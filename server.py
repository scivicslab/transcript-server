#!/usr/bin/env python3
"""
Transcript REST server for quarkus-exdb2 (Video tab).

Downloads a video's audio with yt-dlp and transcribes it with Whisper (faster-whisper
on x86_64, openai-whisper/PyTorch on the aarch64 GB10 hosts; see WHISPER_BACKEND),
returning Whisper segments so that quarkus-exdb2 (Java) can chunk and store them
via TranscriptClient. See VideoTranscriptLifecycle_260601_oo01.

Endpoints:
  POST /transcript   (application/json)
    {"url": "https://www.youtube.com/watch?v=..."}
    -> {"success": true, "title": "...", "segments": [{"start": 0.0, "end": 4.2, "text": "..."}]}
       on failure: {"success": false, "error": "..."}

  GET /health        -> {"loaded": "whisper"|"none", "device": "cuda"|"cpu"}

GPU co-residency with Marker (same host 192.168.5.13, Marker on :8001, this on
:8002): the RTX 4080 (16GB) cannot hold both models at once. This server loads
the Whisper model on demand and UNLOADS it (frees VRAM) after each request, so
the GPU is free for Marker when transcription is not running. A process-wide lock
serializes transcription requests so only one Whisper run holds VRAM at a time.
"""

import os
import gc
import logging
import tempfile
import threading

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcript-server")

# Whisper model size and device come from env; defaults suit the 5.13 RTX 4080.
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
# Which Whisper implementation to run. "faster-whisper" (CTranslate2) is the
# original and the fastest on x86_64 GPUs, but the CTranslate2 wheels for
# aarch64 have no CUDA support (verified on the GB10: 0 CUDA devices), so on the
# GB10 hosts the "openai-whisper" backend (PyTorch, CUDA-13 aarch64 wheels)
# is used instead. Both return the same segments through _Backend.
BACKEND = os.environ.get("WHISPER_BACKEND", "faster-whisper")
# Keep the model resident between requests. Default true: Marker and Whisper run
# concurrently on the 5.13 RTX 4080 (16GB) — Marker ~3.7GB + Whisper ~3-5GB fits —
# so there is no need to unload, and keeping it loaded avoids per-request load wait.
KEEP_LOADED = os.environ.get("WHISPER_KEEP_LOADED", "1") == "1"

app = FastAPI(title="Transcript server (yt-dlp + faster-whisper)")

_model = None
_lock = threading.Lock()   # serialize transcription so only one run holds VRAM


class TranscriptRequest(BaseModel):
    url: str


class _Segment:
    """One transcribed segment; the shape both backends are normalised to."""
    __slots__ = ("start", "end", "text")

    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class _FasterWhisperBackend:
    """faster-whisper (CTranslate2)."""

    def __init__(self):
        global DEVICE, COMPUTE_TYPE
        from faster_whisper import WhisperModel
        try:
            self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
            logger.info("faster-whisper %s loaded on %s/%s", MODEL_SIZE, DEVICE, COMPUTE_TYPE)
        except Exception as e:
            logger.warning("Failed on %s (%s); falling back to cpu/int8", DEVICE, e)
            DEVICE, COMPUTE_TYPE = "cpu", "int8"
            self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    def transcribe(self, audio_path, language):
        segments, _info = self.model.transcribe(audio_path, language=language)
        return [_Segment(s.start, s.end, s.text) for s in segments]


_OPENAI_WHISPER_WORKER = r"""
import json, sys
import torch, whisper
audio, model_size, device, language, fp16 = sys.argv[1:6]
if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"
model = whisper.load_model(model_size, device=device)
result = model.transcribe(audio, language=language, fp16=(fp16 == "1" and device == "cuda"))
json.dump({"device": device,
           "segments": [{"start": s["start"], "end": s["end"], "text": s["text"]}
                        for s in result.get("segments", [])]}, sys.stdout)
"""


class _OpenAiWhisperBackend:
    """openai-whisper (PyTorch). Used on aarch64 GPU hosts (GB10).

    With WHISPER_KEEP_LOADED=1 the model stays resident in this process. With
    WHISPER_KEEP_LOADED=0 every request runs in a short-lived worker process:
    on the GB10 the unified memory a CUDA process has touched is not returned
    to the host by unloading the model (the uvicorn process kept 4.2 GB RSS
    and MemAvailable stayed 8 GB after unloading), and the host is shared with
    a vLLM server that must not be starved, so the memory is released by
    letting the worker process exit. Loading large-v3 from the local cache
    costs about 10 s per request.
    """

    def __init__(self):
        global DEVICE
        self.fp16 = (DEVICE == "cuda" and COMPUTE_TYPE == "float16")
        self.model = None
        if KEEP_LOADED:
            import torch
            import whisper
            if DEVICE == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available to PyTorch; falling back to cpu")
                DEVICE = "cpu"
            self.model = whisper.load_model(MODEL_SIZE, device=DEVICE)
            logger.info("openai-whisper %s loaded on %s (fp16=%s)", MODEL_SIZE, DEVICE, self.fp16)
        else:
            logger.info("openai-whisper %s: one worker process per request on %s", MODEL_SIZE, DEVICE)

    def transcribe(self, audio_path, language):
        if self.model is not None:
            result = self.model.transcribe(audio_path, language=language, fp16=self.fp16)
            return [_Segment(s["start"], s["end"], s["text"]) for s in result.get("segments", [])]
        import json
        import subprocess
        import sys
        proc = subprocess.run(
            [sys.executable, "-c", _OPENAI_WHISPER_WORKER, audio_path, MODEL_SIZE, DEVICE,
             language, "1" if self.fp16 else "0"],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError("whisper worker failed: " + proc.stderr[-2000:])
        result = json.loads(proc.stdout)
        return [_Segment(s["start"], s["end"], s["text"]) for s in result["segments"]]


def _load_model():
    """Load the Whisper backend selected by WHISPER_BACKEND (CUDA, CPU fallback)."""
    global _model
    if _model is not None:
        return _model
    if BACKEND == "openai-whisper":
        _model = _OpenAiWhisperBackend()
    elif BACKEND == "faster-whisper":
        _model = _FasterWhisperBackend()
    else:
        raise ValueError("unknown WHISPER_BACKEND: " + BACKEND)
    return _model


def _unload_model():
    """Free the Whisper model and its VRAM so Marker can use the GPU."""
    global _model
    if _model is None:
        return
    _model = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("Whisper unloaded; VRAM freed")


# YouTube serves a different media URL per "player client" (web/android/ios/tv),
# and the default web URLs intermittently return HTTP 403. The android/ios app
# clients usually return more directly-downloadable URLs that do not 403. So we try
# several client sets in order, each with built-in retries, and only fail if all of
# them fail. This turns the previous "one shot, give up on 403" into "several
# identities x a few retries each". (Persistent 403 on a throttled IP/video would
# still need cookies / a PO token — not handled here.)
_PLAYER_CLIENT_SETS = [
    ["android", "web_safari", "web"],
    ["ios"],
    ["tv"],
]


def _download_audio(url: str, workdir: str) -> str:
    """Download bestaudio to a local file with yt-dlp; return (title, path, thumbnail).

    Tries multiple YouTube player clients (with retries) so an intermittent HTTP 403
    on one client's media URL falls through to another instead of failing the request.
    """
    import yt_dlp
    out_tmpl = os.path.join(workdir, "audio.%(ext)s")
    last_error = None
    for clients in _PLAYER_CLIENT_SETS:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_tmpl,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            # A URL carrying both v= and list= (e.g. a video opened from within a
            # playlist) must fetch only that one video, not the whole playlist.
            "noplaylist": True,
            # Auto-retry transient failures before giving up on this client.
            "retries": 5,
            "fragment_retries": 5,
            "extractor_retries": 3,
            "extractor_args": {"youtube": {"player_client": clients}},
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return (info.get("title"), ydl.prepare_filename(info),
                        info.get("thumbnail"))
        except Exception as e:
            last_error = e
            logger.warning("yt-dlp download failed with player_client=%s: %s", clients, e)
            continue
    # All client sets failed: surface the last error to the caller (-> 500 + message).
    raise last_error


def _fetch_thumbnail_data_uri(thumb_url: str) -> str:
    """Download the thumbnail and return it as a data: URI, or "" on failure.
    Fetching server-side avoids the browser needing direct internet access."""
    if not thumb_url:
        return ""
    try:
        import urllib.request
        req = urllib.request.Request(thumb_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            ctype = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        import base64
        return "data:" + ctype + ";base64," + base64.b64encode(data).decode("ascii")
    except Exception as e:
        logger.warning("thumbnail fetch failed: %s", e)
        return ""


@app.get("/health")
def health():
    return {"loaded": "whisper" if _model is not None else "none", "device": DEVICE,
            "backend": BACKEND}


@app.post("/transcript")
def transcript(req: TranscriptRequest):
    url = (req.url or "").strip()
    if not url:
        return JSONResponse(status_code=400,
                            content={"success": False, "error": "missing url"})
    # Serialize: only one transcription holds VRAM at a time (co-resident Marker).
    with _lock:
        workdir = tempfile.mkdtemp(prefix="transcript_")
        try:
            title, audio_path, thumb_url = _download_audio(url, workdir)
            thumbnail = _fetch_thumbnail_data_uri(thumb_url)
            model = _load_model()
            segments = [
                {"start": round(s.start, 2), "end": round(s.end, 2),
                 "text": s.text.strip()}
                for s in model.transcribe(audio_path, language="en")
                if s.text and s.text.strip()
            ]
            return {"success": True, "title": title or url,
                    "thumbnail": thumbnail, "segments": segments}
        except Exception as e:
            logger.exception("transcription failed")
            return JSONResponse(status_code=500,
                                content={"success": False, "error": str(e)})
        finally:
            # Always clean the temp audio; unload the model unless asked to keep it.
            try:
                import shutil
                shutil.rmtree(workdir, ignore_errors=True)
            except Exception:
                pass
            if not KEEP_LOADED:
                _unload_model()
