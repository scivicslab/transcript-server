#!/usr/bin/env python3
"""
Transcript REST server for quarkus-exdb2 (Video tab).

Downloads a video's audio with yt-dlp and transcribes it with Whisper (faster-whisper
on x86_64, openai-whisper/PyTorch on the aarch64 GB10 hosts; see WHISPER_BACKEND),
returning Whisper segments so that quarkus-exdb2 (Java) can chunk and store them
via TranscriptClient. See VideoTranscriptLifecycle_260601_oo01.

Endpoints:
  POST /transcript   (application/json)
    {"url": "https://www.youtube.com/watch?v=..."}     -- or, for a file on the users' shared
    {"file": "<user>/videos/lecture.mp4"}              --    storage mounted at MEDIA_ROOT
    -> {"success": true, "title": "...", "segments": [{"start": 0.0, "end": 4.2, "text": "..."}]}
       on failure: {"success": false, "error": "..."}

       Also "thumbnail" (data: URI) and, when the download carried a video stream,
       "frames": {"url": "<PUBLIC_BASE_URL>/frames", "token": "...", "hasVideo": true}:
       the media file is kept for FRAME_KEEP_SECONDS so that /frames can cut stills
       from it without a second download.

  POST /frames       (application/json)
    {"token": "...", "times": [12.3, 45.0], "width": 640}      -- from a kept media file
    {"url": "https://...", "times": [...], "width": 640}        -- downloads 360p video again
    -> {"success": true, "frames": [{"time": 12.3, "image": "data:image/jpeg;base64,..."}]}

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
# How long a transcribed video's media file is kept for /frames, and the address a
# client uses to reach THIS server for it. Requests arrive through quarkus-gpu-broker,
# which spreads them over several servers, so the transcript answer must name the
# server that holds the file.
FRAME_KEEP_SECONDS = int(os.environ.get("FRAME_KEEP_SECONDS", "1800"))
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
# Video stream height for stills; 360p keeps a one-hour lecture near 150 MB.
FRAME_VIDEO_HEIGHT = int(os.environ.get("FRAME_VIDEO_HEIGHT", "360"))

app = FastAPI(title="Transcript server (yt-dlp + faster-whisper)")

_model = None
_lock = threading.Lock()   # serialize transcription so only one run holds VRAM


class TranscriptRequest(BaseModel):
    url: str | None = None
    # A media file already on this server's MEDIA_ROOT (the users' shared storage, mounted
    # read-only), as a path relative to that root. Used for files uploaded with File Browser.
    file: str | None = None


class FramesRequest(BaseModel):
    token: str | None = None
    url: str | None = None
    file: str | None = None
    times: list[float]
    width: int = 640


# Where uploaded media can be read from (a read-only mount of the users' storage). Empty = off.
MEDIA_ROOT = os.environ.get("MEDIA_ROOT", "").rstrip("/")


def _resolve_media_file(rel: str) -> str:
    """The absolute path of a client-named file under MEDIA_ROOT, or a ValueError."""
    if not MEDIA_ROOT:
        raise ValueError("local media is not enabled on this server (MEDIA_ROOT unset)")
    rel = (rel or "").strip().lstrip("/")
    if not rel:
        raise ValueError("missing file")
    path = os.path.realpath(os.path.join(MEDIA_ROOT, rel))
    root = os.path.realpath(MEDIA_ROOT)
    if path != root and not path.startswith(root + os.sep):
        raise ValueError("file is outside the media root")
    if not os.path.isfile(path):
        raise ValueError("no such file: " + rel)
    return path


# Media files kept after transcription: token -> (path, expires_at). Guarded by _kept_lock.
_kept = {}
_kept_lock = threading.Lock()


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


# Whisper reads the audio track of an mp4 as readily as an m4a, so one download serves both
# the transcript and the stills: small video plus best audio, merged by ffmpeg; a progressive
# file of that size; and audio only when the site offers no video (then /frames has nothing).
_FORMAT_MEDIA = ("bv*[height<=%d][ext=mp4]+ba[ext=m4a]/bv*[height<=%d]+ba/b[height<=%d]/bestaudio/best"
                 % (FRAME_VIDEO_HEIGHT, FRAME_VIDEO_HEIGHT, FRAME_VIDEO_HEIGHT))
# For /frames by url (a video imported before stills existed): the video stream alone.
_FORMAT_VIDEO_ONLY = "bv*[height<=%d][ext=mp4]/bv*[height<=%d]/b[height<=%d]" % (
    FRAME_VIDEO_HEIGHT, FRAME_VIDEO_HEIGHT, FRAME_VIDEO_HEIGHT)


def _download_audio(url: str, workdir: str, fmt: str = _FORMAT_MEDIA) -> str:
    """Download the media to a local file with yt-dlp; return (title, path, thumbnail).

    Tries multiple YouTube player clients (with retries) so an intermittent HTTP 403
    on one client's media URL falls through to another instead of failing the request.
    """
    import yt_dlp
    out_tmpl = os.path.join(workdir, "media.%(ext)s")
    last_error = None
    for clients in _PLAYER_CLIENT_SETS:
        ydl_opts = {
            "format": fmt,
            "merge_output_format": "mp4",
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
                path = ydl.prepare_filename(info)
                if not os.path.exists(path):
                    # A merged download is written as <name>.mp4 whatever the parts were called.
                    merged = os.path.splitext(path)[0] + ".mp4"
                    if os.path.exists(merged):
                        path = merged
                return (info.get("title"), path, info.get("thumbnail"))
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


def _has_video(path: str) -> bool:
    """Whether the file carries a video stream (ffprobe)."""
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30).stdout
        return "video" in out
    except Exception as e:
        logger.warning("ffprobe failed: %s", e)
        return False


def _keep_media(workdir: str, path: str) -> str:
    """Register the media file for /frames and return its token."""
    import secrets
    import time
    token = secrets.token_urlsafe(16)
    with _kept_lock:
        _kept[token] = (workdir, path, time.time() + FRAME_KEEP_SECONDS)
    return token


def _sweep_kept():
    """Delete kept media whose time is up; runs every minute in a daemon thread."""
    import shutil
    import time
    while True:
        time.sleep(60)
        now = time.time()
        with _kept_lock:
            expired = [t for t, (_, _, exp) in _kept.items() if exp <= now]
            entries = [(_kept.pop(t)) for t in expired]
        for workdir, _, _ in entries:
            if workdir:                      # a downloaded file; a shared-storage file is not ours to delete
                shutil.rmtree(workdir, ignore_errors=True)
        if entries:
            logger.info("removed %d expired media file(s)", len(entries))


threading.Thread(target=_sweep_kept, daemon=True, name="kept-media-sweeper").start()


def _cut_frame(path: str, t: float, width: int) -> str:
    """One still at second t as a data: URI (JPEG, scaled to width). Empty on failure."""
    import base64
    import subprocess
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", "%.3f" % max(0.0, t), "-i", path,
             "-frames:v", "1", "-vf", "scale=%d:-2" % width, "-q:v", "4",
             "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
            capture_output=True, timeout=60)
        if proc.returncode != 0 or not proc.stdout:
            logger.warning("ffmpeg still at %.1fs failed: %s", t, proc.stderr[-300:])
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(proc.stdout).decode("ascii")
    except Exception as e:
        logger.warning("ffmpeg still at %.1fs failed: %s", t, e)
        return ""


@app.post("/frames")
def frames(req: FramesRequest):
    """Stills at the requested seconds, from a kept media file (token) or a fresh 360p download (url)."""
    import shutil
    times = [float(t) for t in (req.times or [])]
    width = max(64, min(int(req.width or 640), 1920))
    if not times:
        return JSONResponse(status_code=400, content={"success": False, "error": "missing times"})
    path = None
    tmp_workdir = None
    if req.file:
        try:
            path = _resolve_media_file(req.file)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"success": False, "error": str(e)})
    elif req.token:
        with _kept_lock:
            entry = _kept.get(req.token)
        if entry is None:
            return JSONResponse(status_code=404,
                                content={"success": False, "error": "unknown or expired token"})
        path = entry[1]
    elif req.url:
        tmp_workdir = tempfile.mkdtemp(prefix="frames_")
        try:
            _, path, _ = _download_audio(req.url.strip(), tmp_workdir, fmt=_FORMAT_VIDEO_ONLY)
        except Exception as e:
            shutil.rmtree(tmp_workdir, ignore_errors=True)
            logger.exception("frames download failed")
            return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    else:
        return JSONResponse(status_code=400, content={"success": False, "error": "token, url or file required"})
    try:
        if not _has_video(path):
            return JSONResponse(status_code=422,
                                content={"success": False, "error": "media has no video stream"})
        out = [{"time": t, "image": _cut_frame(path, t, width)} for t in times]
        return {"success": True, "frames": out}
    finally:
        if tmp_workdir:
            shutil.rmtree(tmp_workdir, ignore_errors=True)


@app.get("/health")
def health():
    return {"loaded": "whisper" if _model is not None else "none", "device": DEVICE,
            "backend": BACKEND}


@app.post("/transcript")
def transcript(req: TranscriptRequest):
    url = (req.url or "").strip()
    if not url and not req.file:
        return JSONResponse(status_code=400,
                            content={"success": False, "error": "missing url or file"})
    # Serialize: only one transcription holds VRAM at a time (co-resident Marker).
    with _lock:
        workdir = tempfile.mkdtemp(prefix="transcript_")
        try:
            if req.file:
                # A file on the shared storage: nothing to download; the title is its name and the
                # poster is its first second. It stays where it is (workdir holds nothing).
                audio_path = _resolve_media_file(req.file)
                title = os.path.splitext(os.path.basename(audio_path))[0]
                url = "file:" + req.file.strip().lstrip("/")
                thumbnail = _cut_frame(audio_path, 1.0, 320) if _has_video(audio_path) else ""
            else:
                title, audio_path, thumb_url = _download_audio(url, workdir)
                thumbnail = _fetch_thumbnail_data_uri(thumb_url)
            model = _load_model()
            segments = [
                {"start": round(s.start, 2), "end": round(s.end, 2),
                 "text": s.text.strip()}
                for s in model.transcribe(audio_path, language="en")
                if s.text and s.text.strip()
            ]
            answer = {"success": True, "title": title or url,
                      "thumbnail": thumbnail, "segments": segments}
            # Keep the media for /frames when it has pictures to cut and a client can find us.
            if PUBLIC_BASE_URL and _has_video(audio_path):
                if req.file:
                    token = _keep_media(None, audio_path)     # shared-storage file: never deleted
                else:
                    token = _keep_media(workdir, audio_path)
                    workdir = None      # now owned by the sweeper
                answer["frames"] = {"url": PUBLIC_BASE_URL + "/frames", "token": token,
                                    "hasVideo": True}
            else:
                answer["frames"] = {"hasVideo": False}
            return answer
        except ValueError as e:
            return JSONResponse(status_code=400, content={"success": False, "error": str(e)})
        except Exception as e:
            logger.exception("transcription failed")
            return JSONResponse(status_code=500,
                                content={"success": False, "error": str(e)})
        finally:
            # Clean the temp media unless it was kept for /frames; unload the model unless asked to keep it.
            if workdir is not None:
                try:
                    import shutil
                    shutil.rmtree(workdir, ignore_errors=True)
                except Exception:
                    pass
            if not KEEP_LOADED:
                _unload_model()
