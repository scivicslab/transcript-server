# Transcript REST server: yt-dlp (audio download) + faster-whisper (ASR).
# Co-resides with Marker on the 5.13 RTX 4080; loads/unloads Whisper per request.
FROM python:3.11-slim

# ffmpeg is required by yt-dlp for audio extraction.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# faster-whisper uses CTranslate2, which needs the CUDA cuBLAS + cuDNN runtime
# at inference time (libcublas.so.12 / libcudnn). The slim base does not provide
# them, so install the pip-packaged CUDA 12 runtime libs explicitly and put them
# on the loader path.
RUN pip install --no-cache-dir \
        faster-whisper \
        yt-dlp \
        fastapi \
        "uvicorn[standard]" \
        pydantic \
        nvidia-cublas-cu12 \
        nvidia-cudnn-cu12

# Expose the pip-installed NVIDIA runtime libraries to the dynamic loader.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

WORKDIR /app
COPY server.py /app/server.py

ENV WHISPER_MODEL=large-v3
ENV WHISPER_DEVICE=cuda
ENV WHISPER_COMPUTE_TYPE=float16
# Keep Whisper resident; runs concurrently with Marker on the 5.13 RTX 4080.
ENV WHISPER_KEEP_LOADED=1
EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
