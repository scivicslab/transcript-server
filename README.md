# Transcript REST サーバ（yt-dlp + faster-whisper）

quarkus-exdb2 の Video タブのバックエンド。動画 URL から音声を yt-dlp で取得し、
faster-whisper で文字起こしして Whisper セグメントを返す。仕様は
`VideoTranscriptLifecycle_260601_oo01` を参照。

## エンドポイント

```
POST /transcript   (application/json)
  {"url": "https://www.youtube.com/watch?v=..."}
  -> {"success": true, "title": "...",
      "segments": [{"start": 0.0, "end": 4.2, "text": "..."}]}
     失敗時: {"success": false, "error": "..."}

GET /health         -> {"loaded": "whisper"|"none", "device": "cuda"|"cpu"}
```

## GPU 同居（5.13 RTX 4080, Marker と併用）

このサーバは Marker（`192.168.5.13:8001`）と同じ 5.13 ホスト・別ポート（`:8003`）で動く。
RTX 4080（16GB）に Marker（約 3.7GB）と Whisper（large-v3, 約 3〜5GB）を**同時に常駐**させて
併用する（合計 8〜9GB で 16GB に収まる）。Whisper はモデルを常駐させたまま
（既定 `WHISPER_KEEP_LOADED=1`）にし、リクエストごとのロード待ちを無くす。プロセス内
ロックで文字起こしは1件ずつ直列化する。

## ビルド

```bash
cd ~/works/transcript-server
docker build -t scivicslab/transcript-server:latest .
```

## 起動（5.13 上、ポート 8003、GPU 使用）

```bash
docker run -d --name transcript-server \
  --gpus all \
  -p 8003:8000 \
  -v hf-cache:/root/.cache/huggingface \
  scivicslab/transcript-server:latest
```

初回のみ Whisper モデル重み（large-v3）のダウンロードが走り `hf-cache` に載る。

## 起動（CPU のみ・GPU が無いホストでの検証用）

```bash
docker run -d --name transcript-server \
  -e WHISPER_DEVICE=cpu -e WHISPER_COMPUTE_TYPE=int8 \
  -e WHISPER_MODEL=base \
  -p 8003:8000 \
  -v hf-cache:/root/.cache/huggingface \
  scivicslab/transcript-server:latest
```

CPU では large-v3 は非現実的に遅いので、検証は `base`/`small` モデルで行う。

## 停止・削除

```bash
docker stop transcript-server && docker rm transcript-server
```

## quarkus-exdb2 からの接続先

`application.properties` の `exdb2.transcript.url=http://192.168.5.13:8003` が向き先。

## 動作確認

```bash
curl -s -XPOST http://192.168.5.13:8003/transcript \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://www.youtube.com/watch?v=<id>"}' | head -c 400
```
