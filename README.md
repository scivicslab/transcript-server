# Transcript REST サーバ（yt-dlp + Whisper）

quarkus-exdb2 の Video タブのバックエンド。動画 URL から音声を yt-dlp で取得し、
Whisper で文字起こしして Whisper セグメントを返す。x86_64 の 5.13 では faster-whisper、aarch64 の GB10（5.18）では openai-whisper（PyTorch）を使う（`WHISPER_BACKEND`）。仕様は
`VideoTranscriptLifecycle_260601_oo01` を参照。

## エンドポイント

```
POST /transcript   (application/json)
  {"url": "https://www.youtube.com/watch?v=..."}
  -> {"success": true, "title": "...",
      "segments": [{"start": 0.0, "end": 4.2, "text": "..."}]}
     失敗時: {"success": false, "error": "..."}

GET /health         -> {"loaded": "whisper"|"none", "device": "cuda"|"cpu", "backend": "faster-whisper"|"openai-whisper"}
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

## GB10（192.168.5.18、aarch64）での 2 台目

5.13 の 1 台では文字起こしが 1 本ずつ直列になり待ちが出るので、2 台目を 5.18 で動かす。
GPU broker（MiniPC `:28005`）はサブネットの `:8003` を起動時に探索して同じ `whisper-transcript`
キューに束ねるので、exdb2 側の変更は要らない。ただし探索は broker の起動時だけなので、
2 台目を立てたあとに broker を再起動する。

faster-whisper が使う CTranslate2 の pip wheel は aarch64 では CUDA 無し（5.18 で実測、CUDA デバイス 0）
なので、GB10 では `openai-whisper`（PyTorch、CUDA 13 の aarch64 wheel）で動かす。image は
`Dockerfile.gb10` で 5.18 上で焼く（x86_64 の image は使えない）。

```bash
scp server.py Dockerfile.gb10 devteam@192.168.5.18:~/transcript-server/
ssh devteam@192.168.5.18
cd ~/transcript-server
sudo docker build -f Dockerfile.gb10 -t scivicslab/transcript-server:gb10 .
sudo docker volume create transcript-cache
sudo docker run -d --name transcript-server \
  --gpus all --restart unless-stopped --memory 14g \
  -e WHISPER_KEEP_LOADED=0 \
  -p 8003:8000 \
  -v transcript-cache:/cache \
  scivicslab/transcript-server:gb10
```

`WHISPER_KEEP_LOADED=0` にする理由: 5.18 は vLLM（Qwen3.8-Flash-Next、ユニファイドメモリ 121 GB
のうち 105 GB）と同居で、空きは 16〜18 GB しか無い。GB10 では CUDA を使ったプロセスが触った
ユニファイドメモリはモデルを解放しても戻らない（常駐させると uvicorn の RSS 4.2 GB、空き 8 GB のまま）。
`0` にすると openai-whisper backend はリクエストごとに worker プロセスを起こし、終了で
メモリが戻る（空き 17 GB に復帰、uvicorn の RSS 0.5 GB）。代償はリクエストごとの重み読み込み約 10 秒で、
19 秒の動画で 1 リクエスト 22〜34 秒だった。`--memory 14g` は、溢れたときに落ちるのが vLLM ではなく
このコンテナになるようにする上限である。

初回は Whisper の重み（2.9 GB）を `transcript-cache` ボリューム（`/cache`）に落とす。
## 停止・削除

```bash
docker stop transcript-server && docker rm transcript-server
```

## quarkus-exdb2 からの接続先

`application.properties` の `exdb2.transcript.url=http://192.168.5.13:8003` が直接の向き先。`gpu.broker.url` を設定した instance は broker の `whisper-transcript` キュー経由で、5.13 と 5.18 の空いている方に振られる。

## 動作確認

```bash
curl -s -XPOST http://192.168.5.13:8003/transcript \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://www.youtube.com/watch?v=<id>"}' | head -c 400
```
