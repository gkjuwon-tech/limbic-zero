#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="/workspace/hf_cache"
export PYTHONUNBUFFERED="1"
mkdir -p "$HF_HOME" /workspace/logs /workspace/habitat /workspace/plasticity

cd /workspace/app

pip uninstall -y torchaudio torchvision >/dev/null 2>&1 || true
pip install -q --upgrade "transformers>=5.5.0" accelerate safetensors sentencepiece protobuf huggingface_hub
pip install -q --no-deps -e /workspace/app

exec python3 /workspace/app/scripts/run_stage5_gemma4_vocal_cord.py \
  --model-id google/gemma-4-12B-it \
  --host 0.0.0.0 \
  --port 8000 \
  --souls-dir /workspace/app/checkpoints/souls \
  --subspace-cache /workspace/hf_cache/gemma4_12b_subspace_v5.pt \
  --habitat-dir /workspace/habitat \
  --plasticity-dir /workspace/plasticity \
  --enable-cloudflare
