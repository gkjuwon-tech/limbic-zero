#!/usr/bin/env bash
# Google Colab Free TPU v5e-1 (16 GiB HBM + ~50 GB Host RAM) 4-bit Launcher for Gemma-4-31B-IT.
# Streams safetensors shards into 4-bit PackedInt4 on TPU v5e-1, opens a Cloudflare Quick Tunnel,
# and exposes the LIMBIC-0 backend so external HTML clients (outside Colab) can chat live.
#
# Usage inside a Colab TPU v5e-1 cell:
#   !git clone https://github.com/gkjuwon-tech/limbic-zero.git && bash limbic-zero/scripts/run_colab_tpu_v5e_4bit.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Keep lightweight soul/LoRA/habitat state on Google Drive (if mounted) so HUMAN-01's memories
# survive Colab disconnects, while keeping the 62 GB HuggingFace shard cache on fast local NVMe
# so we NEVER blow through Free Google Drive's 15 GB storage quota.
if [ -n "${LIMBIC_STATE_DIR:-}" ]; then
  STATE_ROOT="$LIMBIC_STATE_DIR"
elif [ -d "/content/drive/MyDrive" ]; then
  STATE_ROOT="/content/drive/MyDrive/limbic_state"
elif [ -d "/content" ]; then
  STATE_ROOT="/content/limbic_state"
else
  STATE_ROOT="/tmp/limbic_state"
fi

if [ -n "${LIMBIC_HF_HOME:-}" ]; then
  export HF_HOME="$LIMBIC_HF_HOME"
elif [ -d "/content" ]; then
  export HF_HOME="/content/hf_cache"
else
  export HF_HOME="/tmp/hf_cache"
fi

mkdir -p "$HF_HOME" "$STATE_ROOT/subspace" "$STATE_ROOT/habitat" "$STATE_ROOT/plasticity" "$STATE_ROOT/evolution"

export TORCH_TPU_DEFER_AND_FUSE="${TORCH_TPU_DEFER_AND_FUSE:-1}"
if [ -e "/dev/accel0" ] || [ -d "/dev/vfio" ] || [ -n "${COLAB_TPU_ADDR:-}" ] || [ -n "${TPU_NAME:-}" ]; then
  export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
fi
export PYTHONUNBUFFERED="1"
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

if [ "${LIMBIC_SKIP_PIP_INSTALL:-0}" != "1" ]; then
  NEED_TORCH_XLA=$(python3 - <<'PY'
import os
from pathlib import Path

has_tpu_hw = bool(
    os.environ.get("COLAB_TPU_ADDR")
    or os.environ.get("TPU_NAME")
    or os.environ.get("TPU_WORKER_ID")
    or Path("/dev/accel0").exists()
    or Path("/dev/vfio").exists()
)
if not has_tpu_hw:
    print(0)
else:
    found = False
    for mod in ("torch_tpu", "torch_xla"):
        try:
            __import__(mod)
            found = True
            break
        except Exception:
            pass
    print(0 if found else 1)
PY
)
  if [ "$NEED_TORCH_XLA" = "1" ]; then
    echo "[LIMBIC-0 COLAB TPU v5e-1] TPU hardware detected without torch_tpu/torch_xla -> installing torch_xla[tpu]..."
    TORCH_MM=$(python3 -c 'import torch; v=torch.__version__.split("+")[0].split("."); print(f"{v[0]}.{v[1]}.*")')
    pip install -q "torch_xla[tpu]==${TORCH_MM}" -f https://storage.googleapis.com/libtpu-releases/index.html || \
      pip install -q "torch_xla[tpu]" -f https://storage.googleapis.com/libtpu-releases/index.html
  fi
  echo "[LIMBIC-0 COLAB TPU v5e-1] Installing runtime packages (preserving Colab's native torch / TPU bridge)..."
  pip install -q --upgrade "transformers>=5.5.0" "jinja2>=3.1.6" accelerate safetensors sentencepiece protobuf huggingface_hub pillow
  pip install -q --no-deps -e "$REPO_DIR"
fi

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "[LIMBIC-0 COLAB TPU v5e-1] Installing cloudflared binary..."
  CF_TARGET="$HOME/.local/bin/cloudflared"
  mkdir -p "$(dirname "$CF_TARGET")"
  curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" -o "$CF_TARGET"
  chmod +x "$CF_TARGET"
fi

MODEL_ID="${LIMBIC_MODEL_ID:-google/gemma-4-31B-it}"
PORT="${LIMBIC_PORT:-8000}"
HBM_BUDGET_GB="${LIMBIC_TPU_HBM_BUDGET_GB:-12.5}"
GROUP_SIZE="${LIMBIC_QUANT_GROUP_SIZE:-128}"
TUNNEL_URL_FILE="${LIMBIC_TUNNEL_URL_FILE:-/tmp/limbic_cloudflare_url.txt}"
SAFE_SLUG=$(echo "$MODEL_ID" | tr '/-' '__' | tr '[:upper:]' '[:lower:]')

rm -f "$TUNNEL_URL_FILE"

echo "[LIMBIC-0 COLAB TPU v5e-1] Booting ${MODEL_ID} in 4-bit PackedInt4 (HBM budget=${HBM_BUDGET_GB} GiB, group_size=${GROUP_SIZE})..."
echo "[LIMBIC-0 COLAB TPU v5e-1] Persistent Soul State : ${STATE_ROOT}"
echo "[LIMBIC-0 COLAB TPU v5e-1] Ephemeral HF Shard Dir: ${HF_HOME}"

exec python3 "$REPO_DIR/scripts/run_stage5_gemma4_vocal_cord.py" \
  --model-id "$MODEL_ID" \
  --device-backend "${LIMBIC_DEVICE_BACKEND:-auto}" \
  --load-in-4bit \
  --tpu-hbm-budget-gb "$HBM_BUDGET_GB" \
  --quant-group-size "$GROUP_SIZE" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --souls-dir "$REPO_DIR/checkpoints/souls" \
  --subspace-cache "$STATE_ROOT/subspace/${SAFE_SLUG}_subspace_v6.pt" \
  --habitat-dir "$STATE_ROOT/habitat" \
  --plasticity-dir "$STATE_ROOT/plasticity" \
  --evolution-dir "$STATE_ROOT/evolution" \
  --evolution-interval-sec "${LIMBIC_EVOLUTION_INTERVAL_SEC:-150}" \
  --evolution-idle-grace-sec "${LIMBIC_EVOLUTION_IDLE_GRACE_SEC:-28}" \
  --enable-cloudflare \
  --tunnel-url-file "$TUNNEL_URL_FILE" \
  "$@"
