#!/usr/bin/env bash
# Zero-Cost Forever-Free GPU Launcher for Kaggle (2x T4 32GB, 30h/wk) & Google Colab (T4 16GB).
# Usage inside a Kaggle or Colab notebook cell:
#   !git clone https://github.com/gkjuwon-tech/limbic-zero.git && bash limbic-zero/scripts/run_kaggle_colab_free_gpu.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ -d "/kaggle/working" ]; then
  STATE_ROOT="/kaggle/working/limbic_state"
  export HF_HOME="${LIMBIC_HF_HOME:-/kaggle/working/hf_cache}"
elif [ -d "/content/drive/MyDrive" ]; then
  STATE_ROOT="/content/drive/MyDrive/limbic_state"
  export HF_HOME="${LIMBIC_HF_HOME:-/content/hf_cache}"
else
  STATE_ROOT="${LIMBIC_STATE_DIR:-/tmp/limbic_state}"
  export HF_HOME="${LIMBIC_HF_HOME:-/tmp/hf_cache}"
fi

TPU_DETECTED=$(python3 - <<'PY'
import os
if os.environ.get("COLAB_TPU_ADDR") or os.environ.get("TPU_NAME") or os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
    print(1)
else:
    found = 0
    for mod in ("torch_tpu", "torch_xla"):
        try:
            __import__(mod)
            found = 1
            break
        except Exception:
            pass
    print(found)
PY
)

if [ "$TPU_DETECTED" = "1" ]; then
  echo "[LIMBIC-0] Google Cloud TPU runtime detected -> delegating to scripts/run_colab_tpu_v5e_4bit.sh..."
  exec bash "$REPO_DIR/scripts/run_colab_tpu_v5e_4bit.sh" "$@"
fi

mkdir -p "$HF_HOME" "$STATE_ROOT/subspace" "$STATE_ROOT/habitat" "$STATE_ROOT/plasticity" "$STATE_ROOT/evolution"
export PYTHONUNBUFFERED="1"

echo "[LIMBIC-0 FREE GPU] Installing runtime dependencies..."
pip install -q --upgrade "transformers>=5.5.0" "jinja2>=3.1.6" accelerate bitsandbytes safetensors sentencepiece protobuf huggingface_hub pillow
pip install -q --no-deps -e "$REPO_DIR"

TOTAL_VRAM_GB=$(python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    print(0)
else:
    total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count()))
    print(int(total // (1024**3)))
PY
)

EXTRA_ARGS=()
if [ -n "${LIMBIC_MODEL_ID:-}" ]; then
  MODEL_ID="$LIMBIC_MODEL_ID"
elif [ "$TOTAL_VRAM_GB" -ge 70 ]; then
  MODEL_ID="google/gemma-4-31B-it"
elif [ "$TOTAL_VRAM_GB" -ge 26 ]; then
  MODEL_ID="google/gemma-4-31B-it"
  EXTRA_ARGS+=("--load-in-4bit")
else
  MODEL_ID="google/gemma-4-12B-it"
  EXTRA_ARGS+=("--load-in-4bit")
fi

SAFE_SLUG=$(echo "$MODEL_ID" | tr '/-' '__' | tr '[:upper:]' '[:lower:]')
echo "[LIMBIC-0 FREE GPU] Detected ${TOTAL_VRAM_GB}GB total VRAM -> booting ${MODEL_ID} (${EXTRA_ARGS[*]:-native bf16})..."

exec python3 "$REPO_DIR/scripts/run_stage5_gemma4_vocal_cord.py" \
  --model-id "$MODEL_ID" \
  --host 0.0.0.0 \
  --port 8000 \
  --souls-dir "$REPO_DIR/checkpoints/souls" \
  --subspace-cache "$STATE_ROOT/subspace/${SAFE_SLUG}_subspace_v6.pt" \
  --habitat-dir "$STATE_ROOT/habitat" \
  --plasticity-dir "$STATE_ROOT/plasticity" \
  --evolution-dir "$STATE_ROOT/evolution" \
  --evolution-interval-sec 120 \
  --evolution-idle-grace-sec 25 \
  --enable-cloudflare \
  "${EXTRA_ARGS[@]}" \
  "$@"
