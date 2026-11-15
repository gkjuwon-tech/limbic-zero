#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="/workspace/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export PYTHONUNBUFFERED="1"
mkdir -p "$HF_HOME" /workspace/logs /workspace/habitat /workspace/plasticity /workspace/evolution

BOOT_LOG="/workspace/logs/boot.log"
: > "$BOOT_LOG"
exec > >(tee -a "$BOOT_LOG") 2>&1

echo "[BOOT] Starting LIMBIC-0 Stage 12 bootstrap ($(date -u +%FT%TZ))..."

python3 - <<'PY' &
import http.server, json, os, pathlib, socketserver

BOOT_LOG = pathlib.Path("/workspace/logs/boot.log")

class BootHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return
    def do_GET(self):
        tail = BOOT_LOG.read_text(encoding="utf-8", errors="replace")[-6000:] if BOOT_LOG.is_file() else ""
        if self.path.startswith("/boot-log"):
            body = tail.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = json.dumps({"status": "booting", "log_tail": tail[-1500:]}).encode("utf-8")
            self.send_response(503)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

pathlib.Path("/tmp/limbic0_boot_http.pid").write_text(str(os.getpid()), encoding="utf-8")
with ReusableTCPServer(("0.0.0.0", 8000), BootHandler) as httpd:
    httpd.serve_forever()
PY

cd /workspace/app

echo "[BOOT] Installing Python runtime packages..."
pip uninstall -y torchaudio torchvision >/dev/null 2>&1 || true
pip install -q --upgrade "transformers>=5.5.0" accelerate safetensors sentencepiece protobuf huggingface_hub hf_transfer
pip install -q --no-deps -e /workspace/app

echo "[BOOT] Launching run_stage5_gemma4_vocal_cord.py (google/gemma-4-31B-it)..."
exec python3 /workspace/app/scripts/run_stage5_gemma4_vocal_cord.py \
  --model-id google/gemma-4-31B-it \
  --host 0.0.0.0 \
  --port 8000 \
  --souls-dir /workspace/app/checkpoints/souls \
  --subspace-cache /workspace/hf_cache/gemma4_31b_subspace_v6.pt \
  --habitat-dir /workspace/habitat \
  --plasticity-dir /workspace/plasticity \
  --evolution-dir /workspace/evolution \
  --evolution-interval-sec 150 \
  --evolution-idle-grace-sec 28 \
  --enable-cloudflare
