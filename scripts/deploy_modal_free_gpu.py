"""Zero-Cost Cloud GPU Launcher for LIMBIC-0 via Modal Serverless GPU ($30/mo Free Credits).

RunPod credits exhausted? Modal provides $30/month in free serverless GPU credits on signup
(no upfront deposit required), supporting NVIDIA A100-80GB / H100 for `google/gemma-4-31B-it`
in native `bfloat16` with automatic idle spin-down so credits are only consumed while active.

Quickstart (from any laptop/PC terminal):
  1. `pip install modal && modal setup`
  2. (Optional, if gated model) `modal secret create huggingface-secret HF_TOKEN=hf_...`
  3. Ephemeral live dev session:
       `modal serve scripts/deploy_modal_free_gpu.py`
     Or persistent HTTPS deployment (auto-scales to 0 when idle):
       `modal deploy scripts/deploy_modal_free_gpu.py`

Environment overrides (optional):
  - `LIMBIC_GPU="A100-80GB"` (default; or `"H100"`, `"L40S"`, `"A10G"`)
  - `LIMBIC_MODEL_ID="google/gemma-4-31B-it"` (or `"google/gemma-4-E4B-it"` on 24GB GPUs)
  - `LIMBIC_IDLE_SCALEDOWN_SEC="600"` (spin down container after 10 min of HTTP inactivity)
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

GPU_SPEC = os.environ.get("LIMBIC_GPU", "A100-80GB")
MODEL_ID = os.environ.get("LIMBIC_MODEL_ID", "google/gemma-4-31B-it")
SCALEDOWN_WINDOW_SEC = int(os.environ.get("LIMBIC_IDLE_SCALEDOWN_SEC", "600"))

app = modal.App("limbic-zero-stage13")

# Persistent volume preserves evolved `.soul.pt` / `.cortex.pt` checkpoints,
# organism-authored workspace artifacts, and HuggingFace weights across restarts.
state_volume = modal.Volume.from_name("limbic-zero-state", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "tar", "procps")
    .pip_install(
        "torch>=2.4.0",
        "numpy>=1.26.0",
        "transformers>=5.5.0",
        "jinja2>=3.1.6",
        "accelerate>=1.2.0",
        "safetensors>=0.4.5",
        "sentencepiece>=0.2.0",
        "protobuf>=5.27.0",
        "huggingface_hub>=0.28.0",
        "pillow>=10.4.0",
    )
    .add_local_dir(
        str(REPO_ROOT / "src"),
        remote_path="/workspace/app/src",
        copy=True,
    )
    .add_local_dir(
        str(REPO_ROOT / "checkpoints"),
        remote_path="/workspace/app/checkpoints",
        copy=True,
    )
    .add_local_dir(
        str(REPO_ROOT / "scripts"),
        remote_path="/workspace/app/scripts",
        copy=True,
    )
    .add_local_file(
        str(REPO_ROOT / "pyproject.toml"),
        remote_path="/workspace/app/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        str(REPO_ROOT / "README.md"),
        remote_path="/workspace/app/README.md",
        copy=True,
    )
    .run_commands("pip install --no-deps -e /workspace/app")
)

_OPTIONAL_SECRETS = []
if os.environ.get("LIMBIC_USE_HF_SECRET", "0") == "1":
    _OPTIONAL_SECRETS.append(modal.Secret.from_name("huggingface-secret"))
elif os.environ.get("HF_TOKEN"):
    _OPTIONAL_SECRETS.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))


@app.function(
    image=image,
    gpu=GPU_SPEC,
    volumes={"/data": state_volume},
    secrets=_OPTIONAL_SECRETS,
    timeout=86400,
    scaledown_window=SCALEDOWN_WINDOW_SEC,
    max_containers=1,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(8000, startup_timeout=900)
def serve_limbic_zero() -> None:
    """Boot HUMAN-01 × Gemma-4 with sovereign Stage 13 autonomy on port 8000."""
    for sub in ("hf_cache", "habitat", "plasticity", "evolution"):
        Path(f"/data/{sub}").mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["HF_HOME"] = "/data/hf_cache"
    env["HF_XET_HIGH_PERFORMANCE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    safe_model_slug = MODEL_ID.replace("/", "_").replace("-", "_").lower()
    subspace_cache = f"/data/hf_cache/{safe_model_slug}_subspace_v6.pt"

    cmd = [
        "python3",
        "/workspace/app/scripts/run_stage5_gemma4_vocal_cord.py",
        "--model-id",
        MODEL_ID,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--souls-dir",
        "/workspace/app/checkpoints/souls",
        "--subspace-cache",
        subspace_cache,
        "--habitat-dir",
        "/data/habitat",
        "--plasticity-dir",
        "/data/plasticity",
        "--evolution-dir",
        "/data/evolution",
        "--evolution-interval-sec",
        "120",
        "--evolution-idle-grace-sec",
        "25",
    ]
    subprocess.Popen(cmd, cwd="/workspace/app", env=env)
