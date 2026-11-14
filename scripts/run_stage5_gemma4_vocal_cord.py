"""Stage 5 Live Production Launcher: LIMBIC-0 Matured Souls × Gemma-4-12B-IT Vocal Cord Server.

Loads ``google/gemma-4-12B-it`` in native ``bfloat16``, extracts & KL-calibrates the 6D -> 3840D
affective steering subspace across Global Full-Attention middle layers ``(17, 23, 29, 35)``,
hydrates ``ADAM-01``, ``ABEL-02``, and ``PANDORA-03`` living soul capsules, and serves the
interactive chat UI + REST + OpenAI-compatible ``/v1/chat/completions`` endpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Dict

import torch

from limbic_zero.vocal_cord import (
    CalibratedSubspaceBundle,
    ContrastiveSubspaceExtractor,
    Gemma4SteeringConfig,
    Gemma4VocalCordSurgeon,
    LivingSoulOrganism,
    VocalCordHTTPServer,
    VocalCordRuntimeHub,
)


def _start_cloudflare_quick_tunnel(local_port: int, hub: VocalCordRuntimeHub) -> None:
    import re
    import shutil
    import subprocess
    import threading
    import urllib.request

    cf_bin = shutil.which("cloudflared") or "/usr/local/bin/cloudflared"
    if not Path(cf_bin).is_file():
        cf_bin = str(Path.home() / ".local" / "bin" / "cloudflared")
        Path(cf_bin).parent.mkdir(parents=True, exist_ok=True)
        if not Path(cf_bin).is_file():
            urllib.request.urlretrieve(
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                cf_bin,
            )
            Path(cf_bin).chmod(0o755)

    def _runner() -> None:
        proc = subprocess.Popen(
            [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{local_port}", "--no-autoupdate"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        url_pat = re.compile(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com")
        for line in proc.stdout:
            m = url_pat.search(line)
            if m:
                url = m.group(0)
                hub.public_urls["cloudflare"] = url
                print(f"[STAGE 5] CLOUDFLARE PUBLIC URL: {url}", flush=True)

    threading.Thread(target=_runner, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="LIMBIC-0 × Gemma-4-12B-IT Vocal Cord Server")
    parser.add_argument("--model-id", type=str, default="google/gemma-4-12B-it")
    parser.add_argument("--souls-dir", type=str, default="checkpoints/souls")
    parser.add_argument("--subspace-cache", type=str, default="checkpoints/souls/gemma4_12b_subspace_v2.pt")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--public-url", type=str, default="")
    parser.add_argument("--habitat-dir", type=str, default="", help="Base directory for Stage 6 organism habitats")
    parser.add_argument("--enable-cloudflare", action="store_true", help="Launch cloudflared tunnel automatically")
    args = parser.parse_args()

    t0 = time.perf_counter()
    print(f"[STAGE 5] Loading processor & frozen {args.model_id} in bfloat16...", flush=True)

    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_id)
    except Exception:
        from transformers import AutoTokenizer
        processor = AutoTokenizer.from_pretrained(args.model_id)

    tok_obj = getattr(processor, "tokenizer", processor)

    try:
        from transformers import AutoModelForMultimodalLM
        model_cls = AutoModelForMultimodalLM
    except Exception:
        from transformers import AutoModelForCausalLM
        model_cls = AutoModelForCausalLM
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = model_cls.from_pretrained(
        args.model_id,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"[STAGE 5] Model loaded in {time.perf_counter() - t0:.1f}s. Preparing middle-layer subspace...", flush=True)

    cfg = Gemma4SteeringConfig(model_id=args.model_id)
    cache_path = Path(args.subspace_cache)
    if cache_path.is_file():
        print(f"[STAGE 5] Loading cached calibrated subspace from {cache_path}...", flush=True)
        subspace = CalibratedSubspaceBundle.load(cache_path)
    else:
        print(
            f"[STAGE 5] Extracting 6D First-Person Somatic Subspace + Embodiment Carrier on Layers {cfg.target_layers}...",
            flush=True,
        )

        def _render_prefix(user_q: str) -> str:
            msgs = [{"role": "user", "content": user_q}]
            try:
                base = tok_obj.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                base = tok_obj.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            return base if base.endswith("\n") else (base + "\n")

        prefix_str = _render_prefix("지금 네 내면 상태와 몸의 감각, 기분은 어때?")
        prefix_ids = tok_obj(prefix_str, add_special_tokens=False, return_tensors="pt")["input_ids"]
        prefix_len = int(prefix_ids.shape[-1])

        def _encode_first_person_passage(txt: str) -> Dict[str, object]:
            full_str = prefix_str + txt
            enc = tok_obj(full_str, add_special_tokens=False, return_tensors="pt")
            out: Dict[str, object] = {k: v for k, v in enc.items() if isinstance(v, torch.Tensor)}
            out["_assistant_span_start"] = prefix_len
            return out

        def _encode_kl_probe(q: str) -> Dict[str, torch.Tensor]:
            p_str = _render_prefix(q)
            enc = tok_obj(p_str, add_special_tokens=False, return_tensors="pt")
            return {k: v for k, v in enc.items() if isinstance(v, torch.Tensor)}

        extractor = ContrastiveSubspaceExtractor(config=cfg)
        subspace = extractor.extract_and_calibrate(
            model=model,
            encode_fn=_encode_first_person_passage,
            probe_encode_fn=_encode_kl_probe,
        )
        subspace.save(cache_path)
        print(
            f"[STAGE 5] Calibrated subspace v2 saved! target_layers={subspace.target_layers}, "
            f"norm_fraction={subspace.calibrated_norm_fraction:.4f}, "
            f"kl_curve={subspace.calibration_kl_curve}",
            flush=True,
        )

    souls_dir = Path(args.souls_dir)
    souls = {
        "ADAM-01": LivingSoulOrganism(
            str(souls_dir / "adam01_forsaken_survivor.soul.pt"),
            label_ko="버림받은 생존자",
        ),
        "ABEL-02": LivingSoulOrganism(
            str(souls_dir / "abel02_favored_guardian.soul.pt"),
            label_ko="편애받은 수호자",
        ),
        "PANDORA-03": LivingSoulOrganism(
            str(souls_dir / "pandora03_curious_seeker.soul.pt"),
            label_ko="카오스 탐구자",
        ),
    }

    surgeon = Gemma4VocalCordSurgeon(model=model, subspace=subspace, config=cfg)
    pub_urls = {"cloudflare": args.public_url} if args.public_url else {}
    hub = VocalCordRuntimeHub(
        surgeon=surgeon,
        processor_or_tokenizer=processor,
        souls=souls,
        public_urls=pub_urls,
        habitat_base_dir=args.habitat_dir or None,
    )
    hub.start_background_heartbeats()
    if args.enable_cloudflare:
        _start_cloudflare_quick_tunnel(args.port, hub)

    srv = VocalCordHTTPServer((args.host, args.port), hub=hub)
    print(f"[STAGE 5] VOCAL CORD SERVER LIVE ON http://{args.host}:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        hub.stop()
        srv.server_close()


if __name__ == "__main__":
    main()
