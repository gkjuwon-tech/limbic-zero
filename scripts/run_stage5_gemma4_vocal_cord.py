"""Stage 12 Live Production Launcher: Unified Living Human Soul ('HUMAN-01') × Gemma-4-31B-IT.

Loads ``google/gemma-4-31B-it`` (60 layers, 5376D) in native ``bfloat16``, extracts & KL-calibrates
the 6D -> 5376D somatic steering subspace (`subspace_v6`) across Global Full-Attention middle layers,
hydrates the unified evolved organism ``HUMAN-01`` (auto-resuming from ``latest.soul.pt`` &
``latest.cortex.pt`` when available), runs continuous autonomous foraging + cortical LoRA weight
surgery + sleep consolidation + checkpoint generation in the background, and serves the 1:1 chat UI.
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
from limbic_zero.vocal_cord.gemma4_calibrator import resolve_decoder_layers


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
                print(f"[STAGE 12] CLOUDFLARE PUBLIC URL: {url}", flush=True)

    threading.Thread(target=_runner, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="LIMBIC-0 Unified Soul × Gemma-4-31B-IT Server")
    parser.add_argument("--model-id", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--souls-dir", type=str, default="checkpoints/souls")
    parser.add_argument("--subspace-cache", type=str, default="checkpoints/souls/gemma4_31b_subspace_v6.pt")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--public-url", type=str, default="")
    parser.add_argument("--habitat-dir", type=str, default="", help="Base directory for organism habitat")
    parser.add_argument("--plasticity-dir", type=str, default="", help="Directory for cortical LoRA banks")
    parser.add_argument("--evolution-dir", type=str, default="", help="Directory for evolved checkpoints & ledger")
    parser.add_argument("--evolution-interval-sec", type=float, default=150.0)
    parser.add_argument("--evolution-idle-grace-sec", type=float, default=28.0)
    parser.add_argument("--disable-autonomous-evolution", action="store_true")
    parser.add_argument("--legacy-three-souls", action="store_true", help="Load legacy 3-soul comparison bank")
    parser.add_argument("--enable-cloudflare", action="store_true", help="Launch cloudflared tunnel automatically")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in NF4 4-bit (for Kaggle 2xT4 / Colab T4)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    quant_label = "NF4 4-bit" if args.load_in_4bit else "bfloat16"
    print(f"[STAGE 13] Loading processor & frozen {args.model_id} in {quant_label}...", flush=True)

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
    load_kwargs: Dict[str, object] = {
        "dtype": dtype,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if args.load_in_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        compute_dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dt,
        )
    model = model_cls.from_pretrained(args.model_id, **load_kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    num_layers = len(resolve_decoder_layers(model))
    cfg = Gemma4SteeringConfig.for_model(args.model_id, num_hidden_layers=num_layers)
    print(
        f"[STAGE 12] Model loaded in {time.perf_counter() - t0:.1f}s "
        f"(layers={num_layers}, target_layers={cfg.target_layers}). Preparing subspace_v6...",
        flush=True,
    )

    cache_path = Path(args.subspace_cache)
    subspace = None
    if cache_path.is_file():
        print(f"[STAGE 12] Loading cached calibrated subspace from {cache_path}...", flush=True)
        loaded = CalibratedSubspaceBundle.load(cache_path)
        if (
            loaded.format_version == "limbic0.gemma4_subspace.v6"
            and loaded.model_id == args.model_id
            and len(loaded.embodiment_directions) == len(cfg.target_layers)
        ):
            subspace = loaded
        else:
            print(
                f"[STAGE 12] Cached subspace mismatch (format={loaded.format_version}, model={loaded.model_id}). "
                f"Re-extracting subspace_v6...",
                flush=True,
            )

    if subspace is None:
        print(
            f"[STAGE 12] Extracting 6D Somatic Subspace + Everyday Human Speech Carrier on Layers {cfg.target_layers}...",
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
            if "<channel|>" not in base[-32:]:
                if base.endswith("<|channel>thought\n"):
                    base = base + "<channel|>"
                elif base.rstrip().endswith("<|channel>thought"):
                    base = base.rstrip() + "\n<channel|>"
                elif base.rstrip().endswith("<|channel>"):
                    base = base.rstrip() + "thought\n<channel|>"
                elif base.rstrip().endswith("<|turn>model"):
                    base = base.rstrip() + "\n<|channel>thought\n<channel|>"
            return base

        prefix_str = _render_prefix("지금 기분이랑 컨디션은 좀 어때?")
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
            f"[STAGE 12] Calibrated subspace_v6 saved! target_layers={subspace.target_layers}, "
            f"norm_fraction={subspace.calibrated_norm_fraction:.4f}, "
            f"kl_curve={subspace.calibration_kl_curve}",
            flush=True,
        )

    souls_dir = Path(args.souls_dir)
    evo_dir = Path(args.evolution_dir).resolve() if args.evolution_dir else None

    if args.legacy_three_souls:
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
    else:
        latest_evolved_soul = (evo_dir / "latest.soul.pt") if evo_dir is not None else None
        if latest_evolved_soul is not None and latest_evolved_soul.is_file():
            soul_ckpt_path = latest_evolved_soul
            print(f"[STAGE 12] Resuming unified organism HUMAN-01 from evolved checkpoint: {soul_ckpt_path}", flush=True)
        else:
            soul_ckpt_path = souls_dir / "human01_ordinary_evolved.soul.pt"
            print(f"[STAGE 12] Hydrating unified organism HUMAN-01 from seed capsule: {soul_ckpt_path}", flush=True)

        souls = {
            "HUMAN-01": LivingSoulOrganism(
                str(soul_ckpt_path),
                label_ko="진화한 일상 인격체",
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
        plasticity_dir=args.plasticity_dir or None,
        evolution_dir=evo_dir,
        enable_autonomous_evolution=not args.disable_autonomous_evolution,
        evolution_idle_grace_sec=args.evolution_idle_grace_sec,
        evolution_cycle_interval_sec=args.evolution_interval_sec,
    )
    hub.start_background_heartbeats()

    boot_pid_file = Path("/tmp/limbic0_boot_http.pid")
    if boot_pid_file.is_file():
        import os
        import signal

        try:
            bpid = int(boot_pid_file.read_text(encoding="utf-8").strip())
            os.kill(bpid, signal.SIGTERM)
            time.sleep(0.5)
        except Exception:
            pass
        boot_pid_file.unlink(missing_ok=True)

    if args.enable_cloudflare:
        _start_cloudflare_quick_tunnel(args.port, hub)

    srv = VocalCordHTTPServer((args.host, args.port), hub=hub)
    print(f"[STAGE 12] UNIFIED HUMAN SOUL SERVER LIVE ON http://{args.host}:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        hub.stop()
        srv.server_close()


if __name__ == "__main__":
    main()
