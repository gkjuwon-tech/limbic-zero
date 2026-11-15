"""Stage 14 Live Production Launcher: Unified Living Human Soul ('HUMAN-01') × Gemma-4-31B-IT.

Supports both:
  - **Google Cloud TPU v5e-8 (`--backend tpu_spmd`)**: Zero-quantization pure ``bfloat16`` (`58.3 GiB`)
    sharded across all 8 TPU chips (`128 GB` aggregate HBM) via PyTorch/XLA GSPMD + ``XLAPinnedStaticCache``
    bucketed prefill & `O(1)` `dynamic-update-slice` decode.
  - **CUDA / CPU (`--backend cuda` / `cpu`)**: Standard HuggingFace `device_map="auto"` / NF4 fallback.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Dict, Optional

import torch

from limbic_zero.tpu import (
    TPUBucketedStaticGenerator,
    initialize_tpu_spmd_runtime,
    is_tpu_xla_available,
    resolve_kaggle_or_hf_model_source,
    shard_model_to_tpu_spmd,
)
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


def _start_cloudflare_quick_tunnel(
    local_port: int,
    hub: VocalCordRuntimeHub,
    tunnel_url_file: str = "/tmp/limbic_cloudflare_url.txt",
) -> None:
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
                if tunnel_url_file:
                    try:
                        out_p = Path(tunnel_url_file)
                        out_p.parent.mkdir(parents=True, exist_ok=True)
                        out_p.write_text(url + "\n", encoding="utf-8")
                    except Exception:
                        pass
                print(f"[STAGE 14] CLOUDFLARE PUBLIC URL: {url}", flush=True)

    threading.Thread(target=_runner, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="LIMBIC-0 Unified Soul × Gemma-4-31B-IT Server (TPU v5e-8 / TPU v5e-1 4-bit / CUDA)")
    parser.add_argument("--model-id", type=str, default="google/gemma-4-31B-it")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=("auto", "tpu_spmd", "tpu_4bit", "cuda", "cpu"),
        help="Accelerator execution backend (`auto` selects tpu_spmd on v5e-8, tpu_4bit on single-chip Colab v5e-1 / --load-in-4bit, CUDA, or CPU)",
    )
    parser.add_argument(
        "--device-backend",
        type=str,
        default="",
        choices=("", "auto", "tpu", "tpu_4bit", "tpu_spmd", "cuda", "cpu"),
        help="Optional alias for accelerator selection (used by Colab TPU v5e-1 launcher)",
    )
    parser.add_argument("--max-cache-len", type=int, default=2560, help="Pinned static KV-cache length on TPU v5e-8")
    parser.add_argument(
        "--xla-cache-dir",
        type=str,
        default="/kaggle/working/xla_compile_cache",
        help="Persistent XLA HLO compilation cache directory",
    )
    parser.add_argument(
        "--warmup-buckets",
        type=str,
        default="128,256,512",
        help="Comma-separated prefill buckets to pre-compile on TPU before opening background daemons",
    )
    parser.add_argument("--souls-dir", type=str, default="checkpoints/souls")
    parser.add_argument("--subspace-cache", type=str, default="checkpoints/souls/gemma4_31b_subspace_v6.pt")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--public-url", type=str, default="")
    parser.add_argument("--tunnel-url-file", type=str, default="/tmp/limbic_cloudflare_url.txt")
    parser.add_argument("--habitat-dir", type=str, default="", help="Base directory for organism habitat")
    parser.add_argument("--plasticity-dir", type=str, default="", help="Directory for cortical LoRA banks")
    parser.add_argument("--evolution-dir", type=str, default="", help="Directory for evolved checkpoints & ledger")
    parser.add_argument("--evolution-interval-sec", type=float, default=150.0)
    parser.add_argument("--evolution-idle-grace-sec", type=float, default=28.0)
    parser.add_argument("--disable-autonomous-evolution", action="store_true")
    parser.add_argument("--legacy-three-souls", action="store_true", help="Load legacy 3-soul comparison bank")
    parser.add_argument("--enable-cloudflare", action="store_true", help="Launch cloudflared tunnel automatically")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit (PackedInt4 + Hybrid HBM/CPU split on Colab TPU v5e-1, or NF4 on CUDA)",
    )
    parser.add_argument(
        "--tpu-hbm-budget-gb",
        type=float,
        default=12.5,
        help="Max TPU HBM budget (GiB) for 4-bit layers on single-chip Colab TPU v5e-1 (default 12.5 GiB)",
    )
    parser.add_argument("--quant-group-size", type=int, default=128, help="Group size for TPU/CPU 4-bit quantization")
    args = parser.parse_args()

    from limbic_zero.vocal_cord.tpu_quantizer import detect_tpu_device, load_quantized_causal_lm_for_tpu

    requested = args.device_backend if (args.device_backend and args.backend == "auto") else args.backend
    tpu_info = detect_tpu_device() if requested in ("auto", "tpu", "tpu_4bit") else None

    if requested in ("auto", "tpu"):
        if args.load_in_4bit and ((tpu_info is not None and tpu_info.available) or is_tpu_xla_available()):
            active_backend = "tpu_4bit"
        elif is_tpu_xla_available():
            # Check chip count: multi-chip (v5e-8) uses pure-BF16 GSPMD; single-chip (v5e-1) 16GB uses 4-bit streamer
            try:
                import torch_xla.runtime as xr  # type: ignore[import-not-found]
                n_chips = int(xr.global_runtime_device_count())
            except Exception:
                n_chips = 8
            active_backend = "tpu_4bit" if n_chips <= 1 else "tpu_spmd"
        elif tpu_info is not None and tpu_info.available:
            active_backend = "tpu_4bit"
        elif torch.cuda.is_available() and requested != "tpu":
            active_backend = "cuda"
        else:
            active_backend = "cpu"
    else:
        active_backend = requested

    t0 = time.perf_counter()
    spmd_ctx = None
    model_source = resolve_kaggle_or_hf_model_source(args.model_id) if active_backend in ("tpu_spmd", "tpu_4bit") else args.model_id

    if active_backend == "tpu_spmd":
        spmd_ctx = initialize_tpu_spmd_runtime(cache_dir=args.xla_cache_dir or None)
        print(
            f"[STAGE 14] Initialized PyTorch/XLA TPU GSPMD across {spmd_ctx.num_devices} chips "
            f"(device={spmd_ctx.device}, model_source={model_source})...",
            flush=True,
        )
    elif active_backend == "tpu_4bit":
        print(
            f"[STAGE 14] Initializing single-chip Colab TPU v5e-1 4-bit PackedInt4 streamer "
            f"(budget={args.tpu_hbm_budget_gb} GiB, group_size={args.quant_group_size}, source={model_source})...",
            flush=True,
        )
    else:
        quant_label = "4-bit" if args.load_in_4bit else "bfloat16"
        print(f"[STAGE 14] Loading processor & frozen {args.model_id} on {active_backend} ({quant_label})...", flush=True)

    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_source)
    except Exception:
        from transformers import AutoTokenizer
        processor = AutoTokenizer.from_pretrained(model_source)

    tok_obj = getattr(processor, "tokenizer", processor)

    if active_backend == "tpu_4bit" or (args.load_in_4bit and active_backend == "cpu"):
        target_dev = (
            tpu_info.device
            if (active_backend == "tpu_4bit" and tpu_info is not None and tpu_info.available)
            else torch.device("cpu")
        )
        model, _tpu4_summary = load_quantized_causal_lm_for_tpu(
            model_id=model_source,
            tpu_device=target_dev,
            hbm_budget_gb=args.tpu_hbm_budget_gb,
            group_size=args.quant_group_size,
            overflow_device="cpu",
            quantize_overflow_layers=(active_backend == "cpu"),
        )
        model.eval()
        for p in model.parameters():
            if p.is_floating_point():
                p.requires_grad_(False)
    elif active_backend == "tpu_spmd":
        try:
            from transformers import AutoModelForMultimodalLM
            model_cls = AutoModelForMultimodalLM
        except Exception:
            from transformers import AutoModelForCausalLM
            model_cls = AutoModelForCausalLM
        assert spmd_ctx is not None
        print(f"[STAGE 14] Loading pure bfloat16 weights into Host Xeon RAM from {model_source}...", flush=True)
        try:
            model = model_cls.from_pretrained(
                model_source,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
        except TypeError:
            model = model_cls.from_pretrained(
                model_source,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        model.eval()
        print(f"[STAGE 14] Streaming & GSPMD-sharding parameters onto {spmd_ctx.num_devices} TPU v5e chips...", flush=True)
        shard_report = shard_model_to_tpu_spmd(
            model=model,
            spmd_ctx=spmd_ctx,
            drop_unused_multimodal_towers=True,
            target_dtype=torch.bfloat16,
        )
        print(f"[STAGE 14] TPU GSPMD Sharding Complete: {shard_report}", flush=True)
    else:
        try:
            from transformers import AutoModelForMultimodalLM
            model_cls = AutoModelForMultimodalLM
        except Exception:
            from transformers import AutoModelForCausalLM
            model_cls = AutoModelForCausalLM
        dtype = torch.bfloat16 if active_backend == "cuda" else torch.float32
        load_kwargs: Dict[str, object] = {
            "dtype": dtype,
            "device_map": "auto" if active_backend == "cuda" else None,
        }
        if args.load_in_4bit and active_backend == "cuda":
            from transformers import BitsAndBytesConfig

            compute_dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dt,
            )
        model = model_cls.from_pretrained(model_source, **load_kwargs)
        model.eval()
        for p in model.parameters():
            if p.is_floating_point():
                p.requires_grad_(False)

    num_layers = len(resolve_decoder_layers(model))
    cfg = Gemma4SteeringConfig.for_model(args.model_id, num_hidden_layers=num_layers)
    print(
        f"[STAGE 14] Model ready in {time.perf_counter() - t0:.1f}s "
        f"(layers={num_layers}, target_layers={cfg.target_layers}). Preparing subspace_v6...",
        flush=True,
    )

    cache_path = Path(args.subspace_cache)
    subspace = None
    if cache_path.is_file():
        print(f"[STAGE 14] Loading cached calibrated subspace from {cache_path}...", flush=True)
        loaded = CalibratedSubspaceBundle.load(cache_path)
        if (
            loaded.format_version == "limbic0.gemma4_subspace.v6"
            and loaded.model_id == args.model_id
            and len(loaded.embodiment_directions) == len(cfg.target_layers)
        ):
            subspace = loaded
        else:
            print(
                f"[STAGE 14] Cached subspace mismatch (format={loaded.format_version}, model={loaded.model_id}). "
                f"Re-extracting subspace_v6...",
                flush=True,
            )

    if subspace is None:
        print(
            f"[STAGE 14] Extracting 6D Somatic Subspace + Everyday Human Speech Carrier on Layers {cfg.target_layers}...",
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
            f"[STAGE 14] Calibrated subspace_v6 saved! target_layers={subspace.target_layers}, "
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
            print(f"[STAGE 14] Resuming unified organism HUMAN-01 from evolved checkpoint: {soul_ckpt_path}", flush=True)
        else:
            soul_ckpt_path = souls_dir / "human01_ordinary_evolved.soul.pt"
            print(f"[STAGE 14] Hydrating unified organism HUMAN-01 from seed capsule: {soul_ckpt_path}", flush=True)

        souls = {
            "HUMAN-01": LivingSoulOrganism(
                str(soul_ckpt_path),
                label_ko="진화한 일상 인격체",
            ),
        }

    tpu_gen: Optional[TPUBucketedStaticGenerator] = None
    if active_backend == "tpu_spmd":
        tpu_gen = TPUBucketedStaticGenerator(
            model=model,
            spmd_ctx=spmd_ctx,
            max_cache_len=args.max_cache_len,
        )

    surgeon = Gemma4VocalCordSurgeon(
        model=model,
        subspace=subspace,
        config=cfg,
        tpu_generator=tpu_gen,
    )
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

    if tpu_gen is not None and args.warmup_buckets.strip():
        warmup_list = [int(x.strip()) for x in args.warmup_buckets.split(",") if x.strip().isdigit()]
        if warmup_list:
            print(f"[STAGE 14] Pre-compiling XLA GSPMD prefill+decode graphs for buckets {warmup_list}...", flush=True)
            primary_soul = hub.souls[hub.primary_soul_id]
            w_drive, _ = primary_soul.compute_6d_steering_vector()

            def _prep_warmup(d_len: int) -> None:
                surgeon._prepare_persistent_xla_turn_buffers(
                    drive_6d=w_drive,
                    steer_gain=1.0,
                    steer_enabled=True,
                    prompt_len=d_len,
                    is_alive=True,
                    motor_decision_mode=False,
                )

            with surgeon.gen_lock, hub.cortical_surgeon.activate_soul_cortex(hub.primary_soul_id, enabled=True):
                try:
                    w_timings = tpu_gen.warmup_xla_graphs(buckets=warmup_list, prepare_hooks_fn=_prep_warmup)
                finally:
                    surgeon._xla_active_gate.zero_()
            print(f"[STAGE 14] XLA GSPMD Warmup Complete (0s live compile latency): {w_timings}", flush=True)

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
        _start_cloudflare_quick_tunnel(args.port, hub, tunnel_url_file=args.tunnel_url_file)

    srv = VocalCordHTTPServer((args.host, args.port), hub=hub)
    print(f"[STAGE 14] UNIFIED HUMAN SOUL SERVER LIVE ON http://{args.host}:{args.port} (backend={active_backend})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        hub.stop()
        srv.server_close()


if __name__ == "__main__":
    main()

