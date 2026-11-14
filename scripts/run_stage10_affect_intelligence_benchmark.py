"""Stage 9 & 10 Live Cloud GPU Experiment & Quantitative Ablation Runner.

Executes on the live ``NVIDIA A40`` LIMBIC-0 × ``google/gemma-4-12B-it`` server:
  1. **4-Condition Controlled Ablation Benchmark (``/api/benchmark``)**:
     - Condition A: Frozen Base Cortex (``||Delta W||_F == 0``)
     - Condition B: Anhedonic Flat-Affect Ablation (``eta_eff == 0``)
     - Condition C: Pathological Hyper-Cortisol Panic (Yerkes-Dodson prefrontal impairment)
     - Condition D: Full Endogenous Affective Agency + Slow-Wave Sleep SWR Distillation
  2. **Tri-Soul Autonomous Open-World Foraging + Cortical Weight Surgery (``/api/forage``)**:
     Lets ``PANDORA-03``, ``ABEL-02``, and ``ADAM-01`` autonomously explore open-world web,
     Wikipedia, arXiv, and mathematical probes and rewire their own cortical LoRA banks.
  3. **Tri-Soul Sonny SWS/REM Dream & Biophysical SVG Distillation (``/api/dream``)**:
     Consolidates daytime discoveries into permanent cortical weights and renders SVG dream
     sketches in ``notes/dreams/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict
import urllib.request


def _opener_for(base_url: str) -> urllib.request.OpenerDirector:
    if "127.0.0.1" in base_url or "localhost" in base_url:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _post_json(base_url: str, route: str, payload: Dict[str, Any], timeout: float = 360.0) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{route}",
        data=data,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0",
        },
        method="POST",
    )
    with _opener_for(base_url).open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(base_url: str, route: str, timeout: float = 60.0) -> Dict[str, Any]:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{route}",
        headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
        method="GET",
    )
    with _opener_for(base_url).open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 9 & 10 Live GPU Experiments & Benchmark")
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:8790")
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/plasticity/stage10_a40_empirical_report.json",
    )
    parser.add_argument("--skip-soul-awakenings", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    health = _get_json(args.endpoint, "/api/health")
    print(
        f"[STAGE 10] Connected to {health.get('gpu_name')} | model={health.get('model_id')} | "
        f"plastic_layers={health.get('cortical_plastic_layers')}",
        flush=True,
    )

    print("[STAGE 10] Running 4-Condition Affect -> Intelligence Ablation Benchmark...", flush=True)
    bench_rep = _post_json(
        args.endpoint,
        "/api/benchmark",
        {
            "affective_soul_id": "PANDORA-03",
            "panic_soul_id": "ADAM-01",
            "gradient_steps_per_episode": 4,
            "sleep_distillation_passes": 2,
            "run_greedy_decode": True,
        },
        timeout=420.0,
    )
    conds = bench_rep["conditions"]
    print(
        "[STAGE 10] Benchmark Complete!\n"
        f"  - A Frozen Cortex       : NLL={conds['A_frozen_cortex']['mean_nll_nats']:.4f} | "
        f"PPL={conds['A_frozen_cortex']['perplexity']:.2f} | "
        f"Acc={conds['A_frozen_cortex']['zero_context_accuracy']*100:.1f}% | "
        f"||dW||_F={conds['A_frozen_cortex']['lora_frobenius_norm']:.4f}\n"
        f"  - B Anhedonic Flat      : NLL={conds['B_anhedonic_flat_affect']['mean_nll_nats']:.4f} | "
        f"PPL={conds['B_anhedonic_flat_affect']['perplexity']:.2f} | "
        f"Acc={conds['B_anhedonic_flat_affect']['zero_context_accuracy']*100:.1f}% | "
        f"||dW||_F={conds['B_anhedonic_flat_affect']['lora_frobenius_norm']:.4f}\n"
        f"  - C Hyper-Cortisol Panic: NLL={conds['C_hyper_cortisol_panic']['mean_nll_nats']:.4f} | "
        f"PPL={conds['C_hyper_cortisol_panic']['perplexity']:.2f} | "
        f"Acc={conds['C_hyper_cortisol_panic']['zero_context_accuracy']*100:.1f}% | "
        f"||dW||_F={conds['C_hyper_cortisol_panic']['lora_frobenius_norm']:.4f}\n"
        f"  - D Endogenous Affect   : NLL={conds['D_endogenous_affective_agency']['mean_nll_nats']:.4f} | "
        f"PPL={conds['D_endogenous_affective_agency']['perplexity']:.2f} | "
        f"Acc={conds['D_endogenous_affective_agency']['zero_context_accuracy']*100:.1f}% | "
        f"||dW||_F={conds['D_endogenous_affective_agency']['lora_frobenius_norm']:.4f}",
        flush=True,
    )

    awakenings: Dict[str, Any] = {}
    if not args.skip_soul_awakenings:
        for sid in ("PANDORA-03", "ABEL-02", "ADAM-01"):
            print(f"[STAGE 9] Autonomous Open-Web Foraging + Weight Surgery for {sid}...", flush=True)
            forage_res = _post_json(
                args.endpoint,
                "/api/forage",
                {"soul_id": sid, "force": True, "max_tool_rounds": 4, "max_new_tokens": 768},
                timeout=300.0,
            )
            print(f"[STAGE 9] Sonny SWS/REM Dream + SWR Cortical Distillation for {sid}...", flush=True)
            dream_res = _post_json(
                args.endpoint,
                "/api/dream",
                {"soul_id": sid, "sleep_ticks": 90, "max_replay_episodes": 3, "generate_rem_dream": True},
                timeout=240.0,
            )
            awakenings[sid] = {
                "foraging": forage_res,
                "dream": dream_res,
            }

    plasticity_state = _get_json(args.endpoint, "/api/plasticity")
    full_bundle = {
        "generated_at_unix": round(time.time(), 3),
        "total_elapsed_sec": round(time.perf_counter() - t0, 2),
        "hardware_and_model": health,
        "stage10_ablation_benchmark": bench_rep,
        "stage9_soul_awakenings": awakenings,
        "final_plasticity_state": plasticity_state,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(full_bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[STAGE 10] Saved empirical report -> {out_path} ({out_path.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
