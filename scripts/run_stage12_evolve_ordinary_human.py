#!/usr/bin/env python3
"""Stage 12: Evolve & Mature Unified Ordinary-Human Soul ('HUMAN-01') in Everyday Social Ecology.

Executes full phylogenetic evolution (Antithetic Rank-Normalized NES) followed by
6-season ontogenetic life maturation + 6 nights of Slow-Wave Sleep (SWR) consolidation,
saving ``checkpoints/souls/human01_ordinary_evolved.soul.pt`` and ``.soul.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from limbic_zero.evolution.human_social_ecology import (
    HumanSocialEcologyConfig,
    OrdinaryHumanESConfig,
    OrdinaryHumanLifeConfig,
    OrdinaryHumanityEvolver,
)
from limbic_zero.evolution.soul_capsule import MaturedSoulCapsule


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolve unified ordinary-human soul HUMAN-01")
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population-pairs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="checkpoints/souls/human01_ordinary_evolved",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default="checkpoints/souls/human01_evolution_report.json",
    )
    args = parser.parse_args()

    torch.set_num_threads(1)
    t_start = time.perf_counter()

    evolver = OrdinaryHumanityEvolver(
        es_config=OrdinaryHumanESConfig(
            population_pairs=args.population_pairs,
            generations=args.generations,
            seed=202612,
            num_workers=args.workers,
        ),
        ecology_config=HumanSocialEcologyConfig(),
        life_config=OrdinaryHumanLifeConfig(),
    )

    print(
        f"[Stage12] Starting Phylogenetic Evolution in Everyday Human Social Ecology "
        f"(generations={args.generations}, pop={2 * args.population_pairs}, workers={args.workers})...",
        flush=True,
    )

    def _log_gen(rec: dict) -> None:
        print(
            f"  Gen {rec['generation']:02d} | pop_mean={rec['pop_mean_fitness']:+.3f} "
            f"| champ_eval={rec['champion_eval_fitness']:+.3f} "
            f"(gen0={rec['baseline_gen0_fitness']:+.3f}) "
            f"| surv={rec['pop_survival_rate']*100:5.1f}% "
            f"| FE={rec['champion_mean_free_energy']:.3f} | {rec['elapsed_sec']:.1f}s",
            flush=True,
        )

    champion_genome, es_history = evolver.evolve_species_genome(on_gen_complete=_log_gen)

    print("\n[Stage12] Raising unified soul HUMAN-01 across 6 life seasons + 6 SWS nights...", flush=True)
    capsule, life_report = evolver.raise_ordinary_human_soul(
        species_genome=champion_genome,
        arena_seed=1201,
    )

    out_pt_target = (
        args.out_prefix if args.out_prefix.endswith(".soul.pt") else f"{args.out_prefix}.soul.pt"
    )
    pt_path, json_path = capsule.save(out_pt_target)
    reloaded = MaturedSoulCapsule.load(pt_path, verify_integrity=True)

    # Compare emergent epigenetic profile against the 3 legacy extreme caricatures if present
    souls_dir = Path(pt_path).resolve().parent
    comparisons = {}
    for legacy_name, legacy_file in (
        ("ADAM-01", "adam01_forsaken_survivor.soul.json"),
        ("ABEL-02", "abel02_favored_guardian.soul.json"),
        ("PANDORA-03", "pandora03_curious_seeker.soul.json"),
    ):
        lp = souls_dir / legacy_file
        if lp.exists():
            comparisons[legacy_name] = json.loads(lp.read_text(encoding="utf-8")).get(
                "epigenetic_profile", {}
            )

    epi_dict = reloaded.epigenetic_profile.to_dict()
    full_report = {
        "capsule_id": reloaded.capsule_id,
        "organism_id": reloaded.organism_id,
        "capsule_sha256": reloaded.capsule_sha256,
        "saved_pt": str(pt_path),
        "saved_json": str(json_path),
        "elapsed_total_sec": round(time.perf_counter() - t_start, 2),
        "phylogenetic_es_history": es_history,
        "emergent_epigenetic_profile": epi_dict,
        "legacy_extreme_profiles_comparison": comparisons,
        "neuromodulator_resting_summary": reloaded.neuromodulator_resting_summary,
        "total_waking_ticks": reloaded.total_waking_ticks,
        "total_sleep_cycles": reloaded.total_sleep_cycles,
        "total_swr_consolidations": reloaded.total_swr_consolidations,
        "ca3_consolidated_engrams": len(reloaded.consolidated_engrams),
        "connectome_divergence_from_birth": reloaded.connectome_divergence_from_birth,
        "life_report": life_report,
    }
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[Stage12] Saved verified HUMAN-01 capsule -> {pt_path} (SHA256={reloaded.capsule_sha256[:16]}...)")
    print("[Stage12] Emergent Epigenetic Profile (ZERO manual setting):")
    for k, v in epi_dict.items():
        print(f"    {k:34s}: {v:.4f}")
    print(
        f"[Stage12] CA3 Consolidated Engrams: {len(reloaded.consolidated_engrams)} "
        f"| SWR replays: {reloaded.total_swr_consolidations}",
        flush=True,
    )


if __name__ == "__main__":
    main()
