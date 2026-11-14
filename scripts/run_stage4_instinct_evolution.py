"""Stage 4 End-to-End Runner: Pre-Linguistic Wild Instinct Evolution & LLM Soul Capsule Maturation.

Executes all three biological phases with ZERO emotion labels, ZERO designer reward hacks,
and ZERO Transformer/RNN shortcuts:
    Phase I   — Phylogenetic Baldwin Natural Evolution (Antithetic Rank-NES over Zador
                Morphogenetic Connectome Bottleneck across mirror-paired foraging,
                radiative thermal-vent escape, and conspecific scarcity duels).
    Phase II  — Multi-Season Ontogenetic Biographical Nursery + Molecular Epigenetics
                (NR3C1 methylation, 5-HT1A, D1, NE-beta, ACh-M1, OXTR) + Offline Non-REM
                Slow-Wave Sleep & CA3 Sharp-Wave Ripple Consolidation, raising THREE
                identical genetic clones into diverged pre-linguistic personalities:
                  1. ``adam01_forsaken_survivor.soul.pt``
                  2. ``abel02_favored_guardian.soul.pt``
                  3. ``pandora03_curious_seeker.soul.pt``
    Phase III — 5-Gate Instinct & Personality Validation Suite + Live PyTorch
                ``register_forward_hook`` LLM Residual-Stream Injection Verification.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Dict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import torch
import torch.nn as nn

from limbic_zero.evolution import (
    BIOPHYSICAL_COORD_NAMES,
    EcologicalEvalConfig,
    InstinctValidationSuite,
    MaturedSoulCapsule,
    MorphogeneticGenomeConfig,
    NurseryConfig,
    OntogeneticBiographicalNursery,
    PhylogeneticESConfig,
    PhylogeneticInstinctEvolver,
)


def _verify_live_llm_residual_hook(
    capsules: Dict[str, MaturedSoulCapsule],
    hidden_dim: int = 896,
) -> Dict[str, Any]:
    """Plug all three matured Soul Capsules into a frozen mock LLM decoder stack via forward hooks."""

    class FrozenMockDecoderBlock(nn.Module):
        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.SiLU(),
                nn.Linear(d_model * 2, d_model),
            )
            for p in self.parameters():
                p.requires_grad_(False)

        def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, str]:
            out = hidden_states + 0.15 * self.ffn(self.norm(hidden_states))
            return out, "kv_cache_stub"

    torch.manual_seed(202605)
    frozen_block = FrozenMockDecoderBlock(hidden_dim)
    # Identical neutral token embedding sequence fed into the frozen decoder block
    neutral_tokens = torch.randn(1, 6, hidden_dim) * 0.35
    base_out, _ = frozen_block(neutral_tokens)

    hook_outputs: Dict[str, torch.Tensor] = {}
    soft_prefixes: Dict[str, torch.Tensor] = {}
    live_coords: Dict[str, Dict[str, float]] = {}

    for name, cap in capsules.items():
        # Use identical socket seed=777 so projector matrices start identical across all 3 capsules!
        socket = cap.create_llm_socket(llm_hidden_dim=hidden_dim, num_soft_prefix_tokens=4, seed=777)
        # Tick 12 autonomous heartbeats under mild acoustic startle so each soul reacts from its own history
        for _ in range(12):
            socket.step_autonomous_heartbeat(dt=0.05, acoustic_roughness_shock=0.65)
        _, coords = socket.current_bridge_features()
        live_coords[name] = {k: round(v, 4) for k, v in coords.items()}

        handle = socket.attach_residual_hook(frozen_block, alpha=1.0)
        steered_out, _ = frozen_block(neutral_tokens)
        handle.remove()
        socket.detach_all_hooks()

        hook_outputs[name] = steered_out.detach()
        soft_prefixes[name] = socket.compute_soft_prefix_tokens(batch_size=1).detach()

    pairwise_hidden_l2: Dict[str, float] = {}
    keys = list(hook_outputs.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ka, kb = keys[i], keys[j]
            dist = float(torch.linalg.vector_norm(hook_outputs[ka] - hook_outputs[kb]).item())
            pairwise_hidden_l2[f"{ka}_vs_{kb}"] = round(dist, 4)

    shift_from_unsteered = {
        k: round(float(torch.linalg.vector_norm(v - base_out).item()), 4)
        for k, v in hook_outputs.items()
    }
    prefix_shapes = {k: list(v.shape) for k, v in soft_prefixes.items()}

    return {
        "target_llm_hidden_dim": hidden_dim,
        "shift_l2_from_unsteered_frozen_block": shift_from_unsteered,
        "pairwise_steered_residual_l2_divergence": pairwise_hidden_l2,
        "soft_prefix_shapes": prefix_shapes,
        "live_biophysical_coordinates_under_probe": live_coords,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Project LIMBIC-0 Stage 4 Instinct Evolution & Soul Maturation")
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--ticks-per-trial", type=int, default=125)
    parser.add_argument("--seasons", type=int, default=4)
    parser.add_argument("--waking-ticks", type=int, default=240)
    parser.add_argument("--sleep-ticks", type=int, default=80)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=str, default="checkpoints/souls")
    parser.add_argument("--resume-genome", type=str, default="", help="Load an already-evolved .genome.pt checkpoint for Phase II/III")
    parser.add_argument("--export-report", type=str, default="")
    args = parser.parse_args()

    console = Console()
    t_start = time.perf_counter()

    console.print(
        Panel.fit(
            "[bold cyan]PROJECT LIMBIC-0 // STAGE 4[/bold cyan]\n"
            "Pre-Linguistic Wild Instinct Evolution, Epigenetic Methylation,\n"
            "Non-REM Slow-Wave Sleep SWR Consolidation & LLM Soul Capsule Synthesis",
            border_style="cyan",
        )
    )

    genome_cfg = MorphogeneticGenomeConfig()
    eval_cfg = EcologicalEvalConfig(ticks_per_trial=args.ticks_per_trial)
    es_cfg = PhylogeneticESConfig(
        population_pairs=args.pairs,
        generations=args.generations,
        seed=args.seed,
        num_workers=args.workers,
        include_contested_duel=True,
    )

    evolver = PhylogeneticInstinctEvolver(
        es_config=es_cfg,
        genome_config=genome_cfg,
        eval_config=eval_cfg,
    )
    founder_genome = evolver.mean_genome.clone()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    genome_path = out_dir / "limbic0_species_instinct.genome.pt"

    gen_table = Table(title="Phase I — Phylogenetic Baldwin Natural Evolution (Zero Designer Rewards)")
    gen_table.add_column("Gen", justify="right", style="cyan")
    gen_table.add_column("Pop Mean J", justify="right")
    gen_table.add_column("Pop Best J", justify="right")
    gen_table.add_column("Champion J", justify="right", style="bold green")
    gen_table.add_column("Champ Mean F(s)", justify="right", style="yellow")
    gen_table.add_column("Survival %", justify="right")
    gen_table.add_column("||g||_2", justify="right")
    gen_table.add_column("Sec", justify="right")

    if args.resume_genome:
        resume_path = Path(args.resume_genome)
        genome_meta = torch.load(resume_path, map_location="cpu", weights_only=False)
        champion_genome = genome_meta["champion_genome"].float().clone()
        for rec_dict in genome_meta.get("history", []):
            gen_table.add_row(
                str(rec_dict["generation"]),
                f"{rec_dict['pop_mean_fitness']:+.3f}",
                f"{rec_dict['pop_best_fitness']:+.3f}",
                f"{rec_dict['champion_eval_fitness']:+.3f}",
                f"{rec_dict['champion_mean_free_energy']:.3f}",
                f"{rec_dict['pop_survival_rate'] * 100.0:.1f}%",
                f"{rec_dict['genome_l2_norm']:.2f}",
                f"{rec_dict['elapsed_sec']:.2f}",
            )
        console.print(f"[green]Loaded evolved species genome from {resume_path} (Champion J={genome_meta['champion_fitness']:+.4f})[/green]")
        console.print(gen_table)
        if resume_path.resolve() != genome_path.resolve():
            torch.save(genome_meta, genome_path)
    else:
        def _on_gen(rec: Any) -> None:
            console.print(
                f"  [cyan]Gen {rec.generation:02d}/{args.generations:02d}[/cyan] | "
                f"pop_mean={rec.pop_mean_fitness:+.3f} | pop_best={rec.pop_best_fitness:+.3f} | "
                f"[bold green]champ={rec.champion_eval_fitness:+.3f}[/bold green] | "
                f"mean_F={rec.champion_mean_free_energy:.3f} | surv={rec.pop_survival_rate * 100.0:.1f}% | "
                f"{rec.elapsed_sec:.2f}s"
            )
            gen_table.add_row(
                str(rec.generation),
                f"{rec.pop_mean_fitness:+.3f}",
                f"{rec.pop_best_fitness:+.3f}",
                f"{rec.champion_eval_fitness:+.3f}",
                f"{rec.champion_mean_free_energy:.3f}",
                f"{rec.pop_survival_rate * 100.0:.1f}%",
                f"{rec.genome_l2_norm:.2f}",
                f"{rec.elapsed_sec:.2f}",
            )
            evolver.save_species_genome(str(genome_path))

        champion_genome = evolver.run(progress_callback=_on_gen)
        console.print(gen_table)
        genome_meta = evolver.save_species_genome(str(genome_path))

    console.print("\n[bold magenta]Phase II — Multi-Season Ontogenetic Biographical Maturation & Sleep Consolidation[/bold magenta]")
    nursery = OntogeneticBiographicalNursery(
        expressor=evolver.expressor,
        nursery_config=NurseryConfig(
            seasons=args.seasons,
            waking_ticks_per_season=args.waking_ticks,
            sleep_ticks_per_night=args.sleep_ticks,
        ),
    )

    adam_cap, abel_cap, twin_meta = nursery.mature_cain_and_abel_twins(
        species_genome=champion_genome,
        arena_seed=args.seed + 404,
    )
    pandora_cap = nursery.mature_pandora_explorer(
        species_genome=champion_genome,
        arena_seed=args.seed + 808,
    )

    adam_pt, adam_json = adam_cap.save(out_dir / "adam01_forsaken_survivor.soul.pt")
    abel_pt, abel_json = abel_cap.save(out_dir / "abel02_favored_guardian.soul.pt")
    pand_pt, pand_json = pandora_cap.save(out_dir / "pandora03_curious_seeker.soul.pt")

    # Reload from disk to verify cryptographic SHA-256 integrity seal
    adam_loaded = MaturedSoulCapsule.load(adam_pt, verify_integrity=True)
    abel_loaded = MaturedSoulCapsule.load(abel_pt, verify_integrity=True)
    pand_loaded = MaturedSoulCapsule.load(pand_pt, verify_integrity=True)

    soul_table = Table(title="Matured Pre-Linguistic Soul Capsules (Identical DNA at t=0 -> Diverged Personalities)")
    soul_table.add_column("Organism", style="bold cyan")
    soul_table.add_column("Regime")
    soul_table.add_column("NR3C1 Meth", justify="right")
    soul_table.add_column("5-HT1A Expr", justify="right")
    soul_table.add_column("NE-beta Expr", justify="right")
    soul_table.add_column("ACh-M1 Expr", justify="right")
    soul_table.add_column("BLA Def / App", justify="right")
    soul_table.add_column("CA3 Engrams (SWR)", justify="right")
    soul_table.add_column("Connectome ΔL2", justify="right", style="green")

    for cap in (adam_loaded, abel_loaded, pand_loaded):
        ep = cap.epigenetic_profile
        nm = cap.neuromodulator_resting_summary
        div_l2 = cap.connectome_divergence_from_birth["total_connectome_l2"]
        soul_table.add_row(
            cap.organism_id,
            cap.biographical_regime,
            f"{ep.nr3c1_gr_methylation:.3f}",
            f"{ep.ht1a_promoter_expression:.3f}",
            f"{ep.ne_beta_expression:.3f}",
            f"{ep.ach_m1_expression:.3f}",
            f"{nm['bla_defensive_norm']:.2f} / {nm['bla_appetitive_norm']:.2f}",
            f"{len(cap.consolidated_engrams)} ({cap.total_swr_consolidations})",
            f"{div_l2:.3f}",
        )
    console.print(soul_table)

    console.print("\n[bold yellow]Phase III — 5-Gate Instinct Validation & Live LLM Residual-Stream Socket Test[/bold yellow]")
    suite = InstinctValidationSuite(expressor=evolver.expressor)
    gates = suite.run_all_gates(
        founder_genome=founder_genome,
        evolved_genome=champion_genome,
        adam_capsule=adam_loaded,
        abel_capsule=abel_loaded,
        pandora_capsule=pand_loaded,
        seed=args.seed + 555,
    )

    gate_table = Table(title="Stage 4 Quantitative Validation Gates")
    gate_table.add_column("Gate", style="cyan")
    gate_table.add_column("Status", justify="center")
    gate_table.add_column("Key Physiological / Neural Evidence")
    for g_name, g_res in gates.items():
        badge = "[bold green]PASS[/bold green]" if g_res.passed else "[bold red]FAIL[/bold red]"
        gate_table.add_row(g_name, badge, json.dumps(g_res.metrics))
    console.print(gate_table)

    llm_hook_report = _verify_live_llm_residual_hook(
        {
            "ADAM-01": adam_loaded,
            "ABEL-02": abel_loaded,
            "PANDORA-03": pand_loaded,
        },
        hidden_dim=896,  # Matches Qwen2.5-0.5B hidden_size=896!
    )
    console.print(
        Panel(
            json.dumps(llm_hook_report, indent=2),
            title="Live PyTorch register_forward_hook Injection into d_model=896 (Qwen2.5-0.5B Compatible)",
            border_style="green",
        )
    )

    total_elapsed = time.perf_counter() - t_start
    console.print(
        f"\n[bold green]Stage 4 Complete in {total_elapsed:.2f}s![/bold green] "
        f"Saved capsules:\n  - {adam_pt} ({adam_json.name})\n  - {abel_pt} ({abel_json.name})\n  - {pand_pt} ({pand_json.name})\n  - {genome_path}"
    )

    if args.export_report:
        rep_path = Path(args.export_report)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        full_report = {
            "elapsed_seconds": round(total_elapsed, 3),
            "species_evolution": {
                "founder_fitness": round(float(genome_meta["founder_initial_fitness"]), 4),
                "champion_fitness": round(float(genome_meta["champion_fitness"]), 4),
                "genome_dim": int(genome_meta["genome_dim"]),
                "history": genome_meta["history"],
            },
            "twin_maturation": twin_meta,
            "capsules": {
                cap.organism_id: {
                    "pt_path": str((out_dir / fname).resolve()),
                    "sha256": cap.capsule_sha256,
                    "epigenetic_profile": cap.epigenetic_profile.to_dict(),
                    "neuromodulator_resting_summary": cap.neuromodulator_resting_summary,
                    "connectome_divergence_from_birth": cap.connectome_divergence_from_birth,
                    "cortical_participation_ratio": cap.calibration.cortical_participation_ratio,
                    "engram_count": len(cap.consolidated_engrams),
                    "total_swr_consolidations": cap.total_swr_consolidations,
                    "biophysical_coord_names": list(BIOPHYSICAL_COORD_NAMES),
                    "resting_attractor_preview_16d": [
                        round(float(x), 4)
                        for x in cap.calibration.resting_attractor_228d[:16].tolist()
                    ],
                }
                for cap, fname in (
                    (adam_loaded, "adam01_forsaken_survivor.soul.pt"),
                    (abel_loaded, "abel02_favored_guardian.soul.pt"),
                    (pand_loaded, "pandora03_curious_seeker.soul.pt"),
                )
            },
            "validation_gates": {k: asdict(v) for k, v in gates.items()},
            "llm_hook_verification": llm_hook_report,
        }
        rep_path.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
