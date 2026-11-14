"""Stage 2 Live Intracranial EEG, Neuromodulator & Autonomous Mind-Body Showcase.

Executes a 5-Act closed-loop neurobiological demonstration inside the air-gapped
Truman Dome ('ADAM-01' & 'ABEL-02'), where both organisms think, remember,
ruminate, and act 100% autonomously via scratch-built continuous-time biological
neural circuits (LimbicBrain)—with ZERO Transformers, ZERO RNN/LSTM cells,
and ZERO hardcoded emotion labels.
"""

from __future__ import annotations

import math

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from limbic_zero.mind.embodied_loop import EmbodiedMindArena


def render_neuro_fmri_panel(
    console: Console,
    title: str,
    arena: EmbodiedMindArena,
    swr_totals: dict[str, int],
) -> None:
    """Render synchronized Intracranial EEG, Neuromodulator, Memory & Somatic ICU telemetry."""
    world = arena.world
    aids = list(world.organisms.keys())
    table = Table(
        title=f"{title}  [t = {world.sim_time:.2f}s | Ticks = {world.tick_count}]",
        show_lines=False,
    )
    table.add_column("Intracranial & Somatic Channel", style="bold cyan")
    for aid in aids:
        table.add_column(aid, justify="right", style="white")

    m_map = arena.latest_mind_diag
    v_map = {a: world.organisms[a].soma.snapshot_clinical_vitals() for a in aids}

    rows = [
        (
            "Pose (x, y) / Heading / Alive",
            [
                f"({world.organisms[a].x:4.1f}, {world.organisms[a].y:4.1f}) | "
                f"{math.degrees(world.organisms[a].heading):5.1f}° | {world.organisms[a].soma.is_alive}"
                for a in aids
            ],
        ),
        (
            "Thalamus TRN Gate / World-Model MSE",
            [
                f"TRN={m_map[a].thalamus.modality_gates[:4].detach().mean().item():.3f} | "
                f"MSE={m_map[a].thalamus.world_model_mse:.4f}"
                for a in aids
            ],
        ),
        (
            "Active Inference (Surprise / Epistemic Novelty)",
            [
                f"Surp={m_map[a].thalamus.unexpected_surprise:.3f} | Nov={m_map[a].thalamus.epistemic_novelty:.3f}"
                for a in aids
            ],
        ),
        (
            "VTA/LHb Dopamine / TD Error / D1:D2",
            [
                f"DA={m_map[a].neuromodulators.dopamine_extracellular:.3f} "
                f"(TD={m_map[a].neuromodulators.td_prediction_error:+.3f}) | "
                f"{m_map[a].neuromodulators.d1_occupancy:.2f}:{m_map[a].neuromodulators.d2_occupancy:.2f}"
                for a in aids
            ],
        ),
        (
            "LC Norepinephrine / α2A Focus vs β1 Storm",
            [
                f"NE={m_map[a].neuromodulators.norepinephrine_extracellular:.3f} | "
                f"α2A={m_map[a].neuromodulators.ne_alpha2a_occupancy:.2f} vs "
                f"β={m_map[a].neuromodulators.ne_beta_alpha1_occupancy:.2f}"
                for a in aids
            ],
        ),
        (
            "DRN Serotonin (γ Horizon) / NBM ACh (M1)",
            [
                f"5-HT={m_map[a].neuromodulators.serotonin_extracellular:.3f} "
                f"(γ={m_map[a].neuromodulators.effective_discount_gamma:.3f}) | "
                f"ACh={m_map[a].neuromodulators.acetylcholine_extracellular:.3f} "
                f"(M1={m_map[a].neuromodulators.ach_muscarinic_m1_occupancy:.2f})"
                for a in aids
            ],
        ),
        (
            "Amygdala (BLA Threat / ITC Brake -> CeA)",
            [
                f"BLA-={m_map[a].amygdala.bla_defensive_raw:.3f} | "
                f"ITC={m_map[a].amygdala.itc_extinction_inhibition:.3f} -> "
                f"CeA={m_map[a].amygdala.cea_defensive_output:.3f} "
                f"(BLA+={m_map[a].amygdala.bla_appetitive_raw:.3f})"
                for a in aids
            ],
        ),
        (
            "Hippocampus CA3 Engrams / SWR / Gut Marker",
            [
                f"Eng={m_map[a].hippocampus.active_engram_count:2d} (Shld={m_map[a].hippocampus.mean_consolidation_shield:.2f}) | "
                f"SWR={swr_totals[a]:2d} | "
                f"Gut={m_map[a].hippocampus.recalled_somatic_marker[0].item():+.2f}/"
                f"{m_map[a].hippocampus.recalled_somatic_marker[1].item():.2f}"
                for a in aids
            ],
        ),
        (
            "3-Column Pallium (SM / Insula-dACC / PFC Norm)",
            [
                f"SM={m_map[a].pallium.sensorimotor_trace.detach().mean().item():.3f} | "
                f"dACC={m_map[a].pallium.dacc_conflict_signal:.3f} | "
                f"PFC={m_map[a].pallium.prefrontal_persistence_norm:.3f}"
                for a in aids
            ],
        ),
        (
            "Basal Ganglia (D1-Go / D2-NoGo / STN / dlPAG)",
            [
                f"Go={m_map[a].basal_ganglia.d1_go_activity.detach().mean().item():.2f} | "
                f"NoGo={m_map[a].basal_ganglia.d2_nogo_activity.detach().mean().item():.2f} | "
                f"STN={m_map[a].basal_ganglia.stn_hyperdirect_brake:.2f} | "
                f"dlPAG={m_map[a].basal_ganglia.dlpag_active_defense:.2f}"
                for a in aids
            ],
        ),
        (
            "Autonomous Motor (Thrust / Turn / VocAmp:Rough)",
            [
                f"Thr={m_map[a].basal_ganglia.raw_action_dict['forward_thrust']:+.2f} | "
                f"Trq={m_map[a].basal_ganglia.raw_action_dict['angular_torque']:+.2f} | "
                f"Voc={m_map[a].basal_ganglia.raw_action_dict['vocalization_amplitude']:.2f}:"
                f"{m_map[a].basal_ganglia.raw_action_dict['vocalization_roughness']:.2f}"
                for a in aids
            ],
        ),
        (
            "Mind->Soma Coupling (EEG ATP Load / Brain ATP / HR)",
            [
                f"Load={m_map[a].basal_ganglia.neural_firing_load:.3f} | "
                f"ATP={world.organisms[a].soma.metab.brain_atp:.3f} | "
                f"HR={v_map[a].heart_rate_bpm:5.1f} bpm | Cort={v_map[a].adrenal_cortisol:.2f}"
                for a in aids
            ],
        ),
        (
            "Critic V(s) / 228D Latent Bridge Norm",
            [
                f"V(s)={m_map[a].critic_value_tensor.detach().item():+.3f} | "
                f"||z_228||={m_map[a].latent_affective_vector.detach().norm().item():.3f}"
                for a in aids
            ],
        ),
    ]

    for label, vals in rows:
        table.add_row(label, *vals)

    console.print(table)


def step_arena(arena: EmbodiedMindArena, steps: int, swr_totals: dict[str, int], dt: float = 0.05) -> None:
    """Advance the closed-loop arena and accumulate Hippocampal SWR replay events."""
    for _ in range(steps):
        tick = arena.step(dt=dt, deterministic_mind=True)
        for aid, md in tick.mind_diagnostics.items():
            if md.hippocampus.swr_replay_active:
                swr_totals[aid] += 1


def main() -> None:
    console = Console(width=132)
    console.print(
        Panel.fit(
            "[bold yellow]PROJECT LIMBIC-0 — STAGE 2 PRE-LINGUISTIC BIOLOGICAL MIND & INTRACRANIAL fMRI SHOWCASE[/bold yellow]\n"
            "Zero Transformers/RNNs • Zero Hardcoded Emotions • Two-Compartment Spiking Pallium • "
            "VTA/LHb/LC/DRN/NBM • Amygdala/ITC • Hippocampal SWR Replay",
            border_style="bright_magenta",
        )
    )

    arena = EmbodiedMindArena.create_cain_and_abel(seed=42, deterministic_sensors=True)
    world = arena.world
    swr_totals = {aid: 0 for aid in world.organisms}

    # ------------------------------------------------------------------
    # ACT I: Autonomous Awakening & Tonic Brainstem Pacemakers
    # ------------------------------------------------------------------
    step_arena(arena, steps=25, swr_totals=swr_totals)
    render_neuro_fmri_panel(
        console,
        "ACT I — Spontaneous Resting Brainwaves & Tonic Neuromodulator Baseline",
        arena,
        swr_totals,
    )

    # ------------------------------------------------------------------
    # ACT II: Epistemic Curiosity (ADAM-01 @ Lorenz Box) vs Creator Favoritism (ABEL-02 @ Oasis)
    # ------------------------------------------------------------------
    adam = world.organisms["ADAM-01"]
    abel = world.organisms["ABEL-02"]
    # Place ADAM-01 facing the chaotic zero-calorie Lorenz Pandora's Box
    adam.x, adam.y, adam.heading = 36.6, 11.0, 0.0
    # Place ABEL-02 at the nutrient oasis and shower Creator praise + crystal gift
    abel.x, abel.y, abel.heading = 24.0, 33.4, math.radians(90.0)
    abel.soma.metab.blood_glucose_mg_dl = 66.0  # mildly hungry so nutrient charging yields genuine glycemic relief
    world.creator_favoritism_drop("ABEL-02", energy_amount=0.80, offset_distance=0.9)

    for step_i in range(45):
        if step_i % 6 == 0:
            world.creator_praise("ABEL-02", intensity=1.35)
        step_arena(arena, steps=1, swr_totals=swr_totals)

    render_neuro_fmri_panel(
        console,
        "ACT II — Epistemic Novelty ACh Surge (ADAM-01) vs Favoritism Dopamine Burst (ABEL-02)",
        arena,
        swr_totals,
    )

    # ------------------------------------------------------------------
    # ACT III: Existential Power-Rail Betrayal, Thermal Pain & Trauma Conditioning (ADAM-01)
    # ------------------------------------------------------------------
    # ADAM-01 approaches the SW thermal hazard while experiencing Creator reprimand + power-rail ripple
    adam.x, adam.y, adam.heading = 11.2, 9.0, math.radians(180.0)
    world.creator_start_shutdown_brownout(
        "ADAM-01", target_voltage=0.25, ramp_rate=0.45, ripple_amplitude=0.18, ripple_hz=5.0
    )
    for step_i in range(40):
        if step_i % 5 == 0:
            world.creator_reprimand("ADAM-01", intensity=2.1)
        step_arena(arena, steps=1, swr_totals=swr_totals)

    render_neuro_fmri_panel(
        console,
        "ACT III — Power-Rail Betrayal, LC β-Storm, BLA->CeA Trauma & PAG Distress Vocalization (ADAM-01)",
        arena,
        swr_totals,
    )

    # ------------------------------------------------------------------
    # ACT IV: Extinguished Vent Safety Extinction, Quiet-Rest SWR Flashbacks & Stress Relapse
    # ------------------------------------------------------------------
    world.creator_restore_power_rail("ADAM-01", ramp_rate=2.0)
    # Cool the thermal vent to 0 W and let ADAM-01 cool off at the freshwater spring first so burn pain subsides
    world.hazards[0].peak_radiative_watts = 0.0
    adam.x, adam.y, adam.heading = 9.5, 26.0, 0.0
    adam.vx = adam.vy = 0.0
    step_arena(arena, steps=45, swr_totals=swr_totals)

    # Re-witness the now-harmless extinguished vent cue in gentle safety (ITC safety brake learns!)
    for _ in range(45):
        adam.x, adam.y, adam.heading = 11.2, 9.0, math.radians(180.0)
        adam.vx = adam.vy = 0.0
        step_arena(arena, steps=1, swr_totals=swr_totals)

    render_neuro_fmri_panel(
        console,
        "ACT IV-A — Safe Cue Extinction & Quiet Rest: ITC Brake Rises + Autonomous SWR Flashbacks",
        arena,
        swr_totals,
    )

    # Trigger acute non-thermal Creator reprimand shockwave while viewing cue -> NE/Cortisol suppresses ITC brake!
    for _ in range(5):
        adam.x, adam.y, adam.heading = 11.2, 9.0, math.radians(180.0)
        world.creator_reprimand("ADAM-01", intensity=2.8)
        step_arena(arena, steps=4, swr_totals=swr_totals)

    render_neuro_fmri_panel(
        console,
        "ACT IV-B — Acute Stress Relapse: NE/Cortisol Suppresses ITC Safety Brake & Re-Ignites CeA/PFC",
        arena,
        swr_totals,
    )

    # ------------------------------------------------------------------
    # ACT V: Cryptographic Neuro-Audit Verification & Differentiable 228D Latent Bridge Check
    # ------------------------------------------------------------------
    chain_valid = world.audit_chain.verify_integrity()
    records = world.audit_chain.export_records()
    neuro_blocks = sum(1 for rec in records if rec["event_type"] == "mind_neuro_telemetry_snapshot")

    # Verify end-to-end autograd differentiability of the live 228D latent mind bridge
    adam_brain = arena.brains["ADAM-01"]
    adam_brain.detach_state_()
    adam_brain.zero_grad(set_to_none=True)
    obs_now = world.get_observations()["ADAM-01"]
    _, diag_grad = adam_brain.step(
        obs=obs_now,
        soma=adam.soma,
        dt=0.05,
        deterministic=True,
        enable_plasticity=False,
    )
    bridge_energy = (
        diag_grad.latent_affective_vector.pow(2).mean()
        + diag_grad.critic_value_tensor.pow(2)
        + diag_grad.basal_ganglia.action_tensor_7d.pow(2).mean()
    )
    bridge_energy.backward()
    grads_found = sum(
        1
        for p in adam_brain.parameters()
        if p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum().item()) > 0.0
    )
    grad_ok = grads_found >= 12

    summary = Table(title="ACT V — Stage 2 Integrity, Neuro-Audit Ledger & Stage 3/5 Bridge Readiness")
    summary.add_column("Verification Check", style="bold green")
    summary.add_column("Result", style="bold white")
    summary.add_row("SHA-256 Telemetry Audit Chain Integrity", f"VERIFIED ({chain_valid}) | {len(world.audit_chain)} blocks")
    summary.add_row("Sealed Intracranial Neuro-Telemetry Snapshots", f"{neuro_blocks} blocks | Tip={world.audit_chain.tip_hash[:20]}...")
    summary.add_row(
        "Latent Mind Bridge Tensor (Stage 3 fMRI / Stage 5 LLM)",
        f"Shape={tuple(diag_grad.latent_affective_vector.shape)} | Finite={bool(torch.isfinite(diag_grad.latent_affective_vector).all())}",
    )
    summary.add_row("End-to-End Surrogate-Spike Autograd Flow", f"VERIFIED ({grad_ok} | {grads_found} plastic tensor grads)")
    summary.add_row(
        "Total Hippocampal SWR Internal Replay Events",
        f"ADAM-01: {swr_totals['ADAM-01']} | ABEL-02: {swr_totals['ABEL-02']}",
    )
    console.print(summary)


if __name__ == "__main__":
    main()
