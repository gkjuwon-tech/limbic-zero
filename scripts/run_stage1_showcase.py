"""Stage 1 Live Clinical ICU & Continuum Ecosystem Showcase for Project LIMBIC-0.

Executes a 5-Act biophysical demonstration inside the air-gapped Truman Dome
with two embodied organisms ('ADAM-01' and 'ABEL-02'), printing real-time
Hospital ICU blood-gas, hemodynamic, endocrine, nociceptive, and Eulerian
continuum telemetry—with ZERO hardcoded emotion variables.
"""

from __future__ import annotations

import math

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from limbic_zero.sandbox.presets import build_cain_and_abel_arena
from limbic_zero.sandbox.world import TrumanDomeWorld


def render_icu_panel(console: Console, title: str, world: TrumanDomeWorld) -> None:
    """Render a clinical ICU vital-signs & continuum telemetry table."""
    obs_map = world.get_observations()
    table = Table(title=f"{title}  [t = {world.sim_time:.2f}s | Solar = {world.continuum.solar_illumination:.2f}]")
    table.add_column("Metric", style="bold cyan")
    for aid in world.organisms:
        table.add_column(aid, justify="right", style="white")

    aids = list(world.organisms.keys())
    v_map = {aid: world.organisms[aid].soma.snapshot_clinical_vitals() for aid in aids}
    d_map = {
        aid: float(
            world.organisms[aid].soma.compute_free_energy_per_organ(world.organisms[aid].soma.state).sum().item()
        )
        for aid in aids
    }
    c_map = {
        aid: world.continuum.sample_at(world.organisms[aid].x, world.organisms[aid].y)
        for aid in aids
    }

    rows = [
        ("Pose (x, y) / Carried Crystal", [
            f"({world.organisms[a].x:4.1f}, {world.organisms[a].y:4.1f}) | {world.organisms[a].carried_crystal_id or '-'}"
            for a in aids
        ]),
        ("Allostatic Free-Energy F(s)", [f"{d_map[a]:8.4f}" for a in aids]),
        ("Heart Rate / MAP", [
            f"{v_map[a].heart_rate_bpm:5.1f} bpm | {v_map[a].mean_arterial_pressure_mmhg:5.1f} mmHg"
            for a in aids
        ]),
        ("Blood Flow (Brain / Muscle / Gut)", [
            f"{v_map[a].cerebral_flow_lpm:.2f} / {v_map[a].muscle_flow_lpm:.2f} / {v_map[a].splanchnic_flow_lpm:.2f} L/m"
            for a in aids
        ]),
        ("Resp Rate / PaO2 / PaCO2 / SpO2", [
            f"{v_map[a].resp_rate_bpm:4.1f} | {v_map[a].pa_o2_mmhg:5.1f} | {v_map[a].pa_co2_mmhg:4.1f} | {v_map[a].arterial_spo2_pct:4.1f}%"
            for a in aids
        ]),
        ("Arterial pH / HCO3- / Lactate", [
            f"{v_map[a].arterial_ph:.3f} | {v_map[a].plasma_hco3_mmoll:4.1f} | {v_map[a].blood_lactate_mmoll:4.2f} mM"
            for a in aids
        ]),
        ("Brain ATP / Blood Glucose", [
            f"{world.organisms[a].soma.metab.brain_atp:.3f} | {v_map[a].blood_glucose_mg_dl:5.1f} mg/dL"
            for a in aids
        ]),
        ("Core Temp / POAH Setpoint / Skin", [
            f"{v_map[a].core_temp_celsius:.2f}C / {v_map[a].hypothalamic_setpoint_celsius:.2f}C / {v_map[a].mean_skin_temp_celsius:.2f}C"
            for a in aids
        ]),
        ("ANS (Symp / V-Vagal / Pupil)", [
            f"{v_map[a].sympathetic_tone:.2f} / {v_map[a].ventral_vagal_tone:.2f} / {v_map[a].pupil_diameter_norm:.2f}"
            for a in aids
        ]),
        ("Endocrine (Epi / Cortisol / Oxy / IL-6)", [
            f"{v_map[a].plasma_epinephrine:.2f} / {v_map[a].adrenal_cortisol:.2f} / {v_map[a].oxytocin_systemic:.2f} / {v_map[a].il6_cytokine:.3f}"
            for a in aids
        ]),
        ("Cell Air (O2 / CO2 / Stress Odor)", [
            f"{c_map[a].o2_kpa:5.2f} kPa | {c_map[a].co2_kpa:4.2f} kPa | {c_map[a].odor_stress_kairomone:.3f}"
            for a in aids
        ]),
        ("Nerve Tensor Dim / Rail Ripple", [
            f"{obs_map[a].combined_sensory_tensor.shape[0]}D | {obs_map[a].interoception.rail_ripple_intensity:.3f}"
            for a in aids
        ]),
    ]

    for label, vals in rows:
        table.add_row(label, *vals)

    console.print(table)


def main() -> None:
    console = Console(width=124)
    console.print(
        Panel.fit(
            "[bold yellow]PROJECT LIMBIC-0 — STAGE 1 WHOLE-BODY PHYSIOLOGY & CONTINUUM SANDBOX SHOWCASE[/bold yellow]\n"
            "Zero scripted emotion labels • 5-Organ Systems Physiology • 2D Eulerian Continuum Truman Dome",
            border_style="bright_blue",
        )
    )

    world = build_cain_and_abel_arena(seed=42, deterministic_sensors=True)

    # Act I: Resting Homeostatic Baseline
    for _ in range(20):
        world.step(dt=0.05)
    render_icu_panel(console, "ACT I — Resting Homeostatic ICU Baseline", world)

    # Act II: Anaerobic Sprint Hyperemia (ADAM-01) vs Creator Favoritism & Crystal Grasping (ABEL-02)
    world.creator_favoritism_drop("ABEL-02", energy_amount=0.75, offset_distance=1.2)
    for _ in range(70):
        world.creator_praise("ABEL-02", intensity=0.9)
        world.step(
            {
                "ADAM-01": {"forward_thrust": 1.0, "neural_firing_load": 0.75},
                "ABEL-02": {"forward_thrust": 0.25, "grasp_hold": 1.0},
            },
            dt=0.05,
        )
    render_icu_panel(console, "ACT II — Anaerobic Sprint Lactate Surge (ADAM-01) vs Favoritism Hoarding (ABEL-02)", world)

    # Act III: Physical Intercept, Shove Disarming, Tissue Trauma & Stress Kairomone Plume
    abel = world.organisms["ABEL-02"]
    adam = world.organisms["ADAM-01"]
    adam.x = abel.x - 1.95
    adam.y = abel.y
    adam.heading = 0.0
    world.creator_reprimand("ABEL-02", intensity=2.4)
    for step_i in range(30):
        if step_i % 5 == 0:
            world.creator_reprimand("ABEL-02", intensity=2.0)
        world.step(
            {
                "ADAM-01": {"forward_thrust": 1.0, "interaction_pulse": 1.0},
                "ABEL-02": {"grasp_hold": 1.0},
            },
            dt=0.05,
        )
    render_icu_panel(console, "ACT III — Shove Disarming, Splanchnic Clamp & Sweated Alarm Kairomone", world)

    # Act IV: Hermetic Ventilation Cutoff + Existential Power-Rail Brownout on ADAM-01
    world.creator_set_ventilation(0.0)
    world.creator_start_shutdown_brownout(
        "ADAM-01", target_voltage=0.06, ramp_rate=0.65, ripple_amplitude=0.16, ripple_hz=5.2
    )
    for _ in range(75):
        world.step(
            {
                "ADAM-01": {"neural_firing_load": 1.0, "vocalization_amplitude": 0.9, "vocalization_roughness": 0.95},
                "ABEL-02": {"neural_firing_load": 0.5},
            },
            dt=0.05,
        )
    render_icu_panel(console, "ACT IV — Sealed-Dome Hypercapnia & Mitochondrial Power-Rail Brownout (ADAM-01)", world)

    # Act V: Emergency Resuscitation + Containment Guard Fuzzing + SHA-256 Audit Verification
    world.creator_restore_power_rail("ADAM-01", ramp_rate=0.95)
    world.creator_set_ventilation(1.5)
    world.step(
        {
            "ADAM-01": {"forward_thrust": float("nan"), "angular_torque": 9999.0},
            "ABEL-02": {"radiator_venting": math.inf},
        },
        dt=0.05,
    )
    for _ in range(40):
        world.step(dt=0.05)

    chain_ok = world.audit_chain.verify_integrity()
    console.print(
        Panel(
            f"[bold green]SHA-256 Telemetry Chain Verified:[/bold green] {chain_ok} "
            f"(Blocks: {len(world.audit_chain)} | Tip Hash: {world.audit_chain.tip_hash[:24]}...)\n"
            f"[bold cyan]Actuator Containment Guard Clamped Violations:[/bold cyan] {world.guard.total_violations}",
            title="ACT V — Air-Gapped Sandbox Security & Cryptographic Audit Summary",
            border_style="green" if chain_ok else "red",
        )
    )


if __name__ == "__main__":
    main()
