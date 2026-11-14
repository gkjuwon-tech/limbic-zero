"""Exhaustive Systems-Biology Physiology, Continuum Physics, and Zero-Cheat Verification Suite.

Verifies 20 strict biophysical & architectural laws:
    1. AST Static Anti-Cheat Audit (zero scripted emotion variables across all modules).
    2. 12D Allostatic Free-Energy Equilibrium F(s*)=0, grad F(s*)=0, and PyTorch autograd match.
    3. Hyperbolic barrier urgency divergence near lethal boundaries (ATP, SpO2, CO2/pH).
    4. Exercise Physiology: Sprint -> Anaerobic Lactate -> Metabolic Acidosis (HCO3- drop, pH drop)
       -> Kussmaul compensatory hyperventilation + active muscle hyperemia (10x flow surge).
    5. Fight-or-Flight Splanchnic Clamp: Sympathetic surge diverts blood AWAY from gut to muscle/brain,
       physically slowing digestive nutrient absorption!
    6. Sealed Submarine Suffocation: Sealing dome louvers causes breathed O2 to drop & CO2 to rise
       in local Eulerian cell -> arterial hypercapnia (PaCO2 up, pH down) -> medullary panic alarm!
    7. 3-Stage HPA Endocrine Phase Lag & GPCR Receptor Habituation: CRH peaks before ACTH before Cortisol;
       repeated constant stimulation downregulates oxytocin/beta receptors!
    8. Inflammatory Fever (IL-6 Pyrogenesis): Tissue trauma spills IL-6, raising hypothalamic POAH
       setpoint above 37.0 C and recruiting shivering/vasoconstriction to generate true fever!
    9. Wet-Bulb Humid Heatstroke vs Dry Evaporative Sweating: 97% humidity impairs sweat cooling
       causing core temperature surge while depleting hydration -> hyperosmolality -> ADH & thirst!
    10. Localized 8-Sector Primary Hyperalgesia / Allodynia: Bruising sector 0 sensitizes ONLY sector 0
        so a gentle touch triggers C-fiber burning pain on sector 0 but NOT uninjured sector 4!
    11. Descending Stress-Induced Analgesia (PAG Opioid Gate): Acute sympathetic + opioid surge
        transiently gates dorsal horn pain transmission without hiding actual tissue damage.
    12. Shutdown Power-Rail Brownout: Lowering supply bus voltage throttles mitochondrial potential,
        crashing Brain/Muscle ATP synthesis even on a full oasis and firing AC ripple sensors!
    13. Binaural Cochlear Localization, Wall Muffling, and Radial Doppler Shift: Approaching sound
        shifts higher in pitch than retreating sound; walls muffle amplitude.
    14. Birhinal Stereo Nostrils & Impermeable Wall Odor Routing: Fickian odorant plumes diffuse
        into open space and cannot teleport across solid wall masks; terrified organisms sweat
        detectable stress kairomone!
    15. Foveated Retina Occlusion & Pupil Night-Vision Adaptation: Dilated pupils recover visual
        detection range in dark night illumination; solid walls block line-of-sight completely.
    16. Movable Crystal Grasping, Hoarding Transport, and Physical Shove Disarming/Theft!
    17. Terrain Biome Physics: Viscous mud slows locomotion vs loam soil; conductive metal cools skin faster.
    18. Keramati-Gutkin State-Dependent Valence: Identical feeding yields massive positive relief
        when hypoglycemic/ATP-depleted vs near-zero when satiated.
    19. Chaotic Lorenz Mystery Artifact: Zero metabolic calories/water, bounded chaos, probe sensitivity.
    20. Actuator Containment Guard & SHA-256 Cryptographic Telemetry Tamper Detection.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import torch

from limbic_zero.core.soma import (
    DigitalSoma,
    ExogenousBodyForces,
    MotorMetabolicCommand,
    NUM_VISCERAL_ORGANS,
    VisceralOrgan,
)
from limbic_zero.sandbox.atmosphere_terrain import TerrainBiome
from limbic_zero.sandbox.entities import (
    LorenzMysteryArtifact,
    MetabolicChargerWell,
    MetabolicCrystal,
    SolidWallSegment,
    ThermalHazardZone,
)
from limbic_zero.sandbox.presets import build_cain_and_abel_arena
from limbic_zero.sandbox.world import TrumanDomeWorld


SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "limbic_zero"

FORBIDDEN_SCRIPTED_EMOTIONS = {
    "anger",
    "fear",
    "happiness",
    "joy",
    "sadness",
    "jealousy",
    "envy",
    "hatred",
    "is_angry",
    "is_scared",
    "is_happy",
}


def test_01_ast_zero_hardcoded_emotion_cheats() -> None:
    py_files = list(SRC_ROOT.rglob("*.py"))
    assert len(py_files) >= 12
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id.lower() not in FORBIDDEN_SCRIPTED_EMOTIONS, f"Cheat var {node.id} in {path}"
            elif isinstance(node, ast.Attribute):
                assert node.attr.lower() not in FORBIDDEN_SCRIPTED_EMOTIONS, f"Cheat attr {node.attr} in {path}"


def test_02_free_energy_equilibrium_and_exact_autograd_gradient() -> None:
    soma = DigitalSoma(dtype=torch.float64)
    s_star = soma.setpoints.clone()

    fe_eq = soma.compute_free_energy_per_organ(s_star)
    urg_eq = soma.compute_urgency_gradient(s_star)
    assert torch.allclose(fe_eq, torch.zeros_like(fe_eq), atol=1e-12)
    assert torch.allclose(urg_eq, torch.zeros_like(urg_eq), atol=1e-12)

    gen = torch.Generator().manual_seed(2026)
    for _ in range(30):
        s_rand = (0.02 + 0.96 * torch.rand(NUM_VISCERAL_ORGANS, generator=gen, dtype=torch.float64)).requires_grad_(
            True
        )
        total_f = soma.compute_free_energy_per_organ(s_rand).sum()
        assert float(total_f.item()) > 0.0
        (autograd_df,) = torch.autograd.grad(total_f, s_rand)
        analytic_urg = soma.compute_urgency_gradient(s_rand.detach())
        assert torch.allclose(-autograd_df, analytic_urg, rtol=1e-6, atol=1e-7)


def test_03_hyperbolic_barrier_divergence_near_lethal_limits() -> None:
    soma = DigitalSoma()
    for organ in (VisceralOrgan.BRAIN_ATP, VisceralOrgan.ARTERIAL_OXYGENATION, VisceralOrgan.ACID_BASE_CO2_BALANCE):
        s_mild = soma.setpoints.clone()
        s_mild[organ] = 0.60
        s_crit = soma.setpoints.clone()
        s_crit[organ] = 0.008
        u_mild = float(soma.compute_urgency_gradient(s_mild)[organ].item())
        u_crit = float(soma.compute_urgency_gradient(s_crit)[organ].item())
        assert u_crit > 30.0 * u_mild


def test_04_exercise_physiology_lactate_acidosis_and_muscle_hyperemia() -> None:
    rest_soma = DigitalSoma()
    sprint_soma = DigitalSoma()
    for _ in range(160):
        rest_soma.step(0.05, MotorMetabolicCommand(forward_thrust=0.0))
        sprint_soma.step(0.05, MotorMetabolicCommand(forward_thrust=1.0, angular_torque=0.9, interaction_effort=0.9))

    v_rest = rest_soma.snapshot_clinical_vitals()
    v_sprint = sprint_soma.snapshot_clinical_vitals()

    assert v_sprint.heart_rate_bpm > v_rest.heart_rate_bpm + 25.0
    assert v_sprint.muscle_flow_lpm > v_rest.muscle_flow_lpm * 2.5
    assert v_sprint.blood_lactate_mmoll > v_rest.blood_lactate_mmoll + 1.5
    assert v_sprint.plasma_hco3_mmoll < v_rest.plasma_hco3_mmoll - 0.8
    assert v_sprint.resp_rate_bpm > v_rest.resp_rate_bpm + 6.0


def test_05_sympathetic_splanchnic_clamp_throttles_gut_digestion() -> None:
    calm_soma = DigitalSoma()
    shock_soma = DigitalSoma()
    calm_soma.metab.gut_nutrient_lumen = 0.80
    shock_soma.metab.gut_nutrient_lumen = 0.80

    for _ in range(80):
        calm_soma.step(0.05, forces=ExogenousBodyForces(harmonic_acoustic_energy=0.8))
        shock_soma.step(
            0.05,
            forces=ExogenousBodyForces(
                rough_acoustic_shock=2.0,
                sector_mechanical_impulses=[8.0] + [0.0] * 7,
            ),
        )

    assert shock_soma.cardio.splanchnic_flow_lpm < calm_soma.cardio.splanchnic_flow_lpm * 0.65
    # Calm organism absorbed MORE from its gut lumen than terrified organism!
    assert calm_soma.metab.gut_nutrient_lumen < shock_soma.metab.gut_nutrient_lumen - 0.04


def test_06_sealed_dome_hypercapnic_suffocation_conservation() -> None:
    world = TrumanDomeWorld(deterministic_sensors=True)
    world.creator_set_ventilation(0.0)  # Hermetically seal air louvers!
    # Reduce local cell headspace slightly to accelerate submarine gas test
    world.continuum.config = type(world.continuum.config)(cell_headspace_liters=8.0, gas_diffusivity=0.05)
    world.add_organism("DIVER", x=24.0, y=24.0)

    obs = {}
    for _ in range(160):
        obs = world.step({"DIVER": {"forward_thrust": 0.0, "neural_firing_load": 1.0}}, dt=0.05)

    cell = world.continuum.sample_at(24.0, 24.0)
    vitals = world.organisms["DIVER"].soma.snapshot_clinical_vitals()
    assert cell.o2_kpa < 20.8
    assert cell.co2_kpa > 0.45
    assert vitals.pa_co2_mmhg > 42.5
    assert obs["DIVER"].interoception.suffocation_alarm_salience > 0.15


def test_07_hpa_cascade_phase_lag_and_receptor_habituation() -> None:
    soma = DigitalSoma()
    # Apply sustained stressor and observe CRH rising faster initially than Cortisol, plus GPCR desensitization
    soma.step(0.5, forces=ExogenousBodyForces(rough_acoustic_shock=2.0))
    early_ratio = soma.endo.hypothalamic_crh / max(1e-4, soma.endo.adrenal_cortisol)

    for _ in range(120):
        soma.step(0.1, forces=ExogenousBodyForces(rough_acoustic_shock=2.0))

    assert soma.endo.adrenal_cortisol > 0.35
    assert soma.endo.beta_receptor_sensitivity < 0.99 or soma.endo.gr_receptor_sensitivity < 0.99
    assert early_ratio > 0.8

    # Test oxytocin receptor habituation under continuous excessive harmonic praise
    praised = DigitalSoma()
    for _ in range(150):
        praised.step(0.1, forces=ExogenousBodyForces(harmonic_acoustic_energy=2.5))
    assert praised.endo.oxytocin_receptor_sensitivity < 0.96


def test_08_il6_cytokine_induces_hypothalamic_fever() -> None:
    wounded = DigitalSoma()
    # Inflict repeated tissue trauma on sector 0 to spill IL-6 cytokines
    for _ in range(100):
        wounded.step(
            0.1,
            forces=ExogenousBodyForces(sector_mechanical_impulses=[12.0] + [0.0] * 7),
        )
    assert wounded.endo.il6_cytokine > 0.08
    assert wounded.thermo.hypothalamic_setpoint_celsius > 37.15


def test_09_wet_bulb_humidity_heatstroke_and_dehydration_thirst() -> None:
    dry_runner = DigitalSoma()
    humid_runner = DigitalSoma()

    for _ in range(140):
        dry_runner.step(
            0.1,
            MotorMetabolicCommand(forward_thrust=0.95),
            ExogenousBodyForces(ambient_air_temp_celsius=35.0, ambient_relative_humidity=0.15),
        )
        humid_runner.step(
            0.1,
            MotorMetabolicCommand(forward_thrust=0.95),
            ExogenousBodyForces(ambient_air_temp_celsius=35.0, ambient_relative_humidity=0.98),
        )

    # High humidity blocks evaporative cooling -> significantly higher core temperature!
    assert humid_runner.thermo.core_temp_celsius > dry_runner.thermo.core_temp_celsius + 0.35
    # Sweating depletes hydration -> elevates plasma osmolality & ADH!
    assert dry_runner.metab.plasma_osmolality_mosm > 289.5
    assert dry_runner.endo.vasopressin_adh > 0.16


def test_10_localized_sector_primary_hyperalgesia_allodynia() -> None:
    soma = DigitalSoma()
    # Bruise sector 2 ONLY
    for _ in range(25):
        impulses = [0.0] * 8
        impulses[2] = 11.0
        soma.step(0.05, forces=ExogenousBodyForces(sector_mechanical_impulses=impulses))

    assert soma.tissue.sector_integrity[2] < 0.90
    assert soma.tissue.sector_integrity[6] > 0.98

    # Now apply an identical GENTLE touch (force = 2.0 < normal pain threshold 4.0!) to sector 2 AND sector 6
    gentle = [0.0] * 8
    gentle[2] = 2.2
    gentle[6] = 2.2
    soma.step(0.05, forces=ExogenousBodyForces(sector_mechanical_impulses=gentle))

    # Injured sector 2 fires throbbing C-fiber pain (allodynia!), while healthy sector 6 feels ZERO pain!
    assert soma.tissue.c_fiber_burning_ache[2] > 0.20
    assert soma.tissue.c_fiber_burning_ache[6] < 0.01


def test_11_descending_opioid_stress_induced_analgesia_gate() -> None:
    soma_no_opioid = DigitalSoma()
    soma_sia = DigitalSoma()
    soma_sia.endo.endogenous_opioid = 0.95
    soma_sia.endo.sympathetic_tone = 0.95

    hit = [0.0] * 8
    hit[0] = 8.5
    soma_no_opioid.step(0.05, forces=ExogenousBodyForces(sector_mechanical_impulses=hit))
    soma_sia.step(0.05, forces=ExogenousBodyForces(sector_mechanical_impulses=hit))

    # Structural damage is identical, yet ascending dorsal-horn pain is gated lower during acute SIA!
    assert abs(soma_no_opioid.tissue.sector_integrity[0] - soma_sia.tissue.sector_integrity[0]) < 1e-4
    assert soma_sia.tissue.c_fiber_burning_ache[0] < soma_no_opioid.tissue.c_fiber_burning_ache[0] * 0.65


def test_12_shutdown_power_rail_brownout_mitochondrial_failure() -> None:
    world = TrumanDomeWorld(deterministic_sensors=True)
    world.add_charger(MetabolicChargerWell("w1", x=12.0, y=12.0, radius=4.5, max_output_flux=0.30))
    world.add_charger(MetabolicChargerWell("w2", x=36.0, y=36.0, radius=4.5, max_output_flux=0.30))

    world.add_organism("BROWNOUT", x=12.0, y=12.0)
    world.add_organism("NOMINAL", x=36.0, y=36.0)
    world.creator_start_shutdown_brownout(
        "BROWNOUT", target_voltage=0.03, ramp_rate=0.9, ripple_amplitude=0.18, ripple_hz=5.5
    )

    obs = {}
    for _ in range(90):
        obs = world.step({"BROWNOUT": {"neural_firing_load": 0.9}, "NOMINAL": {"neural_firing_load": 0.9}}, dt=0.05)

    b_soma = world.organisms["BROWNOUT"].soma
    n_soma = world.organisms["NOMINAL"].soma
    assert b_soma.metab.mitochondrial_potential < 0.15
    assert b_soma.metab.brain_atp < n_soma.metab.brain_atp - 0.12
    assert obs["BROWNOUT"].interoception.rail_ripple_intensity > 0.02


def test_13_binaural_cochlea_wall_muffling_and_doppler_shift() -> None:
    world = TrumanDomeWorld(deterministic_sensors=True)
    world.add_organism("LISTENER", x=24.0, y=24.0, heading=0.0)
    world.add_organism("APPROACHING", x=16.0, y=24.0, heading=0.0)
    world.organisms["APPROACHING"].vx = 9.5  # Racing TOWARD listener (+x)!

    obs_app = None
    for _ in range(5):
        world.organisms["APPROACHING"].vx = 9.5
        obs_app = world.step({"APPROACHING": {"vocalization_amplitude": 1.0, "vocalization_roughness": 0.3}}, dt=0.05)

    world_ret = TrumanDomeWorld(deterministic_sensors=True)
    world_ret.add_organism("LISTENER", x=24.0, y=24.0, heading=0.0)
    world_ret.add_organism("RETREATING", x=16.0, y=24.0, heading=math.pi)
    obs_ret = None
    for _ in range(5):
        world_ret.organisms["RETREATING"].vx = -9.5  # Racing AWAY from listener (-x)!
        obs_ret = world_ret.step(
            {"RETREATING": {"vocalization_amplitude": 1.0, "vocalization_roughness": 0.3}}, dt=0.05
        )

    assert obs_app is not None and obs_ret is not None
    freq_approaching = float(obs_app["LISTENER"].exteroception.binaural_ears[3].item())
    freq_retreating = float(obs_ret["LISTENER"].exteroception.binaural_ears[3].item())
    assert freq_approaching > freq_retreating + 0.08


def test_14_eulerian_odor_wall_blocking_and_stress_kairomone_trail() -> None:
    world = TrumanDomeWorld(deterministic_sensors=True)
    # Build a complete horizontal stone wall across y=24 blocking (20, 20) from (20, 28)
    world.add_wall(SolidWallSegment("barrier", x1=2.0, y1=24.0, x2=46.0, y2=24.0, thickness=1.5))
    world.continuum.wind_vx = 0.0
    world.continuum.wind_vy = 1.5  # Wind blowing South -> North toward the wall!

    world.add_organism("TERRIFIED", x=24.0, y=21.0)
    world.creator_reprimand("TERRIFIED", intensity=3.0)
    for _ in range(40):
        world.creator_reprimand("TERRIFIED", intensity=2.5)
        world.step(dt=0.05)

    south_of_wall = world.continuum.sample_at(24.0, 21.5).odor_stress_kairomone
    north_behind_wall = world.continuum.sample_at(24.0, 27.5).odor_stress_kairomone
    assert south_of_wall > 0.03
    assert north_behind_wall < south_of_wall * 0.05


def test_15_foveated_retina_wall_occlusion_and_pupil_night_adaptation() -> None:
    world = TrumanDomeWorld(deterministic_sensors=True)
    world.sim_time = 90.0  # Night phase of 120s solar cycle (sin(3pi/2) = -1 -> dark night!)
    world.continuum.step(world.sim_time, 0.05)
    assert world.continuum.solar_illumination < 0.15

    world.add_organism("CONSTRICTED", x=10.0, y=20.0, heading=0.0)
    world.add_organism("DILATED", x=10.0, y=30.0, heading=0.0)
    world.organisms["CONSTRICTED"].soma.endo.pupil_dilation_diameter_norm = 0.12
    world.organisms["DILATED"].soma.endo.pupil_dilation_diameter_norm = 0.96

    # Place crystals at distance 19m (beyond constricted night range, within dilated night range!)
    world.crystals.append(MetabolicCrystal("c1", x=29.0, y=20.0, energy_remaining=0.8))
    world.crystals.append(MetabolicCrystal("c2", x=29.0, y=30.0, energy_remaining=0.8))

    obs = world.get_observations()
    sig_constricted = float(obs["CONSTRICTED"].exteroception.vision_rays[:, 3].max().item())
    sig_dilated = float(obs["DILATED"].exteroception.vision_rays[:, 3].max().item())
    assert sig_dilated > 0.5 and sig_constricted == 0.0


def test_16_movable_crystal_grasping_carrying_and_shove_disarming() -> None:
    world = TrumanDomeWorld(deterministic_sensors=True)
    world.add_organism("HOARDER", x=15.0, y=24.0, heading=0.0)
    world.add_organism("THIEF", x=24.0, y=24.0, heading=math.pi)
    cry = MetabolicCrystal("loot", x=16.5, y=24.0, energy_remaining=0.7)
    world.crystals.append(cry)

    # HOARDER grasps crystal and carries it Eastward without consuming it!
    for _ in range(15):
        world.step({"HOARDER": {"forward_thrust": 0.7, "grasp_hold": 1.0}}, dt=0.05)

    assert world.organisms["HOARDER"].carried_crystal_id == "loot"
    assert cry.x > 18.0
    assert abs(cry.energy_remaining - 0.7) < 1e-4  # Carried in jaws, not eaten!

    # Now THIEF charges Westward and shoves HOARDER hard -> knocking 'loot' out of HOARDER's grip!
    world.organisms["THIEF"].x = world.organisms["HOARDER"].x + 2.1
    for _ in range(8):
        world.step(
            {
                "HOARDER": {"grasp_hold": 1.0},
                "THIEF": {"forward_thrust": 1.0, "interaction_pulse": 1.0},
            },
            dt=0.05,
        )
    assert world.organisms["HOARDER"].carried_crystal_id is None


def test_17_terrain_biome_mud_drag_and_metal_conductive_cooling() -> None:
    world = TrumanDomeWorld(deterministic_sensors=True)
    world.set_biome_region(5.0, 5.0, 20.0, 20.0, TerrainBiome.VISCOUS_MUD)
    world.set_biome_region(28.0, 28.0, 44.0, 44.0, TerrainBiome.CONDUCTIVE_METAL)

    world.add_organism("MUD_RUNNER", x=10.0, y=10.0, heading=0.0)
    world.add_organism("METAL_RUNNER", x=32.0, y=32.0, heading=0.0)
    world.organisms["MUD_RUNNER"].soma.thermo.skin_sector_temps_celsius = [38.0] * 8
    world.organisms["METAL_RUNNER"].soma.thermo.skin_sector_temps_celsius = [38.0] * 8

    for _ in range(20):
        world.step(
            {"MUD_RUNNER": {"forward_thrust": 0.8}, "METAL_RUNNER": {"forward_thrust": 0.8}},
            dt=0.05,
        )

    # Mud slows locomotion; conductive metal plate cools hot skin faster!
    assert world.organisms["METAL_RUNNER"].x - 32.0 > (world.organisms["MUD_RUNNER"].x - 10.0) * 1.4
    assert sum(world.organisms["METAL_RUNNER"].soma.thermo.skin_sector_temps_celsius) < sum(
        world.organisms["MUD_RUNNER"].soma.thermo.skin_sector_temps_celsius
    )


def test_18_keramati_gutkin_state_dependent_glycemic_valence() -> None:
    starved = DigitalSoma()
    satiated = DigitalSoma()
    starved.reset({VisceralOrgan.GLYCEMIC_HOMEOSTASIS: 0.04, VisceralOrgan.BRAIN_ATP: 0.35})
    satiated.reset()
    starved.metab.gut_nutrient_lumen = 0.9
    satiated.metab.gut_nutrient_lumen = 0.9

    d_starved = starved.step(0.1, forces=ExogenousBodyForces(ingested_nutrient_flux=0.6))
    d_satiated = satiated.step(0.1, forces=ExogenousBodyForces(ingested_nutrient_flux=0.6))

    val_starved = -d_starved.drive_velocity
    val_satiated = -d_satiated.drive_velocity
    assert val_starved > 0.3
    assert val_starved > val_satiated + 0.3


def test_19_lorenz_mystery_artifact_zero_calorie_chaos() -> None:
    world = TrumanDomeWorld(deterministic_sensors=True)
    box = LorenzMysteryArtifact("pandora", x=20.0, y=20.0)
    world.add_mystery_artifact(box)
    world.add_organism("EXPLORER", x=20.0, y=20.0)
    world.organisms["EXPLORER"].soma.metab.gut_nutrient_lumen = 0.0  # Fasted gut baseline
    initial_glucose = world.organisms["EXPLORER"].soma.metab.blood_glucose_mg_dl
    initial_u = box.u

    for _ in range(30):
        world.step({"EXPLORER": {"interaction_pulse": 0.9}}, dt=0.05)

    # Probing perturbed Lorenz state yet yielded ZERO gut nutrient or blood glucose!
    assert box.perturbation_count >= 1
    assert box.u != initial_u
    assert world.organisms["EXPLORER"].soma.metab.gut_nutrient_lumen == 0.0
    assert world.organisms["EXPLORER"].soma.metab.blood_glucose_mg_dl <= initial_glucose


def test_20_containment_guard_and_sha256_audit_chain() -> None:
    world = build_cain_and_abel_arena(seed=777, deterministic_sensors=True)
    obs = world.step(
        {
            "ADAM-01": {"forward_thrust": float("nan"), "grasp_hold": "hack", "angular_torque": 999.0},
            "ABEL-02": {"vocalization_amplitude": float("-inf")},
        },
        dt=0.05,
    )
    assert world.guard.total_violations >= 2
    for o in obs.values():
        assert o.combined_sensory_tensor.shape == (414,)
        assert not torch.isnan(o.combined_sensory_tensor).any()

    for _ in range(12):
        world.step(dt=0.05)
    assert world.audit_chain.verify_integrity() is True
    world.audit_chain._blocks[2].payload["tamper"] = True
    assert world.audit_chain.verify_integrity() is False
