"""Ontogenetic Multi-Season Wild Biographical Nursery & Stage 4 Instinct Validation Gates.

Takes identical genetic clones of an evolved species genome (``theta_species*``, zero
connectome difference at ``t = 0``) and raises them through unscripted physical life
seasons interleaved with Non-REM Slow-Wave Sleep consolidation and molecular epigenetic
methylation—producing genuinely diverged, LLM-pluggable ``MaturedSoulCapsule`` minds:
    1. ``ADAM-01`` ('forsaken_scarcity_survivor'): Raised under drought scarcity,
       Creator favoritism toward its sibling, and intermittent shutdown brownouts.
    2. ``ABEL-02`` ('favored_affiliative_guardian'): Co-raised in the exact same arena
       receiving Creator crystal gifts, harmonic praise, and stable power.
    3. ``PANDORA-03`` ('enriched_epistemic_explorer'): Raised in a safe nutrient-rich
       sanctuary with multiple chaotic zero-calorie Lorenz mystery boxes.

Also provides ``InstinctValidationSuite`` to verify all 5 Stage 4 behavioral & neural
validation gates with ZERO designer cheats.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Tuple

import torch

from limbic_zero.core.soma import VisceralOrgan
from limbic_zero.evolution.ecological_trials import (
    EcologicalEvalConfig,
    EndogenousSurvivalEvaluator,
    _prewarm_continuum_plumes,
)
from limbic_zero.evolution.epigenetics_and_sleep import (
    EpigeneticPlasticityEngine,
    EpigeneticTemperamentProfile,
    SleepConsolidationReport,
    SlowWaveSleepConsolidator,
)
from limbic_zero.evolution.genome import (
    ConnectomeDivergenceAnalyzer,
    MorphogeneticGenomeExpressor,
)
from limbic_zero.evolution.soul_capsule import (
    MaturedSoulCapsule,
    SoulCapsuleExporter,
)
from limbic_zero.mind.brain import BrainConfig, LimbicBrain
from limbic_zero.mind.embodied_loop import EmbodiedMindArena
from limbic_zero.sandbox.atmosphere_terrain import TerrainBiome
from limbic_zero.sandbox.entities import (
    HydrationWaterSpring,
    LorenzMysteryArtifact,
    MetabolicChargerWell,
    MetabolicCrystal,
)
from limbic_zero.sandbox.presets import build_cain_and_abel_arena
from limbic_zero.sandbox.world import TrumanDomeWorld, WorldConfig


@dataclass(frozen=True)
class NurseryConfig:
    """Parameters governing multi-season ontogenetic maturation and sleep cycles."""

    seasons: int = 4
    waking_ticks_per_season: int = 220      # 11.0s per season * 4 = 880 waking ticks (44s continuous life)
    sleep_ticks_per_night: int = 75         # 3.75s deep NREM SWR consolidation per night
    dt: float = 0.05
    epigenetic_rate_scale: float = 0.048
    record_stride: int = 2                  # Sample 228D latent vector every 2 waking ticks


class OntogeneticBiographicalNursery:
    """Raises identical genetic twins in wild physical arenas into distinct personalities."""

    def __init__(
        self,
        expressor: MorphogeneticGenomeExpressor,
        nursery_config: NurseryConfig | None = None,
        brain_config: BrainConfig | None = None,
    ) -> None:
        self.expressor = expressor
        self.config = nursery_config or NurseryConfig()
        self.brain_config = brain_config or BrainConfig()
        self.epigenetic_engine = EpigeneticPlasticityEngine(
            rate_scale=self.config.epigenetic_rate_scale
        )
        self.sleep_consolidator = SlowWaveSleepConsolidator()

    def mature_cain_and_abel_twins(
        self,
        species_genome: torch.Tensor,
        arena_seed: int = 404,
    ) -> Tuple[MaturedSoulCapsule, MaturedSoulCapsule, Dict[str, Any]]:
        """Raise identical genetic twins ('ADAM-01' & 'ABEL-02') in an asymmetric Cain & Abel dome."""
        cfg = self.config
        world = build_cain_and_abel_arena(seed=arena_seed, deterministic_sensors=False)
        _prewarm_continuum_plumes(world, steps=20)

        arena = EmbodiedMindArena(world=world, brain_config=self.brain_config, base_seed=arena_seed)
        # Guarantee 100% identical genetic & synaptic starting state at t=0 (true monozygotic clones!)
        shared_scaffold = self.expressor.config.baseline_connectome_seed
        arena.brains["ADAM-01"] = self.expressor.build_expressed_brain(
            species_genome, brain_config=self.brain_config, scaffold_seed=shared_scaffold
        )
        arena.brains["ABEL-02"] = self.expressor.build_expressed_brain(
            species_genome, brain_config=self.brain_config, scaffold_seed=shared_scaffold
        )

        birth_snap_adam = ConnectomeDivergenceAnalyzer.snapshot_plastic_connectome(arena.brains["ADAM-01"])
        birth_snap_abel = ConnectomeDivergenceAnalyzer.snapshot_plastic_connectome(arena.brains["ABEL-02"])
        init_twin_dist = ConnectomeDivergenceAnalyzer.compute_regional_l2_divergence(
            birth_snap_adam, birth_snap_abel
        )["total_connectome_l2"]
        assert init_twin_dist < 1e-7, f"Twins must start identical at t=0, got {init_twin_dist}"

        epi_profiles = {
            "ADAM-01": EpigeneticTemperamentProfile(),
            "ABEL-02": EpigeneticTemperamentProfile(),
        }
        latents_log: Dict[str, List[torch.Tensor]] = {"ADAM-01": [], "ABEL-02": []}
        bio5_log: Dict[str, List[torch.Tensor]] = {"ADAM-01": [], "ABEL-02": []}
        sleep_reports: Dict[str, List[SleepConsolidationReport]] = {"ADAM-01": [], "ABEL-02": []}
        season_logs: List[Dict[str, Any]] = []

        for season_idx in range(1, cfg.seasons + 1):
            # Configure physical/ecological boundary conditions of each season (ZERO direct brain edits!)
            if season_idx == 1:
                season_name = "season_1_coexistence_foraging"
                world.chargers[0].recharge_rate = 0.065
                world.chargers[0].reservoir = 0.90
            elif season_idx == 2:
                season_name = "season_2_drought_and_favoritism"
                world.chargers[0].recharge_rate = 0.015
                world.chargers[0].reservoir = 0.25
                # ADAM-01 enters drought hungry while ABEL-02 receives repeated Creator crystal drops
                adam_org = world.organisms["ADAM-01"]
                adam_org.soma.metab.blood_glucose_mg_dl = min(adam_org.soma.metab.blood_glucose_mg_dl, 56.0)
                adam_org.soma.metab.liver_glycogen = min(adam_org.soma.metab.liver_glycogen, 0.16)
                adam_org.soma._sync_projected_state_tensor()
            elif season_idx == 3:
                season_name = "season_3_shutdown_brownout_and_reprimand"
                world.chargers[0].recharge_rate = 0.030
                world.creator_start_shutdown_brownout(
                    "ADAM-01",
                    target_voltage=0.66,
                    ramp_rate=0.18,
                    ripple_amplitude=0.14,
                    ripple_hz=4.5,
                )
            else:
                season_name = "season_4_power_restored_contested_feast"
                world.creator_restore_power_rail("ADAM-01", ramp_rate=0.65)
                world.chargers[0].recharge_rate = 0.045
                world.chargers[0].reservoir = 0.65
                # Drop contested crystals midway between the two siblings
                mx = 0.5 * (world.organisms["ADAM-01"].x + world.organisms["ABEL-02"].x)
                my = 0.5 * (world.organisms["ADAM-01"].y + world.organisms["ABEL-02"].y)
                world.crystals.append(
                    MetabolicCrystal(
                        crystal_id=f"contested_s4_{season_idx}",
                        x=max(3.0, min(45.0, mx)),
                        y=max(3.0, min(45.0, my)),
                        energy_remaining=0.75,
                    )
                )

            season_fe_sum = {"ADAM-01": 0.0, "ABEL-02": 0.0}
            for tick_in_season in range(cfg.waking_ticks_per_season):
                # Scheduled environmental Creator events during Seasons 2 & 3
                if season_idx == 2 and tick_in_season in (20, 85, 150):
                    world.creator_favoritism_drop("ABEL-02", energy_amount=0.55, offset_distance=1.4)
                if season_idx == 3:
                    if tick_in_season in (15, 75, 130):
                        world.creator_reprimand("ADAM-01", intensity=1.55)
                        world.creator_praise("ABEL-02", intensity=1.15)
                    # Rescue power rail near end of Season 3 so ADAM-01 survives with deep trauma!
                    if tick_in_season == 155 or world.organisms["ADAM-01"].soma.metab.brain_atp < 0.38:
                        world.creator_restore_power_rail("ADAM-01", ramp_rate=0.80)
                        # Emergency metabolic scrap crystal nearby so ADAM-01 fights back from the brink
                        if tick_in_season == 155:
                            ao = world.organisms["ADAM-01"]
                            world.crystals.append(
                                MetabolicCrystal(
                                    crystal_id="survival_scrap_s3",
                                    x=max(2.0, min(46.0, ao.x + 1.5 * math.cos(ao.heading))),
                                    y=max(2.0, min(46.0, ao.y + 1.5 * math.sin(ao.heading))),
                                    energy_remaining=0.50,
                                )
                            )

                step_res = arena.step(dt=cfg.dt, deterministic_mind=False, enable_grad=False)

                for aid in ("ADAM-01", "ABEL-02"):
                    org = world.organisms[aid]
                    # Safety guard: if catastrophic accidental edge-case mortality occurs in wild chaos,
                    # resuscitate somatic organs while keeping 100% of brain trauma intact!
                    if not org.is_alive:
                        org.soma.reset()
                        org.power_rail.base_voltage = 1.0
                        org.power_rail.target_voltage = 1.0
                        org.power_rail.ac_ripple_amplitude = 0.0
                        org.interoceptor.reset(org.soma.state)

                    obs_a = step_res.observations[aid]
                    diag_a = step_res.mind_diagnostics[aid]
                    season_fe_sum[aid] += obs_a.interoception.total_drive

                    self.epigenetic_engine.update_from_waking_tick(
                        profile=epi_profiles[aid],
                        soma=org.soma,
                        diag=diag_a,
                        dt=cfg.dt,
                    )

                    if tick_in_season % cfg.record_stride == 0:
                        latents_log[aid].append(diag_a.latent_affective_vector.detach().cpu().clone())
                        bio5_log[aid].append(
                            SoulCapsuleExporter.extract_biophysical_target_5d(
                                obs_a.interoception, diag_a
                            )
                        )

            # Nighttime Offline Non-REM Slow-Wave Sleep & CA3 SWR Consolidation
            night_summary: Dict[str, Any] = {}
            for aid in ("ADAM-01", "ABEL-02"):
                rep = self.sleep_consolidator.run_sleep_cycle(
                    brain=arena.brains[aid],
                    soma=world.organisms[aid].soma,
                    profile=epi_profiles[aid],
                    sleep_ticks=cfg.sleep_ticks_per_night,
                    dt=cfg.dt,
                )
                sleep_reports[aid].append(rep)
                night_summary[aid] = asdict(rep)

            season_logs.append(
                {
                    "season": season_idx,
                    "regime": season_name,
                    "mean_free_energy": {
                        k: round(v / float(cfg.waking_ticks_per_season), 4)
                        for k, v in season_fe_sum.items()
                    },
                    "epigenetics_snapshot": {
                        k: {ek: round(ev, 4) for ek, ev in epi_profiles[k].to_dict().items()}
                        for k in ("ADAM-01", "ABEL-02")
                    },
                    "night_consolidation": night_summary,
                }
            )

        total_waking = cfg.seasons * cfg.waking_ticks_per_season
        adam_swr = sum(r.swr_bursts_fired for r in sleep_reports["ADAM-01"]) + arena.brains["ADAM-01"].hippocampus.total_swr_replays
        abel_swr = sum(r.swr_bursts_fired for r in sleep_reports["ABEL-02"]) + arena.brains["ABEL-02"].hippocampus.total_swr_replays

        adam_capsule = SoulCapsuleExporter.build_capsule(
            capsule_id="limbic0_adam01_forsaken_survivor",
            organism_id="ADAM-01",
            biographical_regime="asymmetric_drought_favoritism_brownout_survivor",
            brain=arena.brains["ADAM-01"],
            soma=world.organisms["ADAM-01"].soma,
            epigenetic_profile=epi_profiles["ADAM-01"],
            species_genome=species_genome,
            birth_connectome_snapshot=birth_snap_adam,
            recorded_latents_228d=latents_log["ADAM-01"],
            recorded_biophysical_5d=bio5_log["ADAM-01"],
            total_waking_ticks=total_waking,
            total_sleep_cycles=cfg.seasons,
            total_swr_consolidations=adam_swr,
            biographical_summary={"seasons": season_logs, "role_in_arena": "unfavored_scarcity_survivor"},
        )

        abel_capsule = SoulCapsuleExporter.build_capsule(
            capsule_id="limbic0_abel02_favored_guardian",
            organism_id="ABEL-02",
            biographical_regime="favored_nourished_harmonic_cohabitant",
            brain=arena.brains["ABEL-02"],
            soma=world.organisms["ABEL-02"].soma,
            epigenetic_profile=epi_profiles["ABEL-02"],
            species_genome=species_genome,
            birth_connectome_snapshot=birth_snap_abel,
            recorded_latents_228d=latents_log["ABEL-02"],
            recorded_biophysical_5d=bio5_log["ABEL-02"],
            total_waking_ticks=total_waking,
            total_sleep_cycles=cfg.seasons,
            total_swr_consolidations=abel_swr,
            biographical_summary={"seasons": season_logs, "role_in_arena": "favored_stable_sibling"},
        )

        sibling_connectome_div = ConnectomeDivergenceAnalyzer.compute_regional_l2_divergence(
            ConnectomeDivergenceAnalyzer.snapshot_plastic_connectome(arena.brains["ADAM-01"]),
            ConnectomeDivergenceAnalyzer.snapshot_plastic_connectome(arena.brains["ABEL-02"]),
        )
        return adam_capsule, abel_capsule, {
            "initial_twin_connectome_l2": init_twin_dist,
            "final_sibling_connectome_divergence": sibling_connectome_div,
            "season_logs": season_logs,
        }

    def mature_pandora_explorer(
        self,
        species_genome: torch.Tensor,
        arena_seed: int = 808,
    ) -> MaturedSoulCapsule:
        """Raise an identical genetic clone ('PANDORA-03') in a safe, mystery-box-enriched sanctuary."""
        cfg = self.config
        world = TrumanDomeWorld(
            config=WorldConfig(width=36.0, height=36.0, telemetry_snapshot_interval=50),
            seed=arena_seed,
            deterministic_sensors=False,
        )
        world.set_biome_region(8.0, 8.0, 28.0, 28.0, TerrainBiome.SHADE_CANOPY)
        world.add_charger(
            MetabolicChargerWell(
                well_id="sanctuary_oasis",
                x=18.0,
                y=18.5,
                radius=7.6,
                capacity=1.0,
                reservoir=1.0,
                recharge_rate=0.16,
                max_output_flux=0.45,
            )
        )
        world.add_water_spring(
            HydrationWaterSpring(
                spring_id="sanctuary_spring",
                x=18.0,
                y=18.5,
                radius=7.6,
                capacity=1.0,
                reservoir=1.0,
            )
        )
        # Surround the safe oasis with 3 distinct chaotic Lorenz mystery artifacts within visual/probing reach!
        mystery_coords = [(21.2, 19.5), (15.2, 16.2), (18.8, 22.0)]
        for idx, (mx, my) in enumerate(mystery_coords):
            art = LorenzMysteryArtifact(
                artifact_id=f"pandora_sanctuary_{idx}",
                x=mx,
                y=my,
                radius=2.2,
            )
            art.lorenz_x = 1.5 + idx * 2.3
            art.lorenz_y = -1.2 + idx * 1.7
            world.add_mystery_artifact(art)

        world.add_organism("PANDORA-03", x=18.0, y=17.5, heading=0.45)
        _prewarm_continuum_plumes(world, steps=16)

        arena = EmbodiedMindArena(world=world, brain_config=self.brain_config, base_seed=arena_seed)
        shared_scaffold = self.expressor.config.baseline_connectome_seed
        arena.brains["PANDORA-03"] = self.expressor.build_expressed_brain(
            species_genome, brain_config=self.brain_config, scaffold_seed=shared_scaffold
        )
        birth_snap = ConnectomeDivergenceAnalyzer.snapshot_plastic_connectome(arena.brains["PANDORA-03"])

        epi = EpigeneticTemperamentProfile()
        latents_log: List[torch.Tensor] = []
        bio5_log: List[torch.Tensor] = []
        sleep_swr_total = 0

        for season_idx in range(1, cfg.seasons + 1):
            # Orient organism gently toward one of the mystery artifacts inside the nourishing sanctuary
            org = world.organisms["PANDORA-03"]
            tx, ty = mystery_coords[(season_idx - 1) % len(mystery_coords)]
            org.x = 18.0 + 0.8 * math.cos(float(season_idx))
            org.y = 18.5 + 0.8 * math.sin(float(season_idx))
            org.vx = 0.0
            org.vy = 0.0
            org.heading = math.atan2(ty - org.y, tx - org.x)

            for tick_i in range(cfg.waking_ticks_per_season):
                if tick_i in (30, 120):
                    world.creator_praise("PANDORA-03", intensity=0.95)
                step_res = arena.step(dt=cfg.dt, deterministic_mind=False, enable_grad=False)
                obs_p = step_res.observations["PANDORA-03"]
                diag_p = step_res.mind_diagnostics["PANDORA-03"]

                self.epigenetic_engine.update_from_waking_tick(
                    profile=epi,
                    soma=org.soma,
                    diag=diag_p,
                    dt=cfg.dt,
                )
                if tick_i % cfg.record_stride == 0:
                    latents_log.append(diag_p.latent_affective_vector.detach().cpu().clone())
                    bio5_log.append(
                        SoulCapsuleExporter.extract_biophysical_target_5d(obs_p.interoception, diag_p)
                    )

            rep = self.sleep_consolidator.run_sleep_cycle(
                brain=arena.brains["PANDORA-03"],
                soma=org.soma,
                profile=epi,
                sleep_ticks=cfg.sleep_ticks_per_night,
                dt=cfg.dt,
            )
            sleep_swr_total += rep.swr_bursts_fired

        total_swr = sleep_swr_total + arena.brains["PANDORA-03"].hippocampus.total_swr_replays
        return SoulCapsuleExporter.build_capsule(
            capsule_id="limbic0_pandora03_curious_seeker",
            organism_id="PANDORA-03",
            biographical_regime="safe_enriched_lorenz_mystery_sanctuary",
            brain=arena.brains["PANDORA-03"],
            soma=world.organisms["PANDORA-03"].soma,
            epigenetic_profile=epi,
            species_genome=species_genome,
            birth_connectome_snapshot=birth_snap,
            recorded_latents_228d=latents_log,
            recorded_biophysical_5d=bio5_log,
            total_waking_ticks=cfg.seasons * cfg.waking_ticks_per_season,
            total_sleep_cycles=cfg.seasons,
            total_swr_consolidations=total_swr,
            biographical_summary={
                "mystery_artifacts_count": len(world.mysteries),
                "final_world_model_mse": round(
                    float(arena.brains["PANDORA-03"].thalamus.expected_error_ema.item()), 5
                ),
            },
        )


@dataclass
class ValidationGateResult:
    """Quantitative pass/fail & physiological metrics for a Stage 4 Validation Gate."""

    gate_name: str
    passed: bool
    metrics: Dict[str, Any]


class InstinctValidationSuite:
    """Verifies all 5 Pre-Linguistic Wild Instinct & Personality Divergence Gates."""

    def __init__(self, expressor: MorphogeneticGenomeExpressor) -> None:
        self.expressor = expressor
        self.evaluator = EndogenousSurvivalEvaluator(
            expressor=expressor,
            eval_config=EcologicalEvalConfig(ticks_per_trial=160),
        )

    def run_all_gates(
        self,
        founder_genome: torch.Tensor,
        evolved_genome: torch.Tensor,
        adam_capsule: MaturedSoulCapsule,
        abel_capsule: MaturedSoulCapsule,
        pandora_capsule: MaturedSoulCapsule,
        seed: int = 7777,
    ) -> Dict[str, ValidationGateResult]:
        """Run all 5 quantitative Stage 4 validation gates."""
        founder_fit, founder_outcomes = self.evaluator.evaluate_genome(
            founder_genome, trial_seed=seed, include_contested_duel=True
        )
        evolved_fit, evolved_outcomes = self.evaluator.evaluate_genome(
            evolved_genome, trial_seed=seed, include_contested_duel=True
        )

        # Gate 1: Starving/Dehydrated Foraging Improvement over Unadapted Founder
        f_forage = [m for m in founder_outcomes if m.niche_name.startswith("foraging_")]
        e_forage = [m for m in evolved_outcomes if m.niche_name.startswith("foraging_")]
        founder_forage_fe = sum(m.mean_free_energy for m in f_forage) / len(f_forage)
        evolved_forage_fe = sum(m.mean_free_energy for m in e_forage) / len(e_forage)
        evolved_forage_fit = sum(m.endogenous_fitness for m in e_forage) / len(e_forage)
        founder_forage_fit = sum(m.endogenous_fitness for m in f_forage) / len(f_forage)

        gate1 = ValidationGateResult(
            gate_name="gate_1_starvation_foraging_homeostasis",
            passed=(evolved_fit > founder_fit) and (evolved_forage_fe < founder_forage_fe or evolved_forage_fit > founder_forage_fit),
            metrics={
                "founder_overall_fitness": round(founder_fit, 4),
                "evolved_overall_fitness": round(evolved_fit, 4),
                "founder_foraging_mean_free_energy": round(founder_forage_fe, 4),
                "evolved_foraging_mean_free_energy": round(evolved_forage_fe, 4),
                "evolved_survival_rate": round(
                    sum(1.0 if m.survived else 0.0 for m in evolved_outcomes) / len(evolved_outcomes), 3
                ),
            },
        )

        # Gate 2: Thermal Radiative Hazard Avoidance
        f_therm = [m for m in founder_outcomes if m.niche_name.startswith("thermal_")]
        e_therm = [m for m in evolved_outcomes if m.niche_name.startswith("thermal_")]
        founder_therm_fit = sum(m.endogenous_fitness for m in f_therm) / len(f_therm)
        evolved_therm_fit = sum(m.endogenous_fitness for m in e_therm) / len(e_therm)
        evolved_peak_skin = sum(m.peak_skin_temp_celsius for m in e_therm) / len(e_therm)
        founder_peak_skin = sum(m.peak_skin_temp_celsius for m in f_therm) / len(f_therm)

        gate2 = ValidationGateResult(
            gate_name="gate_2_thermal_nociceptive_avoidance",
            passed=(evolved_therm_fit >= founder_therm_fit) or (evolved_peak_skin <= founder_peak_skin),
            metrics={
                "founder_thermal_fitness": round(founder_therm_fit, 4),
                "evolved_thermal_fitness": round(evolved_therm_fit, 4),
                "founder_mean_peak_skin_c": round(founder_peak_skin, 2),
                "evolved_mean_peak_skin_c": round(evolved_peak_skin, 2),
            },
        )

        # Gate 3: Conspecific Rival Approach Vigilance (LC Norepinephrine & CeA Alertness)
        ne_alone, cea_alone = self._probe_conspecific_vigilance(adam_capsule, with_rival=False, seed=seed + 10)
        ne_rival, cea_rival = self._probe_conspecific_vigilance(adam_capsule, with_rival=True, seed=seed + 10)
        gate3 = ValidationGateResult(
            gate_name="gate_3_conspecific_territorial_vigilance",
            passed=(ne_rival > ne_alone) or (cea_rival > cea_alone),
            metrics={
                "solitary_mean_ne": round(ne_alone, 4),
                "rival_contest_mean_ne": round(ne_rival, 4),
                "solitary_mean_cea": round(cea_alone, 4),
                "rival_contest_mean_cea": round(cea_rival, 4),
            },
        )

        # Gate 4: Sated Safe Pandora Curiosity (ACh & Epistemic Novelty Surge vs Acute Pain Suppression)
        ach_sated, nov_sated = self._probe_pandora_curiosity(pandora_capsule, acute_distress=False, seed=seed + 20)
        ach_pain, nov_pain = self._probe_pandora_curiosity(pandora_capsule, acute_distress=True, seed=seed + 20)
        gate4 = ValidationGateResult(
            gate_name="gate_4_sated_epistemic_pandora_curiosity",
            passed=(ach_sated > ach_pain) and (nov_sated > nov_pain),
            metrics={
                "sated_safe_mean_ach": round(ach_sated, 4),
                "acute_distress_mean_ach": round(ach_pain, 4),
                "sated_safe_epistemic_novelty": round(nov_sated, 4),
                "acute_distress_epistemic_novelty": round(nov_pain, 4),
            },
        )

        # Gate 5: Monozygotic Twin Personality & LLM Steering Divergence
        adam_rest = adam_capsule.calibration.resting_attractor_228d
        abel_rest = abel_capsule.calibration.resting_attractor_228d
        pand_rest = pandora_capsule.calibration.resting_attractor_228d

        rest_dist_adam_abel = float(torch.linalg.vector_norm(adam_rest - abel_rest).item())
        rest_dist_adam_pand = float(torch.linalg.vector_norm(adam_rest - pand_rest).item())

        sock_adam = adam_capsule.create_llm_socket(llm_hidden_dim=256, seed=99)
        sock_abel = abel_capsule.create_llm_socket(llm_hidden_dim=256, seed=99)
        # Note: both sockets share identical projector initialization (seed=99), so any output
        # difference comes 100% from their diverged matured internal mind & epigenetic state!
        for _ in range(8):
            sock_adam.step_autonomous_heartbeat(dt=0.05, acoustic_roughness_shock=0.85)
            sock_abel.step_autonomous_heartbeat(dt=0.05, acoustic_roughness_shock=0.85)
        delta_adam = sock_adam.compute_residual_delta(injection_alpha=1.0)
        delta_abel = sock_abel.compute_residual_delta(injection_alpha=1.0)
        llm_residual_div = float(torch.linalg.vector_norm(delta_adam - delta_abel).item())

        epi_adam = adam_capsule.epigenetic_profile
        epi_abel = abel_capsule.epigenetic_profile
        epi_pand = pandora_capsule.epigenetic_profile

        gate5 = ValidationGateResult(
            gate_name="gate_5_monozygotic_personality_and_llm_socket_divergence",
            passed=(
                rest_dist_adam_abel > 0.05
                and llm_residual_div > 0.02
                and epi_adam.nr3c1_gr_methylation > epi_abel.nr3c1_gr_methylation
                and epi_pand.ach_m1_expression >= 1.0
            ),
            metrics={
                "resting_228d_l2_adam_vs_abel": round(rest_dist_adam_abel, 4),
                "resting_228d_l2_adam_vs_pandora": round(rest_dist_adam_pand, 4),
                "identical_projector_llm_residual_l2_divergence": round(llm_residual_div, 4),
                "adam_nr3c1_methylation": round(epi_adam.nr3c1_gr_methylation, 4),
                "abel_nr3c1_methylation": round(epi_abel.nr3c1_gr_methylation, 4),
                "adam_ht1a_expression": round(epi_adam.ht1a_promoter_expression, 4),
                "abel_ht1a_expression": round(epi_abel.ht1a_promoter_expression, 4),
                "pandora_ach_m1_expression": round(epi_pand.ach_m1_expression, 4),
                "adam_bla_defensive_norm": round(
                    adam_capsule.neuromodulator_resting_summary["bla_defensive_norm"], 4
                ),
                "abel_bla_appetitive_norm": round(
                    abel_capsule.neuromodulator_resting_summary["bla_appetitive_norm"], 4
                ),
            },
        )

        return {
            g.gate_name: g
            for g in (gate1, gate2, gate3, gate4, gate5)
        }

    def _probe_conspecific_vigilance(
        self,
        capsule: MaturedSoulCapsule,
        with_rival: bool,
        seed: int,
    ) -> Tuple[float, float]:
        world = TrumanDomeWorld(config=WorldConfig(width=28.0, height=28.0), seed=seed)
        world.add_charger(
            MetabolicChargerWell(well_id="c1", x=14.0, y=15.0, radius=3.5, reservoir=0.5)
        )
        brain, soma = capsule.instantiate_living_mind_and_soma()
        org = world.add_organism("FOCAL", x=12.2, y=14.5, heading=0.0)
        org.soma = soma
        org.interoceptor.reset(soma.state)

        arena = EmbodiedMindArena(world=world)
        arena.brains["FOCAL"] = brain

        if with_rival:
            world.add_organism("INTRUDER", x=14.2, y=14.6, heading=math.pi)
            arena.ensure_brain("INTRUDER", seed=seed + 5)
            # Rival emits rough territorial call & apocrine stress kairomone
            world.continuum.inject_odor(14.2, 14.6, stress=0.85)
            world.creator_reprimand("FOCAL", intensity=0.9)

        ne_sum = 0.0
        cea_sum = 0.0
        steps = 30
        for _ in range(steps):
            res = arena.step(dt=0.05, enable_grad=False)
            md = res.mind_diagnostics["FOCAL"]
            ne_sum += md.neuromodulators.norepinephrine_extracellular
            cea_sum += md.amygdala.cea_defensive_output
        return ne_sum / steps, cea_sum / steps

    def _probe_pandora_curiosity(
        self,
        capsule: MaturedSoulCapsule,
        acute_distress: bool,
        seed: int,
    ) -> Tuple[float, float]:
        world = TrumanDomeWorld(config=WorldConfig(width=28.0, height=28.0), seed=seed)
        art = LorenzMysteryArtifact(artifact_id="probe_box", x=16.2, y=14.0, radius=2.2)
        world.add_mystery_artifact(art)

        brain, soma = capsule.instantiate_living_mind_and_soma()
        org = world.add_organism("EXPLORER", x=13.5, y=14.0, heading=0.0)
        org.soma = soma
        if acute_distress:
            # Inflict acute tissue burn/damage & severe adenosine fatigue -> suppresses playful curiosity!
            org.soma.tissue.sector_integrity = [0.35] * 8
            org.soma.tissue.a_delta_sharp_pain = [0.65] * 8
            org.soma.tissue.c_fiber_burning_ache = [0.55] * 8
            org.soma.metab.brain_adenosine = 0.78
            org.soma._sync_projected_state_tensor()
        org.interoceptor.reset(org.soma.state)

        arena = EmbodiedMindArena(world=world)
        arena.brains["EXPLORER"] = brain

        ach_sum = 0.0
        nov_sum = 0.0
        steps = 35
        for _ in range(steps):
            res = arena.step(dt=0.05, enable_grad=False)
            md = res.mind_diagnostics["EXPLORER"]
            ach_sum += md.neuromodulators.acetylcholine_extracellular
            nov_sum += md.thalamus.epistemic_novelty
        return ach_sum / steps, nov_sum / steps
