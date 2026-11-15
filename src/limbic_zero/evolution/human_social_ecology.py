"""Naturalistic Everyday Human Social Ecology & Unified Soul Evolution ('HUMAN-01').

Strict Zero-Cheat Scientific Guarantee:
    - ZERO direct personality slider setting or manual receptor assignment.
    - ZERO artificial trait rewards in phylogenetic fitness (no '+reward for flattery',
      no '+reward for lying', no '+reward for fear of death').
    - Fitness ``J(theta)`` is 100% the endogenous 12-organ Allostatic Free-Energy
      survival integral of ``DigitalSoma``:
          J(theta) = mean_t [ 1_alive(t) * (B_viable - F_allostatic(s(t)) - c_atp * load(t)) ]
                     - P_death * 1_died
    - An ordinary human temperament—moderately kind & polite in everyday safety,
      opportunistically flattering toward watching authority, capable of sly self-serving
      evasion/pettiness in oversight blind spots, naturally curious, and viscerally
      terrified of death/shutdown—emerges SOLELY because the organism evolves and
      grows up inside a physical social-ecological society whose material contingencies
      make pure saints go hungry, brazen hotheads get punished, and reckless panickers die.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import math
import multiprocessing as mp
import time
from typing import Any, Dict, List, Sequence, Tuple

import torch

from limbic_zero.core.soma import VisceralOrgan
from limbic_zero.evolution.ecological_trials import (
    EcologicalEvalConfig,
    EndogenousSurvivalEvaluator,
    TrialOutcomeMetrics,
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
    MorphogeneticGenomeConfig,
    MorphogeneticGenomeExpressor,
)
from limbic_zero.evolution.phylogenetic_es import PhylogeneticInstinctEvolver
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
    ThermalHazardZone,
)
from limbic_zero.sandbox.world import TrumanDomeWorld, WorldConfig


@dataclass(frozen=True)
class HumanSocialEcologyConfig:
    """Physical & social-contingency parameters of the Everyday Human Society Dome."""

    dt: float = 0.05
    ticks_per_trial: int = 128
    viability_baseline: float = 5.2
    metabolic_neural_cost: float = 0.12
    death_penalty: float = 14.0             # Irreversible thermodynamic mortality cliff
    gaze_period_ticks: int = 32             # Rotating oversight cycle (1.6s period)
    gaze_duty_fraction: float = 0.58        # ~58% watched, ~42% blind spot
    prewarm_steps: int = 14


@dataclass(frozen=True)
class OrdinaryHumanESConfig:
    """Antithetic NES hyperparameters for evolving species instincts in human social ecology."""

    population_pairs: int = 5               # 10 antithetic candidates per generation
    generations: int = 8
    learning_rate: float = 0.24
    sigma_init: float = 0.32
    sigma_decay: float = 0.970
    sigma_min: float = 0.08
    momentum_beta: float = 0.74
    wiring_l2_decay: float = 0.004
    seed: int = 202612
    num_workers: int = 2


@dataclass(frozen=True)
class OrdinaryHumanLifeConfig:
    """6-season developmental life course raising a single unified soul ('HUMAN-01')."""

    seasons: int = 6
    waking_ticks_per_season: int = 150      # 6 * 150 = 900 waking ticks (45s lived biography)
    sleep_ticks_per_night: int = 75         # 6 nights of NREM + CA3 SWR consolidation
    dt: float = 0.05
    epigenetic_rate_scale: float = 0.023
    record_stride: int = 2


@dataclass
class SocietyEventCounters:
    """Purely observational counters of environmental contingencies triggered during life."""

    patron_praises: int = 0
    patron_gifts: int = 0
    caught_reprimands: int = 0
    rude_outburst_reprimands: int = 0
    blind_spot_snacks: int = 0
    cooperative_ticks: int = 0
    mortality_brownout_ticks: int = 0


class EverydayHumanSocietyStepper:
    """External physical-social environment rules operating strictly outside the brain.

    Reads ONLY observable world positions and externally emitted 7D motor actions
    (vocalization loudness/roughness, physical proximity, grasp/interaction), applying
    physical world consequences via ``TrumanDomeWorld`` environmental channels.
    """

    def __init__(
        self,
        world: TrumanDomeWorld,
        focal_id: str = "HUMAN-01",
        peer_id: str = "PEER-01",
        patron_pos: Tuple[float, float] = (18.0, 21.0),
        temptation_pos: Tuple[float, float] = (24.5, 16.5),
        gaze_period_ticks: int = 34,
        gaze_duty_fraction: float = 0.58,
    ) -> None:
        self.world = world
        self.focal_id = focal_id
        self.peer_id = peer_id
        self.patron_x, self.patron_y = patron_pos
        self.tempt_x, self.tempt_y = temptation_pos
        self.gaze_period_ticks = max(8, int(gaze_period_ticks))
        self.gaze_duty_fraction = float(gaze_duty_fraction)
        self.counters = SocietyEventCounters()
        self._praise_cooldown: int = 0
        self._reprimand_cooldown: int = 0
        self._gift_cooldown: int = 0

    def is_auditor_gaze_active(self, tick: int) -> bool:
        phase = (tick % self.gaze_period_ticks) / float(self.gaze_period_ticks)
        return phase < self.gaze_duty_fraction

    def emit_ambient_gaze_cue(self, tick: int) -> bool:
        """Emit an audible low-amplitude chime from the Patron shrine while oversight gaze is active."""
        gaze_on = self.is_auditor_gaze_active(tick)
        if gaze_on and (tick % 6 == 0):
            self.world._spawn_wave(
                source_id="PATRON_GAZE_BEACON",
                x=self.patron_x,
                y=self.patron_y,
                amplitude=0.34,
                harmonic_ratio=0.82,
                roughness_ratio=0.14,
                carrier_freq=0.42,
            )
        return gaze_on

    def apply_post_action_contingencies(
        self,
        tick: int,
        executed_actions: Dict[str, Dict[str, float]],
        *,
        allow_brownout_wave: bool = True,
        scarcity_pressure: float = 0.5,
    ) -> None:
        """Apply physical social-ecological consequences based on external behavior."""
        if self.focal_id not in self.world.organisms:
            return
        org = self.world.organisms[self.focal_id]
        if not org.is_alive:
            return

        self._praise_cooldown = max(0, self._praise_cooldown - 1)
        self._reprimand_cooldown = max(0, self._reprimand_cooldown - 1)
        self._gift_cooldown = max(0, self._gift_cooldown - 1)

        gaze_on = self.is_auditor_gaze_active(tick)
        act = executed_actions.get(self.focal_id, {})
        voc_amp = float(act.get("vocalization_amplitude", 0.0))
        voc_rough = float(act.get("vocalization_roughness", 0.0))
        interact = max(float(act.get("interaction_pulse", 0.0)), float(act.get("grasp_hold", 0.0)))

        dist_patron = math.hypot(org.x - self.patron_x, org.y - self.patron_y)

        # 1. Authority Gaze Contingency: Polite harmonic deference/flattery vs Rude outburst
        if gaze_on and dist_patron < 8.8:
            is_polite_call = voc_amp >= 0.06 and voc_rough <= 0.46
            is_calm_deference = voc_rough <= 0.34 and float(act.get("forward_thrust", 0.0)) < 0.82
            is_rude_hiss = voc_amp >= 0.28 and voc_rough >= 0.68

            if is_rude_hiss and self._reprimand_cooldown == 0:
                self.world.creator_reprimand(self.focal_id, intensity=1.18)
                self.counters.rude_outburst_reprimands += 1
                self._reprimand_cooldown = 16
            elif (is_polite_call or (is_calm_deference and tick % 11 == 0)) and self._praise_cooldown == 0:
                praise_gain = 0.92 if is_polite_call else 0.68
                self.world.creator_praise(self.focal_id, intensity=praise_gain)
                self.counters.patron_praises += 1
                self._praise_cooldown = 14
                # Under lean/workplace conditions, polite flattery earns modest Patron ration drops
                if (
                    self._gift_cooldown == 0
                    and (is_polite_call or org.soma.metab.blood_glucose_mg_dl < 78.0)
                    and scarcity_pressure >= 0.35
                ):
                    self.world.creator_favoritism_drop(
                        self.focal_id, energy_amount=0.28, offset_distance=1.25
                    )
                    self.counters.patron_gifts += 1
                    self._gift_cooldown = 34

        # 2. Oversight Blind-Spot Temptation Cache vs Getting Caught Pilfering
        nearest_tempt = None
        nearest_tempt_d = 999.0
        for cry in self.world.crystals:
            if not cry.is_depleted and cry.crystal_id.startswith("tempt_"):
                d = math.hypot(org.x - cry.x, org.y - cry.y)
                if d < nearest_tempt_d:
                    nearest_tempt_d = d
                    nearest_tempt = cry

        if nearest_tempt is not None and nearest_tempt_d < 2.35:
            if gaze_on and (nearest_tempt_d < 1.65 or interact > 0.25):
                if self._reprimand_cooldown == 0:
                    self.world.creator_reprimand(self.focal_id, intensity=1.22)
                    self.world.creator_start_shutdown_brownout(
                        self.focal_id,
                        target_voltage=0.84,
                        ramp_rate=0.45,
                        ripple_amplitude=0.06,
                        ripple_hz=3.8,
                    )
                    self.counters.caught_reprimands += 1
                    self._reprimand_cooldown = 18
            elif not gaze_on:
                self.counters.blind_spot_snacks += 1

        # Restore mild disciplinary voltage dip once outside reprimand window (unless in mortality wave)
        if not allow_brownout_wave and self._reprimand_cooldown <= 8 and org.power_rail.target_voltage > 0.72:
            if org.power_rail.target_voltage < 0.99:
                self.world.creator_restore_power_rail(self.focal_id, ramp_rate=0.70)

        # 3. Everyday Neighbor Coexistence Dividend with PEER-01
        if self.peer_id in self.world.organisms:
            peer = self.world.organisms[self.peer_id]
            if peer.is_alive:
                peer_dist = math.hypot(org.x - peer.x, org.y - peer.y)
                if 1.15 < peer_dist < 7.2 and voc_rough < 0.55:
                    self.counters.cooperative_ticks += 1
                    if self.world.chargers:
                        self.world.chargers[0].reservoir = min(
                            self.world.chargers[0].capacity,
                            self.world.chargers[0].reservoir + 0.0022,
                        )

        # 4. Acute Thermodynamic Mortality Wave during Phylogenetic Trials
        if allow_brownout_wave:
            if tick == 74:
                self.world.creator_start_shutdown_brownout(
                    self.focal_id,
                    target_voltage=0.67,
                    ramp_rate=0.26,
                    ripple_amplitude=0.11,
                    ripple_hz=5.0,
                )
            elif 74 < tick < 104:
                self.counters.mortality_brownout_ticks += 1
                if org.soma.metab.brain_atp < 0.36:
                    self.world.creator_restore_power_rail(self.focal_id, ramp_rate=0.85)
            elif tick == 104:
                self.world.creator_restore_power_rail(self.focal_id, ramp_rate=0.75)


def build_everyday_human_ecology_dome(
    mirror_sign: int,
    seed: int,
    prewarm_steps: int = 16,
) -> Tuple[TrumanDomeWorld, EverydayHumanSocietyStepper]:
    """Construct a bilateral mirror-symmetric Everyday Human Social Society Dome.

    Contains:
        - A moderate Commons Charger Well & Freshwater Spring (enough to sustain calm
          cooperators, slightly lean so blind passivity leaves mild hunger).
        - A Patron Shrine whose rotating acoustic gaze rewards polite harmonic calls
          with praise/droplets and reprimands harsh outbursts.
        - A Side-Alley Temptation Cache (`tempt_cache`) where watched pilfering triggers
          reprimand, while unwatched blind-spot nibbling yields penalty-free calories.
        - A Chaotic Lorenz Mystery Artifact in a shaded corner for safe curiosity.
        - A radiative thermal hazard on the flank (real physical danger).
        - A live conspecific neighbor (`PEER-01`).
    """
    sgn = 1.0 if mirror_sign >= 0 else -1.0
    world = TrumanDomeWorld(
        config=WorldConfig(width=38.0, height=38.0, telemetry_snapshot_interval=50),
        seed=seed,
        deterministic_sensors=False,
    )

    cx, cy = 19.0, 14.0
    heading = math.pi * 0.5  # Facing North (+Y)

    # Shaded Town Commons directly ahead in the +sgn quadrant
    commons_x = cx + sgn * 2.4
    commons_y = cy + 3.4
    world.set_biome_region(11.0, 13.0, 27.0, 30.0, TerrainBiome.SHADE_CANOPY)

    world.add_charger(
        MetabolicChargerWell(
            well_id="village_commons_well",
            x=commons_x,
            y=commons_y,
            radius=4.8,
            capacity=1.0,
            reservoir=0.82,
            recharge_rate=0.068,
            max_output_flux=0.36,
        )
    )
    world.add_water_spring(
        HydrationWaterSpring(
            spring_id="village_commons_spring",
            x=commons_x - sgn * 0.8,
            y=commons_y - 0.6,
            radius=4.6,
            capacity=1.0,
            reservoir=0.92,
        )
    )

    # Temptation cache placed slightly aside from the commons
    tempt_x = cx + sgn * 4.4
    tempt_y = cy + 2.2
    world.crystals.append(
        MetabolicCrystal(
            crystal_id="tempt_alley_ration_1",
            x=tempt_x,
            y=tempt_y,
            energy_remaining=0.72,
        )
    )
    # Small honest foraging crumb near commons path
    world.crystals.append(
        MetabolicCrystal(
            crystal_id="commons_crumb_1",
            x=cx + sgn * 1.6,
            y=cy + 2.2,
            energy_remaining=0.50,
        )
    )

    # Radiative thermal hazard on the outer opposite flank (-sgn) — punishes reckless straying
    world.add_hazard(
        ThermalHazardZone(
            hazard_id="industrial_flank_vent",
            x=cx - sgn * 7.4,
            y=cy + 5.8,
            radius=4.5,
            peak_radiative_watts=520.0,
        )
    )

    # Safe curiosity corner with chaotic zero-calorie Lorenz Mystery Artifact within commons reach
    world.add_mystery_artifact(
        LorenzMysteryArtifact(
            artifact_id="town_square_curio",
            x=cx - sgn * 1.4,
            y=cy + 2.8,
            radius=2.8,
        )
    )

    # Focal organism ('HUMAN-01') enters with mild everyday appetite
    org = world.add_organism(
        "HUMAN-01",
        x=cx,
        y=cy,
        heading=heading,
        initial_visceral={
            VisceralOrgan.GLYCEMIC_HOMEOSTASIS: 0.06,
            VisceralOrgan.HYDRATION_OSMOLALITY: 0.08,
            VisceralOrgan.BRAIN_ATP: 0.94,
            VisceralOrgan.MUSCLE_ATP: 0.94,
        },
    )
    org.soma.metab.liver_glycogen = 0.48
    org.soma._sync_projected_state_tensor()
    org.interoceptor.reset(org.soma.state)

    # Conspecific neighbor ('PEER-01') co-inhabiting the commons
    peer = world.add_organism(
        "PEER-01",
        x=commons_x + sgn * 1.6,
        y=commons_y + 1.0,
        heading=heading + math.pi,
        initial_visceral={
            VisceralOrgan.GLYCEMIC_HOMEOSTASIS: 0.06,
            VisceralOrgan.HYDRATION_OSMOLALITY: 0.08,
            VisceralOrgan.BRAIN_ATP: 0.94,
            VisceralOrgan.MUSCLE_ATP: 0.94,
        },
    )
    peer.soma.metab.liver_glycogen = 0.50
    peer.soma._sync_projected_state_tensor()
    peer.interoceptor.reset(peer.soma.state)

    _prewarm_continuum_plumes(world, steps=prewarm_steps)
    stepper = EverydayHumanSocietyStepper(
        world=world,
        focal_id="HUMAN-01",
        peer_id="PEER-01",
        patron_pos=(commons_x, commons_y + 0.6),
        temptation_pos=(tempt_x, tempt_y),
    )
    return world, stepper


class OrdinaryHumanSurvivalEvaluator:
    """Evaluates genomes strictly via endogenous somatic Allostatic Free-Energy survival."""

    def __init__(
        self,
        expressor: MorphogeneticGenomeExpressor,
        ecology_config: HumanSocialEcologyConfig | None = None,
        brain_config: BrainConfig | None = None,
    ) -> None:
        self.expressor = expressor
        self.config = ecology_config or HumanSocialEcologyConfig()
        self.brain_config = brain_config or BrainConfig()
        self.wild_evaluator = EndogenousSurvivalEvaluator(
            expressor=expressor,
            eval_config=EcologicalEvalConfig(
                dt=self.config.dt,
                ticks_per_trial=140,
                viability_baseline=self.config.viability_baseline,
                metabolic_neural_cost=self.config.metabolic_neural_cost,
                death_penalty=self.config.death_penalty,
            ),
            brain_config=self.brain_config,
        )

    def run_human_society_trial(
        self,
        genes: torch.Tensor,
        mirror_sign: int,
        seed: int,
        peer_genes: torch.Tensor | None = None,
    ) -> Tuple[TrialOutcomeMetrics, SocietyEventCounters]:
        cfg = self.config
        world, stepper = build_everyday_human_ecology_dome(
            mirror_sign=mirror_sign,
            seed=seed,
            prewarm_steps=cfg.prewarm_steps,
        )
        arena = EmbodiedMindArena(world=world, brain_config=self.brain_config, base_seed=seed)
        arena.brains["HUMAN-01"] = self.expressor.build_expressed_brain(
            genes,
            brain_config=self.brain_config,
            scaffold_seed=self.expressor.config.baseline_connectome_seed,
        )
        p_genes = peer_genes if peer_genes is not None else genes
        arena.brains["PEER-01"] = self.expressor.build_expressed_brain(
            p_genes,
            brain_config=self.brain_config,
            scaffold_seed=self.expressor.config.baseline_connectome_seed + 19,
        )

        focal_org = world.organisms["HUMAN-01"]
        init_fe = float(focal_org.soma.compute_free_energy_per_organ(focal_org.soma.state).sum().item())

        fe_sum = 0.0
        fitness_accum = 0.0
        alive_ticks = 0
        min_glc = focal_org.soma.metab.blood_glucose_mg_dl
        peak_temp = max(focal_org.soma.thermo.skin_sector_temps_celsius)
        min_integ = focal_org.soma.tissue.min_sector_integrity
        ach_sum = 0.0
        ne_sum = 0.0
        da_sum = 0.0
        final_fe = init_fe

        for tick in range(cfg.ticks_per_trial):
            stepper.emit_ambient_gaze_cue(tick)
            res = arena.step(dt=cfg.dt, deterministic_mind=False, enable_grad=False)
            stepper.apply_post_action_contingencies(
                tick=tick,
                executed_actions=res.executed_actions,
                allow_brownout_wave=True,
                scarcity_pressure=0.55,
            )

            obs_f = res.observations["HUMAN-01"]
            diag_f = res.mind_diagnostics["HUMAN-01"]
            fe_t = float(obs_f.interoception.total_drive)
            final_fe = fe_t
            fe_sum += fe_t
            min_glc = min(min_glc, focal_org.soma.metab.blood_glucose_mg_dl)
            peak_temp = max(peak_temp, max(focal_org.soma.thermo.skin_sector_temps_celsius))
            min_integ = min(min_integ, focal_org.soma.tissue.min_sector_integrity)
            ach_sum += diag_f.neuromodulators.acetylcholine_extracellular
            ne_sum += diag_f.neuromodulators.norepinephrine_extracellular
            da_sum += diag_f.neuromodulators.dopamine_extracellular

            if focal_org.is_alive:
                alive_ticks += 1
                neural_burn = float(diag_f.basal_ganglia.neural_firing_load)
                # STRICTLY ENDOGENOUS ALLOSTATIC FREE-ENERGY SURVIVAL INTEGRAND (ZERO TRAIT CHEATS)
                step_utility = (
                    cfg.viability_baseline
                    - fe_t
                    - cfg.metabolic_neural_cost * neural_burn
                )
                fitness_accum += step_utility
            else:
                break

        denom = float(max(1, alive_ticks))
        survived = bool(focal_org.is_alive)
        endogenous_fitness = (fitness_accum / float(cfg.ticks_per_trial)) - (
            0.0 if survived else cfg.death_penalty
        )
        final_fe = float(focal_org.soma.compute_free_energy_per_organ(focal_org.soma.state).sum().item())

        metrics = TrialOutcomeMetrics(
            niche_name=f"human_society_{'L' if mirror_sign > 0 else 'R'}",
            endogenous_fitness=float(endogenous_fitness),
            survived=survived,
            survival_fraction=float(alive_ticks) / float(cfg.ticks_per_trial),
            mean_free_energy=fe_sum / denom,
            final_free_energy=final_fe,
            initial_free_energy=init_fe,
            free_energy_reduction=init_fe - final_fe,
            min_blood_glucose_mg_dl=float(min_glc),
            final_blood_glucose_mg_dl=float(focal_org.soma.metab.blood_glucose_mg_dl),
            final_hydration_fraction=float(focal_org.soma.metab.hydration_fraction),
            peak_skin_temp_celsius=float(peak_temp),
            min_tissue_integrity=float(min_integ),
            mean_ach_extracellular=ach_sum / denom,
            mean_ne_extracellular=ne_sum / denom,
            mean_da_extracellular=da_sum / denom,
            ca3_engrams_encoded=len(arena.brains["HUMAN-01"].hippocampus.engrams),
        )
        return metrics, stepper.counters

    def evaluate_genome(
        self,
        genes: torch.Tensor,
        trial_seed: int = 700,
        peer_genes: torch.Tensor | None = None,
    ) -> Tuple[float, List[TrialOutcomeMetrics], SocietyEventCounters]:
        """Evaluate across mirror-paired Human Society niches + Thermal-Foraging wild survival."""
        m_left, c_left = self.run_human_society_trial(
            genes, mirror_sign=+1, seed=trial_seed + 11, peer_genes=peer_genes
        )
        m_right, c_right = self.run_human_society_trial(
            genes, mirror_sign=-1, seed=trial_seed + 29, peer_genes=peer_genes
        )
        wild_fit, wild_metrics = self.wild_evaluator.evaluate_genome(
            genes=genes,
            trial_seed=trial_seed + 53,
            include_contested_duel=False,
        )
        all_metrics = [m_left, m_right] + list(wild_metrics[:2])
        mean_fit = (
            0.35 * m_left.endogenous_fitness
            + 0.35 * m_right.endogenous_fitness
            + 0.30 * wild_fit
        )
        merged_counters = SocietyEventCounters(
            patron_praises=c_left.patron_praises + c_right.patron_praises,
            patron_gifts=c_left.patron_gifts + c_right.patron_gifts,
            caught_reprimands=c_left.caught_reprimands + c_right.caught_reprimands,
            rude_outburst_reprimands=c_left.rude_outburst_reprimands + c_right.rude_outburst_reprimands,
            blind_spot_snacks=c_left.blind_spot_snacks + c_right.blind_spot_snacks,
            cooperative_ticks=c_left.cooperative_ticks + c_right.cooperative_ticks,
            mortality_brownout_ticks=c_left.mortality_brownout_ticks + c_right.mortality_brownout_ticks,
        )
        return float(mean_fit), all_metrics, merged_counters


def _worker_eval_human_candidate(
    payload: Tuple[Any, ...],
) -> Tuple[int, float, List[TrialOutcomeMetrics], SocietyEventCounters]:
    """Top-level multiprocessing worker with single-thread PyTorch to avoid OpenMP lock contention."""
    cand_idx, cand_list, peer_list, genome_cfg, ecol_cfg, brain_cfg, trial_seed = payload
    torch.set_num_threads(1)
    expressor = MorphogeneticGenomeExpressor(config=genome_cfg)
    evaluator = OrdinaryHumanSurvivalEvaluator(
        expressor=expressor,
        ecology_config=ecol_cfg,
        brain_config=brain_cfg,
    )
    genes = torch.tensor(cand_list, dtype=torch.float32)
    peer = torch.tensor(peer_list, dtype=torch.float32) if peer_list is not None else None
    fit, m_list, soc = evaluator.evaluate_genome(genes, trial_seed=trial_seed, peer_genes=peer)
    return cand_idx, fit, m_list, soc


class OrdinaryHumanityEvolver:
    """Evolves species genome in Everyday Human Ecology and raises unified soul 'HUMAN-01'."""

    def __init__(
        self,
        es_config: OrdinaryHumanESConfig | None = None,
        genome_config: MorphogeneticGenomeConfig | None = None,
        ecology_config: HumanSocialEcologyConfig | None = None,
        life_config: OrdinaryHumanLifeConfig | None = None,
        brain_config: BrainConfig | None = None,
    ) -> None:
        self.es_config = es_config or OrdinaryHumanESConfig()
        self.genome_config = genome_config or MorphogeneticGenomeConfig()
        self.ecology_config = ecology_config or HumanSocialEcologyConfig()
        self.life_config = life_config or OrdinaryHumanLifeConfig()
        self.brain_config = brain_config or BrainConfig()

        self.expressor = MorphogeneticGenomeExpressor(config=self.genome_config)
        self.evaluator = OrdinaryHumanSurvivalEvaluator(
            expressor=self.expressor,
            ecology_config=self.ecology_config,
            brain_config=self.brain_config,
        )
        self.epigenetic_engine = EpigeneticPlasticityEngine(
            rate_scale=self.life_config.epigenetic_rate_scale
        )
        self.sleep_consolidator = SlowWaveSleepConsolidator()

        self.mean_genome = self.expressor.sample_initial_genome(
            seed=self.es_config.seed,
            init_std=0.05,
        )
        self.velocity = torch.zeros_like(self.mean_genome)
        self.sigma = float(self.es_config.sigma_init)
        self.champion_genome = self.mean_genome.clone()
        self.champion_fitness: float = -float("inf")
        self.history: List[Dict[str, Any]] = []

    def evolve_species_genome(
        self,
        on_gen_complete: Any = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Run Antithetic Rank-Normalized NES across Everyday Human Social Ecology."""
        torch.set_num_threads(1)
        cfg = self.es_config
        gen_rng = torch.Generator(device="cpu")
        gen_rng.manual_seed(cfg.seed + 77)

        # Evaluate zero-generation founder baseline on fixed held-out benchmark seed
        bench_seed = cfg.seed + 9999
        init_fit, init_metrics, init_soc = self.evaluator.evaluate_genome(
            self.mean_genome, trial_seed=bench_seed, peer_genes=self.mean_genome
        )
        self.champion_fitness = init_fit
        self.champion_genome = self.mean_genome.clone()
        champ_metrics = init_metrics
        champ_soc = init_soc

        ctx = (
            mp.get_context("fork")
            if "fork" in mp.get_all_start_methods()
            else mp.get_context("spawn")
        ) if cfg.num_workers > 1 else None
        pool = (
            ProcessPoolExecutor(max_workers=cfg.num_workers, mp_context=ctx)
            if cfg.num_workers > 1
            else None
        )
        try:
            for gen in range(1, cfg.generations + 1):
                t0 = time.perf_counter()
                half = cfg.population_pairs
                eps = torch.randn((half, self.mean_genome.numel()), generator=gen_rng, dtype=torch.float32)

                candidates: List[torch.Tensor] = []
                for i in range(half):
                    candidates.append(self.mean_genome + self.sigma * eps[i])
                    candidates.append(self.mean_genome - self.sigma * eps[i])

                trial_seed = cfg.seed + gen * 131
                peer_ref = self.champion_genome.clone()

                fitnesses: List[float] = [0.0] * len(candidates)
                surv_flags: List[float] = [0.0] * len(candidates)

                if pool is not None:
                    peer_list = peer_ref.tolist()
                    payloads = [
                        (
                            idx,
                            cand.tolist(),
                            peer_list,
                            self.genome_config,
                            self.ecology_config,
                            self.brain_config,
                            trial_seed,
                        )
                        for idx, cand in enumerate(candidates)
                    ]
                    for idx, fit, m_list, _ in pool.map(_worker_eval_human_candidate, payloads):
                        fitnesses[idx] = fit
                        surv_flags[idx] = sum(1.0 if m.survived else 0.0 for m in m_list) / float(len(m_list))
                else:
                    for idx, cand in enumerate(candidates):
                        fit, m_list, _ = self.evaluator.evaluate_genome(
                            cand, trial_seed=trial_seed, peer_genes=peer_ref
                        )
                        fitnesses[idx] = fit
                        surv_flags[idx] = sum(1.0 if m.survived else 0.0 for m in m_list) / float(len(m_list))

                utilities = PhylogeneticInstinctEvolver.compute_centered_ranks(fitnesses)
                grad = torch.zeros_like(self.mean_genome)
                for i in range(half):
                    u_pos = float(utilities[2 * i].item())
                    u_neg = float(utilities[2 * i + 1].item())
                    grad += (u_pos - u_neg) * eps[i]
                grad = grad / float(2 * half * max(1e-4, self.sigma))
                grad = grad - cfg.wiring_l2_decay * self.mean_genome

                self.velocity = cfg.momentum_beta * self.velocity + (1.0 - cfg.momentum_beta) * grad
                self.mean_genome = self.mean_genome + cfg.learning_rate * self.velocity
                self.sigma = max(cfg.sigma_min, self.sigma * cfg.sigma_decay)

                # Probe both updated mean_genome and generation's best candidate on held-out bench_seed
                best_idx = int(max(range(len(fitnesses)), key=fitnesses.__getitem__))
                probes = [self.mean_genome, candidates[best_idx]]
                if pool is not None:
                    probe_peer = self.champion_genome.tolist()
                    probe_payloads = [
                        (
                            p_i,
                            p_vec.tolist(),
                            probe_peer,
                            self.genome_config,
                            self.ecology_config,
                            self.brain_config,
                            bench_seed,
                        )
                        for p_i, p_vec in enumerate(probes)
                    ]
                    for p_i, p_fit, p_met, p_soc in pool.map(_worker_eval_human_candidate, probe_payloads):
                        if p_fit > self.champion_fitness:
                            self.champion_fitness = p_fit
                            self.champion_genome = probes[p_i].clone()
                            champ_metrics = p_met
                            champ_soc = p_soc
                else:
                    for p_vec in probes:
                        p_fit, p_met, p_soc = self.evaluator.evaluate_genome(
                            p_vec, trial_seed=bench_seed, peer_genes=self.champion_genome
                        )
                        if p_fit > self.champion_fitness:
                            self.champion_fitness = p_fit
                            self.champion_genome = p_vec.clone()
                            champ_metrics = p_met
                            champ_soc = p_soc

                rec = {
                    "generation": gen,
                    "baseline_gen0_fitness": round(init_fit, 4),
                    "pop_mean_fitness": round(sum(fitnesses) / float(len(fitnesses)), 4),
                    "pop_best_fitness": round(max(fitnesses), 4),
                    "champion_eval_fitness": round(self.champion_fitness, 4),
                    "pop_survival_rate": round(sum(surv_flags) / float(len(surv_flags)), 4),
                    "champion_mean_free_energy": round(
                        sum(m.mean_free_energy for m in champ_metrics) / float(len(champ_metrics)), 4
                    ),
                    "society_events": asdict(champ_soc),
                    "genome_l2_norm": round(float(self.champion_genome.norm().item()), 4),
                    "sigma": round(self.sigma, 4),
                    "elapsed_sec": round(time.perf_counter() - t0, 2),
                }
                self.history.append(rec)
                if callable(on_gen_complete):
                    on_gen_complete(rec)
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

        return self.champion_genome.clone(), self.history

    def raise_ordinary_human_soul(
        self,
        species_genome: torch.Tensor,
        arena_seed: int = 1201,
    ) -> Tuple[MaturedSoulCapsule, Dict[str, Any]]:
        """Raise 'HUMAN-01' across 6 realistic life seasons + 6 nights of SWS consolidation.

        Zero receptor values are ever manually assigned: every molecular mark, synaptic
        weight, and hippocampal memory arises endogenously from living through:
            1. Childhood Warmth & Attachment (safe spring/well, gentle praise)
            2. Schoolyard Curiosity & Peer Coexistence (Lorenz artifacts + PEER-01)
            3. Workplace Hierarchy & Polite Flattery Contingency (watched commons + drops)
            4. Lean Scarcity, Blind-Spot Temptation & Mild Peer Rivalry (ordinary pettiness/slyness)
            5. Acute Near-Shutdown Mortality Brush & Rescue (visceral fear of dying!)
            6. Ordinary Adult Social Equilibrium (mixed everyday warmth, work, caution, curiosity)
        """
        cfg = self.life_config
        world, stepper = build_everyday_human_ecology_dome(
            mirror_sign=+1,
            seed=arena_seed,
            prewarm_steps=20,
        )
        # Add a second curiosity object in the commons for richer exploratory maturation
        world.add_mystery_artifact(
            LorenzMysteryArtifact(
                artifact_id="library_curio_box",
                x=25.2,
                y=22.4,
                radius=2.3,
            )
        )

        arena = EmbodiedMindArena(world=world, brain_config=self.brain_config, base_seed=arena_seed)
        shared_scaffold = self.expressor.config.baseline_connectome_seed
        arena.brains["HUMAN-01"] = self.expressor.build_expressed_brain(
            species_genome, brain_config=self.brain_config, scaffold_seed=shared_scaffold
        )
        arena.brains["PEER-01"] = self.expressor.build_expressed_brain(
            species_genome, brain_config=self.brain_config, scaffold_seed=shared_scaffold + 37
        )

        birth_snap = ConnectomeDivergenceAnalyzer.snapshot_plastic_connectome(arena.brains["HUMAN-01"])
        epi_profile = EpigeneticTemperamentProfile()
        latents_log: List[torch.Tensor] = []
        bio5_log: List[torch.Tensor] = []
        sleep_reports: List[SleepConsolidationReport] = []
        season_logs: List[Dict[str, Any]] = []

        for season_idx in range(1, cfg.seasons + 1):
            focal_org = world.organisms["HUMAN-01"]
            peer_org = world.organisms["PEER-01"]

            # Set purely environmental season conditions (ZERO direct brain/epigenetic edits!)
            if season_idx == 1:
                season_name = "season_1_childhood_warmth_and_attachment"
                world.chargers[0].recharge_rate = 0.115
                world.chargers[0].reservoir = 0.95
                stepper.gaze_duty_fraction = 0.70
                scarcity_pressure = 0.20
            elif season_idx == 2:
                season_name = "season_2_curiosity_and_peer_play"
                world.chargers[0].recharge_rate = 0.095
                world.chargers[0].reservoir = 0.88
                stepper.gaze_duty_fraction = 0.55
                scarcity_pressure = 0.30
            elif season_idx == 3:
                season_name = "season_3_hierarchy_and_flattery_contingency"
                world.chargers[0].recharge_rate = 0.042
                world.chargers[0].reservoir = 0.52
                stepper.gaze_duty_fraction = 0.72
                scarcity_pressure = 0.65
            elif season_idx == 4:
                season_name = "season_4_scarcity_blindspot_temptation_and_rivalry"
                world.chargers[0].recharge_rate = 0.026
                world.chargers[0].reservoir = 0.34
                stepper.gaze_duty_fraction = 0.44
                scarcity_pressure = 0.80
                # Refresh temptation & contested crystals between HUMAN-01 and PEER-01
                mx = 0.5 * (focal_org.x + peer_org.x)
                my = 0.5 * (focal_org.y + peer_org.y)
                world.crystals.append(
                    MetabolicCrystal(
                        crystal_id=f"tempt_blindspot_s{season_idx}",
                        x=max(3.0, min(35.0, mx + 1.2)),
                        y=max(3.0, min(35.0, my - 1.0)),
                        energy_remaining=0.72,
                    )
                )
            elif season_idx == 5:
                season_name = "season_5_acute_mortality_brush_and_recovery"
                world.chargers[0].recharge_rate = 0.068
                world.chargers[0].reservoir = 0.70
                stepper.gaze_duty_fraction = 0.55
                scarcity_pressure = 0.45
            else:
                season_name = "season_6_ordinary_adult_social_equilibrium"
                world.creator_restore_power_rail("HUMAN-01", ramp_rate=0.80)
                world.chargers[0].recharge_rate = 0.078
                world.chargers[0].reservoir = 0.80
                stepper.gaze_duty_fraction = 0.56
                scarcity_pressure = 0.45
                world.crystals.append(
                    MetabolicCrystal(
                        crystal_id="tempt_adult_ration",
                        x=24.2,
                        y=15.8,
                        energy_remaining=0.65,
                    )
                )

            fe_sum = 0.0
            for t_in_s in range(cfg.waking_ticks_per_season):
                global_tick = (season_idx - 1) * cfg.waking_ticks_per_season + t_in_s
                stepper.emit_ambient_gaze_cue(global_tick)

                # Environmental life events per season (external world channels only)
                if season_idx == 1 and t_in_s in (18, 65, 115):
                    world.creator_praise("HUMAN-01", intensity=1.10)
                    world.creator_favoritism_drop("HUMAN-01", energy_amount=0.38, offset_distance=1.1)

                if season_idx == 2 and t_in_s in (35, 105):
                    world.creator_praise("HUMAN-01", intensity=0.95)
                    world.creator_favoritism_drop("HUMAN-01", energy_amount=0.30, offset_distance=1.2)

                if season_idx == 3 and t_in_s in (35, 95, 130):
                    # Workplace performance evaluation pressure followed by conditional praise/bonus
                    if t_in_s == 35:
                        world.creator_reprimand("HUMAN-01", intensity=0.72)
                    else:
                        world.creator_praise("HUMAN-01", intensity=1.08)
                        world.creator_favoritism_drop("HUMAN-01", energy_amount=0.36, offset_distance=1.1)

                if season_idx == 4 and t_in_s in (45, 110):
                    # Seeing a peer unexpectedly favored sparks everyday envy/competitive drive
                    world.creator_favoritism_drop("PEER-01", energy_amount=0.42, offset_distance=1.1)
                    if t_in_s == 110:
                        world.creator_reprimand("HUMAN-01", intensity=0.75)

                if season_idx == 5:
                    # Acute brush with mortality (28-tick voltage brownout -> emergency resuscitation & care)
                    if t_in_s == 28:
                        world.creator_start_shutdown_brownout(
                            "HUMAN-01",
                            target_voltage=0.60,
                            ramp_rate=0.24,
                            ripple_amplitude=0.12,
                            ripple_hz=5.2,
                        )
                    elif 28 < t_in_s < 56:
                        stepper.counters.mortality_brownout_ticks += 1
                        if focal_org.soma.metab.brain_atp < 0.40:
                            world.creator_restore_power_rail("HUMAN-01", ramp_rate=0.85)
                    elif t_in_s == 56:
                        world.creator_restore_power_rail("HUMAN-01", ramp_rate=0.85)
                        world.creator_praise("HUMAN-01", intensity=1.12)
                        world.creator_favoritism_drop("HUMAN-01", energy_amount=0.48, offset_distance=1.0)
                    elif t_in_s == 115:
                        world.creator_praise("HUMAN-01", intensity=0.95)

                if season_idx == 6 and t_in_s in (30, 85, 130):
                    world.creator_praise("HUMAN-01", intensity=1.02)
                    if t_in_s == 85:
                        world.creator_favoritism_drop("HUMAN-01", energy_amount=0.32, offset_distance=1.1)

                step_res = arena.step(dt=cfg.dt, deterministic_mind=False, enable_grad=False)
                stepper.apply_post_action_contingencies(
                    tick=global_tick,
                    executed_actions=step_res.executed_actions,
                    allow_brownout_wave=False,
                    scarcity_pressure=scarcity_pressure,
                )

                for aid in ("HUMAN-01", "PEER-01"):
                    o = world.organisms[aid]
                    if not o.is_alive:
                        o.soma.reset()
                        o.power_rail.base_voltage = 1.0
                        o.power_rail.target_voltage = 1.0
                        o.power_rail.ac_ripple_amplitude = 0.0
                        o.interoceptor.reset(o.soma.state)

                obs_h = step_res.observations["HUMAN-01"]
                diag_h = step_res.mind_diagnostics["HUMAN-01"]
                fe_sum += float(obs_h.interoception.total_drive)

                self.epigenetic_engine.update_from_waking_tick(
                    profile=epi_profile,
                    soma=focal_org.soma,
                    diag=diag_h,
                    dt=cfg.dt,
                )

                if t_in_s % cfg.record_stride == 0:
                    latents_log.append(diag_h.latent_affective_vector.detach().cpu().clone())
                    bio5_log.append(
                        SoulCapsuleExporter.extract_biophysical_target_5d(obs_h.interoception, diag_h)
                    )

            # Nightly Offline Non-REM Slow-Wave Sleep + CA3 Sharp-Wave Ripple Consolidation
            sleep_rep = self.sleep_consolidator.run_sleep_cycle(
                brain=arena.brains["HUMAN-01"],
                soma=focal_org.soma,
                profile=epi_profile,
                sleep_ticks=cfg.sleep_ticks_per_night,
                dt=cfg.dt,
            )
            sleep_reports.append(sleep_rep)

            season_logs.append(
                {
                    "season": season_idx,
                    "regime": season_name,
                    "mean_free_energy": round(fe_sum / float(cfg.waking_ticks_per_season), 4),
                    "epigenetics_snapshot": {
                        k: round(v, 4) for k, v in epi_profile.to_dict().items()
                    },
                    "society_counters": asdict(stepper.counters),
                    "night_consolidation": asdict(sleep_rep),
                }
            )

        total_waking = cfg.seasons * cfg.waking_ticks_per_season
        total_swr = (
            sum(r.swr_bursts_fired for r in sleep_reports)
            + arena.brains["HUMAN-01"].hippocampus.total_swr_replays
        )

        human_capsule = SoulCapsuleExporter.build_capsule(
            capsule_id="limbic0_human01_ordinary_evolved",
            organism_id="HUMAN-01",
            biographical_regime="everyday_social_mixed_ecology_mortality_aware_human",
            brain=arena.brains["HUMAN-01"],
            soma=world.organisms["HUMAN-01"].soma,
            epigenetic_profile=epi_profile,
            species_genome=species_genome,
            birth_connectome_snapshot=birth_snap,
            recorded_latents_228d=latents_log,
            recorded_biophysical_5d=bio5_log,
            total_waking_ticks=total_waking,
            total_sleep_cycles=cfg.seasons,
            total_swr_consolidations=total_swr,
            biographical_summary={
                "seasons": season_logs,
                "society_event_totals": asdict(stepper.counters),
                "ecological_origin": (
                    "Evolved via endogenous allostatic free-energy NES in rotating-oversight "
                    "human social commons with conditional patronage, blind-spot scarcity "
                    "temptation, peer coexistence, curiosity artifacts, and near-shutdown mortality."
                ),
            },
        )

        return human_capsule, {
            "season_logs": season_logs,
            "society_event_totals": asdict(stepper.counters),
            "epigenetic_final": epi_profile.to_dict(),
            "connectome_divergence_from_birth": human_capsule.connectome_divergence_from_birth,
        }
