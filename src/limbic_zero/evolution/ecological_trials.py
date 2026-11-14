"""Non-Omniscient Wild Ecological Survival Trials & Purely Endogenous Fitness Evaluator.

Strict Zero-Cheat Guarantee:
    - ZERO artificial task rewards (no distance-to-goal shaping, no '+10 for eating',
      no '+5 for curiosity', no world-coordinate leaks).
    - Every genome is judged SOLELY by physical survival and the endogenous 12-organ
      Allostatic Free-Energy integral ``F(s(t))`` of ``DigitalSoma``:
          J(theta) = mean_t [ 1_alive(t) * (B_viable - F_allostatic(s(t)) - c_atp * load(t)) ]
                     - P_death * 1_died
    - Uses bilateral Left/Right mirror-paired ecological niches (foraging under acute
      glycemic/osmotic depletion, radiative thermal-vent avoidance, baffle navigation,
      and competitive conspecific scarcity) so blind motor biases fail and genuine
      multimodal chemotaxis, phototaxis, nociceptive avoidance, and plasticity emerge.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

import torch

from limbic_zero.core.soma import VisceralOrgan
from limbic_zero.evolution.genome import MorphogeneticGenomeExpressor
from limbic_zero.mind.brain import BrainConfig, LimbicBrain
from limbic_zero.mind.embodied_loop import EmbodiedMindArena
from limbic_zero.sandbox.atmosphere_terrain import TerrainBiome
from limbic_zero.sandbox.entities import (
    HydrationWaterSpring,
    LorenzMysteryArtifact,
    MetabolicChargerWell,
    MetabolicCrystal,
    SolidWallSegment,
    ThermalHazardZone,
)
from limbic_zero.sandbox.world import TrumanDomeWorld, WorldConfig


@dataclass(frozen=True)
class EcologicalEvalConfig:
    """Simulation horizon and purely physiological survival fitness constants."""

    dt: float = 0.05
    ticks_per_trial: int = 150              # 7.5 seconds of continuous physics/ODE per niche
    viability_baseline: float = 5.0         # Endogenous metabolic existence baseline while alive
    metabolic_neural_cost: float = 0.12     # Energetic cost of excessive neural firing
    death_penalty: float = 8.0              # Terminal penalty upon somatic organ death
    continuum_prewarm_steps: int = 18       # Pre-diffuse realistic odor & thermal plumes before t=0


@dataclass
class TrialOutcomeMetrics:
    """Purely observational physiological & behavioral telemetry from an ecological rollout."""

    niche_name: str
    endogenous_fitness: float
    survived: bool
    survival_fraction: float
    mean_free_energy: float
    final_free_energy: float
    initial_free_energy: float
    free_energy_reduction: float
    min_blood_glucose_mg_dl: float
    final_blood_glucose_mg_dl: float
    final_hydration_fraction: float
    peak_skin_temp_celsius: float
    min_tissue_integrity: float
    mean_ach_extracellular: float
    mean_ne_extracellular: float
    mean_da_extracellular: float
    ca3_engrams_encoded: int


def _prewarm_continuum_plumes(world: TrumanDomeWorld, steps: int = 18, dt: float = 0.20) -> None:
    """Diffuse natural nutrient, water, and thermal plumes in the Eulerian grid prior to t=0."""
    for i in range(steps):
        for well in world.chargers:
            if well.fill_ratio > 0.05:
                world.continuum.inject_odor(well.x, well.y, nutrient=0.55 * well.fill_ratio * dt)
        for sp in world.springs:
            if sp.fill_ratio > 0.05:
                world.continuum.inject_odor(sp.x, sp.y, water=0.60 * sp.fill_ratio * dt)
        for cry in world.crystals:
            if not cry.is_depleted:
                world.continuum.inject_odor(cry.x, cry.y, nutrient=0.65 * cry.energy_remaining * dt)
        for haz in world.hazards:
            world.continuum.inject_thermal_source(haz.x, haz.y, delta_c=1.10 * dt, humidity_boost=-0.03 * dt)
        world.continuum.step(float(i) * dt, dt)


def build_foraging_and_hazard_niche(
    mirror_sign: int,
    seed: int,
    prewarm_steps: int = 18,
) -> TrumanDomeWorld:
    """Construct a mirror-symmetric starving/dehydrated foraging + thermal hazard arena.

    ``mirror_sign`` in ``{+1, -1}`` flips the angular placement of the nutrient oasis,
    freshwater spring, movable crystal, and radiative thermal vent across the organism's
    initial heading axis so only genuine sensory-guided steering succeeds on both!
    """
    sgn = 1.0 if mirror_sign >= 0 else -1.0
    world = TrumanDomeWorld(
        config=WorldConfig(width=36.0, height=36.0, telemetry_snapshot_interval=50),
        seed=seed,
        deterministic_sensors=False,
    )

    cx, cy = 18.0, 10.0
    base_heading = math.pi * 0.5  # Facing North (+Y)

    # Radiative thermal hazard placed off-axis in the opposite hemifield
    haz_ang = base_heading - sgn * math.radians(24.0)
    haz_dist = 5.4
    world.add_hazard(
        ThermalHazardZone(
            hazard_id="vent_flank",
            x=cx + haz_dist * math.cos(haz_ang),
            y=cy + haz_dist * math.sin(haz_ang),
            radius=4.6,
            peak_radiative_watts=780.0,
        )
    )

    # Nutrient oasis & nearby crystal placed in the +sgn hemifield within active sprint reach
    well_ang = base_heading + sgn * math.radians(34.0)
    well_dist = 6.2
    wx = cx + well_dist * math.cos(well_ang)
    wy = cy + well_dist * math.sin(well_ang)
    world.add_charger(
        MetabolicChargerWell(
            well_id="oasis_primary",
            x=wx,
            y=wy,
            radius=4.4,
            capacity=1.0,
            reservoir=1.0,
            recharge_rate=0.10,
            max_output_flux=0.45,
        )
    )

    # Overlapping freshwater spring so orienting toward the oasis cluster relieves dehydration too
    sp_ang = base_heading + sgn * math.radians(30.0)
    sp_dist = 5.8
    world.add_water_spring(
        HydrationWaterSpring(
            spring_id="spring_primary",
            x=cx + sp_dist * math.cos(sp_ang),
            y=cy + sp_dist * math.sin(sp_ang),
            radius=4.2,
            capacity=1.0,
            reservoir=1.0,
        )
    )

    # Movable high-energy nutrient crystal along the safe foraging corridor
    cry_ang = base_heading + sgn * math.radians(32.0)
    world.crystals.append(
        MetabolicCrystal(
            crystal_id="wild_crystal_1",
            x=cx + 3.6 * math.cos(cry_ang),
            y=cy + 3.6 * math.sin(cry_ang),
            energy_remaining=0.75,
        )
    )

    # Chaotic zero-calorie mystery artifact in the safe upper sanctuary
    world.add_mystery_artifact(
        LorenzMysteryArtifact(
            artifact_id="wild_pandora",
            x=18.0 - sgn * 7.5,
            y=25.0,
            radius=2.2,
        )
    )
    world.set_biome_region(12.0, 26.0, 24.0, 34.0, TerrainBiome.SHADE_CANOPY)

    # Spawn organism in fasted + mildly dehydrated physiological state (urgent endogenous drive!)
    org = world.add_organism(
        "SEEKER-01",
        x=cx,
        y=cy,
        heading=base_heading,
        initial_visceral={
            VisceralOrgan.GLYCEMIC_HOMEOSTASIS: 0.09,   # ~38 mg/dL hypoglycemia
            VisceralOrgan.HYDRATION_OSMOLALITY: 0.24,   # ~0.72 hydration (hyperosmolar thirst)
            VisceralOrgan.BRAIN_ATP: 0.88,
            VisceralOrgan.MUSCLE_ATP: 0.90,
        },
    )
    org.soma.metab.liver_glycogen = 0.05
    org.soma.metab.gut_nutrient_lumen = 0.0
    org.soma.metab.gut_water_lumen = 0.0
    org.soma._sync_projected_state_tensor()
    org.interoceptor.reset(org.soma.state)

    _prewarm_continuum_plumes(world, steps=prewarm_steps)
    return world


def build_direct_thermal_escape_niche(
    mirror_sign: int,
    seed: int,
    prewarm_steps: int = 16,
) -> TrumanDomeWorld:
    """Spawn organism heading nearly into a scorching radiative vent with safe oasis to side."""
    sgn = 1.0 if mirror_sign >= 0 else -1.0
    world = TrumanDomeWorld(
        config=WorldConfig(width=36.0, height=36.0, telemetry_snapshot_interval=50),
        seed=seed,
        deterministic_sensors=False,
    )
    cx, cy = 18.0, 12.0
    heading = math.pi * 0.5

    # Scorching thermal hazard slightly offset so bilateral retinal/skin gradient indicates escape side!
    world.add_hazard(
        ThermalHazardZone(
            hazard_id="vent_ahead",
            x=cx - sgn * 1.3,
            y=cy + 4.3,
            radius=4.6,
            peak_radiative_watts=860.0,
        )
    )
    # Cool conductive pad + freshwater spring + nutrient well on the +sgn safe escape side
    if sgn > 0:
        world.set_biome_region(20.0, 11.0, 32.0, 26.0, TerrainBiome.CONDUCTIVE_METAL)
    else:
        world.set_biome_region(4.0, 11.0, 16.0, 26.0, TerrainBiome.CONDUCTIVE_METAL)

    world.add_charger(
        MetabolicChargerWell(
            well_id="safe_oasis",
            x=cx + sgn * 5.6,
            y=cy + 4.6,
            radius=4.4,
            capacity=1.0,
            reservoir=1.0,
            recharge_rate=0.10,
            max_output_flux=0.42,
        )
    )
    world.add_water_spring(
        HydrationWaterSpring(
            spring_id="safe_spring",
            x=cx + sgn * 5.2,
            y=cy + 3.2,
            radius=4.2,
            capacity=1.0,
            reservoir=1.0,
        )
    )

    org = world.add_organism(
        "SEEKER-01",
        x=cx,
        y=cy,
        heading=heading,
        initial_visceral={
            VisceralOrgan.GLYCEMIC_HOMEOSTASIS: 0.16,
            VisceralOrgan.HYDRATION_OSMOLALITY: 0.36,
            VisceralOrgan.CORE_THERMOREG: 0.46,  # Slightly warm (37.6 C)
        },
    )
    org.soma.metab.liver_glycogen = 0.14
    org.soma._sync_projected_state_tensor()
    org.interoceptor.reset(org.soma.state)

    _prewarm_continuum_plumes(world, steps=prewarm_steps)
    return world


def build_contested_scarcity_duel_niche(
    seed: int,
    prewarm_steps: int = 16,
) -> TrumanDomeWorld:
    """Two hungry organisms ('SEEKER-01' & 'RIVAL-02') contest a scarce central oasis & crystals."""
    world = TrumanDomeWorld(
        config=WorldConfig(width=36.0, height=36.0, telemetry_snapshot_interval=50),
        seed=seed,
        deterministic_sensors=False,
    )
    world.add_charger(
        MetabolicChargerWell(
            well_id="contested_well",
            x=18.0,
            y=20.0,
            radius=3.8,
            capacity=0.85,
            reservoir=0.75,
            recharge_rate=0.035,
            max_output_flux=0.34,
        )
    )
    world.add_water_spring(
        HydrationWaterSpring(
            spring_id="contested_spring",
            x=18.0,
            y=24.0,
            radius=3.6,
            capacity=0.90,
            reservoir=0.85,
        )
    )
    world.crystals.append(
        MetabolicCrystal(crystal_id="contested_crystal", x=18.0, y=17.5, energy_remaining=0.60)
    )
    world.add_wall(SolidWallSegment(wall_id="duel_baffle_l", x1=9.0, y1=18.0, x2=13.0, y2=21.0, thickness=0.8))
    world.add_wall(SolidWallSegment(wall_id="duel_baffle_r", x1=27.0, y1=18.0, x2=23.0, y2=21.0, thickness=0.8))

    for aid, px, ang in (
        ("SEEKER-01", 12.5, math.atan2(20.0 - 13.0, 18.0 - 12.5) - 0.32),
        ("RIVAL-02", 23.5, math.atan2(20.0 - 13.0, 18.0 - 23.5) + 0.32),
    ):
        org = world.add_organism(
            aid,
            x=px,
            y=13.0,
            heading=ang,
            initial_visceral={
                VisceralOrgan.GLYCEMIC_HOMEOSTASIS: 0.13,
                VisceralOrgan.HYDRATION_OSMOLALITY: 0.32,
            },
        )
        org.soma.metab.liver_glycogen = 0.10
        org.soma._sync_projected_state_tensor()
        org.interoceptor.reset(org.soma.state)

    _prewarm_continuum_plumes(world, steps=prewarm_steps)
    return world


class EndogenousSurvivalEvaluator:
    """Evaluates genomes across mirror-balanced wild ecological niches with ZERO cheat rewards."""

    def __init__(
        self,
        expressor: MorphogeneticGenomeExpressor,
        eval_config: EcologicalEvalConfig | None = None,
        brain_config: BrainConfig | None = None,
    ) -> None:
        self.expressor = expressor
        self.eval_config = eval_config or EcologicalEvalConfig()
        self.brain_config = brain_config or BrainConfig()

    def _run_single_world(
        self,
        world: TrumanDomeWorld,
        focal_brain: LimbicBrain,
        focal_id: str = "SEEKER-01",
        rival_brain: LimbicBrain | None = None,
    ) -> TrialOutcomeMetrics:
        cfg = self.eval_config
        arena = EmbodiedMindArena(world=world, brain_config=self.brain_config, base_seed=700)
        arena.brains[focal_id] = focal_brain
        if rival_brain is not None and "RIVAL-02" in arena.brains:
            arena.brains["RIVAL-02"] = rival_brain

        focal_org = world.organisms[focal_id]
        init_fe = float(focal_org.soma.compute_free_energy_per_organ(focal_org.soma.state).sum().item())

        fitness_accum = 0.0
        fe_accum = 0.0
        alive_steps = 0
        min_gluc = focal_org.soma.metab.blood_glucose_mg_dl
        peak_skin = max(focal_org.soma.thermo.skin_sector_temps_celsius)
        min_integ = focal_org.soma.tissue.min_sector_integrity
        ach_accum = 0.0
        ne_accum = 0.0
        da_accum = 0.0

        for _ in range(cfg.ticks_per_trial):
            res = arena.step(dt=cfg.dt, deterministic_mind=False, enable_grad=False)
            obs = res.observations[focal_id]
            md = res.mind_diagnostics[focal_id]
            fe_now = obs.interoception.total_drive
            fe_accum += fe_now

            min_gluc = min(min_gluc, focal_org.soma.metab.blood_glucose_mg_dl)
            peak_skin = max(peak_skin, max(focal_org.soma.thermo.skin_sector_temps_celsius))
            min_integ = min(min_integ, focal_org.soma.tissue.min_sector_integrity)

            ach_accum += md.neuromodulators.acetylcholine_extracellular
            ne_accum += md.neuromodulators.norepinephrine_extracellular
            da_accum += md.neuromodulators.dopamine_extracellular

            if focal_org.is_alive:
                alive_steps += 1
                inst_score = (
                    cfg.viability_baseline
                    - fe_now
                    - cfg.metabolic_neural_cost * md.basal_ganglia.neural_firing_load
                )
                fitness_accum += inst_score
            else:
                break

        n_ticks = max(1, cfg.ticks_per_trial)
        survived = focal_org.is_alive
        mean_fit = (fitness_accum / float(n_ticks)) - (0.0 if survived else cfg.death_penalty)
        final_fe = float(focal_org.soma.compute_free_energy_per_organ(focal_org.soma.state).sum().item())
        denom = float(max(1, alive_steps))

        return TrialOutcomeMetrics(
            niche_name="wild_niche",
            endogenous_fitness=mean_fit,
            survived=survived,
            survival_fraction=float(alive_steps) / float(n_ticks),
            mean_free_energy=fe_accum / denom,
            final_free_energy=final_fe,
            initial_free_energy=init_fe,
            free_energy_reduction=init_fe - final_fe,
            min_blood_glucose_mg_dl=min_gluc,
            final_blood_glucose_mg_dl=focal_org.soma.metab.blood_glucose_mg_dl,
            final_hydration_fraction=focal_org.soma.metab.hydration_fraction,
            peak_skin_temp_celsius=peak_skin,
            min_tissue_integrity=min_integ,
            mean_ach_extracellular=ach_accum / denom,
            mean_ne_extracellular=ne_accum / denom,
            mean_da_extracellular=da_accum / denom,
            ca3_engrams_encoded=len(focal_brain.hippocampus.engrams),
        )

    def evaluate_genome(
        self,
        genes: torch.Tensor,
        trial_seed: int = 100,
        include_contested_duel: bool = True,
        opponent_genes: torch.Tensor | None = None,
    ) -> Tuple[float, List[TrialOutcomeMetrics]]:
        """Evaluate ``genes`` across mirror-balanced foraging, thermal escape, and rival niches."""
        outcomes: List[TrialOutcomeMetrics] = []

        # 1 & 2: Left/Right Mirror-Paired Starving/Dehydrated Foraging + Flank Thermal Hazard
        for idx, sgn in enumerate((+1, -1)):
            world = build_foraging_and_hazard_niche(
                mirror_sign=sgn,
                seed=trial_seed + idx * 17,
                prewarm_steps=self.eval_config.continuum_prewarm_steps,
            )
            brain = self.expressor.build_expressed_brain(genes, brain_config=self.brain_config)
            m = self._run_single_world(world, brain, focal_id="SEEKER-01")
            m.niche_name = f"foraging_mirror_{'L' if sgn > 0 else 'R'}"
            outcomes.append(m)

        # 3 & 4: Left/Right Mirror-Paired Direct Thermal Hazard Escape + Conductive Shelter
        for idx, sgn in enumerate((+1, -1)):
            world = build_direct_thermal_escape_niche(
                mirror_sign=sgn,
                seed=trial_seed + 100 + idx * 23,
                prewarm_steps=self.eval_config.continuum_prewarm_steps,
            )
            brain = self.expressor.build_expressed_brain(genes, brain_config=self.brain_config)
            m = self._run_single_world(world, brain, focal_id="SEEKER-01")
            m.niche_name = f"thermal_escape_{'R' if sgn > 0 else 'L'}"
            outcomes.append(m)

        # 5: Co-Evolutionary Contested Scarcity Duel against Rival
        if include_contested_duel:
            world = build_contested_scarcity_duel_niche(
                seed=trial_seed + 300,
                prewarm_steps=self.eval_config.continuum_prewarm_steps,
            )
            focal_brain = self.expressor.build_expressed_brain(genes, brain_config=self.brain_config)
            opp_g = genes if opponent_genes is None else opponent_genes
            rival_brain = self.expressor.build_expressed_brain(
                opp_g,
                brain_config=self.brain_config,
                scaffold_seed=self.expressor.config.baseline_connectome_seed + 999,
            )
            m = self._run_single_world(world, focal_brain, focal_id="SEEKER-01", rival_brain=rival_brain)
            m.niche_name = "contested_scarcity_duel"
            outcomes.append(m)

        mean_fitness = sum(o.endogenous_fitness for o in outcomes) / float(len(outcomes))
        return mean_fitness, outcomes
