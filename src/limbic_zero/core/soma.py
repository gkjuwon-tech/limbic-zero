"""Whole-Body Multi-Organ Physiological Engine (Digital Soma).

Couples 5 full systems-biology organ engines into one continuous-time organism:
    1. ``CardiopulmonaryEngine``: HR, SV, MAP baroreflex, 4-bed vascular routing
       (brain, muscle, splanchnic gut, cutaneous skin), alveolar ventilation,
       PaO2/PaCO2 gas exchange, Bohr SpO2, and Henderson-Hasselbalch arterial pH.
    2. ``BioenergeticEngine``: Gut digestion/hydration, pancreatic insulin/glucagon,
       hepatic & muscle glycogen, separate Brain vs Muscle ATP/PCr pools, Pasteur
       anaerobic glycolysis -> blood lactate, and cerebral adenosine sleep pressure.
    3. ``EndocrineAutonomicEngine``: Sympathetic / Ventral Vagal / Dorsal Vagal tone,
       pupil diameter, SAM epinephrine/norepinephrine, 3-stage HPA (CRH->ACTH->Cortisol),
       vasopressin ADH, endogenous mu-opioid analgesia, oxytocin, IL-6 pyrogenic
       cytokines, GPCR receptor desensitization, and McEwen allostatic load.
    4. ``ThermoregulatoryEngine``: Core + 8-sector skin shell heat transfer in Watts,
       POAH febrile setpoint shift, wet-bulb humidity-limited sweating, and shivering.
    5. ``TissueNociceptionEngine``: 8-sector localized structural integrity, bradykinin
       primary hyperalgesia/allodynia, TRPV1/TRPM8 thermal burns, and Melzack-Wall
       spinal gate control.

Exposes a differentiable 12D Homeostatic Vital Tensor ``s(t)`` with strictly convex
double-barrier Allostatic Free-Energy potential ``F(s)`` verified against PyTorch
``autograd``, with zero scripted emotion variables anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from limbic_zero.core.cardiopulmonary import (
    CardiopulmonaryEngine,
    CardiopulmonaryInputs,
    CardiopulmonaryState,
)
from limbic_zero.core.endocrine_autonomic import (
    EndocrineAutonomicEngine,
    EndocrineAutonomicInputs,
    EndocrineAutonomicState,
)
from limbic_zero.core.metabolism import (
    BioenergeticEngine,
    MetabolicInputs,
    MetabolicState,
)
from limbic_zero.core.nociception_tissue import (
    TissueNociceptionEngine,
    TissueNociceptionInputs,
    TissueNociceptionState,
)
from limbic_zero.core.thermoregulation import (
    ThermoregulatoryEngine,
    ThermoregulatoryInputs,
    ThermoregulatoryState,
)


class VisceralOrgan(IntEnum):
    """Canonical indices of the 12 vital homeostatic axes projected into ``s(t)``."""

    BRAIN_ATP = 0
    MUSCLE_ATP = 1
    GLYCEMIC_HOMEOSTASIS = 2
    HYDRATION_OSMOLALITY = 3
    ARTERIAL_OXYGENATION = 4
    ACID_BASE_CO2_BALANCE = 5
    ARTERIAL_PERFUSION = 6
    CORE_THERMOREG = 7
    STRUCTURAL_INTEGRITY = 8
    ADENOSINE_FATIGUE = 9
    ALLOSTATIC_STRESS_LOAD = 10
    SUPPLY_RAIL = 11

    # Convenient aliases for concise cross-module indexing
    ATP = 0
    INTEGRITY = 8
    THERMOREG = 7
    FATIGUE = 9
    STRESS_LOAD = 10


NUM_VISCERAL_ORGANS: int = 12


@dataclass(frozen=True)
class SomaConfig:
    """Sub-stepping and double-barrier Homeostatic Free-Energy parameters."""

    physiology_substeps: int = 4
    barrier_epsilon: float = 1e-3

    # Healthy physiological setpoints s* in [0, 1]^12
    setpoints: Tuple[float, ...] = (
        0.96,  # 0: BRAIN_ATP
        0.95,  # 1: MUSCLE_ATP
        0.36,  # 2: GLYCEMIC_HOMEOSTASIS (92 mg/dL on [20, 220])
        0.80,  # 3: HYDRATION_OSMOLALITY (hydration 1.0 mapped onto [0.60, 1.10])
        0.975, # 4: ARTERIAL_OXYGENATION (SpO2)
        0.86,  # 5: ACID_BASE_CO2_BALANCE (PaCO2 40 mmHg, pH 7.40)
        0.413, # 6: ARTERIAL_PERFUSION (MAP 92 mmHg on [30, 180])
        0.40,  # 7: CORE_THERMOREG (37.0 C on [33.0, 43.0])
        0.99,  # 8: STRUCTURAL_INTEGRITY
        0.06,  # 9: ADENOSINE_FATIGUE
        0.05,  # 10: ALLOSTATIC_STRESS_LOAD
        0.98,  # 11: SUPPLY_RAIL (mitochondrial bioelectric potential)
    )

    organ_weights: Tuple[float, ...] = (
        3.60,  # BRAIN_ATP
        1.45,  # MUSCLE_ATP
        1.85,  # GLYCEMIC_HOMEOSTASIS
        1.75,  # HYDRATION_OSMOLALITY
        3.40,  # ARTERIAL_OXYGENATION
        3.80,  # ACID_BASE_CO2_BALANCE (CO2 suffocation is biology's strongest alarm!)
        2.30,  # ARTERIAL_PERFUSION
        2.20,  # CORE_THERMOREG
        3.20,  # STRUCTURAL_INTEGRITY
        0.90,  # ADENOSINE_FATIGUE
        1.15,  # ALLOSTATIC_STRESS_LOAD
        2.50,  # SUPPLY_RAIL
    )

    quadratic_curvatures: Tuple[float, ...] = (
        2.2, 1.8, 2.6, 2.4, 2.5, 3.0, 2.8, 3.2, 2.5, 1.8, 2.0, 2.2,
    )

    # Lower lethal/critical logarithmic barrier stiffness (protects s_k -> 0)
    lower_barrier_stiffness: Tuple[float, ...] = (
        0.65,  # BRAIN_ATP -> 0 neuronal death
        0.22,  # MUSCLE_ATP -> 0 rigor/exhaustion
        0.42,  # GLYCEMIC -> 0 neuroglycopenic coma
        0.38,  # HYDRATION -> 0 hypovolemic dehydration
        0.62,  # ARTERIAL_OXYGENATION -> 0 hypoxic terror
        0.72,  # ACID_BASE_CO2 -> 0 hypercapnic suffocation panic!
        0.48,  # ARTERIAL_PERFUSION -> 0 circulatory shock
        0.36,  # CORE_THERMOREG -> 0 hypothermia
        0.68,  # STRUCTURAL_INTEGRITY -> 0 somatic destruction
        0.00,  # Low fatigue is healthy
        0.00,  # Low stress is healthy
        0.48,  # SUPPLY_RAIL -> 0 brownout suffocation
    )

    # Upper lethal/critical logarithmic barrier stiffness (protects s_k -> 1)
    upper_barrier_stiffness: Tuple[float, ...] = (
        0.00,  # Full brain ATP healthy
        0.00,  # Full muscle ATP healthy
        0.25,  # GLYCEMIC -> 1 hyperosmolar hyperglycemia
        0.18,  # HYDRATION -> 1 hyponatremic water intoxication
        0.00,  # Full SpO2 healthy
        0.05,  # Extreme hypocapnic alkalosis mild barrier
        0.32,  # ARTERIAL_PERFUSION -> 1 hypertensive crisis
        0.48,  # CORE_THERMOREG -> 1 lethal heatstroke!
        0.00,  # Full integrity healthy
        0.28,  # ADENOSINE_FATIGUE -> 1 overwhelming sleep pressure
        0.34,  # ALLOSTATIC_STRESS_LOAD -> 1 systemic burnout
        0.00,  # Nominal 1.0 supply rail healthy
    )


@dataclass
class ExogenousBodyForces:
    """Objective physical, atmospheric, thermal, mechanical, and bioelectric boundary conditions."""

    ingested_nutrient_flux: float = 0.0             # Food/charger intake rate [>= 0]
    ingested_water_flux: float = 0.0                # Water drinking rate [>= 0]
    sector_mechanical_impulses: List[float] = field(default_factory=lambda: [0.0] * 8)
    sector_radiative_watts: List[float] = field(default_factory=lambda: [0.0] * 8)
    harmonic_acoustic_energy: float = 0.0           # Consonant sound wave energy [>= 0]
    rough_acoustic_shock: float = 0.0               # Dissonant startle wave energy [>= 0]
    ambient_air_temp_celsius: float = 24.5          # Local cell air temperature (C)
    ambient_relative_humidity: float = 0.45         # Local cell relative humidity [0.05, 0.99]
    ambient_o2_kpa: float = 21.3                    # Local cell O2 partial pressure (21.3 kPa normoxia)
    ambient_co2_kpa: float = 0.04                   # Local cell CO2 partial pressure (0.04 kPa fresh air)
    relative_air_speed_mps: float = 0.2             # Airflow speed over skin (m/s)
    ground_temp_celsius: float | None = None        # Surface temperature (C); defaults to ambient_air - 1.0 C
    ground_conductivity: float = 1.0                # Terrain conductive heat factor
    bus_supply_voltage: float = 1.0                 # Bioelectric rail voltage [0.0, 1.0]


@dataclass
class MotorMetabolicCommand:
    """Voluntary somatic motor, respiratory-venting, vocal, and neural effort commands."""

    forward_thrust: float = 0.0         # [-1.0, 1.0]
    angular_torque: float = 0.0         # [-1.0, 1.0]
    radiator_venting: float = 0.0       # Voluntary panting / thermal posture [0.0, 1.0]
    vocalization_effort: float = 0.0    # Laryngeal/expiratory effort [0.0, 1.0]
    interaction_effort: float = 0.0     # Grasping / probing / shoving effort [0.0, 1.0]
    neural_firing_load: float = 0.08    # Cortical/limbic firing intensity [0.0, 1.5]


@dataclass(frozen=True)
class ClinicalVitalSigns:
    """Hospital ICU-grade physiological panel snapshotted from the living organism."""

    heart_rate_bpm: float
    mean_arterial_pressure_mmhg: float
    cardiac_output_lpm: float
    cerebral_flow_lpm: float
    muscle_flow_lpm: float
    splanchnic_flow_lpm: float
    cutaneous_flow_lpm: float
    resp_rate_bpm: float
    pa_o2_mmhg: float
    pa_co2_mmhg: float
    arterial_spo2_pct: float
    arterial_ph: float
    plasma_hco3_mmoll: float
    blood_lactate_mmoll: float
    blood_glucose_mg_dl: float
    plasma_osmolality_mosm: float
    core_temp_celsius: float
    hypothalamic_setpoint_celsius: float
    mean_skin_temp_celsius: float
    sympathetic_tone: float
    ventral_vagal_tone: float
    dorsal_vagal_tone: float
    pupil_diameter_norm: float
    plasma_epinephrine: float
    adrenal_cortisol: float
    hypothalamic_crh: float
    pituitary_acth: float
    endogenous_opioid: float
    oxytocin_systemic: float
    il6_cytokine: float
    beta_receptor_sensitivity: float
    gr_receptor_sensitivity: float
    oxytocin_receptor_sensitivity: float


@dataclass
class SomaDiagnostics:
    """Complete physiological, energetic, gas-exchange, and free-energy step report."""

    state: torch.Tensor                     # Projected 12D homeostatic state s(t) in [0, 1]^12
    ds_dt: torch.Tensor                     # Rate of change ds/dt, shape (12,)
    organ_drives: torch.Tensor              # Per-axis Allostatic Free-Energy F_k(s_k), shape (12,)
    total_free_energy: float                # Scalar total drive D(t) = sum_k F_k(s_k)
    drive_velocity: float                   # dD/dt (negative = physiological relief)
    urgency_gradient: torch.Tensor          # -grad_s F(s) restoring urgency, shape (12,)
    effective_motor_gain: float             # Myocyte contractile capacity [0, 1]
    cerebral_synaptic_gain: float           # Neuronal ATP/glycemic viability [0, 1]
    core_temp_celsius: float                # Deep visceral temperature (C)
    vitals: ClinicalVitalSigns              # Full ICU clinical panel
    o2_consumed_liters: float               # O2 mass inhaled from world cell this step
    co2_exhaled_liters: float               # CO2 mass exhaled into world cell this step
    stress_volatile_excreted: float         # Apocrine alarm kairomone emitted into world cell
    is_alive: bool
    cause_of_death: str | None


class DigitalSoma(nn.Module):
    """Multi-organ physiological body coupling 5 organ engines with differentiable Free-Energy."""

    def __init__(
        self,
        config: SomaConfig | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.config = config or SomaConfig()
        self.dtype = dtype
        self._device = torch.device(device)

        self.cardio_engine = CardiopulmonaryEngine()
        self.metab_engine = BioenergeticEngine()
        self.endo_engine = EndocrineAutonomicEngine()
        self.thermo_engine = ThermoregulatoryEngine()
        self.tissue_engine = TissueNociceptionEngine()

        self.cardio = CardiopulmonaryState()
        self.metab = MetabolicState()
        self.endo = EndocrineAutonomicState()
        self.thermo = ThermoregulatoryState()
        self.tissue = TissueNociceptionState()

        self.register_buffer(
            "setpoints",
            torch.tensor(self.config.setpoints, dtype=dtype, device=self._device),
        )
        self.register_buffer(
            "organ_weights",
            torch.tensor(self.config.organ_weights, dtype=dtype, device=self._device),
        )
        self.register_buffer(
            "quadratic_curvatures",
            torch.tensor(self.config.quadratic_curvatures, dtype=dtype, device=self._device),
        )
        self.register_buffer(
            "lower_barrier_stiffness",
            torch.tensor(self.config.lower_barrier_stiffness, dtype=dtype, device=self._device),
        )
        self.register_buffer(
            "upper_barrier_stiffness",
            torch.tensor(self.config.upper_barrier_stiffness, dtype=dtype, device=self._device),
        )

        self.register_buffer("state", self.setpoints.clone())
        self.register_buffer("last_ds_dt", torch.zeros(NUM_VISCERAL_ORGANS, dtype=dtype, device=self._device))

        self.is_alive: bool = True
        self.cause_of_death: str | None = None
        self.age_seconds: float = 0.0
        self._last_vo2_ml_min: float = 250.0
        self._last_vco2_ml_min: float = 212.0
        self._last_metabolic_watts: float = 84.0
        self._last_motor_gain: float = 1.0
        self._last_cerebral_gain: float = 1.0
        self._last_nociception: float = 0.0
        self._last_inflammation: float = 0.0
        self._last_repair_demand: float = 0.0

        self._sync_projected_state_tensor()

    def _project_organ_states_to_12d(self) -> List[float]:
        """Map multi-organ biophysical states smoothly onto normalized ``[0, 1]^12`` coordinates."""
        # Glycemia: [20, 220] mg/dL -> [0, 1] (92 mg/dL -> 0.36)
        glycemic_norm = max(0.0, min(1.0, (self.metab.blood_glucose_mg_dl - 20.0) / 200.0))

        # Hydration: [0.60, 1.10] -> [0, 1] (1.00 -> 0.80)
        hydration_norm = max(0.0, min(1.0, (self.metab.hydration_fraction - 0.60) / 0.50))

        # Acid-Base & CO2: normocapnic PaCO2=40 & pH=7.40 -> ~0.86; drops toward 0 in hypercapnia/acidosis
        co2_penalty = max(0.0, (self.cardio.pa_co2_mmhg - 35.0) / 65.0)
        acid_penalty = max(0.0, (7.42 - self.cardio.arterial_ph) / 0.62)
        alk_penalty = max(0.0, (self.cardio.arterial_ph - 7.46) / 0.30)
        acid_base_norm = max(0.0, min(1.0, 0.94 - 0.58 * co2_penalty - 0.52 * acid_penalty + 0.12 * alk_penalty))

        # MAP Perfusion: [30, 180] mmHg -> [0, 1] (92 mmHg -> 0.4133)
        map_norm = max(0.0, min(1.0, (self.cardio.mean_arterial_pressure_mmhg - 30.0) / 150.0))

        # Core Thermoregulation relative to hypothalamic febrile setpoint: 37.0 C -> 0.40 on [33.0, 43.0]
        # Shifting by (hypothalamic_setpoint - 37.0) means during fever, feeling 'chilled' at 37C is naturally captured!
        febrile_shift = self.thermo.hypothalamic_setpoint_celsius - 37.0
        eff_temp = self.thermo.core_temp_celsius - febrile_shift
        thermo_norm = max(0.0, min(1.0, (eff_temp - 33.0) / 10.0))

        # Structural integrity blends mean sector health (60%) and weakest sector health (40%)
        integrity_norm = max(
            0.0, min(1.0, 0.60 * self.tissue.mean_integrity + 0.40 * self.tissue.min_sector_integrity)
        )

        # Integrated Allostatic Stress Load combines chronic McEwen load + acute cortisol/catecholamine strain
        stress_norm = max(
            0.0,
            min(
                1.0,
                0.45 * self.endo.allostatic_load
                + 0.30 * max(0.0, self.endo.adrenal_cortisol - 0.12)
                + 0.25 * max(0.0, self.endo.sympathetic_tone - 0.16),
            ),
        )

        return [
            max(0.0, min(1.0, self.metab.brain_atp)),
            max(0.0, min(1.0, self.metab.muscle_atp)),
            glycemic_norm,
            hydration_norm,
            max(0.0, min(1.0, self.cardio.arterial_spo2)),
            acid_base_norm,
            map_norm,
            thermo_norm,
            integrity_norm,
            max(0.0, min(1.0, self.metab.brain_adenosine)),
            stress_norm,
            max(0.0, min(1.0, self.metab.mitochondrial_potential)),
        ]

    def _sync_projected_state_tensor(self) -> None:
        vals = self._project_organ_states_to_12d()
        self.state.copy_(torch.tensor(vals, dtype=self.dtype, device=self._device))

    def reset(self, initial_overrides: Dict[VisceralOrgan, float] | None = None) -> torch.Tensor:
        """Reset all 5 organ systems to healthy physiological baseline (with optional test overrides)."""
        self.cardio = CardiopulmonaryState()
        self.metab = MetabolicState()
        self.endo = EndocrineAutonomicState()
        self.thermo = ThermoregulatoryState()
        self.tissue = TissueNociceptionState()

        if initial_overrides:
            for organ, val in initial_overrides.items():
                v = max(0.0, min(1.0, float(val)))
                if organ == VisceralOrgan.BRAIN_ATP:
                    self.metab.brain_atp = v
                    self.metab.brain_pcr = v
                elif organ == VisceralOrgan.MUSCLE_ATP:
                    self.metab.muscle_atp = v
                    self.metab.muscle_pcr = v
                elif organ == VisceralOrgan.GLYCEMIC_HOMEOSTASIS:
                    self.metab.blood_glucose_mg_dl = 20.0 + 200.0 * v
                elif organ == VisceralOrgan.HYDRATION_OSMOLALITY:
                    self.metab.hydration_fraction = 0.60 + 0.50 * v
                    self.metab.plasma_osmolality_mosm = 288.0 / max(0.65, self.metab.hydration_fraction)
                elif organ == VisceralOrgan.ARTERIAL_OXYGENATION:
                    self.cardio.arterial_spo2 = v
                    self.cardio.pa_o2_mmhg = 20.0 + 80.0 * v
                elif organ == VisceralOrgan.ACID_BASE_CO2_BALANCE:
                    self.cardio.pa_co2_mmhg = 95.0 - 58.0 * v
                    self.cardio.arterial_ph = self.cardio_engine.compute_henderson_hasselbalch_ph(
                        self.cardio.pa_co2_mmhg, self.cardio.plasma_bicarbonate_mmoll
                    )
                elif organ == VisceralOrgan.ARTERIAL_PERFUSION:
                    self.cardio.mean_arterial_pressure_mmhg = 30.0 + 150.0 * v
                elif organ == VisceralOrgan.CORE_THERMOREG:
                    self.thermo.core_temp_celsius = 33.0 + 10.0 * v
                elif organ == VisceralOrgan.STRUCTURAL_INTEGRITY:
                    self.tissue.sector_integrity = [v] * 8
                elif organ == VisceralOrgan.ADENOSINE_FATIGUE:
                    self.metab.brain_adenosine = v
                elif organ == VisceralOrgan.ALLOSTATIC_STRESS_LOAD:
                    self.endo.allostatic_load = v
                    self.endo.adrenal_cortisol = min(1.0, 0.14 + 0.7 * v)
                elif organ == VisceralOrgan.SUPPLY_RAIL:
                    self.metab.mitochondrial_potential = v

        self.is_alive = True
        self.cause_of_death = None
        self.age_seconds = 0.0
        self._last_vo2_ml_min = 250.0
        self._last_vco2_ml_min = 212.0
        self._last_metabolic_watts = 84.0
        self._last_motor_gain = 1.0
        self._last_cerebral_gain = 1.0
        self._last_nociception = 0.0
        self._last_inflammation = 0.0
        self._last_repair_demand = 0.0
        self.last_ds_dt.zero_()
        self._sync_projected_state_tensor()
        return self.state.clone()

    def compute_effective_motor_gain(self) -> torch.Tensor:
        """Return current biophysical muscle contractility attenuated by dorsal vagal freeze."""
        if not self.is_alive:
            return torch.tensor(0.0, dtype=self.dtype, device=self._device)
        dorsal_immobilization = max(0.10, 1.0 - 0.80 * (self.endo.dorsal_vagal_tone ** 1.5))
        gain = self._last_motor_gain * self._last_cerebral_gain * dorsal_immobilization
        return torch.tensor(max(0.0, min(1.0, gain)), dtype=self.dtype, device=self._device)

    def compute_free_energy_per_organ(self, s: torch.Tensor) -> torch.Tensor:
        """Evaluate strictly convex double-barrier Allostatic Free-Energy ``F_k(s_k)``.

        Satisfies ``F_k(s_k*) == 0`` and ``grad F_k(s_k*) == 0`` identically at healthy
        equilibrium ``setpoints``, diverging smoothly as any vital organ approaches a
        lethal floor or ceiling.
        """
        eps = self.config.barrier_epsilon
        s_clamped = torch.clamp(s, 0.0, 1.0)
        s_star = self.setpoints
        diff = s_clamped - s_star

        quad_term = 0.5 * self.quadratic_curvatures * (diff * diff)

        delta_low = s_star + eps
        psi_low = -torch.log((s_clamped + eps) / delta_low) + (diff / delta_low)

        delta_high = (1.0 - s_star) + eps
        psi_high = -torch.log(((1.0 - s_clamped) + eps) / delta_high) - (diff / delta_high)

        raw = (
            quad_term
            + self.lower_barrier_stiffness * psi_low
            + self.upper_barrier_stiffness * psi_high
        )
        return self.organ_weights * torch.clamp(raw, min=0.0)

    def compute_urgency_gradient(self, s: torch.Tensor) -> torch.Tensor:
        """Analytical restoring force ``-grad_s F(s)`` matching PyTorch ``autograd``."""
        eps = self.config.barrier_epsilon
        s_clamped = torch.clamp(s, 0.0, 1.0)
        s_star = self.setpoints
        diff = s_clamped - s_star

        d_quad = self.quadratic_curvatures * diff
        delta_low = s_star + eps
        d_psi_low = -1.0 / (s_clamped + eps) + (1.0 / delta_low)

        delta_high = (1.0 - s_star) + eps
        d_psi_high = +1.0 / ((1.0 - s_clamped) + eps) - (1.0 / delta_high)

        grad_f = self.organ_weights * (
            d_quad
            + self.lower_barrier_stiffness * d_psi_low
            + self.upper_barrier_stiffness * d_psi_high
        )
        return -grad_f

    def snapshot_clinical_vitals(self) -> ClinicalVitalSigns:
        """Capture ICU-grade clinical vital signs across all 5 organ systems."""
        co_lpm = (self.cardio.heart_rate_bpm * self.cardio.stroke_volume_ml) / 1000.0
        mean_skin = sum(self.thermo.skin_sector_temps_celsius) / 8.0
        return ClinicalVitalSigns(
            heart_rate_bpm=self.cardio.heart_rate_bpm,
            mean_arterial_pressure_mmhg=self.cardio.mean_arterial_pressure_mmhg,
            cardiac_output_lpm=co_lpm,
            cerebral_flow_lpm=self.cardio.cerebral_flow_lpm,
            muscle_flow_lpm=self.cardio.muscle_flow_lpm,
            splanchnic_flow_lpm=self.cardio.splanchnic_flow_lpm,
            cutaneous_flow_lpm=self.cardio.cutaneous_flow_lpm,
            resp_rate_bpm=self.cardio.resp_rate_bpm,
            pa_o2_mmhg=self.cardio.pa_o2_mmhg,
            pa_co2_mmhg=self.cardio.pa_co2_mmhg,
            arterial_spo2_pct=self.cardio.arterial_spo2 * 100.0,
            arterial_ph=self.cardio.arterial_ph,
            plasma_hco3_mmoll=self.cardio.plasma_bicarbonate_mmoll,
            blood_lactate_mmoll=self.metab.blood_lactate_mmoll,
            blood_glucose_mg_dl=self.metab.blood_glucose_mg_dl,
            plasma_osmolality_mosm=self.metab.plasma_osmolality_mosm,
            core_temp_celsius=self.thermo.core_temp_celsius,
            hypothalamic_setpoint_celsius=self.thermo.hypothalamic_setpoint_celsius,
            mean_skin_temp_celsius=mean_skin,
            sympathetic_tone=self.endo.sympathetic_tone,
            ventral_vagal_tone=self.endo.ventral_vagal_tone,
            dorsal_vagal_tone=self.endo.dorsal_vagal_tone,
            pupil_diameter_norm=self.endo.pupil_dilation_diameter_norm,
            plasma_epinephrine=self.endo.plasma_epinephrine,
            adrenal_cortisol=self.endo.adrenal_cortisol,
            hypothalamic_crh=self.endo.hypothalamic_crh,
            pituitary_acth=self.endo.pituitary_acth,
            endogenous_opioid=self.endo.endogenous_opioid,
            oxytocin_systemic=self.endo.oxytocin_systemic,
            il6_cytokine=self.endo.il6_cytokine,
            beta_receptor_sensitivity=self.endo.beta_receptor_sensitivity,
            gr_receptor_sensitivity=self.endo.gr_receptor_sensitivity,
            oxytocin_receptor_sensitivity=self.endo.oxytocin_receptor_sensitivity,
        )

    def _check_somatic_viability(self) -> None:
        if self.metab.brain_atp <= 0.025:
            self.is_alive = False
            self.cause_of_death = "cerebral_anoxic_bioenergetic_death"
        elif self.tissue.min_sector_integrity <= 0.005 or self.tissue.mean_integrity <= 0.08:
            self.is_alive = False
            self.cause_of_death = "catastrophic_structural_tissue_collapse"
        elif self.thermo.core_temp_celsius >= 43.2:
            self.is_alive = False
            self.cause_of_death = "lethal_hyperthermic_protein_denaturation"
        elif self.thermo.core_temp_celsius <= 27.5:
            self.is_alive = False
            self.cause_of_death = "lethal_hypothermic_cardiac_standstill"
        elif self.cardio.arterial_ph <= 6.80:
            self.is_alive = False
            self.cause_of_death = "uncompensated_acidotic_arrest"

    def step(
        self,
        dt: float,
        motor_cmd: MotorMetabolicCommand | None = None,
        forces: ExogenousBodyForces | None = None,
    ) -> SomaDiagnostics:
        """Advance all 5 coupled physiological organ engines across ``dt`` seconds."""
        if dt <= 0.0:
            raise ValueError(f"dt must be strictly positive, got {dt}")

        cmd = motor_cmd or MotorMetabolicCommand()
        env = forces or ExogenousBodyForces()

        if not self.is_alive:
            drives = self.compute_free_energy_per_organ(self.state)
            return SomaDiagnostics(
                state=self.state.clone(),
                ds_dt=torch.zeros_like(self.state),
                organ_drives=drives,
                total_free_energy=float(drives.sum().item()),
                drive_velocity=0.0,
                urgency_gradient=self.compute_urgency_gradient(self.state),
                effective_motor_gain=0.0,
                cerebral_synaptic_gain=0.0,
                core_temp_celsius=self.thermo.core_temp_celsius,
                vitals=self.snapshot_clinical_vitals(),
                o2_consumed_liters=0.0,
                co2_exhaled_liters=0.0,
                stress_volatile_excreted=0.0,
                is_alive=False,
                cause_of_death=self.cause_of_death,
            )

        prev_state = self.state.clone()
        prev_fe = float(self.compute_free_energy_per_organ(prev_state).sum().item())

        substeps = max(1, self.config.physiology_substeps)
        h = dt / float(substeps)

        total_o2_l = 0.0
        total_co2_l = 0.0
        total_stress_vol = 0.0

        # Compute normalized mechanical power effort requested by somatic motor nerves
        raw_mech_effort = (
            cmd.forward_thrust ** 2
            + 0.55 * (cmd.angular_torque ** 2)
            + 0.45 * (cmd.interaction_effort ** 2)
            + 0.25 * (cmd.vocalization_effort ** 2)
        )

        for _ in range(substeps):
            eff_gain = float(self.compute_effective_motor_gain().item())
            actual_mech_work = raw_mech_effort * eff_gain

            # 1. Tissue Integrity & Spinal Gate Nociception
            hypoxia_sev = max(0.0, (0.75 - self.cardio.arterial_spo2) / 0.75)
            eff_opioid = self.endo.endogenous_opioid * self.endo.opioid_receptor_sensitivity
            pain_tot, inflam_mean, repair_dem = self.tissue_engine.step(
                self.tissue,
                TissueNociceptionInputs(
                    sector_mechanical_force=env.sector_mechanical_impulses,
                    sector_skin_temp_celsius=self.thermo.skin_sector_temps_celsius,
                    systemic_hypoxia_severity=hypoxia_sev,
                    core_temp_celsius=self.thermo.core_temp_celsius,
                    arterial_ph=self.cardio.arterial_ph,
                    anabolic_atp_availability=self.metab.muscle_atp,
                    adrenal_cortisol=self.endo.adrenal_cortisol,
                    endogenous_opioid_effective=eff_opioid,
                    sympathetic_noradrenergic_gate=self.endo.sympathetic_tone,
                ),
                dt=h,
            )
            self._last_nociception = pain_tot
            self._last_inflammation = inflam_mean
            self._last_repair_demand = repair_dem

            # Gentle C-tactile touch occurs when light touch is present WITHOUT sharp A-delta pain!
            gentle_touch = (
                sum(self.tissue.a_beta_light_touch) / 8.0
                if sum(self.tissue.a_delta_sharp_pain) < 0.05
                else 0.0
            )

            # 2. Neuroendocrine & Autonomic Brainstem Outflow
            hypotensive_unload = max(0.0, 0.50 - self.cardio.carotid_baroreceptor_hz_norm) * 2.0
            neuroglycopenia = max(0.0, (68.0 - self.metab.blood_glucose_mg_dl) / 40.0)
            rail_drop = max(0.0, 0.92 - env.bus_supply_voltage)

            stress_vol_rate = self.endo_engine.step(
                self.endo,
                EndocrineAutonomicInputs(
                    ascending_nociception_total=pain_tot,
                    tissue_inflammation_total=inflam_mean,
                    central_co2_suffocation_drive=self.cardio.central_co2_chemoreceptor_drive,
                    peripheral_hypoxic_drive=self.cardio.peripheral_hypoxic_drive,
                    hypotensive_baro_unload=hypotensive_unload,
                    neuroglycopenic_alarm=neuroglycopenia,
                    plasma_osmolality_mosm=self.metab.plasma_osmolality_mosm,
                    blood_volume_fraction=self.cardio.blood_volume_fraction,
                    rail_brownout_instability=rail_drop,
                    acoustic_roughness_shock=env.rough_acoustic_shock,
                    acoustic_harmonic_resonance=env.harmonic_acoustic_energy,
                    gentle_mechanosensory_touch=gentle_touch,
                    physical_exertion_level=actual_mech_work,
                    brain_atp_fraction=self.metab.brain_atp,
                ),
                dt=h,
            )
            total_stress_vol += stress_vol_rate * h

            # 3. Multi-Compartment Cellular Bioenergetics (Brain, Muscle, Gut/Liver)
            metab_out = self.metab_engine.step(
                self.metab,
                MetabolicInputs(
                    ingested_nutrient_rate=env.ingested_nutrient_flux,
                    ingested_water_rate=env.ingested_water_flux,
                    mechanical_power_demand=actual_mech_work,
                    shivering_power_demand=self.thermo.shivering_thermogenesis_cmd,
                    neural_firing_intensity=max(0.02, cmd.neural_firing_load),
                    tissue_repair_anabolic_demand=repair_dem,
                    sweat_water_loss_rate=self.thermo.sweat_evaporation_water_rate,
                    cerebral_flow_lpm=self.cardio.cerebral_flow_lpm,
                    muscle_flow_lpm=self.cardio.muscle_flow_lpm,
                    splanchnic_flow_lpm=self.cardio.splanchnic_flow_lpm,
                    arterial_spo2=self.cardio.arterial_spo2,
                    pa_o2_mmhg=self.cardio.pa_o2_mmhg,
                    sympathetic_tone=self.endo.sympathetic_tone,
                    ventral_vagal_tone=self.endo.ventral_vagal_tone,
                    plasma_epinephrine=self.endo.plasma_epinephrine,
                    adrenal_cortisol=self.endo.adrenal_cortisol,
                    bus_supply_voltage=env.bus_supply_voltage,
                ),
                dt=h,
            )
            self._last_vo2_ml_min = metab_out.vo2_ml_min
            self._last_vco2_ml_min = metab_out.vco2_ml_min
            self._last_metabolic_watts = metab_out.metabolic_heat_watts
            self._last_motor_gain = metab_out.muscle_contractile_gain
            self._last_cerebral_gain = metab_out.cerebral_synaptic_viability

            # 4. Cardiopulmonary Hemodynamics, Perfusion Routing, & Blood Gases
            o2_l, co2_l = self.cardio_engine.step(
                self.cardio,
                CardiopulmonaryInputs(
                    sympathetic_tone=self.endo.sympathetic_tone,
                    ventral_vagal_tone=self.endo.ventral_vagal_tone,
                    dorsal_vagal_tone=self.endo.dorsal_vagal_tone,
                    plasma_epinephrine=self.endo.plasma_epinephrine,
                    plasma_norepinephrine=self.endo.plasma_norepinephrine,
                    beta_receptor_sensitivity=self.endo.beta_receptor_sensitivity,
                    muscle_metabolic_vasodilator=self.metab.muscle_vasodilator_signal,
                    cutaneous_vasodilation_cmd=self.thermo.cutaneous_vasodilation_cmd,
                    cutaneous_vasoconstrict_cmd=self.thermo.cutaneous_vasoconstrict_cmd,
                    whole_body_o2_consumption_ml_min=metab_out.vo2_ml_min,
                    whole_body_co2_production_ml_min=metab_out.vco2_ml_min,
                    blood_lactate_mmoll=self.metab.blood_lactate_mmoll,
                    core_temp_celsius=self.thermo.core_temp_celsius,
                    hydration_fraction=self.metab.hydration_fraction,
                    ambient_o2_kpa=env.ambient_o2_kpa,
                    ambient_co2_kpa=env.ambient_co2_kpa,
                ),
                dt=h,
            )
            total_o2_l += o2_l
            total_co2_l += co2_l

            # 5. Core-Shell Thermoregulation & Evaporative Sweat Physics
            self.thermo_engine.step(
                self.thermo,
                ThermoregulatoryInputs(
                    metabolic_heat_watts=metab_out.metabolic_heat_watts,
                    cutaneous_flow_lpm=self.cardio.cutaneous_flow_lpm,
                    il6_cytokine=self.endo.il6_cytokine,
                    hydration_fraction=self.metab.hydration_fraction,
                    voluntary_venting_effort=cmd.radiator_venting,
                    ambient_air_temp_celsius=env.ambient_air_temp_celsius,
                    ambient_relative_humidity=env.ambient_relative_humidity,
                    relative_air_speed_mps=env.relative_air_speed_mps,
                    ground_conductivity_factor=env.ground_conductivity,
                    ground_temp_celsius=(
                        env.ambient_air_temp_celsius - 1.0
                        if env.ground_temp_celsius is None
                        else env.ground_temp_celsius
                    ),
                    sector_radiative_heat_watts=env.sector_radiative_watts,
                ),
                dt=h,
            )

        self.age_seconds += dt
        self._sync_projected_state_tensor()
        self._check_somatic_viability()

        net_ds_dt = (self.state - prev_state) / dt
        self.last_ds_dt.copy_(net_ds_dt)

        drives = self.compute_free_energy_per_organ(self.state)
        total_fe = float(drives.sum().item())
        drive_vel = (total_fe - prev_fe) / dt

        return SomaDiagnostics(
            state=self.state.clone(),
            ds_dt=net_ds_dt.clone(),
            organ_drives=drives,
            total_free_energy=total_fe,
            drive_velocity=drive_vel,
            urgency_gradient=self.compute_urgency_gradient(self.state),
            effective_motor_gain=float(self.compute_effective_motor_gain().item()),
            cerebral_synaptic_gain=self._last_cerebral_gain if self.is_alive else 0.0,
            core_temp_celsius=self.thermo.core_temp_celsius,
            vitals=self.snapshot_clinical_vitals(),
            o2_consumed_liters=total_o2_l,
            co2_exhaled_liters=total_co2_l,
            stress_volatile_excreted=total_stress_vol,
            is_alive=self.is_alive,
            cause_of_death=self.cause_of_death,
        )
