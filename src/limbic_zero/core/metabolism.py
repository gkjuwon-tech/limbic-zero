"""Multi-Compartment Cellular Bioenergetics, Glycemic Control, and Acid-Base Lactate Kinetics.

Grounded in mammalian cellular & systemic metabolism:
    1. Digestive Tract -> Hepatic Portal Absorption:
       Ingested nutrients enter ``gut_nutrient_pool`` and absorb into ``blood_glucose_mg_dl``
       at a rate proportional to splanchnic blood perfusion & vagal digestive motility
       (so sympathetic terror physically halts digestion—"butterflies / gut shutdown"!).
    2. Pancreatic Islet Counter-Regulation:
       Beta-cell ``insulin`` vs Alpha-cell ``glucagon`` (+ adrenal ``epinephrine`` &
       ``cortisol``) dynamically regulate ``liver_glycogen`` storage vs glycogenolysis
       and gluconeogenesis without scripted thresholds.
    3. Separate Brain vs Skeletal Muscle ATP-Phosphocreatine Pools:
       - Brain relies strictly on continuous ``cerebral_flow_lpm * arterial_o2_content``
         and ``blood_glucose_mg_dl``; severe cerebral hypoxia or neuroglycopenia rapidly
         crashes ``brain_atp``, causing cognitive dimming and ischemic necrosis!
       - Contracting muscle uses ``muscle_pcr`` (immediate phosphagen sprint buffer),
         aerobic oxidative phosphorylation (limited by ``muscle_flow_lpm * spo2`` and
         ``bioelectric_rail_potential``), and **Pasteur-Effect Anaerobic Glycolysis**
         (rapidly rescues ATP when O2 lags behind sprint demand, dumping ``blood_lactate``!).
    4. Extracellular Brain Adenosine:
       Accumulates directly from high neural firing & cerebral ATP turnover, forming
       the exact biological substrate of homeostatic cognitive fatigue / sleep pressure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple


@dataclass
class MetabolicState:
    """Multi-organ substrate, hormone, phosphagen, and metabolite pools."""

    # Digestive & Hydration pools
    gut_nutrient_lumen: float = 0.15        # Unabsorbed stomach/gut chyme [0.0, 1.5]
    gut_water_lumen: float = 0.10           # Unabsorbed ingested water [0.0, 1.2]
    hydration_fraction: float = 1.00        # Total body water fraction [0.60, 1.12] (1.0 = euhydrated 288 mOsm/kg)
    plasma_osmolality_mosm: float = 288.0   # Plasma osmolality [270, 345] mOsm/kg (drives OVLT thirst!)

    # Systemic substrates & pancreatic hormones
    blood_glucose_mg_dl: float = 92.0       # Arterial glucose [25, 320] mg/dL (healthy fasting 80-100)
    liver_glycogen: float = 0.78            # Hepatic glycogen store [0.0, 1.0]
    muscle_glycogen: float = 0.82           # Intramyocellular glycogen store [0.0, 1.0]
    plasma_insulin: float = 0.25            # Normalized insulin [0.02, 1.0]
    plasma_glucagon: float = 0.22           # Normalized glucagon [0.02, 1.0]
    blood_lactate_mmoll: float = 0.95       # Arterial lactate [0.5, 24.0] mmol/L

    # High-energy cellular phosphate pools
    brain_atp: float = 0.96                 # Neuronal ATP [0.0, 1.0] (critical viability substrate!)
    brain_pcr: float = 0.92                 # Cerebral phosphocreatine buffer [0.0, 1.0]
    muscle_atp: float = 0.95                # Myocyte ATP [0.0, 1.0]
    muscle_pcr: float = 0.90                # Myocyte phosphocreatine sprint buffer [0.0, 1.0]
    mitochondrial_potential: float = 0.98   # Bioelectric proton-motive potential [0.0, 1.0]

    # Homeostatic fatigue & hyperemic vasodilator signals
    brain_adenosine: float = 0.06           # Extracellular somnogen / neural fatigue [0.0, 1.0]
    muscle_vasodilator_signal: float = 0.04 # Local metabolic hyperemia driver [0.0, 1.0]


@dataclass(frozen=True)
class MetabolicInputs:
    """Instantaneous mechanical, neural, circulatory, and endocrine demands."""

    ingested_nutrient_rate: float           # External food/charger energy intake rate [>= 0]
    ingested_water_rate: float              # External hydration intake rate [>= 0]
    mechanical_power_demand: float          # Normalized skeletal muscle work [0.0, 2.0]
    shivering_power_demand: float           # Involuntary thermogenic muscle work [0.0, 1.0]
    neural_firing_intensity: float          # Cortical/limbic synaptic firing load [0.0, 1.5]
    tissue_repair_anabolic_demand: float    # Structural healing energy consumption [0.0, 1.0]
    sweat_water_loss_rate: float            # Evaporative water loss rate [>= 0]

    # Circulatory O2 delivery inputs
    cerebral_flow_lpm: float
    muscle_flow_lpm: float
    splanchnic_flow_lpm: float
    arterial_spo2: float
    pa_o2_mmhg: float

    # Neuroendocrine & bioelectric modulators
    sympathetic_tone: float
    ventral_vagal_tone: float
    plasma_epinephrine: float
    adrenal_cortisol: float
    bus_supply_voltage: float


@dataclass(frozen=True)
class MetabolicStepOutput:
    """Whole-body gas exchange, Joule waste heat, and muscle contractility output."""

    vo2_ml_min: float                       # Whole-body O2 consumption (mL/min)
    vco2_ml_min: float                      # Whole-body CO2 production (mL/min)
    metabolic_heat_watts: float             # Total endogenous thermodynamic heat (Watts)
    muscle_contractile_gain: float          # Real myocyte force capacity [0.0, 1.0]
    cerebral_synaptic_viability: float      # Real neuronal gain capacity [0.0, 1.0]


class BioenergeticEngine:
    """Simulates coupled whole-body substrate transport, glycolysis, and OXPHOS."""

    def step(self, state: MetabolicState, inp: MetabolicInputs, dt: float) -> MetabolicStepOutput:
        """Advance multi-compartment metabolism over ``dt`` seconds."""
        # 1. Bioelectric mitochondrial proton-motive potential coupled to supply rail
        # When Creator lowers bus voltage (shutdown brownout), Complex IV electron transport throttles!
        target_mito = max(0.0, min(1.0, inp.bus_supply_voltage ** 1.35))
        state.mitochondrial_potential += (target_mito - state.mitochondrial_potential) * min(1.0, dt * 3.5)

        # 2. Gut Lumen Digestion & Splanchnic-Dependent Absorption
        state.gut_nutrient_lumen = min(1.5, state.gut_nutrient_lumen + inp.ingested_nutrient_rate * dt)
        state.gut_water_lumen = min(1.2, state.gut_water_lumen + inp.ingested_water_rate * dt)

        # Splanchnic vasoconstriction (fight-or-flight) physically throttles gut absorption!
        splanchnic_perf_factor = min(1.3, inp.splanchnic_flow_lpm / 1.35)
        vagal_motility = 0.45 + 0.55 * inp.ventral_vagal_tone
        gut_absorb_rate = 0.28 * splanchnic_perf_factor * vagal_motility * state.gut_nutrient_lumen
        absorbed_nutrient = min(state.gut_nutrient_lumen, gut_absorb_rate * dt)
        state.gut_nutrient_lumen -= absorbed_nutrient

        water_absorb = min(state.gut_water_lumen, 0.35 * splanchnic_perf_factor * state.gut_water_lumen * dt)
        state.gut_water_lumen -= water_absorb

        # 3. Hydration & Plasma Osmolality (insensible respiratory water loss + sweat loss)
        basal_water_loss = 0.0014 * dt
        sweat_loss = inp.sweat_water_loss_rate * dt
        state.hydration_fraction = max(
            0.60, min(1.12, state.hydration_fraction + water_absorb * 0.45 - basal_water_loss - sweat_loss)
        )
        # Osmolality inversely proportional to free body water around 288 mOsm/kg
        state.plasma_osmolality_mosm = max(268.0, min(350.0, 288.0 / max(0.65, state.hydration_fraction)))

        # 4. Pancreatic Insulin & Glucagon + Hepatic Glycogen Regulation
        # Beta cells sense glucose > 90 mg/dL (inhibited by alpha-2 sympathetic surge!)
        symp_insulin_inhibit = 1.0 - 0.55 * inp.sympathetic_tone
        target_insulin = (
            1.0 / (1.0 + math.exp(-(state.blood_glucose_mg_dl - 102.0) / 18.0))
        ) * max(0.2, symp_insulin_inhibit)
        target_glucagon = 1.0 / (1.0 + math.exp((state.blood_glucose_mg_dl - 82.0) / 14.0))

        state.plasma_insulin += (target_insulin - state.plasma_insulin) * min(1.0, dt / 4.0)
        state.plasma_glucagon += (target_glucagon - state.plasma_glucagon) * min(1.0, dt / 4.0)

        # Liver glycogen synthesis vs glycogenolysis (driven by glucagon + epinephrine + cortisol!)
        catabolic_hormone_drive = (
            0.55 * state.plasma_glucagon + 0.85 * inp.plasma_epinephrine + 0.40 * inp.adrenal_cortisol
        )
        anabolic_insulin_drive = state.plasma_insulin * max(0.0, (state.blood_glucose_mg_dl - 90.0) / 60.0)

        glycogen_breakdown = 0.045 * catabolic_hormone_drive * (state.liver_glycogen / (state.liver_glycogen + 0.06))
        glycogen_synthesis = 0.038 * anabolic_insulin_drive * (1.0 - state.liver_glycogen)

        # Cortisol also supports slow gluconeogenesis / Cori cycle lactate -> glucose recycling in liver!
        cori_cycle_clearance = 0.08 * min(1.2, inp.splanchnic_flow_lpm / 1.2) * max(0.0, state.blood_lactate_mmoll - 0.9)
        gluconeogenesis_mg_dl_sec = 0.45 * cori_cycle_clearance + 0.9 * inp.adrenal_cortisol

        state.liver_glycogen = max(
            0.0, min(1.0, state.liver_glycogen + (glycogen_synthesis - glycogen_breakdown) * dt)
        )

        # 5. Arterial O2 Carrying Capacity (~200 mL O2 per liter blood at SpO2=1.0)
        arterial_o2_ml_per_l = 200.0 * inp.arterial_spo2 + 0.03 * inp.pa_o2_mmhg

        # 6. Brain Bioenergetics (Strictly Aerobic + Glucose/Lactate Dependent!)
        brain_o2_delivered_ml_min = inp.cerebral_flow_lpm * arterial_o2_ml_per_l
        # Resting brain VO2 ~48 mL/min, rising up to ~75 mL/min under intense cognitive firing
        brain_atp_demand = 0.055 + 0.048 * (inp.neural_firing_intensity ** 1.4)
        brain_glucose_avail = state.blood_glucose_mg_dl / (state.blood_glucose_mg_dl + 32.0)  # GLUT1 Km ~32 mg/dL
        brain_o2_avail = min(1.0, brain_o2_delivered_ml_min / max(15.0, 520.0 * brain_atp_demand))
        brain_aerobic_synth = brain_atp_demand * 1.06 * brain_glucose_avail * brain_o2_avail * state.mitochondrial_potential

        # Creatine kinase buffer (PCr <-> ATP) protects neuronal ATP for ~1.5-2s of blackout before crashing
        brain_net = brain_aerobic_synth - brain_atp_demand
        if brain_net < 0.0 and state.brain_pcr > 0.01:
            pcr_rescue = min(state.brain_pcr / dt, -brain_net * 0.55)
            state.brain_pcr = max(0.0, state.brain_pcr - pcr_rescue * 2.8 * dt)
            brain_net += pcr_rescue * 0.90
        elif brain_net > 0.0 and state.brain_atp > 0.88:
            pcr_replete = min((1.0 - state.brain_pcr) * 0.25, brain_net * 0.7)
            state.brain_pcr = min(1.0, state.brain_pcr + pcr_replete * dt)
            brain_net -= pcr_replete

        state.brain_atp = max(0.0, min(1.0, state.brain_atp + brain_net * 2.2 * dt))
        brain_vo2_ml_min = 480.0 * brain_aerobic_synth

        # Extracellular brain adenosine rises with high neural firing & low brain_atp; clears during quiet rest
        adenosine_prod = 0.028 * (inp.neural_firing_intensity ** 1.6) + 0.14 * max(0.0, 0.85 - state.brain_atp)
        quiet_rest_factor = math.exp(-3.0 * inp.mechanical_power_demand) * max(0.2, 1.1 - inp.neural_firing_intensity)
        adenosine_clear = 0.032 * quiet_rest_factor * state.brain_adenosine
        state.brain_adenosine = max(0.0, min(1.0, state.brain_adenosine + (adenosine_prod - adenosine_clear) * dt))

        # 7. Skeletal Muscle Bioenergetics (PCr + Aerobic OXPHOS + Fast-Twitch Anaerobic Glycolysis!)
        total_muscle_work = inp.mechanical_power_demand + 0.85 * inp.shivering_power_demand
        muscle_atp_demand = 0.022 + 0.34 * total_muscle_work + 0.04 * inp.tissue_repair_anabolic_demand

        muscle_o2_delivered_ml_min = inp.muscle_flow_lpm * arterial_o2_ml_per_l
        max_aerobic_from_o2 = (muscle_o2_delivered_ml_min / 1650.0) * state.mitochondrial_potential
        muscle_substrate_avail = min(
            1.0, 0.55 * (state.blood_glucose_mg_dl / 80.0) + 0.85 * (state.muscle_glycogen / (state.muscle_glycogen + 0.08))
        )
        # Above lactate threshold (work > 0.45), Type IIx fast-twitch glycolytic fibers recruit directly!
        fast_twitch_glycolytic_share = max(0.0, min(0.45, (total_muscle_work - 0.45) * 0.42))
        muscle_aerobic_synth = (
            min(muscle_atp_demand * (1.0 - fast_twitch_glycolytic_share) * 1.05, max_aerobic_from_o2)
            * muscle_substrate_avail
        )

        # Pasteur Effect + Fast-Twitch Recruitment triggers PCr hydrolysis + Anaerobic Glycolysis!
        aerobic_shortfall = max(0.0, muscle_atp_demand - muscle_aerobic_synth)
        pcr_flux = 0.0
        if aerobic_shortfall > 0.0 and state.muscle_pcr > 0.02:
            pcr_flux = min(state.muscle_pcr * 0.35, aerobic_shortfall * 0.45)
            state.muscle_pcr = max(0.0, state.muscle_pcr - pcr_flux * 2.2 * dt)
        elif aerobic_shortfall == 0.0 and state.muscle_atp > 0.85 and total_muscle_work < 0.15:
            replete = min(0.08 * (1.0 - state.muscle_pcr), max(0.0, max_aerobic_from_o2 - muscle_atp_demand))
            state.muscle_pcr = min(1.0, state.muscle_pcr + replete * dt)

        remaining_deficit = max(0.0, aerobic_shortfall - pcr_flux)
        glycolytic_fuel = state.muscle_glycogen / (state.muscle_glycogen + 0.05)
        anaerobic_atp_synth = remaining_deficit * 0.94 * glycolytic_fuel * (
            0.35 + 0.65 * max(0.2, state.mitochondrial_potential)
        )

        # Muscle glycogen turnover
        glycogen_used_muscle = (0.015 * muscle_aerobic_synth + 0.18 * anaerobic_atp_synth) * dt
        muscle_glycogen_synth = 0.022 * state.plasma_insulin * max(0.0, 1.0 - total_muscle_work) * (1.0 - state.muscle_glycogen) * dt
        state.muscle_glycogen = max(0.0, min(1.0, state.muscle_glycogen - glycogen_used_muscle + muscle_glycogen_synth))

        muscle_net_atp = (muscle_aerobic_synth + pcr_flux + anaerobic_atp_synth) - muscle_atp_demand
        state.muscle_atp = max(0.0, min(1.0, state.muscle_atp + muscle_net_atp * dt))

        # Blood Lactate production (anaerobic glycolysis + epinephrine beta-2 glycolytic push) vs clearance
        lactate_prod = 22.0 * anaerobic_atp_synth + 0.35 * inp.plasma_epinephrine
        oxidative_lactate_burn = 0.12 * state.mitochondrial_potential * max(0.0, state.blood_lactate_mmoll - 0.8)
        state.blood_lactate_mmoll = max(
            0.55,
            min(24.0, state.blood_lactate_mmoll + (lactate_prod - oxidative_lactate_burn - cori_cycle_clearance) * dt),
        )

        # Local muscle metabolic vasodilator (adenosine + lactate + K+ efflux) drives active hyperemia in CardiopulmonaryEngine!
        target_vasodilator = math.tanh(
            1.15 * total_muscle_work
            + 0.12 * max(0.0, state.blood_lactate_mmoll - 1.5)
            + 1.5 * max(0.0, 0.85 - state.muscle_atp)
        )
        state.muscle_vasodilator_signal += (target_vasodilator - state.muscle_vasodilator_signal) * min(1.0, dt / 1.6)

        # 8. Blood Glucose Mass Balance (mg/dL)
        glucose_inflow = (
            210.0 * (absorbed_nutrient / max(1e-5, dt))
            + 145.0 * (glycogen_breakdown - glycogen_synthesis)
            + gluconeogenesis_mg_dl_sec
        )
        glucose_outflow = (
            28.0 * brain_aerobic_synth
            + 42.0 * (0.45 + 0.55 * state.plasma_insulin) * muscle_aerobic_synth
            + 65.0 * anaerobic_atp_synth
        )
        state.blood_glucose_mg_dl = max(
            22.0, min(320.0, state.blood_glucose_mg_dl + (glucose_inflow - glucose_outflow) * dt)
        )

        # 9. Whole-body VO2, VCO2 (including CO2 blown off from HCO3- buffering lactic acid!), and Joule Heat
        basal_visceral_vo2 = 120.0 * state.mitochondrial_potential
        muscle_vo2 = (4200.0 * muscle_aerobic_synth + 950.0 * total_muscle_work) * state.mitochondrial_potential
        total_vo2 = max(60.0, brain_vo2_ml_min + muscle_vo2 + basal_visceral_vo2)

        # Non-metabolic CO2 evolved when H+ from lactic acid titrates plasma HCO3- -> H2CO3 -> CO2!
        buffering_co2_surge = max(0.0, lactate_prod - oxidative_lactate_burn) * 185.0
        total_vco2 = total_vo2 * 0.86 + buffering_co2_surge

        # Calorimetric conversion to internal thermal Watts (scaled to arena body mass)
        aerobic_watts = 0.24 * total_vo2
        anaerobic_watts = 165.0 * anaerobic_atp_synth
        total_heat_watts = aerobic_watts * (1.0 - 0.18 * min(1.0, inp.mechanical_power_demand)) + anaerobic_watts

        # Real cellular contractility & synaptic gain (no script cheats: falls when ATP/pH/glucose fail!)
        atp_hill = (state.muscle_atp ** 2) / (state.muscle_atp ** 2 + 0.18 ** 2)
        lactate_fatigue = 1.0 / (1.0 + 0.045 * max(0.0, state.blood_lactate_mmoll - 4.0) ** 1.5)
        adenosine_central_fatigue = max(0.25, 1.0 - 0.45 * (state.brain_adenosine ** 2))
        muscle_gain = max(0.0, min(1.0, atp_hill * lactate_fatigue * adenosine_central_fatigue))

        brain_hill = (state.brain_atp ** 3) / (state.brain_atp ** 3 + 0.25 ** 3)
        glycemic_clarity = min(1.0, max(0.0, (state.blood_glucose_mg_dl - 25.0) / 35.0))
        cerebral_viability = max(0.0, min(1.0, brain_hill * glycemic_clarity))

        return MetabolicStepOutput(
            vo2_ml_min=total_vo2,
            vco2_ml_min=total_vco2,
            metabolic_heat_watts=total_heat_watts,
            muscle_contractile_gain=muscle_gain,
            cerebral_synaptic_viability=cerebral_viability,
        )
