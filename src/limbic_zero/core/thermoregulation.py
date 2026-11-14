"""Two-Compartment Core-Shell Thermoregulation, Febrile Pyrogenesis, and Psychrometric Sweating.

Models real thermodynamic heat exchange in Watts & degrees Celsius:
    - Core compartment (viscera, brain, deep muscle) receives endogenous metabolic Joule heat.
    - Cutaneous blood perfusion ``cutaneous_flow_lpm`` convects heat from Core -> 8-sector Skin Shell.
    - Hypothalamic Preoptic Anterior Area (POAH) compares ``core_temp_c`` + ``0.15 * mean_skin_temp``
      against ``hypothalamic_setpoint_c`` (37.0 C baseline, elevated up to ~39.8 C by ``il6_cytokine``
      prostaglandin E2 signaling -> true biological fever!).
    - Effector loops:
        1. Cutaneous active vasodilation vs adrenergic cold vasoconstriction.
        2. Eccrine sweat secretion + wet-bulb evaporative cooling (throttled at high ambient humidity!).
        3. Involuntary skeletal muscle shivering thermogenesis when core drops below POAH setpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass
class ThermoregulatoryState:
    """Core and 8-sector cutaneous skin shell temperatures & POAH effector commands."""

    core_temp_celsius: float = 37.00
    hypothalamic_setpoint_celsius: float = 37.00
    skin_sector_temps_celsius: List[float] = field(default_factory=lambda: [33.4] * 8)

    # Hypothalamic autonomic thermoregulatory effector outputs
    cutaneous_vasodilation_cmd: float = 0.10   # [0, 1]
    cutaneous_vasoconstrict_cmd: float = 0.05  # [0, 1]
    sweat_evaporation_water_rate: float = 0.0  # Normalized hydration loss/sec
    shivering_thermogenesis_cmd: float = 0.0   # [0, 1] recruits extra muscle ATP burn!


@dataclass(frozen=True)
class ThermoregulatoryInputs:
    """Endogenous heat, cutaneous blood flow, cytokine pyrogens, and environmental physics."""

    metabolic_heat_watts: float                # Total internal heat produced by metabolism (~85W rest, >800W sprint)
    cutaneous_flow_lpm: float                  # Blood flow transferring heat from core to skin
    il6_cytokine: float                        # Systemic pyrogen raising POAH setpoint -> fever!
    hydration_fraction: float                  # Severe dehydration impairs sweating (anhidrotic heatstroke!)
    voluntary_venting_effort: float            # Behavioral panting / radiator fin spreading [0, 1]
    ambient_air_temp_celsius: float            # Local cell air temperature
    ambient_relative_humidity: float           # [0.05, 0.99] limits evaporative cooling!
    relative_air_speed_mps: float              # Body speed + local wind speed increasing convection
    ground_conductivity_factor: float          # Terrain conductive coupling (e.g. cold metal plate vs dry soil)
    ground_temp_celsius: float                 # Surface temperature beneath organism
    sector_radiative_heat_watts: Sequence[float]  # Directional radiant heat from hazards/sun on 8 skin sectors


class ThermoregulatoryEngine:
    """Integrates coupled Core <-> 8-Sector Skin <-> Environment Joule heat balance."""

    CORE_HEAT_CAPACITY_J_PER_C: float = 860.0
    SECTOR_SKIN_HEAT_CAPACITY_J_PER_C: float = 18.0
    BLOOD_SPECIFIC_HEAT_W_PER_LPM_C: float = 64.5

    def step(self, state: ThermoregulatoryState, inp: ThermoregulatoryInputs, dt: float) -> None:
        """Advance core & 8-sector skin temperatures by ``dt`` seconds."""
        # 1. Pyrogenic PGE2 shift of POAH hypothalamic setpoint (IL-6 induces true fever!)
        target_setpoint = 37.00 + 3.15 * (max(0.0, inp.il6_cytokine) ** 0.88)
        state.hypothalamic_setpoint_celsius += (
            target_setpoint - state.hypothalamic_setpoint_celsius
        ) * min(1.0, dt / 2.2)

        mean_skin_c = sum(state.skin_sector_temps_celsius) / 8.0
        core_warm_err = state.core_temp_celsius - state.hypothalamic_setpoint_celsius
        poah_warm_error = core_warm_err + 0.12 * (mean_skin_c - 33.4)

        # 2. POAH Effector Commands
        # Active vasodilation responds strongly whenever Core > Setpoint (without cool evaporating sweat clamping it off!)
        if core_warm_err > 0.0 or poah_warm_error > 0.0:
            dilate_drive = max(0.0, core_warm_err) / 0.55 + 0.12 * max(0.0, mean_skin_c - 33.4)
            state.cutaneous_vasodilation_cmd = min(1.0, dilate_drive)
            state.cutaneous_vasoconstrict_cmd = 0.0
            state.shivering_thermogenesis_cmd = 0.0
        else:
            cold_err = -poah_warm_error
            state.cutaneous_vasodilation_cmd = 0.0
            state.cutaneous_vasoconstrict_cmd = min(1.0, cold_err / 1.1)
            state.shivering_thermogenesis_cmd = min(1.0, max(0.0, cold_err - 0.20) / 1.3)

        # 3. Psychrometric Sweat & Respiratory Panting Evaporative Cooling
        sweat_hydration_gate = max(0.05, min(1.0, (inp.hydration_fraction - 0.65) / 0.22))
        skin_warm_reflex = max(0.0, mean_skin_c - 34.0) * 0.22
        autonomic_sweat_drive = max(0.0, core_warm_err - 0.03) / 0.55 + skin_warm_reflex
        total_evap_drive = min(1.5, autonomic_sweat_drive + 0.55 * inp.voluntary_venting_effort) * sweat_hydration_gate

        # High relative humidity blocks latent evaporation even though sweat water is still secreted & lost!
        vapor_gradient_efficiency = max(0.03, 1.0 - 0.96 * max(0.0, min(0.99, inp.ambient_relative_humidity)))
        wind_evap_boost = 1.0 + 0.25 * math.sqrt(max(0.0, inp.relative_air_speed_mps))
        total_evaporative_cooling_watts = 520.0 * total_evap_drive * vapor_gradient_efficiency * wind_evap_boost

        # Secreted sweat depletes body water regardless of whether high humidity prevented evaporation!
        state.sweat_evaporation_water_rate = 0.011 * total_evap_drive

        # 4. Core <-> 8-Sector Skin Blood Perfusion + Conductive Tissue Coupling
        per_sector_flow_lpm = inp.cutaneous_flow_lpm / 8.0
        h_conv_air = 5.2 + 5.5 * math.sqrt(max(0.0, inp.relative_air_speed_mps))
        sector_area = 0.16

        total_core_to_skin_watts = 0.0
        evap_per_sector = total_evaporative_cooling_watts / 8.0
        cond_ground_per_sector = 1.4 * inp.ground_conductivity_factor

        for idx in range(8):
            t_skin = state.skin_sector_temps_celsius[idx]
            # Core -> sector heat flow via cutaneous blood + subcutaneous fat/tissue conduction (0.75 W/C)
            q_core_to_sec = (
                self.BLOOD_SPECIFIC_HEAT_W_PER_LPM_C * per_sector_flow_lpm + 0.75
            ) * (state.core_temp_celsius - t_skin)
            total_core_to_skin_watts += q_core_to_sec

            q_air_conv = h_conv_air * sector_area * (t_skin - inp.ambient_air_temp_celsius)
            q_ground_cond = cond_ground_per_sector * (t_skin - inp.ground_temp_celsius)
            q_rad_in = float(inp.sector_radiative_heat_watts[idx]) if idx < len(inp.sector_radiative_heat_watts) else 0.0

            net_skin_watts = q_core_to_sec + q_rad_in - q_air_conv - q_ground_cond - evap_per_sector
            new_tskin = t_skin + (net_skin_watts / self.SECTOR_SKIN_HEAT_CAPACITY_J_PER_C) * dt
            state.skin_sector_temps_celsius[idx] = max(12.0, min(52.0, new_tskin))

        respiratory_heat_loss_watts = 0.06 * inp.metabolic_heat_watts * max(
            0.05, (state.core_temp_celsius - inp.ambient_air_temp_celsius) / 15.0
        )
        net_core_watts = inp.metabolic_heat_watts - total_core_to_skin_watts - respiratory_heat_loss_watts
        state.core_temp_celsius = max(
            26.0,
            min(45.0, state.core_temp_celsius + (net_core_watts / self.CORE_HEAT_CAPACITY_J_PER_C) * dt),
        )


import math  # noqa: E402
