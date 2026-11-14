"""Localized 8-Sector Tissue Integrity, Inflammatory Sensitization (Hyperalgesia), and Spinal Gate Control.

Models peripheral somatosensory pathology & pain neurobiology without scripted cheats:
    1. 8 Localized Body Sectors:
       Impacts, rival shoves, or directional thermal burns damage the SPECIFIC body sector
       that was hit rather than a generic single health bar.
    2. Peripheral Inflammatory Soup (Bradykinin / Prostaglandin / Substance P / H+):
       Damaged cells leak DAMP mediators into that sector's interstitial fluid, lowering
       local nociceptor firing thresholds -> **Primary Hyperalgesia & Tactile Allodynia**
       (even a gentle brush on an injured sector triggers C-fiber throbbing pain!).
    3. Melzack-Wall Dorsal Horn Spinal Gate Control + PAG Descending Analgesia:
       - Non-noxious A-beta mechanoreceptor rubbing partially inhibits dorsal horn pain projection.
       - Descending Periaqueductal Gray (PAG) ``endogenous_opioid`` + ``norepinephrine``
         suppresses ascending pain during acute fight-or-flight survival emergencies,
         unmasking delayed soreness once danger subsides!
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


@dataclass
class TissueNociceptionState:
    """Per-sector structural health, local inflammatory soup, and gated pain channels."""

    sector_integrity: List[float] = field(default_factory=lambda: [0.99] * 8)
    sector_inflammation: List[float] = field(default_factory=lambda: [0.0] * 8)

    # Ascending Lamina I Spinothalamic signals per sector [0.0, 1.5]
    a_delta_sharp_pain: List[float] = field(default_factory=lambda: [0.0] * 8)
    c_fiber_burning_ache: List[float] = field(default_factory=lambda: [0.0] * 8)
    a_beta_light_touch: List[float] = field(default_factory=lambda: [0.0] * 8)

    @property
    def mean_integrity(self) -> float:
        return sum(self.sector_integrity) / 8.0

    @property
    def min_sector_integrity(self) -> float:
        return min(self.sector_integrity)


@dataclass(frozen=True)
class TissueNociceptionInputs:
    """Mechanical contact forces, skin temperatures, systemic ischemia, and descending analgesia."""

    sector_mechanical_force: Sequence[float]   # Contact impulse per sector [>= 0]
    sector_skin_temp_celsius: Sequence[float]  # Skin temperature per sector (TRPV1 > 43 C, TRPM8 < 15 C!)
    systemic_hypoxia_severity: float           # Brain/arterial anoxia causing systemic cellular damage [0, 1]
    core_temp_celsius: float                   # Core > 41.5 C causes systemic heatstroke denaturing
    arterial_ph: float                         # Severe acidosis (< 7.05) damages cellular membranes
    anabolic_atp_availability: float           # Muscle/systemic ATP funding tissue repair [0, 1]
    adrenal_cortisol: float                    # High chronic cortisol slows wound healing!
    endogenous_opioid_effective: float         # opioid * receptor_sensitivity gating dorsal horn
    sympathetic_noradrenergic_gate: float      # Acute descending locus coeruleus pain suppression [0, 1]


class TissueNociceptionEngine:
    """Advances 8-sector structural pathology, hyperalgesia, and spinal gate transduction."""

    MECHANICAL_DAMAGE_THRESHOLD: float = 4.0   # Benign touch below 4.0; bruising above 4.0
    TRPV1_HEAT_PAIN_THRESHOLD_C: float = 43.0  # Exact biological TRPV1 capsaicin/heat channel activation!
    TRPM8_COLD_PAIN_THRESHOLD_C: float = 15.0  # Noxious cold nociceptor threshold

    def step(self, state: TissueNociceptionState, inp: TissueNociceptionInputs, dt: float) -> Tuple[float, float, float]:
        """Advance 8-sector tissue pathology & spinal nociception by ``dt`` seconds.

        Returns:
            Tuple ``(total_gated_nociception, total_inflammation_spill, total_repair_demand)``.
        """
        # Systemic metabolic/thermal cellular stress affecting all sectors
        hyperthermic_denature = max(0.0, inp.core_temp_celsius - 41.3) * 0.065
        acidotic_necrosis = max(0.0, 7.02 - inp.arterial_ph) * 0.12
        anoxic_necrosis = (inp.systemic_hypoxia_severity ** 2) * 0.085
        systemic_damage_rate = hyperthermic_denature + acidotic_necrosis + anoxic_necrosis

        # Anabolic wound healing (requires ATP, inhibited by chronic high cortisol!)
        cortisol_healing_penalty = max(0.20, 1.0 - 0.65 * max(0.0, inp.adrenal_cortisol - 0.35))
        base_heal_rate = 0.016 * inp.anabolic_atp_availability * cortisol_healing_penalty

        # Descending PAG endogenous analgesia factor (gates dorsal horn transmission, NOT tissue damage!)
        descending_gate = 1.0 / (
            1.0 + 1.65 * inp.endogenous_opioid_effective + 0.55 * max(0.0, inp.sympathetic_noradrenergic_gate - 0.65)
        )

        total_pain = 0.0
        total_inflam = 0.0
        total_repair = 0.0

        for i in range(8):
            mech = float(inp.sector_mechanical_force[i]) if i < len(inp.sector_mechanical_force) else 0.0
            tskin = float(inp.sector_skin_temp_celsius[i]) if i < len(inp.sector_skin_temp_celsius) else 33.4

            # 1. Noxious mechanical impact & TRPV1 (>43C) / cold (<15C) thermal burn damage
            mech_excess = max(0.0, mech - self.MECHANICAL_DAMAGE_THRESHOLD)
            burn_excess = max(0.0, tskin - self.TRPV1_HEAT_PAIN_THRESHOLD_C)
            freeze_excess = max(0.0, self.TRPM8_COLD_PAIN_THRESHOLD_C - tskin)

            local_trauma_rate = 0.028 * mech_excess + 0.032 * (burn_excess ** 1.3) + 0.015 * freeze_excess
            total_damage_rate = local_trauma_rate + systemic_damage_rate

            # Healing proportional to missing tissue integrity
            missing = 1.0 - state.sector_integrity[i]
            sector_heal = base_heal_rate * missing if total_damage_rate < 0.02 else 0.0
            total_repair += sector_heal

            state.sector_integrity[i] = max(
                0.0, min(1.0, state.sector_integrity[i] + (sector_heal - total_damage_rate) * dt)
            )

            # 2. Local Inflammatory Soup (Bradykinin / PGE2) accumulates from trauma & missing integrity
            inflam_prod = 3.8 * local_trauma_rate + 0.35 * (1.0 - state.sector_integrity[i])
            inflam_clear = 0.05 * state.sector_inflammation[i]
            state.sector_inflammation[i] = max(
                0.0, min(1.0, state.sector_inflammation[i] + (inflam_prod - inflam_clear) * dt)
            )
            inflam = state.sector_inflammation[i]
            total_inflam += inflam

            # 3. Innocuous A-beta Mechanoreceptor Touch (saturates smoothly on gentle contact)
            a_beta = mech / (mech + 1.8) if mech > 1e-5 else 0.0
            state.a_beta_light_touch[i] = a_beta

            # 4. Peripheral Hyperalgesia / Allodynia:
            # Inflammation lowers mechanical pain threshold from 4.0 down toward 0.4!
            effective_mech_thresh = self.MECHANICAL_DAMAGE_THRESHOLD * (1.0 - 0.90 * inflam)
            sensitized_mech_nox = max(0.0, mech - effective_mech_thresh)

            # Fast A-delta sharp nociceptors respond to acute suprathreshold mechanical/thermal insult
            raw_a_delta = 0.22 * mech_excess + 0.25 * burn_excess + 0.18 * freeze_excess

            # Slow C-fiber polymodal ache responds to sensitized contact, thermal burn, and spontaneous inflammatory throbbing!
            raw_c_fiber = (
                0.42 * sensitized_mech_nox
                + 0.30 * burn_excess
                + 0.40 * (inflam ** 1.25)
                + 0.20 * systemic_damage_rate
            )

            # Melzack-Wall Gate Control: gentle A-beta touch (when not noxious) inhibits C-fiber transmission by ~30%
            spinal_a_beta_gate = 1.0 - 0.30 * a_beta * max(0.0, 1.0 - raw_a_delta * 2.0)

            gated_a_delta = min(1.5, raw_a_delta * (0.35 + 0.65 * descending_gate))
            gated_c_fiber = min(1.5, raw_c_fiber * spinal_a_beta_gate * descending_gate)

            state.a_delta_sharp_pain[i] = gated_a_delta
            state.c_fiber_burning_ache[i] = gated_c_fiber
            total_pain += 0.55 * gated_a_delta + 0.45 * gated_c_fiber

        return min(2.5, total_pain), total_inflam / 8.0, min(1.0, total_repair * 8.0)
