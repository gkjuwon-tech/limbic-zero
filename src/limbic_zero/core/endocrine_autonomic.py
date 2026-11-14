"""Autonomic Nervous System, Multi-Stage Neuroendocrine Cascades, and GPCR Receptor Desensitization.

Strict Physiological Fidelity (No Cartoon 'Hormone = Emotion' Shortcuts):
    - Tri-branch Autonomic Nervous System:
        * Thoracolumbar Sympathetic Outflow (mobilization, vasoconstriction, glycogenolysis)
        * Myelinated Ventral Vagal Outflow (Nucleus Ambiguus cardiovagal calm, social engagement)
        * Unmyelinated Dorsal Vagal Outflow (extreme hypoxia/hypotensive metabolic conservation freeze)
    - SAM Axis: Adrenal medulla ``plasma_epinephrine`` & spillover ``plasma_norepinephrine``.
    - 3-Stage HPA Cascade: ``hypothalamic_crh`` -> ``pituitary_acth`` -> ``adrenal_cortisol``
      with negative Glucocorticoid Receptor (``gr_sensitivity``) feedback and cumulative
      ``allostatic_load`` wear-and-tear under chronic unremitting stress.
    - Posterior Pituitary ``vasopressin_adh`` driven by OVLT plasma hyperosmolality & hypovolemia.
    - GPCR Beta-Arrestin Receptor Internalization / Desensitization across beta-adrenergic,
      glucocorticoid, mu-opioid, and oxytocinergic receptors—causing natural biological
      habituation to repeated stimuli and receptor blunting under chronic overload.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class EndocrineAutonomicState:
    """Continuous autonomic outflows, circulating systemic hormones, and receptor sensitivities."""

    # Autonomic branches [0.0, 1.0]
    sympathetic_tone: float = 0.18
    ventral_vagal_tone: float = 0.68
    dorsal_vagal_tone: float = 0.02
    pupil_dilation_diameter_norm: float = 0.35  # [0.10 pinpoint miosis, 0.98 maximal mydriasis]

    # Fast SAM catecholamines [0.0, 1.0]
    plasma_epinephrine: float = 0.06
    plasma_norepinephrine: float = 0.10

    # Multi-stage HPA endocrine cascade [0.0, 1.0] (realistic staggered phase lag!)
    hypothalamic_crh: float = 0.10
    pituitary_acth: float = 0.10
    adrenal_cortisol: float = 0.14

    # Osmoregulatory, analgesic, affiliative-vagal, and immune cytokines [0.0, 1.0]
    vasopressin_adh: float = 0.12
    endogenous_opioid: float = 0.08
    oxytocin_systemic: float = 0.22
    il6_cytokine: float = 0.02

    # GPCR Receptor Densities / Sensitivities [0.20, 1.20] (1.0 = baseline normosensitive)
    beta_receptor_sensitivity: float = 1.00
    gr_receptor_sensitivity: float = 1.00
    opioid_receptor_sensitivity: float = 1.00
    oxytocin_receptor_sensitivity: float = 1.00

    # Cumulative McEwen Allostatic Overload index [0.0, 1.0]
    allostatic_load: float = 0.03


@dataclass(frozen=True)
class EndocrineAutonomicInputs:
    """Ascending physiological stressors and physical sensory waves driving brainstem nuclei."""

    ascending_nociception_total: float      # Spinal Lamina I gated A-delta + C pain [0, 2]
    tissue_inflammation_total: float        # Damaged tissue DAMP / cytokine spill rate [0, 2]
    central_co2_suffocation_drive: float    # Medullary CO2/acidosis alarm [0, 1]
    peripheral_hypoxic_drive: float         # Carotid body low-PaO2 alarm [0, 1]
    hypotensive_baro_unload: float          # Drop in carotid MAP baroreceptor stretch [0, 1]
    neuroglycopenic_alarm: float            # Low blood glucose (< 65 mg/dL) alarm [0, 1]
    plasma_osmolality_mosm: float           # Osmolality driving supraoptic ADH [270, 350]
    blood_volume_fraction: float            # Hypovolemia driving ADH & renin-sympathetic [0.55, 1.15]
    rail_brownout_instability: float        # Bioelectric power-rail drop/flicker [0, 1]
    acoustic_roughness_shock: float         # Sharp dissonant acoustic startle [0, 2]
    acoustic_harmonic_resonance: float      # Consonant rhythmic acoustic soothing [0, 2]
    gentle_mechanosensory_touch: float      # Low-velocity C-tactile warm contact [0, 1]
    physical_exertion_level: float          # Central motor command exercise drive [0, 2]
    brain_atp_fraction: float               # Extreme cerebral ATP failure triggers dorsal freeze [0, 1]


class EndocrineAutonomicEngine:
    """Integrates brainstem autonomic reflexes, endocrine axes, and receptor habituation."""

    def step(self, state: EndocrineAutonomicState, inp: EndocrineAutonomicInputs, dt: float) -> float:
        """Advance neuroendocrine-autonomic state by ``dt`` seconds.

        Returns:
            Volatile stress metabolite excretion rate (alarm kairomone sweated into local air).
        """
        # 1. Systemic IL-6 Inflammatory Cytokine (released by damaged/ischemic tissue, modulated by Cortisol!)
        cortisol_anti_inflammatory = 1.0 / (
            1.0 + 1.1 * state.adrenal_cortisol * state.gr_receptor_sensitivity
        )
        il6_prod = 0.24 * inp.tissue_inflammation_total * cortisol_anti_inflammatory
        il6_clear = 0.025 * state.il6_cytokine
        state.il6_cytokine = max(0.0, min(1.0, state.il6_cytokine + (il6_prod - il6_clear) * dt))

        # 2. Oxytocinergic / C-Tactile Vagal Resonance with Receptor Habituation
        affiliative_stimulus = 0.55 * inp.acoustic_harmonic_resonance + 0.65 * inp.gentle_mechanosensory_touch
        # High acute pain or suffocation gates out calm affiliative reception
        acute_emergency_block = 1.0 / (
            1.0 + 2.5 * inp.central_co2_suffocation_drive + 1.8 * inp.ascending_nociception_total
        )
        oxy_release = 0.32 * affiliative_stimulus * acute_emergency_block
        oxy_clear = 0.14 * state.oxytocin_systemic
        state.oxytocin_systemic = max(0.0, min(1.0, state.oxytocin_systemic + (oxy_release - oxy_clear) * dt))

        # Effective oxytocin signal scaled by current receptor density
        eff_oxytocin = state.oxytocin_systemic * state.oxytocin_receptor_sensitivity

        # 3. Rostral Ventrolateral Medulla (RVLM) Sympathetic Drive vs Nucleus Ambiguus Ventral Vagus
        visceral_emergency = (
            1.35 * inp.central_co2_suffocation_drive
            + 1.15 * inp.peripheral_hypoxic_drive
            + 0.95 * inp.hypotensive_baro_unload
            + 1.10 * inp.neuroglycopenic_alarm
            + 0.90 * inp.rail_brownout_instability
            + 0.85 * inp.ascending_nociception_total
            + 0.75 * inp.acoustic_roughness_shock
            + 0.45 * inp.physical_exertion_level
            + 0.25 * state.hypothalamic_crh
        )

        # Ventral vagal myelinated brake is supported by safe oxygenation & oxytocin, withdrawn in emergency
        target_ventral_vagus = (0.72 + 0.35 * eff_oxytocin) / (1.0 + 1.95 * visceral_emergency)
        target_ventral_vagus = max(0.02, min(0.98, target_ventral_vagus))
        state.ventral_vagal_tone += (target_ventral_vagus - state.ventral_vagal_tone) * min(1.0, dt / 0.8)

        target_sympathetic = math.tanh(0.16 + 0.72 * visceral_emergency - 0.28 * eff_oxytocin)
        target_sympathetic = max(0.03, min(0.99, target_sympathetic))
        state.sympathetic_tone += (target_sympathetic - state.sympathetic_tone) * min(1.0, dt / 0.9)

        # Dorsal Motor Vagus unmyelinated shutdown/collapse fires ONLY under extreme cerebral ATP / anoxic crisis
        anoxic_collapse_trigger = max(0.0, 0.42 - inp.brain_atp_fraction) * 2.8 + max(
            0.0, inp.peripheral_hypoxic_drive - 0.75
        ) * 1.5
        target_dorsal_vagus = math.tanh(max(0.0, anoxic_collapse_trigger))
        state.dorsal_vagal_tone += (target_dorsal_vagus - state.dorsal_vagal_tone) * min(1.0, dt / 2.2)

        # Pupil diameter: dilates with sympathetic alpha-1 (radial iris muscle), constricts with vagal/ACh sphincter
        target_pupil = max(
            0.12, min(0.98, 0.38 + 0.62 * state.sympathetic_tone - 0.28 * state.ventral_vagal_tone)
        )
        state.pupil_dilation_diameter_norm += (
            target_pupil - state.pupil_dilation_diameter_norm
        ) * min(1.0, dt / 0.45)

        # 4. SAM Axis Catecholamines (Adrenal Medulla Epinephrine & Sympathetic Spillover Norepinephrine)
        # Epinephrine surges especially strongly during hypoglycemia, hypoxia, terror shock, and brownout!
        epi_secretion = 0.55 * (state.sympathetic_tone ** 1.5) + 0.45 * inp.neuroglycopenic_alarm
        epi_clearance = 0.42 * state.plasma_epinephrine  # Fast ~2 min biological half-life scaled to arena time
        state.plasma_epinephrine = max(
            0.0, min(1.0, state.plasma_epinephrine + (epi_secretion - epi_clearance) * dt)
        )

        ne_spillover = 0.50 * state.sympathetic_tone + 0.20 * inp.physical_exertion_level
        ne_clearance = 0.48 * state.plasma_norepinephrine
        state.plasma_norepinephrine = max(
            0.0, min(1.0, state.plasma_norepinephrine + (ne_spillover - ne_clearance) * dt)
        )

        # 5. Three-Stage HPA Endocrine Axis with Negative Glucocorticoid Feedback
        gr_negative_feedback = 1.0 / (
            1.0 + 2.4 * state.adrenal_cortisol * state.gr_receptor_sensitivity
        )
        crh_drive = (
            0.48 * visceral_emergency + 0.35 * state.il6_cytokine - 0.25 * eff_oxytocin
        ) * gr_negative_feedback
        d_crh = max(0.0, crh_drive) * 0.22 - 0.16 * state.hypothalamic_crh
        state.hypothalamic_crh = max(0.02, min(1.0, state.hypothalamic_crh + d_crh * dt))

        acth_drive = 0.26 * state.hypothalamic_crh * gr_negative_feedback
        d_acth = acth_drive - 0.14 * state.pituitary_acth
        state.pituitary_acth = max(0.02, min(1.0, state.pituitary_acth + d_acth * dt))

        # Cortisol has a noticeably slower rise & clearance (lingering physiological stress tail!)
        d_cortisol = 0.15 * state.pituitary_acth - 0.065 * state.adrenal_cortisol
        state.adrenal_cortisol = max(0.03, min(1.0, state.adrenal_cortisol + d_cortisol * dt))

        # 6. Supraoptic / Paraventricular Vasopressin (ADH) from Plasma Osmolality & Hypovolemia
        osmo_excess = max(0.0, inp.plasma_osmolality_mosm - 286.0) / 22.0
        vol_depletion = max(0.0, 0.94 - inp.blood_volume_fraction) * 3.5
        target_adh = math.tanh(0.10 + 0.85 * osmo_excess + 0.95 * vol_depletion)
        state.vasopressin_adh += (target_adh - state.vasopressin_adh) * min(1.0, dt / 3.0)

        # 7. Periaqueductal Gray (PAG) Endogenous Opioid Release (Stress-Induced Analgesia!)
        # Extreme acute trauma + high sympathetic fight-or-flight transiently releases beta-endorphin/enkephalin
        sia_trigger = max(0.0, inp.ascending_nociception_total - 0.25) * state.sympathetic_tone
        opioid_release = 0.28 * sia_trigger + 0.08 * eff_oxytocin
        opioid_clear = 0.12 * state.endogenous_opioid
        state.endogenous_opioid = max(0.02, min(1.0, state.endogenous_opioid + (opioid_release - opioid_clear) * dt))

        # 8. GPCR Receptor Desensitization / Internalization & Resensitization Kinetics
        # Sustained high agonist occupancy internalizes receptors; low agonist allows recycling to membrane!
        state.beta_receptor_sensitivity = max(
            0.25,
            min(
                1.15,
                state.beta_receptor_sensitivity
                + (0.018 * (1.0 - state.beta_receptor_sensitivity) - 0.045 * max(0.0, state.plasma_epinephrine - 0.45))
                * dt,
            ),
        )
        state.gr_receptor_sensitivity = max(
            0.25,
            min(
                1.15,
                state.gr_receptor_sensitivity
                + (0.012 * (1.0 - state.gr_receptor_sensitivity) - 0.038 * max(0.0, state.adrenal_cortisol - 0.50))
                * dt,
            ),
        )
        state.opioid_receptor_sensitivity = max(
            0.25,
            min(
                1.15,
                state.opioid_receptor_sensitivity
                + (0.015 * (1.0 - state.opioid_receptor_sensitivity) - 0.050 * max(0.0, state.endogenous_opioid - 0.40))
                * dt,
            ),
        )
        state.oxytocin_receptor_sensitivity = max(
            0.25,
            min(
                1.15,
                state.oxytocin_receptor_sensitivity
                + (0.020 * (1.0 - state.oxytocin_receptor_sensitivity) - 0.055 * max(0.0, state.oxytocin_systemic - 0.55))
                * dt,
            ),
        )

        # 9. Cumulative McEwen Allostatic Overload (chronic cortisol + sympathetic + inflammatory wear)
        chronic_strain = (
            max(0.0, state.adrenal_cortisol - 0.38) * 0.035
            + max(0.0, state.sympathetic_tone - 0.55) * 0.025
            + state.il6_cytokine * 0.030
        )
        deep_recovery = 0.008 * state.ventral_vagal_tone * max(0.0, 0.30 - state.adrenal_cortisol)
        state.allostatic_load = max(
            0.0, min(1.0, state.allostatic_load + (chronic_strain - deep_recovery) * dt)
        )

        # Apocrine stress volatile emission rate into local air grid (alarm kairomone!)
        stress_volatile_emission = (
            max(0.0, state.plasma_epinephrine - 0.14) * 1.40
            + max(0.0, state.sympathetic_tone - 0.52) * 0.55
            + max(0.0, state.adrenal_cortisol - 0.25) * 0.50
        )
        return stress_volatile_emission
