"""Subcortical Brainstem & Basal Forebrain Neuromodulatory Nuclei + Receptor Subtype Kinetics.

Zero Cartoon Shortcuts ('NO dopamine == joy'):
    Models four anatomically distinct subcortical projection nuclei + lateral habenula (LHb),
    presynaptic autoreceptor negative feedback, reuptake transporters (DAT, NET, SERT) &
    acetylcholinesterase (AChE) clearance, postsynaptic receptor subtype affinities, and
    agonist-induced GPCR desensitization:

    1. VTA / SNc + Lateral Habenula (Dopamine ``DA``):
       - Driven by internal Allostatic Temporal-Difference Error ``delta_TD``:
         ``r_intero(t) + gamma * V(z_t) - V(z_{t-dt})`` minus LHb aversive disappointment!
       - High-affinity ``D2`` receptors (Gi, baseline-occupied, sensitive to LHb pauses!) vs
         low-affinity ``D1`` receptors (Gs, recruited by phasic VTA bursts -> Go facilitation & LTP).
    2. Locus Coeruleus (Central Norepinephrine ``NE``):
       - Driven by Unexpected Uncertainty (world-model surprise jumps), spinothalamic nociception,
         medullary CO2 air-hunger, and vagal peripheral epinephrine signaling.
       - High-affinity ``alpha2A`` receptors (stabilize prefrontal working-memory attractors at
         moderate arousal -> Yerkes-Dodson peak!) vs low-affinity ``beta_alpha1`` receptors
         (engaged in panic/rage storms -> reset prefrontal attractors & hand control to amygdala/PAG!).
    3. Dorsal Raphe Nucleus (Central Serotonin ``5-HT``):
       - Driven by homeostatic safety/energy reserves & ventral-vagal tone; suppressed by chronic
         LHb punishment & glucocorticoid (cortisol) allostatic load.
       - ``5-HT1A`` (Gi: stabilizes temporal horizon ``gamma`` & restrains PAG impulsive aggression)
         vs ``5-HT2A`` (Gq: increases neural entropy & cognitive flexibility under extreme crisis).
    4. Nucleus Basalis of Meynert (Central Acetylcholine ``ACh``):
       - Driven by Expected Uncertainty / Epistemic Novelty (safe-enough localized prediction errors,
         e.g., investigating chaotic mystery artifacts).
       - ``nAChR`` (boosts bottom-up thalamic sensory precision) & ``M1`` (blocks K+ adaptation
         to sustain exploratory focus and gate cortical/hippocampal synaptic plasticity).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class NeuromodulatorState:
    """Extracellular volume-transmission concentrations, firing rates, & receptor occupancies."""

    # Subcortical nuclei normalized firing rates [0, 2.0] (0.35 = tonic pacemaker baseline)
    vta_firing_rate: float = 0.35
    lhb_aversive_rate: float = 0.12         # Lateral Habenula ("anti-reward" disappointment nucleus)
    lc_firing_rate: float = 0.32
    drn_firing_rate: float = 0.45
    nbm_firing_rate: float = 0.30

    # Extracellular volume concentrations [0.0, 1.5]
    dopamine_extracellular: float = 0.28
    norepinephrine_extracellular: float = 0.24
    serotonin_extracellular: float = 0.42
    acetylcholine_extracellular: float = 0.25

    # Postsynaptic receptor subtype occupancies [0.0, 1.0] (Hill-Langmuir binding * sensitivity)
    d1_occupancy: float = 0.24              # Low-affinity Gs (phasic burst detector)
    d2_occupancy: float = 0.58              # High-affinity Gi (tonic + dip detector)
    ne_alpha2a_occupancy: float = 0.48      # High-affinity prefrontal stabilizer
    ne_beta_alpha1_occupancy: float = 0.14  # Low-affinity panic/fight-flight network resetter
    ht1a_occupancy: float = 0.52            # Impulse restraint & long temporal discounting
    ht2a_occupancy: float = 0.16            # Crisis flexibility & gain
    ach_nicotinic_occupancy: float = 0.28   # Bottom-up thalamic sensory precision
    ach_muscarinic_m1_occupancy: float = 0.26  # K+ adaptation block & STDP learning gate
    adenosine_a1_occupancy: float = 0.10    # Sleep pressure & presynaptic brake

    # Slow GPCR receptor membrane sensitivities (internalization / habituation)
    d1_sensitivity: float = 1.0
    ne_beta_sensitivity: float = 1.0
    ht1a_sensitivity: float = 1.0
    ach_m1_sensitivity: float = 1.0

    # Emergent computational signals (read out by Basal Ganglia, Pallium, & Hippocampus)
    td_prediction_error: float = 0.0        # Internal homeostatic TD error delta_TD
    value_estimate: float = 0.0             # Critic expected allostatic value V(z)
    effective_discount_gamma: float = 0.95  # 5-HT1A modulated future horizon!
    consolidation_salience_tag: float = 0.0 # Synaptic tag-and-capture strength for memory!


@dataclass(frozen=True)
class NeuromodulatorDriveInputs:
    """Purely endogenous visceral, world-model, and somatic signals feeding brainstem nuclei."""

    homeostatic_valence: float              # -dF/dt from InteroceptiveNerveBundle
    total_free_energy: float                # F(s(t)) >= 0
    somatic_pain_salience: float            # [0, 1] spinothalamic pain
    suffocation_salience: float             # [0, 1] medullary CO2/O2 alarm
    rail_ripple_intensity: float            # >= 0 existential brownout flicker
    unexpected_world_surprise: float        # >= 0 sudden jump in world-model error (Yu & Dayan)
    epistemic_novelty: float                # >= 0 sustained learnable sensory prediction error
    conditioned_threat_cea: float           # [0, 1] Central Amygdala learned fear/rage output
    conditioned_appetitive_bla: float       # [0, 1] Basolateral Amygdala learned cue value
    peripheral_epinephrine: float           # Vagal NTS transduction of systemic adrenaline
    systemic_cortisol: float                # Blood-brain-barrier glucocorticoid level
    systemic_oxytocin: float                # Central/systemic affiliative peptide
    brain_adenosine: float                  # Extracellular cerebral adenosine [0, 1]
    ventral_vagal_tone: float               # [0, 1] autonomic safety state


class SubcorticalNeuromodulatorEngine(nn.Module):
    """Integrates VTA/LHb, LC, DRN, & NBM nuclei + Value Critic + Receptor Pharmacology."""

    def __init__(
        self,
        latent_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.dtype = dtype
        self._device = torch.device(device)

        # Ventral Striatum / Ventromedial Critic estimating expected homeostatic value V(z)
        self.value_net = nn.Sequential(
            nn.Linear(latent_dim, 48, dtype=dtype, device=self._device),
            nn.Tanh(),
            nn.Linear(48, 1, dtype=dtype, device=self._device),
        )
        # Initialize value head gently near 0
        nn.init.normal_(self.value_net[2].weight, mean=0.0, std=0.04)
        nn.init.zeros_(self.value_net[2].bias)

        self.state = NeuromodulatorState()
        self.register_buffer("prev_value", torch.tensor(0.0, dtype=dtype, device=self._device))
        self.register_buffer("value_eligibility", torch.zeros(latent_dim, dtype=dtype, device=self._device))
        self._has_prev_value: bool = False

    def reset_state(self) -> None:
        self.state = NeuromodulatorState()
        self.prev_value.zero_()
        self.value_eligibility.zero_()
        self._has_prev_value = False

    @staticmethod
    def _hill_binding(conc: float, kd: float, hill_n: float = 1.4, sensitivity: float = 1.0) -> float:
        """Biophysical Hill-Langmuir receptor occupancy ``[L]^n / (Kd^n + [L]^n)``."""
        c = max(0.0, conc)
        if c <= 1e-8:
            return 0.0
        cn = c ** hill_n
        raw_occ = cn / (kd ** hill_n + cn)
        return max(0.0, min(1.0, raw_occ * sensitivity))

    def step(
        self,
        mind_latent: torch.Tensor,
        inp: NeuromodulatorDriveInputs,
        dt: float,
        enable_critic_learning: bool = True,
    ) -> Tuple[NeuromodulatorState, torch.Tensor]:
        """Advance brainstem nuclei, neurotransmitter clearance, and receptor occupancies.

        Returns:
            ``(state, value_tensor)`` where ``value_tensor`` retains autograd graph for
            optional policy/value optimization while online TD updates run continuously.
        """
        st = self.state

        # 1. Serotonin 5-HT1A sets temporal discounting horizon gamma in [0.72, 0.985]
        # Low 5-HT -> myopic impulsivity & panic; Healthy 5-HT -> patient long-horizon planning!
        st.effective_discount_gamma = 0.72 + 0.265 * st.ht1a_occupancy

        # 2. Endogenous Allostatic Reward & Lateral Habenula Disappointment -> VTA TD Error
        v_pred_tensor = self.value_net(mind_latent).squeeze(-1)
        v_now = float(v_pred_tensor.detach().item())

        # Purely physiological primary reinforcement: relief (-dF/dt) minus chronic acute misery penalty
        r_intero = (
            1.35 * inp.homeostatic_valence
            - 0.08 * min(10.0, inp.total_free_energy)
            - 0.65 * inp.somatic_pain_salience
            - 0.75 * inp.suffocation_salience
            - 2.20 * inp.rail_ripple_intensity
            + 0.25 * inp.conditioned_appetitive_bla
        )

        if not self._has_prev_value:
            td_err = r_intero
            self._has_prev_value = True
        else:
            # Continuous-time Doya (2000) step-invariant TD error rate: r(t) + (gamma^dt V(t) - V(t-dt)) / dt_ref
            discount = st.effective_discount_gamma ** max(0.01, dt)
            value_deriv = (discount * v_now - float(self.prev_value.item())) / max(0.04, dt)
            td_err = r_intero + 0.25 * max(-4.0, min(4.0, value_deriv))

        td_err_clamped = max(-2.5, min(2.5, td_err))
        st.td_prediction_error = td_err_clamped
        st.value_estimate = v_now

        # Online local TD(0) update of the subcortical Ventral-Striatal Critic (no external labels!)
        if enable_critic_learning:
            with torch.no_grad():
                z_det = mind_latent.detach()
                h = torch.tanh(
                    torch.mv(self.value_net[0].weight, z_det) + self.value_net[0].bias
                )
                lr_v = 0.020 * min(1.0, dt * 8.0)
                self.value_net[2].weight.data.add_(lr_v * td_err_clamped * h.unsqueeze(0))
                self.value_net[2].bias.data.add_(lr_v * td_err_clamped)
        self.prev_value.fill_(v_now)

        # 3. Lateral Habenula (LHb) & VTA/SNc Dopamine Firing + DAT Reuptake
        # LHb bursts on negative TD disappointment, acute pain, brownout, or conditioned threat!
        target_lhb = (
            0.10
            + 0.85 * max(0.0, -td_err_clamped)
            + 0.55 * inp.somatic_pain_salience
            + 0.60 * inp.suffocation_salience
            + 1.40 * inp.rail_ripple_intensity
            + 0.35 * inp.conditioned_threat_cea
        )
        st.lhb_aversive_rate = max(0.02, min(2.0, st.lhb_aversive_rate + (target_lhb - st.lhb_aversive_rate) * min(1.0, dt / 0.12)))

        # Presynaptic D2 autoreceptor brake stabilizes tonic VTA firing around ~0.35
        d2_auto_brake = 0.26 * st.d2_occupancy
        vta_exc = 0.42 + 0.72 * max(0.0, td_err_clamped) + 0.30 * inp.conditioned_appetitive_bla + 0.12 * inp.epistemic_novelty
        vta_inh = 0.65 * st.lhb_aversive_rate + d2_auto_brake
        target_vta = max(0.01, min(1.95, vta_exc - vta_inh))
        st.vta_firing_rate += (target_vta - st.vta_firing_rate) * min(1.0, dt / 0.09)

        # Extracellular DA mass balance: VTA vesicular release - Michaelis-Menten DAT reuptake
        da_release = 0.95 * st.vta_firing_rate
        dat_reuptake = (1.10 * st.dopamine_extracellular) / (0.45 + st.dopamine_extracellular)
        st.dopamine_extracellular = max(
            0.01, min(1.5, st.dopamine_extracellular + (da_release - dat_reuptake) * min(1.0, dt * 4.5))
        )

        # 4. Locus Coeruleus (LC) Norepinephrine + NET Reuptake
        # Driven by Yu & Dayan Unexpected Uncertainty + visceral emergencies + vagal adrenaline!
        alpha2_auto_brake = 0.22 * st.ne_alpha2a_occupancy
        lc_drive = (
            0.34
            + 1.15 * inp.unexpected_world_surprise
            + 0.85 * inp.somatic_pain_salience
            + 1.10 * inp.suffocation_salience
            + 2.40 * inp.rail_ripple_intensity
            + 0.65 * inp.conditioned_threat_cea
            + 0.45 * inp.peripheral_epinephrine
            - alpha2_auto_brake
            - 0.12 * inp.ventral_vagal_tone
        )
        target_lc = max(0.04, min(2.0, lc_drive))
        st.lc_firing_rate += (target_lc - st.lc_firing_rate) * min(1.0, dt / 0.10)

        ne_release = 0.88 * st.lc_firing_rate
        net_reuptake = (1.05 * st.norepinephrine_extracellular) / (0.48 + st.norepinephrine_extracellular)
        st.norepinephrine_extracellular = max(
            0.02, min(1.5, st.norepinephrine_extracellular + (ne_release - net_reuptake) * min(1.0, dt * 3.8))
        )

        # 5. Dorsal Raphe Nucleus (DRN) Serotonin + SERT Reuptake
        # Sustained by physiological safety & vagal/oxytocin calm; eroded by chronic LHb misery & cortisol!
        ht1a_auto_brake = 0.22 * st.ht1a_occupancy
        safety_wellbeing = max(0.0, 1.0 - 0.18 * inp.total_free_energy) * (0.55 + 0.45 * inp.ventral_vagal_tone)
        drn_drive = (
            0.22
            + 0.58 * safety_wellbeing
            + 0.28 * inp.systemic_oxytocin
            - 0.38 * st.lhb_aversive_rate
            - 0.35 * max(0.0, inp.systemic_cortisol - 0.30)
            - 0.45 * inp.rail_ripple_intensity
            - ht1a_auto_brake
        )
        target_drn = max(0.03, min(1.6, drn_drive))
        st.drn_firing_rate += (target_drn - st.drn_firing_rate) * min(1.0, dt / 0.65)

        ht_release = 0.82 * st.drn_firing_rate
        sert_vmax = 0.85 * (1.0 + 0.45 * max(0.0, inp.systemic_cortisol - 0.35))
        sert_reuptake = (sert_vmax * st.serotonin_extracellular) / (0.50 + st.serotonin_extracellular)
        st.serotonin_extracellular = max(
            0.02, min(1.4, st.serotonin_extracellular + (ht_release - sert_reuptake) * min(1.0, dt * 1.8))
        )

        # 6. Nucleus Basalis of Meynert (NBM) Acetylcholine + Fast AChE Hydrolysis
        nbm_drive = (
            0.18
            + 1.15 * inp.epistemic_novelty
            + 0.45 * inp.unexpected_world_surprise
            + 0.25 * inp.conditioned_appetitive_bla
            - 0.55 * inp.brain_adenosine
        )
        target_nbm = max(0.03, min(1.85, nbm_drive))
        st.nbm_firing_rate += (target_nbm - st.nbm_firing_rate) * min(1.0, dt / 0.14)

        ach_release = 0.92 * st.nbm_firing_rate
        ache_hydrolysis = 1.65 * st.acetylcholine_extracellular
        st.acetylcholine_extracellular = max(
            0.02, min(1.5, st.acetylcholine_extracellular + (ach_release - ache_hydrolysis) * min(1.0, dt * 5.0))
        )

        # 7. Postsynaptic Receptor Subtype Binding Affinities (Kd creates opposing functional regimes!)
        st.d2_occupancy = self._hill_binding(st.dopamine_extracellular, kd=0.20, hill_n=1.5)
        st.d1_occupancy = self._hill_binding(
            st.dopamine_extracellular, kd=0.46, hill_n=1.8, sensitivity=st.d1_sensitivity
        )

        # Arnsten (2009) Yerkes-Dodson Prefrontal Switch:
        # Moderate NE engages high-affinity alpha2A; extreme NE storm recruits PKC/cAMP alpha1/beta which shuts down alpha2A efficacy!
        raw_a2a = self._hill_binding(st.norepinephrine_extracellular, kd=0.20, hill_n=1.4)
        st.ne_beta_alpha1_occupancy = self._hill_binding(
            st.norepinephrine_extracellular, kd=0.58, hill_n=2.2, sensitivity=st.ne_beta_sensitivity
        )
        st.ne_alpha2a_occupancy = max(0.04, raw_a2a * (1.0 - 0.82 * (st.ne_beta_alpha1_occupancy ** 1.3)))

        # 5-HT: 5-HT1A high-affinity (Kd=0.30 -> patience/restraint) vs 5-HT2A low-affinity + crisis-recruited (Kd=0.68)
        st.ht1a_occupancy = self._hill_binding(
            st.serotonin_extracellular, kd=0.30, hill_n=1.4, sensitivity=st.ht1a_sensitivity
        )
        crisis_2a_upreg = 0.18 * st.ne_beta_alpha1_occupancy
        st.ht2a_occupancy = min(
            1.0, self._hill_binding(st.serotonin_extracellular, kd=0.68, hill_n=1.8) + crisis_2a_upreg
        )

        # ACh: Nicotinic (Kd=0.32) & Muscarinic M1 (Kd=0.36)
        st.ach_nicotinic_occupancy = self._hill_binding(st.acetylcholine_extracellular, kd=0.32, hill_n=1.5)
        st.ach_muscarinic_m1_occupancy = self._hill_binding(
            st.acetylcholine_extracellular, kd=0.36, hill_n=1.5, sensitivity=st.ach_m1_sensitivity
        )
        st.adenosine_a1_occupancy = self._hill_binding(inp.brain_adenosine, kd=0.35, hill_n=1.6)

        # 8. Slow GPCR Receptor Desensitization & Membrane Recycling
        st.d1_sensitivity = max(
            0.35,
            min(1.10, st.d1_sensitivity + (0.015 * (1.0 - st.d1_sensitivity) - 0.045 * max(0.0, st.d1_occupancy - 0.65)) * dt),
        )
        st.ne_beta_sensitivity = max(
            0.35,
            min(1.10, st.ne_beta_sensitivity + (0.015 * (1.0 - st.ne_beta_sensitivity) - 0.045 * max(0.0, st.ne_beta_alpha1_occupancy - 0.65)) * dt),
        )
        st.ht1a_sensitivity = max(
            0.35,
            min(1.10, st.ht1a_sensitivity + (0.012 * (1.0 - st.ht1a_sensitivity) - 0.035 * max(0.0, inp.systemic_cortisol - 0.55)) * dt),
        )
        st.ach_m1_sensitivity = max(
            0.35,
            min(1.10, st.ach_m1_sensitivity + (0.018 * (1.0 - st.ach_m1_sensitivity) - 0.040 * max(0.0, st.ach_muscarinic_m1_occupancy - 0.70)) * dt),
        )

        # 9. Synaptic Tag-and-Capture Consolidation Strength (Frey & Morris, 1997)
        # Flashbulb / trauma permanence when phasic DA error OR NE beta surge + Cortisol co-occur!
        st.consolidation_salience_tag = min(
            1.0,
            0.85 * st.ne_beta_alpha1_occupancy
            + 0.65 * abs(td_err_clamped)
            + 0.45 * inp.systemic_cortisol * st.ne_beta_alpha1_occupancy
            + 0.50 * inp.rail_ripple_intensity,
        )

        return st, v_pred_tensor
