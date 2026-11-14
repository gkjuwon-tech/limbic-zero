"""Molecular Epigenetic Receptor Methylation & Slow-Wave Sleep (SWS / SWR) Consolidation.

Zero-Emotion-Label Developmental Psychobiology:
    1. ``EpigeneticTemperamentProfile`` & ``EpigeneticPlasticityEngine``:
       Models experience-dependent chromatin remodeling and promoter methylation
       (Meaney, 2001 NR3C1 glucocorticoid receptor methylation; Krishnan & Nestler,
       2008 monoaminergic/striatal adaptations) using strictly molecular quantities:
         - ``nr3c1_gr_methylation``: Blunts hypothalamic/hippocampal GR negative feedback
           under chronic unbuffered cortisol + LHb punishment -> lingering HPA tail.
         - ``ht1a_promoter_expression``: Modulates postsynaptic 5-HT1A sensitivity ->
           alters future discount horizon ``gamma`` and STN/PAG impulse restraint.
         - ``d1_striatal_expression``: Modulates phasic D1 Go sensitivity.
         - ``ne_beta_expression``: Modulates central noradrenergic alarm sensitivity.
         - ``ach_m1_expression``: Upregulated by sustained safe epistemic novelty ->
           environmental enrichment curiosity trait.
         - ``oxtr_expression``: Upregulated by harmonic acoustic resonance & vagal safety.

    2. ``SlowWaveSleepConsolidator``:
       Executes genuine offline Non-REM Thalamocortical Slow-Wave Sleep + CA3 Sharp-Wave
       Ripple (SWR) Systems Consolidation (Born & Wilhelm, 2012) + Tononi & Cirelli (2006)
       Synaptic Homeostasis (SHY), transferring daytime hippocampal episodes into durable
       cortical, amygdalar, and striatal synaptic structure while clearing cerebral adenosine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F

from limbic_zero.core.soma import DigitalSoma
from limbic_zero.mind.amygdala_hippocampus import EpisodicEngram
from limbic_zero.mind.biophysics import DalesLawSynapse
from limbic_zero.mind.brain import LimbicBrain, MindDiagnostics
from limbic_zero.mind.thalamus_sensory import TOTAL_RAW_SENSORY_DIM


@dataclass
class EpigeneticTemperamentProfile:
    """Heritable/consolidated molecular receptor expression & promoter methylation state."""

    nr3c1_gr_methylation: float = 0.08       # [0.0, 0.85] Higher = blunted cortisol shutoff!
    ht1a_promoter_expression: float = 1.00   # [0.42, 1.35] Postsynaptic 5-HT1A density
    d1_striatal_expression: float = 1.00     # [0.48, 1.38] Striatal/cortical D1 receptor density
    ne_beta_expression: float = 1.00         # [0.50, 1.42] Central beta/alpha1 alarm sensitivity
    ach_m1_expression: float = 1.00          # [0.50, 1.45] Cortical/hippocampal M1 density
    oxtr_expression: float = 1.00            # [0.45, 1.45] Oxytocin receptor expression

    # Cumulative developmental exposure integrals (purely biographical telemetry)
    cumulative_adversity_integral: float = 0.0
    cumulative_nurturance_integral: float = 0.0
    cumulative_epistemic_integral: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpigeneticTemperamentProfile":
        valid = {k: float(v) for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


class EpigeneticPlasticityEngine:
    """Slow chromatin/methylation remodeler coupling life history to receptor expression."""

    def __init__(self, rate_scale: float = 0.035) -> None:
        self.rate_scale = float(rate_scale)

    def update_from_waking_tick(
        self,
        profile: EpigeneticTemperamentProfile,
        soma: DigitalSoma,
        diag: MindDiagnostics,
        dt: float,
    ) -> None:
        """Integrate endogenous endocrine & subcortical firing signals into epigenetic marks."""
        nm = diag.neuromodulators
        cort = soma.endo.adrenal_cortisol
        oxy = soma.endo.oxytocin_systemic
        vagal = soma.endo.ventral_vagal_tone
        lhb = nm.lhb_aversive_rate
        novelty = diag.thalamus.epistemic_novelty
        td = nm.td_prediction_error

        # 1. Adversity vs Safety-Nurturance vs Epistemic Enrichment signals
        adversity = max(0.0, cort - 0.24) * 1.4 + max(0.0, lhb - 0.32) * 0.9 + 0.6 * diag.amygdala.cea_defensive_output
        nurturance = max(0.0, oxy - 0.24) * 1.5 + max(0.0, vagal - 0.62) * 1.1 + 0.5 * max(0.0, td)
        safe_epistemic = novelty * max(0.1, vagal) * max(0.0, 1.0 - 1.5 * diag.amygdala.cea_defensive_output)

        profile.cumulative_adversity_integral += adversity * dt
        profile.cumulative_nurturance_integral += nurturance * dt
        profile.cumulative_epistemic_integral += safe_epistemic * dt

        eta = self.rate_scale * dt

        # 2. NR3C1 Glucocorticoid Receptor Promoter Methylation (Meaney, 2001):
        # Methylated by severe cortisol + LHb punishment; slowly demethylated by oxytocin/vagal safety
        d_meth = 0.55 * adversity - 0.30 * nurturance - 0.02 * (profile.nr3c1_gr_methylation - 0.08)
        profile.nr3c1_gr_methylation = max(0.02, min(0.82, profile.nr3c1_gr_methylation + eta * d_meth))

        # 3. Postsynaptic 5-HT1A Promoter Expression:
        # Downregulated by chronic glucocorticoid/LHb load; upregulated by safe vagal/oxytocin tone
        d_ht1a = 0.42 * nurturance - 0.52 * adversity + 0.03 * (1.0 - profile.ht1a_promoter_expression)
        profile.ht1a_promoter_expression = max(0.42, min(1.35, profile.ht1a_promoter_expression + eta * d_ht1a))

        # 4. Striatal/Cortical D1 Expression:
        # Strengthened by mastery (positive TD + safe exploration); blunted by chronic inescapable LHb
        d_d1 = 0.45 * max(0.0, td) + 0.25 * safe_epistemic - 0.30 * max(0.0, lhb - 0.55)
        profile.d1_striatal_expression = max(0.48, min(1.38, profile.d1_striatal_expression + eta * d_d1))

        # 5. Central NE Beta/Alpha1 Alarm Sensitivity:
        # Sensitized by repeated acute threat surges; calmed by nurturance
        d_neb = 0.48 * adversity - 0.28 * nurturance + 0.02 * (1.0 - profile.ne_beta_expression)
        profile.ne_beta_expression = max(0.50, min(1.42, profile.ne_beta_expression + eta * d_neb))

        # 6. Cholinergic M1 Cortical Expression (Environmental Enrichment):
        d_m1 = 0.65 * safe_epistemic - 0.22 * max(0.0, cort - 0.45) + 0.02 * (1.0 - profile.ach_m1_expression)
        profile.ach_m1_expression = max(0.50, min(1.45, profile.ach_m1_expression + eta * d_m1))

        # 7. Oxytocin Receptor (OXTR) Expression:
        d_oxtr = 0.50 * nurturance - 0.32 * adversity + 0.02 * (1.0 - profile.oxtr_expression)
        profile.oxtr_expression = max(0.45, min(1.45, profile.oxtr_expression + eta * d_oxtr))

        self.apply_to_living_substrate(profile, brain_nm_state=nm, soma=soma, blend=min(1.0, 0.25 * dt))

    @staticmethod
    def apply_to_living_substrate(
        profile: EpigeneticTemperamentProfile,
        brain_nm_state: Any,
        soma: DigitalSoma | None = None,
        blend: float = 1.0,
    ) -> None:
        """Bind epigenetic expression baselines onto live neuromodulator & endocrine receptors."""
        b = max(0.0, min(1.0, float(blend)))
        brain_nm_state.ht1a_sensitivity = (
            (1.0 - b) * brain_nm_state.ht1a_sensitivity + b * profile.ht1a_promoter_expression
        )
        brain_nm_state.d1_sensitivity = (
            (1.0 - b) * brain_nm_state.d1_sensitivity + b * profile.d1_striatal_expression
        )
        brain_nm_state.ne_beta_sensitivity = (
            (1.0 - b) * brain_nm_state.ne_beta_sensitivity + b * profile.ne_beta_expression
        )
        brain_nm_state.ach_m1_sensitivity = (
            (1.0 - b) * brain_nm_state.ach_m1_sensitivity + b * profile.ach_m1_expression
        )

        if soma is not None:
            target_gr = max(0.25, min(1.15, 1.05 - 0.85 * profile.nr3c1_gr_methylation))
            soma.endo.gr_receptor_sensitivity = (
                (1.0 - b) * soma.endo.gr_receptor_sensitivity + b * target_gr
            )
            soma.endo.oxytocin_receptor_sensitivity = (
                (1.0 - b) * soma.endo.oxytocin_receptor_sensitivity + b * profile.oxtr_expression
            )
            soma.endo.beta_receptor_sensitivity = (
                (1.0 - b) * soma.endo.beta_receptor_sensitivity + b * min(1.20, profile.ne_beta_expression)
            )


@dataclass
class SleepConsolidationReport:
    """Telemetry summary of a single Non-REM Slow-Wave Sleep & SWR consolidation cycle."""

    sleep_ticks: int
    swr_bursts_fired: int
    cortical_synaptic_delta_norm: float
    striatal_habit_delta_norm: float
    amygdala_consolidation_delta: float
    adenosine_before: float
    adenosine_after: float
    surviving_engrams: int


class SlowWaveSleepConsolidator:
    """Offline Non-REM Slow-Wave Sleep + CA3 SWR Replay + Tononi-Cirelli Synaptic Homeostasis."""

    def __init__(
        self,
        shy_downscale_factor: float = 0.994,
        systems_consolidation_lr: float = 0.032,
    ) -> None:
        self.shy_downscale_factor = float(shy_downscale_factor)
        self.systems_consolidation_lr = float(systems_consolidation_lr)

    @torch.no_grad()
    def run_sleep_cycle(
        self,
        brain: LimbicBrain,
        soma: DigitalSoma,
        profile: EpigeneticTemperamentProfile | None = None,
        sleep_ticks: int = 80,
        dt: float = 0.05,
    ) -> SleepConsolidationReport:
        """Place ``brain`` & ``soma`` in dark quiet rest for offline SWR systems consolidation."""
        if not soma.is_alive:
            return SleepConsolidationReport(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, len(brain.hippocampus.engrams))

        adenosine_before = float(soma.metab.brain_adenosine)
        # Ensure elevated initial sleep pressure so TRN spindle gating & CA3 SWR engage naturally
        soma.metab.brain_adenosine = max(soma.metab.brain_adenosine, 0.58)

        swr_count = 0
        cort_delta_acc = 0.0
        striatal_delta_acc = 0.0
        amyg_delta_acc = 0.0

        # Quiet dark sensory tensor (eyes closed, silent room, normoxic air, resting viscera)
        quiet_sensory = torch.zeros(
            TOTAL_RAW_SENSORY_DIM,
            dtype=brain.dtype,
            device=brain.prev_action_7d.device,
        )

        for step_i in range(sleep_ticks):
            # Adenosine clearance & cerebral phosphagen restoration during slow-wave sleep
            soma.metab.brain_adenosine = max(0.05, soma.metab.brain_adenosine * (1.0 - 0.32 * dt))
            soma.metab.brain_atp = min(0.97, soma.metab.brain_atp + 0.25 * (0.97 - soma.metab.brain_atp) * dt)
            soma.metab.brain_pcr = min(0.94, soma.metab.brain_pcr + 0.25 * (0.94 - soma.metab.brain_pcr) * dt)
            # HPA cortisol clearance depends on NR3C1 GR sensitivity (methylated GR clears slower!)
            gr_eff = soma.endo.gr_receptor_sensitivity
            soma.endo.adrenal_cortisol = max(
                0.12,
                soma.endo.adrenal_cortisol - 0.14 * gr_eff * (soma.endo.adrenal_cortisol - 0.12) * dt,
            )

            # Force hippocampal SWR readiness during deep slow-wave sleep ripples
            brain.hippocampus._swr_cooldown = max(0.0, brain.hippocampus._swr_cooldown - 2.0 * dt)
            prev_ws = brain.pallium.current_workspace

            thal_out = brain.thalamus.step(
                raw_sensory_414d=quiet_sensory,
                pallial_context=prev_ws,
                prev_action_7d=torch.zeros_like(brain.prev_action_7d),
                dt=dt,
                ach_nicotinic=0.08,  # Cholinergic silence during NREM sleep!
                ne_arousal=0.10,
                adenosine_a1=min(0.95, 0.50 + soma.metab.brain_adenosine),
                somatic_pain_salience=0.0,
                enable_world_model_learning=False,
            )

            hippo_out = brain.hippocampus.step(
                sim_time=float(brain.total_steps + step_i) * dt,
                extero_cue=thal_out.extero_cue_vector,
                pallial_context=prev_ws,
                prev_action_7d=torch.zeros_like(brain.prev_action_7d),
                dt=dt,
                consolidation_salience_tag=0.0,
                valence_relief=0.0,
                visceral_trauma=0.0,
                cea_threat=float(brain.amygdala.cea_activity_ema.item()),
                epistemic_novelty=0.0,
                ach_m1_encoding_gate=0.06,  # Low ACh disinhibits CA3->CA1 SWR transfer!
                adenosine_a1_sleep=0.72,
                motor_quiescence=1.0,
            )

            if hippo_out.swr_replay_active and hippo_out.swr_replay_cue is not None and brain.hippocampus.engrams:
                swr_count += 1
                # Identify the replayed consolidated engram
                best_eng: EpisodicEngram = max(
                    brain.hippocampus.engrams,
                    key=lambda e: e.consolidation_shield * e.vitality,
                )
                relief_val = float(best_eng.somatic_marker[0].item())
                trauma_val = float(best_eng.somatic_marker[1].item())
                novelty_val = float(best_eng.somatic_marker[3].item())
                shield = best_eng.consolidation_shield

                # 1. Propagate replayed cue & CA1 context into Pallium (cortical systems consolidation!)
                nm_sleep = brain.neuromod_engine.state
                pall_out = brain.pallium.step(
                    extero_thalamic=hippo_out.swr_replay_cue,
                    intero_thalamic=thal_out.intero_thalamic_vector,
                    amygdala_embed=brain.amygdala.out_proj.bias.detach() * 0.0
                    + torch.zeros(brain.config.amygdala_embed_dim, dtype=brain.dtype, device=prev_ws.device),
                    hippo_ca1_ctx=best_eng.ca1_context_vec,
                    recalled_somatic_marker=best_eng.somatic_marker,
                    neuromod=nm_sleep,
                    dt=dt,
                    brain_atp=soma.metab.brain_atp,
                    core_temp_celsius=soma.thermo.core_temp_celsius,
                    enable_plasticity=True,
                    deterministic=False,
                )

                # Apply replay-tagged plasticity across cortical Dale's Law synapses
                replay_mod = self.systems_consolidation_lr * shield * (
                    0.8 * relief_val + 0.9 * trauma_val + 0.5 * novelty_val
                )
                for mod in brain.pallium.modules():
                    if isinstance(mod, DalesLawSynapse):
                        delta_w = replay_mod * mod.eligibility_trace * dt
                        mod.weight_mag.data.add_(delta_w).mul_(mod.structural_mask).clamp_(0.0, mod.config.weight_max)
                        cort_delta_acc += float(delta_w.abs().mean().item())

                # 2. Offline Amygdala BLA / ITC Consolidation of replayed trauma or safety relief
                la_feat = F.relu(torch.tanh(brain.amygdala.la_encoder(hippo_out.swr_replay_cue)))
                if trauma_val > 0.10:
                    d_bla = self.systems_consolidation_lr * shield * trauma_val * la_feat * dt * 2.5
                    brain.amygdala.w_bla_defensive.add_(d_bla).clamp_(0.0, 2.5)
                    amyg_delta_acc += float(d_bla.abs().mean().item())
                if relief_val > 0.08:
                    d_app = self.systems_consolidation_lr * shield * relief_val * la_feat * dt * 2.2
                    brain.amygdala.w_bla_appetitive.add_(d_app).clamp_(0.0, 2.5)
                    amyg_delta_acc += float(d_app.abs().mean().item())

                # 3. Offline Corticostriatal Habit Consolidation (D1-Go vs D2-NoGo)
                joint_bg = torch.cat(
                    [
                        pall_out.workspace_vector.detach(),
                        torch.zeros(brain.config.amygdala_embed_dim, dtype=brain.dtype, device=prev_ws.device),
                        hippo_out.swr_replay_cue.detach(),
                    ],
                    dim=0,
                )
                s_feat = torch.tanh(brain.basal_ganglia.cortico_striatal_enc(joint_bg))
                s_pos = F.relu(s_feat) + 0.15 * torch.sigmoid(s_feat)

                act7 = best_eng.action_taken
                # Expand 7D executed motor vector onto the 9 push-pull striatal subchannels
                act9 = torch.stack(
                    [
                        F.relu(act7[0]),
                        F.relu(-act7[0]),
                        F.relu(act7[1]),
                        F.relu(-act7[1]),
                        F.relu(act7[2]),
                        F.relu(act7[3]),
                        F.relu(act7[4]),
                        F.relu(act7[5]),
                        F.relu(act7[6]),
                    ]
                )
                net_valence = max(-1.2, min(1.2, relief_val - 1.15 * trauma_val))
                habit_step = (self.systems_consolidation_lr * 0.85 * shield * net_valence * dt) * torch.outer(
                    act9, s_pos
                )
                brain.basal_ganglia.w_d1_go.data.add_(habit_step).clamp_(-1.5, 1.5)
                brain.basal_ganglia.w_d2_nogo.data.add_(-0.85 * habit_step).clamp_(-1.5, 1.5)
                striatal_delta_acc += float(habit_step.abs().mean().item())

        # Tononi & Cirelli (2006) Synaptic Homeostasis Down-Selection (SHY):
        # Gently renormalizes untagged background synaptic growth while preserving consolidated peaks
        for mod in brain.pallium.modules():
            if isinstance(mod, DalesLawSynapse):
                mod.weight_mag.data.mul_(self.shy_downscale_factor)
        brain.basal_ganglia.w_d1_go.data.mul_(self.shy_downscale_factor)
        brain.basal_ganglia.w_d2_nogo.data.mul_(self.shy_downscale_factor)

        if profile is not None:
            EpigeneticPlasticityEngine.apply_to_living_substrate(
                profile,
                brain_nm_state=brain.neuromod_engine.state,
                soma=soma,
                blend=1.0,
            )

        soma._sync_projected_state_tensor()
        return SleepConsolidationReport(
            sleep_ticks=sleep_ticks,
            swr_bursts_fired=swr_count,
            cortical_synaptic_delta_norm=cort_delta_acc,
            striatal_habit_delta_norm=striatal_delta_acc,
            amygdala_consolidation_delta=amyg_delta_acc,
            adenosine_before=adenosine_before,
            adenosine_after=float(soma.metab.brain_adenosine),
            surviving_engrams=len(brain.hippocampus.engrams),
        )
