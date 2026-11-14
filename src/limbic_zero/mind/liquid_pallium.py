"""Multi-Timescale Continuous-Time Liquid-Spiking Pallium (Sensorimotor, Insula/dACC, & Prefrontal).

100% Transformer-Free, Attention-Free Biophysical Recurrent Substrate:
    Implements three anatomically coupled two-compartment liquid-spiking columns with a
    biological hierarchy of intrinsic neuronal time constants (Murray et al., 2014):
        1. ``sensorimotor_col`` (64 neurons, fast ~30ms): Bottom-up exteroceptive & motor loop.
        2. ``insula_cingulate_col`` (64 neurons, medium ~180ms): Visceral VMpo, Amygdala, &
           CA1 somatic integration; computes Dorsal Anterior Cingulate (dACC) conflict!
        3. ``prefrontal_slow_col`` (64 neurons, slow ~1.1s + NMDA plateaus + 450ms STP):
           Persistent working memory, long-horizon intention, and lingering post-event
           rumination ('뒤끝' reverberation long after stimuli disappear!).

    Bottom-up projections drive basal somatic/AMPA conductances, while top-down predictions
    from slow Prefrontal -> Insula/dACC -> Sensorimotor target **apical dendritic NMDA**
    compartments!
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from limbic_zero.mind.biophysics import (
    BiophysicalNeuralPopulation,
    DalesLawSynapse,
    PopulationConfig,
    SynapseConfig,
)
from limbic_zero.mind.neuromodulators import NeuromodulatorState


@dataclass(frozen=True)
class PalliumConfig:
    """Sizes and biophysical sub-stepping parameters of the 3-column Liquid-Spiking Pallium."""

    sensorimotor_dim: int = 64
    insula_cingulate_dim: int = 64
    prefrontal_dim: int = 64
    max_ode_substep_sec: float = 0.0125     # <= 12.5 ms biophysical integration resolution

    @property
    def total_pallial_dim(self) -> int:
        return self.sensorimotor_dim + self.insula_cingulate_dim + self.prefrontal_dim  # 192D


@dataclass
class PalliumStepOutput:
    """Continuous-spiking mind workspace, spike statistics, 80Hz dipole LFP, and dACC conflict."""

    workspace_vector: torch.Tensor          # (192,) concatenated [sensorimotor, insula_dacc, prefrontal]
    sensorimotor_trace: torch.Tensor        # (64,)
    insula_cingulate_trace: torch.Tensor    # (64,)
    prefrontal_trace: torch.Tensor          # (64,) slow persistent intention/rumination state
    mean_firing_rate_norm: float            # Mean population firing rate [0, 1] -> drives brain ATP burn!
    total_spikes_emitted: float             # Total action potentials fired across substeps
    dacc_conflict_signal: float             # [0, 1] Anterior Cingulate competing-drive conflict
    prefrontal_persistence_norm: float      # L2 norm of slow prefrontal working-memory attractor
    mean_synaptic_plasticity_delta: float   # Mean online 3-factor STDP magnitude
    substep_lfp_dipole: torch.Tensor        # (substeps, 3) Buzsaki pyramidal axial dipole LFP [SM, Ins, PFC] (mV)
    substep_dt: float                       # Integration substep duration in seconds (~0.0125s -> 80 Hz)


class MultiTimescaleLiquidPallium(nn.Module):
    """Three-column hierarchical continuous-time spiking-dendritic cortical/pallial engine."""

    def __init__(
        self,
        thalamic_extero_dim: int = 76,
        thalamic_intero_dim: int = 28,
        amygdala_embed_dim: int = 24,
        hippocampal_ctx_dim: int = 192,
        config: PalliumConfig | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.config = config or PalliumConfig()
        self.dtype = dtype
        self._device = torch.device(device)
        cfg = self.config

        # 1. Three populations with biological hierarchy of intrinsic membrane, dendritic, & trace taus
        self.sensorimotor_col = BiophysicalNeuralPopulation(
            PopulationConfig(
                size=cfg.sensorimotor_dim,
                base_tau_soma=0.022,
                base_tau_dendrite=0.035,
                tau_firing_trace=0.028,
                adaptation_jump=0.085,
                dendritic_coupling=0.22,
            ),
            dtype=dtype,
            device=self._device,
        )
        self.insula_cingulate_col = BiophysicalNeuralPopulation(
            PopulationConfig(
                size=cfg.insula_cingulate_dim,
                base_tau_soma=0.160,
                base_tau_dendrite=0.280,
                tau_firing_trace=0.110,
                adaptation_jump=0.060,
            ),
            dtype=dtype,
            device=self._device,
        )
        self.prefrontal_slow_col = BiophysicalNeuralPopulation(
            PopulationConfig(
                size=cfg.prefrontal_dim,
                base_tau_soma=0.950,
                base_tau_dendrite=1.750,
                tau_nmda=0.220,
                tau_firing_trace=0.480,
                adaptation_jump=0.035,
                dendritic_coupling=0.58,
            ),
            dtype=dtype,
            device=self._device,
        )

        # 2. Afferent sensory, visceral, amygdalar, and hippocampal input projections
        self.in_extero_to_sm = nn.Linear(thalamic_extero_dim, cfg.sensorimotor_dim, dtype=dtype, device=self._device)
        self.in_intero_to_ins = nn.Linear(
            thalamic_intero_dim + amygdala_embed_dim + 4,
            cfg.insula_cingulate_dim,
            dtype=dtype,
            device=self._device,
        )
        self.in_hippo_to_pfc = nn.Linear(
            hippocampal_ctx_dim + amygdala_embed_dim,
            cfg.prefrontal_dim,
            dtype=dtype,
            device=self._device,
        )

        # 3. Intra-column Dale's Law Recurrent Synapses (with Tsodyks-Markram STP & 3-Factor STDP)
        fast_syn_cfg = SynapseConfig(tau_facilitation=0.15, tau_recovery=0.35, plasticity_lr=0.010)
        med_syn_cfg = SynapseConfig(tau_facilitation=0.45, tau_recovery=0.30, plasticity_lr=0.012)
        slow_syn_cfg = SynapseConfig(
            u0_baseline_release=0.24,
            tau_facilitation=1.10,  # Strong Ca2+ facilitation sustains prefrontal working-memory reverberation!
            tau_recovery=0.18,
            tau_eligibility=2.60,
            plasticity_lr=0.014,
        )

        self.rec_sm = DalesLawSynapse(
            cfg.sensorimotor_dim,
            cfg.sensorimotor_dim,
            pre_dales_sign=self.sensorimotor_col.dales_sign,
            init_scale=0.24,
            config=fast_syn_cfg,
            dtype=dtype,
            device=self._device,
        )
        self.rec_ins = DalesLawSynapse(
            cfg.insula_cingulate_dim,
            cfg.insula_cingulate_dim,
            pre_dales_sign=self.insula_cingulate_col.dales_sign,
            init_scale=0.55,
            config=med_syn_cfg,
            dtype=dtype,
            device=self._device,
        )
        self.rec_pfc = DalesLawSynapse(
            cfg.prefrontal_dim,
            cfg.prefrontal_dim,
            pre_dales_sign=self.prefrontal_slow_col.dales_sign,
            init_scale=0.82,
            sparsity=0.15,
            config=slow_syn_cfg,
            dtype=dtype,
            device=self._device,
        )

        # 4. Inter-column Feedforward (basal) and Top-Down Predictive (apical dendritic) Pathways
        self.ff_sm_to_ins = DalesLawSynapse(
            cfg.sensorimotor_dim,
            cfg.insula_cingulate_dim,
            pre_dales_sign=self.sensorimotor_col.dales_sign,
            init_scale=0.45,
            config=med_syn_cfg,
            dtype=dtype,
            device=self._device,
        )
        self.ff_ins_to_pfc = DalesLawSynapse(
            cfg.insula_cingulate_dim,
            cfg.prefrontal_dim,
            pre_dales_sign=self.insula_cingulate_col.dales_sign,
            init_scale=0.55,
            config=slow_syn_cfg,
            dtype=dtype,
            device=self._device,
        )
        self.fb_pfc_to_ins_apical = DalesLawSynapse(
            cfg.prefrontal_dim,
            cfg.insula_cingulate_dim,
            pre_dales_sign=self.prefrontal_slow_col.dales_sign,
            init_scale=0.48,
            config=med_syn_cfg,
            dtype=dtype,
            device=self._device,
        )
        self.fb_ins_to_sm_apical = DalesLawSynapse(
            cfg.insula_cingulate_dim,
            cfg.sensorimotor_dim,
            pre_dales_sign=self.insula_cingulate_col.dales_sign,
            init_scale=0.35,
            config=fast_syn_cfg,
            dtype=dtype,
            device=self._device,
        )

    def reset_state(self) -> None:
        self.sensorimotor_col.reset_state()
        self.insula_cingulate_col.reset_state()
        self.prefrontal_slow_col.reset_state()
        for syn in self._all_synapses():
            syn.reset_state()

    def detach_state_(self) -> None:
        self.sensorimotor_col.detach_state_()
        self.insula_cingulate_col.detach_state_()
        self.prefrontal_slow_col.detach_state_()
        for syn in self._all_synapses():
            syn.detach_state_()

    def _all_synapses(self) -> Tuple[DalesLawSynapse, ...]:
        return (
            self.rec_sm,
            self.rec_ins,
            self.rec_pfc,
            self.ff_sm_to_ins,
            self.ff_ins_to_pfc,
            self.fb_pfc_to_ins_apical,
            self.fb_ins_to_sm_apical,
        )

    @property
    def current_workspace(self) -> torch.Tensor:
        return torch.cat(
            [
                self.sensorimotor_col.firing_trace,
                self.insula_cingulate_col.firing_trace,
                self.prefrontal_slow_col.firing_trace,
            ],
            dim=0,
        )

    def step(
        self,
        extero_thalamic: torch.Tensor,
        intero_thalamic: torch.Tensor,
        amygdala_embed: torch.Tensor,
        hippo_ca1_ctx: torch.Tensor,
        recalled_somatic_marker: torch.Tensor,
        neuromod: NeuromodulatorState,
        dt: float,
        *,
        brain_atp: float = 0.92,
        core_temp_celsius: float = 37.0,
        enable_plasticity: bool = True,
        deterministic: bool = False,
    ) -> PalliumStepOutput:
        """Integrate coupled 3-column continuous-time ODE across biophysical substeps."""
        cfg = self.config
        substeps = max(1, int(round(dt / cfg.max_ode_substep_sec)))
        if substeps * cfg.max_ode_substep_sec < dt - 1e-6:
            substeps += 1
        h = dt / float(substeps)

        # Afferent feedforward currents from Thalamus, Amygdala, & Hippocampus
        ext_present = float(extero_thalamic.abs().mean().item())
        ext_coincidence_gate = min(1.0, 0.08 + 2.5 * ext_present)  # Larkum apical coincidence gate!

        ext_drive = self.in_extero_to_sm(extero_thalamic) * min(1.0, ext_present * 4.0)
        ins_in = torch.cat([intero_thalamic, amygdala_embed, recalled_somatic_marker], dim=0)
        visc_drive = self.in_intero_to_ins(ins_in)
        pfc_in = torch.cat([hippo_ca1_ctx, amygdala_embed], dim=0)
        mem_drive = self.in_hippo_to_pfc(pfc_in)

        ext_exc, ext_inh = torch.relu(ext_drive), torch.relu(-ext_drive)
        visc_exc, visc_inh = torch.relu(visc_drive), torch.relu(-visc_drive)
        mem_exc, mem_inh = torch.relu(mem_drive), torch.relu(-mem_drive)

        total_spikes = 0.0
        sm_trace = self.sensorimotor_col.firing_trace
        ins_trace = self.insula_cingulate_col.firing_trace
        pfc_trace = self.prefrontal_slow_col.firing_trace

        common_mod = dict(
            d1_occupancy=neuromod.d1_occupancy,
            d2_occupancy=neuromod.d2_occupancy,
            ne_alpha2a=neuromod.ne_alpha2a_occupancy,
            ne_beta_alpha1=neuromod.ne_beta_alpha1_occupancy,
            ht1a_occupancy=neuromod.ht1a_occupancy,
            ht2a_occupancy=neuromod.ht2a_occupancy,
            ach_m1=neuromod.ach_muscarinic_m1_occupancy,
            adenosine_a1=neuromod.adenosine_a1_occupancy,
            brain_atp=brain_atp,
            core_temp_celsius=core_temp_celsius,
            deterministic=deterministic,
        )

        lfp_rows = []
        for _ in range(substeps):
            r_sm_exc, r_sm_inh = self.rec_sm.forward_currents(sm_trace, h)
            r_ins_exc, r_ins_inh = self.rec_ins.forward_currents(ins_trace, h)
            r_pfc_exc, r_pfc_inh = self.rec_pfc.forward_currents(pfc_trace, h)

            ff1_exc, ff1_inh = self.ff_sm_to_ins.forward_currents(sm_trace, h)
            ff2_exc, ff2_inh = self.ff_ins_to_pfc.forward_currents(ins_trace, h)

            fb_pfc_exc, fb_pfc_inh = self.fb_pfc_to_ins_apical.forward_currents(pfc_trace, h)
            fb_ins_exc, fb_ins_inh = self.fb_ins_to_sm_apical.forward_currents(ins_trace, h)

            # Step 1: Sensorimotor Column (fast sensory-locked; apical input modulates rather than hallucinates!)
            sp_sm, sm_trace = self.sensorimotor_col.step(
                basal_exc_drive=0.80 * ext_exc + 0.25 * r_sm_exc,
                basal_inh_drive=0.50 * ext_inh + 0.75 * r_sm_inh,
                apical_exc_drive=0.45 * fb_ins_exc * ext_coincidence_gate,
                apical_inh_drive=0.45 * fb_ins_inh,
                dt=h,
                **common_mod,
            )

            # Step 2: Anterior Insula & Dorsal Anterior Cingulate Column
            sp_ins, ins_trace = self.insula_cingulate_col.step(
                basal_exc_drive=0.65 * visc_exc + 0.45 * ff1_exc + 0.58 * r_ins_exc,
                basal_inh_drive=0.45 * visc_inh + 0.45 * ff1_inh + 0.65 * r_ins_inh,
                apical_exc_drive=0.55 * fb_pfc_exc,
                apical_inh_drive=0.45 * fb_pfc_inh,
                dt=h,
                **common_mod,
            )

            # Step 3: Slow Prefrontal Working-Memory / Rumination Column
            pfc_reverberation_gain = (1.15 + 0.55 * neuromod.ne_alpha2a_occupancy + 0.35 * neuromod.d1_occupancy) * max(
                0.45, 1.0 - 0.50 * neuromod.ne_beta_alpha1_occupancy
            )
            sp_pfc, pfc_trace = self.prefrontal_slow_col.step(
                basal_exc_drive=0.55 * mem_exc + 0.50 * ff2_exc + 0.88 * pfc_reverberation_gain * r_pfc_exc,
                basal_inh_drive=0.40 * mem_inh + 0.45 * ff2_inh + 0.55 * r_pfc_inh,
                apical_exc_drive=0.45 * mem_exc + 0.45 * r_pfc_exc,
                apical_inh_drive=0.30 * r_pfc_inh,
                dt=h,
                **common_mod,
            )

            total_spikes += float((sp_sm.sum() + sp_ins.sum() + sp_pfc.sum()).item())

            # Biophysical extracellular dipole LFP (Buzsaki et al. 2012, in mV):
            # Apical-to-soma axial return current + perisomatic E/I synaptic dipole
            with torch.no_grad():
                lfp_sm = 100.0 * (
                    (self.sensorimotor_col.v_dend - self.sensorimotor_col.v_soma).mean()
                    + 0.30 * (self.sensorimotor_col.g_ampa + self.sensorimotor_col.g_nmda - self.sensorimotor_col.g_gaba_a).mean()
                )
                lfp_ins = 100.0 * (
                    (self.insula_cingulate_col.v_dend - self.insula_cingulate_col.v_soma).mean()
                    + 0.30 * (self.insula_cingulate_col.g_ampa + self.insula_cingulate_col.g_nmda - self.insula_cingulate_col.g_gaba_a).mean()
                )
                lfp_pfc = 100.0 * (
                    (self.prefrontal_slow_col.v_dend - self.prefrontal_slow_col.v_soma).mean()
                    + 0.30 * (self.prefrontal_slow_col.g_ampa + self.prefrontal_slow_col.g_nmda - self.prefrontal_slow_col.g_gaba_a).mean()
                )
                lfp_rows.append(torch.stack([lfp_sm, lfp_ins, lfp_pfc]))

        substep_lfp = torch.stack(lfp_rows, dim=0)

        # Online Three-Factor Neuromodulated STDP + Homeostatic Scaling across pallial synapses
        third_factor = (
            0.85 * neuromod.td_prediction_error
            + 0.40 * (neuromod.d1_occupancy - 0.45 * neuromod.d2_occupancy)
            + 0.35 * neuromod.ne_beta_alpha1_occupancy
        )
        ach_gate = neuromod.ach_muscarinic_m1_occupancy
        plast_deltas = [
            self.rec_sm.update_eligibility_and_plasticity(sm_trace, sm_trace, third_factor, ach_gate, dt, enable_plasticity),
            self.rec_ins.update_eligibility_and_plasticity(ins_trace, ins_trace, third_factor, ach_gate, dt, enable_plasticity),
            self.rec_pfc.update_eligibility_and_plasticity(pfc_trace, pfc_trace, third_factor, ach_gate, dt, enable_plasticity),
            self.ff_sm_to_ins.update_eligibility_and_plasticity(sm_trace, ins_trace, third_factor, ach_gate, dt, enable_plasticity),
            self.ff_ins_to_pfc.update_eligibility_and_plasticity(ins_trace, pfc_trace, third_factor, ach_gate, dt, enable_plasticity),
            self.fb_pfc_to_ins_apical.update_eligibility_and_plasticity(pfc_trace, ins_trace, third_factor, ach_gate, dt, enable_plasticity),
            self.fb_ins_to_sm_apical.update_eligibility_and_plasticity(ins_trace, sm_trace, third_factor, ach_gate, dt, enable_plasticity),
        ]

        workspace = torch.cat([sm_trace, ins_trace, pfc_trace], dim=0)
        mean_rate = float(workspace.mean().item())

        # Dorsal Anterior Cingulate Conflict (Botvinick et al., 2001):
        # Co-activation of excitatory & inhibitory subpopulations + disagreement between Insula & PFC!
        ins_exc_act = float(ins_trace[:51].mean().item())
        ins_inh_act = float(ins_trace[51:].mean().item())
        mismatch = float((ins_trace - pfc_trace).abs().mean().item())
        dacc_conflict = max(0.0, min(1.0, 2.4 * (ins_exc_act * ins_inh_act) + 0.65 * mismatch))

        pfc_norm = float(torch.linalg.vector_norm(pfc_trace).item()) / (cfg.prefrontal_dim ** 0.5)
        mean_plast = sum(plast_deltas) / len(plast_deltas)

        return PalliumStepOutput(
            workspace_vector=workspace,
            sensorimotor_trace=sm_trace,
            insula_cingulate_trace=ins_trace,
            prefrontal_trace=pfc_trace,
            mean_firing_rate_norm=mean_rate,
            total_spikes_emitted=total_spikes,
            dacc_conflict_signal=dacc_conflict,
            prefrontal_persistence_norm=pfc_norm,
            mean_synaptic_plasticity_delta=mean_plast,
            substep_lfp_dipole=substep_lfp,
            substep_dt=h,
        )
