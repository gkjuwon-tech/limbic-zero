"""Biophysical Spiking-Dendritic Neurons, Dale's Law Synapses, Tsodyks-Markram STP, & Three-Factor STDP.

Implements scratch continuous-time neurobiological primitives WITHOUT any
Transformer, Attention, LSTM, or GRU layers:
    1. Two-Compartment Conductance-Based Liquid-Spiking Population
       (``BiophysicalNeuralPopulation``):
       - Apical/basal dendritic compartment with voltage-gated NMDA Mg2+ unblock.
       - Somatic compartment with leak, excitatory AMPA/NMDA, inhibitory GABA_A/GABA_B,
         Ca2+-activated K+ spike-frequency adaptation (modulated by muscarinic M1 ACh!),
         Q10 temperature kinetics, and cerebral ATP Na+/K+-pump coupling.
       - Differentiable fast-sigmoid surrogate spike emission + continuous rate trace.
    2. Dale's Law Excitatory/Inhibitory Synaptic Projection (``DalesLawSynapse``)
       with Tsodyks-Markram Short-Term Synaptic Plasticity (vesicle depletion ``x_ves``
       & residual Ca2+ facilitation ``u_rel``) AND Three-Factor Neuromodulated
       Eligibility-Trace Plasticity + Turrigiano Homeostatic Synaptic Scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateSpikeFunction(torch.autograd.Function):
    """Binary threshold spike in forward pass; fast-sigmoid derivative in backward pass."""

    @staticmethod
    def forward(ctx, membrane_excess: torch.Tensor, sharpness: float = 12.0) -> torch.Tensor:
        ctx.save_for_backward(membrane_excess)
        ctx.sharpness = sharpness
        return (membrane_excess >= 0.0).to(membrane_excess.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (membrane_excess,) = ctx.saved_tensors
        denom = 1.0 + ctx.sharpness * torch.abs(membrane_excess)
        surrogate_grad = 1.0 / (denom * denom)
        return grad_output * surrogate_grad, None


def surrogate_spike(membrane_excess: torch.Tensor, sharpness: float = 12.0) -> torch.Tensor:
    """Emit differentiable binary action potentials when ``v >= v_threshold``."""
    return SurrogateSpikeFunction.apply(membrane_excess, sharpness)


@dataclass(frozen=True)
class PopulationConfig:
    """Biophysical membrane, ion-channel, and receptor time constants (in seconds)."""

    size: int = 64
    excitatory_fraction: float = 0.80       # 80% glutamatergic pyramidal, 20% GABAergic interneuron
    base_tau_soma: float = 0.035            # 35 ms baseline somatic RC time constant
    base_tau_dendrite: float = 0.055        # 55 ms dendritic RC time constant
    tau_ampa: float = 0.012                 # Fast AMPA glutamatergic decay
    tau_nmda: float = 0.095                 # Slow voltage-gated NMDA decay
    tau_gaba_a: float = 0.015               # Fast GABA_A chloride shunting decay
    tau_gaba_b: float = 0.130               # Slow GABA_B metabotropic K+ decay
    tau_adaptation: float = 0.280           # Ca2+-activated K+ (SK / M-current) decay
    tau_firing_trace: float = 0.065         # Filtered postsynaptic firing rate trace
    v_rest: float = -0.65                   # Normalized resting potential (-65 mV -> -0.65)
    v_reset: float = -0.72                  # Post-spike hyperpolarized reset (-72 mV)
    v_threshold: float = -0.48              # Action potential threshold (-48 mV)
    e_rev_exc: float = 0.00                 # Glutamate reversal potential (0 mV)
    e_rev_inh_a: float = -0.70              # GABA_A Cl- shunting reversal (-70 mV)
    e_rev_inh_b: float = -0.90              # GABA_B K+ hyperpolarizing reversal (-90 mV)
    adaptation_jump: float = 0.085          # Spike-triggered K+ conductance increment
    dendritic_coupling: float = 0.42        # Axial conductance from dendrite -> soma
    q10_factor: float = 2.2                 # Ion-channel thermal rate scaling per 10 C


class BiophysicalNeuralPopulation(nn.Module):
    """Two-compartment conductance-based liquid-spiking cortical/subcortical population."""

    def __init__(
        self,
        config: PopulationConfig,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.config = config
        self.dtype = dtype
        self._device = torch.device(device)
        n = config.size

        n_exc = max(1, min(n - 1, int(round(n * config.excitatory_fraction)))) if n > 1 else 1
        dales_sign = torch.ones(n, dtype=dtype, device=self._device)
        if n > 1:
            dales_sign[n_exc:] = -1.0
        self.register_buffer("dales_sign", dales_sign)
        self.register_buffer("is_excitatory_mask", (dales_sign > 0.0).to(dtype))

        # Heterogeneous intrinsic neuronal time constants across the population (log-spaced)
        tau_spread = torch.linspace(0.65, 1.55, steps=n, dtype=dtype, device=self._device)
        # Fast-spiking GABAergic interneurons have ~2x faster membrane kinetics than pyramidal cells!
        tau_spread = torch.where(dales_sign > 0.0, tau_spread, tau_spread * 0.50)
        self.register_buffer("intrinsic_tau_soma", config.base_tau_soma * tau_spread)
        self.register_buffer("intrinsic_tau_dend", config.base_tau_dendrite * tau_spread)

        # Learnable intrinsic excitability bias (homeostatically regulated)
        self.intrinsic_bias = nn.Parameter(torch.zeros(n, dtype=dtype, device=self._device))

        # Dynamical state buffers
        self.register_buffer("v_soma", torch.full((n,), config.v_rest, dtype=dtype, device=self._device))
        self.register_buffer("v_dend", torch.full((n,), config.v_rest, dtype=dtype, device=self._device))
        self.register_buffer("g_ampa", torch.zeros(n, dtype=dtype, device=self._device))
        self.register_buffer("g_nmda", torch.zeros(n, dtype=dtype, device=self._device))
        self.register_buffer("g_gaba_a", torch.zeros(n, dtype=dtype, device=self._device))
        self.register_buffer("g_gaba_b", torch.zeros(n, dtype=dtype, device=self._device))
        self.register_buffer("g_adapt", torch.zeros(n, dtype=dtype, device=self._device))
        self.register_buffer("firing_trace", torch.zeros(n, dtype=dtype, device=self._device))
        self.register_buffer("last_spikes", torch.zeros(n, dtype=dtype, device=self._device))
        self.register_buffer("homeostatic_rate_ema", torch.full((n,), 0.08, dtype=dtype, device=self._device))

    def reset_state(self) -> None:
        cfg = self.config
        self.v_soma.fill_(cfg.v_rest)
        self.v_dend.fill_(cfg.v_rest)
        self.g_ampa.zero_()
        self.g_nmda.zero_()
        self.g_gaba_a.zero_()
        self.g_gaba_b.zero_()
        self.g_adapt.zero_()
        self.firing_trace.zero_()
        self.last_spikes.zero_()
        self.homeostatic_rate_ema.fill_(0.08)

    def detach_state_(self) -> None:
        """Detach stateful buffers across truncated BPTT boundaries while preserving values."""
        self.v_soma = self.v_soma.detach()
        self.v_dend = self.v_dend.detach()
        self.g_ampa = self.g_ampa.detach()
        self.g_nmda = self.g_nmda.detach()
        self.g_gaba_a = self.g_gaba_a.detach()
        self.g_gaba_b = self.g_gaba_b.detach()
        self.g_adapt = self.g_adapt.detach()
        self.firing_trace = self.firing_trace.detach()
        self.last_spikes = self.last_spikes.detach()

    @staticmethod
    def nmda_mg_unblock(v_membrane: torch.Tensor) -> torch.Tensor:
        """Jahr & Stevens (1990) voltage-dependent Mg2+ unblock of NMDA channels.

        ``v_membrane`` is in normalized Volts (e.g., -0.65 = -65 mV).
        """
        v_mv = v_membrane * 100.0
        return 1.0 / (1.0 + (1.2 / 3.57) * torch.exp(-0.062 * v_mv))

    def step(
        self,
        basal_exc_drive: torch.Tensor,
        basal_inh_drive: torch.Tensor,
        apical_exc_drive: torch.Tensor,
        apical_inh_drive: torch.Tensor,
        dt: float,
        *,
        d1_occupancy: float = 0.35,
        d2_occupancy: float = 0.35,
        ne_alpha2a: float = 0.40,
        ne_beta_alpha1: float = 0.15,
        ht1a_occupancy: float = 0.45,
        ht2a_occupancy: float = 0.20,
        ach_m1: float = 0.35,
        adenosine_a1: float = 0.10,
        brain_atp: float = 0.92,
        core_temp_celsius: float = 37.0,
        spontaneous_noise_std: float = 0.008,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate two-compartment conductance-based dynamics over ``dt`` seconds.

        Returns:
            ``(spikes, blended_activity)`` where ``spikes`` is binary surrogate-differentiable
            action potentials ``{0, 1}^N`` and ``blended_activity`` in ``[0, 1]^N`` combines
            filtered spike train + subthreshold dendritic-somatic plateau potential.
        """
        cfg = self.config

        # 1. Thermal Q10 scaling & Cerebral ATP Na+/K+-ATPase pump integrity
        q10 = float(cfg.q10_factor ** ((core_temp_celsius - 37.0) / 10.0))
        q10 = max(0.45, min(2.4, q10))

        # When cerebral ATP drops critically (< 0.55), Na+/K+ pumps fail -> anoxic depolarization block!
        pump_health = max(0.05, min(1.0, (brain_atp - 0.12) / 0.55))
        effective_v_rest = cfg.v_rest * pump_health + (-0.36) * (1.0 - pump_health)
        effective_v_reset = cfg.v_reset * pump_health + (-0.40) * (1.0 - pump_health)
        synaptic_atp_gate = 0.15 + 0.85 * pump_health

        # 2. Update AMPA, NMDA, GABA_A, GABA_B synaptic conductances
        a_ampa = min(1.0, (dt * q10) / cfg.tau_ampa)
        a_nmda = min(1.0, (dt * q10) / cfg.tau_nmda)
        a_gabaa = min(1.0, (dt * q10) / cfg.tau_gaba_a)
        a_gabab = min(1.0, (dt * q10) / cfg.tau_gaba_b)

        exc_total = F.relu(basal_exc_drive) * synaptic_atp_gate
        apical_exc = F.relu(apical_exc_drive) * synaptic_atp_gate
        inh_total = (F.relu(basal_inh_drive) + 0.65 * F.relu(apical_inh_drive)) * synaptic_atp_gate

        # D1 enhances NMDA persistent currents; Adenosine A1 presynaptically suppresses glutamate release!
        nmda_mod = (1.0 + 0.55 * d1_occupancy + 0.35 * ne_alpha2a) * max(0.35, 1.0 - 0.55 * adenosine_a1)
        ampa_mod = (1.0 + 0.30 * ht2a_occupancy + 0.25 * ne_beta_alpha1) * max(0.40, 1.0 - 0.45 * adenosine_a1)
        gaba_mod = 1.0 + 0.35 * d2_occupancy + 0.40 * ht1a_occupancy

        g_ampa_next = (1.0 - a_ampa) * self.g_ampa + a_ampa * ampa_mod * (exc_total + 0.35 * apical_exc)
        g_nmda_next = (1.0 - a_nmda) * self.g_nmda + a_nmda * nmda_mod * (0.45 * exc_total + 1.10 * apical_exc)
        g_gabaa_next = (1.0 - a_gabaa) * self.g_gaba_a + a_gabaa * gaba_mod * inh_total
        g_gabab_next = (1.0 - a_gabab) * self.g_gaba_b + a_gabab * (0.35 * inh_total + 0.25 * adenosine_a1 + 0.20 * ht1a_occupancy)

        # 3. Apical Dendritic Compartment ODE (nonlinear NMDA plateau potential!)
        mg_gate_dend = self.nmda_mg_unblock(self.v_dend)
        i_dend_nmda = g_nmda_next * mg_gate_dend * (cfg.e_rev_exc - self.v_dend)
        i_dend_ampa = 0.40 * g_ampa_next * (cfg.e_rev_exc - self.v_dend)
        i_dend_inh = 0.50 * g_gabaa_next * (cfg.e_rev_inh_a - self.v_dend)
        i_dend_leak = effective_v_rest - self.v_dend

        alpha_dend = torch.clamp((dt * q10) / self.intrinsic_tau_dend, 0.0, 0.85)
        v_dend_next = torch.clamp(
            self.v_dend + alpha_dend * (i_dend_leak + i_dend_ampa + i_dend_nmda + i_dend_inh),
            -0.95,
            0.10,
        )

        # 4. Somatic Liquid-Spiking Compartment ODE
        # Neuromodulator-warped liquid time constant: NE beta/alpha1 accelerates reactivity; alpha2A stabilizes!
        tau_warp = (1.0 + 0.45 * ne_alpha2a) / (1.0 + 0.85 * ne_beta_alpha1 + 0.25 * ht2a_occupancy)
        eff_tau_soma = self.intrinsic_tau_soma * tau_warp

        # Muscarinic M1 Acetylcholine suppresses K+ M-current adaptation -> sustained attention & firing!
        m1_adaptation_block = max(0.15, 1.0 - 0.72 * ach_m1)
        mg_gate_soma = self.nmda_mg_unblock(self.v_soma)

        i_leak = effective_v_rest - self.v_soma
        i_ampa = g_ampa_next * (cfg.e_rev_exc - self.v_soma)
        i_nmda = 0.55 * g_nmda_next * mg_gate_soma * (cfg.e_rev_exc - self.v_soma)
        i_gabaa = g_gabaa_next * (cfg.e_rev_inh_a - self.v_soma)
        i_gabab = g_gabab_next * (cfg.e_rev_inh_b - self.v_soma)
        i_adapt = (self.g_adapt * m1_adaptation_block) * (cfg.e_rev_inh_b - self.v_soma)
        i_axial = cfg.dendritic_coupling * (v_dend_next - self.v_soma)

        # Spontaneous ion-channel stochasticity (enables autonomous resting brainwaves in total darkness!)
        if not deterministic and spontaneous_noise_std > 0.0:
            noise_scale = spontaneous_noise_std * (1.0 + 0.65 * ht2a_occupancy + 0.55 * ne_beta_alpha1)
            i_noise = torch.randn_like(self.v_soma) * noise_scale
        else:
            pacemaker_spread = torch.linspace(-0.004, 0.006, steps=cfg.size, dtype=self.dtype, device=self._device)
            i_noise = pacemaker_spread * (1.0 - 0.4 * adenosine_a1) * pump_health

        total_somatic_drive = (
            i_leak + i_ampa + i_nmda + i_gabaa + i_gabab + i_adapt + i_axial + 0.02 * torch.tanh(self.intrinsic_bias) + i_noise
        )
        alpha_soma = torch.clamp((dt * q10) / eff_tau_soma, 0.0, 0.90)
        v_pre_spike = torch.clamp(self.v_soma + alpha_soma * total_somatic_drive, -0.96, 0.15)

        # Dynamic spike threshold; severe ATP pump failure (< 0.25) causes Na+-channel depolarization block!
        depol_block_shift = 0.25 * max(0.0, 0.30 - pump_health)
        eff_threshold = (
            cfg.v_threshold
            - 0.018 * d1_occupancy
            - 0.015 * ne_beta_alpha1
            + 0.016 * ht1a_occupancy
            + 0.024 * adenosine_a1
            + depol_block_shift
        )
        spikes = surrogate_spike(v_pre_spike - eff_threshold)

        # Soft refractory reset (respecting ATP-dependent K+ reversal potential)
        v_soma_next = v_pre_spike - spikes * (v_pre_spike - effective_v_reset)

        # 5. Update Ca2+-activated K+ spike-frequency adaptation & filtered firing rate trace
        a_adapt = min(1.0, (dt * q10) / cfg.tau_adaptation)
        g_adapt_next = (1.0 - a_adapt) * self.g_adapt + cfg.adaptation_jump * spikes

        a_trace = min(1.0, dt / cfg.tau_firing_trace)
        # Subthreshold voltage proximity + discrete spike trace gives rich continuous+spiking representation
        subthresh_rate = torch.sigmoid((v_pre_spike - (eff_threshold - 0.045)) * 28.0)
        inst_activity = 0.55 * spikes + 0.45 * subthresh_rate * pump_health
        trace_next = (1.0 - a_trace) * self.firing_trace + a_trace * inst_activity

        # Update buffers (keep autograd graph alive if parameters require grad during training, else clean)
        self.g_ampa = g_ampa_next
        self.g_nmda = g_nmda_next
        self.g_gaba_a = g_gabaa_next
        self.g_gaba_b = g_gabab_next
        self.v_dend = v_dend_next
        self.v_soma = v_soma_next
        self.g_adapt = g_adapt_next
        self.firing_trace = trace_next
        self.last_spikes = spikes

        with torch.no_grad():
            self.homeostatic_rate_ema.lerp_(trace_next.detach(), min(1.0, dt * 0.15))

        return spikes, trace_next


@dataclass(frozen=True)
class SynapseConfig:
    """Tsodyks-Markram short-term vesicle dynamics and three-factor STDP parameters."""

    use_stp: bool = True
    u0_baseline_release: float = 0.32       # Baseline vesicle release probability U0
    tau_facilitation: float = 0.450         # Residual presynaptic Ca2+ decay (450 ms synaptic working memory!)
    tau_recovery: float = 0.320             # Synaptic vesicle pool replenishment (320 ms)
    tau_eligibility: float = 1.80           # Synaptic eligibility tag time constant (1.8 s)
    plasticity_lr: float = 0.012            # Online three-factor weight update rate
    target_post_activity: float = 0.14      # Turrigiano homeostatic target rate
    homeostatic_scale_lr: float = 0.002     # Slow homeostatic synaptic scaling rate
    weight_max: float = 1.25


class DalesLawSynapse(nn.Module):
    """Dale's Law synaptic projection with Tsodyks-Markram STP & Three-Factor Eligibility STDP.

    Separates excitatory glutamatergic vs inhibitory GABAergic currents strictly according
    to the presynaptic neuron's ``pre_dales_sign`` (+1 = Glutamate, -1 = GABA).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        pre_dales_sign: torch.Tensor | None = None,
        init_scale: float = 0.45,
        sparsity: float = 0.25,
        config: SynapseConfig | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or SynapseConfig()
        self.dtype = dtype
        self._device = torch.device(device)

        if pre_dales_sign is None:
            sign = torch.ones(in_features, dtype=dtype, device=self._device)
        else:
            sign = pre_dales_sign.detach().clone().to(dtype=dtype, device=self._device)
        self.register_buffer("exc_mask", (sign > 0.0).to(dtype))
        self.register_buffer("inh_mask", (sign < 0.0).to(dtype))

        # Structural sparse connectome mask (biological circuits are sparse, not dense all-to-all!)
        conn_mask = (torch.rand(out_features, in_features, device=self._device) >= sparsity).to(dtype)
        # Guarantee every postsynaptic neuron receives at least 2 connections
        if in_features >= 2:
            conn_mask[:, :2] = 1.0
        self.register_buffer("structural_mask", conn_mask)

        # Non-negative synaptic efficacy magnitudes (Dale's Law enforced via softplus/relu!)
        fan_in = max(1.0, in_features * (1.0 - sparsity))
        raw_w = torch.rand(out_features, in_features, dtype=dtype, device=self._device) * (
            2.0 * init_scale / math.sqrt(fan_in)
        )
        self.weight_mag = nn.Parameter(raw_w * conn_mask)

        # Tsodyks-Markram Presynaptic Short-Term Plasticity buffers
        self.register_buffer(
            "u_rel",
            torch.full((in_features,), self.config.u0_baseline_release, dtype=dtype, device=self._device),
        )
        self.register_buffer("x_ves", torch.ones(in_features, dtype=dtype, device=self._device))

        # Three-Factor Synaptic Eligibility Trace e_{ij}(t) (shape: out_features x in_features)
        self.register_buffer(
            "eligibility_trace",
            torch.zeros(out_features, in_features, dtype=dtype, device=self._device),
        )
        self.register_buffer("pre_trace", torch.zeros(in_features, dtype=dtype, device=self._device))
        self.register_buffer("post_trace", torch.zeros(out_features, dtype=dtype, device=self._device))

    def reset_state(self) -> None:
        self.u_rel.fill_(self.config.u0_baseline_release)
        self.x_ves.fill_(1.0)
        self.eligibility_trace.zero_()
        self.pre_trace.zero_()
        self.post_trace.zero_()

    def detach_state_(self) -> None:
        self.u_rel = self.u_rel.detach()
        self.x_ves = self.x_ves.detach()
        self.eligibility_trace = self.eligibility_trace.detach()
        self.pre_trace = self.pre_trace.detach()
        self.post_trace = self.post_trace.detach()

    def effective_weights(self) -> torch.Tensor:
        """Strictly non-negative masked synaptic conductances ``W >= 0``."""
        return torch.clamp(F.relu(self.weight_mag), 0.0, self.config.weight_max) * self.structural_mask

    def forward_currents(
        self,
        pre_activity: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transmit presynaptic activity through Tsodyks-Markram vesicles into ``(I_exc, I_inh)``."""
        cfg = self.config
        pre_pos = F.relu(pre_activity)

        if cfg.use_stp:
            # Tsodyks-Markram presynaptic Ca2+ facilitation & vesicle depletion
            a_fac = min(1.0, dt / cfg.tau_facilitation)
            a_rec = min(1.0, dt / cfg.tau_recovery)
            u_next = self.u_rel + a_fac * (cfg.u0_baseline_release - self.u_rel) + (
                cfg.u0_baseline_release * (1.0 - self.u_rel) * pre_pos * min(1.0, dt * 8.0)
            )
            u_next = torch.clamp(u_next, 0.05, 0.95)
            released_frac = u_next * self.x_ves * pre_pos
            x_next = torch.clamp(
                self.x_ves + a_rec * (1.0 - self.x_ves) - released_frac * min(1.0, dt * 6.0),
                0.08,
                1.0,
            )
            self.u_rel = u_next
            self.x_ves = x_next
            # Normalized STP gain relative to baseline release probability
            effective_pre = released_frac / max(0.05, cfg.u0_baseline_release)
        else:
            effective_pre = pre_pos

        w_pos = self.effective_weights()
        exc_pre = effective_pre * self.exc_mask
        inh_pre = effective_pre * self.inh_mask

        i_exc = F.linear(exc_pre, w_pos)
        i_inh = F.linear(inh_pre, w_pos)
        return i_exc, i_inh

    @torch.no_grad()
    def update_eligibility_and_plasticity(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        neuromod_third_factor: float,
        ach_plasticity_gate: float,
        dt: float,
        enable_weight_update: bool = True,
    ) -> float:
        """Update STDP eligibility trace ``e_ij(t)`` and apply three-factor + homeostatic plasticity.

        ``neuromod_third_factor`` combines phasic reward/punishment prediction error (DA D1-D2)
        and noradrenergic surprise consolidation, gated by ``ach_plasticity_gate`` (M1/nAChR).
        """
        cfg = self.config
        a_pre = min(1.0, dt / 0.080)
        a_post = min(1.0, dt / 0.110)

        self.pre_trace.lerp_(F.relu(pre_activity.detach()), a_pre)
        self.post_trace.lerp_(F.relu(post_activity.detach()), a_post)

        # Asymmetric STDP + Bienenstock-Cooper-Munro (BCM) sliding threshold coincidence
        centered_post = post_activity.detach() - 0.55 * self.post_trace
        hebbian_coincidence = torch.outer(centered_post, self.pre_trace) * self.structural_mask

        a_elig = min(1.0, dt / cfg.tau_eligibility)
        self.eligibility_trace.mul_(1.0 - a_elig).add_(hebbian_coincidence * a_elig)
        self.eligibility_trace.clamp_(-1.0, 1.0)

        if not enable_weight_update:
            return 0.0

        # Third-factor neuromodulated weight consolidation + Turrigiano homeostatic scaling
        mod_gain = cfg.plasticity_lr * (0.25 + 0.75 * ach_plasticity_gate) * float(neuromod_third_factor)
        delta_three_factor = mod_gain * self.eligibility_trace * dt

        # Homeostatic scaling pulls excessively silent or hyperactive rows softly toward target rate
        rate_err = (cfg.target_post_activity - self.post_trace).unsqueeze(1)
        delta_homeo = cfg.homeostatic_scale_lr * rate_err * self.weight_mag.data * dt

        self.weight_mag.data.add_(delta_three_factor + delta_homeo)
        self.weight_mag.data.mul_(self.structural_mask)
        self.weight_mag.data.clamp_(0.0, cfg.weight_max)

        return float(delta_three_factor.abs().mean().item())
