"""Unified Continuous-Time Limbic-Pallial-Striatal Brain ('LimbicBrain') & Neuro-Telemetry.

Connects all six scratch-built neurobiological subsystems into a single living,
24/7 ticking pre-linguistic nervous system coupled bidirectionally to Stage 1's
``DigitalSoma`` and ``TrumanDomeWorld``:
    - Modality-Specific Thalamus + TRN Gate + Active-Inference World Model
    - Basolateral / ITC Extinction / Central Amygdala Circuit
    - Trisynaptic Hippocampus (DG -> CA3 -> CA1) with Sharp-Wave Ripple Replay
    - Subcortical Brainstem Nuclei (VTA/LHb, LC, DRN, NBM) & Receptor Subtypes
    - Multi-Timescale 3-Column Liquid-Spiking Pallium (Sensorimotor, Insula/dACC, Prefrontal)
    - Cortico-Basal-Ganglia-Thalamic Action Selector (D1-Go, D2-NoGo, STN, PAG)

Exposes ``latent_affective_state_vector`` (228D continuous mind + neuromodulator + amygdala
state) for Stage 3 Affective fMRI visualization and Stage 5 LLM Residual-Stream Vocal Cord
injection—with ZERO hardcoded emotion labels or Transformer shortcuts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from limbic_zero.core.soma import DigitalSoma
from limbic_zero.mind.amygdala_hippocampus import (
    AmygdalaCircuit,
    AmygdalaOutput,
    HippocampalSomaticMemory,
    HippocampalStepOutput,
)
from limbic_zero.mind.basal_ganglia import BasalGangliaActionSelector, BasalGangliaOutput
from limbic_zero.mind.liquid_pallium import (
    MultiTimescaleLiquidPallium,
    PalliumConfig,
    PalliumStepOutput,
)
from limbic_zero.mind.neuromodulators import (
    NeuromodulatorDriveInputs,
    NeuromodulatorState,
    SubcorticalNeuromodulatorEngine,
)
from limbic_zero.mind.thalamus_sensory import (
    AUDITORY_SLICE,
    TOTAL_RAW_SENSORY_DIM,
    ThalamicConfig,
    ThalamicSensoryComplex,
    ThalamicStepOutput,
)
from limbic_zero.sandbox.world import OrganismObservation


@dataclass(frozen=True)
class BrainConfig:
    """Architectural configuration for ``LimbicBrain``."""

    thalamic: ThalamicConfig = ThalamicConfig()
    pallium: PalliumConfig = PalliumConfig()
    amygdala_embed_dim: int = 24
    hippocampus_dg_dim: int = 256
    hippocampus_max_engrams: int = 128
    striatal_hidden_dim: int = 48
    enable_online_plasticity: bool = True


@dataclass
class MindDiagnostics:
    """Complete Intracranial EEG, Neuromodulator, Memory, & Basal-Ganglia fMRI Snapshot."""

    is_conscious_active: bool
    sim_time: float

    # Subsystem step outputs
    thalamus: ThalamicStepOutput
    amygdala: AmygdalaOutput
    hippocampus: HippocampalStepOutput
    pallium: PalliumStepOutput
    basal_ganglia: BasalGangliaOutput
    neuromodulators: NeuromodulatorState

    # Differentiable value & continuous latent bridge vector (228D) for Stage 3/4/5
    critic_value_tensor: torch.Tensor
    latent_affective_vector: torch.Tensor   # (228,) [192 pallium + 24 amygdala + 12 neuromod/intero]


class LimbicBrain(nn.Module):
    """Autonomous continuous-time biological mind substrate for an ``EmbodiedOrganism``."""

    def __init__(
        self,
        config: BrainConfig | None = None,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.config = config or BrainConfig()
        self.dtype = dtype
        self._device = torch.device(device)
        cfg = self.config

        pallial_dim = cfg.pallium.total_pallial_dim          # 192
        extero_dim = cfg.thalamic.extero_thalamic_dim        # 76
        intero_dim = cfg.thalamic.vmpo_intero_dim            # 28
        amyg_dim = cfg.amygdala_embed_dim                    # 24

        self.thalamus = ThalamicSensoryComplex(
            pallial_context_dim=pallial_dim,
            action_dim=7,
            config=cfg.thalamic,
            dtype=dtype,
            device=self._device,
        )
        self.amygdala = AmygdalaCircuit(
            extero_cue_dim=extero_dim,
            pallial_dim=pallial_dim,
            embed_dim=amyg_dim,
            dtype=dtype,
            device=self._device,
        )
        self.hippocampus = HippocampalSomaticMemory(
            extero_cue_dim=extero_dim,
            context_dim=pallial_dim,
            dg_dim=cfg.hippocampus_dg_dim,
            max_engrams=cfg.hippocampus_max_engrams,
            dtype=dtype,
            device=self._device,
        )
        self.pallium = MultiTimescaleLiquidPallium(
            thalamic_extero_dim=extero_dim,
            thalamic_intero_dim=intero_dim,
            amygdala_embed_dim=amyg_dim,
            hippocampal_ctx_dim=pallial_dim,
            config=cfg.pallium,
            dtype=dtype,
            device=self._device,
        )
        self.neuromod_engine = SubcorticalNeuromodulatorEngine(
            latent_dim=pallial_dim + amyg_dim,
            dtype=dtype,
            device=self._device,
        )
        self.basal_ganglia = BasalGangliaActionSelector(
            pallial_dim=pallial_dim,
            amygdala_dim=amyg_dim,
            extero_cue_dim=extero_dim,
            striatal_hidden=cfg.striatal_hidden_dim,
            dtype=dtype,
            device=self._device,
        )

        self.register_buffer("prev_action_7d", torch.zeros(7, dtype=dtype, device=self._device))
        self.last_diagnostics: MindDiagnostics | None = None
        self.total_steps: int = 0

    @property
    def latent_bridge_dim(self) -> int:
        """Dimension of continuous mind state exposed to Stage 5 Vocal Cord Bridge (228D)."""
        return self.config.pallium.total_pallial_dim + self.config.amygdala_embed_dim + 12

    def reset_state(self) -> None:
        """Reset transient dynamical potentials & working memory while keeping learned weights."""
        self.thalamus.reset_state()
        self.hippocampus.reset_state()
        self.pallium.reset_state()
        self.neuromod_engine.reset_state()
        self.basal_ganglia.reset_state()
        self.prev_action_7d.zero_()
        self.last_diagnostics = None
        self.total_steps = 0

    def detach_state_(self) -> None:
        """Detach recurrent ODE & STP state buffers across truncated BPTT boundaries."""
        self.thalamus.detach_state_()
        self.pallium.detach_state_()
        self.basal_ganglia.detach_state_()
        self.prev_action_7d = self.prev_action_7d.detach()

    def step(
        self,
        obs: OrganismObservation,
        soma: DigitalSoma,
        dt: float = 0.05,
        *,
        deterministic: bool = False,
        enable_plasticity: bool | None = None,
    ) -> Tuple[Dict[str, float], MindDiagnostics]:
        """Advance the entire continuous-time mind by ``dt`` seconds and emit 7D motor commands."""
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        plastic = self.config.enable_online_plasticity if enable_plasticity is None else enable_plasticity
        raw_414 = obs.combined_sensory_tensor.to(dtype=self.dtype, device=self._device)
        if raw_414.numel() != TOTAL_RAW_SENSORY_DIM:
            raise ValueError(f"Expected {TOTAL_RAW_SENSORY_DIM}D sensory nerve tensor, got {raw_414.shape}")

        intero = obs.interoception
        nm_prev = self.neuromod_engine.state
        prev_workspace = self.pallium.current_workspace

        # Handle biological death -> Isoelectric EEG flatline!
        if not soma.is_alive:
            zero_act = {
                "forward_thrust": 0.0,
                "angular_torque": 0.0,
                "radiator_venting": 0.0,
                "vocalization_amplitude": 0.0,
                "vocalization_roughness": 0.0,
                "interaction_pulse": 0.0,
                "grasp_hold": 0.0,
                "neural_firing_load": 0.0,
            }
            self.pallium.sensorimotor_col.firing_trace.mul_(0.5)
            self.pallium.insula_cingulate_col.firing_trace.mul_(0.5)
            self.pallium.prefrontal_slow_col.firing_trace.mul_(0.5)
            if self.last_diagnostics is not None:
                self.last_diagnostics.is_conscious_active = False
                self.last_diagnostics.pallium.substep_lfp_dipole.zero_()
                return zero_act, self.last_diagnostics

        # 1. Thalamic Sensory Relays, TRN Gate, & Forward Active-Inference World Model
        thal_out = self.thalamus.step(
            raw_sensory_414d=raw_414,
            pallial_context=prev_workspace,
            prev_action_7d=self.prev_action_7d,
            dt=dt,
            ach_nicotinic=nm_prev.ach_nicotinic_occupancy,
            ne_arousal=nm_prev.ne_beta_alpha1_occupancy + 0.5 * nm_prev.ne_alpha2a_occupancy,
            adenosine_a1=nm_prev.adenosine_a1_occupancy,
            somatic_pain_salience=intero.somatic_pain_salience,
            enable_world_model_learning=plastic,
        )

        # 2. Hippocampal DG->CA3->CA1 Episodic Somatic Memory & Autonomous SWR Replay
        visceral_trauma_now = min(
            1.5,
            intero.somatic_pain_salience
            + intero.suffocation_alarm_salience
            + 2.5 * intero.rail_ripple_intensity,
        )
        motor_quiescence = max(
            0.0,
            1.0 - float(self.prev_action_7d[:2].abs().mean().item()) * 1.4,
        )
        hippo_out = self.hippocampus.step(
            sim_time=obs.sim_time,
            extero_cue=thal_out.extero_cue_vector,
            pallial_context=prev_workspace,
            prev_action_7d=self.prev_action_7d,
            dt=dt,
            consolidation_salience_tag=nm_prev.consolidation_salience_tag,
            valence_relief=intero.homeostatic_valence,
            visceral_trauma=visceral_trauma_now,
            cea_threat=float(self.amygdala.cea_activity_ema.item()),
            epistemic_novelty=thal_out.epistemic_novelty,
            ach_m1_encoding_gate=nm_prev.ach_muscarinic_m1_occupancy,
            adenosine_a1_sleep=nm_prev.adenosine_a1_occupancy,
            motor_quiescence=motor_quiescence,
        )

        # If CA3 fired a Sharp-Wave Ripple (SWR) replay burst during quiet rest, blend the
        # remembered sensory cue into the Amygdala & Pallium (true internal flashback/rumination!)
        if hippo_out.swr_replay_active and hippo_out.swr_replay_cue is not None:
            replay_w = min(0.65, 0.45 * hippo_out.swr_replay_trauma_intensity)
            effective_cue = (1.0 - replay_w) * thal_out.extero_cue_vector + replay_w * hippo_out.swr_replay_cue
        else:
            effective_cue = thal_out.extero_cue_vector

        # Extract acoustic consonance/roughness from binaural cochlear nerves with vocal corollary discharge
        # (Eliades & Wang 2008: motor efference copy attenuates self-generated vocal roughness!)
        ears = raw_414[AUDITORY_SLICE]
        cochlear_harmonic = float((0.5 * (ears[1] + ears[5])).clamp(min=0.0).item())
        cochlear_rough = float((0.5 * (ears[2] + ears[6])).clamp(min=0.0).item())
        self_vocal_rough_copy = float((self.prev_action_7d[3] * self.prev_action_7d[4]).clamp(0.0, 1.0).item())
        exogenous_rough = max(0.0, cochlear_rough - 0.90 * self_vocal_rough_copy)

        aversive_us = min(
            1.5,
            0.90 * intero.somatic_pain_salience
            + 0.95 * intero.suffocation_alarm_salience
            + 2.20 * intero.rail_ripple_intensity
            + 0.65 * exogenous_rough,
        )
        appetitive_us = min(
            1.5,
            max(0.0, intero.homeostatic_valence) * 0.90 + 0.55 * cochlear_harmonic,
        )

        # 3. Basolateral / ITC Extinction / Central Amygdala Conditioning
        amyg_out = self.amygdala.step(
            extero_cue=effective_cue,
            pallial_state=prev_workspace,
            dt=dt,
            aversive_us_reinforcer=aversive_us,
            appetitive_us_reinforcer=appetitive_us,
            ne_beta_occupancy=nm_prev.ne_beta_alpha1_occupancy,
            d1_occupancy=nm_prev.d1_occupancy,
            ht1a_occupancy=nm_prev.ht1a_occupancy,
            systemic_cortisol=soma.endo.adrenal_cortisol,
            enable_plasticity=plastic,
        )

        # 4. Subcortical Neuromodulator Nuclei (VTA/LHb, LC, DRN, NBM) & Critic TD Update
        critic_input = torch.cat([prev_workspace, amyg_out.amygdala_embedding], dim=0)
        nm_state, v_tensor = self.neuromod_engine.step(
            mind_latent=critic_input,
            inp=NeuromodulatorDriveInputs(
                homeostatic_valence=intero.homeostatic_valence,
                total_free_energy=intero.total_drive,
                somatic_pain_salience=intero.somatic_pain_salience,
                suffocation_salience=intero.suffocation_alarm_salience,
                rail_ripple_intensity=intero.rail_ripple_intensity,
                unexpected_world_surprise=thal_out.unexpected_surprise,
                epistemic_novelty=thal_out.epistemic_novelty,
                conditioned_threat_cea=amyg_out.cea_defensive_output,
                conditioned_appetitive_bla=amyg_out.bla_appetitive_raw,
                peripheral_epinephrine=soma.endo.plasma_epinephrine,
                systemic_cortisol=soma.endo.adrenal_cortisol,
                systemic_oxytocin=soma.endo.oxytocin_systemic,
                brain_adenosine=soma.metab.brain_adenosine,
                ventral_vagal_tone=soma.endo.ventral_vagal_tone,
            ),
            dt=dt,
            enable_critic_learning=plastic,
        )

        # 5. Multi-Timescale 3-Column Continuous-Time Liquid-Spiking Pallium
        pall_out = self.pallium.step(
            extero_thalamic=effective_cue,
            intero_thalamic=thal_out.intero_thalamic_vector,
            amygdala_embed=amyg_out.amygdala_embedding,
            hippo_ca1_ctx=hippo_out.ca1_retrieved_context,
            recalled_somatic_marker=hippo_out.recalled_somatic_marker,
            neuromod=nm_state,
            dt=dt,
            brain_atp=soma.metab.brain_atp,
            core_temp_celsius=soma.thermo.core_temp_celsius,
            enable_plasticity=plastic,
            deterministic=deterministic,
        )

        # 6. Cortico-Basal-Ganglia-Thalamic + PAG Survival Action Selection
        core_temp_excess = soma.thermo.core_temp_celsius - soma.thermo.hypothalamic_setpoint_celsius
        bg_out = self.basal_ganglia.step(
            pallial_workspace=pall_out.workspace_vector,
            amygdala_embed=amyg_out.amygdala_embedding,
            extero_cue=thal_out.extero_cue_vector,
            recalled_action_bias=hippo_out.recalled_action_bias,
            neuromod=nm_state,
            dt=dt,
            dacc_conflict=pall_out.dacc_conflict_signal,
            cea_threat=amyg_out.cea_defensive_output,
            somatic_pain=intero.somatic_pain_salience,
            suffocation_alarm=intero.suffocation_alarm_salience,
            rail_ripple=intero.rail_ripple_intensity,
            core_temp_excess_c=core_temp_excess,
            dorsal_vagal_tone=soma.endo.dorsal_vagal_tone,
            pallial_mean_firing=pall_out.mean_firing_rate_norm,
            enable_plasticity=plastic,
            deterministic=deterministic,
        )

        self.prev_action_7d = bg_out.action_tensor_7d.detach()
        self.total_steps += 1

        # Construct continuous 228D latent mind vector for Stage 3 fMRI & Stage 5 LLM Bridge
        nm_summary = torch.tensor(
            [
                nm_state.dopamine_extracellular,
                nm_state.d1_occupancy,
                nm_state.d2_occupancy,
                nm_state.norepinephrine_extracellular,
                nm_state.ne_alpha2a_occupancy,
                nm_state.ne_beta_alpha1_occupancy,
                nm_state.serotonin_extracellular,
                nm_state.ht1a_occupancy,
                nm_state.acetylcholine_extracellular,
                nm_state.ach_muscarinic_m1_occupancy,
                nm_state.adenosine_a1_occupancy,
                pall_out.dacc_conflict_signal,
            ],
            dtype=self.dtype,
            device=self._device,
        )
        latent_228 = torch.cat(
            [pall_out.workspace_vector, amyg_out.amygdala_embedding, nm_summary],
            dim=0,
        )

        diag = MindDiagnostics(
            is_conscious_active=soma.is_alive,
            sim_time=obs.sim_time,
            thalamus=thal_out,
            amygdala=amyg_out,
            hippocampus=hippo_out,
            pallium=pall_out,
            basal_ganglia=bg_out,
            neuromodulators=nm_state,
            critic_value_tensor=v_tensor,
            latent_affective_vector=latent_228,
        )
        self.last_diagnostics = diag
        return bg_out.raw_action_dict, diag
