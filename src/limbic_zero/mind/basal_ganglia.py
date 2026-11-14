"""Cortico-Basal-Ganglia-Thalamic Action Selection (D1 Go, D2 NoGo, STN Brake) & PAG Survival Columns.

Zero-Scripted-Behavior Motor Architecture:
    Maps continuous pallial, amygdalar, and hippocampal representations onto the 7 physical
    muscle degrees of freedom of ``DigitalSoma`` & ``TrumanDomeWorld`` via authentic vertebrate
    basal ganglia + brainstem selection dynamics:
        1. Striatal ``D1`` Direct Pathway ('Go'): Facilitated by phasic D1 occupancy;
           undergoes positive TD reinforcement LTP and negative TD LTD.
        2. Striatal ``D2`` Indirect Pathway ('NoGo'): Inhibited by tonic D2 occupancy;
           disinhibited during LHb dopamine dips -> vetoes dangerous/futile actions!
        3. Subthalamic Nucleus (``STN``) Hyperdirect Pathway ('Global Conflict Brake'):
           Driven by dACC conflict + LC surprise + 5-HT1A patience; transiently raises
           the GPi/SNr gate threshold before action release (collapses when 5-HT1A is depleted
           -> impulsive lashing out!).
        4. Periaqueductal Gray (``PAG``) Brainstem Columns (``dlPAG`` Active Escape/Defense vs
           ``vlPAG`` Passive Immobility + Laryngeal Distress/Rough Vocalization + Hypothalamic
           Thermoregulatory Venting Drive).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from limbic_zero.mind.neuromodulators import NeuromodulatorState


ACTUATOR_KEYS = (
    "forward_thrust",          # [-1.0, 1.0]
    "angular_torque",          # [-1.0, 1.0]
    "radiator_venting",        # [0.0, 1.0]
    "vocalization_amplitude",  # [0.0, 1.0]
    "vocalization_roughness",  # [0.0, 1.0]
    "interaction_pulse",       # [0.0, 1.0]
    "grasp_hold",              # [0.0, 1.0]
)
NUM_MOTOR_CHANNELS = 7


@dataclass
class BasalGangliaOutput:
    """Selected 7D continuous motor commands, neural metabolic burn, and pathway diagnostics."""

    action_tensor_7d: torch.Tensor          # (7,) continuous somatic muscle activations
    raw_action_dict: Dict[str, float]       # Ready to feed directly into TrumanDomeWorld.step()!
    d1_go_activity: torch.Tensor            # (7,) channel-wise Direct Go drive
    d2_nogo_activity: torch.Tensor          # (7,) channel-wise Indirect NoGo suppression
    stn_hyperdirect_brake: float            # Global Subthalamic conflict/hesitation brake
    gpi_tonic_inhibition: torch.Tensor      # (7,) Globus Pallidus Internus output
    dlpag_active_defense: float             # [0, 1] Brainstem active fight/flight surge
    vlpag_conservation_freeze: float        # [0, 1] Brainstem passive freeze immobility
    neural_firing_load: float               # [0.02, 1.50] Real whole-brain ATP demand fed to DigitalSoma!


class BasalGangliaActionSelector(nn.Module):
    """Striatal D1-Go / D2-NoGo + STN Hyperdirect + GPi/SNr + PAG Brainstem Motor Controller."""

    def __init__(
        self,
        pallial_dim: int = 192,
        amygdala_dim: int = 24,
        extero_cue_dim: int = 76,
        striatal_hidden: int = 48,
        plasticity_lr: float = 0.022,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.in_dim = pallial_dim + amygdala_dim + extero_cue_dim
        self.striatal_hidden = striatal_hidden
        self.plasticity_lr = plasticity_lr
        self.dtype = dtype
        self._device = torch.device(device)

        # Corticostriatal feature projection into Dorsomedial / Dorsolateral Striatum
        self.cortico_striatal_enc = nn.Linear(self.in_dim, striatal_hidden, dtype=dtype, device=self._device)

        # Plastic Striatopallidal Readout Synapses for the 7 motor channels (D1 Go vs D2 NoGo)
        # Signed channels (forward_thrust, angular_torque) use bidirectional push-pull subchannels (9 total -> 7)
        self.num_subchannels = 9  # [fwd+, fwd-, turn_L, turn_R, vent, voc_amp, voc_rough, interact, grasp]

        self.w_d1_go = nn.Parameter(
            torch.randn(self.num_subchannels, striatal_hidden, dtype=dtype, device=self._device) * 0.14
        )
        self.b_d1_go = nn.Parameter(torch.full((self.num_subchannels,), 0.12, dtype=dtype, device=self._device))

        self.w_d2_nogo = nn.Parameter(
            torch.randn(self.num_subchannels, striatal_hidden, dtype=dtype, device=self._device) * 0.10
        )
        self.b_d2_nogo = nn.Parameter(torch.full((self.num_subchannels,), 0.05, dtype=dtype, device=self._device))

        # Subthalamic Nucleus (STN) hyperdirect projection from prefrontal/cingulate
        self.stn_proj = nn.Linear(pallial_dim, 1, dtype=dtype, device=self._device)
        nn.init.zeros_(self.stn_proj.bias)

        # Superior Colliculus / Brainstem orienting projection (unlearned innate retinotopic/binaural orienting seed
        # that plasticity can freely override or refine!)
        self.colliculus_orient = nn.Linear(extero_cue_dim, 4, dtype=dtype, device=self._device)
        nn.init.normal_(self.colliculus_orient.weight, mean=0.0, std=0.08)
        nn.init.zeros_(self.colliculus_orient.bias)

        # Striatal synaptic eligibility traces for Reynolds & Wickens (2002) opposing D1/D2 TD learning
        self.register_buffer(
            "d1_eligibility",
            torch.zeros(self.num_subchannels, striatal_hidden, dtype=dtype, device=self._device),
        )
        self.register_buffer(
            "d2_eligibility",
            torch.zeros(self.num_subchannels, striatal_hidden, dtype=dtype, device=self._device),
        )
        self.register_buffer("motor_smooth_ema", torch.zeros(7, dtype=dtype, device=self._device))
        self.register_buffer("ou_exploration_state", torch.zeros(self.num_subchannels, dtype=dtype, device=self._device))

    def reset_state(self) -> None:
        self.d1_eligibility.zero_()
        self.d2_eligibility.zero_()
        self.motor_smooth_ema.zero_()
        self.ou_exploration_state.zero_()

    def detach_state_(self) -> None:
        self.d1_eligibility = self.d1_eligibility.detach()
        self.d2_eligibility = self.d2_eligibility.detach()
        self.motor_smooth_ema = self.motor_smooth_ema.detach()
        self.ou_exploration_state = self.ou_exploration_state.detach()

    def step(
        self,
        pallial_workspace: torch.Tensor,
        amygdala_embed: torch.Tensor,
        extero_cue: torch.Tensor,
        recalled_action_bias: torch.Tensor,
        neuromod: NeuromodulatorState,
        dt: float,
        *,
        dacc_conflict: float = 0.0,
        cea_threat: float = 0.0,
        somatic_pain: float = 0.0,
        suffocation_alarm: float = 0.0,
        rail_ripple: float = 0.0,
        core_temp_excess_c: float = 0.0,
        dorsal_vagal_tone: float = 0.0,
        pallial_mean_firing: float = 0.12,
        enable_plasticity: bool = True,
        deterministic: bool = False,
    ) -> BasalGangliaOutput:
        """Select 7D somatic action via D1/D2/STN/GPi competition + PAG survival columns."""
        joint_in = torch.cat([pallial_workspace, amygdala_embed, extero_cue], dim=0)
        striatal_feat = torch.tanh(self.cortico_striatal_enc(joint_in))
        striatal_pos = F.relu(striatal_feat) + 0.15 * torch.sigmoid(striatal_feat)

        # 1. Ornstein-Uhlenbeck Motor Exploration Variability (scaled by ACh curiosity + 5-HT2A + NE)
        if not deterministic:
            a_ou = min(1.0, dt / 0.35)
            explore_amp = (
                0.14
                + 0.24 * neuromod.ach_muscarinic_m1_occupancy
                + 0.20 * neuromod.ht2a_occupancy
                + 0.15 * neuromod.ne_beta_alpha1_occupancy
            )
            noise = torch.randn_like(self.ou_exploration_state) * explore_amp
            self.ou_exploration_state = (1.0 - a_ou) * self.ou_exploration_state + (a_ou ** 0.5) * noise
            ou_drive = self.ou_exploration_state
        else:
            ou_drive = torch.zeros_like(self.ou_exploration_state)

        # 2. Direct D1 ('Go') vs Indirect D2 ('NoGo') Medium Spiny Neuron Activation
        # D1 (Gs) amplifies Go excitability; D2 (Gi) suppresses NoGo until DA dips!
        d1_gain = 0.65 + 1.35 * neuromod.d1_occupancy
        d2_suppression = 1.0 / (1.0 + 1.45 * neuromod.d2_occupancy)

        raw_go = F.linear(striatal_pos, self.w_d1_go, self.b_d1_go) + ou_drive
        raw_nogo = F.linear(striatal_pos, self.w_d2_nogo, self.b_d2_nogo)

        go_9 = F.softplus(raw_go) * d1_gain
        nogo_9 = F.softplus(raw_nogo) * (0.45 + 1.35 * d2_suppression)

        # 3. Subthalamic Nucleus (STN) Hyperdirect Global Conflict Brake
        # Strengthened by dACC conflict & 5-HT1A impulse control; weakened when 5-HT1A collapses!
        pfc_stn = float(torch.sigmoid(self.stn_proj(pallial_workspace)).squeeze(-1).detach().item())
        impulse_restraint = 0.35 + 0.85 * neuromod.ht1a_occupancy
        stn_brake = min(
            1.2,
            (0.35 * dacc_conflict + 0.22 * pfc_stn) * impulse_restraint * (1.0 - 0.45 * neuromod.d1_occupancy),
        )

        # 4. Periaqueductal Gray (PAG) Brainstem Survival Columns (dlPAG Active vs vlPAG Freeze)
        acute_emergency = (
            0.70 * cea_threat
            + 0.85 * somatic_pain
            + 0.95 * suffocation_alarm
            + 1.65 * rail_ripple
        )
        # High 5-HT1A restrains excessive dlPAG rage/panic bursts; depleted 5-HT1A disinhibits dlPAG!
        pag_disinhibition = max(0.35, 1.25 - 0.70 * neuromod.ht1a_occupancy)
        dlpag_active = max(
            0.0,
            min(1.0, math.tanh(acute_emergency * pag_disinhibition) * (1.0 - 0.75 * dorsal_vagal_tone)),
        )
        vlpag_freeze = max(
            0.0,
            min(1.0, dorsal_vagal_tone * (0.55 + 0.45 * neuromod.adenosine_a1_occupancy)),
        )

        # 5. Globus Pallidus Internus (GPi / SNr) Tonic Inhibition & Thalamic Motor Gate
        # GPi baseline tonic inhibition is ~0.35; Go disinhibits GPi (< 0.35 -> motor release!)
        gpi_9 = F.relu(0.32 + 0.75 * nogo_9 + stn_brake - 0.95 * go_9)
        thalamic_release_9 = torch.tanh(F.relu(go_9 - 0.65 * nogo_9 - 0.55 * stn_brake))

        # Superior colliculus unconditioned orienting reflex modulation
        collic = torch.tanh(self.colliculus_orient(extero_cue))

        # Map 9 push-pull striato-thalamic subchannels + hippocampal habit bias + PAG onto 7 somatic muscles
        freeze_damp = max(0.08, 1.0 - 0.88 * vlpag_freeze)
        active_boost = 1.0 + 0.65 * dlpag_active

        fwd = (
            (thalamic_release_9[0] - 0.75 * thalamic_release_9[1] + 0.20 * collic[0] + 0.25 * recalled_action_bias[0])
            * active_boost
            * freeze_damp
        )
        turn = (
            (thalamic_release_9[2] - thalamic_release_9[3] + 0.25 * collic[1] + 0.25 * recalled_action_bias[1])
            * active_boost
            * freeze_damp
        )

        # Hypothalamic-medullary thermoregulatory panting reflex blends smoothly with voluntary striatal venting
        reflex_vent = max(0.0, min(1.0, core_temp_excess_c / 1.2))
        vent = torch.clamp(0.75 * thalamic_release_9[4] + 0.65 * reflex_vent + 0.15 * recalled_action_bias[2], 0.0, 1.0)

        # Laryngeal vocalization: dlPAG emergency or social/striatal drive triggers calls; emergency roughens timbre!
        voc_amp = torch.clamp(
            (0.80 * thalamic_release_9[5] + 0.65 * dlpag_active + 0.15 * recalled_action_bias[3]) * freeze_damp,
            0.0,
            1.0,
        )
        voc_rough = torch.clamp(
            0.55 * thalamic_release_9[6]
            + 0.70 * dlpag_active
            + 0.35 * neuromod.ne_beta_alpha1_occupancy
            - 0.45 * neuromod.ht1a_occupancy,
            0.0,
            1.0,
        )

        # Physical interaction / shoving / probing pulse (amplified by ACh curiosity OR disinhibited dlPAG!)
        interact = torch.clamp(
            (
                0.80 * thalamic_release_9[7]
                + 0.30 * neuromod.ach_muscarinic_m1_occupancy * F.relu(collic[2])
                + 0.45 * dlpag_active * F.relu(collic[3])
                + 0.20 * recalled_action_bias[5]
            )
            * freeze_damp,
            0.0,
            1.0,
        )
        grasp = torch.clamp(
            (0.85 * thalamic_release_9[8] + 0.25 * recalled_action_bias[6]) * freeze_damp,
            0.0,
            1.0,
        )

        inst_action_7d = torch.stack(
            [
                torch.clamp(fwd, -1.0, 1.0),
                torch.clamp(turn, -1.0, 1.0),
                vent,
                voc_amp,
                voc_rough,
                interact,
                grasp,
            ]
        )

        # Neuromuscular smoothing filter (~45ms motor unit recruitment tau)
        a_mot = min(1.0, dt / 0.045)
        smooth_7d = (1.0 - a_mot) * self.motor_smooth_ema + a_mot * inst_action_7d
        self.motor_smooth_ema = smooth_7d

        # 6. Online Corticostriatal Opposing D1 / D2 Dopamine Reinforcement Plasticity (Reynolds & Wickens, 2002)
        if enable_plasticity:
            with torch.no_grad():
                a_el = min(1.0, dt / 1.40)
                pre_d = striatal_pos.detach()
                go_act = torch.tanh(go_9.detach())
                nogo_act = torch.tanh(nogo_9.detach())

                self.d1_eligibility.lerp_(torch.outer(go_act, pre_d), a_el)
                self.d2_eligibility.lerp_(torch.outer(nogo_act, pre_d), a_el)

                td = neuromod.td_prediction_error
                lr = self.plasticity_lr * (0.35 + 0.65 * neuromod.ach_muscarinic_m1_occupancy) * dt
                # Positive TD strengthens D1-Go & weakens D2-NoGo; Negative TD strengthens D2-NoGo & weakens D1-Go!
                self.w_d1_go.data.add_(lr * td * self.d1_eligibility)
                self.w_d2_nogo.data.add_(-lr * td * self.d2_eligibility)

                # Soft synaptic weight decay prevents unbounded drift
                self.w_d1_go.data.mul_(1.0 - 0.0015 * dt).clamp_(-1.5, 1.5)
                self.w_d2_nogo.data.mul_(1.0 - 0.0015 * dt).clamp_(-1.5, 1.5)

        # 7. Compute Emergent Whole-Brain Metabolic ATP Demand ('neural_firing_load' -> DigitalSoma!)
        striatal_effort = float((go_9.mean() + nogo_9.mean()).detach().item()) * 0.18
        arousal_burn = 0.28 * neuromod.ne_beta_alpha1_occupancy + 0.18 * neuromod.ach_muscarinic_m1_occupancy
        sleep_saving = 0.25 * neuromod.adenosine_a1_occupancy
        neural_load = max(0.03, min(1.50, 1.85 * pallial_mean_firing + striatal_effort + arousal_burn - sleep_saving))

        go_7 = torch.stack([go_9[0] + go_9[1], go_9[2] + go_9[3], go_9[4], go_9[5], go_9[6], go_9[7], go_9[8]])
        nogo_7 = torch.stack(
            [nogo_9[0] + nogo_9[1], nogo_9[2] + nogo_9[3], nogo_9[4], nogo_9[5], nogo_9[6], nogo_9[7], nogo_9[8]]
        )
        gpi_7 = torch.stack([gpi_9[0] + gpi_9[1], gpi_9[2] + gpi_9[3], gpi_9[4], gpi_9[5], gpi_9[6], gpi_9[7], gpi_9[8]])

        raw_dict = {k: float(smooth_7d[i].detach().item()) for i, k in enumerate(ACTUATOR_KEYS)}
        raw_dict["neural_firing_load"] = neural_load

        return BasalGangliaOutput(
            action_tensor_7d=smooth_7d,
            raw_action_dict=raw_dict,
            d1_go_activity=go_7.detach(),
            d2_nogo_activity=nogo_7.detach(),
            stn_hyperdirect_brake=stn_brake,
            gpi_tonic_inhibition=gpi_7.detach(),
            dlpag_active_defense=dlpag_active,
            vlpag_conservation_freeze=vlpag_freeze,
            neural_firing_load=neural_load,
        )


import math  # noqa: E402
