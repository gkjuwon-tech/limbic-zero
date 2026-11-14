"""Modality-Specific Thalamic Relays, GABAergic TRN Gate, & Active-Inference World Model.

Anatomically routes the 414D raw nerve bundle from Stage 1 WITHOUT omniscient cheats:
    - [0:297]   Foveated 33x9 Retina       -> Lateral Geniculate Nucleus (LGN)
    - [297:305] Binaural Doppler Cochlea   -> Medial Geniculate Nucleus (MGN)
    - [305:313] Birhinal Stereo Nostrils   -> Piriform / Mediodorsal Olfactory Relay
    - [313:328] Cutaneous + Vestibular     -> Ventral Posterolateral Nucleus (VPL)
    - [328:414] Ascending Visceral Nerves  -> Posterior Ventromedial / Insular Relay (VMpo)

Includes:
    1. ``ThalamicReticularGate`` (TRN): Competitive surround-inhibition shell modulated by
       top-down pallial attention, nicotinic ACh sensory precision, LC arousal, and
       Adenosine A1 thalamocortical sleep gating.
    2. ``PredictiveWorldModel``: Action-conditioned forward generative model predicting next
       multimodal thalamic state ``z_hat(t + dt)`` from efference copy ``a(t)`` and pallial
       context, decomposing prediction errors into:
       - ``unexpected_surprise`` (sudden variance-normalized jump -> drives Locus Coeruleus NE)
       - ``epistemic_novelty`` (learnable exteroceptive prediction error -> drives NBM ACh)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


VISION_SLICE = slice(0, 297)       # 33 rays * 9 channels
AUDITORY_SLICE = slice(297, 305)   # 8D binaural Doppler cochlea
OLFACTORY_SLICE = slice(305, 313)  # 8D birhinal stereo nostrils
SOMATO_SLICE = slice(313, 328)     # 8D tactile + 7D vestibular/proprioception
INTERO_SLICE = slice(328, 414)     # 86D spinothalamic + vagal NTS visceral fibers
TOTAL_RAW_SENSORY_DIM = 414


@dataclass(frozen=True)
class ThalamicConfig:
    """Latent widths of the five anatomical thalamic nuclei and world-model learning rate."""

    lgn_visual_dim: int = 36
    mgn_auditory_dim: int = 12
    olfactory_dim: int = 12
    vpl_somato_dim: int = 16
    vmpo_intero_dim: int = 28
    world_model_lr: float = 0.018
    surprise_ema_tau: float = 1.40

    @property
    def total_thalamic_dim(self) -> int:
        return (
            self.lgn_visual_dim
            + self.mgn_auditory_dim
            + self.olfactory_dim
            + self.vpl_somato_dim
            + self.vmpo_intero_dim
        )  # Default: 104D multimodal thalamic representation

    @property
    def extero_thalamic_dim(self) -> int:
        return self.total_thalamic_dim - self.vmpo_intero_dim  # 76D exteroceptive relays


@dataclass
class ThalamicStepOutput:
    """Gated thalamic representations and decomposed Active-Inference prediction errors."""

    gated_thalamic_latent: torch.Tensor     # (104,) TRN-gated multimodal thalamic output
    extero_cue_vector: torch.Tensor         # (76,) pure exteroceptive cue for Amygdala/Hippocampus
    intero_thalamic_vector: torch.Tensor    # (28,) VMpo visceral representation
    modality_gates: torch.Tensor            # (5,) TRN attentional transmission gates [vis, aud, olf, som, int]
    prediction_error_vec: torch.Tensor      # (104,) z_thal(t) - z_hat(t)
    unexpected_surprise: float              # Variance-normalized jump in error (drives LC NE!)
    epistemic_novelty: float                # Sustained exteroceptive error reduction potential (drives NBM ACh!)
    world_model_mse: float                  # Raw forward prediction MSE


class ThalamicSensoryComplex(nn.Module):
    """Five-nucleus thalamic relay + GABAergic TRN gate + Action-Conditioned World Predictor."""

    def __init__(
        self,
        pallial_context_dim: int,
        action_dim: int = 7,
        config: ThalamicConfig | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.config = config or ThalamicConfig()
        self.pallial_context_dim = pallial_context_dim
        self.action_dim = action_dim
        self.dtype = dtype
        self._device = torch.device(device)
        cfg = self.config

        # 1. Modality-specific thalamic relay projections (retinotopic/tonotopic/viscerotopic)
        # Split 33 retinal rays into Left Periphery (11x9=99), Central Fovea (11x9=99), Right Periphery (11x9=99)
        foveal_sub = cfg.lgn_visual_dim // 3
        self.lgn_left = nn.Linear(99, foveal_sub, dtype=dtype, device=self._device)
        self.lgn_fovea = nn.Linear(99, cfg.lgn_visual_dim - 2 * foveal_sub, dtype=dtype, device=self._device)
        self.lgn_right = nn.Linear(99, foveal_sub, dtype=dtype, device=self._device)

        self.mgn_auditory = nn.Linear(8, cfg.mgn_auditory_dim, dtype=dtype, device=self._device)
        self.olfactory_relay = nn.Linear(8, cfg.olfactory_dim, dtype=dtype, device=self._device)
        self.vpl_somato = nn.Linear(15, cfg.vpl_somato_dim, dtype=dtype, device=self._device)
        self.vmpo_intero = nn.Linear(86, cfg.vmpo_intero_dim, dtype=dtype, device=self._device)

        # 2. GABAergic Thalamic Reticular Nucleus (TRN) top-down attentional gate over the 5 relays
        self.trn_attention_proj = nn.Linear(pallial_context_dim, 5, dtype=dtype, device=self._device)
        nn.init.zeros_(self.trn_attention_proj.bias)

        # 3. Efference-Copy Forward Generative World Model: predicts next 104D thalamic state
        wm_in_dim = cfg.total_thalamic_dim + pallial_context_dim + action_dim
        self.world_pred_hidden = nn.Linear(wm_in_dim, 96, dtype=dtype, device=self._device)
        self.world_pred_out = nn.Linear(96, cfg.total_thalamic_dim, dtype=dtype, device=self._device)
        nn.init.normal_(self.world_pred_out.weight, mean=0.0, std=0.03)
        nn.init.zeros_(self.world_pred_out.bias)

        # Stateful buffers for predictive coding & novelty decomposition
        self.register_buffer(
            "predicted_next_thalamic",
            torch.zeros(cfg.total_thalamic_dim, dtype=dtype, device=self._device),
        )
        self.register_buffer("last_wm_hidden", torch.zeros(96, dtype=dtype, device=self._device))
        self.register_buffer("expected_error_ema", torch.tensor(0.08, dtype=dtype, device=self._device))
        self.register_buffer(
            "extero_error_fast_ema",
            torch.zeros(cfg.extero_thalamic_dim, dtype=dtype, device=self._device),
        )
        self.register_buffer(
            "extero_error_slow_ema",
            torch.zeros(cfg.extero_thalamic_dim, dtype=dtype, device=self._device),
        )
        self.register_buffer("spindle_phase", torch.tensor(0.0, dtype=dtype, device=self._device))
        self._has_prediction: bool = False

    def reset_state(self) -> None:
        self.predicted_next_thalamic.zero_()
        self.last_wm_hidden.zero_()
        self.expected_error_ema.fill_(0.08)
        self.extero_error_fast_ema.zero_()
        self.extero_error_slow_ema.zero_()
        self.spindle_phase.zero_()
        self._has_prediction = False

    def detach_state_(self) -> None:
        self.predicted_next_thalamic = self.predicted_next_thalamic.detach()
        self.last_wm_hidden = self.last_wm_hidden.detach()

    def step(
        self,
        raw_sensory_414d: torch.Tensor,
        pallial_context: torch.Tensor,
        prev_action_7d: torch.Tensor,
        dt: float,
        *,
        ach_nicotinic: float = 0.30,
        ne_arousal: float = 0.25,
        adenosine_a1: float = 0.10,
        somatic_pain_salience: float = 0.0,
        enable_world_model_learning: bool = True,
    ) -> ThalamicStepOutput:
        """Transduce 414D nerves through LGN/MGN/Olf/VPL/VMpo, TRN gate, & World Model."""
        cfg = self.config
        x = raw_sensory_414d.to(dtype=self.dtype, device=self._device)

        # 1. Encode each anatomical sensory pathway
        vis_raw = x[VISION_SLICE]
        vis_latent = torch.tanh(
            torch.cat(
                [
                    self.lgn_left(vis_raw[:99]),
                    self.lgn_fovea(vis_raw[99:198]),
                    self.lgn_right(vis_raw[198:297]),
                ],
                dim=0,
            )
        )
        aud_latent = torch.tanh(self.mgn_auditory(x[AUDITORY_SLICE]))
        olf_latent = torch.tanh(self.olfactory_relay(x[OLFACTORY_SLICE]))
        som_latent = torch.tanh(self.vpl_somato(x[SOMATO_SLICE]))
        int_latent = torch.tanh(self.vmpo_intero(x[INTERO_SLICE]))

        raw_thalamic = torch.cat([vis_latent, aud_latent, olf_latent, som_latent, int_latent], dim=0)

        # 2. Compute Active-Inference Forward Prediction Error BEFORE updating predictor
        if not self._has_prediction:
            pred_err = torch.zeros_like(raw_thalamic)
            self._has_prediction = True
        else:
            pred_err = raw_thalamic - self.predicted_next_thalamic

        wm_mse = float((pred_err * pred_err).mean().item())
        extero_dim = cfg.extero_thalamic_dim
        extero_err = pred_err[:extero_dim]
        extero_err_norm = float(torch.linalg.vector_norm(extero_err).item()) / (extero_dim ** 0.5)

        # Online delta-rule predictive coding learning on the forward world model!
        if enable_world_model_learning and self._has_prediction:
            with torch.no_grad():
                lr = (cfg.world_model_lr * 3.5) * (0.35 + 0.65 * ach_nicotinic) * min(1.0, dt * 12.0)
                err_det = pred_err.detach()
                self.world_pred_out.weight.data.add_(lr * torch.outer(err_det, self.last_wm_hidden.detach()))
                self.world_pred_out.bias.data.add_(lr * err_det)

        # Decompose Prediction Error into:
        # (A) Unexpected Surprise (Yu & Dayan): sudden ratio spike above expected baseline error -> LC NE!
        baseline_err = max(0.015, float(self.expected_error_ema.item()))
        inst_total_err = float(pred_err.abs().mean().item())
        unexpected_surprise = max(0.0, min(2.0, (inst_total_err - baseline_err) / baseline_err))

        with torch.no_grad():
            a_surp = min(1.0, dt / cfg.surprise_ema_tau)
            self.expected_error_ema.lerp_(torch.tensor(inst_total_err, dtype=self.dtype, device=self._device), a_surp)

            # (B) Epistemic Novelty (Schmidhuber / Friston epistemic value):
            # Active, non-nociceptive exteroceptive dynamics where fast error differs from slow static noise!
            self.extero_error_fast_ema.lerp_(extero_err.detach().abs(), min(1.0, dt / 0.18))
            self.extero_error_slow_ema.lerp_(extero_err.detach().abs(), min(1.0, dt / 2.20))
            dynamic_change = float((self.extero_error_fast_ema - 0.55 * self.extero_error_slow_ema).clamp(min=0.0).mean().item())
            # Pain/emergency suppresses playful curiosity; safety allows epistemic exploration!
            safety_curiosity_gate = max(0.05, 1.0 - 1.25 * somatic_pain_salience)
            epistemic_novelty = max(0.0, min(1.8, (1.8 * dynamic_change + 0.8 * extero_err_norm) * safety_curiosity_gate))

        # 3. Thalamic Reticular Nucleus (TRN) Competitive Modality Gating
        # Nicotinic ACh boosts bottom-up sensory gain; Adenosine A1 induces rhythmic sleep-spindle attenuation!
        self.spindle_phase.add_(dt * 11.0 * 3.14159265)  # ~5.5 Hz theta/spindle oscillation phase
        sleep_gate = 1.0 - 0.55 * max(0.0, adenosine_a1 - 0.45) * (1.0 - ne_arousal) * (
            0.5 + 0.5 * float(torch.sin(self.spindle_phase).item())
        )

        raw_trn_logits = self.trn_attention_proj(pallial_context)
        # Competitive surround normalization + baseline nicotinic sensory fidelity
        centered_logits = raw_trn_logits - raw_trn_logits.mean()
        # Note: Visceral VMpo pain/emergency always breaks through TRN sleep gating!
        base_precision = 0.55 + 0.45 * ach_nicotinic + 0.20 * ne_arousal
        modality_gates = torch.sigmoid(centered_logits + base_precision) * sleep_gate
        # Guarantee VMpo interoceptive emergency bypass when pain/visceral alarm is high
        vmpo_gate = torch.clamp(modality_gates[4] + 0.55 * somatic_pain_salience, 0.20, 1.0)
        modality_gates = torch.stack(
            [modality_gates[0], modality_gates[1], modality_gates[2], modality_gates[3], vmpo_gate]
        )

        gated_vis = vis_latent * modality_gates[0]
        gated_aud = aud_latent * modality_gates[1]
        gated_olf = olf_latent * modality_gates[2]
        gated_som = som_latent * modality_gates[3]
        gated_int = int_latent * modality_gates[4]

        # Precision-weighted predictive coding boosts unpredicted sensory residuals!
        residual_boost = 0.35 * (0.4 + 0.6 * ach_nicotinic) * pred_err
        gated_thalamic = torch.clamp(
            torch.cat([gated_vis, gated_aud, gated_olf, gated_som, gated_int], dim=0) + residual_boost,
            -1.5,
            1.5,
        )

        # 4. Generate Efference-Copy Forward Prediction for t + dt
        wm_input = torch.cat(
            [
                gated_thalamic,
                pallial_context,
                prev_action_7d.to(dtype=self.dtype, device=self._device),
            ],
            dim=0,
        )
        wm_h = torch.tanh(self.world_pred_hidden(wm_input))
        self.last_wm_hidden = wm_h
        self.predicted_next_thalamic = torch.tanh(self.world_pred_out(wm_h))

        extero_cue = torch.cat([gated_vis, gated_aud, gated_olf, gated_som], dim=0)
        return ThalamicStepOutput(
            gated_thalamic_latent=gated_thalamic,
            extero_cue_vector=extero_cue,
            intero_thalamic_vector=gated_int,
            modality_gates=modality_gates,
            prediction_error_vec=pred_err,
            unexpected_surprise=unexpected_surprise,
            epistemic_novelty=epistemic_novelty,
            world_model_mse=wm_mse,
        )
