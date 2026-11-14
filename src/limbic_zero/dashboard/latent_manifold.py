"""Scratch Real-Time 228D Neural State-Space Manifold & 80Hz LFP Spectral Analytics.

Implements:
1. ``OnlineNeuralManifoldProjector``: 100% PyTorch SVD subspace tracker with Procrustes
   sign-alignment (preventing eigenvector sign-flip jitter), Participation Ratio effective
   dimensionality, anatomical brain-region variance decomposition, and 228D phase velocity.
2. ``IntracranialLFPSpectralEngine``: Hann-windowed ``torch.fft.rfft`` spectral analyzer on
   80 Hz two-compartment pyramidal dipole Local Field Potentials (Buzsaki et al., 2012),
   extracting Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha/Beta (8-22 Hz), Low-Gamma (22-40 Hz),
   and Shannon spectral entropy—with zero synthetic waveforms or sklearn shortcuts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, Dict, List, Tuple

import torch


# Exact anatomical slice boundaries of LimbicBrain's 228D latent_affective_vector:
# [0:64] Sensorimotor, [64:128] Insula/dACC, [128:192] Prefrontal, [192:216] Amygdala, [216:228] Brainstem Neuromod
ANATOMICAL_SLICES: Tuple[Tuple[str, int, int], ...] = (
    ("sensorimotor", 0, 64),
    ("insula_dacc", 64, 128),
    ("prefrontal", 128, 192),
    ("amygdala", 192, 216),
    ("brainstem_neuromod", 216, 228),
)


@dataclass
class ManifoldSnapshot:
    """Real-time 3D state-space projection and population geometry metrics."""

    coords_3d: Dict[str, Tuple[float, float, float]]
    trails_3d: Dict[str, List[Tuple[float, float, float]]]
    phase_velocity: Dict[str, float]
    explained_variance_ratio: Tuple[float, float, float]
    participation_ratio_dim: float
    anatomical_subspace_shares: Dict[str, float]
    inter_organism_divergence_l2: float


class OnlineNeuralManifoldProjector:
    """Rolling SVD + Procrustes sign-locked 3D projector for 228D continuous mind vectors."""

    def __init__(
        self,
        latent_dim: int = 228,
        window_size: int = 240,
        trail_length: int = 90,
        recompute_svd_every: int = 2,
    ) -> None:
        self.latent_dim = latent_dim
        self.window_size = max(16, window_size)
        self.trail_length = max(10, trail_length)
        self.recompute_svd_every = max(1, recompute_svd_every)

        self._history_vectors: Deque[torch.Tensor] = deque(maxlen=self.window_size)
        self._prev_vectors: Dict[str, torch.Tensor] = {}
        self._agent_raw_trails: Dict[str, Deque[torch.Tensor]] = {}

        # Initialize deterministic orthonormal 3x228 basis so first frames project cleanly
        g = torch.Generator()
        g.manual_seed(202603)
        init_mat = torch.randn(self.latent_dim, 3, generator=g, dtype=torch.float32)
        q, _ = torch.linalg.qr(init_mat)
        self._basis_3d: torch.Tensor = q.T.contiguous()  # (3, 228)
        self._mean_vec: torch.Tensor = torch.zeros(self.latent_dim, dtype=torch.float32)
        self._explained_ratio: Tuple[float, float, float] = (0.34, 0.33, 0.33)
        self._participation_ratio: float = 1.0
        self._anatomical_shares: Dict[str, float] = {
            name: round(1.0 / len(ANATOMICAL_SLICES), 4) for name, _, _ in ANATOMICAL_SLICES
        }
        self._step_counter: int = 0

    def reset(self) -> None:
        self._history_vectors.clear()
        self._prev_vectors.clear()
        self._agent_raw_trails.clear()
        self._mean_vec.zero_()
        self._step_counter = 0

    def update(
        self,
        latent_vectors: Dict[str, torch.Tensor],
        dt: float = 0.05,
    ) -> ManifoldSnapshot:
        """Ingest one tick of 228D mind vectors across organisms and return 3D state-space metrics."""
        safe_dt = max(1e-4, float(dt))
        phase_vel: Dict[str, float] = {}
        clean_vecs: Dict[str, torch.Tensor] = {}

        for aid, vec in latent_vectors.items():
            v = vec.detach().to(dtype=torch.float32, device="cpu").flatten()
            if v.numel() != self.latent_dim:
                raise ValueError(f"Expected {self.latent_dim}D latent vector for {aid}, got {v.numel()}")
            v = torch.nan_to_num(v, nan=0.0, posinf=10.0, neginf=-10.0)
            clean_vecs[aid] = v
            self._history_vectors.append(v)

            if aid in self._prev_vectors:
                vel = float(torch.linalg.vector_norm(v - self._prev_vectors[aid]).item()) / safe_dt
            else:
                vel = 0.0
            phase_vel[aid] = round(vel, 4)
            self._prev_vectors[aid] = v

            if aid not in self._agent_raw_trails:
                self._agent_raw_trails[aid] = deque(maxlen=self.trail_length)
            self._agent_raw_trails[aid].append(v)

        self._step_counter += 1
        if len(self._history_vectors) >= 6 and (self._step_counter % self.recompute_svd_every == 0):
            self._recompute_subspace()

        # Project both current coordinates and trailing ribbons onto the sign-locked current 3D basis
        # so the entire ribbon lives in a coherent instantaneous coordinate frame!
        coords_3d: Dict[str, Tuple[float, float, float]] = {}
        trails_3d: Dict[str, List[Tuple[float, float, float]]] = {}

        for aid, v in clean_vecs.items():
            proj = torch.mv(self._basis_3d, v - self._mean_vec)
            coords_3d[aid] = (
                round(float(proj[0].item()), 4),
                round(float(proj[1].item()), 4),
                round(float(proj[2].item()), 4),
            )
            raw_trail = list(self._agent_raw_trails[aid])
            mat = torch.stack(raw_trail, dim=0) - self._mean_vec.unsqueeze(0)  # (T, 228)
            trail_proj = mat @ self._basis_3d.T  # (T, 3)
            trails_3d[aid] = [
                (
                    round(float(row[0].item()), 4),
                    round(float(row[1].item()), 4),
                    round(float(row[2].item()), 4),
                )
                for row in trail_proj
            ]

        aids = sorted(clean_vecs.keys())
        if len(aids) >= 2:
            div_l2 = float(torch.linalg.vector_norm(clean_vecs[aids[0]] - clean_vecs[aids[1]]).item())
        else:
            div_l2 = 0.0

        return ManifoldSnapshot(
            coords_3d=coords_3d,
            trails_3d=trails_3d,
            phase_velocity=phase_vel,
            explained_variance_ratio=self._explained_ratio,
            participation_ratio_dim=round(self._participation_ratio, 3),
            anatomical_subspace_shares=dict(self._anatomical_shares),
            inter_organism_divergence_l2=round(div_l2, 4),
        )

    def _recompute_subspace(self) -> None:
        """Compute exact centered SVD with continuous Procrustes sign/column alignment."""
        data = torch.stack(list(self._history_vectors), dim=0)  # (M, 228)
        mean_now = data.mean(dim=0)
        centered = data - mean_now.unsqueeze(0)

        total_energy = float(centered.pow(2).sum().item())
        if total_energy < 1e-9:
            self._mean_vec = mean_now
            return

        # Economy SVD: centered = U @ diag(S) @ Vh, rows of Vh[:3] are principal directions in R^228
        _, singular_vals, vh = torch.linalg.svd(centered, full_matrices=False)
        eigenvals = singular_vals.pow(2)
        sum_eig = float(eigenvals.sum().item())
        sum_eig_sq = float(eigenvals.pow(2).sum().item())

        if sum_eig_sq > 1e-12:
            self._participation_ratio = (sum_eig * sum_eig) / sum_eig_sq

        k = min(3, vh.shape[0])
        new_basis = self._basis_3d.clone()
        ev_list = [0.0, 0.0, 0.0]

        for i in range(k):
            cand = vh[i].clone()
            # Procrustes sign lock: preserve positive inner product with previous axis i
            if float(torch.dot(cand, self._basis_3d[i]).item()) < 0.0:
                cand.neg_()
            # Smooth exponential blend (0.35) when eigenvalues are close to prevent sudden 90-deg swaps
            blended = 0.72 * self._basis_3d[i] + 0.28 * cand
            norm_b = float(torch.linalg.vector_norm(blended).item())
            new_basis[i] = blended / max(1e-8, norm_b)
            ev_list[i] = round(float(eigenvals[i].item()) / max(1e-12, sum_eig), 4)

        # Re-orthonormalize via modified Gram-Schmidt so axes remain strictly orthogonal!
        v0 = new_basis[0] / torch.linalg.vector_norm(new_basis[0]).clamp(min=1e-8)
        v1 = new_basis[1] - torch.dot(new_basis[1], v0) * v0
        v1 = v1 / torch.linalg.vector_norm(v1).clamp(min=1e-8)
        v2 = new_basis[2] - torch.dot(new_basis[2], v0) * v0 - torch.dot(new_basis[2], v1) * v1
        v2 = v2 / torch.linalg.vector_norm(v2).clamp(min=1e-8)

        self._basis_3d = torch.stack([v0, v1, v2], dim=0)
        self._mean_vec = 0.80 * self._mean_vec + 0.20 * mean_now
        self._explained_ratio = (ev_list[0], ev_list[1], ev_list[2])

        # Compute anatomical variance contribution weighted by explained variance of top 3 modes
        weights = torch.tensor(
            [max(1e-4, ev_list[0]), max(1e-4, ev_list[1]), max(1e-4, ev_list[2])],
            dtype=torch.float32,
        )
        weights = weights / weights.sum()
        weighted_sq = (self._basis_3d.pow(2) * weights.unsqueeze(1)).sum(dim=0)  # (228,)
        denom = max(1e-9, float(weighted_sq.sum().item()))
        self._anatomical_shares = {
            name: round(float(weighted_sq[s:e].sum().item()) / denom, 4)
            for name, s, e in ANATOMICAL_SLICES
        }


@dataclass
class LFPSpectralSnapshot:
    """Biophysical 80 Hz LFP trace tail and real-time FFT band-power decomposition."""

    sampling_hz: float
    recent_sm_mv: List[float]
    recent_ins_mv: List[float]
    recent_pfc_mv: List[float]
    delta_power: float          # 0.5 - 4.0 Hz
    theta_power: float          # 4.0 - 8.0 Hz
    alpha_beta_power: float     # 8.0 - 22.0 Hz
    low_gamma_power: float      # 22.0 - 40.0 Hz
    band_shares: Dict[str, float]
    spectral_entropy: float     # Normalized Shannon entropy in [0, 1]; 0.0 on isoelectric flatline


class IntracranialLFPSpectralEngine:
    """Hann-windowed real FFT analyzer over 80 Hz pyramidal dipole LFP streams."""

    def __init__(self, buffer_samples: int = 256, oscilloscope_tail: int = 120) -> None:
        self.buffer_samples = max(64, buffer_samples)
        self.oscilloscope_tail = max(32, oscilloscope_tail)
        self._buffers: Dict[str, Deque[Tuple[float, float, float]]] = {}
        self._sampling_hz: Dict[str, float] = {}

    def reset(self) -> None:
        self._buffers.clear()
        self._sampling_hz.clear()

    def push_and_analyze(
        self,
        agent_id: str,
        substep_lfp_dipole: torch.Tensor,
        substep_dt: float,
    ) -> LFPSpectralSnapshot:
        """Append new ODE substep LFP samples (mV) and compute Hann-windowed FFT band powers."""
        fs = 1.0 / max(1e-4, float(substep_dt))
        self._sampling_hz[agent_id] = fs
        if agent_id not in self._buffers:
            self._buffers[agent_id] = deque(maxlen=self.buffer_samples)

        buf = self._buffers[agent_id]
        arr = substep_lfp_dipole.detach().to(dtype=torch.float32, device="cpu")
        if arr.ndim == 1:
            arr = arr.unsqueeze(0)
        for row in arr:
            buf.append((float(row[0].item()), float(row[1].item()), float(row[2].item())))

        tail = list(buf)[-self.oscilloscope_tail :]
        sm_tail = [round(x[0], 4) for x in tail]
        ins_tail = [round(x[1], 4) for x in tail]
        pfc_tail = [round(x[2], 4) for x in tail]

        n = len(buf)
        if n < 16:
            return LFPSpectralSnapshot(
                sampling_hz=round(fs, 2),
                recent_sm_mv=sm_tail,
                recent_ins_mv=ins_tail,
                recent_pfc_mv=pfc_tail,
                delta_power=0.0,
                theta_power=0.0,
                alpha_beta_power=0.0,
                low_gamma_power=0.0,
                band_shares={"delta": 0.25, "theta": 0.25, "alpha_beta": 0.25, "low_gamma": 0.25},
                spectral_entropy=0.0,
            )

        # Composite cortical macro-electrode LFP = mean dipole across all 3 columns
        sig = torch.tensor(
            [(x[0] + x[1] + x[2]) / 3.0 for x in buf],
            dtype=torch.float32,
        )
        sig_centered = sig - sig.mean()
        variance = float(sig_centered.pow(2).mean().item())
        if variance < 1e-10:
            # Clinical isoelectric EEG flatline!
            return LFPSpectralSnapshot(
                sampling_hz=round(fs, 2),
                recent_sm_mv=sm_tail,
                recent_ins_mv=ins_tail,
                recent_pfc_mv=pfc_tail,
                delta_power=0.0,
                theta_power=0.0,
                alpha_beta_power=0.0,
                low_gamma_power=0.0,
                band_shares={"delta": 0.0, "theta": 0.0, "alpha_beta": 0.0, "low_gamma": 0.0},
                spectral_entropy=0.0,
            )

        win = torch.hann_window(n, dtype=torch.float32)
        win_norm = float(win.pow(2).sum().item()) * fs
        spec = torch.fft.rfft(sig_centered * win)
        psd = spec.abs().pow(2) / max(1e-9, win_norm)
        freqs = torch.fft.rfftfreq(n, d=1.0 / fs)

        def _band_power(f_lo: float, f_hi: float) -> float:
            mask = (freqs >= f_lo) & (freqs < f_hi)
            if not bool(mask.any()):
                return 0.0
            df = float(freqs[1].item() - freqs[0].item()) if freqs.numel() > 1 else 1.0
            return float(psd[mask].sum().item()) * df

        p_delta = _band_power(0.5, 4.0)
        p_theta = _band_power(4.0, 8.0)
        p_ab = _band_power(8.0, 22.0)
        p_gamma = _band_power(22.0, min(40.0, 0.5 * fs + 0.01))
        p_tot = p_delta + p_theta + p_ab + p_gamma

        if p_tot > 1e-12:
            shares = {
                "delta": round(p_delta / p_tot, 4),
                "theta": round(p_theta / p_tot, 4),
                "alpha_beta": round(p_ab / p_tot, 4),
                "low_gamma": round(p_gamma / p_tot, 4),
            }
        else:
            shares = {"delta": 0.0, "theta": 0.0, "alpha_beta": 0.0, "low_gamma": 0.0}

        # Normalized Shannon spectral entropy across positive AC bins (0.5 Hz .. Nyquist)
        ac_mask = freqs >= 0.5
        ac_psd = psd[ac_mask]
        ac_sum = float(ac_psd.sum().item())
        if ac_sum > 1e-12 and ac_psd.numel() > 1:
            probs = (ac_psd / ac_sum).clamp(min=1e-12)
            ent = -float((probs * probs.log()).sum().item()) / math.log(float(ac_psd.numel()))
            ent = max(0.0, min(1.0, ent))
        else:
            ent = 0.0

        return LFPSpectralSnapshot(
            sampling_hz=round(fs, 2),
            recent_sm_mv=sm_tail,
            recent_ins_mv=ins_tail,
            recent_pfc_mv=pfc_tail,
            delta_power=round(p_delta, 6),
            theta_power=round(p_theta, 6),
            alpha_beta_power=round(p_ab, 6),
            low_gamma_power=round(p_gamma, 6),
            band_shares=shares,
            spectral_entropy=round(ent, 4),
        )
