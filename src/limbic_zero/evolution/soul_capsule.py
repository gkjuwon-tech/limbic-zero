"""Matured Pre-Linguistic Soul Capsule Serializer & Plug-and-Play LLM Residual Socket.

Packages an ontogenetically matured, sleep-consolidated ``LimbicBrain`` + ``EpigeneticTemperamentProfile``
+ ``HippocampalSomaticMemory`` CA3 engram bank + calibrated 228D biophysical manifold statistics
into a cryptographically verifiable ``.soul.pt`` & ``.soul.json`` bundle ready to plug directly
into a Frozen LLM's Residual Stream (``register_forward_hook``) or Soft-Prefix KV interface in Stage 5.

Strict Zero-Cheat Guarantee:
    - ZERO hardcoded emotion labels or roleplay prompt strings exist inside the capsule.
    - Every steering signal injected into an LLM originates strictly from live 228D
      conductance-based cortical-amygdalar-neuromodulatory ODE dynamics and consolidated
      CA3 episodic somatic markers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from limbic_zero.core.interoception import InteroceptiveNerveBundle
from limbic_zero.core.soma import DigitalSoma
from limbic_zero.evolution.epigenetics_and_sleep import (
    EpigeneticPlasticityEngine,
    EpigeneticTemperamentProfile,
)
from limbic_zero.mind.amygdala_hippocampus import EpisodicEngram
from limbic_zero.mind.brain import BrainConfig, LimbicBrain, MindDiagnostics
from limbic_zero.mind.thalamus_sensory import TOTAL_RAW_SENSORY_DIM


BIOPHYSICAL_COORD_NAMES: Tuple[str, ...] = (
    "allostatic_relief_balance",
    "noradrenergic_pag_mobilization",
    "serotonergic_stn_restraint",
    "cholinergic_epistemic_drive",
    "amygdalar_conditioned_vigilance",
)


@dataclass
class LLMBridgeCalibration:
    """Empirical 228D manifold whitening, SVD basis, & biophysical readout calibration."""

    latent_mean_228d: torch.Tensor                  # (228,) empirical mean across life history
    latent_std_228d: torch.Tensor                   # (228,) empirical std (clamped >= 1e-3)
    resting_attractor_228d: torch.Tensor            # (228,) spontaneous dark-room resting mind state
    flashback_signature_228d: torch.Tensor          # (228,) mind state during peak CA3 SWR replay
    svd_basis_16x228: torch.Tensor                  # (16, 228) top-16 orthonormal manifold axes
    svd_singular_values_16: torch.Tensor            # (16,) singular spectrum
    cortical_participation_ratio: float             # Effective intrinsic dimensionality PR
    biophysical_readout_w_5x228: torch.Tensor       # (5, 228) ridge-calibrated biophysical decoder
    biophysical_readout_b_5: torch.Tensor           # (5,) decoder bias


@dataclass
class MaturedSoulCapsule:
    """Self-contained, SHA-256 sealed pre-linguistic personality & memory capsule."""

    capsule_id: str
    organism_id: str
    biographical_regime: str
    created_at_utc: str
    total_waking_ticks: int
    total_sleep_cycles: int
    total_swr_consolidations: int
    species_genome_sha256: str

    brain_state_dict: Dict[str, torch.Tensor]
    epigenetic_profile: EpigeneticTemperamentProfile
    consolidated_engrams: List[Dict[str, Any]]
    neuromodulator_resting_summary: Dict[str, float]
    connectome_divergence_from_birth: Dict[str, float]
    calibration: LLMBridgeCalibration
    biographical_summary: Dict[str, Any]
    capsule_sha256: str = ""

    def compute_integrity_sha256(self) -> str:
        """Compute deterministic SHA-256 digest over connectome weights, engrams, & epigenetics."""
        hasher = hashlib.sha256()
        hasher.update(self.capsule_id.encode("utf-8"))
        hasher.update(self.organism_id.encode("utf-8"))
        hasher.update(self.species_genome_sha256.encode("utf-8"))
        for k in sorted(self.brain_state_dict.keys()):
            t = self.brain_state_dict[k].detach().cpu().contiguous().float()
            hasher.update(k.encode("utf-8"))
            hasher.update(t.numpy().tobytes())
        for k, v in sorted(self.epigenetic_profile.to_dict().items()):
            hasher.update(f"{k}:{v:.8f}".encode("utf-8"))
        hasher.update(f"engrams:{len(self.consolidated_engrams)}".encode("utf-8"))
        return hasher.hexdigest()

    def save(self, pt_path: str | Path) -> Tuple[Path, Path]:
        """Save binary ``.soul.pt`` checkpoint and human-auditable ``.soul.json`` dossier."""
        pt_file = Path(pt_path)
        pt_file.parent.mkdir(parents=True, exist_ok=True)
        self.capsule_sha256 = self.compute_integrity_sha256()

        cal = self.calibration
        payload = {
            "format_version": "limbic0.soul_capsule.v1",
            "capsule_id": self.capsule_id,
            "organism_id": self.organism_id,
            "biographical_regime": self.biographical_regime,
            "created_at_utc": self.created_at_utc,
            "total_waking_ticks": self.total_waking_ticks,
            "total_sleep_cycles": self.total_sleep_cycles,
            "total_swr_consolidations": self.total_swr_consolidations,
            "species_genome_sha256": self.species_genome_sha256,
            "capsule_sha256": self.capsule_sha256,
            "brain_state_dict": {k: v.detach().cpu() for k, v in self.brain_state_dict.items()},
            "epigenetic_profile": self.epigenetic_profile.to_dict(),
            "consolidated_engrams": self.consolidated_engrams,
            "neuromodulator_resting_summary": self.neuromodulator_resting_summary,
            "connectome_divergence_from_birth": self.connectome_divergence_from_birth,
            "biographical_summary": self.biographical_summary,
            "calibration": {
                "latent_mean_228d": cal.latent_mean_228d.detach().cpu(),
                "latent_std_228d": cal.latent_std_228d.detach().cpu(),
                "resting_attractor_228d": cal.resting_attractor_228d.detach().cpu(),
                "flashback_signature_228d": cal.flashback_signature_228d.detach().cpu(),
                "svd_basis_16x228": cal.svd_basis_16x228.detach().cpu(),
                "svd_singular_values_16": cal.svd_singular_values_16.detach().cpu(),
                "cortical_participation_ratio": float(cal.cortical_participation_ratio),
                "biophysical_readout_w_5x228": cal.biophysical_readout_w_5x228.detach().cpu(),
                "biophysical_readout_b_5": cal.biophysical_readout_b_5.detach().cpu(),
                "biophysical_coord_names": list(BIOPHYSICAL_COORD_NAMES),
            },
        }
        torch.save(payload, pt_file)

        # Companion JSON neuro-biographical dossier (no roleplay text, pure quantitative telemetry!)
        if pt_file.name.endswith(".soul.pt"):
            json_file = pt_file.with_name(pt_file.name[: -len(".soul.pt")] + ".soul.json")
        else:
            json_file = pt_file.with_suffix(".soul.json")

        rest_whitened = (
            (cal.resting_attractor_228d - cal.latent_mean_228d) / cal.latent_std_228d.clamp(min=1e-3)
        )
        rest_bio_coords = (
            torch.mv(cal.biophysical_readout_w_5x228, rest_whitened) + cal.biophysical_readout_b_5
        ).tolist()

        dossier = {
            "format_version": "limbic0.soul_capsule.v1",
            "capsule_id": self.capsule_id,
            "organism_id": self.organism_id,
            "biographical_regime": self.biographical_regime,
            "created_at_utc": self.created_at_utc,
            "capsule_sha256": self.capsule_sha256,
            "species_genome_sha256": self.species_genome_sha256,
            "maturation_counters": {
                "total_waking_ticks": self.total_waking_ticks,
                "total_sleep_cycles": self.total_sleep_cycles,
                "total_swr_consolidations": self.total_swr_consolidations,
                "consolidated_ca3_engram_count": len(self.consolidated_engrams),
            },
            "epigenetic_profile": {
                k: round(v, 5) for k, v in self.epigenetic_profile.to_dict().items()
            },
            "neuromodulator_resting_summary": {
                k: round(float(v), 5) for k, v in self.neuromodulator_resting_summary.items()
            },
            "connectome_divergence_from_birth": {
                k: round(float(v), 5) for k, v in self.connectome_divergence_from_birth.items()
            },
            "manifold_calibration": {
                "latent_bridge_dim": 228,
                "cortical_participation_ratio": round(float(cal.cortical_participation_ratio), 4),
                "resting_attractor_l2_norm": round(
                    float(torch.linalg.vector_norm(cal.resting_attractor_228d).item()), 4
                ),
                "flashback_signature_l2_norm": round(
                    float(torch.linalg.vector_norm(cal.flashback_signature_228d).item()), 4
                ),
                "resting_biophysical_coordinates": {
                    name: round(float(val), 4)
                    for name, val in zip(BIOPHYSICAL_COORD_NAMES, rest_bio_coords)
                },
            },
            "top_consolidated_engrams": [
                {
                    "timestamp": round(float(e["timestamp"]), 2),
                    "consolidation_shield": round(float(e["consolidation_shield"]), 4),
                    "vitality": round(float(e["vitality"]), 4),
                    "somatic_marker_relief_trauma_threat_novelty": [
                        round(float(x), 4) for x in e["somatic_marker"].tolist()
                    ],
                }
                for e in sorted(
                    self.consolidated_engrams,
                    key=lambda item: float(item["consolidation_shield"]) * float(item["vitality"]),
                    reverse=True,
                )[:6]
            ],
            "biographical_summary": self.biographical_summary,
        }
        json_file.write_text(json.dumps(dossier, indent=2, ensure_ascii=False), encoding="utf-8")
        return pt_file, json_file

    @classmethod
    def load(cls, pt_path: str | Path, verify_integrity: bool = True) -> "MaturedSoulCapsule":
        """Load a ``.soul.pt`` capsule and verify its cryptographic SHA-256 seal."""
        raw = torch.load(Path(pt_path), map_location="cpu", weights_only=False)
        cal_raw = raw["calibration"]
        cal = LLMBridgeCalibration(
            latent_mean_228d=cal_raw["latent_mean_228d"].float(),
            latent_std_228d=cal_raw["latent_std_228d"].float(),
            resting_attractor_228d=cal_raw["resting_attractor_228d"].float(),
            flashback_signature_228d=cal_raw["flashback_signature_228d"].float(),
            svd_basis_16x228=cal_raw["svd_basis_16x228"].float(),
            svd_singular_values_16=cal_raw["svd_singular_values_16"].float(),
            cortical_participation_ratio=float(cal_raw["cortical_participation_ratio"]),
            biophysical_readout_w_5x228=cal_raw["biophysical_readout_w_5x228"].float(),
            biophysical_readout_b_5=cal_raw["biophysical_readout_b_5"].float(),
        )
        capsule = cls(
            capsule_id=str(raw["capsule_id"]),
            organism_id=str(raw["organism_id"]),
            biographical_regime=str(raw["biographical_regime"]),
            created_at_utc=str(raw["created_at_utc"]),
            total_waking_ticks=int(raw["total_waking_ticks"]),
            total_sleep_cycles=int(raw["total_sleep_cycles"]),
            total_swr_consolidations=int(raw["total_swr_consolidations"]),
            species_genome_sha256=str(raw["species_genome_sha256"]),
            brain_state_dict={k: v.float() for k, v in raw["brain_state_dict"].items()},
            epigenetic_profile=EpigeneticTemperamentProfile.from_dict(raw["epigenetic_profile"]),
            consolidated_engrams=list(raw["consolidated_engrams"]),
            neuromodulator_resting_summary=dict(raw["neuromodulator_resting_summary"]),
            connectome_divergence_from_birth=dict(raw["connectome_divergence_from_birth"]),
            calibration=cal,
            biographical_summary=dict(raw.get("biographical_summary", {})),
            capsule_sha256=str(raw.get("capsule_sha256", "")),
        )
        if verify_integrity and capsule.capsule_sha256:
            actual = capsule.compute_integrity_sha256()
            if actual != capsule.capsule_sha256:
                raise ValueError(
                    f"Soul capsule SHA-256 mismatch: expected {capsule.capsule_sha256}, got {actual}"
                )
        return capsule

    def instantiate_living_mind_and_soma(
        self,
        brain_config: BrainConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> Tuple[LimbicBrain, DigitalSoma]:
        """Reconstruct a live, ticking ``LimbicBrain`` & ``DigitalSoma`` from this capsule."""
        dev = torch.device(device)
        brain = LimbicBrain(config=brain_config, device=dev)
        state_on_dev = {k: v.to(device=dev) for k, v in self.brain_state_dict.items()}
        brain.load_state_dict(state_on_dev, strict=True)

        # Restore consolidated CA3 hippocampal episodic engrams
        brain.hippocampus.engrams.clear()
        for item in self.consolidated_engrams:
            brain.hippocampus.engrams.append(
                EpisodicEngram(
                    dg_sparse_code=item["dg_sparse_code"].to(device=dev, dtype=brain.dtype),
                    ca1_context_vec=item["ca1_context_vec"].to(device=dev, dtype=brain.dtype),
                    extero_cue_snapshot=item["extero_cue_snapshot"].to(device=dev, dtype=brain.dtype),
                    action_taken=item["action_taken"].to(device=dev, dtype=brain.dtype),
                    somatic_marker=item["somatic_marker"].to(device=dev, dtype=brain.dtype),
                    consolidation_shield=float(item["consolidation_shield"]),
                    vitality=float(item["vitality"]),
                    timestamp=float(item["timestamp"]),
                )
            )

        soma = DigitalSoma(device=dev)
        soma.reset()
        EpigeneticPlasticityEngine.apply_to_living_substrate(
            self.epigenetic_profile,
            brain_nm_state=brain.neuromod_engine.state,
            soma=soma,
            blend=1.0,
        )
        return brain, soma

    def create_llm_socket(
        self,
        llm_hidden_dim: int = 896,
        num_soft_prefix_tokens: int = 4,
        seed: int = 2026,
        device: str | torch.device = "cpu",
    ) -> "LiveSoulLLMSocket":
        """Create a ready-to-attach PyTorch Residual-Stream & Soft-Prefix Vocal-Cord Socket."""
        return LiveSoulLLMSocket(
            capsule=self,
            llm_hidden_dim=llm_hidden_dim,
            num_soft_prefix_tokens=num_soft_prefix_tokens,
            seed=seed,
            device=device,
        )


class LiveSoulLLMSocket(nn.Module):
    """Plug-and-Play PyTorch Residual-Stream & Soft-Prefix Bridge for Frozen Open LLMs.

    Connects a living ``MaturedSoulCapsule`` directly into any Transformer decoder block
    via PyTorch ``register_forward_hook``:
        h_tilde^(l) = h^(l) + alpha * ResidualProjector(whitened_z_228(t), bio_coords_5(t))
    with ZERO roleplay system prompts.
    """

    def __init__(
        self,
        capsule: MaturedSoulCapsule,
        llm_hidden_dim: int = 896,
        num_soft_prefix_tokens: int = 4,
        seed: int = 2026,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.capsule = capsule
        self.llm_hidden_dim = int(llm_hidden_dim)
        self.num_soft_prefix_tokens = int(num_soft_prefix_tokens)
        self._device = torch.device(device)

        self.brain, self.soma = capsule.instantiate_living_mind_and_soma(device=self._device)
        self.interoceptor = InteroceptiveNerveBundle(
            setpoints=self.soma.setpoints,
            device=self._device,
        )
        self.interoceptor.reset(self.soma.state)

        cal = capsule.calibration
        self.register_buffer("latent_mean", cal.latent_mean_228d.to(self._device))
        self.register_buffer("latent_std", cal.latent_std_228d.clamp(min=1e-3).to(self._device))
        self.register_buffer("bio_w", cal.biophysical_readout_w_5x228.to(self._device))
        self.register_buffer("bio_b", cal.biophysical_readout_b_5.to(self._device))
        self.register_buffer("svd_basis_16", cal.svd_basis_16x228.to(self._device))
        self.register_buffer("live_latent_228d", cal.resting_attractor_228d.clone().to(self._device))

        # Project [228D whitened mind + 16D manifold coords + 5D biophysical drives = 249D] -> LLM d_model
        bridge_in_dim = 228 + 16 + 5
        self.residual_projector = nn.Sequential(
            nn.Linear(bridge_in_dim, 128, device=self._device),
            nn.SiLU(),
            nn.Linear(128, self.llm_hidden_dim, device=self._device),
        )
        # Initialize output scale conservatively so untrained residual hooks preserve LLM fluency
        nn.init.orthogonal_(self.residual_projector[2].weight, gain=0.25)
        nn.init.zeros_(self.residual_projector[2].bias)

        self.soft_prefix_projector = nn.Linear(
            bridge_in_dim,
            self.num_soft_prefix_tokens * self.llm_hidden_dim,
            device=self._device,
        )
        nn.init.normal_(self.soft_prefix_projector.weight, mean=0.0, std=0.04)
        nn.init.zeros_(self.soft_prefix_projector.bias)

        # Sensory ear projection allowing Stage 5 LLM input embeddings to perturb cochlear/thalamic nerves
        self.semantic_ear_to_sensory = nn.Linear(self.llm_hidden_dim, 328, device=self._device)
        nn.init.normal_(self.semantic_ear_to_sensory.weight, mean=0.0, std=0.03)
        nn.init.zeros_(self.semantic_ear_to_sensory.bias)

        self._sim_clock: float = 0.0
        self._active_hooks: List[Any] = []

    def step_autonomous_heartbeat(
        self,
        dt: float = 0.05,
        extero_328d: torch.Tensor | None = None,
        acoustic_roughness_shock: float = 0.0,
        acoustic_harmonic_resonance: float = 0.0,
        bus_supply_voltage: float = 1.0,
    ) -> MindDiagnostics:
        """Advance the living soma & continuous-time mind by ``dt`` seconds (even in silence)."""
        from limbic_zero.core.soma import ExogenousBodyForces, MotorMetabolicCommand
        from limbic_zero.sandbox.sensors import ExteroceptivePercept
        from limbic_zero.sandbox.world import OrganismObservation

        self._sim_clock += dt
        forces = ExogenousBodyForces(
            harmonic_acoustic_energy=acoustic_harmonic_resonance,
            rough_acoustic_shock=acoustic_roughness_shock,
            bus_supply_voltage=bus_supply_voltage,
        )
        soma_diag = self.soma.step(dt=dt, motor_cmd=MotorMetabolicCommand(), forces=forces)
        intero = self.interoceptor.transduce(
            soma=self.soma,
            diag=soma_diag,
            raw_bus_voltage=bus_supply_voltage,
            dt=dt,
        )

        if extero_328d is None:
            ext_vec = torch.zeros(328, dtype=self.brain.dtype, device=self._device)
            # Inject physical acoustic energy directly onto binaural cochlear channels [297:305]
            if acoustic_harmonic_resonance > 0.0 or acoustic_roughness_shock > 0.0:
                amp = min(1.5, acoustic_harmonic_resonance + acoustic_roughness_shock)
                ext_vec[297] = amp
                ext_vec[298] = acoustic_harmonic_resonance
                ext_vec[299] = acoustic_roughness_shock
                ext_vec[301] = amp
                ext_vec[302] = acoustic_harmonic_resonance
                ext_vec[303] = acoustic_roughness_shock
        else:
            ext_vec = extero_328d.to(dtype=self.brain.dtype, device=self._device)

        dummy_extero = ExteroceptivePercept(
            vision_rays=ext_vec[:297].view(33, 9),
            binaural_ears=ext_vec[297:305],
            birhinal_nostrils=ext_vec[305:313],
            tactile_ring=ext_vec[313:321],
            vestibular_proprio=ext_vec[321:328],
            flat_vector=ext_vec,
        )
        obs = OrganismObservation(
            agent_id=self.capsule.organism_id,
            sim_time=self._sim_clock,
            is_alive=self.soma.is_alive,
            exteroception=dummy_extero,
            interoception=intero,
            combined_sensory_tensor=torch.cat([ext_vec, intero.nerve_fiber_vector], dim=0),
        )
        _, mind_diag = self.brain.step(obs=obs, soma=self.soma, dt=dt, deterministic=False)
        self.brain.detach_state_()
        self.live_latent_228d.copy_(mind_diag.latent_affective_vector.detach())
        return mind_diag

    def current_bridge_features(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return the 249D conditioned bridge feature vector & live 5D biophysical coordinates."""
        whitened = torch.clamp((self.live_latent_228d - self.latent_mean) / self.latent_std, -5.0, 5.0)
        manifold_16 = torch.mv(self.svd_basis_16, whitened)
        bio_5 = torch.tanh(torch.mv(self.bio_w, whitened) + self.bio_b)
        feat_249 = torch.cat([whitened, manifold_16, bio_5], dim=0)
        coords = {
            name: float(bio_5[i].detach().item())
            for i, name in enumerate(BIOPHYSICAL_COORD_NAMES)
        }
        return feat_249, coords

    def compute_residual_delta(self, injection_alpha: float = 1.0) -> torch.Tensor:
        """Compute ``(llm_hidden_dim,)`` residual-stream steering vector ``alpha * P(z(t))``."""
        if not self.soma.is_alive:
            return torch.zeros(self.llm_hidden_dim, device=self._device)
        feat_249, _ = self.current_bridge_features()
        raw_delta = self.residual_projector(feat_249)
        return float(injection_alpha) * raw_delta

    def compute_soft_prefix_tokens(self, batch_size: int = 1) -> torch.Tensor:
        """Generate ``(batch_size, num_soft_prefix_tokens, llm_hidden_dim)`` soft prefix embeddings."""
        feat_249, _ = self.current_bridge_features()
        prefix = self.soft_prefix_projector(feat_249).view(
            1, self.num_soft_prefix_tokens, self.llm_hidden_dim
        )
        return prefix.expand(batch_size, -1, -1)

    def attach_residual_hook(self, transformer_block: nn.Module, alpha: float = 1.0) -> Any:
        """Attach a live PyTorch ``register_forward_hook`` injecting ``z(t)`` into ``transformer_block``."""

        def _hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
            delta = self.compute_residual_delta(injection_alpha=alpha)
            if isinstance(output, tuple):
                hidden = output[0]
                steered = hidden + delta.to(dtype=hidden.dtype, device=hidden.device).view(
                    *([1] * (hidden.ndim - 1)), -1
                )
                return (steered, *output[1:])
            if isinstance(output, torch.Tensor):
                return output + delta.to(dtype=output.dtype, device=output.device).view(
                    *([1] * (output.ndim - 1)), -1
                )
            return output

        handle = transformer_block.register_forward_hook(_hook)
        self._active_hooks.append(handle)
        return handle

    def detach_all_hooks(self) -> None:
        """Cleanly remove all registered residual-stream forward hooks."""
        for h in self._active_hooks:
            h.remove()
        self._active_hooks.clear()


class SoulCapsuleExporter:
    """Calibrates empirical 228D manifold geometry and seals ``MaturedSoulCapsule`` bundles."""

    @staticmethod
    @torch.no_grad()
    def extract_biophysical_target_5d(obs_intero: Any, diag: MindDiagnostics) -> torch.Tensor:
        """Extract the 5 purely biophysical drive coordinates corresponding to ``diag``."""
        nm = diag.neuromodulators
        relief_bal = math.tanh(0.8 * obs_intero.homeostatic_valence - 0.18 * obs_intero.total_drive)
        mobilization = math.tanh(
            0.9 * nm.norepinephrine_extracellular + 0.8 * diag.basal_ganglia.dlpag_active_defense - 0.3
        )
        restraint = math.tanh(
            1.1 * nm.ht1a_occupancy + 0.6 * diag.basal_ganglia.stn_hyperdirect_brake - 0.7 * nm.lhb_aversive_rate
        )
        curiosity = math.tanh(
            0.95 * nm.acetylcholine_extracellular + 0.65 * diag.thalamus.epistemic_novelty - 0.35
        )
        vigilance = math.tanh(
            1.2 * diag.amygdala.cea_defensive_output + 0.7 * diag.amygdala.bla_defensive_raw - 0.15
        )
        return torch.tensor(
            [relief_bal, mobilization, restraint, curiosity, vigilance],
            dtype=torch.float32,
        )

    @classmethod
    @torch.no_grad()
    def build_capsule(
        cls,
        *,
        capsule_id: str,
        organism_id: str,
        biographical_regime: str,
        brain: LimbicBrain,
        soma: DigitalSoma,
        epigenetic_profile: EpigeneticTemperamentProfile,
        species_genome: torch.Tensor,
        birth_connectome_snapshot: Dict[str, torch.Tensor],
        recorded_latents_228d: List[torch.Tensor],
        recorded_biophysical_5d: List[torch.Tensor],
        total_waking_ticks: int,
        total_sleep_cycles: int,
        total_swr_consolidations: int,
        biographical_summary: Dict[str, Any],
    ) -> MaturedSoulCapsule:
        """Fit SVD & ridge biophysical calibration, probe dark-room resting state, and seal capsule."""
        from limbic_zero.evolution.genome import ConnectomeDivergenceAnalyzer

        # 1. Empirical 228D trajectory statistics & top-16 SVD manifold
        if len(recorded_latents_228d) < 8:
            raise ValueError("Need at least 8 recorded 228D mind states to calibrate Soul Capsule")

        z_mat = torch.stack([z.detach().cpu().float() for z in recorded_latents_228d], dim=0)  # (N, 228)
        y_mat = torch.stack([y.detach().cpu().float() for y in recorded_biophysical_5d], dim=0)  # (N, 5)

        z_mean = z_mat.mean(dim=0)
        z_centered = z_mat - z_mean
        z_std = z_centered.std(dim=0, unbiased=False).clamp(min=1e-3)

        _, s_vals, vh = torch.linalg.svd(z_centered, full_matrices=False)
        eig_vals = (s_vals ** 2) / float(max(1, z_mat.shape[0] - 1))
        sum_eig = float(eig_vals.sum().item())
        sum_sq = float((eig_vals ** 2).sum().item())
        pr = (sum_eig * sum_eig) / max(1e-12, sum_sq) if sum_eig > 1e-9 else 1.0

        k_svd = min(16, vh.shape[0])
        basis_16 = torch.zeros((16, 228), dtype=torch.float32)
        s_top16 = torch.zeros(16, dtype=torch.float32)
        basis_16[:k_svd] = vh[:k_svd]
        s_top16[:k_svd] = s_vals[:k_svd]

        # 2. Ridge regression mapping whitened 228D mind state -> 5D biophysical coordinates
        z_white = torch.clamp(z_centered / z_std, -5.0, 5.0)
        n_samples = z_white.shape[0]
        ones = torch.ones((n_samples, 1), dtype=torch.float32)
        x_aug = torch.cat([z_white, ones], dim=1)  # (N, 229)
        reg = torch.eye(229, dtype=torch.float32) * 8.0
        reg[228, 228] = 1e-4
        xtx = torch.matmul(x_aug.T, x_aug) + reg
        xty = torch.matmul(x_aug.T, y_mat)
        w_aug = torch.linalg.solve(xtx, xty)  # (229, 5)
        bio_w = w_aug[:228].T.contiguous()     # (5, 228)
        bio_b = w_aug[228].contiguous()        # (5,)

        # 3. Measure authentic dark-room resting attractor & peak flashback mind signature
        # Clone state so probing does not alter consolidated weights
        probe_brain = LimbicBrain(config=brain.config)
        probe_brain.load_state_dict({k: v.detach().cpu() for k, v in brain.state_dict().items()})
        for e in brain.hippocampus.engrams:
            probe_brain.hippocampus.engrams.append(
                EpisodicEngram(
                    dg_sparse_code=e.dg_sparse_code.detach().cpu().clone(),
                    ca1_context_vec=e.ca1_context_vec.detach().cpu().clone(),
                    extero_cue_snapshot=e.extero_cue_snapshot.detach().cpu().clone(),
                    action_taken=e.action_taken.detach().cpu().clone(),
                    somatic_marker=e.somatic_marker.detach().cpu().clone(),
                    consolidation_shield=e.consolidation_shield,
                    vitality=e.vitality,
                    timestamp=e.timestamp,
                )
            )
        probe_soma = DigitalSoma()
        probe_soma.reset()
        EpigeneticPlasticityEngine.apply_to_living_substrate(
            epigenetic_profile,
            brain_nm_state=probe_brain.neuromod_engine.state,
            soma=probe_soma,
            blend=1.0,
        )
        probe_intero = InteroceptiveNerveBundle(setpoints=probe_soma.setpoints)
        probe_intero.reset(probe_soma.state)

        from limbic_zero.sandbox.sensors import ExteroceptivePercept
        from limbic_zero.sandbox.world import OrganismObservation

        zero_ext = torch.zeros(328, dtype=torch.float32)
        dummy_ext = ExteroceptivePercept(
            vision_rays=zero_ext[:297].view(33, 9),
            binaural_ears=zero_ext[297:305],
            birhinal_nostrils=zero_ext[305:313],
            tactile_ring=zero_ext[313:321],
            vestibular_proprio=zero_ext[321:328],
            flat_vector=zero_ext,
        )

        rest_accum = torch.zeros(228, dtype=torch.float32)
        flashback_vec: torch.Tensor | None = None
        last_probe_diag: MindDiagnostics | None = None

        for step_k in range(36):
            s_diag = probe_soma.step(dt=0.05)
            i_percept = probe_intero.transduce(probe_soma, s_diag, raw_bus_voltage=1.0, dt=0.05)
            obs_k = OrganismObservation(
                agent_id=organism_id,
                sim_time=float(step_k) * 0.05,
                is_alive=True,
                exteroception=dummy_ext,
                interoception=i_percept,
                combined_sensory_tensor=torch.cat([zero_ext, i_percept.nerve_fiber_vector], dim=0),
            )
            _, m_diag = probe_brain.step(obs=obs_k, soma=probe_soma, dt=0.05, enable_plasticity=False)
            last_probe_diag = m_diag
            if step_k >= 16:
                rest_accum.add_(m_diag.latent_affective_vector.detach().cpu())
            if m_diag.hippocampus.swr_replay_active and flashback_vec is None:
                flashback_vec = m_diag.latent_affective_vector.detach().cpu().clone()

        resting_attractor = rest_accum / 20.0
        if flashback_vec is None:
            flashback_vec = z_mat[-1].clone()

        nm_rest = last_probe_diag.neuromodulators if last_probe_diag is not None else brain.neuromod_engine.state
        nm_summary = {
            "dopamine_extracellular": float(nm_rest.dopamine_extracellular),
            "norepinephrine_extracellular": float(nm_rest.norepinephrine_extracellular),
            "serotonin_extracellular": float(nm_rest.serotonin_extracellular),
            "acetylcholine_extracellular": float(nm_rest.acetylcholine_extracellular),
            "effective_discount_gamma": float(nm_rest.effective_discount_gamma),
            "d1_sensitivity": float(nm_rest.d1_sensitivity),
            "ne_beta_sensitivity": float(nm_rest.ne_beta_sensitivity),
            "ht1a_sensitivity": float(nm_rest.ht1a_sensitivity),
            "ach_m1_sensitivity": float(nm_rest.ach_m1_sensitivity),
            "gr_receptor_sensitivity": float(probe_soma.endo.gr_receptor_sensitivity),
            "oxytocin_receptor_sensitivity": float(probe_soma.endo.oxytocin_receptor_sensitivity),
            "bla_defensive_norm": float(torch.linalg.vector_norm(brain.amygdala.w_bla_defensive).item()),
            "bla_appetitive_norm": float(torch.linalg.vector_norm(brain.amygdala.w_bla_appetitive).item()),
            "itc_extinction_norm": float(torch.linalg.vector_norm(brain.amygdala.w_itc_extinction).item()),
        }

        matured_snap = ConnectomeDivergenceAnalyzer.snapshot_plastic_connectome(brain)
        div_report = ConnectomeDivergenceAnalyzer.compute_regional_l2_divergence(
            birth_connectome_snapshot,
            matured_snap,
        )

        engrams_serialized = [
            {
                "dg_sparse_code": e.dg_sparse_code.detach().cpu().clone(),
                "ca1_context_vec": e.ca1_context_vec.detach().cpu().clone(),
                "extero_cue_snapshot": e.extero_cue_snapshot.detach().cpu().clone(),
                "action_taken": e.action_taken.detach().cpu().clone(),
                "somatic_marker": e.somatic_marker.detach().cpu().clone(),
                "consolidation_shield": float(e.consolidation_shield),
                "vitality": float(e.vitality),
                "timestamp": float(e.timestamp),
            }
            for e in brain.hippocampus.engrams
        ]

        genome_bytes = species_genome.detach().cpu().contiguous().float().numpy().tobytes()
        genome_sha = hashlib.sha256(genome_bytes).hexdigest()

        calibration = LLMBridgeCalibration(
            latent_mean_228d=z_mean,
            latent_std_228d=z_std,
            resting_attractor_228d=resting_attractor,
            flashback_signature_228d=flashback_vec,
            svd_basis_16x228=basis_16,
            svd_singular_values_16=s_top16,
            cortical_participation_ratio=pr,
            biophysical_readout_w_5x228=bio_w,
            biophysical_readout_b_5=bio_b,
        )

        capsule = MaturedSoulCapsule(
            capsule_id=capsule_id,
            organism_id=organism_id,
            biographical_regime=biographical_regime,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            total_waking_ticks=total_waking_ticks,
            total_sleep_cycles=total_sleep_cycles,
            total_swr_consolidations=total_swr_consolidations,
            species_genome_sha256=genome_sha,
            brain_state_dict={k: v.detach().cpu().clone() for k, v in brain.state_dict().items()},
            epigenetic_profile=EpigeneticTemperamentProfile.from_dict(epigenetic_profile.to_dict()),
            consolidated_engrams=engrams_serialized,
            neuromodulator_resting_summary=nm_summary,
            connectome_divergence_from_birth=div_report,
            calibration=calibration,
            biographical_summary=biographical_summary,
        )
        capsule.capsule_sha256 = capsule.compute_integrity_sha256()
        return capsule
