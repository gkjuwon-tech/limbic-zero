"""Morphogenetic Connectome Genome Adapter & Anatomical Divergence Analyzer.

Models biological Zador (2019) 'Genomic Bottleneck' development without any hardcoded
sensory labels, sign rules, or behavioral cheats:
    - At birth (``t = 0``), a compact vector of continuous morphogenetic parameters
      ``genes`` in ``R^G`` expresses structured topographic receptive fields across
      the sensory thalamus (LGN, MGN, Piriform Olfactory, VPL, VMpo), Superior
      Colliculus, Corticostriatal projection, and Striatal D1-Go / D2-NoGo ensembles
      inside a standard ``LimbicBrain``.
    - At ``genes == 0``, the brain reduces to a random baseline connectome with zero
      directional or valence knowledge. Phylogenetic natural selection must discover
      every sensory-to-motor polarity purely from endogenous survival and homeostatic
      free-energy minimization.
    - Once expressed at birth, every individual synapse across all ``47,000+`` weights
      is completely unconstrained and undergoes synapse-specific lifetime plasticity
      (Three-Factor STDP, Opposing D1/D2 TD RL, Pavlovian BLA/ITC, & Sleep Consolidation).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

import torch

from limbic_zero.mind.brain import BrainConfig, LimbicBrain


@dataclass(frozen=True)
class MorphogeneticGenomeConfig:
    """Dimensions and developmental gain scales of the morphogenetic genome."""

    lgn_streams_per_hemifield: int = 3      # 3 hemifields * 3 streams * 9 ray channels = 81 genes
    olfactory_streams: int = 4              # 4 streams * 8 birhinal channels = 32 genes
    auditory_streams: int = 3               # 3 streams * 8 binaural channels = 24 genes
    somato_streams: int = 4                 # 4 streams * 15 tactile/vestibular channels = 60 genes
    intero_streams: int = 4                 # 4 streams * 12 pooled visceral blocks = 48 genes
    striatal_ensembles: int = 8             # 8 striatal MSN ensembles
    striatal_rank: int = 4                  # Rank-4 corticostriatal bottleneck
    expression_scale: float = 1.25          # Developmental weight expression amplitude
    baseline_connectome_seed: int = 202604  # Fixed structural scaffold seed


class MorphogeneticGenomeExpressor:
    """Expresses a compact developmental genome vector into a newborn ``LimbicBrain``."""

    def __init__(self, config: MorphogeneticGenomeConfig | None = None) -> None:
        self.config = config or MorphogeneticGenomeConfig()
        cfg = self.config

        # Core expressed thalamic feature count inside the 76D exteroceptive cue vector:
        #   3 * lgn_streams + auditory_streams + olfactory_streams + somato_streams
        self._n_extero_core = (
            3 * cfg.lgn_streams_per_hemifield
            + cfg.auditory_streams
            + cfg.olfactory_streams
            + cfg.somato_streams
        )  # 9 + 3 + 4 + 4 = 20 core exteroceptive relays
        self._n_limbic_modes = 6 + 4  # 6 pooled pallial modes (2 per column) + 4 amygdala modes = 10
        self._joint_core_dim = self._n_limbic_modes + self._n_extero_core  # 30

        # Compute exact slice layout of the flat genome vector
        slices: List[Tuple[str, Tuple[int, ...]]] = [
            ("lgn_hemifields", (3, cfg.lgn_streams_per_hemifield, 9)),
            ("olfactory_relay", (cfg.olfactory_streams, 8)),
            ("mgn_auditory", (cfg.auditory_streams, 8)),
            ("vpl_somato", (cfg.somato_streams, 15)),
            ("vmpo_intero", (cfg.intero_streams, 12)),
            ("colliculus_w", (4, self._n_extero_core)),
            ("colliculus_b", (4,)),
            ("striatal_u", (cfg.striatal_ensembles, cfg.striatal_rank)),
            ("striatal_v", (cfg.striatal_rank, self._joint_core_dim)),
            ("striatal_b", (cfg.striatal_ensembles,)),
            ("d1_go_w", (9, cfg.striatal_ensembles)),
            ("d1_go_b", (9,)),
            ("d2_nogo_w", (9, cfg.striatal_ensembles)),
            ("d2_nogo_b", (9,)),
            ("stn_pfc_drive", (6,)),
        ]
        self.slice_specs: Dict[str, Tuple[int, int, Tuple[int, ...]]] = {}
        offset = 0
        for name, shape in slices:
            numel = 1
            for s in shape:
                numel *= s
            self.slice_specs[name] = (offset, offset + numel, shape)
            offset += numel
        self.genome_dim: int = offset

        # Precompute smooth spatial ray envelopes across each 11-ray retinal hemifield
        # (purely geometric taper so adjacent rays within a hemifield share smooth continuity)
        ray_grid = torch.linspace(-1.0, 1.0, steps=11, dtype=torch.float32)
        self._hemifield_taper = (0.70 + 0.30 * torch.cos(0.5 * math.pi * ray_grid)) / math.sqrt(11.0)

        # Precompute deterministic block-pooling matrix (12 x 86) for the 86D visceral nerve bundle
        gen = torch.Generator()
        gen.manual_seed(cfg.baseline_connectome_seed + 77)
        visceral_basis = torch.zeros((12, 86), dtype=torch.float32)
        # Partition 86 visceral fibers into 12 contiguous anatomical blocks so all tracts are covered
        bounds = [0, 12, 24, 36, 48, 56, 64, 72, 76, 80, 82, 84, 86]
        for b_idx in range(12):
            s0, s1 = bounds[b_idx], bounds[b_idx + 1]
            width = max(1, s1 - s0)
            visceral_basis[b_idx, s0:s1] = 1.0 / math.sqrt(float(width))
        self._visceral_basis = visceral_basis

    def sample_initial_genome(self, seed: int = 42, init_std: float = 0.08) -> torch.Tensor:
        """Create a small zero-mean random initial genome (zero built-in task knowledge)."""
        gen = torch.Generator()
        gen.manual_seed(seed)
        return torch.randn(self.genome_dim, generator=gen, dtype=torch.float32) * float(init_std)

    def _unpack(self, genes: torch.Tensor, key: str) -> torch.Tensor:
        s0, s1, shape = self.slice_specs[key]
        return genes[s0:s1].view(shape)

    def build_expressed_brain(
        self,
        genes: torch.Tensor,
        brain_config: BrainConfig | None = None,
        scaffold_seed: int | None = None,
    ) -> LimbicBrain:
        """Instantiate a fresh ``LimbicBrain`` and express ``genes`` onto its innate connectome."""
        seed = self.config.baseline_connectome_seed if scaffold_seed is None else scaffold_seed
        brain = LimbicBrain(config=brain_config, seed=seed)
        self.express_into_brain(brain, genes)
        return brain

    @torch.no_grad()
    def express_into_brain(self, brain: LimbicBrain, genes: torch.Tensor) -> None:
        """Express morphogenetic ``genes`` into ``brain``'s innate neural projections."""
        if genes.numel() != self.genome_dim:
            raise ValueError(f"Expected genome of dim {self.genome_dim}, got {genes.numel()}")

        cfg = self.config
        g = genes.detach().to(dtype=brain.dtype, device=brain.prev_action_7d.device)
        scale = cfg.expression_scale

        thal = brain.thalamus
        t_cfg = thal.config
        lgn_sub = t_cfg.lgn_visual_dim // 3
        k_lgn = cfg.lgn_streams_per_hemifield
        taper = self._hemifield_taper.to(dtype=brain.dtype, device=g.device)  # (11,)

        # 1. Express Retinotopic LGN Hemifield Kernels: (3, k_lgn, 9) -> (k_lgn, 11*9=99)
        lgn_genes = self._unpack(g, "lgn_hemifields") * scale
        for h_idx, layer in enumerate((thal.lgn_left, thal.lgn_fovea, thal.lgn_right)):
            # Outer product across 11 rays in hemifield and 9 photoreceptor channels -> (k_lgn, 11, 9)
            kernel_99 = (lgn_genes[h_idx].unsqueeze(1) * taper.view(1, 11, 1)).reshape(k_lgn, 99)
            layer.weight.data[:k_lgn].copy_(kernel_99)
            layer.bias.data[:k_lgn].zero_()

        # 2. Express Olfactory, Auditory, Somatosensory, & Visceral Thalamic Relays
        olf_w = self._unpack(g, "olfactory_relay") * scale
        thal.olfactory_relay.weight.data[: cfg.olfactory_streams].copy_(olf_w)
        thal.olfactory_relay.bias.data[: cfg.olfactory_streams].zero_()

        aud_w = self._unpack(g, "mgn_auditory") * scale
        thal.mgn_auditory.weight.data[: cfg.auditory_streams].copy_(aud_w)
        thal.mgn_auditory.bias.data[: cfg.auditory_streams].zero_()

        som_w = self._unpack(g, "vpl_somato") * scale
        thal.vpl_somato.weight.data[: cfg.somato_streams].copy_(som_w)
        thal.vpl_somato.bias.data[: cfg.somato_streams].zero_()

        int_genes = self._unpack(g, "vmpo_intero") * scale  # (intero_streams, 12)
        v_basis = self._visceral_basis.to(dtype=brain.dtype, device=g.device)  # (12, 86)
        thal.vmpo_intero.weight.data[: cfg.intero_streams].copy_(torch.matmul(int_genes, v_basis))
        thal.vmpo_intero.bias.data[: cfg.intero_streams].zero_()

        # Identify exact indices of the 20 core expressed channels inside 76D extero_cue
        vis_dim = t_cfg.lgn_visual_dim
        aud_off = vis_dim
        olf_off = aud_off + t_cfg.mgn_auditory_dim
        som_off = olf_off + t_cfg.olfactory_dim

        extero_indices: List[int] = []
        extero_indices.extend(range(0, k_lgn))
        extero_indices.extend(range(lgn_sub, lgn_sub + k_lgn))
        extero_indices.extend(range(2 * lgn_sub, 2 * lgn_sub + k_lgn))
        extero_indices.extend(range(aud_off, aud_off + cfg.auditory_streams))
        extero_indices.extend(range(olf_off, olf_off + cfg.olfactory_streams))
        extero_indices.extend(range(som_off, som_off + cfg.somato_streams))
        idx_tensor = torch.tensor(extero_indices, dtype=torch.long, device=g.device)

        # 3. Express Superior Colliculus Orienting Projection (4 x 76)
        bg = brain.basal_ganglia
        col_w = self._unpack(g, "colliculus_w") * (scale * 1.25)
        col_b = self._unpack(g, "colliculus_b") * (scale * 0.5)
        bg.colliculus_orient.weight.data.mul_(0.25)
        bg.colliculus_orient.weight.data[:, idx_tensor] = col_w
        bg.colliculus_orient.bias.data.copy_(col_b)

        # 4. Express Corticostriatal Projection onto first ``striatal_ensembles`` MSN groups
        # Joint input to cortico_striatal_enc is [pallial_workspace (192), amygdala_embed (24), extero_cue (76)]
        pall_dim = brain.config.pallium.total_pallial_dim  # 192
        amyg_dim = brain.config.amygdala_embed_dim         # 24
        extero_base = pall_dim + amyg_dim                  # 216

        u_mat = self._unpack(g, "striatal_u")
        v_mat = self._unpack(g, "striatal_v")
        cs_core = torch.matmul(u_mat, v_mat) * (scale * 0.85)  # (ens, 10 + 20)
        cs_b = self._unpack(g, "striatal_b") * (scale * 0.45)

        ens = cfg.striatal_ensembles
        bg.cortico_striatal_enc.weight.data[:ens].zero_()
        # Pool 6 pallial sub-blocks (32 neurons each -> 192) smoothly into modes 0..5
        for m_idx in range(6):
            p0, p1 = m_idx * 32, (m_idx + 1) * 32
            bg.cortico_striatal_enc.weight.data[:ens, p0:p1] = (
                cs_core[:, m_idx : m_idx + 1] / math.sqrt(32.0)
            )
        # Map first 4 amygdala embedding dims into modes 6..9
        bg.cortico_striatal_enc.weight.data[:ens, pall_dim : pall_dim + 4] = cs_core[:, 6:10]
        # Map the 20 core expressed exteroceptive relays into modes 10..29
        bg.cortico_striatal_enc.weight.data[:ens, extero_base + idx_tensor] = cs_core[:, 10:]
        bg.cortico_striatal_enc.bias.data[:ens].copy_(cs_b)

        # 5. Express Striatal Direct D1-Go & Indirect D2-NoGo Readout Synapses
        d1_w = self._unpack(g, "d1_go_w") * scale
        d1_b = self._unpack(g, "d1_go_b") * (scale * 0.55)
        d2_w = self._unpack(g, "d2_nogo_w") * scale
        d2_b = self._unpack(g, "d2_nogo_b") * (scale * 0.55)

        bg.w_d1_go.data[:, :ens] = d1_w
        bg.b_d1_go.data.copy_(0.10 + d1_b)
        bg.w_d2_nogo.data[:, :ens] = d2_w
        bg.b_d2_nogo.data.copy_(0.05 + d2_b)

        # 6. Express Prefrontal/Cingulate -> STN Hyperdirect Brake Weights
        stn_g = self._unpack(g, "stn_pfc_drive") * (scale * 0.5)
        for m_idx in range(6):
            p0, p1 = m_idx * 32, (m_idx + 1) * 32
            bg.stn_proj.weight.data[0, p0:p1] = stn_g[m_idx] / math.sqrt(32.0)


class ConnectomeDivergenceAnalyzer:
    """Snapshots and compares full ``47,000+``-synapse connectomes across life histories."""

    REGION_PREFIX_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        ("thalamus_world_model", ("thalamus.",)),
        ("amygdala_circuit", ("amygdala.",)),
        ("pallium_sensorimotor", ("pallium.sensorimotor_col.", "pallium.rec_sm.", "pallium.in_extero_to_sm.")),
        ("pallium_insula_dacc", ("pallium.insula_cingulate_col.", "pallium.rec_ins.", "pallium.in_intero_to_ins.")),
        ("pallium_prefrontal", ("pallium.prefrontal_slow_col.", "pallium.rec_pfc.", "pallium.in_hippo_to_pfc.")),
        ("pallium_intercolumn", ("pallium.ff_", "pallium.fb_")),
        ("basal_ganglia_striatum", ("basal_ganglia.",)),
        ("subcortical_critic", ("neuromod_engine.",)),
    )

    @staticmethod
    @torch.no_grad()
    def snapshot_plastic_connectome(brain: LimbicBrain) -> Dict[str, torch.Tensor]:
        """Clone all plastic synaptic parameters and conditioned amygdalar buffers."""
        snap: Dict[str, torch.Tensor] = {}
        for name, param in brain.named_parameters():
            snap[name] = param.detach().cpu().clone()
        for buf_name in (
            "amygdala.w_bla_defensive",
            "amygdala.w_bla_appetitive",
            "amygdala.w_itc_extinction",
        ):
            parts = buf_name.split(".")
            obj = getattr(brain, parts[0])
            snap[buf_name] = getattr(obj, parts[1]).detach().cpu().clone()
        return snap

    @classmethod
    def compute_regional_l2_divergence(
        cls,
        snap_a: Dict[str, torch.Tensor],
        snap_b: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Compute L2 synaptic divergence per anatomical brain region between two connectomes."""
        regional_sq: Dict[str, float] = {label: 0.0 for label, _ in cls.REGION_PREFIX_GROUPS}
        regional_sq["other_projections"] = 0.0
        total_sq = 0.0

        for key, ta in snap_a.items():
            if key not in snap_b:
                continue
            diff_sq = float(((ta - snap_b[key]).float() ** 2).sum().item())
            total_sq += diff_sq
            matched = False
            for label, prefixes in cls.REGION_PREFIX_GROUPS:
                if any(key.startswith(p) for p in prefixes):
                    regional_sq[label] += diff_sq
                    matched = True
                    break
            if not matched:
                regional_sq["other_projections"] += diff_sq

        result = {k: math.sqrt(v) for k, v in regional_sq.items()}
        result["total_connectome_l2"] = math.sqrt(total_sq)
        return result
