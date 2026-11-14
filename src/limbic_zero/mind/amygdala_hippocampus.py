"""Basolateral/Central Amygdala Conditioning & Trisynaptic Hippocampal Somatic Memory (DG->CA3->CA1).

Zero-Cheat Neurobiological Memory & Associative Conditioning:
    1. ``AmygdalaCircuit`` (BLA -> ITC Extinction Gate -> CeA):
       - Rapid single-trial/few-trial Pavlovian association between 76D exteroceptive cues
         (visual rival, glowing thermal vent, acoustic timbre, sweat odor plume) and visceral
         pain/suffocation/brownout (defensive pathway) or visceral relief (appetitive pathway).
       - Infralimbic prefrontal / Intercalated GABAergic Cells (ITC) learn an inhibitory
         *extinction* mask WITHOUT erasing BLA trauma weights -> acute stress (high NE_beta +
         Cortisol) suppresses ITC and triggers spontaneous trauma relapse!
    2. ``HippocampalSomaticMemory`` (DG -> CA3 -> CA1 + Sharp-Wave Ripple Replay):
       - Dentate Gyrus (DG) expands cues into 256D sparse k-Winners-Take-All (k=16) codes
         for orthogonal pattern separation.
       - CA3 autoassociative ring stores episodic engrams tagged with ``consolidation_shield``
         (Frey & Morris synaptic tag-and-capture driven by catecholamine + glucocorticoid surges).
       - Mundane memories decay exponentially; flashbulb traumas/rescues persist ~25x longer!
       - During quiet rest or elevated Adenosine A1, CA3 fires spontaneous **Sharp-Wave Ripple
         (SWR) Replay**, re-injecting consolidated past episodes into the pallium & amygdala
         so the organism ruminates on past betrayals or near-death events in total silence!
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AmygdalaOutput:
    """Associative defensive, appetitive, and extinction-gated outputs from BLA & CeA."""

    bla_defensive_raw: float            # Raw unextinguished trauma association [0, 1]
    bla_appetitive_raw: float           # Learned incentive cue value [0, 1]
    itc_extinction_inhibition: float    # Infralimbic/ITC inhibitory safety brake [0, 1]
    cea_defensive_output: float         # Final Central Amygdala defensive drive [0, 1]
    amygdala_embedding: torch.Tensor    # (24,) projection into pallium & striatum


class AmygdalaCircuit(nn.Module):
    """Basolateral (BLA), Intercalated Extinction (ITC), & Central Amygdala (CeA) microcircuit."""

    def __init__(
        self,
        extero_cue_dim: int,
        pallial_dim: int,
        embed_dim: int = 24,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.extero_cue_dim = extero_cue_dim
        self.pallial_dim = pallial_dim
        self.embed_dim = embed_dim
        self.dtype = dtype
        self._device = torch.device(device)

        # Cue feature expansion in Lateral Amygdala (LA)
        self.la_encoder = nn.Linear(extero_cue_dim, 48, dtype=dtype, device=self._device)

        # Plastic non-negative synaptic weights onto BLA defensive, BLA appetitive, & ITC extinction cells
        self.register_buffer("w_bla_defensive", torch.full((48,), 0.02, dtype=dtype, device=self._device))
        self.register_buffer("w_bla_appetitive", torch.full((48,), 0.02, dtype=dtype, device=self._device))
        self.register_buffer("w_itc_extinction", torch.zeros(48, dtype=dtype, device=self._device))
        self.register_buffer("cue_eligibility", torch.zeros(48, dtype=dtype, device=self._device))

        # Infralimbic mPFC top-down facilitation of ITC extinction neurons
        self.il_pfc_gate = nn.Linear(pallial_dim, 1, dtype=dtype, device=self._device)
        self.out_proj = nn.Linear(48 + 3, embed_dim, dtype=dtype, device=self._device)

        self.register_buffer("cea_activity_ema", torch.tensor(0.05, dtype=dtype, device=self._device))

    def reset_state(self) -> None:
        self.w_bla_defensive.fill_(0.02)
        self.w_bla_appetitive.fill_(0.02)
        self.w_itc_extinction.zero_()
        self.cue_eligibility.zero_()
        self.cea_activity_ema.fill_(0.05)

    def step(
        self,
        extero_cue: torch.Tensor,
        pallial_state: torch.Tensor,
        dt: float,
        *,
        aversive_us_reinforcer: float = 0.0,
        appetitive_us_reinforcer: float = 0.0,
        ne_beta_occupancy: float = 0.15,
        d1_occupancy: float = 0.25,
        ht1a_occupancy: float = 0.50,
        systemic_cortisol: float = 0.18,
        enable_plasticity: bool = True,
    ) -> AmygdalaOutput:
        """Evaluate conditioned cue associations & update single-trial BLA / ITC synapses."""
        # Sparse rectified Lateral Amygdala cue representation
        la_features = F.relu(torch.tanh(self.la_encoder(extero_cue)))

        # Update presynaptic CS cue trace (~0.9s window for trace conditioning!)
        with torch.no_grad():
            a_trace = min(1.0, dt / 0.90)
            self.cue_eligibility.lerp_(la_features.detach(), a_trace)

        bla_def = float(torch.tanh(torch.dot(self.w_bla_defensive, la_features)).item())
        bla_app = float(torch.tanh(torch.dot(self.w_bla_appetitive, la_features)).item())

        # Infralimbic PFC + 5-HT1A facilitates ITC extinction; acute NE_beta + Cortisol stress COLLAPSES ITC!
        il_drive = torch.sigmoid(self.il_pfc_gate(pallial_state)).squeeze(-1)
        stress_relapse_block = max(0.10, 1.0 - 0.75 * ne_beta_occupancy * (0.5 + 0.5 * systemic_cortisol))
        itc_inh = float(
            torch.tanh(torch.dot(self.w_itc_extinction, la_features)).item()
            * (0.45 + 0.55 * float(il_drive.detach().item()))
            * (0.55 + 0.45 * ht1a_occupancy)
            * stress_relapse_block
        )

        # Central Amygdala (CeA) integrates BLA_defensive minus ITC extinction & appetitive safety
        raw_cea = max(0.0, bla_def - 0.85 * itc_inh - 0.25 * bla_app)
        a_cea = min(1.0, dt / 0.12)
        cea_next = (1.0 - a_cea) * float(self.cea_activity_ema.item()) + a_cea * raw_cea
        self.cea_activity_ema.fill_(cea_next)

        # Online Pavlovian BLA & ITC synaptic plasticity
        if enable_plasticity:
            with torch.no_grad():
                # 1. Fast trauma conditioning in BLA_defensive when aversive US strikes (boosted by NE + Cortisol!)
                if aversive_us_reinforcer > 0.04:
                    us_error = max(0.0, aversive_us_reinforcer - bla_def)
                    trauma_lr = 0.28 * (1.0 + 1.4 * ne_beta_occupancy + 0.8 * systemic_cortisol)
                    self.w_bla_defensive.add_(trauma_lr * us_error * self.cue_eligibility * dt)
                    self.w_bla_defensive.clamp_(0.0, 2.5)
                elif bla_def > 0.12 and aversive_us_reinforcer < 0.01:
                    # 2. Safe exposure to previously feared cue builds ITC extinction (does NOT erase w_bla_defensive!)
                    ext_lr = 0.14 * (0.4 + 0.6 * ht1a_occupancy)
                    self.w_itc_extinction.add_(ext_lr * bla_def * la_features.detach() * dt)
                    self.w_itc_extinction.clamp_(0.0, 2.5)

                # 3. Appetitive incentive conditioning in BLA_appetitive gated by D1 occupancy
                if appetitive_us_reinforcer > 0.03:
                    app_err = max(0.0, appetitive_us_reinforcer - bla_app)
                    app_lr = 0.22 * (0.5 + 1.2 * d1_occupancy)
                    self.w_bla_appetitive.add_(app_lr * app_err * self.cue_eligibility * dt)
                    self.w_bla_appetitive.clamp_(0.0, 2.5)

                # Very slow homeostatic decay on ITC & appetitive weights; consolidated defensive traces resist decay
                self.w_bla_appetitive.mul_(max(0.0, 1.0 - 0.002 * dt))
                self.w_itc_extinction.mul_(max(0.0, 1.0 - 0.004 * dt))

        summary_vec = torch.tensor(
            [cea_next, bla_app, itc_inh], dtype=self.dtype, device=self._device
        )
        embed = torch.tanh(self.out_proj(torch.cat([la_features, summary_vec], dim=0)))
        return AmygdalaOutput(
            bla_defensive_raw=bla_def,
            bla_appetitive_raw=bla_app,
            itc_extinction_inhibition=itc_inh,
            cea_defensive_output=cea_next,
            amygdala_embedding=embed,
        )


@dataclass
class EpisodicEngram:
    """Single stored hippocampal episode tagged with somatic outcome & consolidation shield."""

    dg_sparse_code: torch.Tensor        # (256,) L2-normalized sparse Dentate Gyrus index
    ca1_context_vec: torch.Tensor       # (context_dim,) pallial-sensory state snapshot
    extero_cue_snapshot: torch.Tensor   # (extero_dim,) sensory cue for replay re-injection!
    action_taken: torch.Tensor          # (7,) somatic motor vector executed
    somatic_marker: torch.Tensor        # (4,) [valence_relief, visceral_trauma, threat_cea, novelty]
    consolidation_shield: float         # [0, 1] catecholamine/cortisol permanence shield!
    vitality: float                     # Current memory trace strength in (0, 1]
    timestamp: float                    # Simulation time of encoding


@dataclass
class HippocampalStepOutput:
    """Retrieved somatic intuition + autonomous Sharp-Wave Ripple (SWR) replay status."""

    ca1_retrieved_context: torch.Tensor     # (context_dim,) recalled episodic context
    recalled_somatic_marker: torch.Tensor   # (4,) expected [relief, trauma, threat, novelty] from similar past!
    recalled_action_bias: torch.Tensor      # (7,) past successful/habitual motor vector
    swr_replay_active: bool                 # True if CA3 fired an autonomous SWR replay burst this step!
    swr_replay_cue: torch.Tensor | None     # Replayed exteroceptive cue injected into Amygdala/Pallium
    swr_replay_trauma_intensity: float      # Magnitude of replayed trauma/triumph [0, 1]
    active_engram_count: int
    mean_consolidation_shield: float


class HippocampalSomaticMemory(nn.Module):
    """Trisynaptic DG (k-WTA) -> CA3 (Autoassociative Engrams) -> CA1 Somatic Marker & SWR Replay."""

    def __init__(
        self,
        extero_cue_dim: int,
        context_dim: int,
        dg_dim: int = 256,
        dg_k_winners: int = 20,
        max_engrams: int = 128,
        base_decay_per_sec: float = 0.045,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.extero_cue_dim = extero_cue_dim
        self.context_dim = context_dim
        self.dg_dim = dg_dim
        self.dg_k_winners = dg_k_winners
        self.max_engrams = max_engrams
        self.base_decay_per_sec = base_decay_per_sec
        self.dtype = dtype
        self._device = torch.device(device)

        # Fixed random sparse orthogonal Dentate Gyrus expansion (mossy fiber pattern separation!)
        dg_proj = torch.randn(dg_dim, extero_cue_dim + context_dim, dtype=dtype, device=self._device) * (
            1.0 / ((extero_cue_dim + context_dim) ** 0.5)
        )
        self.register_buffer("dg_expansion_matrix", dg_proj)

        self.engrams: List[EpisodicEngram] = []
        self._encode_cooldown: float = 0.0
        self._swr_cooldown: float = 0.0
        self.total_swr_replays: int = 0

    def reset_state(self) -> None:
        self.engrams.clear()
        self._encode_cooldown = 0.0
        self._swr_cooldown = 0.0
        self.total_swr_replays = 0

    def compute_dg_sparse_code(self, extero_cue: torch.Tensor, pallial_context: torch.Tensor) -> torch.Tensor:
        """Dentate Gyrus k-Winners-Take-All orthogonal pattern separation."""
        joint = torch.cat([extero_cue.detach(), pallial_context.detach()], dim=0)
        raw = torch.mv(self.dg_expansion_matrix, joint)
        k = min(self.dg_k_winners, self.dg_dim)
        top_vals, top_idx = torch.topk(raw, k=k, dim=0)
        sparse = torch.zeros_like(raw)
        sparse.scatter_(0, top_idx, F.relu(top_vals) + 1e-4)
        norm = torch.linalg.vector_norm(sparse).clamp(min=1e-6)
        return sparse / norm

    def step(
        self,
        sim_time: float,
        extero_cue: torch.Tensor,
        pallial_context: torch.Tensor,
        prev_action_7d: torch.Tensor,
        dt: float,
        *,
        consolidation_salience_tag: float = 0.0,
        valence_relief: float = 0.0,
        visceral_trauma: float = 0.0,
        cea_threat: float = 0.0,
        epistemic_novelty: float = 0.0,
        ach_m1_encoding_gate: float = 0.30,
        adenosine_a1_sleep: float = 0.10,
        motor_quiescence: float = 1.0,
    ) -> HippocampalStepOutput:
        """Decay traces, retrieve content-matched CA1 somatic markers, encode salient events, & fire SWR replay."""
        self._encode_cooldown = max(0.0, self._encode_cooldown - dt)
        self._swr_cooldown = max(0.0, self._swr_cooldown - dt)

        # 1. Emotional Salience Decay Shielding:
        # Mundane memories (shield ~ 0.05) decay at ~0.043/s; flashbulb traumas (shield ~ 0.98) decay 30x slower!
        surviving: List[EpisodicEngram] = []
        for eng in self.engrams:
            eff_decay = self.base_decay_per_sec * max(0.025, 1.0 - 0.975 * eng.consolidation_shield)
            eng.vitality *= max(0.0, 1.0 - eff_decay * dt)
            if eng.vitality > 0.06:
                surviving.append(eng)
        self.engrams = surviving

        # 2. Pattern-Separated Dentate Gyrus Query -> CA3 Content-Addressable Retrieval
        dg_query = self.compute_dg_sparse_code(extero_cue, pallial_context)

        ca1_ctx = torch.zeros(self.context_dim, dtype=self.dtype, device=self._device)
        somatic_marker = torch.zeros(4, dtype=self.dtype, device=self._device)
        action_bias = torch.zeros(7, dtype=self.dtype, device=self._device)

        if self.engrams:
            sims = torch.stack([torch.dot(dg_query, e.dg_sparse_code) * (0.4 + 0.6 * e.vitality) for e in self.engrams])
            max_sim = float(sims.max().item())
            if max_sim > 0.18:
                attn = F.softmax(sims * 12.0, dim=0)
                for w, eng in zip(attn, self.engrams):
                    wf = float(w.item())
                    if wf > 0.01:
                        ca1_ctx.add_(eng.ca1_context_vec, alpha=wf)
                        somatic_marker.add_(eng.somatic_marker, alpha=wf)
                        # Weight action bias positively if past episode brought relief, negatively if trauma!
                        net_outcome = float(eng.somatic_marker[0].item() - 1.2 * eng.somatic_marker[1].item())
                        action_bias.add_(eng.action_taken, alpha=wf * max(-1.0, min(1.0, net_outcome)))

        # 3. Autonomous CA3 Sharp-Wave Ripple (SWR) Replay during quiet waking / high-adenosine rest!
        # Low ACh + quiet movement disinhibits CA3 recurrent collaterals -> high-salience memories replay!
        swr_active = False
        swr_cue: torch.Tensor | None = None
        swr_intensity = 0.0

        swr_readiness = motor_quiescence * (0.55 + 0.85 * adenosine_a1_sleep) * max(0.2, 1.0 - 0.55 * ach_m1_encoding_gate)
        if self.engrams and self._swr_cooldown <= 0.0 and swr_readiness > 0.42:
            # Select highest-priority consolidated engram (trauma or major relief)
            priorities = [
                e.consolidation_shield * e.vitality * (0.3 + float(e.somatic_marker[1].item() + abs(e.somatic_marker[0].item())))
                for e in self.engrams
            ]
            best_idx = int(max(range(len(priorities)), key=priorities.__getitem__))
            if priorities[best_idx] > 0.14:
                chosen = self.engrams[best_idx]
                swr_active = True
                swr_cue = chosen.extero_cue_snapshot.clone()
                swr_intensity = float(chosen.consolidation_shield * chosen.vitality)
                # Replay strengthens consolidation shield slightly (systems consolidation!)
                chosen.consolidation_shield = min(1.0, chosen.consolidation_shield + 0.04)
                chosen.vitality = min(1.0, chosen.vitality + 0.08)
                ca1_ctx.add_(chosen.ca1_context_vec, alpha=0.55 * swr_intensity)
                somatic_marker.add_(chosen.somatic_marker, alpha=0.55 * swr_intensity)
                self._swr_cooldown = 0.45
                self.total_swr_replays += 1

        # 4. Episodic Encoding into CA3 (gated by Muscarinic M1 ACh or high emotional salience!)
        encode_drive = max(
            consolidation_salience_tag,
            0.45 * ach_m1_encoding_gate + 0.40 * epistemic_novelty + 0.65 * visceral_trauma,
        )
        if self._encode_cooldown <= 0.0 and encode_drive > 0.22:
            marker = torch.tensor(
                [
                    max(-1.5, min(1.5, valence_relief)),
                    max(0.0, min(1.5, visceral_trauma)),
                    max(0.0, min(1.0, cea_threat)),
                    max(0.0, min(1.5, epistemic_novelty)),
                ],
                dtype=self.dtype,
                device=self._device,
            )
            shield = max(0.03, min(1.0, consolidation_salience_tag))

            # Merge if nearly identical DG engram already exists, otherwise insert new episode
            merged = False
            for eng in self.engrams:
                if float(torch.dot(dg_query, eng.dg_sparse_code).item()) > 0.88:
                    eng.somatic_marker.lerp_(marker, 0.5)
                    eng.consolidation_shield = max(eng.consolidation_shield, shield)
                    eng.vitality = 1.0
                    eng.timestamp = sim_time
                    merged = True
                    break

            if not merged:
                if len(self.engrams) >= self.max_engrams:
                    # Evict weakest unconsolidated mundane memory, NEVER consolidated trauma!
                    weakest_idx = min(
                        range(len(self.engrams)),
                        key=lambda i: self.engrams[i].vitality * (0.1 + self.engrams[i].consolidation_shield),
                    )
                    self.engrams.pop(weakest_idx)

                self.engrams.append(
                    EpisodicEngram(
                        dg_sparse_code=dg_query,
                        ca1_context_vec=pallial_context.detach().clone(),
                        extero_cue_snapshot=extero_cue.detach().clone(),
                        action_taken=prev_action_7d.detach().clone().to(dtype=self.dtype, device=self._device),
                        somatic_marker=marker,
                        consolidation_shield=shield,
                        vitality=1.0,
                        timestamp=sim_time,
                    )
                )
            self._encode_cooldown = 0.20

        mean_shield = (
            sum(e.consolidation_shield for e in self.engrams) / len(self.engrams)
            if self.engrams
            else 0.0
        )
        return HippocampalStepOutput(
            ca1_retrieved_context=ca1_ctx,
            recalled_somatic_marker=somatic_marker,
            recalled_action_bias=action_bias,
            swr_replay_active=swr_active,
            swr_replay_cue=swr_cue,
            swr_replay_trauma_intensity=swr_intensity,
            active_engram_count=len(self.engrams),
            mean_consolidation_shield=mean_shield,
        )
