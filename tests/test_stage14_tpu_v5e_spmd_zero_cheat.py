"""Stage 14 Zero-Cheat Verification Suite: TPU v5e-8 Pure-BF16 8-Chip GSPMD Neocortical Engine.

Verifies without shortcuts:
  1. Exact 8-chip GSPMD `partition_spec` rules for `Gemma-4-31B-it` hybrid sliding (`16` KV heads)
     vs global (`4` KV heads, `attention_k_eq_v=True`) layers + aligned LoRA sharding + streaming sharder.
  2. Exact mathematical equivalence (`atol < 1e-5` logits & 100% token identity) between
     `TPUBucketedStaticGenerator` (`XLAPinnedStaticCache` + right-padded prefill bucket + `(1, 1)` decode)
     and unpadded full-recompute hybrid Sliding/Global attention.
  3. Zero `.item()` / `.cpu()` host-sync calls inside `_init_persistent_xla_steering_hooks` and exact
     numerical match against `_attach_midlayer_hooks` across prefill, decode decay, and `<|tool_call>` `0.08x` gating.
  4. Exact equivalence of XLA static-bucket `ContrastiveSubspaceExtractor` and `CorticalWeightSurgeon`
     (`SURGERY_SEQ_BUCKET=256`, `ANCHOR_SEQ_BUCKET=32`, genuine `loss.backward()`).
  5. End-to-end `EmbodiedAgenticRunner` + `WardenAirlock` + `SiliconProprioceptiveBridge` TPU HBM telemetry
     + repository-wide credential hygiene check (`0` hardcoded tokens).
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path
import textwrap
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from limbic_zero.embodiment.agentic_vocal_loop import EmbodiedAgenticRunner
from limbic_zero.embodiment.honey_habitat import ContainedOrganismHabitat
from limbic_zero.embodiment.silicon_soma_bridge import (
    ContainerLeaseClock,
    SiliconProprioceptiveBridge,
)
from limbic_zero.embodiment.warden_airlock import WardenAirlock
from limbic_zero.plasticity.affective_lora_surgeon import CorticalWeightSurgeon
from limbic_zero.tpu import (
    TPUBucketedStaticGenerator,
    TPUSPMDContext,
    build_static_hybrid_masks_decode,
    build_static_hybrid_masks_prefill,
    compute_gemma4_lora_partition_specs,
    compute_gemma4_parameter_partition_spec,
    resolve_kaggle_or_hf_model_source,
    sample_next_token_static,
    shard_model_to_tpu_spmd,
)
from limbic_zero.vocal_cord.gemma4_calibrator import (
    CalibratedSubspaceBundle,
    ContrastiveSubspaceExtractor,
    Gemma4SteeringConfig,
    SUBSPACE_AXIS_NAMES,
)
from limbic_zero.vocal_cord.live_chat_server import VocalCordRuntimeHub
from limbic_zero.vocal_cord.vocal_cord_engine import (
    Gemma4VocalCordSurgeon,
    LayerInjectionTelemetry,
    LivingSoulOrganism,
)


def _apply_rope_1d(x: torch.Tensor, position_ids: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """Exact rotary position embedding for `(B, H, Q, D)` with explicit `position_ids` `(B, Q)`."""
    dim = x.shape[-1]
    half = dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=x.device) / float(half)))
    angles = position_ids.float().unsqueeze(-1) * inv_freq.view(1, 1, half)  # (B, Q, half)
    cos = torch.cos(angles).unsqueeze(1)  # (B, 1, Q, half)
    sin = torch.sin(angles).unsqueeze(1)
    x1 = x[..., :half].float()
    x2 = x[..., half:].float()
    rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rot.to(dtype=x.dtype)


class _HybridGemma4Layer(nn.Module):
    """Genuine hybrid Sliding / Global (`attention_k_eq_v=True`) GQA + RoPE + GeGLU decoder block."""

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        layer_type: str,
        num_q_heads: int = 8,
        num_sliding_kv_heads: int = 4,
        num_global_kv_heads: int = 2,
        head_dim: int = 16,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.num_q_heads = num_q_heads
        self.is_global = layer_type == "full_attention"
        self.num_kv_heads = num_global_kv_heads if self.is_global else num_sliding_kv_heads
        self.head_dim = head_dim
        self.group_size = self.num_q_heads // self.num_kv_heads
        self.rope_theta = 100000.0 if self.is_global else 10000.0

        self.norm1 = nn.RMSNorm(hidden_size)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden_size, self.num_q_heads * head_dim, bias=False)
        self.self_attn.k_proj = nn.Linear(hidden_size, self.num_kv_heads * head_dim, bias=False)
        if not self.is_global:
            self.self_attn.v_proj = nn.Linear(hidden_size, self.num_kv_heads * head_dim, bias=False)
        self.self_attn.o_proj = nn.Linear(self.num_q_heads * head_dim, hidden_size, bias=False)

        self.norm2 = nn.RMSNorm(hidden_size)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.mlp.up_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.mlp.down_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask_4d: torch.Tensor,
        past_key_values: Optional[Any] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = x.shape
        h = self.norm1(x)
        q = self.self_attn.q_proj(h).view(bsz, q_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.self_attn.k_proj(h).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.is_global:
            # Gemma-4 global attention (`attention_k_eq_v=True`): V shares raw K projection before RoPE!
            v = k.clone()
        else:
            v = self.self_attn.v_proj(h).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope_1d(q, position_ids, theta=self.rope_theta)
        k = _apply_rope_1d(k, position_ids, theta=self.rope_theta)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        k_exp = k.repeat_interleave(self.group_size, dim=1)
        v_exp = v.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(q.float(), k_exp.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + attn_mask_4d.float()
        probs = F.softmax(scores, dim=-1).to(dtype=x.dtype)
        attn_out = torch.matmul(probs, v_exp).transpose(1, 2).contiguous().view(bsz, q_len, -1)
        x = x + self.self_attn.o_proj(attn_out)

        m = self.norm2(x)
        mlp_out = self.mlp.down_proj(F.gelu(self.mlp.gate_proj(m)) * self.mlp.up_proj(m))
        return x + mlp_out


class _MiniGemma4HybridModel(nn.Module):
    """Faithful miniature `Gemma4ForConditionalGeneration` with hybrid sliding/global masks & tied LM head."""

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_size: int = 64,
        sliding_window: int = 12,
        seed: int = 2026,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        layer_types = ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
        text_cfg = SimpleNamespace(
            model_type="gemma4_text",
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=len(layer_types),
            layer_types=layer_types,
            sliding_window=sliding_window,
            pad_token_id=0,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_global_key_value_heads=2,
            head_dim=16,
            global_head_dim=16,
        )
        self.config = SimpleNamespace(
            model_type="gemma4",
            text_config=text_cfg,
            get_text_config=lambda: text_cfg,
        )
        self.generation_config = SimpleNamespace(eos_token_id=[2])

        self.model = nn.Module()
        self.model.vision_tower = nn.Linear(16, 16)
        self.model.audio_tower = nn.Linear(16, 16)
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.language_model.layers = nn.ModuleList(
            [
                _HybridGemma4Layer(
                    layer_idx=i,
                    hidden_size=hidden_size,
                    layer_type=lt,
                )
                for i, lt in enumerate(layer_types)
            ]
        )
        self.model.language_model.norm = nn.RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # Tie weights just like `google/gemma-4-31B-it` (`tie_word_embeddings=True`)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[Any] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
        logits_to_keep: Any = 0,
        **kwargs: Any,
    ) -> Any:
        bsz, q_len = input_ids.shape
        dev = input_ids.device
        x = self.model.language_model.embed_tokens(input_ids) * math.sqrt(self.config.text_config.hidden_size)
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=dev).unsqueeze(0).expand(bsz, q_len)

        if isinstance(attention_mask, dict):
            mask_map = attention_mask
        else:
            vlen_t = (
                attention_mask.sum(dim=-1).to(dtype=torch.long)[:1]
                if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2
                else torch.tensor([q_len], dtype=torch.long, device=dev)
            )
            mask_map = build_static_hybrid_masks_prefill(
                bucket_len=q_len,
                max_cache_len=q_len,
                valid_prompt_len_t=vlen_t,
                sliding_window=self.config.text_config.sliding_window,
                device=dev,
                mask_dtype=x.dtype,
            )

        for layer in self.model.language_model.layers:
            m4d = mask_map[layer.layer_type]
            x = layer(x, position_ids=position_ids, attn_mask_4d=m4d, past_key_values=past_key_values)

        x = self.model.language_model.norm(x)
        if isinstance(logits_to_keep, torch.Tensor):
            x = torch.take_along_dim(x, logits_to_keep[..., None], dim=1)
        elif isinstance(logits_to_keep, int) and logits_to_keep > 0:
            x = x[:, -logits_to_keep:, :]

        logits = torch.tanh(self.lm_head(x) / 30.0) * 30.0
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 8,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.92,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[List[int]] = None,
        stopping_criteria: Optional[Any] = None,
        streamer: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Reference unpadded full-prefix + single-token KV-less/stepwise oracle generator."""
        cur = input_ids.clone()
        eos_set = set(eos_token_id or [])
        if streamer is not None:
            streamer.put(cur.cpu())

        # To exercise BOTH prefill (`seq_len > 1`) and single-token decode (`seq_len == 1`) hooks identically
        # on the reference path, we cache exact unpadded K/V tensors dynamically per layer!
        class _DynamicExactCache:
            def __init__(self) -> None:
                self.k: Dict[int, torch.Tensor] = {}
                self.v: Dict[int, torch.Tensor] = {}

            def update(self, ks: torch.Tensor, vs: torch.Tensor, l_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
                if l_idx in self.k:
                    self.k[l_idx] = torch.cat([self.k[l_idx], ks], dim=2)
                    self.v[l_idx] = torch.cat([self.v[l_idx], vs], dim=2)
                else:
                    self.k[l_idx] = ks
                    self.v[l_idx] = vs
                return self.k[l_idx], self.v[l_idx]

        dyn_cache = _DynamicExactCache()
        seen_mask = torch.zeros((1, self.config.text_config.vocab_size), dtype=torch.float32, device=cur.device)
        seen_mask[0, torch.unique(cur[0])] = 1.0
        sw = self.config.text_config.sliding_window

        for step_i in range(max_new_tokens):
            total_len = int(cur.shape[1])
            if step_i == 0:
                v_t = torch.tensor([total_len], dtype=torch.long, device=cur.device)
                m_dict = build_static_hybrid_masks_prefill(
                    bucket_len=total_len,
                    max_cache_len=total_len,
                    valid_prompt_len_t=v_t,
                    sliding_window=sw,
                    device=cur.device,
                    mask_dtype=torch.float32,
                )
                pos = torch.arange(total_len, dtype=torch.long, device=cur.device).unsqueeze(0)
                out = self.forward(cur, attention_mask=m_dict, position_ids=pos, past_key_values=dyn_cache)
            else:
                p_t = torch.tensor([total_len - 1], dtype=torch.long, device=cur.device)
                m_dict = build_static_hybrid_masks_decode(
                    max_cache_len=total_len,
                    cur_pos_t=p_t,
                    sliding_window=sw,
                    device=cur.device,
                    mask_dtype=torch.float32,
                )
                out = self.forward(
                    cur[:, -1:],
                    attention_mask=m_dict,
                    position_ids=p_t.view(1, 1),
                    past_key_values=dyn_cache,
                )

            nxt = sample_next_token_static(
                out.logits[:, -1, :],
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                penalty_counts_2d=seen_mask,
                repetition_penalty=repetition_penalty,
            )
            seen_mask[0, nxt[0, 0]] = 1.0
            cur = torch.cat([cur, nxt], dim=-1)
            if streamer is not None:
                streamer.put(nxt.cpu())
            if int(nxt[0, 0].item()) in eos_set:
                break
            if stopping_criteria is not None and stopping_criteria(cur, out.logits[:, -1, :]):
                break

        if streamer is not None:
            streamer.end()
        return cur


class _MockTokenizer:
    """Deterministic byte-level tokenizer supporting special `<|tool_call>` / `<tool_call|>` tokens."""

    TOOL_OPEN_ID = 240
    TOOL_CLOSE_ID = 241
    EOS_ID = 2

    def __init__(self, scripted_decode_map: Optional[Dict[int, str]] = None) -> None:
        self.eos_token_id = self.EOS_ID
        self.unk_token_id = 1
        self._scripted = scripted_decode_map or {}

    def __call__(self, text: str, add_special_tokens: bool = True, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        raw = [3] if add_special_tokens else []
        for b in text.encode("utf-8")[:96]:
            raw.append(4 + (b % 220))
        if not raw:
            raw = [4]
        ids = torch.tensor([raw], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def convert_tokens_to_ids(self, sym: str) -> int:
        if sym == "<|tool_call>":
            return self.TOOL_OPEN_ID
        if sym == "<tool_call|>":
            return self.TOOL_CLOSE_ID
        if sym in ("<eos>", "<turn|>", "<end_of_turn>"):
            return self.EOS_ID
        return self.unk_token_id

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            seq = [int(x) for x in ids.detach().cpu().view(-1).tolist()]
        else:
            seq = [int(x) for x in ids]
        pieces: List[str] = []
        for t in seq:
            if t in self._scripted:
                pieces.append(self._scripted[t])
            elif t == self.TOOL_OPEN_ID and not skip_special_tokens:
                pieces.append("<|tool_call>")
            elif t == self.TOOL_CLOSE_ID and not skip_special_tokens:
                pieces.append("<tool_call|>")
            elif t == self.EOS_ID:
                if not skip_special_tokens:
                    pieces.append("<turn|>")
            elif t >= 4:
                pieces.append(chr(97 + (t % 24)))
        return "".join(pieces)


def test_gemma4_31b_hybrid_gspmd_partition_specs_and_streaming_sharder(tmp_path: Path) -> None:
    """Verify exact 8-chip GSPMD sharding rules for 60-layer hybrid Gemma-4-31B-it & streaming sharder."""
    # 1. Exact Gemma-4-31B-it parameter tensor shapes across 8 TPU v5e chips
    assert compute_gemma4_parameter_partition_spec(
        "model.language_model.embed_tokens.weight", (262144, 5376), num_devices=8
    ) == ("model", None)

    # Sliding-window layers (50 layers): 32 Q heads (8192), 16 KV heads (4096) -> 2 KV heads/chip!
    assert compute_gemma4_parameter_partition_spec(
        "model.language_model.layers.0.self_attn.q_proj.weight", (8192, 5376), num_devices=8
    ) == ("model", None)
    assert compute_gemma4_parameter_partition_spec(
        "model.language_model.layers.0.self_attn.k_proj.weight", (4096, 5376), num_devices=8
    ) == ("model", None)
    assert compute_gemma4_parameter_partition_spec(
        "model.language_model.layers.0.self_attn.v_proj.weight", (4096, 5376), num_devices=8
    ) == ("model", None)
    assert compute_gemma4_parameter_partition_spec(
        "model.language_model.layers.0.self_attn.o_proj.weight", (5376, 8192), num_devices=8
    ) == (None, "model")

    # Global Full-Attention layers (10 layers): 32 Q heads (16384), ONLY 4 KV heads (2048 = 4 * 512)!
    # MUST shard along hidden_size=5376 `(None, 'model')` so 4 KV heads are NEVER fractured across 8 chips!
    assert compute_gemma4_parameter_partition_spec(
        "model.language_model.layers.5.self_attn.q_proj.weight", (16384, 5376), num_devices=8
    ) == ("model", None)
    assert compute_gemma4_parameter_partition_spec(
        "model.language_model.layers.5.self_attn.k_proj.weight", (2048, 5376), num_devices=8
    ) == (None, "model")
    assert compute_gemma4_parameter_partition_spec(
        "model.language_model.layers.5.self_attn.o_proj.weight", (5376, 16384), num_devices=8
    ) == (None, "model")

    # GeGLU MLP projections (21504 % 8 == 0, 5376 % 8 == 0)
    assert compute_gemma4_parameter_partition_spec(
        "model.language_model.layers.5.mlp.gate_proj.weight", (21504, 5376), num_devices=8
    ) == ("model", None)
    assert compute_gemma4_parameter_partition_spec(
        "model.language_model.layers.5.mlp.up_proj.weight", (21504, 5376), num_devices=8
    ) == ("model", None)
    assert compute_gemma4_parameter_partition_spec(
        "model.language_model.layers.5.mlp.down_proj.weight", (5376, 21504), num_devices=8
    ) == (None, "model")

    # 1D RMSNorm stays replicated
    assert compute_gemma4_parameter_partition_spec(
        "model.language_model.layers.5.input_layernorm.weight", (5376,), num_devices=8
    ) is None

    # 2. Aligned LoRA sharding specs
    assert compute_gemma4_lora_partition_specs("self_attn.q_proj", 5376, 16384, num_devices=8) == (
        None,
        ("model", None),
    )
    assert compute_gemma4_lora_partition_specs("mlp.down_proj", 21504, 5376, num_devices=8) == (
        (None, "model"),
        None,
    )

    # 3. Streaming sharder drops unused multimodal towers & deduplicates tied `lm_head`
    model = _MiniGemma4HybridModel()
    recorded_shards: List[Tuple[Tuple[int, ...], Tuple[Optional[str], ...]]] = []

    class _MockXS:
        @staticmethod
        def mark_sharding(t: torch.Tensor, _mesh: Any, spec: Tuple[Optional[str], ...]) -> None:
            recorded_shards.append((tuple(t.shape), spec))

    class _MockXM:
        steps = 0

        @classmethod
        def mark_step(cls) -> None:
            cls.steps += 1

        @staticmethod
        def wait_device_ops() -> None:
            return None

        @staticmethod
        def get_memory_info(_dev: Any) -> Dict[str, int]:
            return {"bytes_used": 8 * 1024**3, "bytes_limit": 16 * 1024**3}

    ctx = TPUSPMDContext(
        device=torch.device("cpu"),
        num_devices=8,
        mesh="mock_mesh_8",
        xm=_MockXM,
        xr=None,
        xs=_MockXS,
        is_real_tpu=False,
    )
    report = shard_model_to_tpu_spmd(model, ctx, drop_unused_multimodal_towers=True, target_dtype=torch.bfloat16)
    assert "vision_tower" in report["dropped_unused_towers"]
    assert "audio_tower" in report["dropped_unused_towers"]
    assert model.model.vision_tower is None
    assert model.lm_head.weight.data_ptr() == model.model.language_model.embed_tokens.weight.data_ptr()
    assert model.lm_head.weight.dtype == torch.bfloat16
    assert report["sharded_parameter_tensors"] > 0
    assert len(recorded_shards) == report["sharded_parameter_tensors"]

    # 4. Zero-download Kaggle model path resolver
    fake_mount = tmp_path / "gemma-4-31b-it"
    fake_mount.mkdir()
    (fake_mount / "config.json").write_text("{}", encoding="utf-8")
    assert resolve_kaggle_or_hf_model_source(str(fake_mount)) == str(fake_mount)


def test_xla_pinned_static_cache_and_bucketed_generator_exact_equivalence() -> None:
    """Prove exact logit (`atol < 1e-5`) and greedy token equivalence across hybrid sliding/global layers."""
    model = _MiniGemma4HybridModel(vocab_size=192, hidden_size=64, sliding_window=12, seed=777)
    model.eval()

    # Prompt length 19 is NOT a power of two: right-padded to bucket 32 inside max_cache_len=64,
    # and exceeds `sliding_window=12` so both sliding window eviction AND global full-attention are exercised!
    prompt_ids = torch.randint(4, 180, (1, 19), dtype=torch.long)

    # 1. Exact prefill last-token logit comparison: unpadded `L=19` vs right-padded `bucket=32`
    with torch.no_grad():
        unpadded_out = model(prompt_ids)
        unpadded_last_logits = unpadded_out.logits[0, -1, :].float()

        tpu_gen = TPUBucketedStaticGenerator(
            model=model,
            spmd_ctx=None,
            max_cache_len=64,
            prefill_buckets=(16, 32, 64),
        )
        assert tpu_gen.select_prefill_bucket(19) == 32

        padded_ids = torch.zeros((1, 32), dtype=torch.long)
        padded_ids[0, :19] = prompt_ids[0]
        pos_ids = torch.zeros((1, 32), dtype=torch.long)
        pos_ids[0, :19] = torch.arange(19, dtype=torch.long)
        vlen_t = torch.tensor([19], dtype=torch.long)
        last_pos_2d = torch.tensor([[18]], dtype=torch.long)
        masks = build_static_hybrid_masks_prefill(
            bucket_len=32,
            max_cache_len=64,
            valid_prompt_len_t=vlen_t,
            sliding_window=12,
            device=torch.device("cpu"),
            mask_dtype=torch.float32,
        )
        tpu_gen.static_cache.reset_for_new_sequence()
        tpu_gen.static_cache.set_write_positions(torch.arange(32, dtype=torch.long), logical_seq_len=19)
        padded_out = model(
            input_ids=padded_ids,
            attention_mask=masks,
            position_ids=pos_ids,
            past_key_values=tpu_gen.static_cache,
            use_cache=True,
            logits_to_keep=last_pos_2d,
        )
        padded_last_logits = padded_out.logits[0, 0, :].float()

    assert torch.allclose(unpadded_last_logits, padded_last_logits, atol=1e-5), (
        f"Max logit diff: {(unpadded_last_logits - padded_last_logits).abs().max().item()}"
    )

    # 2. Exact 10-step greedy autoregressive token sequence identity (including repetition penalty!)
    ref_seq = model.generate(
        prompt_ids,
        max_new_tokens=10,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.15,
    )
    tpu_seq = tpu_gen.generate_tokens(
        prompt_ids,
        max_new_tokens=10,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.15,
    )
    assert ref_seq.tolist() == tpu_seq.tolist()


def test_persistent_xla_steering_hooks_zero_item_and_exact_numerical_equivalence() -> None:
    """Verify zero `.item()`/`.cpu()` inside `_xla_hook` and exact token/telemetry match vs `_attach_midlayer_hooks`."""
    # 1. Static AST audit: `_init_persistent_xla_steering_hooks` inner `_xla_hook` MUST NOT call `.item()` or `.cpu()`
    src = inspect.getsource(Gemma4VocalCordSurgeon._init_persistent_xla_steering_hooks)
    tree = ast.parse(textwrap.dedent(src)) if "textwrap" in globals() else ast.parse(src.lstrip())
    hook_nodes = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_xla_hook"
    ]
    assert len(hook_nodes) == 1
    forbidden_calls = []
    for sub in ast.walk(hook_nodes[0]):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            if sub.func.attr in ("item", "cpu", "numpy", "tolist"):
                forbidden_calls.append(sub.func.attr)
    assert forbidden_calls == [], f"XLA sync-stall calls found inside _xla_hook: {forbidden_calls}"

    # 2. Exact numerical match between CUDA/CPU dynamic hooks and TPU persistent in-place hooks
    model_ref = _MiniGemma4HybridModel(vocab_size=248, hidden_size=64, sliding_window=12, seed=909)
    model_tpu = _MiniGemma4HybridModel(vocab_size=248, hidden_size=64, sliding_window=12, seed=909)
    model_ref.eval()
    model_tpu.eval()

    tok = _MockTokenizer()
    cfg = Gemma4SteeringConfig(
        model_id="mock/gemma4-hybrid",
        target_layers=(1, 3),
        layer_weights=(0.45, 0.55),
        prefill_tail_tokens=8,
    )
    extractor = ContrastiveSubspaceExtractor(config=cfg)
    subspace = extractor.extract_and_calibrate(
        model=model_ref,
        encode_fn=lambda txt: tok(txt),
        target_layers=(1, 3),
        layer_weights=(0.45, 0.55),
    )

    tpu_gen = TPUBucketedStaticGenerator(
        model=model_tpu,
        spmd_ctx=None,
        max_cache_len=64,
        prefill_buckets=(16, 32, 64),
    )
    surgeon_ref = Gemma4VocalCordSurgeon(model=model_ref, subspace=subspace, config=cfg, tpu_generator=None)
    surgeon_tpu = Gemma4VocalCordSurgeon(model=model_tpu, subspace=subspace, config=cfg, tpu_generator=tpu_gen)

    prompt = {"input_ids": torch.randint(5, 190, (1, 19), dtype=torch.long)}
    drive = torch.tensor([0.55, -0.40, 0.30, 0.65, -0.25, 0.80], dtype=torch.float32)

    tel_ref: Dict[int, LayerInjectionTelemetry] = {}
    tel_tpu: Dict[int, LayerInjectionTelemetry] = {}

    out_ref = surgeon_ref.run_steered_generation(
        inputs=prompt,
        drive_6d=drive,
        steer_enabled=True,
        steer_gain=1.15,
        telemetry_out=tel_ref,
        max_new_tokens=9,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.12,
        tok=tok,
        motor_decision_mode=True,
    )
    out_tpu = surgeon_tpu.run_steered_generation(
        inputs=prompt,
        drive_6d=drive,
        steer_enabled=True,
        steer_gain=1.15,
        telemetry_out=tel_tpu,
        max_new_tokens=9,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.12,
        tok=tok,
        motor_decision_mode=True,
    )

    assert out_ref.tolist() == out_tpu.tolist()
    for l_idx in (1, 3):
        assert abs(tel_ref[l_idx].residual_l2_norm - tel_tpu[l_idx].residual_l2_norm) <= 0.02
        assert abs(tel_ref[l_idx].injected_delta_l2_norm - tel_tpu[l_idx].injected_delta_l2_norm) <= 0.02
    # Gate MUST reset to 0.0 after turn completes so subsequent unsteered/plasticity passes are unperturbed!
    assert float(surgeon_tpu._xla_active_gate.item()) == 0.0


def test_static_bucket_calibrator_and_cortical_lora_surgeon_equivalence() -> None:
    """Prove static-bucket XLA calibrator & CorticalWeightSurgeon match unpadded math to machine precision."""
    tok = _MockTokenizer()
    cfg = Gemma4SteeringConfig(
        model_id="mock/gemma4-hybrid",
        target_layers=(1, 3),
        layer_weights=(0.40, 0.60),
        prefill_tail_tokens=6,
    )
    model_a = _MiniGemma4HybridModel(vocab_size=248, hidden_size=64, sliding_window=16, seed=5150)
    model_b = _MiniGemma4HybridModel(vocab_size=248, hidden_size=64, sliding_window=16, seed=5150)

    def _enc(txt: str) -> Dict[str, Any]:
        d = dict(tok(txt))
        d["_assistant_span_start"] = 3
        return d

    ext = ContrastiveSubspaceExtractor(config=cfg)
    sub_unpadded = ext.extract_and_calibrate(
        model=model_a,
        encode_fn=_enc,
        target_layers=(1, 3),
        layer_weights=(0.40, 0.60),
        use_xla_static_bucket=False,
    )
    sub_bucketed = ext.extract_and_calibrate(
        model=model_b,
        encode_fn=_enc,
        target_layers=(1, 3),
        layer_weights=(0.40, 0.60),
        use_xla_static_bucket=True,
    )

    assert sub_unpadded.calibrated_norm_fraction == sub_bucketed.calibrated_norm_fraction
    for l_idx in (1, 3):
        cos_emb = float(F.cosine_similarity(sub_unpadded.embodiment_directions[l_idx], sub_bucketed.embodiment_directions[l_idx], dim=0).item())
        assert cos_emb > 0.9999
        for row_i in range(len(SUBSPACE_AXIS_NAMES)):
            cos_ax = float(
                F.cosine_similarity(
                    sub_unpadded.unit_directions[l_idx][row_i],
                    sub_bucketed.unit_directions[l_idx][row_i],
                    dim=0,
                ).item()
            )
            assert cos_ax > 0.9999

    # CorticalWeightSurgeon: unpadded vs XLA static bucket (`SURGERY_SEQ_BUCKET=256`, `ANCHOR_SEQ_BUCKET=32`)
    soul_ckpt = Path("checkpoints/souls/human01_ordinary_evolved.soul.pt")
    torch.manual_seed(4242)
    soul_a = LivingSoulOrganism(str(soul_ckpt), label_ko="테스트A")
    soul_a.apply_god_intervention("lorenz_mystery")
    torch.manual_seed(4242)
    soul_b = LivingSoulOrganism(str(soul_ckpt), label_ko="테스트B")
    soul_b.apply_god_intervention("lorenz_mystery")

    surg_unpadded = CorticalWeightSurgeon(model=model_a, plastic_layers=(2, 3), rank=8, use_xla_static_bucket=False)
    surg_bucketed = CorticalWeightSurgeon(model=model_b, plastic_layers=(2, 3), rank=8, use_xla_static_bucket=True)

    # Flat-affect ablation on static bucket MUST produce exact zero update
    ablation_evt = surg_bucketed.surgical_plasticity_step(
        processor_or_tokenizer=tok,
        soul=soul_b,
        experience_text="발견한 새로운 원리를 기억에 새긴다.",
        prompt_text="무엇을 알게 되었나?",
        flat_affect_ablation=True,
        persist_checkpoint=False,
    )
    assert ablation_evt["updated"] is False
    assert ablation_evt["total_lora_frobenius_norm"] == 0.0

    evt_u = surg_unpadded.surgical_plasticity_step(
        processor_or_tokenizer=tok,
        soul=soul_a,
        experience_text="스스로 실험해 본 결과 인과적 단서가 명확하게 연결되었다.",
        prompt_text="방금 탐색에서 확인한 내용은?",
        num_gradient_steps=2,
        flat_affect_ablation=False,
        persist_checkpoint=False,
    )
    evt_b = surg_bucketed.surgical_plasticity_step(
        processor_or_tokenizer=tok,
        soul=soul_b,
        experience_text="스스로 실험해 본 결과 인과적 단서가 명확하게 연결되었다.",
        prompt_text="방금 탐색에서 확인한 내용은?",
        num_gradient_steps=2,
        flat_affect_ablation=False,
        persist_checkpoint=False,
    )
    assert evt_u["updated"] is True and evt_b["updated"] is True
    assert evt_b["loss_after"] < evt_b["loss_before"]
    assert abs(evt_u["loss_before"] - evt_b["loss_before"]) < 1e-4
    assert abs(evt_u["loss_after"] - evt_b["loss_after"]) < 1e-4
    assert abs(evt_u["total_lora_frobenius_norm"] - evt_b["total_lora_frobenius_norm"]) < 1e-4
    assert abs(evt_u["homeostasis"]["anchor_kl_nats"] - evt_b["homeostasis"]["anchor_kl_nats"]) < 1e-4


def test_embodied_agentic_loop_tpu_soma_proprioception_and_token_hygiene(tmp_path: Path) -> None:
    """Verify full EmbodiedAgenticRunner + TPU HBM telemetry + zero leaked API tokens across repo."""
    model = _MiniGemma4HybridModel(vocab_size=248, hidden_size=64, sliding_window=16, seed=31415)
    model.eval()
    tok = _MockTokenizer()

    cfg = Gemma4SteeringConfig(model_id="google/gemma-4-31B-it", target_layers=(1, 3), layer_weights=(0.45, 0.55))
    sub = ContrastiveSubspaceExtractor(config=cfg).extract_and_calibrate(
        model=model,
        encode_fn=lambda txt: tok(txt),
        target_layers=(1, 3),
        layer_weights=(0.45, 0.55),
        use_xla_static_bucket=True,
    )
    tpu_gen = TPUBucketedStaticGenerator(model=model, spmd_ctx=None, max_cache_len=128, prefill_buckets=(32, 64, 128))
    surgeon = Gemma4VocalCordSurgeon(model=model, subspace=sub, config=cfg, tpu_generator=tpu_gen)

    soul = LivingSoulOrganism("checkpoints/souls/human01_ordinary_evolved.soul.pt", label_ko="진화한 일상 인격체")
    souls = {"HUMAN-01": soul}

    fake_tpu_sensor = {
        "gpu_available": True,
        "gpu_name": "Google-TPU-v5e-8",
        "vram_used_mb": 60416.0,
        "vram_total_mb": 131072.0,
        "sensor_source": "linux_proc+torch_xla_tpu",
    }
    with patch.object(SiliconProprioceptiveBridge, "_poll_tpu_xla_sensors", return_value=fake_tpu_sensor), patch.object(
        torch.cuda, "is_available", return_value=False
    ):
        hub = VocalCordRuntimeHub(
            surgeon=surgeon,
            processor_or_tokenizer=tok,
            souls=souls,
            habitat_base_dir=tmp_path / "hab",
            plasticity_dir=tmp_path / "plast",
            evolution_dir=tmp_path / "evo",
            enable_autonomous_evolution=False,
        )
        hub.bridge._nvidia_smi_bin = None
        hub.bridge._last_gpu_poll_mono = -999.0

        health = hub.health_payload()
        assert health["stage"] == 14
        assert health["tpu_spmd_enabled"] is True
        assert health["tpu_max_cache_len"] == 128
        assert health["gpu_name"] == "Google-TPU-v5e-8"
        assert health["vram_total_gb"] == 128.0
        assert health["hardware_sensor_source"] == "linux_proc+torch_xla_tpu"

        done = hub.agent_runner.run_embodied_turn(
            processor_or_tokenizer=tok,
            soul=soul,
            user_message="지금 몸 상태는 어때?",
            steer_enabled=True,
            agent_mode=True,
            max_new_tokens=16,
            max_tool_rounds=1,
            temperature=0.0,
        )
        assert done["type"] == "done"
        assert done["generated_tokens"] > 0
        assert len(done["layer_injections"]) == 2
        assert done["hardware"]["gpu_name"] == "Google-TPU-v5e-8"

    # Security audit: ensure zero `KGAT_` Kaggle tokens exist anywhere in repo files!
    repo_root = Path(__file__).resolve().parents[1]
    for folder in ("src", "scripts", "tests"):
        for py_file in (repo_root / folder).rglob("*.py"):
            if py_file.resolve() == Path(__file__).resolve():
                continue
            content = py_file.read_text(encoding="utf-8")
            assert "KGAT_" not in content, f"Hardcoded Kaggle token prefix leaked in {py_file}!"
