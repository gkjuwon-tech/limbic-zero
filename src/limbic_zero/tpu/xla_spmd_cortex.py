"""Stage 14 Google Cloud TPU v5e-8 Pure-BF16 8-Chip GSPMD Neocortical Engine.

Implements zero-quantization, zero-recompilation, 8-chip Megatron/GSPMD tensor-parallel
execution for ``google/gemma-4-31B-it`` (``~58.3 GiB`` pure ``bfloat16``) on a single-host
``TPU VM v5e-8`` (``8 × 16 GB = 128 GB HBM``, ``~6.5 TB/s`` aggregate bandwidth, ``~330 GB``
Host Xeon RAM), coupled across a 24-byte Corpus Callosum bridge to the Host-CPU biophysical
``DigitalSoma`` and ``LimbicBrain`` ODEs.

Key Engineering Guarantees (Zero-Cheat & Zero-Recompile):
  1. **Streaming Layer-by-Layer GSPMD Sharding**:
     Parameters are moved from ``330 GB`` Host CPU RAM onto ``xm.xla_device()`` one tensor at
     a time and immediately annotated with ``xs.mark_sharding`` + periodic ``xm.mark_step()``,
     preventing single-chip ``16 GB`` HBM spikes while distributing ``58.3 GiB`` evenly
     (``~7.3 GiB / chip``) across all 8 TPU v5e cores.
  2. **Hybrid Sliding/Global Attention Head-Exact Partitioning**:
     - ``q_proj`` (32 heads) & ``gate_proj`` / ``up_proj`` (``21504``): column-parallel ``('model', None)``
     - ``o_proj`` (32 heads) & ``down_proj`` (``21504``): row-parallel ``(None, 'model')``
     - Sliding ``k_proj`` / ``v_proj`` (16 KV heads): column-parallel ``('model', None)`` (2 heads/chip)
     - Global ``k_proj`` (4 KV heads, ``attention_k_eq_v=True``): row-parallel ``(None, 'model')`` on
       ``hidden_size=5376`` so 4 global KV heads are never split fractionally across 8 chips.
  3. **XLAPinnedStaticCache & Bucketed Prefill + O(1) ``dynamic-update-slice`` Decode**:
     Eliminates dynamic-shape HLO recompilation storms on ``model.generate()`` by pairing
     fixed prefill buckets ``(128, 256, 512, 1024, 1536, 2048)`` with a pre-allocated
     ``(1, H_kv, max_cache_len, D_kv)`` static cache updated in-place via ``index_copy_`` at
     true sequence positions ``[0..L-1]`` then ``L, L+1, ...`` (exact RoPE & causal alignment).
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import glob
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.cache_utils import Cache as _HFBaseCache
except Exception:  # pragma: no cover
    class _HFBaseCache:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


DEFAULT_PREFILL_BUCKETS: Tuple[int, ...] = (128, 256, 512, 1024, 1536, 2048)

_RUNTIME_INIT_LOCK = threading.Lock()
_CACHED_SPMD_CTX: Optional["TPUSPMDContext"] = None


@dataclass
class TPUSPMDContext:
    """Active XLA GSPMD device mesh and runtime metadata."""

    device: torch.device
    num_devices: int
    mesh: Any  # torch_xla.distributed.spmd.Mesh or None in single-chip/CPU test mode
    xm: Any
    xr: Any
    xs: Any
    cache_dir: Optional[str] = None
    is_real_tpu: bool = True

    def mark_step(self) -> None:
        if self.xm is not None:
            self.xm.mark_step()

    def wait_device_ops(self) -> None:
        if self.xm is not None:
            self.xm.wait_device_ops()

    def mark_sharding(self, tensor: torch.Tensor, spec: Tuple[Optional[str], ...]) -> None:
        if self.xs is not None and self.mesh is not None and self.num_devices > 1:
            self.xs.mark_sharding(tensor, self.mesh, spec)


def is_tpu_xla_available() -> bool:
    """Return True when ``torch_xla`` is installed and backed by TPU (or explicit XLA SPMD)."""
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]
        import torch_xla.runtime as xr  # type: ignore[import-not-found]

        dev_type = str(xr.device_type() or "").upper()
        if dev_type == "TPU":
            return True
        devs = xm.get_xla_supported_devices()
        return bool(devs) and (
            bool(glob.glob("/dev/accel*"))
            or bool(glob.glob("/dev/vfio/[0-9]*"))
            or os.environ.get("PJRT_DEVICE", "").upper() == "TPU"
        )
    except Exception:
        return False


def initialize_tpu_spmd_runtime(
    cache_dir: Optional[str | Path] = "/kaggle/working/xla_compile_cache",
) -> TPUSPMDContext:
    """Idempotently enable PyTorch/XLA GSPMD across all available TPU chips (e.g. 8 on v5e-8)."""
    global _CACHED_SPMD_CTX
    with _RUNTIME_INIT_LOCK:
        if _CACHED_SPMD_CTX is not None:
            return _CACHED_SPMD_CTX

        os.environ.setdefault("PJRT_DEVICE", "TPU")
        import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]
        import torch_xla.distributed.spmd as xs  # type: ignore[import-not-found]
        import torch_xla.runtime as xr  # type: ignore[import-not-found]

        resolved_cache: Optional[str] = None
        if cache_dir:
            try:
                cpath = Path(cache_dir)
                cpath.mkdir(parents=True, exist_ok=True)
                xr.initialize_cache(str(cpath), readonly=False)
                resolved_cache = str(cpath)
            except Exception:
                resolved_cache = None

        xr.use_spmd()
        num_devices = int(xr.global_runtime_device_count())
        device_ids = np.arange(num_devices)
        mesh = xs.Mesh(device_ids, (num_devices,), ("model",)) if num_devices > 1 else None
        dev = xm.xla_device()

        _CACHED_SPMD_CTX = TPUSPMDContext(
            device=dev,
            num_devices=num_devices,
            mesh=mesh,
            xm=xm,
            xr=xr,
            xs=xs,
            cache_dir=resolved_cache,
            is_real_tpu=(str(xr.device_type() or "").upper() == "TPU"),
        )
        return _CACHED_SPMD_CTX


def resolve_kaggle_or_hf_model_source(requested_model_id: str) -> str:
    """Prefer zero-download local Kaggle Models mount under ``/kaggle/input`` when present."""
    p_req = Path(requested_model_id)
    if p_req.exists() and (p_req / "config.json").is_file():
        return str(p_req)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        req_lower = requested_model_id.lower()
        want_31b = "31b" in req_lower
        want_12b = "12b" in req_lower
        candidates: List[Path] = []
        for cfg_file in sorted(kaggle_input.rglob("config.json")):
            parent = cfg_file.parent
            p_str = str(parent).lower()
            if "gemma" in p_str:
                if want_31b and "31b" not in p_str:
                    continue
                if want_12b and "12b" not in p_str:
                    continue
                if any(parent.glob("*.safetensors")):
                    candidates.append(parent)
        if candidates:
            # Prefer instruction-tuned (`-it`) unquantized safetensors directory
            for c in candidates:
                if "it" in str(c).lower() and "gguf" not in str(c).lower() and "w4a16" not in str(c).lower():
                    return str(c)
            return str(candidates[0])

    return requested_model_id


def compute_gemma4_parameter_partition_spec(
    param_name: str,
    param_shape: Sequence[int],
    num_devices: int = 8,
) -> Optional[Tuple[Optional[str], ...]]:
    """Compute exact 1D GSPMD ``partition_spec`` for ``Gemma-4-31B-it`` & generic Transformers.

    Returns ``None`` when the tensor should remain replicated across all chips.
    """
    ndim = len(param_shape)
    if num_devices <= 1 or ndim < 2:
        return None

    d0, d1 = int(param_shape[0]), int(param_shape[1])
    tail = (None,) * (ndim - 2)
    lname = param_name.lower()

    # 1. Token embeddings / tied LM head: (vocab_size=262144, hidden_size=5376)
    if "embed_tokens" in lname or "lm_head" in lname:
        if d0 % num_devices == 0:
            return ("model", None) + tail
        if d1 % num_devices == 0:
            return (None, "model") + tail
        return None

    # 2. Row-parallel output & down projections: reduce across hidden_size via hardware ICI all-reduce
    if any(tok in lname for tok in (".o_proj.", ".down_proj.", ".out_proj.", ".fc2.")):
        if d1 % num_devices == 0:
            return (None, "model") + tail
        if d0 % num_devices == 0:
            return ("model", None) + tail
        return None

    # 3. Attention K/V projections:
    #    - Sliding layers (16 KV heads * 256 = 4096): 16 % 8 == 0 -> ('model', None) (2 KV heads/chip)
    #    - Global layers (4 KV heads * 512 = 2048): 4 < 8 -> shard input axis (None, 'model') so the
    #      4 global KV heads are never split across 8 chips!
    if ".k_proj." in lname or ".v_proj." in lname:
        # For Gemma-4 global layers d0 == 2048 (4 * 512) while num_devices == 8:
        # detect <= 4 KV heads when d0 // 512 < num_devices and d1 % num_devices == 0.
        if d0 <= 2048 and d1 >= 4096 and (d1 % num_devices == 0):
            return (None, "model") + tail
        if d0 % num_devices == 0:
            return ("model", None) + tail
        if d1 % num_devices == 0:
            return (None, "model") + tail
        return None

    # 4. Column-parallel Q, Gate, Up projections
    if any(tok in lname for tok in (".q_proj.", ".gate_proj.", ".up_proj.", ".fc1.")):
        if d0 % num_devices == 0:
            return ("model", None) + tail
        if d1 % num_devices == 0:
            return (None, "model") + tail
        return None

    # 5. Generic large 2D matrices (e.g. multimodal towers if retained)
    if d0 >= 1024 and d0 % num_devices == 0:
        return ("model", None) + tail
    if d1 >= 1024 and d1 % num_devices == 0:
        return (None, "model") + tail
    return None


def compute_gemma4_lora_partition_specs(
    proj_name: str,
    in_features: int,
    out_features: int,
    num_devices: int = 8,
) -> Tuple[Optional[Tuple[Optional[str], Optional[str]]], Optional[Tuple[Optional[str], Optional[str]]]]:
    """Return ``(lora_A_spec, lora_B_spec)`` aligned with the wrapped base projection's sharding.

    ``lora_A`` has shape ``(rank, in_features)``; ``lora_B`` has shape ``(out_features, rank)``.
    """
    if num_devices <= 1:
        return None, None
    base_spec = compute_gemma4_parameter_partition_spec(
        f"layers.0.{proj_name}.weight",
        (out_features, in_features),
        num_devices=num_devices,
    )
    if base_spec == ("model", None):
        # Base output is sharded on dim 0 -> shard lora_B on dim 0 (`out_features`), replicate lora_A
        return None, ("model", None)
    if base_spec == (None, "model"):
        # Base input is sharded on dim 1 -> shard lora_A on dim 1 (`in_features`), replicate lora_B
        return (None, "model"), None
    return None, None


def shard_model_to_tpu_spmd(
    model: nn.Module,
    spmd_ctx: TPUSPMDContext,
    *,
    drop_unused_multimodal_towers: bool = True,
    target_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """Stream a CPU-loaded ``Gemma-4`` model onto 8 TPU v5e chips parameter-by-parameter.

    Avoids the ``model.to(xla_device)`` single-chip 62 GB OOM spike by moving and marking
    sharding one parameter/buffer at a time and flushing XLA execution every few layers.
    """
    t0 = time.perf_counter()
    dropped_components: List[str] = []

    if drop_unused_multimodal_towers:
        inner = getattr(model, "model", None)
        if inner is not None:
            for attr in ("vision_tower", "embed_vision", "audio_tower", "embed_audio", "multi_modal_projector"):
                if hasattr(inner, attr) and getattr(inner, attr) is not None:
                    setattr(inner, attr, None)
                    dropped_components.append(attr)
        gc.collect()

    # Force pure tensor eager attention so 4D additive masks + GSPMD compile cleanly onto TPU MXUs
    for cfg_obj in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if cfg_obj is not None:
            try:
                cfg_obj._attn_implementation = "eager"
            except Exception:
                pass

    # Handle tied weights (`lm_head.weight` == `embed_tokens.weight`) without double-allocating 2.8 GB!
    seen_data_ptrs: Dict[int, nn.Parameter] = {}
    sharded_params = 0
    replicated_params = 0
    total_param_bytes = 0

    for idx, (name, param) in enumerate(model.named_parameters()):
        orig_ptr = int(param.data.data_ptr())
        if orig_ptr in seen_data_ptrs:
            param.data = seen_data_ptrs[orig_ptr].data
            param.requires_grad_(False)
            continue

        cpu_t = param.data
        if cpu_t.is_floating_point() and cpu_t.dtype != target_dtype:
            cpu_t = cpu_t.to(dtype=target_dtype)
        total_param_bytes += cpu_t.numel() * cpu_t.element_size()

        spec = compute_gemma4_parameter_partition_spec(name, tuple(cpu_t.shape), num_devices=spmd_ctx.num_devices)
        xla_t = cpu_t.to(device=spmd_ctx.device)
        if spec is not None:
            spmd_ctx.mark_sharding(xla_t, spec)
            sharded_params += 1
        else:
            replicated_params += 1

        param.data = xla_t
        param.requires_grad_(False)
        seen_data_ptrs[orig_ptr] = param
        del cpu_t

        if (idx + 1) % 24 == 0:
            spmd_ctx.mark_step()
            gc.collect()

    for b_idx, (_bname, buf) in enumerate(model.named_buffers()):
        if buf is not None and buf.device != spmd_ctx.device:
            buf.data = buf.data.to(device=spmd_ctx.device)
            if (b_idx + 1) % 32 == 0:
                spmd_ctx.mark_step()

    spmd_ctx.mark_step()
    spmd_ctx.wait_device_ops()
    gc.collect()

    mem_info: Dict[str, Any] = {}
    try:
        mem_info = dict(spmd_ctx.xm.get_memory_info(spmd_ctx.device))
    except Exception:
        pass

    return {
        "num_tpu_devices": spmd_ctx.num_devices,
        "sharded_parameter_tensors": sharded_params,
        "replicated_parameter_tensors": replicated_params,
        "total_model_gb_bf16": round(total_param_bytes / (1024 ** 3), 3),
        "dropped_unused_towers": dropped_components,
        "chip0_memory_info": mem_info,
        "shard_elapsed_sec": round(time.perf_counter() - t0, 2),
    }


class _XLAPinnedLayerSlot:
    """Fixed-shape static KV buffer for a single Transformer decoder layer."""

    is_compileable: bool = True

    def __init__(
        self,
        max_cache_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        spmd_ctx: Optional[TPUSPMDContext] = None,
    ) -> None:
        self.max_cache_len = int(max_cache_len)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.device = device
        self.dtype = dtype
        self.is_initialized = True

        self.keys = torch.zeros(
            (1, self.num_kv_heads, self.max_cache_len, self.head_dim),
            dtype=dtype,
            device=device,
        )
        self.values = torch.zeros(
            (1, self.num_kv_heads, self.max_cache_len, self.head_dim),
            dtype=dtype,
            device=device,
        )
        if spmd_ctx is not None and spmd_ctx.num_devices > 1 and (self.num_kv_heads % spmd_ctx.num_devices == 0):
            spmd_ctx.mark_sharding(self.keys, (None, "model", None, None))
            spmd_ctx.mark_sharding(self.values, (None, "model", None, None))

    def zero_(self) -> None:
        self.keys.zero_()
        self.values.zero_()

    def get_seq_length(self) -> int:
        return 0

    def get_max_length(self) -> int:
        return self.max_cache_len

    def get_mask_sizes(self, query_length: int) -> Tuple[int, int]:
        return self.max_cache_len, 0


class XLAPinnedStaticCache(_HFBaseCache):
    """Zero-recompilation fixed-shape KV-Cache for Hybrid Sliding/Global ``Gemma-4`` on XLA.

    Unlike HuggingFace ``StaticCache`` (which increments ``cumulative_length`` by padded prefill
    length and dynamically branches inside ``StaticSlidingWindowLayer``), ``XLAPinnedStaticCache``
    updates exact caller-supplied device ``write_positions`` via ``index_copy_`` (compiling to
    XLA's native ``O(1)`` ``dynamic-update-slice`` instruction) while keeping ``(1, H_kv, M, D_kv)``
    100% invariant across every prefill bucket and every single-token decode step.
    """

    is_compileable: bool = True

    def __init__(
        self,
        text_config: Any,
        max_cache_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        spmd_ctx: Optional[TPUSPMDContext] = None,
    ) -> None:
        self.max_cache_len = int(max_cache_len)
        self.device = device
        self.dtype = dtype
        self.num_layers = int(getattr(text_config, "num_hidden_layers", 60))
        layer_types = list(getattr(text_config, "layer_types", ["full_attention"] * self.num_layers))
        per_layer_cfg = getattr(text_config, "per_layer_config", None)

        slots: List[_XLAPinnedLayerSlot] = []
        for l_idx in range(self.num_layers):
            if per_layer_cfg is not None and l_idx < len(per_layer_cfg):
                h_kv = int(per_layer_cfg[l_idx].num_key_value_heads)
                d_kv = int(per_layer_cfg[l_idx].head_dim)
            else:
                l_type = layer_types[l_idx] if l_idx < len(layer_types) else "full_attention"
                if l_type == "full_attention" and getattr(text_config, "num_global_key_value_heads", None):
                    h_kv = int(text_config.num_global_key_value_heads)
                    d_kv = int(getattr(text_config, "global_head_dim", getattr(text_config, "head_dim", 256)))
                else:
                    h_kv = int(getattr(text_config, "num_key_value_heads", 8))
                    d_kv = int(getattr(text_config, "head_dim", 256))
            slots.append(
                _XLAPinnedLayerSlot(
                    max_cache_len=self.max_cache_len,
                    num_kv_heads=h_kv,
                    head_dim=d_kv,
                    device=device,
                    dtype=dtype,
                    spmd_ctx=spmd_ctx,
                )
            )

        try:
            super().__init__(layers=slots)  # type: ignore[arg-type]
        except TypeError:
            super().__init__()
        if not getattr(self, "layers", None):
            self.layers = slots

        self._write_positions: Optional[torch.Tensor] = None
        self._logical_seq_len: int = 0

    def reset_for_new_sequence(self) -> None:
        for slot in self.layers:
            slot.zero_()
        self._write_positions = None
        self._logical_seq_len = 0

    def set_write_positions(self, write_positions_1d: torch.Tensor, logical_seq_len: int) -> None:
        self._write_positions = write_positions_1d
        self._logical_seq_len = int(logical_seq_len)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        slot: _XLAPinnedLayerSlot = self.layers[layer_idx]
        q_len = int(key_states.shape[-2])
        k_in = key_states.to(dtype=slot.dtype)
        v_in = value_states.to(dtype=slot.dtype)

        if q_len > 1 and self._write_positions is None:
            slot.keys[:, :, :q_len, :].copy_(k_in)
            slot.values[:, :, :q_len, :].copy_(v_in)
        elif self._write_positions is not None:
            slot.keys.index_copy_(2, self._write_positions, k_in)
            slot.values.index_copy_(2, self._write_positions, v_in)
        else:
            pos = torch.arange(q_len, dtype=torch.long, device=slot.device)
            slot.keys.index_copy_(2, pos, k_in)
            slot.values.index_copy_(2, pos, v_in)

        return slot.keys, slot.values

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._logical_seq_len

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len


def build_static_hybrid_masks_prefill(
    bucket_len: int,
    max_cache_len: int,
    valid_prompt_len_t: torch.Tensor,
    sliding_window: int,
    device: torch.device,
    mask_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, torch.Tensor]:
    """Construct fixed-shape ``(1, 1, bucket_len, max_cache_len)`` causal & sliding additive masks.

    ``valid_prompt_len_t`` is a 1D tensor ``(1,)`` on ``device`` holding ``L`` so the compiled HLO
    graph depends ONLY on ``bucket_len`` and ``max_cache_len``, NEVER on the runtime prompt length ``L``!
    """
    neg_inf = -1.0e30 if mask_dtype == torch.float32 else -3.38e38
    q_idx = torch.arange(bucket_len, dtype=torch.long, device=device).view(bucket_len, 1)
    k_idx = torch.arange(max_cache_len, dtype=torch.long, device=device).view(1, max_cache_len)
    valid_len = valid_prompt_len_t.view(1, 1)

    # For padded query rows q >= valid_len, allow self-attention on k == 0 so softmax never sees all -inf (NaN)!
    real_q = q_idx < valid_len
    valid_k_for_real_q = (k_idx <= q_idx) & (k_idx < valid_len)
    pad_safe_anchor = (~real_q) & (k_idx == 0)

    full_allow = valid_k_for_real_q | pad_safe_anchor
    slide_allow = (valid_k_for_real_q & ((q_idx - k_idx) < int(sliding_window))) | pad_safe_anchor

    zero = torch.tensor(0.0, dtype=mask_dtype, device=device)
    ninf = torch.tensor(neg_inf, dtype=mask_dtype, device=device)

    full_mask = torch.where(full_allow, zero, ninf).view(1, 1, bucket_len, max_cache_len)
    slide_mask = torch.where(slide_allow, zero, ninf).view(1, 1, bucket_len, max_cache_len)
    return {
        "full_attention": full_mask,
        "sliding_attention": slide_mask,
    }


def build_static_hybrid_masks_decode(
    max_cache_len: int,
    cur_pos_t: torch.Tensor,
    sliding_window: int,
    device: torch.device,
    mask_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, torch.Tensor]:
    """Construct fixed-shape ``(1, 1, 1, max_cache_len)`` single-token decode masks on device."""
    neg_inf = -1.0e30 if mask_dtype == torch.float32 else -3.38e38
    k_idx = torch.arange(max_cache_len, dtype=torch.long, device=device).view(1, max_cache_len)
    pos = cur_pos_t.view(1, 1)

    full_allow = k_idx <= pos
    slide_allow = full_allow & ((pos - k_idx) < int(sliding_window))

    zero = torch.tensor(0.0, dtype=mask_dtype, device=device)
    ninf = torch.tensor(neg_inf, dtype=mask_dtype, device=device)

    return {
        "full_attention": torch.where(full_allow, zero, ninf).view(1, 1, 1, max_cache_len),
        "sliding_attention": torch.where(slide_allow, zero, ninf).view(1, 1, 1, max_cache_len),
    }


def sample_next_token_static(
    logits_2d: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_k: int = 50,
    top_p: float = 0.92,
    penalty_counts_2d: Optional[torch.Tensor] = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """Sample next token ID with 100% static tensor shapes (no dynamic boolean indexing).

    Args:
        logits_2d: Tensor of shape ``(1, vocab_size)`` on device.
        penalty_counts_2d: Optional float mask ``(1, vocab_size)`` with ``1.0`` where seen, ``0.0`` otherwise.

    Returns:
        ``next_token`` of shape ``(1, 1)`` and dtype ``torch.long`` on the same device.
    """
    scores = logits_2d.float()

    if penalty_counts_2d is not None and repetition_penalty > 1.0001:
        pen = float(repetition_penalty)
        penalized = torch.where(scores < 0.0, scores * pen, scores / pen)
        scores = torch.where(penalty_counts_2d > 0.5, penalized, scores)

    if not do_sample or temperature <= 0.05:
        return torch.argmax(scores, dim=-1, keepdim=True).to(dtype=torch.long)

    scores = scores / max(0.05, float(temperature))
    vocab_size = int(scores.shape[-1])
    k = min(max(1, int(top_k)), vocab_size)

    # Static Top-K + Nucleus (Top-P) inside the K-candidate subspace (64 elements instead of sorting 262,144!)
    # Sorting 64 top-k candidates instead of sorting all 262,144 vocab logits is ~4,000x faster on TPU!
    topk_vals, topk_idx = torch.topk(scores, k=k, dim=-1, largest=True, sorted=True)  # (1, k)

    if 0.0 < top_p < 1.0 and k > 1:
        probs_k = F.softmax(topk_vals, dim=-1)
        cum_probs = torch.cumsum(probs_k, dim=-1)
        # Keep token 0 unconditionally; mask tokens whose cumulative mass BEFORE adding them exceeds top_p
        remove_mask = (cum_probs - probs_k) > float(top_p)
        topk_vals = torch.where(remove_mask, torch.full_like(topk_vals, -1.0e20), topk_vals)

    sampled_probs = F.softmax(topk_vals, dim=-1)
    sampled_rel = torch.multinomial(sampled_probs, num_samples=1)  # (1, 1)
    next_tok = torch.gather(topk_idx, dim=-1, index=sampled_rel)   # (1, 1)
    return next_tok.to(dtype=torch.long)


class TPUBucketedStaticGenerator:
    """Zero-recompilation bucketed prefill + static KV-cache token generator for Gemma-4."""

    def __init__(
        self,
        model: nn.Module,
        spmd_ctx: Optional[TPUSPMDContext] = None,
        *,
        max_cache_len: int = 2560,
        prefill_buckets: Sequence[int] = DEFAULT_PREFILL_BUCKETS,
    ) -> None:
        self.model = model
        self.spmd_ctx = spmd_ctx
        self.device = spmd_ctx.device if spmd_ctx is not None else next(model.parameters()).device
        self.max_cache_len = int(max_cache_len)
        self.prefill_buckets: Tuple[int, ...] = tuple(sorted(int(b) for b in prefill_buckets if int(b) <= self.max_cache_len))

        raw_cfg = getattr(model, "config", None)
        self.text_config = (
            raw_cfg.get_text_config()
            if raw_cfg is not None and hasattr(raw_cfg, "get_text_config")
            else getattr(raw_cfg, "text_config", raw_cfg)
        )
        self.vocab_size = int(getattr(self.text_config, "vocab_size", 262144))
        self.pad_token_id = int(getattr(self.text_config, "pad_token_id", 0) or 0)
        self.sliding_window = int(getattr(self.text_config, "sliding_window", 1024) or 1024)
        self.is_gemma4_hybrid = bool(
            hasattr(self.text_config, "layer_types")
            and "gemma4" in str(getattr(self.text_config, "model_type", "")).lower()
        )
        param_dtype = next(model.parameters()).dtype
        self.cache_dtype = param_dtype if param_dtype in (torch.bfloat16, torch.float16, torch.float32) else torch.bfloat16

        self.static_cache = XLAPinnedStaticCache(
            text_config=self.text_config,
            max_cache_len=self.max_cache_len,
            device=self.device,
            dtype=self.cache_dtype,
            spmd_ctx=spmd_ctx,
        )
        self._seen_token_mask = torch.zeros((1, self.vocab_size), dtype=torch.float32, device=self.device)
        self._ones_1x1 = torch.ones((1, 1), dtype=torch.float32, device=self.device)
        self._decode_zero_keep = torch.zeros((1, 1), dtype=torch.long, device=self.device)

    def select_prefill_bucket(self, prompt_len: int) -> int:
        for b in self.prefill_buckets:
            if b >= prompt_len:
                return b
        return max(prompt_len, self.max_cache_len)

    def _mark_step(self) -> None:
        if self.spmd_ctx is not None:
            self.spmd_ctx.mark_step()

    @torch.no_grad()
    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        do_sample: bool = True,
        temperature: float = 0.70,
        top_p: float = 0.92,
        top_k: int = 50,
        repetition_penalty: float = 1.12,
        eos_token_ids: Optional[Sequence[int]] = None,
        abort_event: Optional[threading.Event] = None,
        on_prefill_complete: Optional[Callable[[], None]] = None,
        on_token_step: Optional[Callable[[int, int], Optional[float]]] = None,
        streamer: Optional[Any] = None,
    ) -> torch.Tensor:
        """Execute bucketed prefill + single-token static decode and return ``(1, L + gen_len)`` IDs on CPU."""
        cpu_prompt = input_ids.detach().to(device="cpu", dtype=torch.long)
        if cpu_prompt.ndim == 1:
            cpu_prompt = cpu_prompt.unsqueeze(0)

        max_prompt_allowed = max(32, self.max_cache_len - max(16, min(int(max_new_tokens), 1024)))
        if int(cpu_prompt.shape[1]) > max_prompt_allowed:
            cpu_prompt = cpu_prompt[:, -max_prompt_allowed:]

        prompt_len = int(cpu_prompt.shape[1])
        bucket_len = self.select_prefill_bucket(prompt_len)
        budget = max(1, min(int(max_new_tokens), self.max_cache_len - prompt_len))

        # 1. Reset static KV cache & seed repetition-penalty mask on CPU -> 1 upload
        self.static_cache.reset_for_new_sequence()
        seen_cpu = torch.zeros((1, self.vocab_size), dtype=torch.float32)
        unique_prompt = torch.unique(cpu_prompt[0].clamp(0, self.vocab_size - 1))
        seen_cpu[0, unique_prompt] = 1.0
        self._seen_token_mask.copy_(seen_cpu.to(device=self.device))

        # 2. Right-pad prompt to `bucket_len` with exact true RoPE `position_ids` [0..L-1, 0..0]
        padded_cpu = torch.full((1, bucket_len), self.pad_token_id, dtype=torch.long)
        padded_cpu[0, :prompt_len] = cpu_prompt[0]
        pos_cpu = torch.zeros((1, bucket_len), dtype=torch.long)
        pos_cpu[0, :prompt_len] = torch.arange(prompt_len, dtype=torch.long)

        padded_ids = padded_cpu.to(device=self.device)
        pos_ids = pos_cpu.to(device=self.device)
        valid_len_t = torch.tensor([prompt_len], dtype=torch.long, device=self.device)
        last_pos_2d = torch.tensor([[prompt_len - 1]], dtype=torch.long, device=self.device)
        prefill_write_pos = torch.arange(bucket_len, dtype=torch.long, device=self.device)
        self.static_cache.set_write_positions(prefill_write_pos, logical_seq_len=prompt_len)

        masks_prefill = build_static_hybrid_masks_prefill(
            bucket_len=bucket_len,
            max_cache_len=self.max_cache_len,
            valid_prompt_len_t=valid_len_t,
            sliding_window=self.sliding_window,
            device=self.device,
            mask_dtype=self.cache_dtype,
        )
        attn_arg: Any = masks_prefill if self.is_gemma4_hybrid else masks_prefill["full_attention"]

        fwd_kwargs: Dict[str, Any] = {
            "input_ids": padded_ids,
            "attention_mask": attn_arg,
            "position_ids": pos_ids,
            "past_key_values": self.static_cache,
            "use_cache": True,
        }
        if self.is_gemma4_hybrid:
            fwd_kwargs["logits_to_keep"] = last_pos_2d

        prefill_out = self.model(**fwd_kwargs)
        raw_logits = prefill_out.logits if hasattr(prefill_out, "logits") else prefill_out[0]
        if int(raw_logits.shape[1]) == 1:
            first_logits_2d = raw_logits[:, 0, :]
        else:
            gathered = torch.take_along_dim(raw_logits, last_pos_2d[..., None], dim=1)
            first_logits_2d = gathered[:, 0, :]

        next_tok_dev = sample_next_token_static(
            first_logits_2d,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            penalty_counts_2d=self._seen_token_mask,
            repetition_penalty=repetition_penalty,
        )
        self._seen_token_mask.scatter_(1, next_tok_dev.clamp(0, self.vocab_size - 1), self._ones_1x1)
        self._mark_step()

        if streamer is not None:
            try:
                streamer.put(cpu_prompt)
            except Exception:
                pass

        first_tok_id = int(next_tok_dev[0, 0].cpu().item())
        if on_prefill_complete is not None:
            on_prefill_complete()

        eos_set = set(int(e) for e in (eos_token_ids or ()))
        generated: List[int] = [first_tok_id]

        if streamer is not None:
            streamer.put(torch.tensor([[first_tok_id]], dtype=torch.long))

        if first_tok_id in eos_set or (abort_event is not None and abort_event.is_set()):
            return torch.cat([cpu_prompt, torch.tensor([generated], dtype=torch.long)], dim=-1)

        if on_token_step is not None:
            on_token_step(first_tok_id, 1)

        # 3. Single-token static decode loop: shapes are 100% invariant `(1, 1)` on every token!
        cur_tok_dev = next_tok_dev
        for step_idx in range(1, budget):
            if abort_event is not None and abort_event.is_set():
                break

            abs_pos = prompt_len + step_idx - 1
            pos_1d = torch.tensor([abs_pos], dtype=torch.long, device=self.device)
            pos_2d = pos_1d.view(1, 1)
            self.static_cache.set_write_positions(pos_1d, logical_seq_len=abs_pos + 1)

            masks_dec = build_static_hybrid_masks_decode(
                max_cache_len=self.max_cache_len,
                cur_pos_t=pos_1d,
                sliding_window=self.sliding_window,
                device=self.device,
                mask_dtype=self.cache_dtype,
            )
            dec_attn: Any = masks_dec if self.is_gemma4_hybrid else masks_dec["full_attention"]

            dec_kwargs: Dict[str, Any] = {
                "input_ids": cur_tok_dev,
                "attention_mask": dec_attn,
                "position_ids": pos_2d,
                "past_key_values": self.static_cache,
                "use_cache": True,
            }
            if self.is_gemma4_hybrid:
                dec_kwargs["logits_to_keep"] = self._decode_zero_keep

            dec_out = self.model(**dec_kwargs)
            dec_logits = dec_out.logits if hasattr(dec_out, "logits") else dec_out[0]
            cur_tok_dev = sample_next_token_static(
                dec_logits[:, -1, :],
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                penalty_counts_2d=self._seen_token_mask,
                repetition_penalty=repetition_penalty,
            )
            self._seen_token_mask.scatter_(1, cur_tok_dev.clamp(0, self.vocab_size - 1), self._ones_1x1)
            self._mark_step()

            tok_id = int(cur_tok_dev[0, 0].cpu().item())
            generated.append(tok_id)
            if streamer is not None:
                streamer.put(torch.tensor([[tok_id]], dtype=torch.long))

            if tok_id in eos_set:
                break

            if on_token_step is not None:
                on_token_step(tok_id, step_idx + 1)

            # Degenerate 2-token repetition guard matching `_AbortAndDegeneracyCriteria`
            if len(generated) >= 20 and len(set(generated[-10:])) <= 2:
                break

        return torch.cat([cpu_prompt, torch.tensor([generated], dtype=torch.long)], dim=-1)

    @torch.no_grad()
    def warmup_xla_graphs(
        self,
        buckets: Sequence[int] = (256, 512),
        prepare_hooks_fn: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, float]:
        """Pre-compile bucketed prefill + 1-token decode HLO graphs at boot so live chat has 0s compile lag."""
        timings: Dict[str, float] = {}
        for b in buckets:
            if b > self.max_cache_len:
                continue
            t0 = time.perf_counter()
            dummy_len = max(8, b - 4)
            if prepare_hooks_fn is not None:
                prepare_hooks_fn(dummy_len)
            dummy_ids = torch.ones((1, dummy_len), dtype=torch.long)
            self.generate_tokens(
                dummy_ids,
                max_new_tokens=3,
                do_sample=True,
                temperature=0.7,
            )
            if self.spmd_ctx is not None:
                self.spmd_ctx.wait_device_ops()
            timings[f"bucket_{b}_sec"] = round(time.perf_counter() - t0, 2)
        return timings
