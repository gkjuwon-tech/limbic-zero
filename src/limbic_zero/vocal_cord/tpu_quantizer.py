"""TPU v5e-1 Streaming 4-bit Quantizer & Hybrid HBM/CPU Split Engine for Gemma-4-31B-IT.

Enables running ``google/gemma-4-31B-it`` (60 layers, 5376D, ~62 GB in ``bfloat16``) inside
Google Colab's free single-chip ``TPU v5e-1`` runtime (``16 GiB`` HBM + ``~50.5 GB`` Host RAM)
where CUDA-only ``bitsandbytes`` is unavailable and full-model ``bf16`` loading would OOM.

Key Architectural Guarantees:
  1. **Zero-Host-OOM Shard Streaming**: Instantiates ``nn.Linear`` and ``nn.Embedding`` weights on
     ``meta`` (0 MB RAM) while keeping small norms and RoPE ``inv_freq`` buffers live on CPU, then
     streams HuggingFace ``.safetensors`` shards one file at a time, quantizing on-the-fly.
  2. **Transient Single-Layer Unpacking (``_StreamingInt4LinearFn``)**: Stores weights as packed
     ``uint8`` (2 x signed int4 nibbles per byte, ``0.5 B/param``) + groupwise ``bfloat16`` scales
     (``group_size=128``). Unpacks only one active projection at a time during ``forward()`` AND
     ``backward()`` (saving 0 extra bytes on the autograd tape during ``CorticalWeightSurgeon``
     online LoRA weight surgery).
  3. **Hybrid TPU HBM + Host CPU Split**: Places ``embed_tokens`` + middle-inclusive decoder
     blocks ``0 .. split_idx - 1`` (covering 100% of ``TARGET_LAYERS_60`` steering layers ``21..47``
     and ``DEFAULT_PLASTIC_LAYERS_60`` LoRA layers ``23..47``) + ``norm`` + tied ``lm_head`` onto
     ``tpu:0`` within ``hbm_budget_gb`` (default ``12.5 GiB``), and places overflow upper layers
     onto Host CPU ``bfloat16`` with automatic cross-device ``register_forward_pre_hook`` handoff.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TPUAcceleratorInfo:
    """Detected Google Cloud / Colab TPU runtime device & HBM capacity."""

    available: bool
    backend: str  # "torch_tpu" | "torch_xla" | "none"
    device: torch.device
    device_name: str
    hbm_total_mb: float
    hbm_used_mb: float


def detect_tpu_device() -> TPUAcceleratorInfo:
    """Auto-detect ``torch_tpu`` (Colab TPU v5e-1 default) or ``torch_xla`` accelerator."""
    os.environ.setdefault("TORCH_TPU_DEFER_AND_FUSE", "1")

    # 1. Modern Colab PyTorch-TPU plugin (`torch_tpu`)
    try:
        import torch_tpu  # type: ignore[import-not-found]  # noqa: F401

        tpu_mod = getattr(torch, "tpu", None)
        if tpu_mod is not None and callable(getattr(tpu_mod, "is_available", None)) and tpu_mod.is_available():
            dev = torch.device("tpu:0")
            total_mb = 16384.0
            used_mb = 0.0
            dev_name = "Google Cloud TPU v5e-1"
            try:
                if hasattr(tpu_mod, "get_device_name"):
                    raw_name = str(tpu_mod.get_device_name(0)).strip()
                    if raw_name:
                        dev_name = raw_name
                if hasattr(tpu_mod, "get_device_properties"):
                    props = tpu_mod.get_device_properties(0)
                    tot_b = getattr(props, "total_memory", 0)
                    if isinstance(tot_b, (int, float)) and tot_b > 0:
                        total_mb = round(float(tot_b) / (1024.0 ** 2), 1)
                if hasattr(tpu_mod, "memory_allocated"):
                    used_b = tpu_mod.memory_allocated(0)
                    if isinstance(used_b, (int, float)) and used_b >= 0:
                        used_mb = round(float(used_b) / (1024.0 ** 2), 1)
            except Exception:
                pass
            return TPUAcceleratorInfo(
                available=True,
                backend="torch_tpu",
                device=dev,
                device_name=dev_name,
                hbm_total_mb=total_mb,
                hbm_used_mb=used_mb,
            )
    except Exception:
        pass

    # 2. PyTorch/XLA (`torch_xla`) runtime
    if (
        os.environ.get("COLAB_TPU_ADDR")
        or os.environ.get("TPU_NAME")
        or os.environ.get("TPU_WORKER_ID")
        or Path("/dev/accel0").exists()
        or Path("/dev/vfio").exists()
    ):
        os.environ.setdefault("PJRT_DEVICE", "TPU")

    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]

        dev = xm.xla_device()
        hw_kind = str(getattr(xm, "xla_device_hw", lambda _d: "TPU")(dev) or "TPU").upper()
        if hw_kind != "CPU":
            total_mb = 16384.0
            used_mb = 0.0
            dev_name = f"Google Cloud {hw_kind} v5e-1"
            try:
                mem_info = xm.get_memory_info(dev)
                if isinstance(mem_info, dict):
                    lim_b = mem_info.get("bytes_limit") or mem_info.get("kb_total", 0) * 1024
                    use_b = mem_info.get("bytes_used", 0)
                    if lim_b and lim_b > 0:
                        total_mb = round(float(lim_b) / (1024.0 ** 2), 1)
                    if use_b and use_b >= 0:
                        used_mb = round(float(use_b) / (1024.0 ** 2), 1)
            except Exception:
                pass
            return TPUAcceleratorInfo(
                available=True,
                backend="torch_xla",
                device=dev,
                device_name=dev_name,
                hbm_total_mb=total_mb,
                hbm_used_mb=used_mb,
            )
    except Exception:
        pass

    return TPUAcceleratorInfo(
        available=False,
        backend="none",
        device=torch.device("cpu"),
        device_name="CPU",
        hbm_total_mb=0.0,
        hbm_used_mb=0.0,
    )


def sync_tpu_device(device: Optional[torch.device] = None) -> None:
    """Flush lazy XLA / TPU execution graph safely if running on a TPU backend."""
    dev_type = "" if device is None else str(device.type).lower()
    if dev_type == "tpu" or hasattr(torch, "tpu"):
        try:
            tpu_mod = getattr(torch, "tpu", None)
            if tpu_mod is not None and hasattr(tpu_mod, "synchronize"):
                tpu_mod.synchronize()
                return
        except Exception:
            pass
    if dev_type == "xla":
        try:
            import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]

            xm.mark_step()
        except Exception:
            pass


def resolve_model_input_device(model: nn.Module) -> torch.device:
    """Determine the primary input device where ``input_ids`` should be placed."""
    if hasattr(model, "get_input_embeddings"):
        try:
            emb = model.get_input_embeddings()
            if emb is not None:
                for p in emb.parameters(recurse=True):
                    if not p.is_meta:
                        return p.device
                for b in emb.buffers(recurse=True):
                    if not b.is_meta:
                        return b.device
        except Exception:
            pass
    for p in model.parameters():
        if not p.is_meta:
            return p.device
    for b in model.buffers():
        if not b.is_meta:
            return b.device
    return torch.device("cpu")


def _effective_even_group_size(in_features: int, requested_group_size: int) -> Tuple[int, int]:
    """Return ``(group_size, padded_in_features)`` with even ``group_size >= 2``."""
    g = max(2, int(requested_group_size))
    if g % 2 != 0:
        g += 1
    if in_features < g:
        g = max(2, int(in_features) + (int(in_features) % 2))
    num_groups = math.ceil(max(1, int(in_features)) / g)
    padded_in = num_groups * g
    return g, padded_in


@torch.no_grad()
def quantize_weight_int4_packed(
    weight: torch.Tensor,
    group_size: int = 128,
    row_chunk_size: int = 16384,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Quantize 2D ``(out_features, in_features)`` weight into packed ``uint8`` + ``bfloat16`` scales.

    Processes rows in chunks so even ``262,144 x 5,376`` vocabulary embeddings use ``< 360 MB``
    of temporary ``float32`` CPU RAM during quantization.

    Returns ``(packed_u8, scales_bf16, eff_group_size, padded_in_features)`` where:
      - ``packed_u8`` has shape ``(out_features, num_groups, eff_group_size // 2)`` (``uint8``)
      - ``scales_bf16`` has shape ``(out_features, num_groups)`` (``bfloat16``)
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight tensor, got shape {tuple(weight.shape)}")
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    eff_g, padded_in = _effective_even_group_size(in_features, group_size)
    num_groups = padded_in // eff_g
    half_g = eff_g // 2

    packed_out = torch.empty((out_features, num_groups, half_g), dtype=torch.uint8, device="cpu")
    scales_out = torch.empty((out_features, num_groups), dtype=torch.bfloat16, device="cpu")

    step = max(1, int(row_chunk_size))
    for r0 in range(0, out_features, step):
        r1 = min(out_features, r0 + step)
        chunk_f32 = weight[r0:r1].detach().to(device="cpu", dtype=torch.float32)
        if padded_in > in_features:
            chunk_f32 = F.pad(chunk_f32, (0, padded_in - in_features), value=0.0)
        grouped = chunk_f32.view(r1 - r0, num_groups, eff_g)
        absmax = grouped.abs().amax(dim=-1, keepdim=True)
        scale = (absmax / 7.0).clamp(min=1e-7)
        q_signed = torch.round(grouped / scale).clamp(-8.0, 7.0).to(torch.int8)
        q_uint = (q_signed + 8).to(torch.uint8)
        packed_out[r0:r1] = q_uint[:, :, :half_g] | (q_uint[:, :, half_g:] << 4)
        scales_out[r0:r1] = scale.squeeze(-1).to(torch.bfloat16)
        del chunk_f32, grouped, absmax, scale, q_signed, q_uint

    return packed_out, scales_out, eff_g, padded_in


def dequantize_weight_int4_packed(
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    in_features: int,
    padded_in_features: int,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Transiently unpack ``(out_features, in_features)`` weight on its resident device."""
    out_features = int(packed_weight.shape[0])
    unpacked = torch.cat((packed_weight & 0x0F, packed_weight >> 4), dim=-1)
    w = ((unpacked.to(compute_dtype) - 8.0) * scales.to(compute_dtype).unsqueeze(-1)).reshape(
        out_features, padded_in_features
    )
    if padded_in_features != in_features:
        w = w[:, :in_features]
    return w


class _StreamingInt4LinearFn(torch.autograd.Function):
    """Autograd-safe streaming 4-bit linear projection with ZERO saved unpacked weights in HBM."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: torch.Tensor,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        bias: Optional[torch.Tensor],
        in_features: int,
        padded_in_features: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(packed_weight, scales)
        ctx.in_features = int(in_features)
        ctx.padded_in_features = int(padded_in_features)
        compute_dt = x.dtype if x.is_floating_point() else torch.bfloat16
        w = dequantize_weight_int4_packed(
            packed_weight=packed_weight,
            scales=scales,
            in_features=ctx.in_features,
            padded_in_features=ctx.padded_in_features,
            compute_dtype=compute_dt,
        )
        b = bias.to(device=x.device, dtype=compute_dt) if bias is not None else None
        out = F.linear(x.to(compute_dt), w, b)
        return out.to(x.dtype) if out.dtype != x.dtype else out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore[override]
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None
        packed_weight, scales = ctx.saved_tensors
        compute_dt = grad_output.dtype if grad_output.is_floating_point() else torch.bfloat16
        w = dequantize_weight_int4_packed(
            packed_weight=packed_weight,
            scales=scales,
            in_features=ctx.in_features,
            padded_in_features=ctx.padded_in_features,
            compute_dtype=compute_dt,
        )
        grad_x = torch.matmul(grad_output.to(compute_dt), w)
        if grad_x.dtype != grad_output.dtype:
            grad_x = grad_x.to(grad_output.dtype)
        return grad_x, None, None, None, None, None


class PackedInt4Linear(nn.Module):
    """Pure-PyTorch TPU/XLA/CPU-compatible 4-bit groupwise Linear layer (0.516 bytes/param)."""

    is_quantized_int4: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 128,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        eff_g, padded_in = _effective_even_group_size(self.in_features, group_size)
        self.group_size = eff_g
        self.padded_in_features = padded_in
        num_groups = padded_in // eff_g

        dev = torch.device(device)
        self.register_buffer(
            "packed_weight",
            torch.zeros((self.out_features, num_groups, eff_g // 2), dtype=torch.uint8, device=dev),
        )
        self.register_buffer(
            "scales",
            torch.ones((self.out_features, num_groups), dtype=torch.bfloat16, device=dev),
        )
        if bias:
            self.register_buffer("bias", torch.zeros(self.out_features, dtype=torch.bfloat16, device=dev))
        else:
            self.bias = None

    @property
    def weight(self) -> torch.Tensor:
        """Expose floating-point ``scales`` buffer for ``.weight.device`` / ``.weight.dtype`` compatibility."""
        return self.scales

    @property
    def device(self) -> torch.device:
        return self.packed_weight.device

    def storage_bytes(self) -> int:
        b = self.packed_weight.numel() * self.packed_weight.element_size()
        b += self.scales.numel() * self.scales.element_size()
        if self.bias is not None:
            b += self.bias.numel() * self.bias.element_size()
        return int(b)

    @classmethod
    @torch.no_grad()
    def from_tensor(
        cls,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        group_size: int = 128,
        device: torch.device | str = "cpu",
    ) -> "PackedInt4Linear":
        out_f, in_f = int(weight.shape[0]), int(weight.shape[1])
        packed_u8, scales_bf16, eff_g, padded_in = quantize_weight_int4_packed(weight, group_size=group_size)
        dev = torch.device(device)
        mod = cls(
            in_features=in_f,
            out_features=out_f,
            bias=(bias is not None),
            group_size=eff_g,
            device="meta",
        )
        mod.padded_in_features = padded_in
        mod.register_buffer("packed_weight", packed_u8.to(device=dev))
        mod.register_buffer("scales", scales_bf16.to(device=dev))
        if bias is not None:
            mod.register_buffer("bias", bias.detach().to(device=dev, dtype=torch.bfloat16))
        else:
            mod.bias = None
        return mod

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
        device: Optional[torch.device | str] = None,
    ) -> "PackedInt4Linear":
        target_dev = torch.device(device) if device is not None else linear.weight.device
        return cls.from_tensor(
            weight=linear.weight,
            bias=linear.bias,
            group_size=group_size,
            device=target_dev,
        )

    from_float = from_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.packed_weight.device:
            x = x.to(self.packed_weight.device)
        if torch.is_grad_enabled() and x.requires_grad:
            return _StreamingInt4LinearFn.apply(
                x,
                self.packed_weight,
                self.scales,
                self.bias,
                self.in_features,
                self.padded_in_features,
            )
        compute_dt = x.dtype if x.is_floating_point() else torch.bfloat16
        w = dequantize_weight_int4_packed(
            packed_weight=self.packed_weight,
            scales=self.scales,
            in_features=self.in_features,
            padded_in_features=self.padded_in_features,
            compute_dtype=compute_dt,
        )
        b = self.bias.to(device=x.device, dtype=compute_dt) if self.bias is not None else None
        return F.linear(x.to(compute_dt), w, b).to(x.dtype)


class PackedInt4Embedding(nn.Module):
    """4-bit groupwise Embedding table with optional Gemma ``embed_scale`` (``sqrt(hidden_size)``)."""

    is_quantized_int4: bool = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        group_size: int = 128,
        embed_scale: float = 1.0,
        padding_idx: Optional[int] = None,
        output_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.padding_idx = padding_idx
        self.embed_scale = float(embed_scale)
        self.output_dtype = output_dtype
        eff_g, padded_dim = _effective_even_group_size(self.embedding_dim, group_size)
        self.group_size = eff_g
        self.padded_embedding_dim = padded_dim
        num_groups = padded_dim // eff_g

        dev = torch.device(device)
        self.register_buffer(
            "packed_weight",
            torch.zeros((self.num_embeddings, num_groups, eff_g // 2), dtype=torch.uint8, device=dev),
        )
        self.register_buffer(
            "scales",
            torch.ones((self.num_embeddings, num_groups), dtype=torch.bfloat16, device=dev),
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.scales

    def storage_bytes(self) -> int:
        return int(
            self.packed_weight.numel() * self.packed_weight.element_size()
            + self.scales.numel() * self.scales.element_size()
        )

    @classmethod
    @torch.no_grad()
    def from_tensor(
        cls,
        weight: torch.Tensor,
        group_size: int = 128,
        embed_scale: float = 1.0,
        padding_idx: Optional[int] = None,
        output_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
    ) -> "PackedInt4Embedding":
        num_emb, emb_dim = int(weight.shape[0]), int(weight.shape[1])
        packed_u8, scales_bf16, eff_g, padded_dim = quantize_weight_int4_packed(weight, group_size=group_size)
        dev = torch.device(device)
        mod = cls(
            num_embeddings=num_emb,
            embedding_dim=emb_dim,
            group_size=eff_g,
            embed_scale=embed_scale,
            padding_idx=padding_idx,
            output_dtype=output_dtype,
            device="meta",
        )
        mod.padded_embedding_dim = padded_dim
        mod.register_buffer("packed_weight", packed_u8.to(device=dev))
        mod.register_buffer("scales", scales_bf16.to(device=dev))
        return mod

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.device != self.packed_weight.device:
            input_ids = input_ids.to(self.packed_weight.device)
        orig_shape = input_ids.shape
        flat_ids = input_ids.reshape(-1)
        p_rows = self.packed_weight.index_select(0, flat_ids)
        s_rows = self.scales.index_select(0, flat_ids)
        unpacked = torch.cat((p_rows & 0x0F, p_rows >> 4), dim=-1)
        emb = ((unpacked.to(self.output_dtype) - 8.0) * s_rows.to(self.output_dtype).unsqueeze(-1)).reshape(
            flat_ids.shape[0], self.padded_embedding_dim
        )
        if self.padded_embedding_dim != self.embedding_dim:
            emb = emb[:, : self.embedding_dim]
        if self.padding_idx is not None:
            pad_mask = (flat_ids == int(self.padding_idx)).unsqueeze(-1)
            emb = emb.masked_fill(pad_mask, 0.0)
        if abs(self.embed_scale - 1.0) > 1e-6:
            emb = emb * self.embed_scale
        return emb.view(*orig_shape, self.embedding_dim)


class PackedInt4TiedLMHead(nn.Module):
    """Zero-extra-HBM ``lm_head`` sharing ``PackedInt4Embedding``'s buffers without ``embed_scale``."""

    is_quantized_int4: bool = True

    def __init__(self, embedding: PackedInt4Embedding) -> None:
        super().__init__()
        self.in_features = embedding.embedding_dim
        self.out_features = embedding.num_embeddings
        self.padded_in_features = embedding.padded_embedding_dim
        self.bias: Optional[torch.Tensor] = None
        # Share exact underlying buffer storage with `embed_tokens` (0 extra bytes!)
        self.register_buffer("packed_weight", embedding.packed_weight, persistent=False)
        self.register_buffer("scales", embedding.scales, persistent=False)

    @property
    def weight(self) -> torch.Tensor:
        return self.scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.packed_weight.device:
            x = x.to(self.packed_weight.device)
        if torch.is_grad_enabled() and x.requires_grad:
            return _StreamingInt4LinearFn.apply(
                x,
                self.packed_weight,
                self.scales,
                None,
                self.in_features,
                self.padded_in_features,
            )
        compute_dt = x.dtype if x.is_floating_point() else torch.bfloat16
        w = dequantize_weight_int4_packed(
            packed_weight=self.packed_weight,
            scales=self.scales,
            in_features=self.in_features,
            padded_in_features=self.padded_in_features,
            compute_dtype=compute_dt,
        )
        return F.linear(x.to(compute_dt), w, None)


def _move_nested_tensors(obj: Any, target_device: torch.device) -> Any:
    """Recursively move tensors inside ``args`` / ``kwargs`` while preserving ``Cache`` objects."""
    if isinstance(obj, torch.Tensor):
        return obj.to(target_device) if obj.device != target_device else obj
    if isinstance(obj, tuple):
        return tuple(_move_nested_tensors(item, target_device) for item in obj)
    if isinstance(obj, list):
        return [_move_nested_tensors(item, target_device) for item in obj]
    if isinstance(obj, dict):
        return {k: _move_nested_tensors(v, target_device) for k, v in obj.items()}
    return obj


def install_hybrid_device_hooks(
    model: nn.Module,
    layers: Sequence[nn.Module],
    split_layer_idx: int,
    primary_device: torch.device,
    overflow_device: torch.device = torch.device("cpu"),
) -> List[Any]:
    """Attach ``register_forward_pre_hook`` cross-device handoffs for hybrid ``tpu:0 + cpu`` split."""
    handles: List[Any] = []
    num_layers = len(layers)

    def _make_pre_hook(target_dev: torch.device):
        def _pre_hook(_mod: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
            new_args = _move_nested_tensors(args, target_dev)
            new_kwargs = _move_nested_tensors(kwargs, target_dev)
            return new_args, new_kwargs

        return _pre_hook

    for idx in range(num_layers):
        dev = primary_device if idx < split_layer_idx else overflow_device
        handles.append(layers[idx].register_forward_pre_hook(_make_pre_hook(dev), with_kwargs=True))

    # Move hidden_states back to `primary_device` right before final `norm` and `lm_head`
    for tail_attr in ("norm", "final_layernorm", "ln_f"):
        for owner in (model, getattr(model, "model", None), getattr(model, "language_model", None)):
            if owner is not None and hasattr(owner, tail_attr):
                norm_mod = getattr(owner, tail_attr)
                if isinstance(norm_mod, nn.Module):
                    handles.append(norm_mod.register_forward_pre_hook(_make_pre_hook(primary_device), with_kwargs=True))
    if hasattr(model, "lm_head") and isinstance(getattr(model, "lm_head"), nn.Module):
        handles.append(model.lm_head.register_forward_pre_hook(_make_pre_hook(primary_device), with_kwargs=True))

    setattr(model, "_limbic_hybrid_hook_handles", handles)
    return handles


@torch.no_grad()
def quantize_model_in_place_int4(
    model: nn.Module,
    group_size: int = 128,
    primary_device: torch.device | str = "cpu",
    overflow_device: torch.device | str = "cpu",
    split_layer_idx: Optional[int] = None,
    quantize_overflow_layers: bool = True,
) -> Dict[str, Any]:
    """Quantize an already-instantiated ``nn.Module`` in-place and optionally split across two devices."""
    from limbic_zero.vocal_cord.gemma4_calibrator import resolve_decoder_layers

    prim_dev = torch.device(primary_device)
    over_dev = torch.device(overflow_device)
    layers = resolve_decoder_layers(model)
    num_layers = len(layers)
    split_idx = num_layers if split_layer_idx is None else max(0, min(num_layers, int(split_layer_idx)))

    # 1. Quantize input embeddings & tie lm_head if tied
    orig_emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if orig_emb is None:
        for owner in (getattr(model, "model", None), getattr(getattr(model, "model", None), "language_model", None), model):
            if owner is not None:
                for attr in ("embed_tokens", "embed"):
                    cand = getattr(owner, attr, None)
                    if isinstance(cand, nn.Embedding):
                        orig_emb = cand
                        break
            if orig_emb is not None:
                break
    tied_head = False
    if orig_emb is not None and isinstance(orig_emb, nn.Embedding):
        emb_scale = _extract_embedding_scale(orig_emb)
        lm_head_mod = getattr(model, "lm_head", None)
        is_tied = (
            isinstance(lm_head_mod, nn.Linear)
            and lm_head_mod.weight.data_ptr() == orig_emb.weight.data_ptr()
        )
        q_emb = PackedInt4Embedding.from_tensor(
            weight=orig_emb.weight,
            group_size=group_size,
            embed_scale=emb_scale,
            padding_idx=orig_emb.padding_idx,
            device=prim_dev,
        )
        _replace_Submodule_by_ref(model, orig_emb, q_emb)
        if is_tied:
            model.lm_head = PackedInt4TiedLMHead(q_emb)
            tied_head = True

    # 2. Quantize decoder blocks & align remaining floating-point norms/overflow weights to bfloat16
    quantized_linears = 0
    for idx, block in enumerate(layers):
        target_dev = prim_dev if idx < split_idx else over_dev
        do_q = True if idx < split_idx else bool(quantize_overflow_layers)
        if do_q:
            quantized_linears += _quantize_linears_in_subtree(block, group_size=group_size, device=target_dev)
        for p in block.parameters(recurse=True):
            if not p.is_meta and p.is_floating_point():
                p.data = p.data.to(device=target_dev, dtype=torch.bfloat16)
        for b_name, b_val in list(block.named_buffers(recurse=False)):
            if not b_val.is_meta:
                block._buffers[b_name] = b_val.to(device=target_dev)

    # 3. Quantize untied lm_head if still nn.Linear
    if not tied_head and hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
        model.lm_head = PackedInt4Linear.from_linear(model.lm_head, group_size=group_size, device=prim_dev)
        quantized_linears += 1

    # Move remaining root/norm modules onto primary_device (casting norm parameters to bfloat16)
    for owner in (model, getattr(model, "model", None), getattr(getattr(model, "model", None), "language_model", None)):
        if owner is None:
            continue
        for attr in ("norm", "final_layernorm", "final_norm", "ln_f"):
            mod = getattr(owner, attr, None)
            if isinstance(mod, nn.Module):
                mod.to(device=prim_dev, dtype=torch.bfloat16)
        for attr in ("rotary_emb", "rotary_emb_local"):
            mod = getattr(owner, attr, None)
            if isinstance(mod, nn.Module):
                mod.to(device=prim_dev)

    if split_idx < num_layers and prim_dev != over_dev:
        install_hybrid_device_hooks(
            model=model,
            layers=layers,
            split_layer_idx=split_idx,
            primary_device=prim_dev,
            overflow_device=over_dev,
        )

    return {
        "num_layers": num_layers,
        "split_layer_idx": split_idx,
        "quantized_linears": quantized_linears,
        "tied_lm_head": tied_head,
        "primary_device": str(prim_dev),
        "overflow_device": str(over_dev),
    }


def _quantize_linears_in_subtree(root: nn.Module, group_size: int, device: torch.device) -> int:
    count = 0
    for child_name, child_mod in list(root.named_children()):
        if isinstance(child_mod, nn.Linear):
            setattr(root, child_name, PackedInt4Linear.from_linear(child_mod, group_size=group_size, device=device))
            count += 1
        else:
            count += _quantize_linears_in_subtree(child_mod, group_size=group_size, device=device)
    return count


def _replace_Submodule_by_ref(root: nn.Module, target: nn.Module, replacement: nn.Module) -> bool:
    for child_name, child_mod in list(root.named_children()):
        if child_mod is target:
            setattr(root, child_name, replacement)
            return True
        if _replace_Submodule_by_ref(child_mod, target, replacement):
            return True
    return False


def _extract_embedding_scale(emb_mod: nn.Module) -> float:
    emb_dim = int(getattr(emb_mod, "embedding_dim", 0) or 1)
    if hasattr(emb_mod, "embed_scale"):
        raw = getattr(emb_mod, "embed_scale")
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, torch.Tensor):
            if not raw.is_meta:
                return float(raw.detach().cpu().item())
            return math.sqrt(float(emb_dim))
    if "ScaledWordEmbedding" in emb_mod.__class__.__name__:
        return math.sqrt(float(emb_dim))
    return 1.0


@contextmanager
def _meta_linear_and_embedding_init() -> Iterator[None]:
    """Allocate ``nn.Linear`` & ``nn.Embedding`` weights on ``meta`` (0 MB) while keeping RoPE/norm buffers on CPU."""
    orig_linear_init = nn.Linear.__init__
    orig_emb_init = nn.Embedding.__init__

    def _patched_linear_init(self: nn.Linear, in_features: int, out_features: int, bias: bool = True, device: Any = None, dtype: Any = None) -> None:
        orig_linear_init(self, in_features, out_features, bias=bias, device="meta" if device is None else device, dtype=dtype)

    def _patched_emb_init(
        self: nn.Embedding,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[torch.Tensor] = None,
        _freeze: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        orig_emb_init(
            self,
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight,
            _freeze=_freeze,
            device="meta" if device is None else device,
            dtype=dtype,
        )

    nn.Linear.__init__ = _patched_linear_init  # type: ignore[assignment]
    nn.Embedding.__init__ = _patched_emb_init  # type: ignore[assignment]
    try:
        yield
    finally:
        nn.Linear.__init__ = orig_linear_init  # type: ignore[assignment]
        nn.Embedding.__init__ = orig_emb_init  # type: ignore[assignment]


_MULTIMODAL_SKIP_PREFIXES: Tuple[str, ...] = (
    "vision_tower.",
    "model.vision_tower.",
    "embed_vision.",
    "model.embed_vision.",
    "vision_embedder.",
    "model.vision_embedder.",
    "audio_tower.",
    "model.audio_tower.",
    "embed_audio.",
    "model.embed_audio.",
    "multi_modal_projector.",
    "model.multi_modal_projector.",
    "multimodal_embedder.",
    "model.multimodal_embedder.",
)


def _candidate_causal_lm_keys(raw_key: str) -> List[str]:
    """Generate candidate CausalLM parameter keys from multimodal/text checkpoint keys."""
    cands = [raw_key]
    if raw_key.startswith("model.language_model."):
        tail = raw_key[len("model.language_model.") :]
        cands.append(f"model.{tail}")
        cands.append(tail)
    elif raw_key.startswith("language_model.model."):
        tail = raw_key[len("language_model.model.") :]
        cands.append(f"model.{tail}")
        cands.append(tail)
    elif raw_key.startswith("language_model."):
        tail = raw_key[len("language_model.") :]
        cands.append(f"model.{tail}")
        cands.append(tail)
    return cands


def _estimate_layer_int4_bytes(block: nn.Module, group_size: int) -> int:
    total = 0
    for mod in block.modules():
        if isinstance(mod, nn.Linear):
            eff_g, padded_in = _effective_even_group_size(mod.in_features, group_size)
            num_groups = padded_in // eff_g
            total += mod.out_features * num_groups * (eff_g // 2)  # uint8
            total += mod.out_features * num_groups * 2  # bf16 scales
            if mod.bias is not None:
                total += mod.out_features * 2
    for p in block.parameters(recurse=True):
        if not p.is_meta:
            total += p.numel() * 2
    return total


@torch.no_grad()
def load_quantized_causal_lm_for_tpu(
    model_id: str,
    tpu_device: Optional[torch.device | str] = None,
    *,
    hbm_budget_gb: float = 12.5,
    group_size: int = 128,
    overflow_device: torch.device | str = "cpu",
    quantize_overflow_layers: bool = False,
    cache_dir: Optional[str | Path] = None,
    evict_shards_on_low_disk: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Stream-quantize a HuggingFace model (e.g., ``google/gemma-4-31B-it``) onto TPU v5e-1 + Host CPU.

    Never materializes the full 62 GB ``bf16`` checkpoint in Host RAM or disk if free disk is tight.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
    from limbic_zero.vocal_cord.gemma4_calibrator import resolve_decoder_layers

    t0 = time.perf_counter()
    if tpu_device is None:
        tpu_info = detect_tpu_device()
        prim_dev = tpu_info.device if tpu_info.available else torch.device("cpu")
    else:
        prim_dev = torch.device(tpu_device)
    over_dev = torch.device(overflow_device)

    raw_cfg = AutoConfig.from_pretrained(model_id, cache_dir=str(cache_dir) if cache_dir else None)
    text_cfg = getattr(raw_cfg, "text_config", None) or raw_cfg
    tie_embeddings = bool(getattr(text_cfg, "tie_word_embeddings", getattr(raw_cfg, "tie_word_embeddings", True)))
    if hasattr(text_cfg, "tie_word_embeddings"):
        text_cfg.tie_word_embeddings = tie_embeddings

    print(
        f"[TPU-4BIT] Building skeleton CausalLM from text_config "
        f"(layers={getattr(text_cfg, 'num_hidden_layers', '?')}, "
        f"hidden={getattr(text_cfg, 'hidden_size', '?')}, tie_embeddings={tie_embeddings})...",
        flush=True,
    )
    with _meta_linear_and_embedding_init():
        model = AutoModelForCausalLM.from_config(text_cfg, dtype=torch.bfloat16)
    model.eval()

    try:
        gen_cfg = GenerationConfig.from_pretrained(model_id, cache_dir=str(cache_dir) if cache_dir else None)
        model.generation_config = gen_cfg
    except Exception:
        pass

    layers = resolve_decoder_layers(model)
    num_layers = len(layers)
    orig_emb = model.get_input_embeddings()
    emb_scale = _extract_embedding_scale(orig_emb) if orig_emb is not None else 1.0
    vocab_size = int(getattr(orig_emb, "num_embeddings", getattr(text_cfg, "vocab_size", 262144)))
    hidden_size = int(getattr(orig_emb, "embedding_dim", getattr(text_cfg, "hidden_size", 5376)))
    eff_g_emb, pad_h_emb = _effective_even_group_size(hidden_size, group_size)
    emb_bytes = vocab_size * (pad_h_emb // eff_g_emb) * ((eff_g_emb // 2) + 2)
    untied_head_bytes = 0 if tie_embeddings else emb_bytes

    budget_bytes = int(float(hbm_budget_gb) * (1024 ** 3))
    used_hbm_bytes = emb_bytes + untied_head_bytes + (32 * 1024 * 1024)  # 32 MiB norm/RoPE reserve
    split_idx = 0
    if prim_dev.type == "cpu":
        split_idx = num_layers
    else:
        for idx, blk in enumerate(layers):
            blk_b = _estimate_layer_int4_bytes(blk, group_size=group_size)
            if used_hbm_bytes + blk_b <= budget_bytes:
                used_hbm_bytes += blk_b
                split_idx = idx + 1
            else:
                break

    print(
        f"[TPU-4BIT] Hybrid placement plan: embed_tokens + layers[0..{max(0, split_idx - 1)}] + norm + lm_head "
        f"-> {prim_dev} (~{used_hbm_bytes / (1024 ** 3):.2f} GiB / budget {hbm_budget_gb:.2f} GiB); "
        f"overflow layers[{split_idx}..{num_layers - 1}] ({num_layers - split_idx} layers) -> {over_dev}.",
        flush=True,
    )

    # Index target submodules & parameters
    module_map: Dict[str, nn.Module] = dict(model.named_modules())
    expected_keys: Dict[str, Tuple[str, str]] = {}  # full_param_key -> (module_path, attr_name)
    for mod_path, mod in module_map.items():
        if isinstance(mod, (nn.Linear, nn.Embedding)):
            w_key = f"{mod_path}.weight" if mod_path else "weight"
            expected_keys[w_key] = (mod_path, "weight")
            if isinstance(mod, nn.Linear) and mod.bias is not None:
                b_key = f"{mod_path}.bias" if mod_path else "bias"
                expected_keys[b_key] = (mod_path, "bias")
        else:
            for p_name, _p in mod.named_parameters(recurse=False):
                full_k = f"{mod_path}.{p_name}" if mod_path else p_name
                expected_keys[full_k] = (mod_path, p_name)

    def _device_for_module_path(mod_path: str) -> Tuple[torch.device, bool]:
        """Return ``(target_device, quantize_to_int4)`` for ``mod_path``."""
        parts = mod_path.split(".")
        for i, tok_p in enumerate(parts):
            if tok_p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                l_i = int(parts[i + 1])
                if l_i < split_idx:
                    return prim_dev, True
                return over_dev, bool(quantize_overflow_layers)
        return prim_dev, True

    # Resolve local directory or remote HF shards
    local_dir = Path(model_id) if Path(model_id).is_dir() else None
    shard_to_keys: Dict[str, List[Tuple[str, str]]] = {}  # shard_file -> [(raw_key, mapped_key)]

    if local_dir is not None:
        idx_file = local_dir / "model.safetensors.index.json"
        if idx_file.is_file():
            w_map = json.loads(idx_file.read_text(encoding="utf-8")).get("weight_map", {})
            for raw_k, shard_name in w_map.items():
                if raw_k.startswith(_MULTIMODAL_SKIP_PREFIXES):
                    continue
                for cand in _candidate_causal_lm_keys(raw_k):
                    if cand in expected_keys:
                        shard_to_keys.setdefault(shard_name, []).append((raw_k, cand))
                        break
        else:
            single_sf = local_dir / "model.safetensors"
            with safe_open(str(single_sf), framework="pt", device="cpu") as sf:
                for raw_k in sf.keys():
                    if raw_k.startswith(_MULTIMODAL_SKIP_PREFIXES):
                        continue
                    for cand in _candidate_causal_lm_keys(raw_k):
                        if cand in expected_keys:
                            shard_to_keys.setdefault("model.safetensors", []).append((raw_k, cand))
                            break
    else:
        try:
            idx_path = hf_hub_download(
                repo_id=model_id,
                filename="model.safetensors.index.json",
                cache_dir=str(cache_dir) if cache_dir else None,
            )
            w_map = json.loads(Path(idx_path).read_text(encoding="utf-8")).get("weight_map", {})
            for raw_k, shard_name in w_map.items():
                if raw_k.startswith(_MULTIMODAL_SKIP_PREFIXES):
                    continue
                for cand in _candidate_causal_lm_keys(raw_k):
                    if cand in expected_keys:
                        shard_to_keys.setdefault(shard_name, []).append((raw_k, cand))
                        break
        except Exception:
            shard_to_keys["model.safetensors"] = []

    # Determine whether local disk is tight (< 75 GB free) so we stream-evict temporary shard files
    dl_root = Path(cache_dir) if cache_dir else Path("/tmp/limbic_tpu_shards")
    dl_root.mkdir(parents=True, exist_ok=True)
    free_disk_gb = shutil.disk_usage(dl_root).free / (1024 ** 3)
    use_ephemeral_shard_dir = (
        local_dir is None
        and evict_shards_on_low_disk
        and free_disk_gb < 78.0
        and os.environ.get("LIMBIC_KEEP_HF_SHARDS", "0") != "1"
    )
    if use_ephemeral_shard_dir:
        print(
            f"[TPU-4BIT] Free disk is {free_disk_gb:.1f} GiB (< 78 GiB): enabling shard-by-shard "
            f"streaming eviction so peak disk usage stays < 6 GiB!",
            flush=True,
        )

    pending_linear_biases: Dict[str, torch.Tensor] = {}
    quantized_emb_module: Optional[PackedInt4Embedding] = None
    loaded_lm_head_explicitly = False
    shards_list = sorted(shard_to_keys.keys())

    for s_idx, shard_name in enumerate(shards_list, start=1):
        if local_dir is not None:
            shard_path = str(local_dir / shard_name)
        elif use_ephemeral_shard_dir:
            ephem_dir = dl_root / "_ephemeral_shard"
            ephem_dir.mkdir(parents=True, exist_ok=True)
            shard_path = hf_hub_download(
                repo_id=model_id,
                filename=shard_name,
                local_dir=str(ephem_dir),
            )
        else:
            shard_path = hf_hub_download(
                repo_id=model_id,
                filename=shard_name,
                cache_dir=str(cache_dir) if cache_dir else None,
            )

        print(f"[TPU-4BIT] Streaming & quantizing shard [{s_idx}/{len(shards_list)}] {shard_name}...", flush=True)
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            pairs = shard_to_keys[shard_name]
            if not pairs:
                for raw_k in sf.keys():
                    if raw_k.startswith(_MULTIMODAL_SKIP_PREFIXES):
                        continue
                    for cand in _candidate_causal_lm_keys(raw_k):
                        if cand in expected_keys:
                            pairs.append((raw_k, cand))
                            break

            # Process `.bias` entries before or alongside `.weight`
            for raw_k, mapped_k in pairs:
                mod_path, attr_name = expected_keys[mapped_k]
                if attr_name == "bias" and isinstance(module_map.get(mod_path), (nn.Linear, PackedInt4Linear)):
                    b_ten = sf.get_tensor(raw_k)
                    cur_mod = module_map[mod_path]
                    if isinstance(cur_mod, PackedInt4Linear):
                        cur_mod.register_buffer(
                            "bias", b_ten.to(device=cur_mod.device, dtype=torch.bfloat16)
                        )
                    elif isinstance(cur_mod, nn.Linear) and not cur_mod.weight.is_meta:
                        cur_mod.bias = nn.Parameter(
                            b_ten.to(device=cur_mod.weight.device, dtype=cur_mod.weight.dtype),
                            requires_grad=False,
                        )
                    else:
                        pending_linear_biases[mod_path] = b_ten

            for raw_k, mapped_k in pairs:
                mod_path, attr_name = expected_keys[mapped_k]
                if attr_name == "bias" and mod_path in pending_linear_biases:
                    continue
                target_mod = module_map[mod_path]
                target_dev, do_quant = _device_for_module_path(mod_path)
                ten = sf.get_tensor(raw_k)

                if isinstance(target_mod, nn.Embedding) and attr_name == "weight":
                    q_emb = PackedInt4Embedding.from_tensor(
                        weight=ten,
                        group_size=group_size,
                        embed_scale=emb_scale,
                        padding_idx=target_mod.padding_idx,
                        device=prim_dev,
                    )
                    _set_submodule_by_path(model, mod_path, q_emb)
                    module_map[mod_path] = q_emb
                    quantized_emb_module = q_emb
                elif isinstance(target_mod, nn.Linear) and attr_name == "weight":
                    bias_ten = pending_linear_biases.pop(mod_path, None)
                    if mod_path == "lm_head":
                        loaded_lm_head_explicitly = True
                    if do_quant:
                        q_lin = PackedInt4Linear.from_tensor(
                            weight=ten,
                            bias=bias_ten,
                            group_size=group_size,
                            device=target_dev,
                        )
                        _set_submodule_by_path(model, mod_path, q_lin)
                        module_map[mod_path] = q_lin
                    else:
                        dense_lin = nn.Linear(
                            target_mod.in_features,
                            target_mod.out_features,
                            bias=(bias_ten is not None),
                            device=target_dev,
                            dtype=torch.bfloat16,
                        )
                        dense_lin.weight.data.copy_(ten.to(device=target_dev, dtype=torch.bfloat16))
                        dense_lin.weight.requires_grad_(False)
                        if bias_ten is not None and dense_lin.bias is not None:
                            dense_lin.bias.data.copy_(bias_ten.to(device=target_dev, dtype=torch.bfloat16))
                            dense_lin.bias.requires_grad_(False)
                        _set_submodule_by_path(model, mod_path, dense_lin)
                        module_map[mod_path] = dense_lin
                else:
                    # 1D RMSNorm / LayerNorm / scalar parameters
                    param_dt = torch.bfloat16 if ten.is_floating_point() else ten.dtype
                    new_param = nn.Parameter(ten.to(device=target_dev, dtype=param_dt), requires_grad=False)
                    setattr(target_mod, attr_name, new_param)

                del ten

        if use_ephemeral_shard_dir:
            try:
                Path(shard_path).unlink(missing_ok=True)
            except Exception:
                pass
        sync_tpu_device(prim_dev)
        gc.collect()

    # Bind tied `lm_head` to `PackedInt4Embedding` with ZERO extra HBM
    if quantized_emb_module is not None and (tie_embeddings or not loaded_lm_head_explicitly):
        if hasattr(model, "lm_head"):
            model.lm_head = PackedInt4TiedLMHead(quantized_emb_module)

    # Move non-layer buffers (RoPE `inv_freq`, etc.) onto their owning device
    for idx, blk in enumerate(layers):
        b_dev = prim_dev if idx < split_idx else over_dev
        _move_direct_buffers_and_params(blk, b_dev)
    for owner in (model, getattr(model, "model", None)):
        if owner is None:
            continue
        for child_name, child_mod in owner.named_children():
            if child_name == "layers":
                continue
            _move_all_non_meta_tensors(child_mod, prim_dev)

    if split_idx < num_layers and prim_dev != over_dev:
        install_hybrid_device_hooks(
            model=model,
            layers=layers,
            split_layer_idx=split_idx,
            primary_device=prim_dev,
            overflow_device=over_dev,
        )

    # Sanity-verify zero leftover `meta` tensors
    leftover_meta = [
        n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.is_meta
    ]
    if leftover_meta:
        raise RuntimeError(f"Unloaded meta tensors remain after streaming quantization: {leftover_meta[:8]}")

    setattr(model, "_limbic_tpu_4bit_hybrid", True)
    sync_tpu_device(prim_dev)
    elapsed = round(time.perf_counter() - t0, 2)
    summary = {
        "model_id": model_id,
        "primary_device": str(prim_dev),
        "overflow_device": str(over_dev),
        "num_hidden_layers": num_layers,
        "tpu_layers": split_idx,
        "cpu_overflow_layers": num_layers - split_idx,
        "estimated_tpu_hbm_gb": round(used_hbm_bytes / (1024 ** 3), 3),
        "group_size": group_size,
        "tie_word_embeddings": tie_embeddings,
        "load_seconds": elapsed,
    }
    print(f"[TPU-4BIT] Streaming 4-bit load complete in {elapsed}s: {summary}", flush=True)
    return model, summary


def _set_submodule_by_path(root: nn.Module, dotted_path: str, new_mod: nn.Module) -> None:
    parts = dotted_path.split(".")
    parent: nn.Module = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


def _move_direct_buffers_and_params(root: nn.Module, target_dev: torch.device) -> None:
    for mod in root.modules():
        for p_name, p in list(mod._parameters.items()):
            if p is not None and not p.is_meta and p.device != target_dev:
                mod._parameters[p_name] = nn.Parameter(p.to(target_dev), requires_grad=p.requires_grad)
        for b_name, b in list(mod._buffers.items()):
            if b is not None and not b.is_meta and b.device != target_dev:
                mod._buffers[b_name] = b.to(target_dev)


def _move_all_non_meta_tensors(root: nn.Module, target_dev: torch.device) -> None:
    _move_direct_buffers_and_params(root, target_dev)
