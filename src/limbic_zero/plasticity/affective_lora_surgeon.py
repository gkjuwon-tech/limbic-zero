"""Stage 8 Affect-Driven Cortical Weight Surgery on Frozen Gemma-4-12B-IT.

Implants per-soul switchable low-rank plastic cortical synapses (``AffectivePlasticLoRALinear``)
into middle-to-upper Transformer decoder blocks (``L21, L23, L27, L29, L33, L35, L39``) of
``google/gemma-4-12B-it`` so a single ``23 GB`` frozen base cortex in GPU VRAM simultaneously
hosts independent, live-evolving synaptic weight manifolds for ``ADAM-01``, ``ABEL-02``,
``PANDORA-03``, and experimental ablation banks.

Scientific & Zero-Cheat Guarantees:
  1. At initialization ``lora_B == 0`` -> ``Delta W == 0`` (exact bitwise identity with base LLM).
  2. Every cortical weight change ``Delta W != 0`` comes strictly from genuine PyTorch
     ``loss.backward()`` on causal prediction error gated by the organism's live
     ``NeuromodulatedGateReading`` (``ACh M1``, ``DA D1/RPE``, ``NE alpha2A``, ``Cortisol``).
  3. Under Flat-Affect Ablation (``flat_affect_ablation=True``), ``eta_eff == 0.0`` and
     ``||Delta W||_F`` remains strictly unchanged.
  4. Exact Frobenius norm ``||B @ A||_F`` is computed in ``O(r^2 (d_in + d_out))`` via
     ``Tr((B^T B)(A A^T))`` without materializing dense matrices.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from limbic_zero.plasticity.weight_homeostasis import (
    HomeostasisAudit,
    NeuromodulatedGateReading,
    WeightHomeostasisGuard,
    compute_neuromodulated_plasticity_gate,
)


DEFAULT_PLASTIC_LAYERS_48: Tuple[int, ...] = (21, 23, 27, 29, 33, 35, 39)
DEFAULT_PLASTIC_LAYERS_60: Tuple[int, ...] = (23, 27, 29, 35, 39, 41, 47)
DEFAULT_TARGET_PROJ_PATHS: Tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.v_proj",
    "mlp.down_proj",
)


class _PlasticRoutingState(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.active_bank_id: Optional[str] = None
        self.bypass_plasticity: bool = False


class SoulLoRABank(nn.Module):
    """Low-rank plastic synaptic matrices ``(lora_A, lora_B)`` for one organism on one projection."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        device: torch.device | str = "cpu",
        seed: int = 1337,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / max(1, int(rank))

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        a_init = torch.empty((self.rank, self.in_features), dtype=torch.float32)
        nn.init.kaiming_uniform_(a_init, a=math.sqrt(5), generator=gen)

        dev = torch.device(device)
        self.lora_A = nn.Parameter(a_init.to(device=dev), requires_grad=False)
        self.lora_B = nn.Parameter(
            torch.zeros((self.out_features, self.rank), dtype=torch.float32, device=dev),
            requires_grad=False,
        )

    @torch.no_grad()
    def reset_zero_(self) -> None:
        self.lora_B.zero_()

    @torch.no_grad()
    def scale_delta_(self, factor: float) -> None:
        self.lora_B.mul_(float(factor))

    @torch.no_grad()
    def delta_frobenius_norm(self) -> float:
        """Exact ``||(alpha/r) * B @ A||_F`` via ``sqrt(sum((B^T B) * (A A^T)))``."""
        btb = torch.matmul(self.lora_B.T, self.lora_B)  # (r, r)
        aat = torch.matmul(self.lora_A, self.lora_A.T)  # (r, r)
        sq = float(torch.sum(btb * aat).clamp(min=0.0).item())
        return self.scaling * math.sqrt(sq)

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
        orig_dev = x.device
        x_in = x.to(self.lora_A.device) if orig_dev != self.lora_A.device else x
        x_f32 = x_in.float()
        low_rank = F.linear(x_f32, self.lora_A)
        delta = F.linear(low_rank, self.lora_B) * self.scaling
        out = delta.to(dtype=x.dtype)
        return out.to(orig_dev) if out.device != orig_dev else out


def _is_wrappable_linear_module(mod: Any) -> bool:
    return isinstance(mod, nn.Linear) or bool(getattr(mod, "is_quantized_int4", False))


class AffectivePlasticLoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` or ``PackedInt4Linear`` with per-soul ``SoulLoRABank`` synapses."""

    def __init__(
        self,
        base_linear: nn.Module,
        routing_state: _PlasticRoutingState,
        layer_idx: int,
        proj_name: str,
        rank: int = 16,
        alpha: float = 32.0,
    ) -> None:
        super().__init__()
        self.base_linear = base_linear
        w_ref = getattr(self.base_linear, "weight", None)
        if isinstance(w_ref, torch.Tensor) and w_ref.is_floating_point():
            w_ref.requires_grad_(False)
        b_ref = getattr(self.base_linear, "bias", None)
        if isinstance(b_ref, torch.Tensor) and b_ref.is_floating_point():
            b_ref.requires_grad_(False)

        self._routing = routing_state
        self.layer_idx = int(layer_idx)
        self.proj_name = str(proj_name)
        self.in_features = int(base_linear.in_features)  # type: ignore[arg-type]
        self.out_features = int(base_linear.out_features)  # type: ignore[arg-type]
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.banks = nn.ModuleDict()

    @property
    def weight(self) -> torch.Tensor:
        return self.base_linear.weight  # type: ignore[return-value]

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base_linear.bias  # type: ignore[return-value]

    def ensure_bank(self, bank_id: str) -> SoulLoRABank:
        if bank_id not in self.banks:
            seed_bytes = hashlib.sha256(f"{bank_id}:{self.layer_idx}:{self.proj_name}".encode("utf-8")).digest()
            seed_int = int.from_bytes(seed_bytes[:4], "big") & 0x7FFFFFFF
            dev = self.base_linear.weight.device  # type: ignore[union-attr]
            bank = SoulLoRABank(
                in_features=self.in_features,
                out_features=self.out_features,
                rank=self.rank,
                alpha=self.alpha,
                device=dev,
                seed=seed_int,
            )
            if dev.type == "xla" and not getattr(self.base_linear, "is_quantized_int4", False):
                try:
                    from limbic_zero.tpu.xla_spmd_cortex import (
                        compute_gemma4_lora_partition_specs,
                        initialize_tpu_spmd_runtime,
                    )

                    spmd_ctx = initialize_tpu_spmd_runtime(cache_dir=None)
                    if spmd_ctx.num_devices > 1:
                        spec_a, spec_b = compute_gemma4_lora_partition_specs(
                            self.proj_name,
                            self.in_features,
                            self.out_features,
                            num_devices=spmd_ctx.num_devices,
                        )
                        if spec_a is not None:
                            spmd_ctx.mark_sharding(bank.lora_A.data, spec_a)
                        if spec_b is not None:
                            spmd_ctx.mark_sharding(bank.lora_B.data, spec_b)
                except Exception:
                    pass
            self.banks[bank_id] = bank
        return self.banks[bank_id]  # type: ignore[return-value]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        if self._routing.bypass_plasticity:
            return base_out
        bank_id = self._routing.active_bank_id
        if not bank_id or bank_id not in self.banks:
            return base_out
        bank: SoulLoRABank = self.banks[bank_id]  # type: ignore[assignment]
        return base_out + bank.forward_delta(x).to(base_out.device)


def _extract_logits_tensor(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        return model_output
    if hasattr(model_output, "logits") and isinstance(model_output.logits, torch.Tensor):
        return model_output.logits
    if isinstance(model_output, (tuple, list)) and len(model_output) > 0 and isinstance(model_output[0], torch.Tensor):
        return model_output[0]
    raise RuntimeError(f"Cannot extract logits tensor from model output of type {type(model_output).__name__}")


class CorticalWeightSurgeon:
    """Manages per-soul plastic cortical LoRA synapses, neuromodulated backprop, & homeostasis."""

    SURGERY_SEQ_BUCKET: int = 256
    ANCHOR_SEQ_BUCKET: int = 32

    def __init__(
        self,
        model: nn.Module,
        gen_lock: Optional[threading.Lock] = None,
        *,
        plastic_layers: Optional[Sequence[int]] = None,
        target_proj_paths: Sequence[str] = DEFAULT_TARGET_PROJ_PATHS,
        rank: int = 16,
        alpha: float = 32.0,
        base_lr: float = 2.8e-4,
        checkpoint_dir: Optional[str | Path] = None,
        homeostasis: Optional[WeightHomeostasisGuard] = None,
        use_xla_static_bucket: Optional[bool] = None,
    ) -> None:
        self.model = model
        self.gen_lock = gen_lock or threading.Lock()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.base_lr = float(base_lr)
        self.homeostasis = homeostasis or WeightHomeostasisGuard()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        from limbic_zero.vocal_cord.gemma4_calibrator import resolve_decoder_layers
        from limbic_zero.vocal_cord.tpu_quantizer import resolve_model_input_device

        self._routing = _PlasticRoutingState()
        self.layers = resolve_decoder_layers(model)
        num_layers = len(self.layers)
        self.device = resolve_model_input_device(model)
        is_4bit_hybrid = bool(getattr(model, "_limbic_tpu_4bit_hybrid", False))
        self.is_xla_static: bool = (
            (self.device.type == "xla" and not is_4bit_hybrid)
            if use_xla_static_bucket is None
            else bool(use_xla_static_bucket)
        )

        if plastic_layers is not None:
            chosen = [int(i) for i in plastic_layers if 0 <= int(i) < num_layers]
        elif num_layers >= 56:
            chosen = [i for i in DEFAULT_PLASTIC_LAYERS_60 if i < num_layers]
        elif num_layers >= 40:
            chosen = [i for i in DEFAULT_PLASTIC_LAYERS_48 if i < num_layers]
        else:
            # Proportional middle-to-upper cortical layers for compact/mock Transformers
            start_l = max(0, num_layers // 2)
            chosen = list(range(start_l, num_layers))
        self.plastic_layers: Tuple[int, ...] = tuple(sorted(set(chosen)))

        self.wrapped_modules: List[AffectivePlasticLoRALinear] = []
        self._install_plastic_wrappers(target_proj_paths)

        self._optimizers: Dict[str, torch.optim.AdamW] = {}
        self._surgery_history: Dict[str, List[Dict[str, Any]]] = {}
        self._step_counts: Dict[str, int] = {}

    def _install_plastic_wrappers(self, target_proj_paths: Sequence[str]) -> None:
        for l_idx in self.plastic_layers:
            block = self.layers[l_idx]
            wrapped_in_layer = 0
            for dotted_path in target_proj_paths:
                parts = dotted_path.split(".")
                parent: Any = block
                ok = True
                for attr in parts[:-1]:
                    if hasattr(parent, attr):
                        parent = getattr(parent, attr)
                    else:
                        ok = False
                        break
                if not ok:
                    continue
                leaf_name = parts[-1]
                if not hasattr(parent, leaf_name):
                    continue
                leaf = getattr(parent, leaf_name)
                if isinstance(leaf, AffectivePlasticLoRALinear):
                    self.wrapped_modules.append(leaf)
                    wrapped_in_layer += 1
                elif _is_wrappable_linear_module(leaf):
                    wrapper = AffectivePlasticLoRALinear(
                        base_linear=leaf,
                        routing_state=self._routing,
                        layer_idx=l_idx,
                        proj_name=dotted_path,
                        rank=self.rank,
                        alpha=self.alpha,
                    )
                    setattr(parent, leaf_name, wrapper)
                    self.wrapped_modules.append(wrapper)
                    wrapped_in_layer += 1

            # Fallback for compact mock decoder blocks without HuggingFace `self_attn`/`mlp` naming
            if wrapped_in_layer == 0:
                for child_name, child_mod in list(block.named_children()):
                    if _is_wrappable_linear_module(child_mod):
                        wrapper = AffectivePlasticLoRALinear(
                            base_linear=child_mod,
                            routing_state=self._routing,
                            layer_idx=l_idx,
                            proj_name=child_name,
                            rank=self.rank,
                            alpha=self.alpha,
                        )
                        setattr(block, child_name, wrapper)
                        self.wrapped_modules.append(wrapper)

    def register_soul_bank(self, bank_id: str, auto_load_checkpoint: bool = True) -> None:
        for mod in self.wrapped_modules:
            mod.ensure_bank(bank_id)
        self._step_counts.setdefault(bank_id, 0)
        self._surgery_history.setdefault(bank_id, [])
        if auto_load_checkpoint and self.checkpoint_dir is not None:
            ckpt = self.checkpoint_dir / f"{bank_id}_cortical_lora.pt"
            if ckpt.is_file():
                try:
                    self.load_soul_synapses(bank_id, ckpt)
                except Exception:
                    pass

    @contextmanager
    def activate_soul_cortex(
        self,
        bank_id: Optional[str],
        enabled: bool = True,
    ) -> Iterator[None]:
        prev_bank = self._routing.active_bank_id
        prev_bypass = self._routing.bypass_plasticity
        try:
            if enabled and bank_id:
                self._routing.active_bank_id = bank_id
                self._routing.bypass_plasticity = False
            else:
                self._routing.active_bank_id = None
                self._routing.bypass_plasticity = True
            yield
        finally:
            self._routing.active_bank_id = prev_bank
            self._routing.bypass_plasticity = prev_bypass

    def _get_bank_parameters(self, bank_id: str) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for mod in self.wrapped_modules:
            bank = mod.ensure_bank(bank_id)
            params.append(bank.lora_A)
            params.append(bank.lora_B)
        return params

    def _get_optimizer(self, bank_id: str, lr: float) -> torch.optim.AdamW:
        params = self._get_bank_parameters(bank_id)
        if bank_id not in self._optimizers:
            self._optimizers[bank_id] = torch.optim.AdamW(
                params,
                lr=lr,
                betas=(0.9, 0.98),
                weight_decay=1e-3,
            )
        else:
            for pg in self._optimizers[bank_id].param_groups:
                pg["lr"] = lr
        return self._optimizers[bank_id]

    @torch.no_grad()
    def reset_soul_synapses(self, bank_id: str) -> Dict[str, Any]:
        for mod in self.wrapped_modules:
            if bank_id in mod.banks:
                mod.banks[bank_id].reset_zero_()  # type: ignore[operator]
        self._optimizers.pop(bank_id, None)
        self._step_counts[bank_id] = 0
        self._surgery_history[bank_id] = []
        return self.inspect_soul_synapses(bank_id)

    @torch.no_grad()
    def inspect_soul_synapses(self, bank_id: str) -> Dict[str, Any]:
        """Return exact layer-by-layer Frobenius norms and surgical history for ``bank_id``."""
        self.register_soul_bank(bank_id, auto_load_checkpoint=False)
        layer_norms: Dict[int, float] = {l_idx: 0.0 for l_idx in self.plastic_layers}
        module_details: List[Dict[str, Any]] = []
        total_sq = 0.0
        trainable_scalars = 0

        for mod in self.wrapped_modules:
            bank = mod.ensure_bank(bank_id)
            fnorm = bank.delta_frobenius_norm()
            layer_norms[mod.layer_idx] = math.hypot(layer_norms.get(mod.layer_idx, 0.0), fnorm)
            total_sq += fnorm * fnorm
            trainable_scalars += bank.lora_A.numel() + bank.lora_B.numel()
            module_details.append(
                {
                    "layer_idx": mod.layer_idx,
                    "proj": mod.proj_name,
                    "frobenius_norm": round(fnorm, 6),
                }
            )

        hist = self._surgery_history.get(bank_id, [])
        last_evt = hist[-1] if hist else None
        return {
            "bank_id": bank_id,
            "rank": self.rank,
            "alpha": self.alpha,
            "plastic_layers": list(self.plastic_layers),
            "wrapped_module_count": len(self.wrapped_modules),
            "trainable_parameter_count": trainable_scalars,
            "surgical_steps": self._step_counts.get(bank_id, 0),
            "total_lora_frobenius_norm": round(math.sqrt(total_sq), 6),
            "per_layer_frobenius_norm": {str(k): round(v, 6) for k, v in sorted(layer_norms.items())},
            "last_surgery": last_evt,
            "recent_history": hist[-10:],
        }

    def _encode_training_pair_cpu(
        self,
        processor_or_tokenizer: Any,
        experience_text: str,
        prompt_text: Optional[str] = None,
        max_seq_tokens: int = 384,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode ``(prompt_text, experience_text)`` on CPU into ``(input_ids_cpu, labels_cpu)``."""
        tok = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)

        if prompt_text and hasattr(tok, "apply_chat_template") and not tok.__class__.__name__.startswith("_Mock"):
            user_msgs = [{"role": "user", "content": prompt_text}]
            full_msgs = [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": experience_text},
            ]
            try:
                p_enc = tok.apply_chat_template(
                    user_msgs,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                f_enc = tok.apply_chat_template(
                    full_msgs,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                p_ids = p_enc["input_ids"] if isinstance(p_enc, dict) or hasattr(p_enc, "keys") else p_enc
                f_ids = f_enc["input_ids"] if isinstance(f_enc, dict) or hasattr(f_enc, "keys") else f_enc
                prompt_len = int(p_ids.shape[-1])
                input_ids_cpu = f_ids[:, :max_seq_tokens].detach().to(device="cpu", dtype=torch.long)
                labels_cpu = input_ids_cpu.clone()
                mask_len = min(prompt_len, int(labels_cpu.shape[-1]) - 1)
                if mask_len > 0:
                    labels_cpu[:, :mask_len] = -100
                return input_ids_cpu, labels_cpu
            except Exception:
                pass

        combined = f"{prompt_text}\n{experience_text}".strip() if prompt_text else experience_text.strip()
        enc = tok(combined, return_tensors="pt")
        raw_ids = enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "keys") else enc
        input_ids_cpu = raw_ids[:, :max_seq_tokens].detach().to(device="cpu", dtype=torch.long)
        labels_cpu = input_ids_cpu.clone()
        if prompt_text:
            p_enc = tok(prompt_text, return_tensors="pt")
            p_ids = p_enc["input_ids"] if isinstance(p_enc, dict) or hasattr(p_enc, "keys") else p_enc
            mask_len = min(int(p_ids.shape[-1]), max(0, int(labels_cpu.shape[-1]) - 2))
            if mask_len > 0:
                labels_cpu[:, :mask_len] = -100
        return input_ids_cpu, labels_cpu

    def _encode_training_pair(
        self,
        processor_or_tokenizer: Any,
        experience_text: str,
        prompt_text: Optional[str] = None,
        max_seq_tokens: int = 384,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode ``(prompt_text, experience_text)`` into ``(input_ids, attention_mask, labels)``."""
        ids_cpu, labels_cpu = self._encode_training_pair_cpu(
            processor_or_tokenizer=processor_or_tokenizer,
            experience_text=experience_text,
            prompt_text=prompt_text,
            max_seq_tokens=max_seq_tokens,
        )
        input_ids = ids_cpu.to(self.device)
        labels = labels_cpu.to(self.device)
        attn = torch.ones_like(input_ids, device=self.device)
        return input_ids, attn, labels

    def _build_static_bucket_forward_kwargs(
        self,
        ids_cpu: torch.Tensor,
        bucket_len: int,
    ) -> Tuple[Dict[str, Any], int]:
        """Pad ``ids_cpu`` ``(1, L)`` to fixed ``bucket_len`` with exact hybrid 4D causal/sliding masks."""
        from limbic_zero.tpu.xla_spmd_cortex import build_static_hybrid_masks_prefill

        raw_cfg = getattr(self.model, "config", None)
        text_cfg = (
            raw_cfg.get_text_config()
            if raw_cfg is not None and hasattr(raw_cfg, "get_text_config")
            else getattr(raw_cfg, "text_config", raw_cfg)
        )
        pad_id = int(getattr(text_cfg, "pad_token_id", 0) or 0)
        sliding_window = int(getattr(text_cfg, "sliding_window", 1024) or 1024)
        is_gemma4_hybrid = bool(
            hasattr(text_cfg, "layer_types")
            and "gemma4" in str(getattr(text_cfg, "model_type", "")).lower()
        )
        param_dtype = next(self.model.parameters()).dtype
        mask_dtype = param_dtype if param_dtype in (torch.bfloat16, torch.float16, torch.float32) else torch.bfloat16

        vlen = min(bucket_len, int(ids_cpu.shape[-1]))
        padded_cpu = torch.full((1, bucket_len), pad_id, dtype=torch.long)
        padded_cpu[0, :vlen] = ids_cpu[0, :vlen]

        if raw_cfg is not None:
            pos_cpu = torch.zeros((1, bucket_len), dtype=torch.long)
            pos_cpu[0, :vlen] = torch.arange(vlen, dtype=torch.long)
            valid_len_t = torch.tensor([vlen], dtype=torch.long, device=self.device)
            masks = build_static_hybrid_masks_prefill(
                bucket_len=bucket_len,
                max_cache_len=bucket_len,
                valid_prompt_len_t=valid_len_t,
                sliding_window=sliding_window,
                device=self.device,
                mask_dtype=mask_dtype,
            )
            return (
                {
                    "input_ids": padded_cpu.to(self.device),
                    "attention_mask": masks if is_gemma4_hybrid else masks["full_attention"],
                    "position_ids": pos_cpu.to(self.device),
                    "use_cache": False,
                },
                vlen,
            )

        attn_cpu = torch.zeros((1, bucket_len), dtype=torch.long)
        attn_cpu[0, :vlen] = 1
        return (
            {
                "input_ids": padded_cpu.to(self.device),
                "attention_mask": attn_cpu.to(self.device),
            },
            vlen,
        )

    @staticmethod
    def _mark_xla_step_if_needed(dev: torch.device) -> None:
        if dev.type == "xla":
            try:
                import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]
                xm.mark_step()
            except Exception:
                pass

    @staticmethod
    def _causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    @torch.no_grad()
    def _enforce_homeostasis(
        self,
        bank_id: str,
        probe_ids: torch.Tensor,
        probe_mask: torch.Tensor,
        *,
        raw_probe_ids_cpu: Optional[torch.Tensor] = None,
    ) -> HomeostasisAudit:
        pre_snap = self.inspect_soul_synapses(bank_id)
        pre_norm = float(pre_snap["total_lora_frobenius_norm"])
        clamped_mods = 0

        # 1. Per-module Frobenius ceiling
        for mod in self.wrapped_modules:
            bank = mod.ensure_bank(bank_id)
            fn = bank.delta_frobenius_norm()
            if fn > self.homeostasis.max_module_frobenius:
                bank.scale_delta_(self.homeostasis.max_module_frobenius / max(1e-6, fn))
                clamped_mods += 1

        # 2. Global Frobenius ceiling
        mid_norm = math.sqrt(sum(mod.ensure_bank(bank_id).delta_frobenius_norm() ** 2 for mod in self.wrapped_modules))
        global_scale = 1.0
        if mid_norm > self.homeostasis.max_total_frobenius:
            global_scale = self.homeostasis.max_total_frobenius / max(1e-6, mid_norm)
            for mod in self.wrapped_modules:
                mod.ensure_bank(bank_id).scale_delta_(global_scale)

        # 3. Exact KL divergence check vs frozen base cortex P_base on last <= 24 real tokens
        if self.is_xla_static and raw_probe_ids_cpu is not None:
            tail_cpu = raw_probe_ids_cpu[:, -min(24, int(raw_probe_ids_cpu.shape[-1])) :]
            anchor_kw, anchor_vlen = self._build_static_bucket_forward_kwargs(
                tail_cpu, bucket_len=self.ANCHOR_SEQ_BUCKET
            )
            with self.activate_soul_cortex(bank_id, enabled=False):
                base_logits = _extract_logits_tensor(self.model(**anchor_kw))[:, :anchor_vlen, :].float()
            with self.activate_soul_cortex(bank_id, enabled=True):
                plastic_logits = _extract_logits_tensor(self.model(**anchor_kw))[:, :anchor_vlen, :].float()
            self._mark_xla_step_if_needed(self.device)
        else:
            anchor_tail = probe_ids[:, -min(24, int(probe_ids.shape[-1])) :]
            anchor_mask = probe_mask[:, -min(24, int(probe_mask.shape[-1])) :]
            anchor_kw = {"input_ids": anchor_tail, "attention_mask": anchor_mask}
            anchor_vlen = int(anchor_tail.shape[-1])
            with self.activate_soul_cortex(bank_id, enabled=False):
                base_logits = _extract_logits_tensor(self.model(**anchor_kw)).float()
            with self.activate_soul_cortex(bank_id, enabled=True):
                plastic_logits = _extract_logits_tensor(self.model(**anchor_kw)).float()

        log_p_base = F.log_softmax(base_logits, dim=-1)
        log_q_plast = F.log_softmax(plastic_logits, dim=-1)
        p_base = log_p_base.exp()
        kl_nats = float(torch.sum(p_base * (log_p_base - log_q_plast), dim=-1).mean().item())

        kl_rescaled = False
        if kl_nats > self.homeostasis.max_anchor_kl_nats and kl_nats > 1e-6:
            shrink = math.sqrt(self.homeostasis.max_anchor_kl_nats / kl_nats) * 0.92
            shrink = max(0.25, min(0.95, shrink))
            for mod in self.wrapped_modules:
                mod.ensure_bank(bank_id).scale_delta_(shrink)
            global_scale *= shrink
            kl_rescaled = True
            with self.activate_soul_cortex(bank_id, enabled=True):
                plastic_logits2 = _extract_logits_tensor(self.model(**anchor_kw))[:, :anchor_vlen, :].float()
            self._mark_xla_step_if_needed(self.device)
            log_q2 = F.log_softmax(plastic_logits2, dim=-1)
            kl_nats = float(torch.sum(p_base * (log_p_base - log_q2), dim=-1).mean().item())

        post_norm = math.sqrt(sum(mod.ensure_bank(bank_id).delta_frobenius_norm() ** 2 for mod in self.wrapped_modules))
        return HomeostasisAudit(
            pre_frobenius_norm=round(pre_norm, 6),
            post_frobenius_norm=round(post_norm, 6),
            anchor_kl_nats=round(max(0.0, kl_nats), 6),
            kl_rescaled=kl_rescaled,
            norm_clamped_modules=clamped_mods,
            scale_applied=round(global_scale, 5),
        )

    def surgical_plasticity_step(
        self,
        processor_or_tokenizer: Any,
        soul: Any,
        experience_text: str,
        *,
        prompt_text: Optional[str] = None,
        bank_id: Optional[str] = None,
        num_gradient_steps: int = 2,
        flat_affect_ablation: bool = False,
        sleep_swr_boost: float = 1.0,
        source_tag: str = "online_experience",
        persist_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """Run genuine neuromodulated cortical backpropagation ``loss.backward()`` on ``Gemma-4``."""
        t0 = time.perf_counter()
        target_bank = bank_id or getattr(soul, "organism_id", "DEFAULT-SOUL")
        self.register_soul_bank(target_bank, auto_load_checkpoint=False)

        gate = compute_neuromodulated_plasticity_gate(
            soul=soul,
            base_lr=self.base_lr,
            flat_affect_ablation=flat_affect_ablation,
            sleep_swr_boost=sleep_swr_boost,
        )

        if gate.effective_lr <= 0.0 or not (experience_text or "").strip():
            snap_now = self.inspect_soul_synapses(target_bank)
            evt = {
                "timestamp_unix": round(time.time(), 3),
                "bank_id": target_bank,
                "source_tag": source_tag,
                "updated": False,
                "reason": "zero_neuromodulated_gate" if gate.effective_lr <= 0.0 else "empty_experience",
                "gate": gate.to_dict(),
                "loss_before": None,
                "loss_after": None,
                "grad_l2_norm": 0.0,
                "total_lora_frobenius_norm": snap_now["total_lora_frobenius_norm"],
                "latency_sec": round(time.perf_counter() - t0, 4),
            }
            self._surgery_history.setdefault(target_bank, []).append(evt)
            return evt

        max_toks = self.SURGERY_SEQ_BUCKET if self.is_xla_static else 384
        raw_ids_cpu, raw_labels_cpu = self._encode_training_pair_cpu(
            processor_or_tokenizer=processor_or_tokenizer,
            experience_text=experience_text,
            prompt_text=prompt_text,
            max_seq_tokens=max_toks,
        )
        if int(raw_ids_cpu.shape[-1]) < 3 or bool((raw_labels_cpu[:, 1:] == -100).all().item()):
            raw_labels_cpu = raw_ids_cpu.clone()

        if self.is_xla_static:
            fwd_kwargs, vlen = self._build_static_bucket_forward_kwargs(
                raw_ids_cpu, bucket_len=self.SURGERY_SEQ_BUCKET
            )
            padded_labels_cpu = torch.full((1, self.SURGERY_SEQ_BUCKET), -100, dtype=torch.long)
            padded_labels_cpu[0, :vlen] = raw_labels_cpu[0, :vlen]
            labels = padded_labels_cpu.to(self.device)
            input_ids = fwd_kwargs["input_ids"]
            attn_mask = torch.ones_like(input_ids)
        else:
            input_ids = raw_ids_cpu.to(self.device)
            attn_mask = torch.ones_like(input_ids, device=self.device)
            labels = raw_labels_cpu.to(self.device)
            fwd_kwargs = {"input_ids": input_ids, "attention_mask": attn_mask}

        steps_n = max(1, min(8, int(num_gradient_steps)))
        params = self._get_bank_parameters(target_bank)

        with self.gen_lock:
            self.model.eval()
            for p in params:
                p.requires_grad_(True)

            loss_before_val = 0.0
            grad_norm_val = 0.0
            try:
                opt = self._get_optimizer(target_bank, lr=gate.effective_lr)
                with self.activate_soul_cortex(target_bank, enabled=True):
                    for step_idx in range(steps_n):
                        opt.zero_grad(set_to_none=True)
                        with torch.enable_grad():
                            out = self.model(**fwd_kwargs)
                            logits = _extract_logits_tensor(out)
                            loss = self._causal_lm_loss(logits, labels)
                            loss_detached = loss.detach()
                            loss.backward()
                            gnorm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                            opt.step()
                        self._mark_xla_step_if_needed(self.device)
                        if step_idx == 0:
                            loss_before_val = float(loss_detached.cpu().item())
                        grad_norm_val = (
                            float(gnorm.detach().cpu().item()) if isinstance(gnorm, torch.Tensor) else float(gnorm)
                        )
                    opt.zero_grad(set_to_none=True)

                homeo = self._enforce_homeostasis(
                    target_bank,
                    input_ids,
                    attn_mask,
                    raw_probe_ids_cpu=raw_ids_cpu,
                )

                with torch.no_grad(), self.activate_soul_cortex(target_bank, enabled=True):
                    out_after = self.model(**fwd_kwargs)
                    loss_after_t = self._causal_lm_loss(_extract_logits_tensor(out_after), labels)
                    self._mark_xla_step_if_needed(self.device)
                    loss_after_val = float(loss_after_t.cpu().item())
            finally:
                for p in params:
                    p.grad = None
                    p.requires_grad_(False)

        # Couple cortical LTP consolidation back into living LimbicBrain pallial synapses + DA relief
        if soul is not None and not flat_affect_ablation:
            try:
                loss_drop = max(0.0, loss_before_val - loss_after_val)
                from limbic_zero.mind.biophysics import DalesLawSynapse
                from limbic_zero.vocal_cord.vocal_cord_engine import SensoryAfferentStimulus

                with soul.lock, torch.no_grad():
                    for mod in soul.brain.pallium.modules():
                        if isinstance(mod, DalesLawSynapse):
                            d_pal = 0.012 * gate.affect_gate * mod.eligibility_trace
                            mod.weight_mag.data.add_(d_pal).mul_(mod.structural_mask).clamp_(0.0, mod.config.weight_max)
                    if loss_drop > 0.01:
                        soul._step_internal(
                            ticks=6,
                            stim=SensoryAfferentStimulus(
                                prosodic_warmth=min(0.45, 0.20 * loss_drop),
                                epistemic_novelty=0.18,
                            ),
                        )
            except Exception:
                pass

        self._step_counts[target_bank] = self._step_counts.get(target_bank, 0) + steps_n
        if persist_checkpoint and self.checkpoint_dir is not None:
            try:
                self.save_soul_synapses(target_bank)
            except Exception:
                pass

        snap_after = self.inspect_soul_synapses(target_bank)
        evt = {
            "timestamp_unix": round(time.time(), 3),
            "bank_id": target_bank,
            "source_tag": source_tag,
            "updated": True,
            "gradient_steps": steps_n,
            "gate": gate.to_dict(),
            "loss_before": round(loss_before_val, 5),
            "loss_after": round(loss_after_val, 5),
            "loss_reduction": round(loss_before_val - loss_after_val, 5),
            "grad_l2_norm": round(grad_norm_val, 5),
            "homeostasis": homeo.to_dict(),
            "total_lora_frobenius_norm": snap_after["total_lora_frobenius_norm"],
            "per_layer_frobenius_norm": snap_after["per_layer_frobenius_norm"],
            "surgical_steps_total": self._step_counts[target_bank],
            "latency_sec": round(time.perf_counter() - t0, 4),
        }
        hist = self._surgery_history.setdefault(target_bank, [])
        hist.append(evt)
        if len(hist) > 60:
            self._surgery_history[target_bank] = hist[-60:]
        return evt

    @torch.no_grad()
    def apply_sleep_synaptic_scaling(self, bank_id: str) -> Dict[str, Any]:
        """Apply Tononi-Cirelli (2006) Synaptic Homeostasis down-selection during Slow-Wave Sleep."""
        self.register_soul_bank(bank_id, auto_load_checkpoint=False)
        pre = self.inspect_soul_synapses(bank_id)["total_lora_frobenius_norm"]
        factor = self.homeostasis.sleep_shy_decay
        for mod in self.wrapped_modules:
            mod.ensure_bank(bank_id).scale_delta_(factor)
        post = self.inspect_soul_synapses(bank_id)["total_lora_frobenius_norm"]
        return {
            "bank_id": bank_id,
            "shy_decay_factor": factor,
            "frobenius_before": pre,
            "frobenius_after": post,
        }

    @torch.no_grad()
    def save_soul_synapses(self, bank_id: str, path: Optional[str | Path] = None) -> Path:
        if path is not None:
            dest = Path(path)
        elif self.checkpoint_dir is not None:
            dest = self.checkpoint_dir / f"{bank_id}_cortical_lora.pt"
        else:
            raise ValueError("No checkpoint path or checkpoint_dir configured.")
        dest.parent.mkdir(parents=True, exist_ok=True)

        tensors: Dict[str, Dict[str, torch.Tensor]] = {}
        hasher = hashlib.sha256()
        for mod in self.wrapped_modules:
            bank = mod.ensure_bank(bank_id)
            key = f"L{mod.layer_idx}:{mod.proj_name}"
            a_cpu = bank.lora_A.detach().cpu().float()
            b_cpu = bank.lora_B.detach().cpu().float()
            tensors[key] = {"lora_A": a_cpu, "lora_B": b_cpu}
            hasher.update(key.encode("utf-8"))
            hasher.update(a_cpu.numpy().tobytes())
            hasher.update(b_cpu.numpy().tobytes())

        payload = {
            "schema": "limbic0.cortical_lora.v1",
            "bank_id": bank_id,
            "rank": self.rank,
            "alpha": self.alpha,
            "plastic_layers": list(self.plastic_layers),
            "surgical_steps": self._step_counts.get(bank_id, 0),
            "sha256": hasher.hexdigest(),
            "saved_unix": time.time(),
            "tensors": tensors,
        }
        torch.save(payload, dest)
        return dest

    @torch.no_grad()
    def load_soul_synapses(self, bank_id: str, path: str | Path) -> Dict[str, Any]:
        raw = torch.load(Path(path), map_location="cpu", weights_only=False)
        tensors = raw.get("tensors", {})
        loaded = 0
        for mod in self.wrapped_modules:
            key = f"L{mod.layer_idx}:{mod.proj_name}"
            if key in tensors:
                bank = mod.ensure_bank(bank_id)
                t_pair = tensors[key]
                if tuple(t_pair["lora_A"].shape) == tuple(bank.lora_A.shape):
                    bank.lora_A.copy_(t_pair["lora_A"].to(device=bank.lora_A.device, dtype=torch.float32))
                    bank.lora_B.copy_(t_pair["lora_B"].to(device=bank.lora_B.device, dtype=torch.float32))
                    loaded += 1
        self._step_counts[bank_id] = int(raw.get("surgical_steps", self._step_counts.get(bank_id, 0)))
        return {
            "bank_id": bank_id,
            "loaded_modules": loaded,
            "sha256": str(raw.get("sha256", ""))[:16],
            "surgical_steps": self._step_counts[bank_id],
        }

    def snapshot_all_banks(self, bank_ids: Sequence[str], souls: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        banks_out: Dict[str, Any] = {}
        for bid in bank_ids:
            info = self.inspect_soul_synapses(bid)
            if souls and bid in souls:
                info["live_neuromodulated_gate"] = compute_neuromodulated_plasticity_gate(
                    souls[bid], base_lr=self.base_lr
                ).to_dict()
            banks_out[bid] = info
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "base_lr": self.base_lr,
            "plastic_layers": list(self.plastic_layers),
            "wrapped_module_count": len(self.wrapped_modules),
            "banks": banks_out,
        }
