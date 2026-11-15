"""Verification Suite for Colab TPU v5e-1 4-Bit Quantizer, LoRA Plasticity over Int4,
CORS / External HTML Chat Client Export, and Colab TPU Notebook Integrity.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import threading
from typing import Any, Dict, List
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

from limbic_zero.plasticity import CorticalWeightSurgeon
from limbic_zero.vocal_cord import (
    ContrastiveSubspaceExtractor,
    Gemma4SteeringConfig,
    Gemma4VocalCordSurgeon,
    LivingSoulOrganism,
    PackedInt4Embedding,
    PackedInt4Linear,
    PackedInt4TiedLMHead,
    VocalCordHTTPServer,
    VocalCordRuntimeHub,
    install_hybrid_device_hooks,
    quantize_model_in_place_int4,
)
from limbic_zero.vocal_cord.tpu_quantizer import (
    dequantize_weight_int4_packed,
    quantize_weight_int4_packed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOULS_DIR = REPO_ROOT / "checkpoints" / "souls"


class _ScaledEmbedding(nn.Embedding):
    """Mimics Gemma4TextScaledWordEmbedding with embed_scale = sqrt(hidden_size)."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = float(embedding_dim ** 0.5)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale


class _DecoderBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()

        class _Attn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 0.20 * self.o_proj(self.q_proj(x) + self.v_proj(x))

        class _MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
                self.up_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
                self.down_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 0.20 * self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))

        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = _Attn()
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = _MLP()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h + self.self_attn(self.norm1(h))
        h = h + self.mlp(self.norm2(h))
        return h


class _TiedMiniGemma4ForCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 320, hidden_size: int = 128, num_layers: int = 6) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        class _TextBody(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = _ScaledEmbedding(vocab_size, hidden_size)
                self.layers = nn.ModuleList([_DecoderBlock(hidden_size) for _ in range(num_layers)])
                self.norm = nn.LayerNorm(hidden_size)

        self.model = _TextBody()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **_kw: Any) -> Any:
        ids = input_ids.clamp(0, self.vocab_size - 1)
        h = self.model.embed_tokens(ids)
        for blk in self.model.layers:
            h = blk(h)
        h = self.model.norm(h)
        logits = self.lm_head(h)
        return type("Out", (), {"logits": logits})()

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 8, **_kw: Any) -> torch.Tensor:
        cur = input_ids
        for _ in range(min(8, max_new_tokens)):
            out = self.forward(cur)
            assert out.logits.device == cur.device
            nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=-1)
        return cur


class _SimpleTokenizer:
    def _enc(self, text: str) -> Dict[str, torch.Tensor]:
        ids = [2] + [int(b) % 300 + 10 for b in text.encode("utf-8")[:96]]
        t = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def __call__(self, text: str, **_kw: Any) -> Dict[str, torch.Tensor]:
        return self._enc(text)

    def apply_chat_template(self, messages: List[Dict[str, Any]], **_kw: Any) -> Dict[str, torch.Tensor]:
        joined = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        return self._enc(joined)

    def decode(self, token_ids: torch.Tensor, **_kw: Any) -> str:
        vals = [int(x) for x in (token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids)]
        return "응, 잘 들려! (" + ",".join(str(v) for v in vals[:6]) + ")"


def test_01_packed_int4_quantize_dequantize_compression_and_streaming_autograd() -> None:
    torch.manual_seed(77)
    # Odd rows & non-divisible-by-128 columns to stress padding & trimming
    w_orig = torch.randn(97, 300, dtype=torch.float32) * 0.08
    packed, scales, eff_g, padded_in = quantize_weight_int4_packed(w_orig, group_size=128)

    assert packed.dtype == torch.uint8
    assert scales.dtype == torch.bfloat16
    assert eff_g == 128
    assert padded_in == 384
    # 3 groups of 128 (=384 padded cols -> 64 packed bytes per group)
    assert packed.shape == (97, 3, 64)
    assert scales.shape == (97, 3)

    w_rec = dequantize_weight_int4_packed(
        packed_weight=packed,
        scales=scales,
        in_features=300,
        padded_in_features=padded_in,
        compute_dtype=torch.float32,
    )
    assert w_rec.shape == w_orig.shape
    cos = F.cosine_similarity(w_orig.flatten(), w_rec.flatten(), dim=0)
    assert float(cos.item()) > 0.990

    # Streaming autograd backward wrt input activations (used by CorticalWeightSurgeon)
    dense = nn.Linear(256, 192, bias=True)
    q_lin = PackedInt4Linear.from_linear(dense, group_size=128)
    assert q_lin.storage_bytes() < (dense.weight.numel() * 2) * 0.30  # ~4x smaller than fp16/bf16

    x_ref = torch.randn(2, 7, 256, requires_grad=True)
    x_q = x_ref.detach().clone().requires_grad_(True)

    y_ref = dense(x_ref)
    y_q = q_lin(x_q)
    assert float(F.cosine_similarity(y_ref.flatten(), y_q.flatten(), dim=0).item()) > 0.990

    grad_out = torch.randn_like(y_ref)
    (y_ref * grad_out).sum().backward()
    (y_q * grad_out).sum().backward()
    assert x_q.grad is not None and x_ref.grad is not None
    grad_cos = F.cosine_similarity(x_ref.grad.flatten(), x_q.grad.flatten(), dim=0)
    assert float(grad_cos.item()) > 0.990


def test_02_in_place_int4_causal_lm_tied_head_steering_and_online_lora_surgery() -> None:
    torch.manual_seed(123)
    model = _TiedMiniGemma4ForCausalLM(vocab_size=320, hidden_size=128, num_layers=6)
    tok = _SimpleTokenizer()

    sample_ids = torch.tensor([[2, 15, 29, 44, 88]], dtype=torch.long)
    with torch.no_grad():
        ref_logits = model(sample_ids).logits.float()

    summary = quantize_model_in_place_int4(
        model,
        group_size=64,
        split_layer_idx=4,
        quantize_overflow_layers=False,
    )
    assert summary["split_layer_idx"] == 4
    assert summary["quantized_linears"] == 4 * 7  # 4 quantized blocks * 7 Linear projections/block
    assert summary["tied_lm_head"] is True
    assert isinstance(model.model.embed_tokens, PackedInt4Embedding)
    assert abs(model.model.embed_tokens.embed_scale - (128 ** 0.5)) < 1e-5
    assert isinstance(model.lm_head, PackedInt4TiedLMHead)
    assert model.lm_head.packed_weight.data_ptr() == model.model.embed_tokens.packed_weight.data_ptr()
    assert isinstance(model.model.layers[0].self_attn.q_proj, PackedInt4Linear)
    # Upper overflow layers >= split_layer_idx remain standard Linear for CPU overflow compatibility
    assert isinstance(model.model.layers[5].self_attn.q_proj, nn.Linear)

    install_hybrid_device_hooks(
        model=model,
        layers=list(model.model.layers),
        split_layer_idx=4,
        primary_device=torch.device("cpu"),
        overflow_device=torch.device("cpu"),
    )

    with torch.no_grad():
        q_logits = model(sample_ids).logits.float()
    logits_cos = F.cosine_similarity(ref_logits.flatten(), q_logits.flatten(), dim=0)
    assert float(logits_cos.item()) > 0.980

    # Subspace calibration + 6D steering on the 4-bit quantized model
    cfg = Gemma4SteeringConfig(model_id="mock/gemma4-tpu-int4", target_layers=(1, 2, 3), layer_weights=(0.3, 0.4, 0.3))
    subspace = ContrastiveSubspaceExtractor(config=cfg).extract_and_calibrate(model, encode_fn=tok._enc)
    vocal_surgeon = Gemma4VocalCordSurgeon(model=model, subspace=subspace, config=cfg)
    human = LivingSoulOrganism(str(SOULS_DIR / "human01_ordinary_evolved.soul.pt"))

    gen_res = vocal_surgeon.generate_with_soul(tok, human, "오늘 컨디션 어때?", steer_enabled=True, max_new_tokens=6)
    assert len(gen_res["layer_injections"]) == 3
    assert all(item["injected_delta_l2_norm"] > 0.0 for item in gen_res["layer_injections"])

    # Online CorticalWeightSurgeon LoRA update over PackedInt4Linear layers
    packed_snapshot = model.model.layers[2].self_attn.q_proj.base_linear.packed_weight.clone() if hasattr(
        model.model.layers[2].self_attn.q_proj, "base_linear"
    ) else model.model.layers[2].self_attn.q_proj.packed_weight.clone()
    weight_surgeon = CorticalWeightSurgeon(
        model=model,
        plastic_layers=(1, 2, 3),
        rank=4,
    )
    step_report = weight_surgeon.surgical_plasticity_step(
        processor_or_tokenizer=tok,
        soul=human,
        experience_text="4비트 그룹 양자화로 HBM 대역폭을 아끼면서 시냅스 가소성을 유지했어.",
        prompt_text="방금 새로 배운 양자화 원리를 정리해 보자.",
        num_gradient_steps=3,
        persist_checkpoint=False,
    )
    assert step_report["updated"] is True
    assert step_report["loss_after"] < step_report["loss_before"]
    assert step_report["total_lora_frobenius_norm"] > 0.0
    # Base int4 packed weights remain 100% immutable inside AffectivePlasticLoRALinear.base_linear
    wrapped_q = model.model.layers[2].self_attn.q_proj
    assert torch.equal(wrapped_q.base_linear.packed_weight, packed_snapshot)


def test_03_cors_client_html_export_and_colab_tpu_notebook_integrity() -> None:
    torch.manual_seed(909)
    model = _TiedMiniGemma4ForCausalLM(vocab_size=256, hidden_size=64, num_layers=4)
    quantize_model_in_place_int4(model, group_size=64)
    tok = _SimpleTokenizer()
    cfg = Gemma4SteeringConfig(model_id="mock/gemma4-tpu-int4", target_layers=(1, 2), layer_weights=(0.5, 0.5))
    subspace = ContrastiveSubspaceExtractor(config=cfg).extract_and_calibrate(model, encode_fn=tok._enc)
    surgeon = Gemma4VocalCordSurgeon(model=model, subspace=subspace, config=cfg)
    souls = {"HUMAN-01": LivingSoulOrganism(str(SOULS_DIR / "human01_ordinary_evolved.soul.pt"))}
    fake_cf = "https://limbic-v5e-demo.trycloudflare.com"
    hub = VocalCordRuntimeHub(
        surgeon=surgeon,
        processor_or_tokenizer=tok,
        souls=souls,
        public_urls={"cloudflare": fake_cf},
        enable_autonomous_evolution=False,
    )

    srv = VocalCordHTTPServer(("127.0.0.1", 0), hub=hub)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    try:
        # 1. CORS preflight for external file:// or cross-origin HTML chat client
        opt_req = urllib.request.Request(f"http://127.0.0.1:{port}/api/chat/stream", method="OPTIONS")
        with opener.open(opt_req, timeout=5.0) as opt_resp:
            assert opt_resp.status == 204
            assert opt_resp.headers.get("Access-Control-Allow-Origin") == "*"
            assert opt_resp.headers.get("Access-Control-Allow-Private-Network") == "true"

        # 2. Pre-configured external HTML download endpoint
        with opener.open(f"http://127.0.0.1:{port}/api/client-html", timeout=5.0) as html_resp:
            assert html_resp.status == 200
            assert "limbic_external_chat.html" in (html_resp.headers.get("Content-Disposition") or "")
            exported_html = html_resp.read().decode("utf-8")
            assert f'window.LIMBIC_DEFAULT_SERVER_URL = "{fake_cf}"' in exported_html
    finally:
        srv.shutdown()
        srv.server_close()

    # 3. Standalone web/limbic_chat_client.html & Colab TPU notebook verification
    standalone_html = (REPO_ROOT / "web" / "limbic_chat_client.html").read_text(encoding="utf-8")
    assert "LIMBIC-0 · External Chat Client" in standalone_html
    assert "LIMBIC_DEFAULT_SERVER_URL" in standalone_html
    assert "trycloudflare.com" in standalone_html

    nb_path = REPO_ROOT / "notebooks" / "limbic_zero_colab_tpu_v5e_4bit.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["metadata"]["accelerator"] == "TPU"
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 4
    for cell in code_cells:
        ast.parse("".join(cell["source"]), filename=cell["id"])
