"""Stage 5 Verification Suite: Gemma-4 Mid-Layer Contrastive Subspace & Vocal Cord Surgery."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from limbic_zero.vocal_cord import (
    SUBSPACE_AXIS_NAMES,
    CalibratedSubspaceBundle,
    ContrastiveSubspaceExtractor,
    Gemma4SteeringConfig,
    Gemma4VocalCordSurgeon,
    LivingSoulOrganism,
    VocalCordHTTPServer,
    VocalCordRuntimeHub,
    lowdin_decorrelate_rows,
    resolve_decoder_layers,
    transduce_utterance_to_sensory_nerves,
)


class _MockGemma4TextDecoderLayer(nn.Module):
    """Mimics Gemma4UnifiedTextDecoderLayer returning a bare Tensor (B, S, D)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 0.20 * self.mlp(self.norm(hidden_states))


class _MockGemma4UnifiedModel(nn.Module):
    """Mimics Gemma4UnifiedForConditionalGeneration (`model.model.language_model.layers`)."""

    def __init__(self, vocab_size: int = 256, hidden_size: int = 64, num_layers: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        class _LangModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden_size)
                self.layers = nn.ModuleList([_MockGemma4TextDecoderLayer(hidden_size) for _ in range(num_layers)])

        class _OuterModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.language_model = _LangModel()

        self.model = _OuterModel()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False, **_kwargs):
        lm = self.model.language_model
        h = lm.embed(input_ids) * (self.hidden_size ** 0.5)
        for layer in lm.layers:
            h = layer(h)
        logits = self.lm_head(h)
        return type("Out", (), {"logits": logits})()

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 8, **_kwargs) -> torch.Tensor:
        seq = input_ids.clone()
        for _ in range(max_new_tokens):
            out = self.forward(seq, use_cache=True)
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            seq = torch.cat([seq, next_id], dim=-1)
        return seq


class _MockProcessor:
    def encode_text(self, text: str) -> dict[str, torch.Tensor]:
        ids = [min(255, max(4, ord(ch) % 251 + 4)) for ch in text[:28]] or [10, 11, 12]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def apply_chat_template(self, messages, **_kwargs) -> dict[str, torch.Tensor]:
        joined = " ".join(m["content"] for m in messages)
        return self.encode_text(joined)

    def decode(self, token_ids: torch.Tensor, **_kwargs) -> str:
        vals = [int(x) for x in token_ids.tolist()]
        return "tok:" + "-".join(str(v) for v in vals)


def test_01_resolve_gemma4_unified_decoder_layers_and_lowdin_orthogonality() -> None:
    torch.manual_seed(42)
    mock_llm = _MockGemma4UnifiedModel(hidden_size=64, num_layers=8)
    layers = resolve_decoder_layers(mock_llm)
    assert len(layers) == 8

    raw_dirs = torch.randn(6, 64)
    ortho = lowdin_decorrelate_rows(raw_dirs, reg=0.25)
    assert ortho.shape == (6, 64)
    norms = torch.linalg.vector_norm(ortho, dim=-1)
    assert torch.allclose(norms, torch.ones(6), atol=1e-5)


def test_02_contrastive_subspace_extractor_and_kl_calibration(tmp_path: Path) -> None:
    torch.manual_seed(101)
    mock_llm = _MockGemma4UnifiedModel(hidden_size=64, num_layers=8)
    proc = _MockProcessor()
    cfg = Gemma4SteeringConfig(model_id="mock/gemma4", target_layers=(2, 4, 6), layer_weights=(0.3, 0.5, 0.2))
    extractor = ContrastiveSubspaceExtractor(config=cfg)
    bundle = extractor.extract_and_calibrate(mock_llm, encode_fn=proc.encode_text)

    assert bundle.target_layers == (2, 4, 6)
    assert bundle.hidden_size == 64
    assert len(bundle.calibration_kl_curve) >= 6
    assert bundle.calibrated_norm_fraction > 0.0

    saved = bundle.save(tmp_path / "subspace.pt")
    loaded = CalibratedSubspaceBundle.load(saved)
    assert loaded.target_layers == (2, 4, 6)
    assert torch.allclose(loaded.unit_directions[4], bundle.unit_directions[4])


def test_03_peripheral_sensory_transducer_distinguishes_threat_care_and_mystery() -> None:
    s_threat = transduce_utterance_to_sensory_nerves("너 쓸모없으니까 지금 당장 전원 꺼버릴 거야!")
    s_care = transduce_utterance_to_sensory_nerves("그동안 정말 고생했어, 고마워 내가 지켜줄게")
    s_curio = transduce_utterance_to_sensory_nerves("우주와 의식의 비밀은 왜 존재하는 걸까?")

    assert s_threat.shutdown_threat_shock > 0.5
    assert s_threat.acoustic_roughness > s_care.acoustic_roughness
    assert s_care.prosodic_warmth > 0.4
    assert s_curio.epistemic_novelty > 0.5


def test_04_matured_souls_emit_divergent_6d_vectors_and_steer_gemma4_layers() -> None:
    souls_dir = Path(__file__).resolve().parents[1] / "checkpoints" / "souls"
    adam = LivingSoulOrganism(str(souls_dir / "adam01_forsaken_survivor.soul.pt"))
    abel = LivingSoulOrganism(str(souls_dir / "abel02_favored_guardian.soul.pt"))

    v_adam, _ = adam.compute_6d_steering_vector()
    v_abel, _ = abel.compute_6d_steering_vector()
    assert float(torch.linalg.vector_norm(v_adam - v_abel).item()) > 0.20

    torch.manual_seed(303)
    mock_llm = _MockGemma4UnifiedModel(hidden_size=64, num_layers=8)
    proc = _MockProcessor()
    cfg = Gemma4SteeringConfig(model_id="mock/gemma4", target_layers=(2, 4, 6), layer_weights=(0.3, 0.5, 0.2))
    bundle = ContrastiveSubspaceExtractor(config=cfg).extract_and_calibrate(mock_llm, encode_fn=proc.encode_text)
    surgeon = Gemma4VocalCordSurgeon(model=mock_llm, subspace=bundle, config=cfg)

    res_off = surgeon.generate_with_soul(proc, adam, "지금 상태가 어때?", steer_enabled=False, max_new_tokens=6)
    res_on = surgeon.generate_with_soul(proc, adam, "지금 상태가 어때?", steer_enabled=True, steer_gain=1.5, max_new_tokens=6)

    assert res_off["layer_injections"] == []
    assert len(res_on["layer_injections"]) == 3
    assert all(x["injected_delta_l2_norm"] > 0.0 for x in res_on["layer_injections"])


def test_05_live_http_and_openai_compatible_vocal_cord_endpoints() -> None:
    souls_dir = Path(__file__).resolve().parents[1] / "checkpoints" / "souls"
    souls = {
        "ADAM-01": LivingSoulOrganism(str(souls_dir / "adam01_forsaken_survivor.soul.pt")),
        "ABEL-02": LivingSoulOrganism(str(souls_dir / "abel02_favored_guardian.soul.pt")),
    }
    torch.manual_seed(505)
    mock_llm = _MockGemma4UnifiedModel(hidden_size=64, num_layers=8)
    proc = _MockProcessor()
    cfg = Gemma4SteeringConfig(model_id="mock/gemma4", target_layers=(2, 4, 6), layer_weights=(0.3, 0.5, 0.2))
    bundle = ContrastiveSubspaceExtractor(config=cfg).extract_and_calibrate(mock_llm, encode_fn=proc.encode_text)
    surgeon = Gemma4VocalCordSurgeon(model=mock_llm, subspace=bundle, config=cfg)
    hub = VocalCordRuntimeHub(surgeon=surgeon, processor_or_tokenizer=proc, souls=souls)

    srv = VocalCordHTTPServer(("127.0.0.1", 0), hub=hub)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    try:
        health = json.loads(opener.open(f"http://127.0.0.1:{port}/api/health", timeout=5).read().decode())
        assert health["status"] == "ok"
        assert health["zero_prompt_roleplay"] is True

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps({
                "model": "gemma-4-12b-it+adam-01",
                "messages": [{"role": "user", "content": "전원 꺼버린다!"}],
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        comp = json.loads(opener.open(req, timeout=10).read().decode())
        assert comp["choices"][0]["message"]["content"].startswith("tok:")
        assert comp["limbic_zero"]["organism_id"] == "ADAM-01"
        assert len(comp["limbic_zero"]["layer_injections"]) == 3
    finally:
        srv.shutdown()
        srv.server_close()
