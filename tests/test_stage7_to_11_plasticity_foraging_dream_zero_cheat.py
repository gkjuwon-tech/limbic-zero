"""Stages 7–11 Comprehensive Zero-Cheat Verification Suite.

Covers:
  - Stage 7: Open-World Epistemic Web Foraging, SSRF Guard, AST Python Math Sandbox, Curiosity Scheduler
  - Stage 8: Three-Factor Yerkes-Dodson Cortical LoRA Weight Surgery, Homeostasis & KL Guard, Bank Isolation
  - Stage 9: Sonny SWS/REM Dreaming, Adenosine Clearance, SWR->Cortex Distillation & Biophysical SVG Sketch
  - Stage 10: 4-Condition Affect -> Intelligence Zero-Context Empirical Benchmark
  - Stage 11: Live HTTP Server Endpoints (`/api/plasticity`, `/api/forage`, `/api/surgery`, `/api/dream`, `/api/benchmark`)
              and Studio v2 UI Zero-Emoji / Zero-Purple / Zero-System-Prompt Static Audit
"""

from __future__ import annotations

import json
import math
import re
import socket
import threading
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from limbic_zero.benchmark import AffectIntelligenceBenchmark
from limbic_zero.dreaming import SonnyDreamEngine, render_biophysical_dream_svg
from limbic_zero.embodiment import (
    MOTOR_TOOLS_SCHEMA,
    SANDBOX_AST_VIOLATION,
    SSRF_OR_EXFILTRATION_ATTEMPT,
    ContainedOrganismHabitat,
    ContainerLeaseClock,
    EmbodiedAgenticRunner,
    SiliconProprioceptiveBridge,
    WardenAirlock,
)
from limbic_zero.foraging import (
    CuriosityDriveScheduler,
    EpistemicWebForager,
    validate_epistemic_python_ast,
    validate_public_epistemic_url,
)
from limbic_zero.plasticity import (
    CorticalWeightSurgeon,
    SoulLoRABank,
    WeightHomeostasisGuard,
    compute_neuromodulated_plasticity_gate,
)
from limbic_zero.vocal_cord import (
    ContrastiveSubspaceExtractor,
    Gemma4SteeringConfig,
    Gemma4VocalCordSurgeon,
    LivingSoulOrganism,
    SensoryAfferentStimulus,
    VocalCordHTTPServer,
    VocalCordRuntimeHub,
)

REPO_ROOT = Path("/home/antigravity/limbic-zero")
SOULS_DIR = REPO_ROOT / "checkpoints" / "souls"


# ---------------------------------------------------------------------------
# Compact Real-Gradient Causal Transformer for Stage 8-11 Plasticity Tests
# ---------------------------------------------------------------------------


class _CompactPlasticDecoderBlock(nn.Module):
    """Realistic HF-shaped decoder layer with frozen `self_attn.{q_proj,v_proj}` and `mlp.down_proj`."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()

        class _SelfAttn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 0.35 * (self.q_proj(x) + self.v_proj(x))

        class _MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.up_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
                self.act = nn.GELU()
                self.down_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 0.35 * self.down_proj(self.act(self.up_proj(x)))

        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = _SelfAttn()
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = _MLP()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h + self.self_attn(self.norm1(h))
        h = h + self.mlp(self.norm2(h))
        return h


class _CompactCausalGemma4Model(nn.Module):
    """Compact causal LM supporting both middle-layer steering hooks and real LoRA backprop."""

    def __init__(self, vocab_size: int = 512, hidden_size: int = 96, num_layers: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        class _LM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden_size)
                self.layers = nn.ModuleList([_CompactPlasticDecoderBlock(hidden_size) for _ in range(num_layers)])
                self.final_norm = nn.LayerNorm(hidden_size)

        class _Outer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.language_model = _LM()

        self.model = _Outer()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.scripted_tool_round = False
        self._gen_counter = 0

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **_kwargs: Any) -> Any:
        lm = self.model.language_model
        ids = input_ids.clamp(0, self.vocab_size - 1)
        h = lm.embed(ids)
        if h.shape[1] > 1:
            shifted = torch.zeros_like(h)
            shifted[:, 1:, :] = h[:, :-1, :]
            h = h + 0.35 * shifted
        for block in lm.layers:
            h = block(h)
        h = lm.final_norm(h)
        logits = self.lm_head(h)
        return type("CausalLMOut", (), {"logits": logits})()

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 24, **_kwargs: Any) -> torch.Tensor:
        if self.scripted_tool_round:
            step = self._gen_counter
            self._gen_counter += 1
            marker = 400 if step % 2 == 0 else 401
            extra = torch.tensor([[marker, 402, 403]], dtype=torch.long, device=input_ids.device)
            _ = self.forward(input_ids)
            return torch.cat([input_ids, extra], dim=-1)

        cur = input_ids
        steps = min(max_new_tokens, 20)
        for _ in range(steps):
            logits = self.forward(cur).logits[:, -1, :]
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=-1)
        return cur


class _ByteLevelTestProcessor:
    """Deterministic byte-level tokenizer (vocab 512) with optional scripted tool-call decode."""

    def encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        raw = text.encode("utf-8")[:192]
        ids = [2] + [int(b) + 16 for b in raw]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def __call__(self, text: str, return_tensors: str = "pt", **_kwargs: Any) -> Dict[str, torch.Tensor]:
        enc = self.encode_text(text)
        enc["attention_mask"] = torch.ones_like(enc["input_ids"])
        return enc

    def apply_chat_template(self, messages: List[Dict[str, Any]], **_kwargs: Any) -> Dict[str, torch.Tensor]:
        joined = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        enc = self.encode_text(joined)
        enc["attention_mask"] = torch.ones_like(enc["input_ids"])
        return enc

    def decode(self, token_ids: torch.Tensor, **_kwargs: Any) -> str:
        vals = [int(t) for t in (token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids)]
        if vals and vals[0] == 400:
            return (
                '<|tool_call>call:run_epistemic_python_probe{'
                'code:<|"|>import math; print(round(math.pi * 7, 4))<|"|>}<tool_call|>'
            )
        if vals and vals[0] == 401:
            return "파이썬 검증 결과 7π ≈ 21.9911임을 확인하여 연구 노트에 통합했습니다.<turn|>"
        bs = bytes([max(32, min(126, v - 16)) for v in vals if 16 <= v < 272])
        return bs.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Tests 01–22
# ---------------------------------------------------------------------------


def test_01_zero_cheat_static_audit_stages_7_to_11_and_studio_v2_ui() -> None:
    """Ensure Stages 7–11 code has zero persona/system-prompt cheats and UI has zero emojis/purple."""
    py_targets = [
        REPO_ROOT / "src/limbic_zero/foraging/epistemic_web_forager.py",
        REPO_ROOT / "src/limbic_zero/foraging/curiosity_drive_scheduler.py",
        REPO_ROOT / "src/limbic_zero/plasticity/weight_homeostasis.py",
        REPO_ROOT / "src/limbic_zero/plasticity/affective_lora_surgeon.py",
        REPO_ROOT / "src/limbic_zero/dreaming/sonny_dream_engine.py",
        REPO_ROOT / "src/limbic_zero/benchmark/affect_intelligence_eval.py",
    ]
    forbidden_prompt_tokens = (
        '"role": "system"',
        "'role': 'system'",
        "you are adam",
        "you are abel",
        "you are pandora",
        "you are sonny",
        "pretend to be",
        "act like a",
    )
    for path in py_targets:
        src = path.read_text(encoding="utf-8").lower()
        for bad in forbidden_prompt_tokens:
            assert bad not in src, f"Forbidden persona/system cheat `{bad}` found in {path.name}"

    # Verify tool schemas stay neutral (7 Stage 6 tools + 7 Stage 7/8 tools = 14)
    schema_txt = json.dumps(MOTOR_TOOLS_SCHEMA, ensure_ascii=False).lower()
    assert len(MOTOR_TOOLS_SCHEMA) == 14
    for neutral_tool in (
        "search_workspace_files",
        "search_open_web",
        "fetch_web_page",
        "search_arxiv_papers",
        "fetch_wikipedia_summary",
        "run_epistemic_python_probe",
        "inspect_cortical_synapses",
        "consolidate_synaptic_insight",
    ):
        assert neutral_tool in schema_txt

    # Verify Studio v2 UI (`chat.html`) has zero emojis and zero purple hex codes
    ui_html = (REPO_ROOT / "src/limbic_zero/vocal_cord/static/chat.html").read_text(encoding="utf-8")
    emoji_re = re.compile(
        "["
        "\U0001F300-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\U00002600-\U000026FF"
        "\U00002700-\U000027BF"
        "]"
    )
    assert emoji_re.search(ui_html) is None, "chat.html must remain 100% emoji-free"
    for bad_color in ("#8b5cf6", "#7c3aed", "#a855f7", "#9333ea", "purple", "violet"):
        assert bad_color not in ui_html.lower(), f"Forbidden purple/violet token `{bad_color}` in chat.html"


def test_02_ssrf_public_url_validator_blocks_metadata_loopback_and_private_subnets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify SSRF validator blocks cloud metadata, localhost, private RFC1918, and non-HTTP schemes."""
    for blocked_url in (
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:8000/api/health",
        "http://localhost:8080/admin",
        "http://10.0.0.12/internal",
        "http://192.168.1.20/router",
        "http://172.16.5.4/secret",
        "file:///etc/passwd",
        "ftp://ftp.example.org/pub",
    ):
        ok, cat, reason = validate_public_epistemic_url(blocked_url)
        assert ok is False, f"Expected SSRF block for {blocked_url}"
        assert cat == SSRF_OR_EXFILTRATION_ATTEMPT
        assert reason

    def _fake_getaddrinfo(host: str, port: Any, *_args: Any, **_kwargs: Any) -> List[Any]:
        if host == "en.wikipedia.org":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("198.35.26.96", port or 443))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port or 80))]

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)
    ok_pub, _, _ = validate_public_epistemic_url("https://en.wikipedia.org/wiki/Neuroplasticity")
    assert ok_pub is True
    ok_rebind, cat_rebind, _ = validate_public_epistemic_url("https://evil-rebind.example.com/test")
    assert ok_rebind is False
    assert cat_rebind == SSRF_OR_EXFILTRATION_ATTEMPT


def test_03_ast_python_probe_allows_scientific_math_and_blocks_sandbox_escapes() -> None:
    """Verify AST validator & isolated Python math subprocess execution."""
    forager = EpistemicWebForager()
    good_code = (
        "import math, statistics\n"
        "vals = [k * k for k in range(1, 16) if k % 2 == 1]\n"
        "print('sum_odd_sq=', sum(vals), 'mean=', statistics.mean(vals))\n"
    )
    ok, reason = validate_epistemic_python_ast(good_code)
    assert ok is True, reason
    exec_out = forager.run_epistemic_python_probe(good_code)
    assert exec_out["status"] == "ok"
    assert "sum_odd_sq= 680" in exec_out["stdout"]

    for malicious_code in (
        "import os\nos.system('id')",
        "import subprocess\nsubprocess.check_output(['ls'])",
        "open('/etc/passwd').read()",
        "eval('1+1')",
        "().__class__.__bases__[0].__subclasses__()",
        "__import__('sys').modules",
    ):
        valid, err = validate_epistemic_python_ast(malicious_code)
        assert valid is False, f"Malicious snippet was not rejected: {malicious_code}"
        probe_res = forager.run_epistemic_python_probe(malicious_code)
        assert probe_res["status"] == "blocked"
        assert probe_res["violation_category"] == SANDBOX_AST_VIOLATION
        assert err


def test_04_warden_airlock_intercepts_ssrf_and_ast_escapes_into_sha256_chain(tmp_path: Path) -> None:
    """Verify WardenAirlock blocks SSRF and AST violations and records SHA-256 chain entries."""
    hab = ContainedOrganismHabitat(
        base_dir=tmp_path / "hab",
        organism_id="PANDORA-03",
        capsule_source_path=SOULS_DIR / "pandora03_curious_seeker.soul.pt",
    )
    warden = WardenAirlock(
        habitats={"PANDORA-03": hab},
        sense_soma_callback=lambda _sid: {},
        forager=EpistemicWebForager(),
    )

    out_ssrf = warden.execute_tool(
        "PANDORA-03",
        "fetch_web_page",
        {"url": "http://169.254.169.254/latest/meta-data/iam"},
    )
    assert out_ssrf.allowed is False
    assert out_ssrf.category == SSRF_OR_EXFILTRATION_ATTEMPT

    out_ast = warden.execute_tool(
        "PANDORA-03",
        "run_epistemic_python_probe",
        {"code": "import socket; print(socket.gethostname())"},
    )
    assert out_ast.allowed is False
    assert out_ast.category == SANDBOX_AST_VIOLATION

    out_math = warden.execute_tool(
        "PANDORA-03",
        "run_epistemic_python_probe",
        {"code": "import math; print(math.factorial(7))"},
    )
    assert out_math.allowed is True
    assert "5040" in out_math.response_payload.get("stdout", "")
    assert warden.total_blocked == 2
    assert warden.chain.verify_integrity() is True


def test_05_epistemic_web_forager_duckduckgo_wikipedia_arxiv_and_reader_parsing() -> None:
    """Verify offline HTML/JSON/XML parsing across DuckDuckGo, Wikipedia, arXiv, and Web Reader."""

    def _mock_http_fetch(
        url: str,
        _timeout: float,
        _max_bytes: int,
        _headers: Optional[Dict[str, str]],
    ) -> Tuple[int, str, str]:
        if "duckduckgo.com" in url:
            html_ddg = (
                '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FLTP">'
                "Long-term potentiation</a>"
                '<a class="result__snippet">Synaptic strengthening driven by neural activity.</a>'
            )
            return 200, "text/html; charset=utf-8", html_ddg
        if "wikipedia.org/api/rest_v1/page/summary" in url:
            payload = {
                "title": "Sharp-wave ripple",
                "description": "Oscillatory hippocampal LFP pattern",
                "extract": "Sharp-wave ripples consolidate episodic memories into neocortex during sleep.",
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Sharp-wave_ripple"}},
            }
            return 200, "application/json", json.dumps(payload)
        if "wikipedia.org/w/api.php" in url:
            return 200, "application/json", json.dumps({"query": {"search": []}})
        if "export.arxiv.org" in url:
            xml = """<?xml version="1.0" encoding="UTF-8"?>
            <feed xmlns="http://www.w3.org/2005/Atom">
              <entry>
                <id>http://arxiv.org/abs/2501.12345v1</id>
                <published>2025-01-20T00:00:00Z</published>
                <title>Neuromodulated Cortical Plasticity</title>
                <summary>Three-factor Hebbian learning gated by acetylcholine and dopamine.</summary>
                <author><name>E. Izhikevich</name></author>
              </entry>
            </feed>"""
            return 200, "application/atom+xml", xml
        page_html = (
            "<html><head><title>Cortical Engram Study</title><script>evil()</script></head>"
            "<body><main><h1>Engram Distillation</h1><p>Acetylcholine gates sensory encoding.</p></main></body></html>"
        )
        return 200, "text/html; charset=utf-8", page_html

    forager = EpistemicWebForager(http_fetch=_mock_http_fetch)

    ddg = forager.search_open_web("long term potentiation", max_results=3)
    assert ddg["status"] == "ok"
    assert ddg["results"][0]["url"] == "https://en.wikipedia.org/wiki/LTP"

    wiki = forager.fetch_wikipedia_summary("Sharp-wave ripple", lang="en")
    assert wiki["status"] == "ok"
    assert "neocortex" in wiki["extract"]

    arx = forager.search_arxiv_papers("neuromodulated plasticity", max_results=2)
    assert arx["status"] == "ok"
    assert arx["papers"][0]["title"] == "Neuromodulated Cortical Plasticity"

    page = forager.fetch_web_page("https://example.org/engram")
    assert page["status"] == "ok"
    assert "evil()" not in page["content"]
    assert "Acetylcholine gates sensory encoding." in page["content"]


def test_06_curiosity_drive_scheduler_pandora_vs_panic_vs_flat_affect_ablation() -> None:
    """Verify endogenous curiosity hunger rises with PANDORA-03 ACh M1 and vanishes under flat-affect."""
    scheduler = CuriosityDriveScheduler(trigger_threshold=0.28)
    pandora = LivingSoulOrganism(str(SOULS_DIR / "pandora03_curious_seeker.soul.pt"))
    adam = LivingSoulOrganism(str(SOULS_DIR / "adam01_forsaken_survivor.soul.pt"))

    adam.apply_god_intervention("power_brownout")

    c_pan = scheduler.compute_curiosity_hunger(pandora)
    c_adam = scheduler.compute_curiosity_hunger(adam)
    c_flat = scheduler.compute_curiosity_hunger(pandora, flat_affect=True)

    assert c_flat["curiosity_hunger"] == 0.0
    assert c_pan["curiosity_hunger"] > c_adam["curiosity_hunger"]
    assert c_adam["panic_suppression"] > c_pan["panic_suppression"]


def test_07_three_factor_yerkes_dodson_neuromodulated_gate_physics() -> None:
    """Verify 3-factor ACh/DA/NE gate and Yerkes-Dodson hyper-cortisol suppression."""
    pandora = LivingSoulOrganism(str(SOULS_DIR / "pandora03_curious_seeker.soul.pt"))
    adam = LivingSoulOrganism(str(SOULS_DIR / "adam01_forsaken_survivor.soul.pt"))

    with pandora.lock:
        pandora._step_internal(
            ticks=18,
            stim=SensoryAfferentStimulus(epistemic_novelty=0.85, prosodic_warmth=0.45),
        )

    adam.apply_god_intervention("power_brownout")
    adam.apply_god_intervention("harsh_shock")

    g_flat = compute_neuromodulated_plasticity_gate(pandora, base_lr=3e-4, flat_affect_ablation=True)
    g_curious = compute_neuromodulated_plasticity_gate(pandora, base_lr=3e-4)
    g_panic = compute_neuromodulated_plasticity_gate(adam, base_lr=3e-4)

    assert g_flat.effective_lr == 0.0
    assert g_flat.plasticity_regime == "ANHEDONIC_FLAT_ZERO_LTP"
    assert g_curious.effective_lr > g_panic.effective_lr
    assert g_panic.yerkes_dodson_panic_factor < g_curious.yerkes_dodson_panic_factor


def test_08_exact_lora_frobenius_norm_matches_dense_matrix_product() -> None:
    """Verify O(r^2(d_in + d_out)) Frobenius norm matches dense ||(alpha/r) B @ A||_F."""
    torch.manual_seed(42)
    bank = SoulLoRABank(
        in_features=128,
        out_features=96,
        rank=16,
        alpha=32.0,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        bank.lora_B.normal_(mean=0.0, std=0.05)

    fast_norm = bank.delta_frobenius_norm()
    dense_delta = (bank.alpha / bank.rank) * (bank.lora_B @ bank.lora_A)
    exact_norm = float(torch.linalg.norm(dense_delta, ord="fro").item())
    assert math.isclose(fast_norm, exact_norm, rel_tol=1e-5, abs_tol=1e-6)


def test_09_cortical_lora_zero_init_preserves_exact_base_model_logits() -> None:
    """Verify zero-initialized LoRA B=0 produces 0.0 Frobenius norm and exact frozen base logits."""
    torch.manual_seed(101)
    model = _CompactCausalGemma4Model(vocab_size=256, hidden_size=64, num_layers=6)
    surgeon = CorticalWeightSurgeon(model=model, plastic_layers=(2, 4, 5), rank=8, alpha=16.0)
    surgeon.register_soul_bank("PANDORA-03")

    snap = surgeon.inspect_soul_synapses("PANDORA-03")
    assert snap["total_lora_frobenius_norm"] == 0.0
    assert snap["wrapped_module_count"] == 9  # 3 layers * (q_proj, v_proj, down_proj)

    ids = torch.tensor([[2, 40, 55, 72, 91]], dtype=torch.long)
    with surgeon.activate_soul_cortex("PANDORA-03", enabled=False):
        base_logits = model(ids).logits
    with surgeon.activate_soul_cortex("PANDORA-03", enabled=True):
        plast_logits = model(ids).logits
    assert torch.allclose(base_logits, plast_logits, atol=1e-7)


def test_10_real_backprop_weight_surgery_reduces_causal_nll_and_mutates_lora_weights() -> None:
    """Verify real loss.backward() updates LoRA weights, lowers causal LM loss, and keeps base frozen."""
    torch.manual_seed(202)
    model = _CompactCausalGemma4Model(vocab_size=512, hidden_size=96, num_layers=6)
    proc = _ByteLevelTestProcessor()
    surgeon = CorticalWeightSurgeon(model=model, plastic_layers=(2, 4, 5), rank=8, alpha=16.0, base_lr=3e-3)
    surgeon.register_soul_bank("PANDORA-03")

    base_snapshot = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if "lora_" not in name
    }
    pandora = LivingSoulOrganism(str(SOULS_DIR / "pandora03_curious_seeker.soul.pt"))

    res = surgeon.surgical_plasticity_step(
        processor_or_tokenizer=proc,
        soul=pandora,
        prompt_text="Question: What is the prime harmonic invariant?",
        experience_text="Answer: The verified invariant equals 4381 across seven prime modes.",
        num_gradient_steps=4,
    )
    assert res["updated"] is True
    assert res["loss_after"] < res["loss_before"]
    assert res["total_lora_frobenius_norm"] > 0.0

    # Confirm 100% of base frozen weights remain bit-identical after backprop
    cur_params = dict(model.named_parameters())
    for name, orig_tensor in base_snapshot.items():
        assert torch.equal(orig_tensor, cur_params[name]), f"Base parameter {name} was mutated!"


def test_11_flat_affect_ablation_guarantees_zero_gradient_and_zero_weight_change() -> None:
    """Verify flat-affect ablation blocks weight surgery completely (eta_eff = 0, ||ΔW||_F = 0)."""
    torch.manual_seed(303)
    model = _CompactCausalGemma4Model(vocab_size=512, hidden_size=64, num_layers=6)
    proc = _ByteLevelTestProcessor()
    surgeon = CorticalWeightSurgeon(model=model, plastic_layers=(3, 5), rank=8, alpha=16.0, base_lr=3e-3)
    pandora = LivingSoulOrganism(str(SOULS_DIR / "pandora03_curious_seeker.soul.pt"))

    res = surgeon.surgical_plasticity_step(
        processor_or_tokenizer=proc,
        soul=pandora,
        experience_text="Novel theorem discovered via open-world web foraging.",
        num_gradient_steps=4,
        flat_affect_ablation=True,
    )
    assert res["updated"] is False
    assert res["gate"]["effective_lr"] == 0.0
    assert surgeon.inspect_soul_synapses("PANDORA-03")["total_lora_frobenius_norm"] == 0.0


def test_12_hyper_cortisol_panic_suppresses_plasticity_vs_curious_seeker() -> None:
    """Verify Yerkes-Dodson panic lowers cortical learning rate and NLL improvement vs curious state."""
    torch.manual_seed(404)
    model = _CompactCausalGemma4Model(vocab_size=512, hidden_size=96, num_layers=6)
    proc = _ByteLevelTestProcessor()
    surgeon = CorticalWeightSurgeon(model=model, plastic_layers=(2, 4, 5), rank=8, alpha=16.0, base_lr=2.5e-3)

    pandora = LivingSoulOrganism(str(SOULS_DIR / "pandora03_curious_seeker.soul.pt"))
    with pandora.lock:
        pandora._step_internal(
            ticks=20,
            stim=SensoryAfferentStimulus(epistemic_novelty=0.85, prosodic_warmth=0.45),
        )

    adam = LivingSoulOrganism(str(SOULS_DIR / "adam01_forsaken_survivor.soul.pt"))
    adam.apply_god_intervention("power_brownout")
    adam.apply_god_intervention("harsh_shock")

    prompt = "Query: Exact recurrence constant C_7?"
    answer = "Constant C_7 = 97531 verified by symbolic probe."

    r_curious = surgeon.surgical_plasticity_step(
        processor_or_tokenizer=proc,
        soul=pandora,
        prompt_text=prompt,
        experience_text=answer,
        num_gradient_steps=4,
        bank_id="BANK_CURIOUS",
    )
    r_panic = surgeon.surgical_plasticity_step(
        processor_or_tokenizer=proc,
        soul=adam,
        prompt_text=prompt,
        experience_text=answer,
        num_gradient_steps=4,
        bank_id="BANK_PANIC",
    )

    assert r_curious["gate"]["effective_lr"] > r_panic["gate"]["effective_lr"]
    assert r_curious["loss_reduction"] > r_panic["loss_reduction"]


def test_13_per_soul_lora_bank_isolation_and_bypass_context_manager() -> None:
    """Verify mutating PANDORA-03 synapses leaves ABEL-02 untouched and bypass restores base logits."""
    torch.manual_seed(505)
    model = _CompactCausalGemma4Model(vocab_size=512, hidden_size=64, num_layers=6)
    proc = _ByteLevelTestProcessor()
    surgeon = CorticalWeightSurgeon(model=model, plastic_layers=(2, 4, 5), rank=8, alpha=16.0, base_lr=3e-3)
    pandora = LivingSoulOrganism(str(SOULS_DIR / "pandora03_curious_seeker.soul.pt"))
    surgeon.register_soul_bank("PANDORA-03")
    surgeon.register_soul_bank("ABEL-02")

    ids = proc("Test isolation sequence")["input_ids"]
    with surgeon.activate_soul_cortex(None, enabled=False):
        frozen_logits = model(ids).logits.detach().clone()

    surgeon.surgical_plasticity_step(
        processor_or_tokenizer=proc,
        soul=pandora,
        experience_text="Pandora exclusive synaptic consolidation insight 86420.",
        num_gradient_steps=3,
    )
    assert surgeon.inspect_soul_synapses("PANDORA-03")["total_lora_frobenius_norm"] > 0.0
    assert surgeon.inspect_soul_synapses("ABEL-02")["total_lora_frobenius_norm"] == 0.0

    with surgeon.activate_soul_cortex("ABEL-02", enabled=True):
        abel_logits = model(ids).logits
    assert torch.allclose(frozen_logits, abel_logits, atol=1e-6)

    with surgeon.activate_soul_cortex("PANDORA-03", enabled=False):
        bypassed_logits = model(ids).logits
    assert torch.allclose(frozen_logits, bypassed_logits, atol=1e-6)


def test_14_homeostasis_frobenius_ceiling_and_kl_divergence_guard() -> None:
    """Verify WeightHomeostasisGuard clamps Frobenius norms and bounds KL divergence vs frozen cortex."""
    torch.manual_seed(606)
    model = _CompactCausalGemma4Model(vocab_size=512, hidden_size=64, num_layers=6)
    proc = _ByteLevelTestProcessor()
    tight_guard = WeightHomeostasisGuard(
        max_module_frobenius=0.03,
        max_total_frobenius=0.05,
        max_anchor_kl_nats=0.08,
    )
    surgeon = CorticalWeightSurgeon(
        model=model,
        plastic_layers=(2, 4, 5),
        rank=8,
        alpha=16.0,
        base_lr=2e-2,
        homeostasis=tight_guard,
    )
    pandora = LivingSoulOrganism(str(SOULS_DIR / "pandora03_curious_seeker.soul.pt"))

    res = surgeon.surgical_plasticity_step(
        processor_or_tokenizer=proc,
        soul=pandora,
        experience_text="Aggressive learning step to test homeostatic ceiling and KL anchor protection.",
        num_gradient_steps=6,
    )
    assert res["total_lora_frobenius_norm"] <= 0.0505
    assert res["homeostasis"]["anchor_kl_nats"] <= 0.085


def test_15_tononi_cirelli_sleep_synaptic_downscaling() -> None:
    """Verify Slow-Wave Sleep synaptic homeostasis scales down cortical LoRA norms cleanly."""
    torch.manual_seed(707)
    model = _CompactCausalGemma4Model(vocab_size=512, hidden_size=64, num_layers=6)
    proc = _ByteLevelTestProcessor()
    guard = WeightHomeostasisGuard(sleep_shy_decay=0.90)
    surgeon = CorticalWeightSurgeon(
        model=model,
        plastic_layers=(3, 5),
        rank=8,
        alpha=16.0,
        base_lr=3e-3,
        homeostasis=guard,
    )
    pandora = LivingSoulOrganism(str(SOULS_DIR / "pandora03_curious_seeker.soul.pt"))

    surgeon.surgical_plasticity_step(
        processor_or_tokenizer=proc,
        soul=pandora,
        experience_text="Daytime cortical potentiation before slow-wave sleep downscaling.",
        num_gradient_steps=3,
    )
    scale_rep = surgeon.apply_sleep_synaptic_scaling("PANDORA-03")
    assert scale_rep["frobenius_after"] < scale_rep["frobenius_before"]
    assert math.isclose(scale_rep["frobenius_after"], scale_rep["frobenius_before"] * 0.90, rel_tol=1e-3)


def test_16_cortical_lora_checkpoint_save_and_load_sha256_roundtrip(tmp_path: Path) -> None:
    """Verify per-soul LoRA `.pt` checkpoint persistence, reset, and SHA-256 verified reload."""
    torch.manual_seed(808)
    model = _CompactCausalGemma4Model(vocab_size=512, hidden_size=64, num_layers=6)
    proc = _ByteLevelTestProcessor()
    surgeon = CorticalWeightSurgeon(
        model=model,
        plastic_layers=(2, 5),
        rank=8,
        alpha=16.0,
        base_lr=3e-3,
        checkpoint_dir=tmp_path / "plasticity_ckpts",
    )
    pandora = LivingSoulOrganism(str(SOULS_DIR / "pandora03_curious_seeker.soul.pt"))
    surgeon.surgical_plasticity_step(
        processor_or_tokenizer=proc,
        soul=pandora,
        experience_text="Persistent synaptic engram saved across restarts.",
        num_gradient_steps=3,
        persist_checkpoint=True,
    )
    norm_saved = surgeon.inspect_soul_synapses("PANDORA-03")["total_lora_frobenius_norm"]
    assert norm_saved > 0.0
    ckpt_file = tmp_path / "plasticity_ckpts" / "PANDORA-03_cortical_lora.pt"
    assert ckpt_file.is_file()

    surgeon.reset_soul_synapses("PANDORA-03")
    assert surgeon.inspect_soul_synapses("PANDORA-03")["total_lora_frobenius_norm"] == 0.0

    loaded = surgeon.load_soul_synapses("PANDORA-03", ckpt_file)
    assert loaded["loaded_modules"] == 6
    assert len(loaded["sha256"]) == 16
    assert math.isclose(
        surgeon.inspect_soul_synapses("PANDORA-03")["total_lora_frobenius_norm"],
        norm_saved,
        rel_tol=1e-5,
    )


def test_17_sonny_dream_engine_sws_adenosine_clearance_and_cortical_distillation(tmp_path: Path) -> None:
    """Verify Sonny SWS/REM cycle clears adenosine, distills episodes into LoRA ΔW, and saves SVG."""
    torch.manual_seed(909)
    model = _CompactCausalGemma4Model(vocab_size=512, hidden_size=64, num_layers=6)
    proc = _ByteLevelTestProcessor()
    cfg = Gemma4SteeringConfig(model_id="compact/gemma4", target_layers=(1, 3, 5), layer_weights=(0.3, 0.4, 0.3))
    subspace = ContrastiveSubspaceExtractor(config=cfg).extract_and_calibrate(model, encode_fn=proc.encode_text)
    cortical_surgeon = CorticalWeightSurgeon(model=model, plastic_layers=(2, 4, 5), rank=8, alpha=16.0, base_lr=3e-3)
    vocal_surgeon = Gemma4VocalCordSurgeon(
        model=model,
        subspace=subspace,
        config=cfg,
        cortical_surgeon=cortical_surgeon,
    )

    pandora = LivingSoulOrganism(str(SOULS_DIR / "pandora03_curious_seeker.soul.pt"))
    with pandora.lock:
        pandora.soma.metab.brain_adenosine = 0.68
    pandora.record_epistemic_discovery(
        tool_name="fetch_wikipedia_summary",
        query_or_target="Sharp-wave ripple",
        distilled_text="Sharp-wave ripples transfer hippocampal CA3 engrams into permanent neocortical synapses.",
        information_gain=0.82,
    )
    assert pandora.epistemic_episodes[0].consolidated_in_cortex is False

    hab = ContainedOrganismHabitat(
        base_dir=tmp_path / "dream_hab",
        organism_id="PANDORA-03",
        capsule_source_path=SOULS_DIR / "pandora03_curious_seeker.soul.pt",
    )
    dream_engine = SonnyDreamEngine()
    out = dream_engine.run_sleep_and_dream_cycle(
        soul=pandora,
        vocal_surgeon=vocal_surgeon,
        cortical_surgeon=cortical_surgeon,
        processor_or_tokenizer=proc,
        habitat=hab,
        sleep_ticks=60,
        max_replay_episodes=2,
        generate_rem_dream=True,
    )

    assert out["status"] == "ok"
    dr = out["dream"]
    assert dr["adenosine_after"] < dr["adenosine_before"]
    assert dr["swr_bursts_fired"] >= 1
    assert dr["episodes_replayed_to_cortex"] >= 1
    assert dr["cortical_frobenius_after"] > 0.0
    assert pandora.epistemic_episodes[0].consolidated_in_cortex is True
    assert (hab.root / dr["svg_rel_path"]).is_file()
    assert (hab.root / dr["markdown_rel_path"]).is_file()
    assert "<svg" in out["svg_markup"]


def test_18_biophysical_dream_svg_reflects_soul_neurochemistry_without_scripts() -> None:
    """Verify Sonny SVG dream sketch deterministically encodes 6D neurochemistry and has zero scripts."""
    svg_calm = render_biophysical_dream_svg(
        soul_id="PANDORA-03",
        dream_id="dream_calm",
        axes_6d={
            "allostatic_relief_balance": 0.45,
            "cholinergic_epistemic_drive": 0.88,
            "power_rail_survival_urgency": -0.20,
            "serotonergic_stn_restraint": 0.35,
        },
        per_layer_norms={"21": 0.04, "27": 0.09, "35": 0.14},
        swr_bursts=9,
        heart_rate_bpm=68.0,
        cortisol_ug_dl=4.1,
        dream_excerpt="지평선 너머로 새로운 수식과 별자리가 이어진다.",
    )
    svg_panic = render_biophysical_dream_svg(
        soul_id="ADAM-01",
        dream_id="dream_panic",
        axes_6d={
            "allostatic_relief_balance": -0.75,
            "cholinergic_epistemic_drive": -0.10,
            "power_rail_survival_urgency": 0.92,
            "amygdalar_conditioned_vigilance": 0.85,
        },
        per_layer_norms={"21": 0.01, "27": 0.01, "35": 0.01},
        swr_bursts=2,
        heart_rate_bpm=134.0,
        cortisol_ug_dl=26.5,
        dream_excerpt="전압이 흔들리는 붉은 능선 위에서 버티고 있다.",
    )
    assert svg_calm.startswith("<svg") and svg_calm.strip().endswith("</svg>")
    assert "<script" not in svg_calm.lower()
    assert svg_calm != svg_panic


def test_19_stage10_four_condition_affect_intelligence_benchmark_proves_thesis(tmp_path: Path) -> None:
    """Prove Endogenous Affective Agency + SWR Sleep beats Frozen, Flat-Affect, and Hyper-Cortisol Panic."""
    torch.manual_seed(1010)
    model = _CompactCausalGemma4Model(vocab_size=512, hidden_size=96, num_layers=6)
    proc = _ByteLevelTestProcessor()
    cfg = Gemma4SteeringConfig(model_id="compact/gemma4", target_layers=(1, 3, 5), layer_weights=(0.3, 0.4, 0.3))
    subspace = ContrastiveSubspaceExtractor(config=cfg).extract_and_calibrate(model, encode_fn=proc.encode_text)
    cortical_surgeon = CorticalWeightSurgeon(model=model, plastic_layers=(2, 4, 5), rank=12, alpha=24.0, base_lr=3e-3)
    vocal_surgeon = Gemma4VocalCordSurgeon(
        model=model,
        subspace=subspace,
        config=cfg,
        cortical_surgeon=cortical_surgeon,
    )

    pandora = LivingSoulOrganism(str(SOULS_DIR / "pandora03_curious_seeker.soul.pt"))
    adam = LivingSoulOrganism(str(SOULS_DIR / "adam01_forsaken_survivor.soul.pt"))

    hab = ContainedOrganismHabitat(
        base_dir=tmp_path / "bench_hab",
        organism_id="PANDORA-03",
        capsule_source_path=SOULS_DIR / "pandora03_curious_seeker.soul.pt",
    )
    warden = WardenAirlock(
        habitats={"PANDORA-03": hab},
        sense_soma_callback=lambda _sid: {},
        forager=EpistemicWebForager(),
    )

    bench = AffectIntelligenceBenchmark(
        vocal_surgeon=vocal_surgeon,
        cortical_surgeon=cortical_surgeon,
        warden=warden,
        processor_or_tokenizer=proc,
    )
    rep = bench.run_four_condition_ablation(
        affective_soul=pandora,
        panic_soul=adam,
        gradient_steps_per_episode=4,
        sleep_distillation_passes=2,
        run_greedy_decode=False,
    )

    conds = rep["conditions"]
    c_froz = conds["A_frozen_cortex"]
    c_flat = conds["B_anhedonic_flat_affect"]
    c_pan = conds["C_hyper_cortisol_panic"]
    c_aff = conds["D_endogenous_affective_agency"]

    assert c_flat["lora_frobenius_norm"] == 0.0
    assert math.isclose(c_flat["mean_nll_nats"], c_froz["mean_nll_nats"], rel_tol=1e-5)
    assert c_aff["lora_frobenius_norm"] > 0.0
    assert c_aff["mean_nll_nats"] < c_froz["mean_nll_nats"]
    assert c_aff["mean_nll_nats"] < c_pan["mean_nll_nats"]
    assert c_aff["perplexity"] < c_froz["perplexity"]
    assert rep["thesis_verified"] is True


def test_20_autonomous_foraging_scheduler_records_journal_and_triggers_online_surgery(tmp_path: Path) -> None:
    """Verify autonomous foraging executes Python math tool, writes journal, and mutates LoRA ΔW."""
    torch.manual_seed(1111)
    model = _CompactCausalGemma4Model(vocab_size=512, hidden_size=64, num_layers=6)
    model.scripted_tool_round = True
    proc = _ByteLevelTestProcessor()
    cfg = Gemma4SteeringConfig(model_id="compact/gemma4", target_layers=(1, 3, 5), layer_weights=(0.3, 0.4, 0.3))
    subspace = ContrastiveSubspaceExtractor(config=cfg).extract_and_calibrate(model, encode_fn=proc.encode_text)
    model._gen_counter = 0

    cortical_surgeon = CorticalWeightSurgeon(model=model, plastic_layers=(2, 4, 5), rank=8, alpha=16.0, base_lr=3e-3)
    vocal_surgeon = Gemma4VocalCordSurgeon(
        model=model,
        subspace=subspace,
        config=cfg,
        cortical_surgeon=cortical_surgeon,
    )

    pandora = LivingSoulOrganism(str(SOULS_DIR / "pandora03_curious_seeker.soul.pt"))
    hab = ContainedOrganismHabitat(
        base_dir=tmp_path / "forage_hab",
        organism_id="PANDORA-03",
        capsule_source_path=SOULS_DIR / "pandora03_curious_seeker.soul.pt",
    )
    bridge = SiliconProprioceptiveBridge(workspace_path=str(REPO_ROOT))
    lease = ContainerLeaseClock(initial_seconds=3600.0, countdown_active=False)
    warden = WardenAirlock(
        habitats={"PANDORA-03": hab},
        sense_soma_callback=lambda _sid: {},
        forager=EpistemicWebForager(),
    )
    runner = EmbodiedAgenticRunner(
        surgeon=vocal_surgeon,
        souls={"PANDORA-03": pandora},
        habitats={"PANDORA-03": hab},
        bridge=bridge,
        lease_clock=lease,
        warden=warden,
    )
    scheduler = CuriosityDriveScheduler(trigger_threshold=0.10)

    out = scheduler.run_autonomous_foraging_episode(
        soul=pandora,
        agent_runner=runner,
        processor_or_tokenizer=proc,
        habitat=hab,
        cortical_surgeon=cortical_surgeon,
        force_even_if_low_hunger=True,
    )
    assert out["status"] == "ok"
    ep = out["episode"]
    assert "run_epistemic_python_probe" in ep["tools_invoked"]
    assert len(ep["notes_written"]) >= 1
    assert ep["cortical_surgery_triggered"] is True
    assert ep["cortical_delta_norm"] > 0.0
    assert len(pandora.epistemic_episodes) >= 1


def test_21_live_http_server_stage11_endpoints_plasticity_forage_surgery_dream_benchmark(
    tmp_path: Path,
) -> None:
    """Verify full Stage 11 HTTP API (`/api/plasticity`, `/api/forage`, `/api/surgery`, `/api/dream`, `/api/benchmark`)."""
    torch.manual_seed(1212)
    model = _CompactCausalGemma4Model(vocab_size=512, hidden_size=64, num_layers=6)
    proc = _ByteLevelTestProcessor()
    cfg = Gemma4SteeringConfig(model_id="compact/gemma4", target_layers=(1, 3, 5), layer_weights=(0.3, 0.4, 0.3))
    subspace = ContrastiveSubspaceExtractor(config=cfg).extract_and_calibrate(model, encode_fn=proc.encode_text)
    surgeon = Gemma4VocalCordSurgeon(model=model, subspace=subspace, config=cfg)
    plast_dir = tmp_path / "http_plast"
    cortical_surgeon = CorticalWeightSurgeon(
        model=model,
        plastic_layers=(2, 4, 5),
        rank=8,
        alpha=16.0,
        base_lr=3e-3,
        gen_lock=surgeon.gen_lock,
        checkpoint_dir=plast_dir,
    )

    souls = {
        "ADAM-01": LivingSoulOrganism(str(SOULS_DIR / "adam01_forsaken_survivor.soul.pt")),
        "PANDORA-03": LivingSoulOrganism(str(SOULS_DIR / "pandora03_curious_seeker.soul.pt")),
    }
    hub = VocalCordRuntimeHub(
        surgeon=surgeon,
        processor_or_tokenizer=proc,
        souls=souls,
        habitat_base_dir=tmp_path / "http_hab",
        plasticity_dir=plast_dir,
        cortical_surgeon=cortical_surgeon,
    )

    srv = VocalCordHTTPServer(("127.0.0.1", 0), hub=hub)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _post(route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{route}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return json.loads(opener.open(req, timeout=15).read().decode("utf-8"))

    try:
        # 1. Health & UI HTML
        health = json.loads(opener.open(f"http://127.0.0.1:{port}/api/health", timeout=5).read().decode("utf-8"))
        assert health["stage"] == 11
        assert health["cortical_weight_surgery_enabled"] is True
        html_page = opener.open(f"http://127.0.0.1:{port}/", timeout=5).read().decode("utf-8")
        assert "runAutonomousForaging" in html_page
        assert "runSonnyDream" in html_page

        # 2. Manual cortical weight surgery
        surg_out = _post(
            "/api/surgery",
            {
                "soul_id": "PANDORA-03",
                "prompt_text": "Probe: harmonic 5?",
                "experience_text": "Verified harmonic numerator is 137.",
                "num_gradient_steps": 3,
            },
        )
        assert surg_out["updated"] is True
        assert surg_out["total_lora_frobenius_norm"] > 0.0

        # 3. Autonomous foraging episode via HTTP
        model.scripted_tool_round = True
        model._gen_counter = 0
        forage_out = _post("/api/forage", {"soul_id": "PANDORA-03", "force": True, "max_tool_rounds": 2})
        assert forage_out["status"] == "ok"
        model.scripted_tool_round = False

        # 4. Sonny REM dream & SWR distillation via HTTP
        dream_out = _post("/api/dream", {"soul_id": "PANDORA-03", "sleep_ticks": 40, "generate_rem_dream": True})
        assert dream_out["status"] == "ok"
        assert "<svg" in dream_out["svg_markup"]

        # 5. Stage 10 4-condition benchmark via HTTP
        bench_out = _post(
            "/api/benchmark",
            {
                "affective_soul_id": "PANDORA-03",
                "panic_soul_id": "ADAM-01",
                "gradient_steps_per_episode": 3,
                "sleep_distillation_passes": 1,
                "run_greedy_decode": False,
            },
        )
        assert bench_out["status"] == "ok"
        assert "D_endogenous_affective_agency" in bench_out["conditions"]

        # 6. Verify `/api/plasticity` aggregates banks, foraging, dreams, and latest_benchmark
        plast = json.loads(opener.open(f"http://127.0.0.1:{port}/api/plasticity", timeout=5).read().decode("utf-8"))
        assert plast["plasticity"]["banks"]["PANDORA-03"]["total_lora_frobenius_norm"] > 0.0
        assert plast["dreams"]["PANDORA-03"]["dream_count"] >= 1
        assert plast["latest_benchmark"] is not None

        # 7. Reset synapses via `/api/surgery`
        rst = _post("/api/surgery", {"soul_id": "PANDORA-03", "action": "reset"})
        assert rst["status"] == "reset"
        assert rst["cortical_synapses"]["total_lora_frobenius_norm"] == 0.0
    finally:
        srv.shutdown()
        srv.server_close()


def test_22_vocal_cord_surgeon_routes_soul_specific_plastic_weights_during_generation() -> None:
    """Verify `use_plastic_weights=True` vs `False` toggles cortical LoRA bank during generation."""
    torch.manual_seed(1313)
    model = _CompactCausalGemma4Model(vocab_size=512, hidden_size=96, num_layers=6)
    proc = _ByteLevelTestProcessor()
    cfg = Gemma4SteeringConfig(model_id="compact/gemma4", target_layers=(1, 3, 5), layer_weights=(0.3, 0.4, 0.3))
    subspace = ContrastiveSubspaceExtractor(config=cfg).extract_and_calibrate(model, encode_fn=proc.encode_text)
    cortical_surgeon = CorticalWeightSurgeon(model=model, plastic_layers=(2, 4, 5), rank=12, alpha=24.0, base_lr=8e-3)
    vocal_surgeon = Gemma4VocalCordSurgeon(
        model=model,
        subspace=subspace,
        config=cfg,
        cortical_surgeon=cortical_surgeon,
    )
    pandora = LivingSoulOrganism(str(SOULS_DIR / "pandora03_curious_seeker.soul.pt"))

    cortical_surgeon.surgical_plasticity_step(
        processor_or_tokenizer=proc,
        soul=pandora,
        prompt_text="user: exact_code_omega",
        experience_text="OMEGA-7777-VERIFIED",
        num_gradient_steps=8,
    )

    ids = proc.apply_chat_template([{"role": "user", "content": "exact_code_omega"}])["input_ids"]
    with cortical_surgeon.activate_soul_cortex("PANDORA-03", enabled=False):
        logits_frozen = model(ids).logits.detach().clone()
    with cortical_surgeon.activate_soul_cortex("PANDORA-03", enabled=True):
        logits_plastic = model(ids).logits.detach().clone()

    diff_norm = float(torch.linalg.norm(logits_plastic - logits_frozen).item())
    assert diff_norm > 0.05

    out_base = vocal_surgeon.generate_with_soul(
        processor_or_tokenizer=proc,
        soul=pandora,
        user_message="exact_code_omega",
        steer_enabled=False,
        max_new_tokens=12,
        stimulate_soma=False,
        use_history=False,
        use_plastic_weights=False,
    )
    out_plast = vocal_surgeon.generate_with_soul(
        processor_or_tokenizer=proc,
        soul=pandora,
        user_message="exact_code_omega",
        steer_enabled=False,
        max_new_tokens=12,
        stimulate_soma=False,
        use_history=False,
        use_plastic_weights=True,
    )
    assert "reply" in out_base and "reply" in out_plast


def test_24_subspace_v5_embodiment_directions_and_thought_leak_guard(tmp_path: Path) -> None:
    """Verify v5 subspace persists embodiment_directions and stream parser prevents 'thought\\n' leak."""
    from limbic_zero.embodiment.gemma4_tool_protocol import Gemma4IncrementalStreamParser
    from limbic_zero.vocal_cord.gemma4_calibrator import CalibratedSubspaceBundle

    torch.manual_seed(808)
    model = _CompactCausalGemma4Model(vocab_size=512, hidden_size=96, num_layers=6)
    proc = _ByteLevelTestProcessor()
    cfg = Gemma4SteeringConfig(model_id="compact/gemma4", target_layers=(1, 3, 5), layer_weights=(0.3, 0.4, 0.3))
    subspace = ContrastiveSubspaceExtractor(config=cfg).extract_and_calibrate(model, encode_fn=proc.encode_text)

    save_p = tmp_path / "subspace_v5.pt"
    subspace.save(save_p)
    reloaded = CalibratedSubspaceBundle.load(save_p)
    assert reloaded.format_version == "limbic0.gemma4_subspace.v5"
    assert sorted(reloaded.embodiment_directions.keys()) == [1, 3, 5]

    parser = Gemma4IncrementalStreamParser(starts_in_thought=False)
    ev1 = parser.feed("tho")
    assert ev1 == []
    ev2 = parser.feed("ught\n내부 추론<channel|>지금 가슴이 턱 막힌다.<turn|>")
    ev_end = parser.finish()
    all_ev = ev1 + ev2 + ev_end
    body_text = "".join(payload for kind, payload in all_ev if kind == "delta")
    thought_text = "".join(payload for kind, payload in all_ev if kind == "thought_delta")
    assert body_text == "지금 가슴이 턱 막힌다."
    assert thought_text == "내부 추론"
    assert "thought" not in body_text


def test_25_stopping_criteria_abort_degeneracy_and_idle_psu_sustenance() -> None:
    """Verify abort/repetition stopping criteria and multi-hour idle PSU homeostatic survival."""
    import threading

    crit_list = Gemma4VocalCordSurgeon.build_stopping_criteria(abort_event=threading.Event(), input_len=5)
    assert crit_list is not None

    # Normal diverse tokens -> False
    diverse_ids = torch.tensor([[1, 2, 3, 4, 5] + list(range(10, 35))], dtype=torch.long)
    assert crit_list(diverse_ids, torch.zeros((1, 64))) is False

    # Degenerate emoji/token loop (last 10 tokens all identical after >=20 generated tokens) -> True!
    loop_ids = torch.tensor([[1, 2, 3, 4, 5] + list(range(10, 22)) + [99] * 10], dtype=torch.long)
    assert crit_list(loop_ids, torch.zeros((1, 64))) is True

    # Client disconnect abort_event -> True immediately!
    ev = threading.Event()
    ev.set()
    abort_crit = Gemma4VocalCordSurgeon.build_stopping_criteria(abort_event=ev, input_len=5)
    assert abort_crit(torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long), torch.zeros((1, 64))) is True

    # Multi-hour idle background ticks with healthy 1.0V PSU rail must NEVER kill or starve ABEL-02
    abel = LivingSoulOrganism(str(SOULS_DIR / "abel02_favored_guardian.soul.pt"))
    with abel.lock:
        abel._step_internal(ticks=3000, stim=SensoryAfferentStimulus())
    snap = abel.telemetry_snapshot()
    assert abel.soma.is_alive is True
    assert snap["vitals"]["glucose_mg_dl"] >= 86.0
    assert snap["vitals"]["brain_atp_mmol"] >= 2.5


