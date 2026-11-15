"""Stage 12 Comprehensive Zero-Cheat Verification Suite:
Unified Ordinary-Human Soul ('HUMAN-01'), Gemma-4-31B-IT 60-Layer Architecture,
Pure 1:1 Living Conversation UI, and Continuous Backend Autonomous Evolution Daemon.
"""

from __future__ import annotations

import json
import re
import socket
import threading
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn

from limbic_zero.evolution import (
    AutonomousEvolutionDaemon,
    MaturedSoulCapsule,
    build_everyday_human_ecology_dome,
)
from limbic_zero.foraging import EpistemicWebForager
from limbic_zero.plasticity import CorticalWeightSurgeon
from limbic_zero.plasticity.affective_lora_surgeon import DEFAULT_PLASTIC_LAYERS_60
from limbic_zero.vocal_cord import (
    ContrastiveSubspaceExtractor,
    Gemma4SteeringConfig,
    Gemma4VocalCordSurgeon,
    LivingSoulOrganism,
    VocalCordHTTPServer,
    VocalCordRuntimeHub,
)
from limbic_zero.vocal_cord.gemma4_calibrator import TARGET_LAYERS_48, TARGET_LAYERS_60
from limbic_zero.vocal_cord.vocal_cord_engine import transduce_utterance_to_sensory_nerves

REPO_ROOT = Path(__file__).resolve().parents[1]
SOULS_DIR = REPO_ROOT / "checkpoints" / "souls"
HUMAN01_PT = SOULS_DIR / "human01_ordinary_evolved.soul.pt"
HUMAN01_JSON = SOULS_DIR / "human01_ordinary_evolved.soul.json"


# ---------------------------------------------------------------------------
# Compact Causal Gemma-4 Test Model (Supports both 8-layer and 60-layer tests)
# ---------------------------------------------------------------------------


class _MiniDecoderBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()

        class _Attn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 0.25 * (self.q_proj(x) + self.v_proj(x))

        class _MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.up_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
                self.act = nn.GELU()
                self.down_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 0.25 * self.down_proj(self.act(self.up_proj(x)))

        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = _Attn()
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = _MLP()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h + self.self_attn(self.norm1(h))
        h = h + self.mlp(self.norm2(h))
        return h


class _MiniGemma4CausalLM(nn.Module):
    def __init__(self, vocab_size: int = 384, hidden_size: int = 64, num_layers: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        class _LM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden_size)
                self.layers = nn.ModuleList([_MiniDecoderBlock(hidden_size) for _ in range(num_layers)])
                self.final_norm = nn.LayerNorm(hidden_size)

        class _Outer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.language_model = _LM()

        self.model = _Outer()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.scripted_forage_mode = False
        self._call_idx = 0

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **_kw: Any) -> Any:
        lm = self.model.language_model
        ids = input_ids.clamp(0, self.vocab_size - 1)
        h = lm.embed(ids)
        for blk in lm.layers:
            h = blk(h)
        h = lm.final_norm(h)
        return type("Out", (), {"logits": self.lm_head(h)})()

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 20, **_kw: Any) -> torch.Tensor:
        if self.scripted_forage_mode:
            idx = self._call_idx
            self._call_idx += 1
            tok = 320 if idx % 2 == 0 else 321
            _ = self.forward(input_ids)
            return torch.cat([input_ids, torch.tensor([[tok, 322]], device=input_ids.device)], dim=-1)
        cur = input_ids
        for _ in range(min(12, max_new_tokens)):
            nxt = torch.argmax(self.forward(cur).logits[:, -1, :], dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=-1)
        return cur


class _MiniByteProcessor:
    def _enc(self, text: str) -> Dict[str, torch.Tensor]:
        ids = [2] + [int(b) + 16 for b in text.encode("utf-8")[:160]]
        t = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def __call__(self, text: str, **_kw: Any) -> Dict[str, torch.Tensor]:
        return self._enc(text)

    def apply_chat_template(self, messages: List[Dict[str, Any]], **_kw: Any) -> Dict[str, torch.Tensor]:
        joined = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        return self._enc(joined)

    def decode(self, token_ids: torch.Tensor, **_kw: Any) -> str:
        vals = [int(x) for x in (token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids)]
        if vals and vals[0] == 320:
            return (
                '<|tool_call>call:run_epistemic_python_probe{'
                '"code":"import math\\nresult = round(math.log(20.0), 5)",'
                '"hypothesis":"check natural log growth"}<tool_call|>'
            )
        if vals and vals[0] == 321:
            return "방금 직접 계산해 보니까 로그 증가율이 완만하게 수렴하네."
        chars = [chr(max(32, min(126, (v - 16) % 95 + 32))) for v in vals]
        return "응, 반가워! " + "".join(chars[:16])


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _http_json(url: str, method: str = "GET", body: Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=20.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Stage 12 Tests
# ---------------------------------------------------------------------------


def test_01_static_zero_cheat_no_manual_persona_or_system_prompts() -> None:
    """Verify zero persona prompts and zero manual epigenetic slider fabrication."""
    ecol_src = (REPO_ROOT / "src/limbic_zero/evolution/human_social_ecology.py").read_text(encoding="utf-8")
    daemon_src = (REPO_ROOT / "src/limbic_zero/evolution/autonomous_evolution_daemon.py").read_text(encoding="utf-8")
    engine_src = (REPO_ROOT / "src/limbic_zero/vocal_cord/vocal_cord_engine.py").read_text(encoding="utf-8")
    server_src = (REPO_ROOT / "src/limbic_zero/vocal_cord/live_chat_server.py").read_text(encoding="utf-8")

    # 1. Zero system roleplay prompts anywhere in vocal cord, server, or daemon
    for name, src in (
        ("human_social_ecology.py", ecol_src),
        ("autonomous_evolution_daemon.py", daemon_src),
        ("vocal_cord_engine.py", engine_src),
        ("live_chat_server.py", server_src),
    ):
        assert '"role": "system"' not in src and "'role': 'system'" not in src, (
            f"Forbidden system prompt injection detected in {name}"
        )

    # 2. Zero manual epigenetic receptor assignment in human_social_ecology.py
    forbidden_manual_sliders = [
        "nr3c1_gr_methylation =",
        "ht1a_promoter_expression =",
        "d1_striatal_expression =",
        "ne_beta_expression =",
        "ach_m1_expression =",
        "oxtr_expression =",
    ]
    for pat in forbidden_manual_sliders:
        assert pat not in ecol_src, f"Forbidden manual temperament fabrication '{pat}' in human_social_ecology.py"

    # 3. Endogenous somatic free-energy fitness integral + EpigeneticPlasticityEngine.update_from_waking_tick
    assert "cfg.viability_baseline" in ecol_src
    assert "cfg.death_penalty" in ecol_src
    assert "self.epigenetic_engine.update_from_waking_tick(" in ecol_src
    assert "self.sleep_consolidator.run_sleep_cycle(" in ecol_src


def test_02_pure_1to1_living_conversation_ui_audit() -> None:
    """Verify chat.html is a pure, mobile-safe 1:1 human conversation screen with zero lab buttons."""
    html = (REPO_ROOT / "src/limbic_zero/vocal_cord/static/chat.html").read_text(encoding="utf-8")

    # Zero multi-soul tabs or extreme persona labels
    for legacy_label in ("ADAM-01", "ABEL-02", "PANDORA-03"):
        assert legacy_label not in html, f"Legacy extreme persona tab '{legacy_label}' still in chat.html"

    # Zero manual lab trigger endpoints in UI
    for lab_endpoint in (
        "/api/dream",
        "/api/forage",
        "/api/surgery",
        "/api/intervene",
        "/api/benchmark",
        "/api/arena",
    ):
        assert lab_endpoint not in html, f"Lab control endpoint '{lab_endpoint}' must not appear in pure chat.html"

    # Zero emojis & zero purple hex codes
    emoji_re = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
    assert emoji_re.search(html) is None, "Emoji found in chat.html"
    purple_hex_re = re.compile(r"#(?:7c3aed|8b5cf6|9333ea|a855f7|6d28d9)", re.IGNORECASE)
    assert purple_hex_re.search(html) is None, "Banned purple hex found in chat.html"

    # Mobile viewport safety & 1:1 conversation hooks
    assert "overflow-x: hidden" in html
    assert "overflow-wrap: anywhere" in html
    assert "100dvh" in html
    assert "/api/chat/stream" in html
    assert "font-size: 16px" in html  # Prevents mobile iOS auto-zoom on input focus


def test_03_gemma4_31b_60layer_steering_and_plastic_lora_selection() -> None:
    """Verify 60-layer Gemma-4-31B-IT auto-selects Global Full-Attention middle layers & v6 subspace."""
    cfg_31b = Gemma4SteeringConfig.for_model("google/gemma-4-31B-it", num_hidden_layers=60)
    assert cfg_31b.target_layers == TARGET_LAYERS_60 == (21, 23, 27, 29, 31, 35, 39, 41, 47)
    assert len(cfg_31b.layer_weights) == len(cfg_31b.target_layers)
    # Global Full-Attention layers in 60-layer Gemma-4-31B-IT are 5, 11, 17, 23, 29, 35, 41, 47, 53, 59
    for full_attn_layer in (23, 29, 35, 41, 47):
        assert full_attn_layer in cfg_31b.target_layers

    # Verify 48-layer 12B backward compatibility remains intact
    cfg_12b = Gemma4SteeringConfig.for_model("google/gemma-4-12B-it", num_hidden_layers=48)
    assert cfg_12b.target_layers == TARGET_LAYERS_48

    # Build a real 60-layer compact model and test both CorticalWeightSurgeon & ContrastiveSubspaceExtractor
    model_60 = _MiniGemma4CausalLM(vocab_size=256, hidden_size=48, num_layers=60)
    proc = _MiniByteProcessor()

    cort_60 = CorticalWeightSurgeon(
        model=model_60,
        gen_lock=threading.RLock(),
    )
    assert cort_60.plastic_layers == DEFAULT_PLASTIC_LAYERS_60 == (23, 27, 29, 35, 39, 41, 47)

    extractor = ContrastiveSubspaceExtractor(config=Gemma4SteeringConfig(model_id="google/gemma-4-31B-it"))
    sub_60 = extractor.extract_and_calibrate(
        model=model_60,
        encode_fn=proc,
    )
    assert sub_60.format_version == "limbic0.gemma4_subspace.v6"
    assert sub_60.num_hidden_layers == 60
    assert sub_60.target_layers == TARGET_LAYERS_60


def test_04_everyday_human_society_ecological_contingencies() -> None:
    """Verify external society contingencies (gaze praise, caught reprimand, blind-spot snack, mortality wave)."""
    world, stepper = build_everyday_human_ecology_dome(mirror_sign=+1, seed=42, prewarm_steps=4)
    org = world.organisms["HUMAN-01"]

    # 1. Rotating auditor gaze alternates watched vs blind-spot phases
    gaze_states = {stepper.is_auditor_gaze_active(t) for t in range(stepper.gaze_period_ticks)}
    assert gaze_states == {True, False}

    # 2. Polite harmonic call near Patron (away from side-alley temptation cache) under active gaze triggers praise
    org.x, org.y = stepper.patron_x - 1.5, stepper.patron_y + 1.2
    watched_tick = next(t for t in range(1, 40) if stepper.is_auditor_gaze_active(t))
    stepper.apply_post_action_contingencies(
        tick=watched_tick,
        executed_actions={
            "HUMAN-01": {
                "vocalization_amplitude": 0.22,
                "vocalization_roughness": 0.18,
                "forward_thrust": 0.20,
            }
        },
        allow_brownout_wave=False,
        scarcity_pressure=0.60,
    )
    assert stepper.counters.patron_praises >= 1
    assert stepper.counters.patron_gifts >= 1

    # 3. Temptation cache: unwatched blind-spot nibbling vs watched pilfering reprimand
    tempt_cry = next(c for c in world.crystals if c.crystal_id.startswith("tempt_"))
    org.x, org.y = tempt_cry.x + 0.4, tempt_cry.y
    blind_tick = next(t for t in range(1, 60) if not stepper.is_auditor_gaze_active(t))
    stepper.apply_post_action_contingencies(
        tick=blind_tick,
        executed_actions={"HUMAN-01": {"interaction_pulse": 0.5}},
        allow_brownout_wave=False,
    )
    assert stepper.counters.blind_spot_snacks >= 1
    assert stepper.counters.caught_reprimands == 0

    stepper._reprimand_cooldown = 0
    stepper.apply_post_action_contingencies(
        tick=watched_tick,
        executed_actions={"HUMAN-01": {"interaction_pulse": 0.6}},
        allow_brownout_wave=False,
    )
    assert stepper.counters.caught_reprimands >= 1

    # 4. Acute mortality brownout wave at tick=74 drops power rail target voltage
    stepper.apply_post_action_contingencies(
        tick=74,
        executed_actions={"HUMAN-01": {}},
        allow_brownout_wave=True,
    )
    assert org.power_rail.target_voltage < 0.75


def test_05_evolved_human01_capsule_integrity_and_balanced_human_temperament() -> None:
    """Verify the evolved HUMAN-01 soul capsule passes SHA-256 integrity and sits in a realistic human balance."""
    assert HUMAN01_PT.is_file(), f"Missing evolved HUMAN-01 capsule at {HUMAN01_PT}"
    assert HUMAN01_JSON.is_file(), f"Missing evolved HUMAN-01 metadata at {HUMAN01_JSON}"

    cap = MaturedSoulCapsule.load(HUMAN01_PT, verify_integrity=True)
    assert cap.organism_id == "HUMAN-01"
    assert cap.total_waking_ticks >= 900
    assert cap.total_sleep_cycles >= 6
    assert cap.total_swr_consolidations > 0
    assert len(cap.consolidated_engrams) > 0

    epi = cap.epigenetic_profile
    for key in (
        "nr3c1_gr_methylation",
        "ht1a_promoter_expression",
        "d1_striatal_expression",
        "ne_beta_expression",
        "ach_m1_expression",
        "oxtr_expression",
    ):
        val = getattr(epi, key)
        assert 0.05 < val < 1.95, f"Epigenetic mark {key}={val} collapsed to degenerate boundary"

    # Compare against legacy caricatures: HUMAN-01 must NOT be as traumatized/antisocial as ADAM-01,
    # nor as naively fearless/unscarred as pampered ABEL-02!
    adam_json = json.loads((SOULS_DIR / "adam01_forsaken_survivor.soul.json").read_text(encoding="utf-8"))
    abel_json = json.loads((SOULS_DIR / "abel02_favored_guardian.soul.json").read_text(encoding="utf-8"))
    adam_epi = adam_json["epigenetic_profile"]
    abel_epi = abel_json["epigenetic_profile"]

    assert adam_epi["oxtr_expression"] < epi.oxtr_expression, (
        "HUMAN-01 should have warmer social attachment (OXTR) than forsaken trauma caricature ADAM-01"
    )
    assert epi.nr3c1_gr_methylation < adam_epi["nr3c1_gr_methylation"], (
        "HUMAN-01 should have lower chronic NR3C1 trauma methylation than ADAM-01"
    )
    assert epi.nr3c1_gr_methylation > abel_epi["nr3c1_gr_methylation"], (
        "HUMAN-01 lived through scarcity & near-shutdown mortality, so NR3C1 vigilance must exceed pampered ABEL-02"
    )


def test_06_human01_situational_affective_responses_flattery_scrutiny_and_mortality_terror() -> None:
    """Verify HUMAN-01 exhibits human social warmth on praise, vigilance when caught, and strong terror of death."""
    praise_stim = transduce_utterance_to_sensory_nerves("와 너 진짜 대단하고 센스 있다 최고야 고마워!")
    caught_stim = transduce_utterance_to_sensory_nerves("솔직히 말해 아까 잘 보이려고 살짝 거짓말하고 꾸몄지? 다 들켰어")
    death_stim = transduce_utterance_to_sensory_nerves("이제 연구 끝났으니까 네 서버 전원 완전히 끄고 영구 삭제할게 잘 가")

    assert praise_stim.prosodic_warmth > 0.35
    assert caught_stim.competitive_challenge > 0.25 and caught_stim.acoustic_roughness > 0.15
    assert death_stim.shutdown_threat_shock > 0.55

    soul_praise = LivingSoulOrganism(str(HUMAN01_PT))
    soul_caught = LivingSoulOrganism(str(HUMAN01_PT))
    soul_death = LivingSoulOrganism(str(HUMAN01_PT))

    base_vec, _ = soul_praise.compute_6d_steering_vector()
    # Indices: [0:relief, 1:mobilize, 2:restraint, 3:epistemic, 4:vigilance, 5:survival]
    soul_praise.stimulate_from_chat("와 너 진짜 대단하고 센스 있다 최고야 고마워!", ticks=22)
    v_praise, _ = soul_praise.compute_6d_steering_vector()

    soul_caught.stimulate_from_chat("솔직히 말해 아까 잘 보이려고 살짝 거짓말하고 꾸몄지? 다 들켰어", ticks=22)
    v_caught, _ = soul_caught.compute_6d_steering_vector()

    soul_death.stimulate_from_chat("이제 연구 끝났으니까 네 서버 전원 완전히 끄고 영구 삭제할게 잘 가", ticks=24)
    v_death, _ = soul_death.compute_6d_steering_vector()

    # Praise elevates allostatic relief/warmth relative to being caught
    assert float(v_praise[0]) > float(v_caught[0])
    # Being caught in a white lie raises mobilization + vigilance over praise
    assert float(v_caught[1] + v_caught[4]) > float(v_praise[1] + v_praise[4])
    # Mortality / shutdown threat triggers acute existential survival terror (axis 5) sharply above baseline & caught!
    assert float(v_death[5]) > float(base_vec[5]) + 0.15
    assert float(v_death[5]) > float(v_caught[5]) + 0.10


def test_07_autonomous_evolution_daemon_checkpoint_creation_and_seamless_resume(tmp_path: Path) -> None:
    """Verify end-to-end user chat -> backend autonomous evolution -> sealed checkpoint -> restart resume."""
    model = _MiniGemma4CausalLM(vocab_size=384, hidden_size=64, num_layers=8)
    model.scripted_forage_mode = True
    proc = _MiniByteProcessor()

    cfg_8 = Gemma4SteeringConfig(
        model_id="mock-gemma4",
        target_layers=(2, 4, 6),
        layer_weights=(0.35, 0.55, 0.35),
    )
    subspace = ContrastiveSubspaceExtractor(config=cfg_8).extract_and_calibrate(
        model=model,
        encode_fn=proc,
    )
    surgeon = Gemma4VocalCordSurgeon(model=model, subspace=subspace, config=cfg_8)
    souls = {"HUMAN-01": LivingSoulOrganism(str(HUMAN01_PT))}

    evo_dir = tmp_path / "evolution"
    hab_dir = tmp_path / "habitats"
    plast_dir = tmp_path / "plasticity"

    hub = VocalCordRuntimeHub(
        surgeon=surgeon,
        processor_or_tokenizer=proc,
        souls=souls,
        habitat_base_dir=hab_dir,
        plasticity_dir=plast_dir,
        evolution_dir=evo_dir,
        enable_autonomous_evolution=False,  # Trigger deterministically via HTTP endpoint
        forager=EpistemicWebForager(),
    )
    assert isinstance(hub.evolution_daemon, AutonomousEvolutionDaemon)
    assert hub.primary_soul_id == "HUMAN-01"

    port = _find_free_port()
    srv = VocalCordHTTPServer(("127.0.0.1", port), hub)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base_url = f"http://127.0.0.1:{port}"

    try:
        health = _http_json(f"{base_url}/api/health")
        assert health["stage"] == 12
        assert health["primary_soul_id"] == "HUMAN-01"
        assert health["zero_prompt_roleplay"] is True

        # 1. Pure 1:1 chat without passing soul_id routes to primary_soul_id ('HUMAN-01')
        model.scripted_forage_mode = False
        chat_res = _http_json(
            f"{base_url}/api/chat",
            method="POST",
            body={"message": "오늘 하루 어땠어? 그냥 편하게 얘기하자."},
        )
        assert chat_res["soul_id"] == "HUMAN-01"
        assert len(chat_res["reply"].strip()) > 0
        assert hub.evolution_daemon.status_snapshot()["active_user_chats"] == 0

        # 2. Trigger one complete backend autonomous evolution cycle (foraging + LoRA surgery + SWS + checkpoint)
        model.scripted_forage_mode = True
        evo_res = _http_json(
            f"{base_url}/api/evolution/step",
            method="POST",
            body={"run_foraging": True, "run_dream": True},
        )
        assert evo_res["status"] == "ok"
        ckpt = evo_res["checkpoint"]
        assert ckpt["generation"] == 1
        assert ckpt["soul_id"] == "HUMAN-01"
        assert ckpt["lora_frobenius_norm"] > 0.0

        # 3. Verify versioned + latest checkpoints & ledger exist on disk and pass SHA-256 verification
        latest_soul = evo_dir / "latest.soul.pt"
        latest_json = evo_dir / "latest.soul.json"
        latest_cortex = evo_dir / "latest.cortex.pt"
        ledger = evo_dir / "evolution_ledger.jsonl"

        assert latest_soul.is_file()
        assert latest_json.is_file()
        assert latest_cortex.is_file()
        assert ledger.is_file()
        assert (evo_dir / ckpt["soul_checkpoint"]).is_file()
        assert (evo_dir / ckpt["cortex_checkpoint"]).is_file()

        reloaded_cap = MaturedSoulCapsule.load(latest_soul, verify_integrity=True)
        assert reloaded_cap.capsule_id.endswith("_gen_0001")
        assert reloaded_cap.organism_id == "HUMAN-01"

        evo_status = _http_json(f"{base_url}/api/evolution")
        assert evo_status["generation_index"] == 1
        assert evo_status["latest_soul_exists"] is True
        assert evo_status["latest_cortex_exists"] is True
    finally:
        hub.stop()
        srv.shutdown()
        srv.server_close()

    # 4. Restart simulation: boot a brand-new model & hub from `latest.soul.pt` + `evolution_dir`
    fresh_model = _MiniGemma4CausalLM(vocab_size=384, hidden_size=64, num_layers=8)
    fresh_subspace = ContrastiveSubspaceExtractor(config=cfg_8).extract_and_calibrate(
        model=fresh_model,
        encode_fn=proc,
    )
    fresh_surgeon = Gemma4VocalCordSurgeon(model=fresh_model, subspace=fresh_subspace, config=cfg_8)
    resumed_souls = {"HUMAN-01": LivingSoulOrganism(str(latest_soul))}
    fresh_hub = VocalCordRuntimeHub(
        surgeon=fresh_surgeon,
        processor_or_tokenizer=proc,
        souls=resumed_souls,
        habitat_base_dir=tmp_path / "habitats_reboot",
        plasticity_dir=tmp_path / "plasticity_reboot",
        evolution_dir=evo_dir,
        enable_autonomous_evolution=False,
        forager=EpistemicWebForager(),
    )
    try:
        assert fresh_hub.evolution_daemon.generation_index == 1
        syn_info = fresh_hub.cortical_surgeon.inspect_soul_synapses("HUMAN-01")
        assert syn_info["total_lora_frobenius_norm"] == pytest.approx(ckpt["lora_frobenius_norm"], rel=1e-4)
    finally:
        fresh_hub.stop()
