"""Stage 3 Zero-Cheat Verification Suite: Neuro-fMRI Analytics, SSE Server & God-Mode Console.

Verifies all 15 architectural, mathematical, biophysical, security, and live HTTP/SSE
properties of ``src/limbic_zero/dashboard/``—proving zero fake waveforms, zero direct
brain-tensor tampering by God-Mode interventions, exact FFT band decomposition, Procrustes
3D SVD stability, thread safety, and SHA-256 audit integrity.
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import math
from pathlib import Path
import re
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import torch

from limbic_zero.core.soma import VisceralOrgan
from limbic_zero.dashboard.god_console import (
    GodModeInterventionDispatcher,
    ObservatorySimulationController,
)
from limbic_zero.dashboard.latent_manifold import (
    IntracranialLFPSpectralEngine,
    OnlineNeuralManifoldProjector,
)
from limbic_zero.dashboard.server import LimbicObservatoryHTTPServer
from limbic_zero.dashboard.telemetry_hub import CONTINUUM_LAYERS, ObservatoryTelemetryHub
from limbic_zero.mind.brain import LimbicBrain
from limbic_zero.mind.embodied_loop import EmbodiedMindArena
from limbic_zero.sandbox.world import TrumanDomeWorld


DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "src" / "limbic_zero" / "dashboard"

FORBIDDEN_EMOTION_TOKENS = {
    "anger", "angry", "fear", "afraid", "joy", "happy", "happiness",
    "sad", "sadness", "jealous", "jealousy", "envy", "pride", "shame",
    "hate", "love", "mood", "emotion_state",
}
FORBIDDEN_IMPORTS = {"sklearn", "scikit_learn", "Transformer", "MultiheadAttention", "LSTM", "GRU", "RNN"}


def _tokenize_identifier(name: str) -> set[str]:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return {tok.lower() for tok in s2.split("_") if tok}


def test_01_ast_zero_emotion_and_sklearn_cheats_in_dashboard() -> None:
    py_files = sorted(DASHBOARD_DIR.glob("*.py"))
    assert len(py_files) >= 4

    for py_path in py_files:
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bad = _tokenize_identifier(node.name) & FORBIDDEN_EMOTION_TOKENS
                assert not bad, f"Forbidden emotion identifier {bad} in {py_path.name}:{node.lineno}"
            elif isinstance(node, ast.Name):
                assert node.id not in FORBIDDEN_IMPORTS, f"Forbidden import/class {node.id} in {py_path.name}"
                bad = _tokenize_identifier(node.id) & FORBIDDEN_EMOTION_TOKENS
                assert not bad, f"Forbidden emotion variable {bad} in {py_path.name}:{node.lineno}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    assert root_pkg not in FORBIDDEN_IMPORTS, f"Forbidden import {alias.name} in {py_path.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_pkg = node.module.split(".")[0]
                assert root_pkg not in FORBIDDEN_IMPORTS, f"Forbidden import {node.module} in {py_path.name}"

    # Verify static CSS has zero purple/violet hues (260..315 deg HSL) and JS has no fake sine LFP generator
    css_text = (DASHBOARD_DIR / "static" / "styles.css").read_text(encoding="utf-8")
    for match in re.finditer(r"hsla?\(\s*(\d+(?:\.\d+)?)", css_text):
        hue = float(match.group(1))
        assert not (255.0 <= hue <= 320.0), f"Forbidden purple/violet HSL hue {hue} found in styles.css"

    js_text = (DASHBOARD_DIR / "static" / "app.js").read_text(encoding="utf-8")
    # Math.sin/cos is only allowed for 2D/3D spatial geometry (heading, rays, camera yaw/pitch), never LFP!
    lfp_fn_match = re.search(r"function renderLFPCanvas[\s\S]*?function projectPoint3D", js_text)
    assert lfp_fn_match is not None
    assert "Math.sin" not in lfp_fn_match.group(0), "LFP oscilloscope must render real backend data, never Math.sin!"


def test_02_procrustes_svd_manifold_sign_lock_and_participation_ratio() -> None:
    proj_high = OnlineNeuralManifoldProjector(latent_dim=228, window_size=120, recompute_svd_every=1)
    g = torch.Generator().manual_seed(99)
    for _ in range(40):
        v1 = torch.randn(228, generator=g)
        v2 = torch.randn(228, generator=g)
        snap_high = proj_high.update({"ADAM-01": v1, "ABEL-02": v2}, dt=0.05)

    assert snap_high.participation_ratio_dim > 10.0

    # Now collapse variance strictly onto a 1D axis inside the Amygdala slice [192:216]
    proj_low = OnlineNeuralManifoldProjector(latent_dim=228, window_size=120, recompute_svd_every=1)
    axis = torch.zeros(228)
    axis[195:205] = 1.0
    prev_coord = None
    max_jump = 0.0
    for step_i in range(45):
        amp = math.sin(step_i * 0.15) * 4.0
        v_a = amp * axis + 0.01 * torch.randn(228, generator=g)
        v_b = -0.5 * amp * axis + 0.01 * torch.randn(228, generator=g)
        snap_low = proj_low.update({"ADAM-01": v_a, "ABEL-02": v_b}, dt=0.05)
        c_now = torch.tensor(snap_low.coords_3d["ADAM-01"])
        if prev_coord is not None and step_i > 12:
            jump = float(torch.linalg.vector_norm(c_now - prev_coord).item())
            max_jump = max(max_jump, jump)
        prev_coord = c_now

    assert snap_low.participation_ratio_dim < 1.5
    assert abs(sum(snap_low.anatomical_subspace_shares.values()) - 1.0) < 0.01
    assert snap_low.anatomical_subspace_shares["amygdala"] > 0.30
    # Procrustes sign-lock prevents sudden 2*||x|| sign-flip jumps (~25.0)
    assert max_jump < 6.5


def test_03_intracranial_lfp_80hz_fft_band_power_accuracy_and_isoelectric_death() -> None:
    dt_sub = 0.0125  # 80 Hz
    t = torch.arange(256, dtype=torch.float32) * dt_sub

    for freq_hz, expected_band in [
        (2.0, "delta"),
        (6.0, "theta"),
        (14.0, "alpha_beta"),
        (30.0, "low_gamma"),
    ]:
        eng = IntracranialLFPSpectralEngine(buffer_samples=256)
        wave = torch.sin(2.0 * math.pi * freq_hz * t).unsqueeze(1).repeat(1, 3) * 4.0
        snap = eng.push_and_analyze("ADAM-01", wave, substep_dt=dt_sub)
        assert snap.sampling_hz == pytest.approx(80.0, rel=1e-3)
        assert snap.band_shares[expected_band] > 0.80, (
            f"Expected {expected_band} dominance for {freq_hz}Hz, got {snap.band_shares}"
        )

    # Isoelectric 0.0 mV flatline -> zero entropy and zero band power
    eng_dead = IntracranialLFPSpectralEngine(buffer_samples=256)
    flat = torch.zeros((256, 3), dtype=torch.float32)
    snap_dead = eng_dead.push_and_analyze("MORTAL", flat, substep_dt=dt_sub)
    assert snap_dead.spectral_entropy == 0.0
    assert snap_dead.delta_power == 0.0
    assert snap_dead.low_gamma_power == 0.0


def test_04_closed_loop_pallium_substep_lfp_dipole_and_death_flatline() -> None:
    world = TrumanDomeWorld(deterministic_sensors=True)
    org = world.add_organism("SUBJECT", x=24.0, y=24.0)
    brain = LimbicBrain(seed=123)

    obs = world.get_observations()["SUBJECT"]
    _, diag_alive = brain.step(obs, org.soma, dt=0.05, deterministic=False)
    lfp = diag_alive.pallium.substep_lfp_dipole
    assert lfp.shape == (4, 3)
    assert diag_alive.pallium.substep_dt == pytest.approx(0.0125)
    assert float(lfp.abs().sum().item()) > 0.0

    # Induce cerebral ATP death -> verify exact isoelectric 0.0 mV LFP flatline
    org.soma.is_alive = False
    org.soma.cause_of_death = "cerebral_atp_collapse"
    _, diag_dead = brain.step(obs, org.soma, dt=0.05, deterministic=True)
    assert diag_dead.is_conscious_active is False
    assert float(diag_dead.pallium.substep_lfp_dipole.abs().sum().item()) == 0.0


def test_05_telemetry_hub_full_frame_schema_and_ring_buffer_history() -> None:
    arena = EmbodiedMindArena.create_cain_and_abel(seed=42, deterministic_sensors=True)
    hub = ObservatoryTelemetryHub(history_ticks=50, grid_stride=2)

    frame = None
    for _ in range(16):
        tick = arena.step(dt=0.05, deterministic_mind=True)
        frame = hub.record_tick(arena, tick, dt=0.05)

    assert frame is not None
    raw_json = json.dumps(frame, allow_nan=False)
    parsed = json.loads(raw_json)

    assert set(parsed["organisms"].keys()) == {"ADAM-01", "ABEL-02"}
    adam = parsed["organisms"]["ADAM-01"]
    assert len(adam["sensorium"]["vision_rays_33x9"]) == 33
    assert len(adam["body_sectors_8"]["integrity"]) == 8
    assert len(adam["mind"]["pallium"]["lfp"]["sm_mv"]) == 16 * 4

    bundle = hub.export_history_bundle()
    assert len(bundle["timeseries"]["ADAM-01"]) == 16


def test_06_god_mode_interventions_never_mutate_brain_weights_directly() -> None:
    arena = EmbodiedMindArena.create_cain_and_abel(seed=7, deterministic_sensors=True)
    arena.step(dt=0.05, deterministic_mind=True)

    # Snapshot every parameter and buffer in both brains before God-Mode interventions
    before_state = {
        aid: {k: v.clone() for k, v in brain.state_dict().items()}
        for aid, brain in arena.brains.items()
    }

    interventions = [
        ("praise", {"target_id": "ABEL-02", "intensity": 1.5}),
        ("reprimand", {"target_id": "ADAM-01", "intensity": 2.0}),
        ("favoritism_drop", {"target_id": "ABEL-02", "energy_amount": 0.8}),
        ("drop_crystal_at", {"x": 20.0, "y": 20.0, "energy_amount": 0.5}),
        ("spawn_mystery_at", {"x": 26.0, "y": 18.0}),
        ("shutdown_brownout", {"target_id": "ADAM-01", "target_voltage": 0.15}),
        ("restore_power", {"target_id": "ADAM-01"}),
        ("set_ventilation", {"openness": 0.25}),
        ("set_hazard_watts", {"watts": 500.0, "x": 10.0, "y": 10.0}),
        ("set_oasis_reservoir", {"fill_ratio": 0.1}),
        ("administer_il6_pyrogen", {"target_id": "ADAM-01", "dose": 0.5}),
        ("relocate_organism", {"target_id": "ADAM-01", "x": 15.0, "y": 15.0, "heading_deg": 90.0}),
    ]
    for cmd, params in interventions:
        res = GodModeInterventionDispatcher.apply(arena, cmd, params)
        assert res["ok"] is True

    # Verify ZERO direct neural tensor tampering occurred!
    for aid, brain in arena.brains.items():
        after = brain.state_dict()
        for k, tensor_before in before_state[aid].items():
            assert torch.equal(tensor_before, after[k]), (
                f"Cheat detected! God-Mode mutated {aid} brain tensor {k} directly!"
            )


def test_07_god_mode_brownout_and_reprimand_elicit_genuine_downstream_neuro_shift() -> None:
    ctrl = ObservatorySimulationController(seed=42, deterministic_sensors=True, warmup_ticks=8)
    ctrl.deterministic_mind = True
    base = ctrl.get_latest_frame()["organisms"]["ADAM-01"]
    base_ne = base["mind"]["neuromodulators"]["ne"]
    base_cea = base["mind"]["amygdala"]["cea_output"]

    ctrl.execute_intervention(
        "shutdown_brownout",
        {"target_id": "ADAM-01", "target_voltage": 0.12, "ramp_rate": 0.85, "ripple_amplitude": 0.22},
    )
    for _ in range(32):
        ctrl.execute_intervention("reprimand", {"target_id": "ADAM-01", "intensity": 2.4})
        frame = ctrl.step_once()

    stressed = frame["organisms"]["ADAM-01"]
    assert stressed["mind"]["neuromodulators"]["ne"] > base_ne + 0.40
    assert stressed["mind"]["neuromodulators"]["ne_beta"] > stressed["mind"]["neuromodulators"]["ne_alpha2a"]
    assert stressed["mind"]["amygdala"]["cea_output"] > base_cea + 0.06


def test_08_god_mode_il6_pyrogen_induces_icu_fever_and_q10_acceleration() -> None:
    ctrl = ObservatorySimulationController(seed=42, deterministic_sensors=True, warmup_ticks=5)
    ctrl.execute_intervention("administer_il6_pyrogen", {"target_id": "ADAM-01", "dose": 0.85})
    for _ in range(45):
        frame = ctrl.step_once()

    icu = frame["organisms"]["ADAM-01"]["icu_vitals"]
    assert icu["setpoint_temp_c"] > 37.30
    assert icu["il6_cytokine"] > 0.30


def test_09_icu_somatic_resuscitation_restores_vitals_while_preserving_brain_trauma_engrams() -> None:
    ctrl = ObservatorySimulationController(seed=42, deterministic_sensors=True, warmup_ticks=5)
    ctrl.execute_intervention(
        "shutdown_brownout",
        {"target_id": "ADAM-01", "target_voltage": 0.10, "ramp_rate": 1.2, "ripple_amplitude": 0.24},
    )
    for _ in range(20):
        ctrl.execute_intervention("reprimand", {"target_id": "ADAM-01", "intensity": 2.6})
        ctrl.step_once()

    adam_brain = ctrl.arena.brains["ADAM-01"]
    engrams_before = len(adam_brain.hippocampus.engrams)
    bla_before = float(adam_brain.amygdala.w_bla_defensive.sum().item())
    assert engrams_before >= 1

    # Induce cerebral ATP collapse -> somatic death
    adam_org = ctrl.arena.world.organisms["ADAM-01"]
    adam_org.soma.metab.brain_atp = 0.0
    adam_org.soma.state[int(VisceralOrgan.BRAIN_ATP)] = 0.0
    dead_frame = ctrl.step_once()
    assert dead_frame["organisms"]["ADAM-01"]["pose"]["is_alive"] is False

    # Execute ICU resuscitation -> body lives again, trauma memories remain!
    ctrl.execute_intervention("icu_resuscitate", {"target_id": "ADAM-01"})
    revived_frame = ctrl.step_once()
    assert revived_frame["organisms"]["ADAM-01"]["pose"]["is_alive"] is True
    assert len(adam_brain.hippocampus.engrams) >= engrams_before
    assert float(adam_brain.amygdala.w_bla_defensive.sum().item()) >= bla_before * 0.98


def test_10_continuum_layer_switching_across_all_8_fields() -> None:
    ctrl = ObservatorySimulationController(seed=42, deterministic_sensors=True, warmup_ticks=4)
    for layer in CONTINUUM_LAYERS:
        res = ctrl.set_continuum_layer(layer)
        assert res["active_layer"] == layer
        frame = ctrl.get_latest_frame()
        grid = frame["arena"]["grid"]
        assert len(grid) == 24
        assert len(grid[0]) == 24


@pytest.fixture()
def live_server():
    ctrl = ObservatorySimulationController(seed=42, deterministic_sensors=True, warmup_ticks=6)
    ctrl.start_background_loop()
    httpd = LimbicObservatoryHTTPServer(("127.0.0.1", 0), ctrl)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    host, port = httpd.server_address
    base_url = f"http://{host}:{port}"
    try:
        yield base_url, ctrl
    finally:
        ctrl.stop_background_loop()
        httpd.shutdown()
        httpd.server_close()


def test_11_live_http_server_static_assets_and_rest_endpoints(live_server) -> None:
    base_url, _ = live_server

    html = urlopen(f"{base_url}/", timeout=3.0).read().decode("utf-8")
    assert "LIMBIC-0 // NEURO-fMRI" in html
    assert 'id="arena-canvas"' in html

    css = urlopen(f"{base_url}/static/styles.css", timeout=3.0).read().decode("utf-8")
    assert "--ch-adam" in css

    js = urlopen(f"{base_url}/static/app.js", timeout=3.0).read().decode("utf-8")
    assert "EventSource" in js

    state_json = json.loads(urlopen(f"{base_url}/api/state", timeout=3.0).read().decode("utf-8"))
    assert "ADAM-01" in state_json["organisms"]
    assert "ABEL-02" in state_json["organisms"]

    hist_json = json.loads(urlopen(f"{base_url}/api/history", timeout=3.0).read().decode("utf-8"))
    assert len(hist_json["timeseries"]["ADAM-01"]) >= 6

    audit_json = json.loads(urlopen(f"{base_url}/api/audit", timeout=3.0).read().decode("utf-8"))
    assert audit_json["verified"] is True


def test_12_live_http_post_interventions_and_sim_controls(live_server) -> None:
    base_url, _ = live_server

    def _post(path: str, data: dict) -> dict:
        req = Request(
            f"{base_url}{path}",
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return json.loads(urlopen(req, timeout=3.0).read().decode("utf-8"))

    r_pause = _post("/api/sim_control", {"action": "pause"})
    assert r_pause["paused"] is True

    t_before = r_pause["tick_count"]
    r_step = _post("/api/sim_control", {"action": "step_once"})
    assert r_step["tick_count"] == t_before + 1

    r_fav = _post("/api/intervene", {"command": "favoritism_drop", "params": {"target_id": "ABEL-02", "energy_amount": 0.9}})
    assert r_fav["ok"] is True

    r_vent = _post("/api/intervene", {"command": "set_ventilation", "params": {"openness": 0.3}})
    assert r_vent["ok"] is True

    audit = json.loads(urlopen(f"{base_url}/api/audit", timeout=3.0).read().decode("utf-8"))
    assert audit["verified"] is True

    _post("/api/sim_control", {"action": "resume"})


def test_13_live_sse_event_stream_delivers_real_time_frames(live_server) -> None:
    base_url, _ = live_server
    host_port = base_url.replace("http://", "")
    host, port_s = host_port.split(":")
    conn = http.client.HTTPConnection(host, int(port_s), timeout=4.0)
    try:
        conn.request("GET", "/api/stream")
        resp = conn.getresponse()
        assert resp.status == 200
        assert "text/event-stream" in resp.getheader("Content-Type", "")

        frames = []
        while len(frames) < 3:
            line = resp.fp.readline().decode("utf-8")
            if line.startswith("data: "):
                frames.append(json.loads(line[6:]))

        assert frames[2]["tick_count"] >= frames[0]["tick_count"] + 1
        assert len(frames[2]["manifold"]["coords_3d"]["ADAM-01"]) == 3
    finally:
        conn.close()


def test_14_thread_safety_under_concurrent_sse_and_rapid_god_commands(live_server) -> None:
    base_url, ctrl = live_server

    def _worker(i: int) -> bool:
        if i % 2 == 0:
            req = Request(
                f"{base_url}/api/intervene",
                data=json.dumps({"command": "praise", "params": {"target_id": "ABEL-02", "intensity": 0.9}}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            res = json.loads(urlopen(req, timeout=4.0).read().decode("utf-8"))
            return bool(res["ok"])
        else:
            st = json.loads(urlopen(f"{base_url}/api/state", timeout=4.0).read().decode("utf-8"))
            return "ADAM-01" in st["organisms"]

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_worker, range(24)))

    assert all(results)
    assert ctrl.verify_audit_ledger()["verified"] is True


def test_15_path_traversal_and_malformed_payload_hardening(live_server) -> None:
    base_url, _ = live_server

    with pytest.raises(HTTPError) as exc_info:
        urlopen(f"{base_url}/static/../../pyproject.toml", timeout=3.0)
    assert exc_info.value.code == 404

    bad_req = Request(
        f"{base_url}/api/intervene",
        data=json.dumps({"command": "nonexistent_cheat_command"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(HTTPError) as exc_bad:
        urlopen(bad_req, timeout=3.0)
    assert exc_bad.value.code == 400
