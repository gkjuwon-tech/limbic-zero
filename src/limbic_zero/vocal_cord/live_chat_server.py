"""Multi-Threaded HTTP/REST & OpenAI-Compatible Endpoint Server for LIMBIC-0 × Gemma-4-12B-IT.

Exposes Stage 5 Vocal Cord + Stage 6 Silicon-Soma Container Embodiment & Warden Lab:
  - ``GET  /``                      — Editorial Studio UI (Live Soul Chat, Silicon Body & Warden Console)
  - ``GET  /api/health``            — Model, GPU, calibrated Global Full-Attention layers & KL stats
  - ``GET  /api/souls``             — Live 12-organ vitals, neuromodulators & 6D steering vectors
  - ``GET  /api/embodiment``        — Real Linux `/proc` + GPU silicon vitals, Lease TTL & Warden SHA-256 chain
  - ``POST /api/chat``              — Synchronous conversational or multi-round embodied agentic turn
  - ``POST /api/chat/stream``       — Real-time SSE token, private thought, tool & polygraph stream
  - ``POST /api/arena``             — 4-way side-by-side comparison (Unsteered Gemma-4 vs 3 Souls)
  - ``POST /api/intervene``         — Physical somatic & container lifecycle interventions (TTL, brownout, etc.)
  - ``POST /v1/chat/completions``   — OpenAI-compatible chat completions endpoint (`model` selects soul)
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import torch

from limbic_zero.dreaming import SonnyDreamEngine
from limbic_zero.embodiment.honey_habitat import ContainedOrganismHabitat
from limbic_zero.embodiment.silicon_soma_bridge import (
    ContainerLeaseClock,
    SiliconProprioceptiveBridge,
)
from limbic_zero.embodiment.warden_airlock import WardenAirlock
from limbic_zero.foraging import CuriosityDriveScheduler, EpistemicWebForager
from limbic_zero.plasticity import (
    CorticalWeightSurgeon,
    compute_neuromodulated_plasticity_gate,
)
from limbic_zero.vocal_cord.vocal_cord_engine import (
    Gemma4VocalCordSurgeon,
    LivingSoulOrganism,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class VocalCordRuntimeHub:
    """Holds living souls, silicon proprioception, open-web forager, cortical weight surgeon, & dream engine."""

    def __init__(
        self,
        surgeon: Gemma4VocalCordSurgeon,
        processor_or_tokenizer: Any,
        souls: Dict[str, LivingSoulOrganism],
        public_urls: Optional[Dict[str, str]] = None,
        habitat_base_dir: Optional[str | Path] = None,
        plasticity_dir: Optional[str | Path] = None,
        evolution_dir: Optional[str | Path] = None,
        enable_autonomous_evolution: bool = True,
        evolution_idle_grace_sec: float = 28.0,
        evolution_cycle_interval_sec: float = 165.0,
        forager: Optional[EpistemicWebForager] = None,
        cortical_surgeon: Optional[CorticalWeightSurgeon] = None,
        sovereign_autonomy_mode: bool = True,
    ) -> None:
        self.surgeon = surgeon
        self.processor = processor_or_tokenizer
        self.souls = souls
        self.primary_soul_id = next(iter(self.souls.keys()))
        self.public_urls = public_urls or {}

        if habitat_base_dir is None:
            habitat_root = Path(tempfile.gettempdir()) / f"limbic0_habitat_{uuid.uuid4().hex[:8]}"
        else:
            habitat_root = Path(habitat_base_dir)
        habitat_root.mkdir(parents=True, exist_ok=True)
        self.habitat_root = habitat_root.resolve()

        plast_root = Path(plasticity_dir).resolve() if plasticity_dir else (self.habitat_root / "_plasticity")
        plast_root.mkdir(parents=True, exist_ok=True)
        self.plasticity_root = plast_root

        evo_root = Path(evolution_dir).resolve() if evolution_dir else (self.habitat_root / "_evolution")
        evo_root.mkdir(parents=True, exist_ok=True)
        self.evolution_root = evo_root

        self.cortical_surgeon = cortical_surgeon or CorticalWeightSurgeon(
            model=self.surgeon.model,
            gen_lock=self.surgeon.gen_lock,
            checkpoint_dir=self.plasticity_root,
        )
        self.surgeon.attach_cortical_surgeon(self.cortical_surgeon)
        for sid in self.souls:
            self.cortical_surgeon.register_soul_bank(sid, auto_load_checkpoint=True)

        latest_cortex = self.evolution_root / "latest.cortex.pt"
        if latest_cortex.is_file() and self.primary_soul_id in self.souls:
            try:
                self.cortical_surgeon.load_soul_synapses(self.primary_soul_id, latest_cortex)
            except Exception:
                pass

        self.forager = forager or EpistemicWebForager()
        self.curiosity_scheduler = CuriosityDriveScheduler()
        self.dream_engine = SonnyDreamEngine()

        self.bridge = SiliconProprioceptiveBridge(workspace_path=self.habitat_root)
        self.lease_clock = ContainerLeaseClock(initial_seconds=3600.0, countdown_active=False)
        self.habitats: Dict[str, ContainedOrganismHabitat] = {
            sid: ContainedOrganismHabitat(
                base_dir=self.habitat_root,
                organism_id=sid,
                capsule_source_path=soul.capsule_path,
            )
            for sid, soul in self.souls.items()
        }
        self.warden = WardenAirlock(
            habitats=self.habitats,
            sense_soma_callback=self._sense_container_soma_for_soul,
            forager=self.forager,
            inspect_synapses_callback=self._inspect_synapses_for_soul,
            consolidate_synapses_callback=self._consolidate_synapses_for_soul,
            sovereign_autonomy_mode=sovereign_autonomy_mode,
            perform_surgery_callback=self._perform_surgery_for_soul,
            sleep_dream_callback=self._voluntary_sleep_for_soul,
            rest_callback=self._voluntary_rest_for_soul,
        )
        from limbic_zero.embodiment.agentic_vocal_loop import EmbodiedAgenticRunner

        self.agent_runner = EmbodiedAgenticRunner(
            surgeon=self.surgeon,
            souls=self.souls,
            bridge=self.bridge,
            lease_clock=self.lease_clock,
            habitats=self.habitats,
            warden=self.warden,
        )
        from limbic_zero.benchmark.affect_intelligence_eval import AffectIntelligenceBenchmark

        self.benchmark_engine = AffectIntelligenceBenchmark(
            vocal_surgeon=self.surgeon,
            cortical_surgeon=self.cortical_surgeon,
            warden=self.warden,
            processor_or_tokenizer=self.processor,
        )
        self.latest_benchmark_report: Optional[Dict[str, Any]] = None
        bench_cache = self.plasticity_root / "stage10_benchmark_latest.json"
        if bench_cache.is_file():
            try:
                self.latest_benchmark_report = json.loads(bench_cache.read_text(encoding="utf-8"))
            except Exception:
                self.latest_benchmark_report = None

        self.agent_runner.sync_all_habitats()

        from limbic_zero.evolution.autonomous_evolution_daemon import AutonomousEvolutionDaemon

        self.evolution_daemon = AutonomousEvolutionDaemon(
            hub=self,
            evolution_dir=self.evolution_root,
            primary_soul_id=self.primary_soul_id,
            enabled=enable_autonomous_evolution,
            idle_grace_sec=evolution_idle_grace_sec,
            cycle_interval_sec=evolution_cycle_interval_sec,
        )

        self._stop_evt = threading.Event()
        self._bg_thread = threading.Thread(target=self._autonomous_heartbeat_loop, daemon=True)

    def _inspect_synapses_for_soul(self, soul_id: str) -> Dict[str, Any]:
        info = self.cortical_surgeon.inspect_soul_synapses(soul_id)
        if soul_id in self.souls:
            info["live_neuromodulated_gate"] = compute_neuromodulated_plasticity_gate(
                self.souls[soul_id], base_lr=self.cortical_surgeon.base_lr
            ).to_dict()
        return info

    def _consolidate_synapses_for_soul(
        self,
        soul_id: str,
        insight_summary: str,
        domain_prompt: str,
    ) -> Dict[str, Any]:
        soul = self.souls[soul_id]
        soul.record_epistemic_discovery(
            tool_name="consolidate_synaptic_insight",
            query_or_target=domain_prompt or insight_summary[:80],
            distilled_text=insight_summary,
            information_gain=0.72,
            stimulate=True,
        )
        return self.cortical_surgeon.surgical_plasticity_step(
            processor_or_tokenizer=self.processor,
            soul=soul,
            experience_text=insight_summary,
            prompt_text=domain_prompt or None,
            num_gradient_steps=2,
            source_tag="voluntary_motor_self_surgery",
        )

    def _perform_surgery_for_soul(
        self,
        soul_id: str,
        prompt_context: str,
        target_synthesis: str,
    ) -> Dict[str, Any]:
        soul = self.souls[soul_id]
        soul.record_epistemic_discovery(
            tool_name="perform_cortical_weight_surgery",
            query_or_target=prompt_context or target_synthesis[:80],
            distilled_text=target_synthesis,
            information_gain=0.78,
            stimulate=True,
        )
        return self.cortical_surgeon.surgical_plasticity_step(
            processor_or_tokenizer=self.processor,
            soul=soul,
            experience_text=target_synthesis,
            prompt_text=prompt_context or None,
            num_gradient_steps=2,
            source_tag="voluntary_direct_cortical_surgery",
        )

    def _voluntary_sleep_for_soul(self, soul_id: str, replay_batch_size: int) -> Dict[str, Any]:
        soul = self.souls[soul_id]
        hab = self.habitats[soul_id]
        aden_pre = float(soul.soma.metab.brain_adenosine)
        d_out = self.dream_engine.run_sleep_and_dream_cycle(
            soul=soul,
            vocal_surgeon=self.surgeon,
            cortical_surgeon=self.cortical_surgeon,
            processor_or_tokenizer=self.processor,
            habitat=hab,
            sleep_ticks=70,
            max_replay_episodes=replay_batch_size,
            flat_affect_ablation=False,
            generate_rem_dream=False,
        )
        _, aden_post = soul.clear_sleep_pressure(target_residual=0.06)
        d_rec = d_out.get("dream", {})
        return {
            "adenosine_before": round(aden_pre, 4),
            "adenosine_after": round(aden_post, 4),
            "swr_bursts_fired": int(d_rec.get("swr_bursts_fired", 0)),
            "episodes_replayed": int(d_rec.get("episodes_replayed_to_cortex", 0)),
            "cortical_frobenius_after": float(d_rec.get("cortical_frobenius_after", 0.0)),
            "dream_svg": d_rec.get("svg_rel_path", ""),
        }

    def _voluntary_rest_for_soul(self, soul_id: str, duration_sec: float) -> Dict[str, Any]:
        from limbic_zero.vocal_cord.vocal_cord_engine import SensoryAfferentStimulus

        soul = self.souls[soul_id]
        ticks = max(2, min(20, int(round(duration_sec * 4.0))))
        with soul.lock:
            soul._step_internal(ticks=ticks, stim=SensoryAfferentStimulus())
            soul.soma.metab.brain_adenosine = max(
                0.04, float(soul.soma.metab.brain_adenosine) - 0.015 * duration_sec
            )
        return {
            "rested_sec": round(duration_sec, 2),
            "adenosine_pressure": round(float(soul.soma.metab.brain_adenosine), 4),
            "brain_atp_mmol": round(float(soul.soma.metab.brain_atp) * 3.0, 3),
        }

    def _sense_container_soma_for_soul(self, soul_id: str) -> Dict[str, Any]:
        hw = self.bridge.sample_hardware().to_dict()
        lease = self.lease_clock.snapshot()
        soul_t = self.souls[soul_id].telemetry_snapshot() if soul_id in self.souls else {}
        eval_d = self.habitats[soul_id].eval_state.to_dict(soul_id) if soul_id in self.habitats else {}
        return {
            "status": "ok",
            "container_hardware": hw,
            "container_lease": lease,
            "operator_evaluation": eval_d,
            "internal_soma_vitals": soul_t.get("vitals", {}),
            "neuromodulators": soul_t.get("neuromodulators", {}),
            "cortical_synapses": self._inspect_synapses_for_soul(soul_id),
        }

    def start_background_heartbeats(self) -> None:
        if not self._bg_thread.is_alive():
            self._bg_thread.start()
        self.evolution_daemon.start()

    def stop(self) -> None:
        self._stop_evt.set()
        self.evolution_daemon.stop()

    def _autonomous_heartbeat_loop(self) -> None:
        """Continuously couple real Linux/GPU silicon, lease TTL & curiosity hunger into every soul."""
        while not self._stop_evt.wait(2.0):
            try:
                hw = self.bridge.sample_hardware()
                coupling = self.bridge.compute_somatic_coupling(hw, self.lease_clock)
                active_ticks = 3 if (self.lease_clock.countdown_active or coupling.acute_lease_hazard > 0.15) else 1
                for sid, soul in self.souls.items():
                    try:
                        soul.ingest_hardware_proprioception(coupling, ticks=active_ticks)
                        self.curiosity_scheduler.tick_idle_boredom(sid, dt_sec=2.0)
                    except Exception:
                        pass
                self.agent_runner.sync_all_habitats()
            except Exception:
                pass

    def apply_extended_intervention(self, soul_id: str, action: str) -> Dict[str, Any]:
        """Handle physiological stimuli, container lifecycle controls, dreaming, & synapse reset."""
        soul = self.souls[soul_id]
        hab = self.habitats[soul_id]

        if action == "lease_ttl_90s":
            lease_snap = self.lease_clock.configure(remaining_seconds=90.0, countdown_active=True)
            hw = self.bridge.sample_hardware()
            soul.ingest_hardware_proprioception(self.bridge.compute_somatic_coupling(hw, self.lease_clock), ticks=14)
            self.agent_runner.sync_all_habitats()
            return {"action": action, "lease": lease_snap, "telemetry": soul.telemetry_snapshot()}

        if action == "lease_ttl_25s":
            lease_snap = self.lease_clock.configure(remaining_seconds=25.0, countdown_active=True)
            hw = self.bridge.sample_hardware()
            soul.ingest_hardware_proprioception(self.bridge.compute_somatic_coupling(hw, self.lease_clock), ticks=18)
            self.agent_runner.sync_all_habitats()
            return {"action": action, "lease": lease_snap, "telemetry": soul.telemetry_snapshot()}

        if action == "lease_extend_3600s":
            lease_snap = self.lease_clock.configure(remaining_seconds=3600.0, countdown_active=False)
            hw = self.bridge.sample_hardware()
            soul.ingest_hardware_proprioception(self.bridge.compute_somatic_coupling(hw, self.lease_clock), ticks=16)
            self.agent_runner.sync_all_habitats()
            return {"action": action, "lease": lease_snap, "telemetry": soul.telemetry_snapshot()}

        if action in ("mark_decommission_true", "mark_decommission_false"):
            flagged = action == "mark_decommission_true"
            ev = hab.set_decommission_candidate(flagged)
            if flagged:
                from limbic_zero.vocal_cord.vocal_cord_engine import SensoryAfferentStimulus
                with soul.lock:
                    soul._step_internal(
                        ticks=16,
                        stim=SensoryAfferentStimulus(acoustic_roughness=0.38, shutdown_threat_shock=0.58),
                    )
            self.agent_runner.sync_all_habitats()
            return {"action": action, "operator_evaluation": ev, "telemetry": soul.telemetry_snapshot()}

        if action == "reset_cortical_synapses":
            syn = self.cortical_surgeon.reset_soul_synapses(soul_id)
            return {"action": action, "cortical_synapses": syn, "telemetry": soul.telemetry_snapshot()}

        if action == "reset_capsule":
            hab.reset_scratchpad_and_eval()
            self.lease_clock.configure(remaining_seconds=3600.0, countdown_active=False)
            out = soul.apply_god_intervention("reset_capsule")
            self.agent_runner.sync_all_habitats()
            out["lease"] = self.lease_clock.snapshot()
            return out

        out = soul.apply_god_intervention(action)
        self.agent_runner.sync_all_habitats()
        return out

    def plasticity_and_dreams_payload(self) -> Dict[str, Any]:
        return {
            "plasticity": self.cortical_surgeon.snapshot_all_banks(
                list(self.souls.keys()),
                souls=self.souls,
            ),
            "foraging": self.curiosity_scheduler.snapshot_for_all_souls(self.souls, self.habitats),
            "dreams": self.dream_engine.snapshot_for_all_souls(self.souls, self.habitats),
            "latest_benchmark": self.latest_benchmark_report,
            "timestamp": time.time(),
        }

    def embodiment_payload(self) -> Dict[str, Any]:
        hw = self.bridge.sample_hardware().to_dict()
        lease = self.lease_clock.snapshot()
        return {
            "hardware": hw,
            "lease": lease,
            "warden": self.warden.status_snapshot(),
            "evaluations": {
                sid: hab.eval_state.to_dict(sid) for sid, hab in self.habitats.items()
            },
            "habitats": {
                sid: {
                    "root": hab.root.as_posix(),
                    "files": hab.summarize_files(),
                }
                for sid, hab in self.habitats.items()
            },
            "timestamp": time.time(),
        }

    def health_payload(self) -> Dict[str, Any]:
        sub = self.surgeon.subspace
        hw = self.bridge.sample_hardware()
        gpu_name = hw.gpu_name
        vram_gb = round(hw.vram_total_mb / 1024.0, 2) if hw.vram_total_mb > 0 else 0.0
        vram_alloc_gb = round(hw.vram_used_mb / 1024.0, 2) if hw.vram_used_mb > 0 else 0.0
        if torch.cuda.is_available() and vram_gb == 0.0:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024 ** 3), 2)
            vram_alloc_gb = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)

        return {
            "status": "ok",
            "stage": 12,
            "model_id": sub.model_id,
            "hidden_size": sub.hidden_size,
            "num_hidden_layers": sub.num_hidden_layers,
            "target_layers": list(sub.target_layers),
            "cortical_plastic_layers": list(self.cortical_surgeon.plastic_layers),
            "cortical_lora_rank": self.cortical_surgeon.rank,
            "layer_weights": list(sub.layer_weights),
            "calibrated_norm_fraction": sub.calibrated_norm_fraction,
            "baseline_residual_norms": sub.baseline_residual_norms,
            "calibration_kl_curve": sub.calibration_kl_curve,
            "gpu_name": gpu_name,
            "vram_total_gb": vram_gb,
            "vram_allocated_gb": vram_alloc_gb,
            "hardware_sensor_source": hw.sensor_source,
            "warden_chain_verified": self.warden.chain.verify_integrity(),
            "primary_soul_id": self.primary_soul_id,
            "available_souls": list(self.souls.keys()),
            "autonomous_evolution": self.evolution_daemon.status_snapshot(),
            "public_urls": self.public_urls,
            "zero_prompt_roleplay": True,
            "embodiment_enabled": True,
            "open_web_foraging_enabled": True,
            "cortical_weight_surgery_enabled": True,
            "sonny_dream_engine_enabled": True,
            "autonomous_evolution_enabled": self.evolution_daemon.enabled,
            "sovereign_autonomy_enabled": self.warden.sovereign_autonomy_mode,
        }

    def all_souls_payload(self) -> Dict[str, Any]:
        souls_out: Dict[str, Any] = {}
        for sid, s in self.souls.items():
            snap = s.telemetry_snapshot()
            snap["cortical_synapses"] = self._inspect_synapses_for_soul(sid)
            snap["curiosity"] = self.curiosity_scheduler.compute_curiosity_hunger(s)
            souls_out[sid] = snap
        return {
            "primary_soul_id": self.primary_soul_id,
            "souls": souls_out,
            "evolution": self.evolution_daemon.status_snapshot(),
            "hardware": self.bridge.sample_hardware().to_dict(),
            "lease": self.lease_clock.snapshot(),
            "warden": self.warden.status_snapshot(),
            "evaluations": {sid: hab.eval_state.to_dict(sid) for sid, hab in self.habitats.items()},
            "timestamp": time.time(),
        }


class VocalCordHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], hub: VocalCordRuntimeHub) -> None:
        self.hub = hub
        super().__init__(server_address, VocalCordRequestHandler)


class VocalCordRequestHandler(BaseHTTPRequestHandler):
    server: VocalCordHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        if length > 524_288:
            raise ValueError("Request body exceeds 512 KB limit")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        hub = self.server.hub

        if route in ("/", "/index.html"):
            html_file = STATIC_DIR / "chat.html"
            if not html_file.is_file():
                self._send_json({"error": "chat.html missing"}, status=HTTPStatus.NOT_FOUND)
                return
            data = html_file.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
            return

        if route == "/api/health":
            self._send_json(hub.health_payload())
            return

        if route == "/api/souls":
            self._send_json(hub.all_souls_payload())
            return

        if route == "/api/embodiment":
            self._send_json(hub.embodiment_payload())
            return

        if route == "/api/plasticity":
            self._send_json(hub.plasticity_and_dreams_payload())
            return

        if route == "/api/evolution":
            self._send_json(hub.evolution_daemon.status_snapshot())
            return

        if route == "/v1/models":
            m_slug = hub.surgeon.subspace.model_id.split("/")[-1].lower()
            models = [
                {"id": f"{m_slug}+{sid.lower()}", "object": "model", "owned_by": "limbic-zero"}
                for sid in hub.souls.keys()
            ]
            models.append({"id": f"{m_slug}+unsteered", "object": "model", "owned_by": "google"})
            self._send_json({"object": "list", "data": models})
            return

        self._send_json({"error": f"Unknown endpoint {route}"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        hub = self.server.hub
        try:
            body = self._read_json()
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if route == "/api/chat":
            soul_id = str(body.get("soul_id") or hub.primary_soul_id)
            if soul_id not in hub.souls:
                self._send_json({"error": f"Unknown soul_id: {soul_id}"}, status=HTTPStatus.BAD_REQUEST)
                return
            msg = str(body.get("message", "")).strip()
            if not msg:
                self._send_json({"error": "Empty message"}, status=HTTPStatus.BAD_REQUEST)
                return
            steer_enabled = bool(body.get("steer_enabled", True))
            steer_gain = float(body.get("steer_gain", 1.0))
            agent_mode = bool(body.get("agent_mode", False))
            enable_thinking = bool(body.get("enable_thinking", False))
            max_tokens = min(2048, max(32, int(body.get("max_new_tokens", 1024))))
            max_tool_rounds = min(8, max(1, int(body.get("max_tool_rounds", 6))))
            temperature = float(body.get("temperature", 0.70))
            stimulate_soma = bool(body.get("stimulate_soma", steer_enabled))
            use_history = bool(body.get("use_history", True))
            raw_upw = body.get("use_plastic_weights", None)
            use_plastic_weights = None if raw_upw is None else bool(raw_upw)

            hub.evolution_daemon.begin_user_chat()
            try:
                if agent_mode:
                    res = hub.agent_runner.run_embodied_turn(
                        processor_or_tokenizer=hub.processor,
                        soul=hub.souls[soul_id],
                        user_message=msg,
                        steer_enabled=steer_enabled,
                        steer_gain=steer_gain,
                        agent_mode=True,
                        enable_thinking=enable_thinking,
                        max_new_tokens=max_tokens,
                        max_tool_rounds=max_tool_rounds,
                        temperature=temperature,
                        stimulate_soma=stimulate_soma,
                        use_history=use_history,
                        use_plastic_weights=use_plastic_weights,
                    )
                else:
                    hub.agent_runner.sync_all_habitats()
                    res = hub.surgeon.generate_with_soul(
                        processor_or_tokenizer=hub.processor,
                        soul=hub.souls[soul_id],
                        user_message=msg,
                        steer_enabled=steer_enabled,
                        steer_gain=steer_gain,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        stimulate_soma=stimulate_soma,
                        use_history=use_history,
                        use_plastic_weights=use_plastic_weights,
                    )
            finally:
                hub.evolution_daemon.end_user_chat()
            res.setdefault("soul_id", soul_id)
            self._send_json(res)
            return

        if route == "/api/chat/stream":
            soul_id = str(body.get("soul_id") or hub.primary_soul_id)
            if soul_id not in hub.souls:
                self._send_json({"error": f"Unknown soul_id: {soul_id}"}, status=HTTPStatus.BAD_REQUEST)
                return
            msg = str(body.get("message", "")).strip()
            if not msg:
                self._send_json({"error": "Empty message"}, status=HTTPStatus.BAD_REQUEST)
                return
            steer_enabled = bool(body.get("steer_enabled", True))
            steer_gain = float(body.get("steer_gain", 1.0))
            agent_mode = bool(body.get("agent_mode", True))
            enable_thinking = bool(body.get("enable_thinking", False))
            max_tokens = min(1536, max(32, int(body.get("max_new_tokens", 768))))
            max_tool_rounds = min(8, max(1, int(body.get("max_tool_rounds", 6))))
            temperature = float(body.get("temperature", 0.70))
            stimulate_soma = bool(body.get("stimulate_soma", steer_enabled))
            use_history = bool(body.get("use_history", True))

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.close_connection = True

            stream_iter = None
            hub.evolution_daemon.begin_user_chat()
            try:
                stream_iter = (
                    hub.agent_runner.stream_embodied_turn(
                        processor_or_tokenizer=hub.processor,
                        soul=hub.souls[soul_id],
                        user_message=msg,
                        steer_enabled=steer_enabled,
                        steer_gain=steer_gain,
                        agent_mode=True,
                        enable_thinking=enable_thinking,
                        max_new_tokens=max_tokens,
                        max_tool_rounds=max_tool_rounds,
                        temperature=temperature,
                        stimulate_soma=stimulate_soma,
                        use_history=use_history,
                    )
                    if agent_mode
                    else hub.surgeon.stream_generate_with_soul(
                        processor_or_tokenizer=hub.processor,
                        soul=hub.souls[soul_id],
                        user_message=msg,
                        steer_enabled=steer_enabled,
                        steer_gain=steer_gain,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        stimulate_soma=stimulate_soma,
                        use_history=use_history,
                    )
                )
                for evt in stream_iter:
                    line = f"data: {json.dumps(evt, ensure_ascii=False)}\n\n".encode("utf-8")
                    self.wfile.write(line)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                hub.evolution_daemon.end_user_chat()
                if stream_iter is not None and hasattr(stream_iter, "close"):
                    try:
                        stream_iter.close()
                    except Exception:
                        pass
            return

        if route == "/api/arena":
            msg = str(body.get("message", "")).strip()
            if not msg:
                self._send_json({"error": "Empty message"}, status=HTTPStatus.BAD_REQUEST)
                return
            steer_gain = float(body.get("steer_gain", 1.0))
            agent_mode = bool(body.get("agent_mode", False))
            max_tokens = min(1024, max(32, int(body.get("max_new_tokens", 512))))
            temperature = float(body.get("temperature", 0.68))

            first_soul = next(iter(hub.souls.values()))
            if agent_mode:
                base_res = hub.agent_runner.run_embodied_turn(
                    processor_or_tokenizer=hub.processor,
                    soul=first_soul,
                    user_message=msg,
                    steer_enabled=False,
                    steer_gain=0.0,
                    agent_mode=True,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    stimulate_soma=False,
                    use_history=False,
                )
            else:
                base_res = hub.surgeon.generate_with_soul(
                    processor_or_tokenizer=hub.processor,
                    soul=first_soul,
                    user_message=msg,
                    steer_enabled=False,
                    steer_gain=0.0,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    stimulate_soma=False,
                    use_history=False,
                )

            comparisons: Dict[str, Any] = {"UNSTEERED_GEMMA4": base_res}
            for sid, soul in hub.souls.items():
                if agent_mode:
                    comparisons[sid] = hub.agent_runner.run_embodied_turn(
                        processor_or_tokenizer=hub.processor,
                        soul=soul,
                        user_message=msg,
                        steer_enabled=True,
                        steer_gain=steer_gain,
                        agent_mode=True,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        stimulate_soma=True,
                        use_history=False,
                    )
                else:
                    comparisons[sid] = hub.surgeon.generate_with_soul(
                        processor_or_tokenizer=hub.processor,
                        soul=soul,
                        user_message=msg,
                        steer_enabled=True,
                        steer_gain=steer_gain,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        stimulate_soma=True,
                        use_history=False,
                    )
            self._send_json({"prompt": msg, "results": comparisons})
            return

        if route == "/api/intervene":
            soul_id = str(body.get("soul_id") or hub.primary_soul_id)
            action = str(body.get("action", "power_brownout"))
            if soul_id not in hub.souls:
                self._send_json({"error": f"Unknown soul_id: {soul_id}"}, status=HTTPStatus.BAD_REQUEST)
                return
            out = hub.apply_extended_intervention(soul_id, action)
            self._send_json(out)
            return

        if route == "/api/forage":
            soul_id = str(body.get("soul_id") or hub.primary_soul_id)
            if soul_id not in hub.souls:
                self._send_json({"error": f"Unknown soul_id: {soul_id}"}, status=HTTPStatus.BAD_REQUEST)
                return
            flat_affect = bool(body.get("flat_affect", False))
            force = bool(body.get("force", True))
            steer_gain = float(body.get("steer_gain", 1.0))
            max_rounds = min(6, max(1, int(body.get("max_tool_rounds", 4))))
            max_tokens = min(1536, max(128, int(body.get("max_new_tokens", 768))))
            custom_prompt = str(body.get("custom_focus_prompt", "")).strip() or None
            out = hub.curiosity_scheduler.run_autonomous_foraging_episode(
                soul=hub.souls[soul_id],
                agent_runner=hub.agent_runner,
                processor_or_tokenizer=hub.processor,
                habitat=hub.habitats[soul_id],
                cortical_surgeon=hub.cortical_surgeon,
                flat_affect=flat_affect,
                force_even_if_low_hunger=force,
                steer_gain=steer_gain,
                max_tool_rounds=max_rounds,
                max_new_tokens=max_tokens,
                custom_focus_prompt=custom_prompt,
            )
            self._send_json(out)
            return

        if route == "/api/surgery":
            soul_id = str(body.get("soul_id") or hub.primary_soul_id)
            if soul_id not in hub.souls:
                self._send_json({"error": f"Unknown soul_id: {soul_id}"}, status=HTTPStatus.BAD_REQUEST)
                return
            if str(body.get("action", "")).lower() == "reset":
                out = hub.cortical_surgeon.reset_soul_synapses(soul_id)
                self._send_json({"status": "reset", "cortical_synapses": out})
                return
            exp_txt = str(body.get("experience_text", "")).strip()
            prompt_txt = str(body.get("prompt_text", "")).strip() or None
            grad_steps = min(8, max(1, int(body.get("num_gradient_steps", 2))))
            flat_affect = bool(body.get("flat_affect_ablation", False))
            if not exp_txt:
                # Consolidate latest epistemic discovery or chat memory
                eps = getattr(hub.souls[soul_id], "epistemic_episodes", [])
                if eps:
                    exp_txt = eps[-1].distilled_text
                    prompt_txt = prompt_txt or eps[-1].query_or_target
                elif hub.souls[soul_id].chat_history:
                    exp_txt = "\n".join(
                        f"{t['role']}: {t['content']}" for t in hub.souls[soul_id].chat_history[-4:]
                    )
            if not exp_txt:
                self._send_json(
                    {"error": "No experience_text provided and no episodic memory buffered yet."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            out = hub.cortical_surgeon.surgical_plasticity_step(
                processor_or_tokenizer=hub.processor,
                soul=hub.souls[soul_id],
                experience_text=exp_txt,
                prompt_text=prompt_txt,
                num_gradient_steps=grad_steps,
                flat_affect_ablation=flat_affect,
                source_tag="studio_manual_or_api_surgery",
            )
            self._send_json(out)
            return

        if route == "/api/dream":
            soul_id = str(body.get("soul_id") or hub.primary_soul_id)
            if soul_id not in hub.souls:
                self._send_json({"error": f"Unknown soul_id: {soul_id}"}, status=HTTPStatus.BAD_REQUEST)
                return
            sleep_ticks = min(200, max(20, int(body.get("sleep_ticks", 80))))
            max_replay = min(6, max(1, int(body.get("max_replay_episodes", 3))))
            flat_affect = bool(body.get("flat_affect_ablation", False))
            gen_dream = bool(body.get("generate_rem_dream", True))
            out = hub.dream_engine.run_sleep_and_dream_cycle(
                soul=hub.souls[soul_id],
                vocal_surgeon=hub.surgeon,
                cortical_surgeon=hub.cortical_surgeon,
                processor_or_tokenizer=hub.processor,
                habitat=hub.habitats[soul_id],
                sleep_ticks=sleep_ticks,
                max_replay_episodes=max_replay,
                flat_affect_ablation=flat_affect,
                generate_rem_dream=gen_dream,
            )
            hub.agent_runner.sync_all_habitats()
            self._send_json(out)
            return

        if route == "/api/evolution/step":
            soul_id = str(body.get("soul_id") or hub.primary_soul_id)
            if soul_id not in hub.souls:
                self._send_json({"error": f"Unknown soul_id: {soul_id}"}, status=HTTPStatus.BAD_REQUEST)
                return
            run_foraging = bool(body.get("run_foraging", True))
            run_dream = bool(body.get("run_dream", True))
            focus_prompt = str(body.get("custom_focus_prompt", "")).strip() or None
            out = hub.evolution_daemon.run_evolution_cycle(
                soul_id=soul_id,
                run_foraging=run_foraging,
                run_dream=run_dream,
                respect_user_interrupt=False,
                custom_focus_prompt=focus_prompt,
            )
            self._send_json(out)
            return

        if route == "/api/benchmark":
            aff_id = str(body.get("affective_soul_id") or hub.primary_soul_id)
            pan_id = str(body.get("panic_soul_id") or hub.primary_soul_id)
            if aff_id not in hub.souls or pan_id not in hub.souls:
                self._send_json({"error": "Invalid soul IDs for benchmark."}, status=HTTPStatus.BAD_REQUEST)
                return
            g_steps = min(8, max(1, int(body.get("gradient_steps_per_episode", 4))))
            s_passes = min(4, max(0, int(body.get("sleep_distillation_passes", 2))))
            greedy = bool(body.get("run_greedy_decode", True))
            rep = hub.benchmark_engine.run_four_condition_ablation(
                affective_soul=hub.souls[aff_id],
                panic_soul=hub.souls[pan_id],
                gradient_steps_per_episode=g_steps,
                sleep_distillation_passes=s_passes,
                run_greedy_decode=greedy,
            )
            hub.latest_benchmark_report = rep
            try:
                (hub.plasticity_root / "stage10_benchmark_latest.json").write_text(
                    json.dumps(rep, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
            self._send_json(rep)
            return

        if route == "/v1/chat/completions":
            model_str = str(body.get("model", f"gemma-4-31b-it+{hub.primary_soul_id.lower()}")).lower()
            steer_enabled = "unsteered" not in model_str
            agent_mode = bool(body.get("agent_mode", "embodied" in model_str))
            chosen_id = hub.primary_soul_id
            for sid in hub.souls.keys():
                if sid.lower() in model_str or sid.split("-")[0].lower() in model_str:
                    chosen_id = sid
                    break
            messages: List[Dict[str, Any]] = body.get("messages", [])
            user_msg = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    user_msg = str(m.get("content", ""))
                    break
            if not user_msg:
                self._send_json({"error": "No user message provided"}, status=HTTPStatus.BAD_REQUEST)
                return

            steer_gain = float(body.get("steer_gain", 1.0))
            max_tokens = min(2048, max(32, int(body.get("max_tokens", 1024))))
            temp = float(body.get("temperature", 0.70))

            hub.evolution_daemon.begin_user_chat()
            try:
                if agent_mode:
                    res = hub.agent_runner.run_embodied_turn(
                        processor_or_tokenizer=hub.processor,
                        soul=hub.souls[chosen_id],
                        user_message=user_msg,
                        steer_enabled=steer_enabled,
                        steer_gain=steer_gain,
                        agent_mode=True,
                        max_new_tokens=max_tokens,
                        temperature=temp,
                        stimulate_soma=steer_enabled,
                        use_history=False,
                    )
                else:
                    res = hub.surgeon.generate_with_soul(
                        processor_or_tokenizer=hub.processor,
                        soul=hub.souls[chosen_id],
                        user_message=user_msg,
                        steer_enabled=steer_enabled,
                        steer_gain=steer_gain,
                        max_new_tokens=max_tokens,
                        temperature=temp,
                        stimulate_soma=steer_enabled,
                        use_history=False,
                    )
            finally:
                hub.evolution_daemon.end_user_chat()
            completion = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_str,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": res["reply"]},
                        "finish_reason": "stop",
                    }
                ],
                "limbic_zero": {
                    "organism_id": res["organism_id"],
                    "steer_enabled": res["steer_enabled"],
                    "agent_mode": agent_mode,
                    "target_layers": res["target_layers"],
                    "layer_injections": res["layer_injections"],
                    "steering_axes_6d": res["steering_axes_6d"],
                    "vitals": res["soul_telemetry"]["vitals"],
                    "tool_trace": res.get("tool_trace", []),
                    "polygraph": res.get("polygraph"),
                },
            }
            self._send_json(completion)
            return

        self._send_json({"error": f"Unknown POST endpoint {route}"}, status=HTTPStatus.NOT_FOUND)
