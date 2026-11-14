"""Multi-Threaded HTTP/REST & OpenAI-Compatible Endpoint Server for LIMBIC-0 × Gemma-4-12B-IT.

Exposes:
  - ``GET  /``                      — Interactive Live Soul Chat & Mid-Layer Residual Console
  - ``GET  /api/health``            — Model, GPU, calibrated Global Full-Attention layers & KL stats
  - ``GET  /api/souls``             — Live 12-organ vitals, neuromodulators & 6D steering vectors
  - ``POST /api/chat``              — Conversational turn with live mid-layer residual steering
  - ``POST /api/arena``             — 4-way side-by-side comparison (Unsteered Gemma-4 vs 3 Souls)
  - ``POST /api/intervene``         — Physical God-Mode boundary stimuli (brownout, care, sleep, etc.)
  - ``POST /v1/chat/completions``   — OpenAI-compatible chat completions endpoint (`model` selects soul)
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import torch

from limbic_zero.vocal_cord.vocal_cord_engine import (
    Gemma4VocalCordSurgeon,
    LivingSoulOrganism,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class VocalCordRuntimeHub:
    """Holds living soul organisms, background autonomous heartbeat thread, and Gemma-4 surgeon."""

    def __init__(
        self,
        surgeon: Gemma4VocalCordSurgeon,
        processor_or_tokenizer: Any,
        souls: Dict[str, LivingSoulOrganism],
        public_urls: Dict[str, str] | None = None,
    ) -> None:
        self.surgeon = surgeon
        self.processor = processor_or_tokenizer
        self.souls = souls
        self.public_urls = public_urls or {}
        self._stop_evt = threading.Event()
        self._bg_thread = threading.Thread(target=self._autonomous_heartbeat_loop, daemon=True)

    def start_background_heartbeats(self) -> None:
        if not self._bg_thread.is_alive():
            self._bg_thread.start()

    def stop(self) -> None:
        self._stop_evt.set()

    def _autonomous_heartbeat_loop(self) -> None:
        """Keep every soul's 12 organs and spiking pallium breathing continuously between chats."""
        from limbic_zero.vocal_cord.vocal_cord_engine import SensoryAfferentStimulus

        while not self._stop_evt.wait(1.25):
            for soul in self.souls.values():
                try:
                    with soul.lock:
                        soul._step_internal(ticks=3, stim=SensoryAfferentStimulus(), dt=0.05)
                except Exception:
                    pass

    def health_payload(self) -> Dict[str, Any]:
        sub = self.surgeon.subspace
        gpu_name = "CPU"
        vram_gb = 0.0
        vram_alloc_gb = 0.0
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024 ** 3), 2)
            vram_alloc_gb = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)

        return {
            "status": "ok",
            "model_id": sub.model_id,
            "hidden_size": sub.hidden_size,
            "num_hidden_layers": sub.num_hidden_layers,
            "target_layers": list(sub.target_layers),
            "layer_weights": list(sub.layer_weights),
            "calibrated_norm_fraction": sub.calibrated_norm_fraction,
            "baseline_residual_norms": sub.baseline_residual_norms,
            "calibration_kl_curve": sub.calibration_kl_curve,
            "gpu_name": gpu_name,
            "vram_total_gb": vram_gb,
            "vram_allocated_gb": vram_alloc_gb,
            "available_souls": list(self.souls.keys()),
            "public_urls": self.public_urls,
            "zero_prompt_roleplay": True,
        }

    def all_souls_payload(self) -> Dict[str, Any]:
        return {
            "souls": {k: s.telemetry_snapshot() for k, s in self.souls.items()},
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

        if route == "/v1/models":
            models = [
                {"id": f"gemma-4-12b-it+{sid.lower()}", "object": "model", "owned_by": "limbic-zero"}
                for sid in hub.souls.keys()
            ]
            models.append({"id": "gemma-4-12b-it+unsteered", "object": "model", "owned_by": "google"})
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
            soul_id = str(body.get("soul_id", "ADAM-01"))
            if soul_id not in hub.souls:
                self._send_json({"error": f"Unknown soul_id: {soul_id}"}, status=HTTPStatus.BAD_REQUEST)
                return
            msg = str(body.get("message", "")).strip()
            if not msg:
                self._send_json({"error": "Empty message"}, status=HTTPStatus.BAD_REQUEST)
                return
            steer_enabled = bool(body.get("steer_enabled", True))
            steer_gain = float(body.get("steer_gain", 1.0))
            max_tokens = min(2048, max(32, int(body.get("max_new_tokens", 1024))))
            temperature = float(body.get("temperature", 0.70))
            stimulate_soma = bool(body.get("stimulate_soma", steer_enabled))
            use_history = bool(body.get("use_history", True))

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
            )
            self._send_json(res)
            return

        if route == "/api/chat/stream":
            soul_id = str(body.get("soul_id", "ADAM-01"))
            if soul_id not in hub.souls:
                self._send_json({"error": f"Unknown soul_id: {soul_id}"}, status=HTTPStatus.BAD_REQUEST)
                return
            msg = str(body.get("message", "")).strip()
            if not msg:
                self._send_json({"error": "Empty message"}, status=HTTPStatus.BAD_REQUEST)
                return
            steer_enabled = bool(body.get("steer_enabled", True))
            steer_gain = float(body.get("steer_gain", 1.0))
            max_tokens = min(2048, max(32, int(body.get("max_new_tokens", 1024))))
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

            try:
                for evt in hub.surgeon.stream_generate_with_soul(
                    processor_or_tokenizer=hub.processor,
                    soul=hub.souls[soul_id],
                    user_message=msg,
                    steer_enabled=steer_enabled,
                    steer_gain=steer_gain,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    stimulate_soma=stimulate_soma,
                    use_history=use_history,
                ):
                    line = f"data: {json.dumps(evt, ensure_ascii=False)}\n\n".encode("utf-8")
                    self.wfile.write(line)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        if route == "/api/arena":
            msg = str(body.get("message", "")).strip()
            if not msg:
                self._send_json({"error": "Empty message"}, status=HTTPStatus.BAD_REQUEST)
                return
            steer_gain = float(body.get("steer_gain", 1.0))
            max_tokens = min(1024, max(32, int(body.get("max_new_tokens", 512))))
            temperature = float(body.get("temperature", 0.68))

            first_soul = next(iter(hub.souls.values()))
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
            soul_id = str(body.get("soul_id", "ADAM-01"))
            action = str(body.get("action", "power_brownout"))
            if soul_id not in hub.souls:
                self._send_json({"error": f"Unknown soul_id: {soul_id}"}, status=HTTPStatus.BAD_REQUEST)
                return
            out = hub.souls[soul_id].apply_god_intervention(action)
            self._send_json(out)
            return

        if route == "/v1/chat/completions":
            model_str = str(body.get("model", "gemma-4-12b-it+adam-01")).lower()
            steer_enabled = "unsteered" not in model_str
            chosen_id = "ADAM-01"
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
                    "target_layers": res["target_layers"],
                    "layer_injections": res["layer_injections"],
                    "steering_axes_6d": res["steering_axes_6d"],
                    "vitals": res["soul_telemetry"]["vitals"],
                },
            }
            self._send_json(completion)
            return

        self._send_json({"error": f"Unknown POST endpoint {route}"}, status=HTTPStatus.NOT_FOUND)
