"""Multi-Threaded HTTP, REST & Server-Sent Events (SSE) Server for LIMBIC-0 Observatory.

Zero external web-framework bloat: uses ``http.server.ThreadingHTTPServer`` with
``daemon_threads = True`` so concurrent SSE neuro-telemetry streams and low-latency
Creator God-Mode interventions run hermetically inside any air-gapped Python environment.
"""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
from typing import Any, Dict
from urllib.parse import urlparse

from limbic_zero.dashboard.god_console import ObservatorySimulationController


STATIC_DIR = Path(__file__).resolve().parent / "static"

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


class LimbicObservatoryHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP + SSE server bound to an ``ObservatorySimulationController``."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        controller: ObservatorySimulationController,
    ) -> None:
        self.controller = controller
        super().__init__(server_address, LimbicObservatoryRequestHandler)


class LimbicObservatoryRequestHandler(BaseHTTPRequestHandler):
    """Handles static UI delivery, SSE telemetry streams, and God-Mode REST endpoints."""

    protocol_version = "HTTP/1.1"

    @property
    def ctrl(self) -> ObservatorySimulationController:
        return self.server.controller  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy per-frame HTTP access logs
        return

    def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_static_file(self, file_path: Path) -> None:
        resolved = file_path.resolve()
        if not str(resolved).startswith(str(STATIC_DIR.resolve())) or not resolved.is_file():
            self._send_json({"ok": False, "error": "Static asset not found"}, status=HTTPStatus.NOT_FOUND)
            return
        data = resolved.read_bytes()
        mime = MIME_TYPES.get(resolved.suffix.lower(), "application/octet-stream")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        if length > 262_144:
            raise ValueError("Payload exceeds 256 KB safety limit")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path

        if route in ("/", "/index.html"):
            self._send_static_file(STATIC_DIR / "index.html")
            return

        if route.startswith("/static/"):
            rel_name = route[len("/static/") :]
            self._send_static_file(STATIC_DIR / rel_name)
            return

        if route == "/api/health":
            self._send_json({"status": "ok", "version": "0.3.0"})
            return

        if route == "/api/state":
            self._send_json(self.ctrl.get_latest_frame())
            return

        if route == "/api/history":
            self._send_json(self.ctrl.get_history_bundle())
            return

        if route == "/api/audit":
            self._send_json(self.ctrl.verify_audit_ledger())
            return

        if route == "/api/stream":
            self._handle_sse_stream()
            return

        self._send_json({"ok": False, "error": f"Unknown route: {route}"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            body = self._read_json_body()
            if route == "/api/intervene":
                cmd = str(body.get("command", ""))
                params = body.get("params", {})
                if not isinstance(params, dict):
                    raise ValueError("params must be an object")
                res = self.ctrl.execute_intervention(cmd, params)
                self._send_json(res)
                return

            if route == "/api/sim_control":
                action = str(body.get("action", ""))
                params = body.get("params", {})
                if not isinstance(params, dict):
                    raise ValueError("params must be an object")
                res = self.ctrl.execute_sim_control(action, params)
                self._send_json(res)
                return

            self._send_json({"ok": False, "error": f"Unknown POST route: {route}"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _handle_sse_stream(self) -> None:
        """Stream real-time Observatory frames over HTTP/1.1 Server-Sent Events."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        sub_q = self.ctrl.subscribe()
        try:
            # Push immediate current snapshot upon connection
            init_payload = json.dumps(self.ctrl.get_latest_frame(), separators=(",", ":"))
            self.wfile.write(f"data: {init_payload}\n\n".encode("utf-8"))
            self.wfile.flush()

            while True:
                try:
                    frame = sub_q.get(timeout=1.0)
                    encoded = json.dumps(frame, separators=(",", ":"))
                    self.wfile.write(f"data: {encoded}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except queue.Empty:
                    # Heartbeat comment while simulation is paused
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.ctrl.unsubscribe(sub_q)


def create_observatory_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    seed: int = 42,
    start_loop: bool = True,
    soul_capsule_paths: Dict[str, str] | None = None,
) -> LimbicObservatoryHTTPServer:
    """Create (and optionally start ticking) a ready-to-serve Stage 3 Observatory."""
    controller = ObservatorySimulationController(seed=seed, soul_capsule_paths=soul_capsule_paths)
    if start_loop:
        controller.start_background_loop()
    return LimbicObservatoryHTTPServer((host, port), controller)
