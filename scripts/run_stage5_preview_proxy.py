"""Smart Local Preview Gateway & Live SSE Relay for LIMBIC-0 × Gemma-4-12B-IT (Stage 6).

Serves local ``chat.html`` and relays native SSE ``/api/chat/stream`` (including live
``thought_delta``, ``tool_call``, ``tool_result``, ``polygraph``, and ``delta`` events)
plus all REST endpoints (`GET /api/souls`, `GET /api/embodiment`, `POST /api/intervene`)
directly to the upstream Cloud GPU server.
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CHAT_HTML = REPO_ROOT / "src" / "limbic_zero" / "vocal_cord" / "static" / "chat.html"


def make_proxy_handler(upstream_base: str):
    upstream = upstream_base.rstrip("/")

    class SmartGatewayHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, obj: Dict[str, Any], status: int = 200) -> None:
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(raw)
            self.wfile.flush()

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_HEAD(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            if route in ("/", "/index.html") and LOCAL_CHAT_HTML.is_file():
                data = LOCAL_CHAT_HTML.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return
            self._passthrough("HEAD")

        def do_GET(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            if route in ("/", "/index.html") and LOCAL_CHAT_HTML.is_file():
                data = LOCAL_CHAT_HTML.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()
                return
            self._passthrough("GET")

        def do_POST(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            if route == "/api/chat/stream":
                self._relay_sse_stream()
                return
            self._passthrough("POST")

        def _relay_sse_stream(self) -> None:
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length > 0 else b"{}"
            req = urllib.request.Request(
                f"{upstream}/api/chat/stream",
                data=body,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Accept": "text/event-stream",
                    "User-Agent": "Mozilla/5.0",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=240) as resp:
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache, no-transform")
                    self.send_header("Connection", "close")
                    self.send_header("X-Accel-Buffering", "no")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.close_connection = True

                    while True:
                        line = resp.readline()
                        if not line:
                            break
                        self.wfile.write(line)
                        if line in (b"\n", b"\r\n"):
                            self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                if not getattr(self, "close_connection", False):
                    self._send_json({"error": str(exc)}, status=502)

        def _passthrough(self, method: str) -> None:
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length > 0 else None
            target_url = f"{upstream}{self.path}"
            headers = {
                "Accept": self.headers.get("Accept", "*/*"),
                "User-Agent": "Mozilla/5.0",
            }
            if body is not None:
                headers["Content-Type"] = self.headers.get("Content-Type", "application/json")
            req = urllib.request.Request(target_url, data=body, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=240) as resp:
                    resp_body = resp.read()
                    self.send_response(resp.status)
                    self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                    self.send_header("Content-Length", str(len(resp_body)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    if method != "HEAD":
                        self.wfile.write(resp_body)
                        self.wfile.flush()
            except urllib.error.HTTPError as exc:
                err_body = exc.read()
                self.send_response(exc.code)
                self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(err_body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(err_body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                try:
                    self._send_json({"error": str(exc)}, status=502)
                except (BrokenPipeError, ConnectionResetError):
                    pass

    return SmartGatewayHandler


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6 Smart Local Preview Gateway & Live SSE Relay")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--upstream", required=True)
    args = parser.parse_args()

    server = ReusableThreadingHTTPServer((args.host, args.port), make_proxy_handler(args.upstream))
    print(f"[SmartGateway] Serving Stage 6 Studio on http://{args.host}:{args.port} -> {args.upstream}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
