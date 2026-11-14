"""Smart Local Gateway & SSE Streaming Adapter for LIMBIC-0 × Gemma-4-12B-IT.

Serves the latest local ``chat.html`` UI and bridges ``/api/chat/stream`` (plus ``/api/chat``
and ``/v1/chat/completions``) to the live GPU Vocal Cord server with full-budget (`512` per hop,
auto-chained up to `2048` tokens) generation so long Korean responses never truncate mid-sentence.
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List
from urllib.parse import urlparse
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CHAT_HTML = REPO_ROOT / "src" / "limbic_zero" / "vocal_cord" / "static" / "chat.html"
UPSTREAM_HOP_MAX_TOKENS = 512


def _upstream_json(upstream: str, path: str, method: str = "GET", payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
    }
    if data is not None:
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(f"{upstream}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _needs_continuation(reply_text: str, gen_tokens: int, hop_limit: int) -> bool:
    if gen_tokens < hop_limit - 8:
        return False
    stripped = reply_text.rstrip()
    if not stripped:
        return False
    # If it hit the token ceiling and did not end on terminal punctuation, continue seamlessly
    terminal_chars = (".", "!", "?", "…", "다", "요", "죠", "네", "군", "어", "야", "지", "\"", "'", ")", "]")
    if stripped[-1] not in (".", "!", "?", "…"):
        return True
    return gen_tokens >= hop_limit - 2 and stripped[-1] not in terminal_chars


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

        def _read_json(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

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
                self._handle_chat_stream()
                return
            if route == "/api/chat":
                self._handle_chat_sync()
                return
            self._passthrough("POST")

        def _write_sse(self, evt: Dict[str, Any]) -> None:
            chunk = f"data: {json.dumps(evt, ensure_ascii=False)}\n\n".encode("utf-8")
            self.send_chunk_or_raw(chunk)

        def send_chunk_or_raw(self, raw: bytes) -> None:
            # HTTP/1.1 chunked encoding so proxies and browsers flush immediately
            header = f"{len(raw):X}\r\n".encode("ascii")
            self.wfile.write(header + raw + b"\r\n")
            self.wfile.flush()

        def _handle_chat_stream(self) -> None:
            try:
                body = self._read_json()
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
                return

            soul_id = str(body.get("soul_id", "ADAM-01"))
            message = str(body.get("message", "")).strip()
            if not message:
                self._send_json({"error": "Empty message"}, status=400)
                return
            steer_enabled = bool(body.get("steer_enabled", True))
            steer_gain = float(body.get("steer_gain", 1.0))
            total_budget = min(2048, max(64, int(body.get("max_new_tokens", 1024))))
            temperature = float(body.get("temperature", 0.70))
            use_history = bool(body.get("use_history", True))

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            t0 = time.perf_counter()
            max_hops = max(1, math.ceil(total_budget / UPSTREAM_HOP_MAX_TOKENS))
            remaining = total_budget
            combined_parts: List[str] = []
            total_tokens = 0
            last_res: Dict[str, Any] = {}

            try:
                for hop_idx in range(max_hops):
                    hop_budget = min(UPSTREAM_HOP_MAX_TOKENS, remaining)
                    if hop_budget < 24:
                        break
                    hop_prompt = (
                        message
                        if hop_idx == 0
                        else "방금 답변이 토큰 길이 제한 때문에 문장 중간에서 끊겼어. 앞 내용을 요약하거나 인사하지 말고, 정확히 끊긴 지점 바로 뒷말부터 자연스럽게 끝까지 이어서 완성해줘."
                    )
                    hop_res = _upstream_json(
                        upstream,
                        "/api/chat",
                        method="POST",
                        payload={
                            "soul_id": soul_id,
                            "message": hop_prompt,
                            "steer_enabled": steer_enabled,
                            "steer_gain": steer_gain,
                            "max_new_tokens": hop_budget,
                            "temperature": temperature,
                        },
                    )
                    last_res = hop_res
                    if hop_idx == 0:
                        self._write_sse(
                            {
                                "type": "meta",
                                "organism_id": hop_res.get("organism_id", soul_id),
                                "steer_enabled": hop_res.get("steer_enabled", steer_enabled),
                                "steer_gain": hop_res.get("steer_gain", steer_gain),
                                "target_layers": hop_res.get("target_layers", []),
                                "steering_axes_6d": hop_res.get("steering_axes_6d", {}),
                                "soul_telemetry": hop_res.get("soul_telemetry"),
                            }
                        )
                        self._write_sse(
                            {
                                "type": "injections",
                                "layer_injections": hop_res.get("layer_injections", []),
                            }
                        )

                    piece_text = str(hop_res.get("reply", ""))
                    gen_tok = int(hop_res.get("generated_tokens", 0))
                    total_tokens += gen_tok
                    remaining -= gen_tok

                    if piece_text:
                        sep = " " if (hop_idx > 0 and combined_parts and not combined_parts[-1].endswith(("\n", " "))) else ""
                        emit_str = sep + piece_text
                        combined_parts.append(emit_str)
                        # Stream smoothly in 18-char slices for immediate progressive rendering
                        step = 18
                        for i in range(0, len(emit_str), step):
                            self._write_sse({"type": "delta", "delta": emit_str[i : i + step]})

                    # Only chain a second hop if stateful chat_history is enabled and hop hit ceiling mid-thought
                    if not (use_history and steer_enabled):
                        break
                    if not _needs_continuation(piece_text, gen_tok, hop_budget):
                        break

                full_reply = "".join(combined_parts).strip()
                elapsed = round(time.perf_counter() - t0, 3)
                self._write_sse(
                    {
                        "type": "done",
                        "organism_id": last_res.get("organism_id", soul_id),
                        "reply": full_reply,
                        "layer_injections": last_res.get("layer_injections", []),
                        "soul_telemetry": last_res.get("soul_telemetry"),
                        "latency_sec": elapsed,
                        "generated_tokens": total_tokens,
                    }
                )
            except Exception as exc:
                try:
                    self._write_sse({"type": "done", "reply": "".join(combined_parts), "error": str(exc), "latency_sec": round(time.perf_counter() - t0, 3), "generated_tokens": total_tokens})
                except Exception:
                    pass
            finally:
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except Exception:
                    pass

        def _handle_chat_sync(self) -> None:
            try:
                body = self._read_json()
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
                return

            total_budget = min(2048, max(64, int(body.get("max_new_tokens", 1024))))
            soul_id = str(body.get("soul_id", "ADAM-01"))
            message = str(body.get("message", "")).strip()
            steer_enabled = bool(body.get("steer_enabled", True))
            steer_gain = float(body.get("steer_gain", 1.0))
            temperature = float(body.get("temperature", 0.70))

            t0 = time.perf_counter()
            max_hops = max(1, math.ceil(total_budget / UPSTREAM_HOP_MAX_TOKENS))
            remaining = total_budget
            parts: List[str] = []
            total_tok = 0
            first_res: Dict[str, Any] = {}
            last_res: Dict[str, Any] = {}

            try:
                for hop_idx in range(max_hops):
                    hop_budget = min(UPSTREAM_HOP_MAX_TOKENS, remaining)
                    if hop_budget < 24:
                        break
                    hop_prompt = (
                        message
                        if hop_idx == 0
                        else "방금 답변이 문장 중간에서 끊겼어. 앞 내용 반복 없이 끊긴 부분 바로 뒷말부터 끝까지 자연스럽게 마무리해줘."
                    )
                    res = _upstream_json(
                        upstream,
                        "/api/chat",
                        method="POST",
                        payload={
                            "soul_id": soul_id,
                            "message": hop_prompt,
                            "steer_enabled": steer_enabled,
                            "steer_gain": steer_gain,
                            "max_new_tokens": hop_budget,
                            "temperature": temperature,
                        },
                    )
                    if hop_idx == 0:
                        first_res = res
                    last_res = res
                    piece = str(res.get("reply", ""))
                    gtok = int(res.get("generated_tokens", 0))
                    total_tok += gtok
                    remaining -= gtok
                    if piece:
                        parts.append((" " if hop_idx > 0 else "") + piece)
                    if not steer_enabled or not _needs_continuation(piece, gtok, hop_budget):
                        break

                merged = dict(first_res)
                merged["reply"] = "".join(parts).strip()
                merged["generated_tokens"] = total_tok
                merged["latency_sec"] = round(time.perf_counter() - t0, 3)
                if "soul_telemetry" in last_res:
                    merged["soul_telemetry"] = last_res["soul_telemetry"]
                self._send_json(merged)
            except Exception as exc:
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
                with urllib.request.urlopen(req, timeout=180) as resp:
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
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=502)

    return SmartGatewayHandler


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 Smart Local Gateway & SSE Streaming Proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--upstream", required=True)
    args = parser.parse_args()

    server = ReusableThreadingHTTPServer((args.host, args.port), make_proxy_handler(args.upstream))
    print(f"[SmartGateway] Serving local chat.html + full-token SSE on http://{args.host}:{args.port} -> {args.upstream}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
