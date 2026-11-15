"""Stage 14 Kaggle TPU v5e-8 Interactive Provisioner & Remote Deployment Commander.

Provisions a zero-queue interactive ``TPU v5e-8`` (`128 GB` HBM, `330 GB` Host RAM) VM on Kaggle
with the official ``google/gemma-4/transformers/gemma-4-31b-it/1`` (`58.3 GiB` safetensors) model
mounted locally under ``/kaggle/input``, launches the Stage 14 GSPMD `LIMBIC-0` server in the
background, and streams startup telemetry + the live ``https://*.trycloudflare.com`` URL.

Security invariant:
  - Reads ``KAGGLE_API_TOKEN`` strictly from ``os.environ`` (never hardcoded or persisted to repo).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import ssl
import struct
import subprocess
import tempfile
import time
from typing import Any, Dict, Optional, Tuple
import urllib.error
import urllib.parse
import urllib.request
import uuid


DEFAULT_STATE_FILE = Path("/tmp/limbic0_kaggle_tpu_session.json")
DEFAULT_MODEL_SOURCE = "google/gemma-4/transformers/gemma-4-31b-it/1"


def _require_kaggle_token() -> str:
    tok = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not tok:
        raise RuntimeError(
            "KAGGLE_API_TOKEN environment variable is required (never hardcode tokens in source files)."
        )
    return tok


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
) -> Any:
    hdrs = {"Accept": "application/json", "User-Agent": "limbic0-stage14-commander/1.0"}
    if headers:
        hdrs.update(headers)
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw.strip() else {}


class _MinimalJupyterWebSocket:
    """Zero-dependency RFC-6455 TLS WebSocket client for Jupyter Kernel/Terminal channels."""

    def __init__(self, ws_url: str, extra_headers: Optional[Dict[str, str]] = None, timeout: float = 30.0) -> None:
        parsed = urllib.parse.urlparse(ws_url)
        use_tls = parsed.scheme == "wss"
        host = parsed.hostname or ""
        port = parsed.port or (443 if use_tls else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY")
        raw_sock: socket.socket
        if proxy_url:
            p = urllib.parse.urlparse(proxy_url)
            p_host = p.hostname or "127.0.0.1"
            p_port = p.port or 8080
            raw_sock = socket.create_connection((p_host, p_port), timeout=timeout)
            connect_lines = [
                f"CONNECT {host}:{port} HTTP/1.1",
                f"Host: {host}:{port}",
            ]
            if p.username:
                pw = p.password or ""
                b64 = base64.b64encode(f"{urllib.parse.unquote(p.username)}:{urllib.parse.unquote(pw)}".encode()).decode()
                connect_lines.append(f"Proxy-Authorization: Basic {b64}")
            connect_req = ("\r\n".join(connect_lines) + "\r\n\r\n").encode("ascii")
            raw_sock.sendall(connect_req)
            resp_buf = b""
            while b"\r\n\r\n" not in resp_buf:
                chunk = raw_sock.recv(4096)
                if not chunk:
                    raise RuntimeError("Proxy closed connection during CONNECT handshake.")
                resp_buf += chunk
            first_line = resp_buf.split(b"\r\n", 1)[0].decode("latin1", errors="replace")
            if " 200 " not in first_line:
                raise RuntimeError(f"Proxy CONNECT failed: {first_line}")
        else:
            raw_sock = socket.create_connection((host, port), timeout=timeout)

        if use_tls:
            ctx = ssl.create_default_context()
            ca_bundle = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
            if ca_bundle and Path(ca_bundle).is_file():
                ctx.load_verify_locations(cafile=ca_bundle)
            self.sock: socket.socket = ctx.wrap_socket(raw_sock, server_hostname=host)
        else:
            self.sock = raw_sock

        self.sock.settimeout(timeout)
        ws_key = base64.b64encode(os.urandom(16)).decode("ascii")
        req_lines = [
            f"GET {path} HTTP/1.1",
            f"Host: {host}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {ws_key}",
            "Sec-WebSocket-Version: 13",
        ]
        if extra_headers:
            for hk, hv in extra_headers.items():
                req_lines.append(f"{hk}: {hv}")
        self.sock.sendall(("\r\n".join(req_lines) + "\r\n\r\n").encode("utf-8"))

        hdr_buf = b""
        while b"\r\n\r\n" not in hdr_buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError("WebSocket closed before HTTP 101 Switching Protocols.")
            hdr_buf += chunk
        status_line, self._leftover = hdr_buf.split(b"\r\n\r\n", 1)
        first = status_line.split(b"\r\n", 1)[0].decode("latin1", errors="replace")
        if " 101 " not in first:
            raise RuntimeError(f"WebSocket upgrade rejected: {first}")

    def _recv_exact(self, n: int) -> bytes:
        out = bytearray()
        if self._leftover:
            take = min(n, len(self._leftover))
            out.extend(self._leftover[:take])
            self._leftover = self._leftover[take:]
        while len(out) < n:
            chunk = self.sock.recv(n - len(out))
            if not chunk:
                raise EOFError("WebSocket connection closed.")
            out.extend(chunk)
        return bytes(out)

    def send_text(self, text: str) -> None:
        payload = text.encode("utf-8")
        mask_key = os.urandom(4)
        header = bytearray([0x81])  # FIN + text frame
        plen = len(payload)
        if plen < 126:
            header.append(0x80 | plen)
        elif plen < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", plen))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", plen))
        header.extend(mask_key)
        masked = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(bytes(header) + masked)

    def recv_text(self) -> Optional[str]:
        while True:
            b0, b1 = self._recv_exact(2)
            opcode = b0 & 0x0F
            masked = bool(b1 & 0x80)
            plen = b1 & 0x7F
            if plen == 126:
                plen = struct.unpack("!H", self._recv_exact(2))[0]
            elif plen == 127:
                plen = struct.unpack("!Q", self._recv_exact(8))[0]
            mkey = self._recv_exact(4) if masked else b""
            data = self._recv_exact(plen) if plen > 0 else b""
            if masked:
                data = bytes(b ^ mkey[i % 4] for i, b in enumerate(data))
            if opcode == 0x8:  # Close
                return None
            if opcode == 0x9:  # Ping -> Pong
                pong = bytearray([0x8A, 0x80 | len(data)])
                pk = os.urandom(4)
                pong.extend(pk)
                pong.extend(bytes(b ^ pk[i % 4] for i, b in enumerate(data)))
                self.sock.sendall(bytes(pong))
                continue
            if opcode in (0x1, 0x2):
                return data.decode("utf-8", errors="replace")

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


def run_python_on_jupyter_proxy(
    jupyter_base_url: str,
    python_code: str,
    timeout_sec: float = 300.0,
) -> Tuple[str, Optional[str]]:
    """Execute Python code inside the remote Kaggle TPU VM via Jupyter Kernel REST + WebSocket."""
    base = jupyter_base_url.rstrip("/")
    k_info = _http_json(f"{base}/api/kernels", method="POST", payload={"name": "python3"}, timeout=30.0)
    kid = k_info["id"]
    ws_base = "wss://" + base[len("https://") :] if base.startswith("https://") else "ws://" + base[len("http://") :]
    session_id = uuid.uuid4().hex
    msg_id = uuid.uuid4().hex
    ws_url = f"{ws_base}/api/kernels/{kid}/channels?session_id={session_id}"

    ws = _MinimalJupyterWebSocket(ws_url, timeout=min(60.0, timeout_sec))
    stdout_parts = []
    err_str: Optional[str] = None
    try:
        exec_msg = {
            "header": {
                "msg_id": msg_id,
                "username": "limbic0",
                "session": session_id,
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": python_code,
                "silent": False,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "buffers": [],
            "channel": "shell",
        }
        ws.send_text(json.dumps(exec_msg))

        deadline = time.monotonic() + timeout_sec
        got_reply = False
        is_idle = False
        while time.monotonic() < deadline:
            raw = ws.recv_text()
            if raw is None:
                break
            msg = json.loads(raw)
            parent_id = msg.get("parent_header", {}).get("msg_id")
            if parent_id != msg_id:
                continue
            mtype = msg.get("msg_type") or msg.get("header", {}).get("msg_type")
            content = msg.get("content", {})
            if mtype == "stream":
                txt = content.get("text", "")
                stdout_parts.append(txt)
                print(txt, end="", flush=True)
            elif mtype == "error":
                tb = "\n".join(content.get("traceback", []))
                err_str = f"{content.get('ename')}: {content.get('evalue')}\n{tb}"
            elif mtype == "execute_reply":
                got_reply = True
                if content.get("status") == "error" and err_str is None:
                    err_str = f"{content.get('ename')}: {content.get('evalue')}"
            elif mtype == "status" and content.get("execution_state") == "idle" and got_reply:
                is_idle = True
            if got_reply and is_idle:
                break
    finally:
        ws.close()
        try:
            _http_json(f"{base}/api/kernels/{kid}", method="DELETE", timeout=15.0)
        except Exception:
            pass

    return "".join(stdout_parts), err_str


def provision_kaggle_tpu_v5e8_session(
    owner: str,
    kernel_slug_name: str = "limbic0-stage14-tpu-v5e8",
    kaggle_cli_bin: str = "kaggle",
    model_source: str = DEFAULT_MODEL_SOURCE,
    skip_push: bool = False,
    state_file: Path = DEFAULT_STATE_FILE,
) -> Dict[str, Any]:
    """Allocate a live interactive TPU v5e-8 VM (optionally mounting a Kaggle Model source)."""
    _require_kaggle_token()
    from kagglesdk import KaggleClient  # type: ignore[import-not-found]
    from kagglesdk.kernels.types.kernels_api_service import (  # type: ignore[import-not-found]
        ApiCreateKernelSessionRequest,
        ApiGetAcceleratorQuotaStatisticsRequest,
    )

    full_slug = f"{owner}/{kernel_slug_name}"
    if not skip_push:
        with tempfile.TemporaryDirectory(prefix="limbic0_kpush_") as tmpdir:
            tpath = Path(tmpdir)
            meta = {
                "id": full_slug,
                "title": kernel_slug_name,
                "code_file": "runner.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": False,
                "enable_tpu": True,
                "enable_internet": True,
                "machine_shape": "TpuV5E8",
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [model_source] if model_source else [],
            }
            (tpath / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            nb = {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": ["print('LIMBIC-0 Stage 14 TPU v5e-8 Interactive Host Ready')\n"],
                    }
                ],
                "metadata": {
                    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"},
                },
                "nbformat": 4,
                "nbformat_minor": 5,
            }
            (tpath / "runner.ipynb").write_text(json.dumps(nb, indent=2), encoding="utf-8")

            push_cmd = [kaggle_cli_bin, "kernels", "push", "--no-run", "-p", str(tpath)]
            print(f"[COMMANDER] Pushing kernel definition `{full_slug}` with `{model_source}`...", flush=True)
            subprocess.run(push_cmd, check=True)

    client = KaggleClient()
    req = ApiCreateKernelSessionRequest()
    req.slug = full_slug
    req.language = "python"
    req.kernel_type = "notebook"
    req.machine_shape = "TpuV5E8"
    req.enable_internet = True

    t0 = time.perf_counter()
    op = client.kernels.kernels_api_client.create_kernel_session(req)
    op_name = op.name
    print(f"[COMMANDER] CreateKernelSession started (`{op_name}`, slug=`{full_slug}`). Waiting for TPU v5e-8 VM...", flush=True)

    jupyter_url = ""
    session_resp: Dict[str, Any] = {}
    for step_idx in range(90):
        time.sleep(5.0)
        st = client.common.operations_client.get_operation(name=op_name)
        if step_idx % 3 == 0 and not st.done:
            q = client.kernels.kernels_api_client.get_accelerator_quota_statistics(
                ApiGetAcceleratorQuotaStatisticsRequest()
            )
            tpu_res = q.tpu_quota.time_reserved if q.tpu_quota else "0s"
            print(
                f"[COMMANDER] [{time.perf_counter() - t0:.1f}s] Provisioning GCE TPU v5e-8 slice (reserved={tpu_res})...",
                flush=True,
            )
        if st.done:
            if st.error and getattr(st.error, "message", ""):
                raise RuntimeError(f"CreateKernelSession failed: {st.error.message}")
            raw_resp = st.response
            if isinstance(raw_resp, dict):
                session_resp = raw_resp
            elif hasattr(raw_resp, "to_dict"):
                session_resp = raw_resp.to_dict()
            else:
                session_resp = dict(raw_resp or {})
            jupyter_url = str(
                session_resp.get("tokenizedJupyterUrl")
                or session_resp.get("tokenized_jupyter_url")
                or session_resp.get("jupyterUrl")
                or session_resp.get("jupyter_url")
                or ""
            )
            break

    if not jupyter_url:
        raise RuntimeError(f"Timed out or missing tokenizedJupyterUrl from operation `{op_name}`: {session_resp}")

    m_sid = re.search(r"/k/(\d+)/", jupyter_url)
    kernel_session_id = int(
        session_resp.get("kernelSessionId")
        or session_resp.get("kernel_session_id")
        or session_resp.get("id")
        or (m_sid.group(1) if m_sid else 0)
    )

    elapsed = round(time.perf_counter() - t0, 1)
    print(f"[COMMANDER] TPU v5e-8 VM allocated in {elapsed}s (session_id={kernel_session_id})!", flush=True)

    saved = {
        "slug": full_slug,
        "kernel_session_id": kernel_session_id,
        "operation_name": op_name,
        "jupyter_url": jupyter_url,
        "provisioned_at_unix": time.time(),
        "provision_elapsed_sec": elapsed,
    }
    state_file.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    state_file.chmod(0o600)
    return saved


def stop_kaggle_tpu_session(
    kernel_session_id: int,
    state_file: Path = DEFAULT_STATE_FILE,
) -> None:
    """Cancel an active interactive Kaggle TPU session cleanly to preserve weekly TPU quota."""
    _require_kaggle_token()
    from kagglesdk import KaggleClient  # type: ignore[import-not-found]
    from kagglesdk.kernels.types.kernels_api_service import (  # type: ignore[import-not-found]
        ApiCancelKernelSessionRequest,
    )

    client = KaggleClient()
    req = ApiCancelKernelSessionRequest()
    req.kernel_session_id = int(kernel_session_id)
    client.kernels.kernels_api_client.cancel_kernel_session(req)
    print(f"[COMMANDER] Cancelled interactive TPU session `{kernel_session_id}`.", flush=True)
    if state_file.is_file():
        state_file.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 14 Kaggle TPU v5e-8 Provisioner & Commander")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prov = sub.add_parser("provision", help="Allocate interactive TPU v5e-8 VM with Gemma-4-31B mount")
    p_prov.add_argument("--owner", type=str, required=True)
    p_prov.add_argument("--slug-name", type=str, default="limbic0-stage14-tpu-v5e8")
    p_prov.add_argument("--kaggle-bin", type=str, default=shutil.which("kaggle") or "kaggle")
    p_prov.add_argument("--model-source", type=str, default=DEFAULT_MODEL_SOURCE)
    p_prov.add_argument("--skip-push", action="store_true", help="Skip SaveKernel push if kernel slug already exists")

    p_exec = sub.add_parser("exec-py", help="Execute Python snippet on active Kaggle TPU VM")
    p_exec.add_argument("--code", type=str, required=True)
    p_exec.add_argument("--timeout", type=float, default=300.0)

    p_stop = sub.add_parser("stop", help="Cancel active Kaggle TPU session")
    p_stop.add_argument("--session-id", type=int, default=0)

    args = parser.parse_args()
    if args.cmd == "provision":
        info = provision_kaggle_tpu_v5e8_session(
            owner=args.owner,
            kernel_slug_name=args.slug_name,
            kaggle_cli_bin=args.kaggle_bin,
            model_source=args.model_source,
            skip_push=args.skip_push,
        )
        print(json.dumps({k: v for k, v in info.items() if k != "jupyter_url"}, indent=2))
    elif args.cmd == "exec-py":
        st = json.loads(DEFAULT_STATE_FILE.read_text(encoding="utf-8"))
        out, err = run_python_on_jupyter_proxy(st["jupyter_url"], args.code, timeout_sec=args.timeout)
        if err:
            raise SystemExit(f"Remote execution error:\n{err}")
    elif args.cmd == "stop":
        sid = int(args.session_id)
        if not sid and DEFAULT_STATE_FILE.is_file():
            sid = int(json.loads(DEFAULT_STATE_FILE.read_text(encoding="utf-8")).get("kernel_session_id", 0))
        if not sid:
            raise SystemExit("Provide --session-id or have an active session state file.")
        stop_kaggle_tpu_session(sid)


if __name__ == "__main__":
    main()
