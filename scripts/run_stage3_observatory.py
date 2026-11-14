"""Launch the Stage 3 Live Intracranial Neuro-fMRI & Creator God-Mode Web Observatory."""

from __future__ import annotations

import argparse
import signal
import sys

from limbic_zero.dashboard.server import create_observatory_server


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Project LIMBIC-0 Stage 3 Live Neuro-fMRI & Biosafety Observatory"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument("--seed", type=int, default=42, help="Initial Truman Dome seed (default: 42)")
    parser.add_argument("--adam-soul", default="", help="Optional .soul.pt path for ADAM-01")
    parser.add_argument("--abel-soul", default="", help="Optional .soul.pt path for ABEL-02")
    args = parser.parse_args()

    soul_map = {}
    if args.adam_soul:
        soul_map["ADAM-01"] = args.adam_soul
    if args.abel_soul:
        soul_map["ABEL-02"] = args.abel_soul

    httpd = create_observatory_server(
        host=args.host,
        port=args.port,
        seed=args.seed,
        start_loop=True,
        soul_capsule_paths=soul_map or None,
    )

    def _shutdown(_sig: int, _frame: object) -> None:
        httpd.controller.stop_background_loop()
        httpd.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(
        f"[LIMBIC-0 Observatory] Serving live Neuro-fMRI & God-Mode Console on "
        f"http://{args.host}:{args.port} (seed={args.seed})",
        flush=True,
    )
    try:
        httpd.serve_forever()
    finally:
        httpd.controller.stop_background_loop()
        httpd.server_close()


if __name__ == "__main__":
    main()
