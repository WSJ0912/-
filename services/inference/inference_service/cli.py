from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="127.0.0.1-only medical imaging service")
    parser.add_argument("--root", type=Path, default=Path("runtime"))
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(f"uvicorn import failed: {exc}") from exc
    from .main import create_app

    port = args.port or _free_port()
    app = create_app(args.root)
    print(json.dumps({"host": "127.0.0.1", "port": port, "processToken": app.state.process_token}, ensure_ascii=True), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
