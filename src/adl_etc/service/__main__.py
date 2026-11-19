"""``python -m adl_etc.service [--config configs/service.yaml] [--override key=value ...]``

Runs the service with uvicorn, one worker, bound to ``127.0.0.1`` (correction S9). Checks the port
first: on Windows a killed ``uvicorn`` can leave a process holding it (see ``docs/demo.md``).
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/service.yaml")
    parser.add_argument("--override", nargs="*", default=[], help="dotted key=value overrides")
    args = parser.parse_args(argv)

    import uvicorn

    from adl_etc.service.app import create_app
    from adl_etc.utils.config import load_config

    cfg = load_config(args.config, args.override)
    host, port = str(cfg.host), int(cfg.port)
    if host not in ("127.0.0.1", "localhost"):
        print(f"refusing to bind {host}: the service has no authentication (correction S9)")
        return 2
    if not port_free(host, port):
        print(
            f"port {port} is in use. On Windows a killed uvicorn can leave a worker holding it:\n"
            f"  netstat -ano | findstr :{port}     then   taskkill /PID <pid> /F\n"
            f"or start on another port: --override port={port + 1}"
        )
        return 1
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    app = create_app(args.config, args.override)
    print(f"Service on http://{host}:{port}  (UI at /, API docs at /docs)")
    uvicorn.run(app, host=host, port=port, log_level="warning", workers=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
