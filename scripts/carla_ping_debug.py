"""Minimal CARLA connectivity log for Cursor vs interactive terminal comparison."""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "runs" / "carla_ping_debug.txt"


def log(msg: str) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def main() -> int:
    if OUT.exists():
        OUT.unlink()
    log(f"sys.executable={sys.executable}")
    log(f"cwd={os.getcwd()}")
    log(f"USER={os.environ.get('USERNAME')}")
    log(f"PID={os.getpid()}")
    host = os.environ.get("CARLA_HOST", "127.0.0.1")
    port = int(os.environ.get("CARLA_PORT", "2000"))
    try:
        s = socket.create_connection((host, port), timeout=5.0)
        log(f"tcp connect {host}:{port} OK local={s.getsockname()}")
        s.close()
    except Exception as e:
        log(f"tcp connect FAIL: {e!r}")
        return 1
    try:
        import carla

        log(f"carla={carla.__file__}")
        client = carla.Client(host, port)
        client.set_timeout(60.0)
        log("calling get_server_version (60s timeout)...")
        ver = client.get_server_version()
        log(f"server_version={ver}")
        world = client.get_world()
        log(f"map={world.get_map().name}")
        return 0
    except Exception as e:
        log(f"carla RPC FAIL: {type(e).__name__}: {e!r}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
