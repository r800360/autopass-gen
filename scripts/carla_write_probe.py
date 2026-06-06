"""Write CARLA connectivity probe results for the Cursor agent to read.

Run from your integrated terminal (Ctrl+J), not the agent shell:

    cd C:\\Users\\bsach\\Documents\\autopass-gen
    .\\.venv\\Scripts\\activate
    python scripts/carla_write_probe.py
"""
from __future__ import annotations

import getpass
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "carla_watch" / "agent_probe.json"


def main() -> int:
    result: dict = {
        "ok": False,
        "timestamp": time.time(),
        "user": getpass.getuser(),
        "username_env": os.environ.get("USERNAME"),
        "python": sys.executable,
    }
    try:
        import carla
    except ImportError as exc:
        result["error"] = f"import carla failed: {exc}"
        _write(result)
        print(json.dumps(result, indent=2))
        return 1

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    try:
        result["server_version"] = client.get_server_version()
        result["map"] = client.get_world().get_map().name
        result["ok"] = True
    except Exception as exc:
        result["error"] = str(exc)

    _write(result)
    print(json.dumps(result, indent=2))
    if result.get("ok"):
        print("\nNext: keep CarlaUE4 running and start the agent bridge in this terminal:")
        print("  python scripts/carla_agent_bridge.py")
    return 0 if result.get("ok") else 1


def _write(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
