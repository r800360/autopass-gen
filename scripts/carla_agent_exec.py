"""Run a shell command via the CARLA agent bridge (for Cursor agent shell).

The bridge must be running in your integrated terminal:
    python scripts/carla_agent_bridge.py
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = os.environ.get("AUTOPASS_CARLA_BRIDGE_URL", "http://127.0.0.1:17800")
REQUEST_DIR = ROOT / "runs" / "carla_watch" / "bridge_requests"
RESPONSE_DIR = ROOT / "runs" / "carla_watch" / "bridge_responses"


def _post(path: str, payload: dict) -> dict:
    url = DEFAULT_URL.rstrip("/") + path
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=float(payload.get("timeout", 620))) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _health_http() -> dict | None:
    url = DEFAULT_URL.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=1.5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _bridge_hint() -> str:
    return (
        "CARLA bridge is not running. In your integrated terminal (Ctrl+J), run:\n"
        "  cd C:\\Users\\bsach\\Documents\\autopass-gen\n"
        "  .\\.venv\\Scripts\\activate\n"
        "  python scripts/carla_agent_bridge.py"
    )


def _emit_result(result: dict) -> int:
    if result.get("stdout"):
        sys.stdout.write(result["stdout"])
        if not result["stdout"].endswith("\n"):
            sys.stdout.write("\n")
    if result.get("stderr"):
        sys.stderr.write(result["stderr"])
        if not result["stderr"].endswith("\n"):
            sys.stderr.write("\n")
    if not result.get("ok"):
        if result.get("error"):
            print(result["error"], file=sys.stderr)
        err = (result.get("stderr") or "") + (result.get("stdout") or "")
        if "time-out" in err and "simulator" in err:
            print(
                "\nBridge subprocess could not reach CARLA. Start the bridge from YOUR terminal (Ctrl+J), "
                "not the agent shell:\n  python scripts/carla_agent_bridge.py",
                file=sys.stderr,
            )
        return int(result.get("exit_code") or 1)
    return int(result.get("exit_code") or 0)


def _exec_via_files(payload: dict) -> dict:
    REQUEST_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    req_id = uuid.uuid4().hex
    req_path = REQUEST_DIR / f"{req_id}.json"
    resp_path = RESPONSE_DIR / f"{req_id}.json"
    req_path.write_text(json.dumps(payload), encoding="utf-8")

    deadline = time.time() + float(payload.get("timeout", 600)) + 30.0
    while time.time() < deadline:
        if resp_path.exists():
            try:
                return json.loads(resp_path.read_text(encoding="utf-8"))
            finally:
                resp_path.unlink(missing_ok=True)
                req_path.unlink(missing_ok=True)
        if not req_path.exists() and not resp_path.exists():
            return {"ok": False, "error": "bridge dropped request", "exit_code": 127}
        time.sleep(0.05)
    req_path.unlink(missing_ok=True)
    return {"ok": False, "error": "bridge file response timeout", "exit_code": 124}


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute via CARLA agent bridge")
    parser.add_argument("--shell-b64", help="Base64-encoded shell command")
    parser.add_argument("-c", "--code", help="Python code to run")
    parser.add_argument("--cwd", default=str(ROOT))
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("argv", nargs="*", help="Command argv after --")
    args = parser.parse_args()

    payload: dict = {"cwd": args.cwd, "timeout": args.timeout}
    if args.shell_b64:
        payload["shell"] = base64.b64decode(args.shell_b64.encode("ascii")).decode("utf-8")
    elif args.code:
        payload["code"] = args.code
    elif args.argv:
        payload["argv"] = args.argv
    else:
        print("Provide --shell-b64, -c, or argv after --", file=sys.stderr)
        return 2

    health = _health_http()
    if health and health.get("ok"):
        try:
            return _emit_result(_post("/exec", payload))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(detail, file=sys.stderr)
            return exc.code if exc.code else 1
        except Exception:
            pass

    if not (REQUEST_DIR.parent / "bridge.pid").exists() and health is None:
        print(_bridge_hint(), file=sys.stderr)
        return 127

    result = _exec_via_files(payload)
    if result.get("error") == "bridge file response timeout":
        print(_bridge_hint(), file=sys.stderr)
    return _emit_result(result)


if __name__ == "__main__":
    raise SystemExit(main())
