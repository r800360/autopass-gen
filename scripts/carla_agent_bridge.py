"""Localhost bridge so Cursor agent shell can run CARLA commands.

Start once per session from your integrated terminal (Ctrl+J), not the agent shell:

    cd C:\\Users\\bsach\\Documents\\autopass-gen
    .\\.venv\\Scripts\\activate
    python scripts/carla_agent_bridge.py

Leave this running while CarlaUE4.exe is open and the agent is working on CARLA tasks.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 17800
PID_FILE = ROOT / "runs" / "carla_watch" / "bridge.pid"
REQUEST_DIR = ROOT / "runs" / "carla_watch" / "bridge_requests"
RESPONSE_DIR = ROOT / "runs" / "carla_watch" / "bridge_responses"
POLL_INTERVAL_S = 0.05


def _carla_health() -> dict[str, Any]:
    try:
        import carla

        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(5.0)
        version = client.get_server_version()
        world = client.get_world()
        return {"carla_ok": True, "server_version": version, "map": world.get_map().name}
    except Exception as exc:
        return {"carla_ok": False, "error": str(exc)}


def _run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cwd = payload.get("cwd") or str(ROOT)
    cwd = str(Path(cwd).resolve())
    env = os.environ.copy()
    venv_scripts = ROOT / ".venv" / "Scripts"
    if venv_scripts.is_dir():
        env["PATH"] = str(venv_scripts) + os.pathsep + env.get("PATH", "")
    env["AUTOPASS_CARLA_VIA_BRIDGE"] = "1"

    try:
        if "shell" in payload:
            proc = subprocess.run(
                payload["shell"],
                shell=True,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=float(payload.get("timeout", 600)),
            )
        elif "argv" in payload:
            proc = subprocess.run(
                payload["argv"],
                shell=False,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=float(payload.get("timeout", 600)),
            )
        elif "code" in payload:
            proc = subprocess.run(
                [sys.executable, "-c", payload["code"]],
                shell=False,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=float(payload.get("timeout", 600)),
            )
        else:
            return {"ok": False, "error": "payload must include shell, argv, or code"}
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": f"timeout after {exc.timeout}s",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "exit_code": 124,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}

    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _process_file_queue(stop: threading.Event) -> None:
    REQUEST_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    while not stop.is_set():
        for req_path in sorted(REQUEST_DIR.glob("*.json")):
            req_id = req_path.stem
            resp_path = RESPONSE_DIR / f"{req_id}.json"
            try:
                payload = json.loads(req_path.read_text(encoding="utf-8"))
                result = _run_payload(payload)
                resp_path.write_text(json.dumps(result), encoding="utf-8")
            except Exception as exc:
                resp_path.write_text(
                    json.dumps({"ok": False, "error": str(exc), "traceback": traceback.format_exc()}),
                    encoding="utf-8",
                )
            finally:
                req_path.unlink(missing_ok=True)
        stop.wait(POLL_INTERVAL_S)


class BridgeHandler(BaseHTTPRequestHandler):
    server_version = "carla-agent-bridge/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[bridge] {self.address_string()} {fmt % args}")

    def _json_response(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._json_response(200, {"ok": True, **_carla_health()})
            return
        self._json_response(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/exec":
            self._json_response(404, {"ok": False, "error": "not found"})
            return
        try:
            payload = self._read_json()
            result = _run_payload(payload)
            code = 200 if result.get("ok") else 500
            self._json_response(code, result)
        except Exception as exc:
            self._json_response(500, {"ok": False, "error": str(exc), "traceback": traceback.format_exc()})


def main() -> int:
    parser = argparse.ArgumentParser(description="CARLA localhost bridge for Cursor agent shell")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()) + "\n", encoding="utf-8")

    stop = threading.Event()
    file_thread = threading.Thread(target=_process_file_queue, args=(stop,), daemon=True)
    file_thread.start()

    httpd = ThreadingHTTPServer((args.host, args.port), BridgeHandler)
    print(f"CARLA agent bridge listening on http://{args.host}:{args.port}")
    print(f"File queue: {REQUEST_DIR}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbridge stopped")
    finally:
        stop.set()
        file_thread.join(timeout=1.0)
        PID_FILE.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
