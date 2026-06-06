#!/usr/bin/env python3
"""Run the 20-clip visual credibility campaign (production LLM + CARLA + video).

Requires CarlaUE4.exe and the agent bridge running in your terminal:
  python scripts/carla_agent_bridge.py

From agent shell, prefer:
  python scripts/run_visual_campaign.py --via-bridge --clip clip_01_highway_urgent_safe_pass
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _apply_production_env() -> None:
    from autopass.config import apply_production_defaults

    apply_production_defaults()
    os.environ.setdefault("AUTOPASS_TEST_MODE", "0")
    os.environ.setdefault("AUTOPASS_MOCK_LLM", "0")
    os.environ.setdefault("AUTOPASS_DECISION_ORACLE", "0")
    os.environ.setdefault("AUTOPASS_LLM_TEMPERATURE", "0.4")
    os.environ.setdefault("AUTOPASS_EXECUTE_DT_S", "0.35")
    os.environ.setdefault("AUTOPASS_VIDEO_REALTIME", "1")
    os.environ.setdefault("AUTOPASS_CARLA_MAX_CORRIDOR_REPICK", "2")
    os.environ.setdefault("AUTOPASS_CARLA_MIN_PASSING_HORIZON_M", "20")
    os.environ.setdefault("AUTOPASS_PERCEPTION_BACKEND", "carla")
    os.environ.setdefault("AUTOPASS_CARLA_CURATED_CORRIDOR", "1")
    for key in (
        "AUTOPASS_CARLA_MAX_STEER",
        "AUTOPASS_CARLA_LANE_DEPARTURE_FAIL_M",
        "AUTOPASS_CARLA_LANE_CHANGE_STEER_CAP_MULT",
    ):
        os.environ.pop(key, None)


def _run_clip_direct(clip, out_dir: Path, *, steps: int, fast: bool) -> dict:
    from autopass.benchmark_catalog import apply_urgency
    from autopass.scenarios import assert_carla_environment_allowed, showcase_map_for_environment
    from autopass.visual_campaign import VisualCampaignClip
    from demo_carla_watch import run_agentic_carla_loop
    from visual_world import curated_demo_scenarios, initialize_world

    assert isinstance(clip, VisualCampaignClip)
    demos = curated_demo_scenarios()
    if clip.demo_index < 0 or clip.demo_index >= len(demos):
        raise ValueError(f"invalid demo_index {clip.demo_index}")
    if demos[clip.demo_index].scenario_id.startswith("demo_07_"):
        raise ValueError("demo_07 is excluded from the visual campaign")

    os.environ["AUTOPASS_ENVIRONMENT"] = clip.environment
    map_name = clip.carla_map or showcase_map_for_environment(clip.environment)
    os.environ["AUTOPASS_CARLA_MAP"] = map_name
    # Allow unvalidated for all non-highway maps or any non-Town04 map
    non_highway_envs = ("town", "local", "suburban", "multilane")
    non_highway_maps = ("Town01", "Town02", "Town03", "Town05", "Town10HD")
    if clip.environment in non_highway_envs or map_name in non_highway_maps:
        os.environ["AUTOPASS_CARLA_ALLOW_UNVALIDATED"] = "1"
    os.environ["AUTOPASS_CARLA_CORRIDOR_MODE"] = "presentation"
    # Bypass environment gate for diverse-map campaign
    os.environ["AUTOPASS_CARLA_ALLOW_UNVALIDATED"] = "1"

    spec = apply_urgency(demos[clip.demo_index], clip.urgency)
    spec = replace(spec, route=replace(spec.route, town=map_name))
    sub = out_dir / clip.clip_id
    sub.mkdir(parents=True, exist_ok=True)
    world = initialize_world(spec)
    ticks = 6 if fast else 10
    if fast:
        os.environ["AUTOPASS_VIDEO_REALTIME"] = "0"
        os.environ["AUTOPASS_DEMO_DENSE_FRAMES"] = "0"
    else:
        os.environ.setdefault("AUTOPASS_VIDEO_REALTIME", "1")
        os.environ.setdefault("AUTOPASS_DEMO_DENSE_FRAMES", "1")

    print(f"\n[campaign] {clip.clip_id}: {spec.scenario_id} env={clip.environment} map={map_name} urg={clip.urgency}")
    state = run_agentic_carla_loop(
        spec,
        world,
        sub,
        ticks,
        steps,
        policy=clip.policy,
        mission_urgency=clip.urgency,
    )
    metrics = state.get("metrics", {})
    row = {
        "clip_id": clip.clip_id,
        "scenario_id": spec.scenario_id,
        "environment": clip.environment,
        "carla_map": map_name,
        "urgency": clip.urgency,
        "expected": clip.expected,
        "narrative": clip.narrative,
        "failure_type": metrics.get("failure_type"),
        "route_completed": metrics.get("route_completed"),
        "collision": metrics.get("collision"),
        "passed": metrics.get("passed"),
        "agency": metrics.get("agency"),
        "out_dir": str(sub),
    }
    return row


def _bridge_command(inner: str, timeout_s: float) -> list[str]:
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    exec_py = ROOT / "scripts" / "carla_agent_exec.py"
    return [str(py), str(exec_py), "--timeout", str(timeout_s), "--shell-b64", inner]


def _run_clip_via_bridge(clip, out_dir: Path, *, steps: int, fast: bool, timeout_s: float) -> dict:
    import base64

    fast_flag = " --fast" if fast else ""
    inner = (
        f'cd /d "{ROOT}" && set AUTOPASS_TEST_MODE=0&& set AUTOPASS_MOCK_LLM=0&& '
        f'set AUTOPASS_DECISION_ORACLE=0&& set AUTOPASS_LLM_TEMPERATURE=0.4&& '
        f'set AUTOPASS_EXECUTE_DT_S=0.35&& set AUTOPASS_VIDEO_REALTIME=1&& '
        f'"{ROOT / ".venv" / "Scripts" / "python.exe"}" '
        f'"{ROOT / "scripts" / "run_visual_campaign.py"}" '
        f'--clip {clip.clip_id} --out-dir "{out_dir}" --steps {steps}{fast_flag}'
    )
    b64 = base64.b64encode(inner.encode("utf-8")).decode("ascii")
    cmd = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        str(ROOT / "scripts" / "carla_agent_exec.py"),
        "--timeout",
        str(timeout_s),
        "--shell-b64",
        b64,
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True)
    summary_path = out_dir / f"{clip.clip_id}_bridge_result.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {"clip_id": clip.clip_id, "exit_code": proc.returncode, "bridge": True}


def main() -> int:
    parser = argparse.ArgumentParser(description="20-clip CARLA visual credibility campaign")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/visual_campaign"))
    parser.add_argument("--clip", type=str, default="", help="Single clip_id (default: all 20)")
    parser.add_argument("--start", type=int, default=0, help="Start index into campaign list")
    parser.add_argument("--count", type=int, default=0, help="Number of clips (0 = all remaining)")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--via-bridge", action="store_true", help="Dispatch through carla_agent_bridge")
    parser.add_argument("--bridge-timeout", type=float, default=2400.0)
    args = parser.parse_args()

    from autopass.config import require_runtime
    from autopass.visual_campaign import VISUAL_CAMPAIGN_20, campaign_clip_by_id

    _apply_production_env()
    if args.via_bridge:
        require_runtime(need_carla=False, need_openai=True)
    else:
        require_runtime(need_carla=True, need_openai=True)

    clips = VISUAL_CAMPAIGN_20
    if args.clip:
        clips = [campaign_clip_by_id(args.clip)]
    else:
        end = len(clips) if args.count <= 0 else min(len(clips), args.start + args.count)
        clips = clips[args.start:end]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for clip in clips:
        if args.via_bridge:
            row = _run_clip_via_bridge(
                clip, args.out_dir, steps=args.steps, fast=args.fast, timeout_s=args.bridge_timeout
            )
        else:
            row = _run_clip_direct(clip, args.out_dir, steps=args.steps, fast=args.fast)
            sidecar = args.out_dir / f"{clip.clip_id}_bridge_result.json"
            sidecar.write_text(json.dumps(row, indent=2), encoding="utf-8")
        summary.append(row)
        print(
            f"[campaign] {clip.clip_id}: failure={row.get('failure_type')} "
            f"route_ok={row.get('route_completed')} expected={clip.expected}"
        )

    out_json = args.out_dir / "campaign_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[campaign] Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
