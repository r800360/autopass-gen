#!/usr/bin/env python3
"""Run several curated CARLA agentic demos (not only demo_07) and summarize metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="CARLA multi-scenario gallery (agentic + video)")
    parser.add_argument(
        "--demos",
        type=str,
        default="0,1,2,3,4,5",
        help="Comma-separated curated_demo_scenarios indices",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/carla_gallery"))
    parser.add_argument("--policy", choices=["autopass", "no_pass"], default="autopass")
    parser.add_argument("--urgency", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--steps", type=int, default=28, help="Max execute steps per scenario")
    parser.add_argument("--fast", action="store_true", help="Faster frames (no realtime pacing)")
    args = parser.parse_args()

    from dataclasses import replace

    from autopass.config import apply_production_defaults, require_runtime
    from autopass.scenarios import showcase_map_for_environment
    from demo_carla_watch import run_agentic_carla_loop
    from visual_world import curated_demo_scenarios, initialize_world

    apply_production_defaults()
    require_runtime(need_carla=True, need_openai=True)

    import os

    os.environ["AUTOPASS_ENVIRONMENT"] = "highway"
    os.environ["AUTOPASS_CARLA_CURATED_CORRIDOR"] = "1"
    os.environ["AUTOPASS_CARLA_CORRIDOR_MODE"] = "presentation"
    map_name = showcase_map_for_environment("highway")
    os.environ["AUTOPASS_CARLA_MAP"] = map_name
    if args.fast:
        os.environ["AUTOPASS_VIDEO_REALTIME"] = "0"

    indices = [int(x.strip()) for x in args.demos.split(",") if x.strip()]
    demos = curated_demo_scenarios()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for idx in indices:
        if idx < 0 or idx >= len(demos):
            print(f"[gallery] skip invalid index {idx}")
            continue
        spec = replace(demos[idx], route=replace(demos[idx].route, town=map_name))
        if idx == 6:
            print("[gallery] skip demo_07 (perception hero — use demo_carla_watch --hero-pass separately)")
            continue
        sub = args.out_dir / spec.scenario_id
        sub.mkdir(parents=True, exist_ok=True)
        print(f"\n[gallery] === {idx}: {spec.scenario_id} ===")
        world = initialize_world(spec)
        ticks = 6 if args.fast else 10
        state = run_agentic_carla_loop(
            spec,
            world,
            sub,
            ticks,
            args.steps,
            policy=args.policy,
            mission_urgency=args.urgency,
        )
        metrics = state.get("metrics", {})
        row = {
            "index": idx,
            "scenario_id": spec.scenario_id,
            "failure_type": metrics.get("failure_type"),
            "failure_taxonomy": metrics.get("failure_taxonomy"),
            "route_completed": metrics.get("route_completed"),
            "collision": metrics.get("collision"),
            "agency": metrics.get("agency"),
            "out_dir": str(sub),
        }
        summary.append(row)
        print(
            f"[gallery] {spec.scenario_id}: failure={row['failure_type']} "
            f"route_ok={row['route_completed']} collision={row['collision']}"
        )

    out_json = args.out_dir / "gallery_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[gallery] Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
