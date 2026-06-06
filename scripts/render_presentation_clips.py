#!/usr/bin/env python3
"""
Render audience-facing CARLA clips from presentation_catalog (batch videos + index).

Example (overnight batch):
  python scripts/render_presentation_clips.py --start 0 --count 10 --fast
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("runs/presentation"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10, help="How many clips this invocation")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--write-catalog", action="store_true", help="Only write clips_100.json")
    args = parser.parse_args()

    from autopass.presentation_catalog import build_presentation_clips

    clips = build_presentation_clips()
    catalog_path = args.out_dir / "clips_100.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps([asdict(c) for c in clips], indent=2),
        encoding="utf-8",
    )
    print(f"[presentation] Wrote {catalog_path} ({len(clips)} clips)")
    if args.write_catalog:
        return 0

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
    os.environ["AUTOPASS_SKIP_STAGNANT_FRAMES"] = "1"
    map_name = showcase_map_for_environment("highway")
    os.environ["AUTOPASS_CARLA_MAP"] = map_name
    if args.fast:
        os.environ["AUTOPASS_VIDEO_REALTIME"] = "0"

    demos = curated_demo_scenarios()
    index_rows = []
    end = min(len(clips), args.start + args.count)
    for clip in clips[args.start:end]:
        spec = replace(demos[clip.demo_index], route=replace(demos[clip.demo_index].route, town=map_name))
        sub = args.out_dir / clip.clip_id
        sub.mkdir(parents=True, exist_ok=True)
        print(f"\n[presentation] {clip.clip_id}: {clip.audience_hook}")
        world = initialize_world(spec)
        state = run_agentic_carla_loop(
            spec,
            world,
            sub,
            ticks_per_step=6 if args.fast else 10,
            max_steps=clip.max_execute_steps,
            policy=clip.policy,
            mission_urgency=clip.urgency,
        )
        metrics = state.get("metrics", {})
        video = sub / f"{spec.scenario_id}_highway_agentic.mp4"
        row = {
            "clip_id": clip.clip_id,
            "title": clip.title,
            "hook": clip.audience_hook,
            "claim_axis": clip.claim_axis,
            "video": str(video) if video.exists() else None,
            "failure_type": metrics.get("failure_type"),
            "route_completed": metrics.get("route_completed"),
            "collision": metrics.get("collision"),
        }
        index_rows.append(row)
        print(f"  -> {row['failure_type']} collision={row['collision']} video={row['video']}")

    idx_path = args.out_dir / f"render_index_{args.start}_{end}.json"
    idx_path.write_text(json.dumps(index_rows, indent=2), encoding="utf-8")
    print(f"\n[presentation] Index: {idx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
