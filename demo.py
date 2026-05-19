#!/usr/bin/env python3
"""
Student demo launcher — AutoPass integrated LangGraph + visual perception.

Usage:
  py -3 demo.py --mode quick          # ~30s, no API key (mock LLMs)
  py -3 demo.py --mode visual         # your visual closed-loop + saved frames
  py -3 demo.py --mode multi_agent    # friend's full graph with visual scenarios
  py -3 demo.py --mode all            # both pipelines
  py -3 -m pytest -q                  # correctness tests
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agents"))

os.environ.setdefault("AUTOPASS_MOCK_LLM", "1")


def run_visual_demo(out_dir: Path, n: int) -> None:
    from autopass_langgraph_demo import curated_demo_scenarios, run_batch

    specs = curated_demo_scenarios()[:n]
    rows = run_batch(specs, ["autopass"], out_dir / "visual")
    print("\n=== Visual LangGraph (RGB + segmentation + depth) ===")
    for r in rows:
        print(f"  {r['scenario_id']}: {r['failure_type']}, passes={r['approved_passes']}, time={r['time_to_goal_s']}s")
    print(f"  Frames: {out_dir / 'visual' / 'frames'}")


def run_multi_agent_demo(out_dir: Path, scenario_idx: int) -> None:
    from langgraph.checkpoint.memory import MemorySaver

    from agents.autopassing import build_autopassing_graph
    from perception.context import set_context
    from visual_world import curated_demo_scenarios, initialize_world

    spec = curated_demo_scenarios()[scenario_idx]
    world = initialize_world(spec)
    set_context(spec, world, backend=os.environ.get("AUTOPASS_PERCEPTION_BACKEND", "visual"))

    app = build_autopassing_graph(checkpointer=MemorySaver())
    init = {
        "travel_request": f"Take me from downtown to {spec.request.goal}, I'm in a hurry",
        "navigation_plan": [],
        "passing_signal": "",
        "passing_instructions": {"overtake_maneuver": [], "merge_back_maneuver": []},
        "visual_scenario": {"spec": asdict(spec), "world": asdict(world), "backend": "visual"},
        "messages": [],
    }
    cfg = {"configurable": {"thread_id": f"demo-{spec.scenario_id}"}}
    result = app.invoke(init, config=cfg)

    trace_path = out_dir / "multi_agent" / f"{spec.scenario_id}_messages.txt"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [getattr(m, "content", str(m)) for m in result.get("messages", [])]
    trace_path.write_text("\n\n---\n\n".join(lines), encoding="utf-8")

    print("\n=== Multi-Agent LangGraph (paper flow + redesign LLMs) ===")
    print(f"  Scenario: {spec.scenario_id}")
    print(f"  Goal: {result.get('goal')} | Aggression: {result.get('aggressive_level')}")
    print(f"  Arrived: {result.get('arrived')} | Waypoints: {len(result.get('navigation_plan', []))}")
    print(f"  Last signal: {result.get('passing_signal', '')} | Maneuver: {result.get('maneuver_state', 'normal')}")
    print(f"  Trace: {trace_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoPass-Gen student demo")
    parser.add_argument("--mode", choices=["quick", "visual", "multi_agent", "all"], default="quick")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/demo"))
    parser.add_argument("--scenario", type=int, default=0, help="Curated scenario index for multi_agent")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--carla", action="store_true", help="Use live CARLA sensors (Python 3.7 egg)")
    args = parser.parse_args()

    if args.carla:
        os.environ["AUTOPASS_PERCEPTION_BACKEND"] = "carla"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("AutoPass-Gen Demo")
    print(f"  Mock LLMs: {os.environ.get('AUTOPASS_MOCK_LLM', '1')}")
    print(f"  Perception: {os.environ.get('AUTOPASS_PERCEPTION_BACKEND', 'visual')}")
    print(f"  Output: {args.out_dir}\n")

    if args.mode in ("quick", "all", "visual"):
        run_visual_demo(args.out_dir, min(args.n, 3))
    if args.mode in ("quick", "all", "multi_agent"):
        run_multi_agent_demo(args.out_dir, args.scenario)


if __name__ == "__main__":
    main()
