#!/usr/bin/env python3
"""
Student demo launcher — AutoPass integrated LangGraph + visual perception.

Production defaults: real OpenAI LLM + CARLA perception + vehicle control.
Offline tests use AUTOPASS_TEST_MODE=1 (set automatically by pytest).

Usage:
  py -3 demo.py --mode quick          # visual closed-loop (requires OPENAI_API_KEY)
  py -3 demo.py --mode visual         # saved frames under runs/demo
  py -3 demo.py --mode multi_agent    # full agentic graph
  py -3 demo.py --carla               # live CARLA (requires CarlaUE4.exe)
  py -3 demo_carla_watch.py           # CARLA video closed-loop (production path)
  py -3 -m pytest -q                  # offline correctness tests
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

from autopass.config import apply_production_defaults, require_runtime

apply_production_defaults()


def run_visual_demo(out_dir: Path, n: int) -> None:
    from autopass_langgraph_demo import curated_demo_scenarios, run_batch

    specs = curated_demo_scenarios()[:n]
    rows = run_batch(specs, ["autopass"], out_dir / "visual")
    print("\n=== Visual LangGraph (RGB + segmentation + depth) ===")
    for r in rows:
        print(f"  {r['scenario_id']}: {r['failure_type']}, passes={r['approved_passes']}, time={r['time_to_goal_s']}s")
    print(f"  Frames: {out_dir / 'visual' / 'frames'}")


def run_multi_agent_demo(out_dir: Path, scenario_idx: int, carla_map: str = "Town04") -> None:
    from autopass.graph import run_agentic_episode
    from perception.context import set_context
    from visual_world import curated_demo_scenarios, initialize_world

    spec = curated_demo_scenarios()[scenario_idx]
    world = initialize_world(spec)
    backend = os.environ.get("AUTOPASS_PERCEPTION_BACKEND", "visual")
    set_context(spec, world, backend=backend)

    if backend == "carla":
        from autopass.config import AutopassConfigurationError
        from perception.carla_scenario import bootstrap_carla_scenario

        if not bootstrap_carla_scenario(spec, world, map_name=carla_map):
            raise AutopassConfigurationError(
                "CARLA bootstrap failed. Start CarlaUE4.exe, verify pip install carla==0.9.16, "
                "then run: python carla_smoke.py"
            )

    result = run_agentic_episode(spec, policy="autopass", perception_backend=backend, max_drive_steps=30)

    trace_path = out_dir / "multi_agent" / f"{spec.scenario_id}_agentic_trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    trace_path.write_text(json.dumps(result.get("trace", []), indent=2), encoding="utf-8")
    dsl_path = out_dir / "multi_agent" / f"{spec.scenario_id}_dsl.json"
    dsl_path.write_text(json.dumps(result.get("dsl", {}), indent=2), encoding="utf-8")

    m = result.get("metrics", {})
    print("\n=== Agentic LangGraph (planner / tools / critic / DSL) ===")
    print(f"  Scenario: {spec.scenario_id}")
    print(f"  Failure: {m.get('failure_type')} | DSL revision: {result.get('dsl', {}).get('revision')}")
    print(f"  Planner rounds: {m.get('planner_rounds')} | Perception tools: {len(result.get('dsl', {}).get('perception_log', []))}")
    print(f"  Trace: {trace_path}")
    print(f"  DSL: {dsl_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoPass-Gen student demo")
    parser.add_argument(
        "--mode",
        choices=["quick", "visual", "multi_agent", "all"],
        default="quick",
        help="Use demo_carla_watch.py for CARLA video closed-loop",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/demo"))
    parser.add_argument("--scenario", type=int, default=0, help="Curated scenario index for multi_agent")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--carla", action="store_true", help="Use live CARLA (spawn scenario in simulator)")
    parser.add_argument(
        "--carla-map",
        default=os.environ.get("AUTOPASS_CARLA_MAP", "Town04"),
        help="CARLA map for highway demo (Town04 recommended)",
    )
    args = parser.parse_args()

    require_runtime(need_carla=args.carla, need_openai=True)

    if args.carla:
        os.environ["AUTOPASS_PERCEPTION_BACKEND"] = "carla"
        os.environ["AUTOPASS_CONTROL_MODE"] = "vehicle"
        os.environ["AUTOPASS_CARLA_MAP"] = args.carla_map

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("AutoPass-Gen Demo")
    from autopass.config import control_mode, get_perception_backend, mock_llm_enabled

    print(f"  Mock LLMs: {mock_llm_enabled()}")
    print(f"  Perception: {get_perception_backend()}")
    print(f"  Control: {control_mode()}")
    if args.carla:
        print(f"  CARLA map: {args.carla_map}  (start CarlaUE4.exe FIRST)")
    print(f"  Output: {args.out_dir}\n")

    if args.mode in ("quick", "all", "visual"):
        run_visual_demo(args.out_dir, min(args.n, 3))
    if args.mode in ("quick", "all", "multi_agent"):
        run_multi_agent_demo(args.out_dir, args.scenario, carla_map=args.carla_map)

    if args.carla:
        print(
            "\n[CARLA] demo.py only spawns the scene and runs the graph once (no step-by-step motion or video).\n"
            "        For the full closed loop in the simulator + MP4:\n"
            "          python demo_carla_watch.py --scenario 0 --steps 40\n"
            "          python demo_carla_watch.py --hero-pass --scenario clear_safe_pass --steps 40\n"
            "        Output videos: runs/carla_watch/*.mp4\n"
        )


if __name__ == "__main__":
    main()
