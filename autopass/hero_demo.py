"""
CARLA hero pass demo — full LangGraph agentic loop with video and trace output.

Uses the same closed-loop path as demo_carla_watch.py --hero-pass (planner → tools →
critic → CARLA VehicleControl → belief refresh), not a scripted pass FSM.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from autopass.benchmark_catalog import BenchmarkCase, FAMILY_TO_DEMO_ID, UrgencyLevel, apply_urgency
from autopass.benchmark_metrics import derive_run_metrics
from autopass.scenarios import assert_carla_environment_allowed, showcase_map_for_environment
from visual_world import ScenarioSpec, WorldState, initialize_world, spec_to_dict


def resolve_hero_scenario(
    scenario_family: str,
    *,
    urgency: UrgencyLevel = "high",
    environment: str = "highway",
) -> Tuple[ScenarioSpec, BenchmarkCase]:
    from visual_world import curated_demo_scenarios

    demos = {s.scenario_id: s for s in curated_demo_scenarios()}
    demo_id = FAMILY_TO_DEMO_ID.get(scenario_family, scenario_family)
    if demo_id not in demos:
        raise ValueError(f"Unknown hero scenario family: {scenario_family}")
    base = demos[demo_id]
    spec = apply_urgency(base, urgency)
    if environment != "synthetic":
        map_name = showcase_map_for_environment(environment)
        case_id = f"{environment}_{scenario_family}_{urgency}"
        spec = replace(
            spec,
            scenario_id=case_id,
            route=replace(spec.route, town=map_name),
        )
    else:
        case_id = f"{scenario_family}_{urgency}"
        spec = replace(spec, scenario_id=case_id)
    case = BenchmarkCase(
        scenario_id=spec.scenario_id,
        scenario_family=scenario_family,
        urgency=urgency,
        environment=environment,
        spec=spec,
        base_demo_id=demo_id,
    )
    return spec, case


def format_action_timeline(trace: List[Dict[str, Any]]) -> str:
    lines = ["# Hero pass action timeline (LangGraph agentic loop)", ""]
    for i, entry in enumerate(trace):
        node = entry.get("node", "?")
        if node == "execute":
            action = entry.get("action", "?")
            phase = entry.get("pass_phase") or entry.get("scripted_phase") or ""
            lane = entry.get("ego_lane_id", entry.get("ego_lane", ""))
            speed = entry.get("speed_mps", entry.get("measured_speed_mps", ""))
            steer = entry.get("steer", "")
            lead = entry.get("lead_gap_m", entry.get("belief", {}).get("front_gap_m", ""))
            lines.append(
                f"{i:3d} EXECUTE {action:6s} phase={phase!s:12s} lane={lane} "
                f"v={speed} steer={steer} lead_gap={lead}"
            )
        elif node in ("planner", "run_tool", "critique_maneuver", "critique_tool", "critique_post_exec"):
            detail = (
                entry.get("decision")
                or entry.get("tool")
                or entry.get("maneuver")
                or entry.get("verdict")
                or entry.get("label", "")
            )
            lines.append(f"{i:3d} {node.upper():16s} {detail}")
        else:
            lines.append(f"{i:3d} {node.upper():16s} {entry}")
    return "\n".join(lines) + "\n"


def run_hero_pass_demo(
    spec: ScenarioSpec,
    case: BenchmarkCase,
    *,
    policy: str = "autopass",
    out_dir: Path,
    ticks_per_step: float = 0.05,
    max_steps: int = 40,
) -> Dict[str, Any]:
    from demo_carla_watch import run_agentic_carla_loop

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AUTOPASS_CARLA_SKIP_PASS_BOOT_VALIDATE", "1")
    os.environ.setdefault("AUTOPASS_CONTROL_MODE", "vehicle")
    os.environ.setdefault("AUTOPASS_EXECUTE_DT_S", "0.35")
    os.environ.setdefault("AUTOPASS_VIDEO_REALTIME", "1")

    map_name = spec.route.town or showcase_map_for_environment(case.environment)
    os.environ["AUTOPASS_CARLA_MAP"] = map_name
    os.environ["AUTOPASS_ENVIRONMENT"] = case.environment
    assert_carla_environment_allowed(case.environment)

    world = initialize_world(spec)
    ticks = max(4, int(round(float(ticks_per_step) * 200)))

    final_state = run_agentic_carla_loop(
        spec,
        world,
        out_dir,
        ticks_per_step=ticks,
        max_steps=max_steps,
        policy=policy,
    )

    trace = final_state.get("trace", [])
    dsl = final_state.get("dsl", {})
    sim_world = WorldState(**final_state["world"]) if "world" in final_state else world

    metrics_row = derive_run_metrics(
        case,
        policy,
        {
            "world": asdict(sim_world),
            "trace": trace,
            "dsl": dsl,
            "metrics": final_state.get("metrics", {}),
        },
    )

    timeline = format_action_timeline(trace)
    (out_dir / f"{spec.scenario_id}_{policy}_timeline.txt").write_text(timeline, encoding="utf-8")

    video_stem = f"{spec.scenario_id}_{case.environment}_agentic"
    composite = out_dir / f"{video_stem}.mp4"
    if not composite.is_file():
        composite = out_dir / f"{spec.scenario_id}_{policy}_hero.mp4"

    trace_doc = {
        "scenario_id": spec.scenario_id,
        "scenario_family": case.scenario_family,
        "policy": policy,
        "urgency": case.urgency,
        "environment": case.environment,
        "spec": spec_to_dict(spec),
        "metrics": metrics_row,
        "agentic_metrics": final_state.get("metrics", {}),
        "trace": trace,
        "dsl": dsl,
        "videos": {"composite": str(composite) if composite.is_file() else None},
        "loop": "langgraph_agentic",
    }
    trace_path = out_dir / f"{spec.scenario_id}_{policy}_trace.json"
    trace_path.write_text(json.dumps(trace_doc, indent=2), encoding="utf-8")

    summary_path = out_dir / f"{spec.scenario_id}_{policy}_summary.json"
    summary_path.write_text(json.dumps(metrics_row, indent=2), encoding="utf-8")

    print(f"[HERO] policy={policy} urgency={case.urgency} pass_attempts={metrics_row.get('pass_attempts')}")
    print(f"[HERO] passed={sim_world.passed} collision={sim_world.collision}")
    if composite.is_file():
        print(f"[HERO] Video: {composite}")
    print(f"[HERO] Trace: {trace_path}")
    print(f"[HERO] Timeline: {out_dir / f'{spec.scenario_id}_{policy}_timeline.txt'}")
    print(f"[HERO] Summary: {summary_path}")

    return trace_doc
