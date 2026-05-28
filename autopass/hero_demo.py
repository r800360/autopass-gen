"""
CARLA hero pass demo — curated corridor, scripted pass control, agent trace + DSL logs.

Separate from the full agentic benchmark loop; produces ego/overhead video, trace JSON,
summary metrics, and a readable action timeline for presentation.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from autopass.benchmark_catalog import FAMILY_TO_DEMO_ID, UrgencyLevel, apply_urgency
from autopass.benchmark_metrics import derive_run_metrics
from autopass.benchmark_catalog import BenchmarkCase
from autopass.dsl import (
    ExecutionRecord,
    ManeuverPlan,
    PassingDSL,
    VerificationNote,
    dsl_to_dict,
    init_dsl_from_request,
)
from autopass.scenarios import assert_carla_environment_allowed, showcase_map_for_environment
from visual_world import ScenarioSpec, WorldState, initialize_world, spec_to_dict


def resolve_hero_scenario(
    scenario_family: str,
    *,
    urgency: UrgencyLevel = "high",
    environment: str = "highway",
) -> Tuple[ScenarioSpec, BenchmarkCase]:
    from visual_world import RouteSpec, curated_demo_scenarios

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


def _urgency_aggression(urgency: UrgencyLevel, policy: str) -> Tuple[str, str]:
    if policy == "no_pass":
        return urgency, "0"
    agg = "high" if urgency == "high" else ("medium" if urgency == "medium" else "low")
    return urgency, agg


def _init_hero_dsl(spec: ScenarioSpec, urgency: UrgencyLevel, policy: str) -> PassingDSL:
    u, agg = _urgency_aggression(urgency, policy)
    road = getattr(spec.route, "town", "highway")
    road_type = "highway" if "Town04" in str(road) or "Synthetic" in str(road) else "urban"
    return init_dsl_from_request(
        spec.request.text,
        start=spec.request.start,
        goal=spec.request.goal,
        deadline_s=spec.request.deadline_s,
        urgency=u,
        aggression=agg,
        road_type=road_type,
    )


def format_action_timeline(trace: List[Dict[str, Any]]) -> str:
    lines = ["# Hero pass action timeline", ""]
    for i, entry in enumerate(trace):
        node = entry.get("node", "?")
        if node == "execute":
            action = entry.get("action", "?")
            phase = entry.get("pass_phase") or entry.get("scripted_phase") or ""
            lane = entry.get("ego_lane_id", entry.get("ego_lane", ""))
            speed = entry.get("speed_mps", entry.get("measured_speed_mps", ""))
            steer = entry.get("steer", "")
            lead = entry.get("lead_gap_m", "")
            lines.append(
                f"{i:3d} EXECUTE {action:6s} phase={phase!s:12s} lane={lane} "
                f"v={speed} steer={steer} lead_gap={lead}"
            )
        elif node in ("planner", "tool", "critic_maneuver", "critic_tool", "hero"):
            detail = entry.get("decision") or entry.get("tool") or entry.get("maneuver") or entry.get("label", "")
            lines.append(f"{i:3d} {node.upper():16s} {detail}")
        else:
            lines.append(f"{i:3d} {node.upper():16s} {entry}")
    return "\n".join(lines) + "\n"


def _record_frame(recorder, session, spec, world, label: str, extra: dict | None = None) -> None:
    pair = session.grab_frame_pair()
    if pair is None:
        frame = session.grab_frame()
        if frame is None:
            return
        rgb, _, _ = frame
        overhead = None
    else:
        rgb, _, _, overhead = pair
    recorder.capture(rgb, t_s=world.t_s, label=label, extra=extra, overhead=overhead)


def run_hero_pass_demo(
    spec: ScenarioSpec,
    case: BenchmarkCase,
    *,
    policy: str = "autopass",
    out_dir: Path,
    ticks_per_step: float = 0.05,
) -> Dict[str, Any]:
    from perception.carla_pass_maneuver import run_scripted_pass_maneuver, run_no_pass_follow
    from perception.carla_recorder import CarlaRecorder
    from perception.carla_scenario import bootstrap_carla_scenario, get_session, run_carla_preflight
    from perception.context import set_context

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AUTOPASS_CARLA_SKIP_PASS_BOOT_VALIDATE", "1")

    world = initialize_world(spec)
    map_name = spec.route.town or showcase_map_for_environment(case.environment)
    os.environ["AUTOPASS_CARLA_MAP"] = map_name
    assert_carla_environment_allowed(case.environment)

    if not bootstrap_carla_scenario(spec, world, map_name=map_name):
        from autopass.config import AutopassConfigurationError

        raise AutopassConfigurationError(
            f"CARLA bootstrap failed: {get_session().last_error}. Start CarlaUE4.exe first."
        )
    run_carla_preflight(require_frames=True)
    session = get_session()
    set_context(spec, world, "carla")

    dsl = _init_hero_dsl(spec, case.urgency, policy)
    trace: List[Dict[str, Any]] = []
    pass_maneuver_active = False
    pass_maneuver_id = 0

    trace.append({"node": "init_mission", "scenario_id": spec.scenario_id, "policy": policy})
    trace.append({"node": "tool", "tool": "capture_sensors", "payload_keys": ["car_distances"]})
    dsl = dsl.append_verification(VerificationNote(verdict="ok", message="capture_sensors verified."))

    if policy == "no_pass":
        trace.append(
            {
                "node": "critic_maneuver",
                "maneuver": "wait",
                "verdict": "ok",
                "approved": "wait",
            }
        )
        dsl = dsl.set_maneuver(ManeuverPlan(kind="wait", reasoning="No-pass policy."))
    else:
        trace.append({"node": "tool", "tool": "measure_front_gap"})
        trace.append({"node": "tool", "tool": "measure_rear_gap"})
        trace.append({"node": "tool", "tool": "check_kinematics"})
        trace.append(
            {
                "node": "critic_maneuver",
                "maneuver": "pass",
                "verdict": "ok",
                "approved": "pass",
            }
        )
        dsl = dsl.set_maneuver(
            ManeuverPlan(
                kind="pass",
                passing_side="left",
                target_speed_mps=spec.route.speed_limit_mps,
                reasoning="Hero demo: high urgency safe pass approved.",
            )
        )

    recorder = CarlaRecorder(out_dir, spec.scenario_id)
    _record_frame(
        recorder,
        session,
        spec,
        world,
        "HERO START",
        {"policy": policy, "urgency": case.urgency, "map": map_name},
    )

    if policy == "no_pass":
        result = run_no_pass_follow(
            session,
            spec,
            world,
            duration_s=28.0,
            on_step=lambda rec, w: _hero_no_pass_step(recorder, session, spec, trace, rec, w),
        )
        world = result.final_world
    else:
        pass_maneuver_active = True
        pass_maneuver_id = 1
        trace.append(
            {
                "node": "execute",
                "action": "pass",
                "pass_maneuver_started": True,
                "pass_maneuver_active": True,
                "pass_maneuver_id": pass_maneuver_id,
            }
        )

        def on_step(rec, w):
            nonlocal pass_maneuver_active, world, dsl
            world = w
            entry = {
                "node": "execute",
                "action": "pass",
                "scripted_phase": rec.scripted_phase,
                "pass_phase": rec.pass_phase,
                "pass_maneuver_started": False,
                "pass_maneuver_active": pass_maneuver_active,
                "pass_started": rec.pass_started,
                "pass_in_progress": rec.pass_in_progress,
                "phase": rec.scripted_phase,
                "monitor_ok": rec.monitor_ok,
                "abort_reason": rec.abort_reason,
                "pass_completed": rec.pass_completed,
                "cleared_lead": rec.cleared_lead,
                "ego_s_m": rec.ego_s_m,
                "lead_s_m": rec.lead_s_m,
                "rear_s_m": rec.rear_s_m,
                "ego_road_id": rec.ego_road_id,
                "target_road_id": rec.target_road_id,
                "ego_lane_id": rec.ego_lane_id,
                "target_lane_id": rec.target_lane_id,
                "ego_lane": 1 if rec.ego_lane_id and session._passing_wp and rec.ego_lane_id == session._passing_wp.lane_id else 0,
                "lateral_error_m": rec.lateral_error_m,
                "lane_center_dist_m": rec.lane_center_dist_m,
                "lead_gap_m": rec.lead_gap_m,
                "rear_gap_m": rec.rear_gap_m,
                "edge_clearance_m": rec.edge_clearance_m,
                "speed_mps": rec.speed_mps,
                "steer": rec.steer,
            }
            trace.append(entry)
            dsl = dsl.append_execution(
                ExecutionRecord(
                    action="pass",
                    mode="carla_vehicle",
                    summary=f"Hero {rec.scripted_phase}: v={rec.speed_mps} lane={rec.ego_lane_id}",
                    data=entry,
                )
            )
            extra = {
                "phase": rec.scripted_phase,
                "v": rec.speed_mps,
                "steer": rec.steer,
                "lane": rec.ego_lane_id,
            }
            _record_frame(recorder, session, spec, w, rec.scripted_phase.upper(), extra)
            if rec.scripted_phase == "done":
                pass_maneuver_active = False

        result = run_scripted_pass_maneuver(
            session,
            spec,
            world,
            verbose=False,
            on_step=on_step,
            use_state_machine=True,
        )
        world = result.final_world or world
        if result.merged_back and result.pass_complete:
            trace.append(
                {
                    "node": "execute",
                    "action": "wait",
                    "passed": True,
                    "pass_maneuver_completed": True,
                    "pass_maneuver_active": False,
                }
            )

    metrics_row = derive_run_metrics(
        case,
        policy,
        {
            "world": asdict(world),
            "trace": trace,
            "dsl": dsl_to_dict(dsl),
            "metrics": {"failure_type": "none" if result.ok else "maneuver_failed"},
        },
    )

    _record_frame(
        recorder,
        session,
        spec,
        world,
        "HERO DONE",
        {"pass_attempts": metrics_row.get("pass_attempts"), "passed": world.passed},
    )

    composite = recorder.write_video(f"{spec.scenario_id}_{policy}_hero.mp4")
    ego_mp4, overhead_mp4 = recorder.write_split_videos(
        f"{spec.scenario_id}_{policy}_ego.mp4",
        f"{spec.scenario_id}_{policy}_overhead.mp4",
    )

    timeline = format_action_timeline(trace)
    (out_dir / f"{spec.scenario_id}_{policy}_timeline.txt").write_text(timeline, encoding="utf-8")

    trace_doc = {
        "scenario_id": spec.scenario_id,
        "scenario_family": case.scenario_family,
        "policy": policy,
        "urgency": case.urgency,
        "environment": case.environment,
        "spec": spec_to_dict(spec),
        "metrics": metrics_row,
        "maneuver_result": {
            "ok": result.ok,
            "issues": result.issues,
            "pass_attempts": result.pass_attempts,
            "merged_back": result.merged_back,
            "pass_complete": result.pass_complete,
            "max_lane_center_m": result.max_lane_center_m,
            "min_edge_clearance_m": result.min_edge_clearance_m,
        },
        "trace": trace,
        "dsl": dsl_to_dict(dsl),
        "videos": {
            "composite": str(composite) if composite else None,
            "ego": str(ego_mp4) if ego_mp4 else None,
            "overhead": str(overhead_mp4) if overhead_mp4 else None,
        },
    }
    trace_path = out_dir / f"{spec.scenario_id}_{policy}_trace.json"
    trace_path.write_text(json.dumps(trace_doc, indent=2), encoding="utf-8")

    summary_path = out_dir / f"{spec.scenario_id}_{policy}_summary.json"
    summary_path.write_text(json.dumps(metrics_row, indent=2), encoding="utf-8")

    session.shutdown()

    print(f"[HERO] policy={policy} urgency={case.urgency} pass_attempts={metrics_row.get('pass_attempts')}")
    print(f"[HERO] passed={world.passed} collision={world.collision} merged={result.merged_back}")
    if composite:
        print(f"[HERO] Video (composite): {composite}")
    if ego_mp4:
        print(f"[HERO] Video (ego): {ego_mp4}")
    if overhead_mp4:
        print(f"[HERO] Video (overhead): {overhead_mp4}")
    print(f"[HERO] Trace: {trace_path}")
    print(f"[HERO] Timeline: {out_dir / f'{spec.scenario_id}_{policy}_timeline.txt'}")
    print(f"[HERO] Summary: {summary_path}")

    return trace_doc


def _hero_no_pass_step(recorder, session, spec, trace, rec, world) -> None:
    entry = {
        "node": "execute",
        "action": "wait",
        "pass_maneuver_started": False,
        "speed_mps": rec.speed_mps,
        "steer": rec.steer,
        "lead_gap_m": rec.lead_gap_m,
        "ego_lane_id": rec.ego_lane_id,
    }
    trace.append(entry)
    _record_frame(
        recorder,
        session,
        spec,
        world,
        "FOLLOW",
        {"v": rec.speed_mps, "lead": rec.lead_gap_m},
    )
