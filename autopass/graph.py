"""
Unified agentic LangGraph — planner / tools / critic / executor / replan.

Every pass/wait/replan is justified by vision evidence; no planner/critic bypass
during pass execution.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Dict, List, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from autopass.critic import critique_maneuver_proposal, critique_post_execution, critique_tool_result
from autopass.dsl import ManeuverPlan, PassingDSL, dsl_from_dict, dsl_to_dict, init_dsl_from_request
from autopass.learning import mutate_from_failure
from autopass.perception_state import (
    MAX_UNRESOLVED_FRONT_RESENSE,
    deadline_pressure,
    sync_world_from_belief,
    urgency_level,
)
from autopass.planner import plan_next
from autopass.tools import run_tool
from visual_world import ScenarioSpec, WorldState, dict_to_spec, initialize_world, spec_to_dict


class AgenticState(TypedDict, total=False):
    spec: Dict[str, Any]
    world: Dict[str, Any]
    dsl: Dict[str, Any]
    policy: str
    trace: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    planner_rounds: int
    phase: str
    last_verdict: str
    last_tool: str
    max_planner_rounds: int
    max_drive_steps: int
    perception_backend: str
    proposed_maneuver: str
    approved_maneuver: str
    tool_payload: Dict[str, Any]
    pass_in_progress: bool
    pass_maneuver_id: int
    learning_round: int
    mutated_spec: Dict[str, Any]
    insufficient_counts_by_tool: Dict[str, int]
    last_insufficient_tool: str
    perception_retry_count: int
    fallback_reason: str
    measure_front_insufficient_streak: int
    capture_refresh_seq: int
    unresolved_front_resense_count: int


def _spec(state: AgenticState) -> ScenarioSpec:
    return dict_to_spec(state["spec"])


def _world(state: AgenticState) -> WorldState:
    return WorldState(**state["world"])


def _dsl(state: AgenticState) -> PassingDSL:
    return dsl_from_dict(state["dsl"])


def node_init_mission(state: AgenticState) -> Dict[str, Any]:
    spec = _spec(state)
    world = _world(state)
    u = urgency_level(spec, world)
    aggression = "high" if u == "high" else "low"
    if state.get("policy") == "no_pass":
        aggression = "0"
    road = getattr(spec.route, "town", "highway")
    road_type = "highway" if "Town04" in str(road) or "Synthetic" in str(road) else (
        "urban" if "Town03" in str(road) else "local"
    )
    dsl = init_dsl_from_request(
        spec.request.text,
        start=spec.request.start,
        goal=spec.request.goal,
        deadline_s=spec.request.deadline_s,
        urgency=u,
        aggression=aggression,
        road_type=road_type,
    )
    return {
        "dsl": dsl_to_dict(dsl),
        "phase": "plan",
        "planner_rounds": 0,
        "last_verdict": "",
        "trace": [],
        "metrics": {},
        "pass_in_progress": False,
        "pass_maneuver_id": 0,
        "learning_round": state.get("learning_round", 0),
        "insufficient_counts_by_tool": {},
        "last_insufficient_tool": "",
        "perception_retry_count": 0,
        "fallback_reason": "",
        "measure_front_insufficient_streak": 0,
        "capture_refresh_seq": 0,
        "unresolved_front_resense_count": 0,
    }


def node_planner(state: AgenticState) -> Dict[str, Any]:
    spec = _spec(state)
    world = _world(state)
    dsl = _dsl(state)
    rounds = state.get("planner_rounds", 0) + 1
    if rounds > state.get("max_planner_rounds", 12) * 4:
        return {"phase": "done", "planner_rounds": rounds, "trace": list(state.get("trace", []))}

    pass_active = bool(state.get("pass_in_progress", False))
    insufficient = dict(state.get("insufficient_counts_by_tool", {}))
    mf_streak = int(state.get("measure_front_insufficient_streak", 0))
    unresolved = int(state.get("unresolved_front_resense_count", 0))
    fallback_reason = state.get("fallback_reason", "") or ""
    from autopass.config import decision_oracle_enabled

    decision = None
    forced_phase = ""
    forced_tool = ""
    agency_source = "llm" if not __import__("agents.llm_agents", fromlist=["use_mock_llm"]).use_mock_llm() else "rules"

    if unresolved >= MAX_UNRESOLVED_FRONT_RESENSE:
        forced_phase = "decide"
        fallback_reason = (
            f"front perception unresolved after {unresolved} resense attempts"
        )
    elif mf_streak >= 2:
        capture_insuf = int(insufficient.get("capture_sensors", 0))
        if capture_insuf < 2 and unresolved < MAX_UNRESOLVED_FRONT_RESENSE:
            forced_phase = "tool"
            forced_tool = "capture_sensors"
            fallback_reason = "measure_front insufficient repeatedly; forcing re-sense"
        else:
            forced_phase = "decide"
            fallback_reason = "insufficient validated front perception; safe follow_lead"

    block_front = unresolved >= MAX_UNRESOLVED_FRONT_RESENSE or mf_streak >= 2
    if decision is None and not forced_phase:
        decision = plan_next(
            dsl,
            spec,
            world,
            max_tool_rounds=state.get("max_planner_rounds", 12),
            pass_in_progress=pass_active,
            block_front_measure=block_front,
        )
        from autopass.perception_state import needed_tools
        from autopass.planner import PlannerDecision

        missing = needed_tools(dsl, spec, world, block_front_measure=block_front)
        if missing and decision.action == "run_tool":
            required_tool, required_why = missing[0]
            if decision.tool != required_tool:
                decision = PlannerDecision(
                    action="run_tool",
                    tool=required_tool,
                    reasoning=f"Enforcing required tool order: {required_why}",
                )
    from autopass.pass_gates import evaluate_pass_gates
    from autopass.tools import perception_summary

    gate_eval = evaluate_pass_gates(dsl, spec, world, summary=perception_summary(dsl))
    trace = list(state.get("trace", []))
    trace.append(
        {
            "node": "planner",
            "decision": decision.model_dump() if decision is not None else {
                "action": "run_tool" if forced_phase == "tool" else "decide_maneuver",
                "tool": forced_tool or None,
                "maneuver": "wait" if forced_phase == "decide" else None,
                "reasoning": fallback_reason,
            },
            "revision": dsl.revision,
            "pass_in_progress": pass_active,
            "insufficient_counts_by_tool": insufficient,
            "last_insufficient_tool": state.get("last_insufficient_tool", ""),
            "perception_retry_count": state.get("perception_retry_count", 0),
            "unresolved_front_resense_count": unresolved,
            "fallback_reason": fallback_reason,
            "belief_source": dsl.world_belief.source,
            "front_gap_m": dsl.world_belief.front_gap_m,
            "front_valid": dsl.world_belief.front_valid,
            "lead_speed_mps": dsl.world_belief.lead_speed_mps,
            "lead_speed_valid": dsl.world_belief.front_valid and dsl.world_belief.lead_speed_mps is not None,
            "rear_gap_m": dsl.world_belief.rear_gap_m,
            "rear_valid": dsl.world_belief.rear_valid,
            "oncoming_gap_m": dsl.world_belief.oncoming_gap_m,
            "oncoming_valid": dsl.world_belief.oncoming_valid,
            "oncoming_available": dsl.world_belief.oncoming_available,
            "oncoming_unavailable_reason": dsl.world_belief.oncoming_unavailable_reason,
            "depth_confidence": dsl.world_belief.depth_confidence,
            "car_distances_count": len(dsl.world_belief.car_distances),
            "pass_preconditions": gate_eval.get("pass_preconditions"),
            "pass_blockers": gate_eval.get("pass_blockers"),
            "pass_motivation": gate_eval.get("pass_motivation"),
            "can_pass": gate_eval.get("can_pass"),
            "oncoming_required": gate_eval.get("oncoming_required"),
            "passing_topology": gate_eval.get("passing_topology"),
            "decision_rule_source": gate_eval.get("decision_rule_source"),
            "agency_source": agency_source,
            "decision_oracle_enabled": decision_oracle_enabled(),
            "belief_calibrated_source": (dsl.world_belief.car_distances[0].get("calibrated_gap_source") if dsl.world_belief.car_distances else None),
        }
    )
    out: Dict[str, Any] = {
        "planner_rounds": rounds,
        "trace": trace,
        "phase": decision.action if decision is not None else ("run_tool" if forced_phase == "tool" else "decide"),
        "last_tool": (decision.tool or "") if decision is not None else forced_tool,
        "fallback_reason": fallback_reason,
    }
    if decision is None and forced_phase == "tool":
        out["phase"] = "tool"
        out["last_tool"] = "capture_sensors"
    elif decision is None and forced_phase == "decide":
        out["phase"] = "decide"
        out["proposed_maneuver"] = "wait"
    elif decision.action == "run_tool" and decision.tool:
        out["phase"] = "tool"
        out["last_tool"] = decision.tool
    elif decision.action == "decide_maneuver":
        out["phase"] = "decide"
        out["proposed_maneuver"] = decision.maneuver or "wait"
    elif decision.action == "finish":
        out["phase"] = "done"
    return out


def node_run_tool(state: AgenticState) -> Dict[str, Any]:
    from autopass.config import get_perception_backend
    from autopass.perception_state import InsufficientPerceptionError
    from perception.context import set_context

    spec = _spec(state)
    world = _world(state)
    dsl = _dsl(state)
    tool = state.get("last_tool") or "capture_sensors"
    set_context(spec, world, state.get("perception_backend", get_perception_backend()))
    try:
        dsl, payload = run_tool(tool, dsl, spec, world)
    except InsufficientPerceptionError as e:
        wb = dsl.world_belief
        missing = []
        if not wb.front_valid or wb.front_gap_m is None:
            missing.append("front_gap_m")
        if wb.front_valid and wb.lead_speed_mps is None:
            missing.append("lead_speed_mps")
        payload = {
            "error_type": "insufficient_perception",
            "tool": tool,
            "message": str(e),
            "missing_fields": missing,
            "belief_source": wb.source,
            "front_valid": wb.front_valid,
            "rear_valid": wb.rear_valid,
            "oncoming_valid": wb.oncoming_valid,
            "oncoming_available": wb.oncoming_available,
            "oncoming_unavailable_reason": wb.oncoming_unavailable_reason,
            "recommendation": "resense" if tool != "capture_sensors" else "wait_for_valid_belief",
        }
    trace = list(state.get("trace", []))
    tool_trace: dict = {"node": "tool", "tool": tool, "payload_keys": list(payload.keys())}
    if tool == "capture_sensors":
        for key in ("actor_continuity_before", "actor_continuity_after"):
            if key in payload:
                tool_trace[key] = payload[key]
    trace.append(tool_trace)
    return {"dsl": dsl_to_dict(dsl), "trace": trace, "tool_payload": payload, "phase": "critique_tool"}


def node_critique_tool(state: AgenticState) -> Dict[str, Any]:
    spec = _spec(state)
    world = _world(state)
    dsl = _dsl(state)
    tool = state.get("last_tool", "")
    payload = state.get("tool_payload", {})
    dsl, verdict = critique_tool_result(dsl, tool, payload, spec, world)
    counts = dict(state.get("insufficient_counts_by_tool", {}))
    mf_streak = int(state.get("measure_front_insufficient_streak", 0))
    retry_count = int(state.get("perception_retry_count", 0))
    unresolved = int(state.get("unresolved_front_resense_count", 0))
    last_ins_tool = state.get("last_insufficient_tool", "")
    if verdict in ("insufficient", "reject"):
        counts[tool] = int(counts.get(tool, 0)) + 1
        retry_count += 1
        last_ins_tool = tool
        if tool == "measure_front_gap":
            mf_streak += 1
    elif verdict == "ok":
        counts[tool] = 0
        if tool == "capture_sensors":
            if dsl.world_belief.front_valid:
                mf_streak = 0
                retry_count = 0
                unresolved = 0
            else:
                unresolved += 1
        elif tool == "measure_front_gap" and dsl.world_belief.front_valid:
            mf_streak = 0
    trace = list(state.get("trace", []))
    trace.append(
        {
            "node": "critic_tool",
            "tool": tool,
            "verdict": verdict,
            "tool_payload_accepted": verdict == "ok",
        }
    )
    phase = "plan"
    if verdict == "reject":
        phase = "plan"
    return {
        "dsl": dsl_to_dict(dsl),
        "trace": trace,
        "last_verdict": verdict,
        "phase": phase,
        "insufficient_counts_by_tool": counts,
        "last_insufficient_tool": last_ins_tool,
        "perception_retry_count": retry_count,
        "measure_front_insufficient_streak": mf_streak,
        "unresolved_front_resense_count": unresolved,
    }


def node_critique_maneuver(state: AgenticState) -> Dict[str, Any]:
    from autopass.policy import clamp_maneuver_for_policy

    spec = _spec(state)
    world = _world(state)
    dsl = _dsl(state)
    maneuver = clamp_maneuver_for_policy(state.get("policy", "autopass"), state.get("proposed_maneuver", "wait"))
    if maneuver == "abort_pass":
        maneuver = "abort_pass"
    dsl, verdict, plan = critique_maneuver_proposal(dsl, maneuver, spec, world)
    if state.get("policy") == "no_pass":
        plan = replace(plan, kind="wait", reasoning="No-pass policy: wait only.")
        verdict = "ok"
    trace = list(state.get("trace", []))
    from autopass.pass_gates import evaluate_pass_gates
    from autopass.tools import perception_summary

    gate_eval = evaluate_pass_gates(dsl, spec, world, summary=perception_summary(dsl))
    trace.append(
        {
            "node": "critic_maneuver",
            "maneuver": maneuver,
            "verdict": verdict,
            "approved": plan.kind,
            "proposed_maneuver": maneuver,
            "pass_preconditions": gate_eval.get("pass_preconditions"),
            "pass_blockers": gate_eval.get("pass_blockers"),
        }
    )
    pass_in_progress = state.get("pass_in_progress", False)
    if plan.kind == "pass":
        pass_in_progress = True
    if maneuver == "abort_pass" or plan.kind == "replan":
        pass_in_progress = False
    if plan.kind == "replan":
        phase = "replan"
    else:
        phase = "execute"
    return {
        "dsl": dsl_to_dict(dsl),
        "trace": trace,
        "last_verdict": verdict,
        "approved_maneuver": plan.kind,
        "pass_in_progress": pass_in_progress,
        "phase": phase,
    }


def _pass_maneuver_complete(world_after: WorldState, feedback: Dict[str, Any]) -> bool:
    if feedback.get("ego_lane", world_after.ego_lane) != 0:
        return False
    if not feedback.get("clear_of_lead"):
        return False
    if feedback.get("pass_phase") not in ("merge", "cruise"):
        return False
    return True


def node_execute(state: AgenticState) -> Dict[str, Any]:
    from autopass.config import get_perception_backend
    from autopass.executor import execute_step
    from autopass.policy import clamp_maneuver_for_policy

    spec = _spec(state)
    world = _world(state)
    dsl = _dsl(state)
    policy = state.get("policy", "autopass")
    approved = state.get("approved_maneuver", "wait")
    action = clamp_maneuver_for_policy(policy, approved)
    pass_active = bool(state.get("pass_in_progress", False))
    if approved == "pass" and not pass_active:
        pass_active = True
    # Do not override planner/critic wait with forced pass — executor handles safe abort/hold.
    if pass_active and approved == "pass":
        action = "pass"
    elif pass_active and approved == "abort_pass":
        action = "wait"
    elif pass_active and approved == "wait":
        action = "wait"
    maneuver_started = False

    world_before = world
    pre_belief_front_m = dsl.world_belief.front_gap_m
    backend = state.get("perception_backend", get_perception_backend())
    import os

    execute_dt = float(os.environ.get("AUTOPASS_EXECUTE_DT_S", "0.5"))
    world_after, dsl, feedback = execute_step(spec, world, dsl, action, backend=backend, dt=execute_dt)
    from autopass.belief import observed_front_gap_m

    obs_payload = feedback.get("observation") or {}
    post_observed_front_m = observed_front_gap_m(obs_payload, feedback)
    post_belief_front_m = dsl.world_belief.front_gap_m

    progress = float(feedback.get("progress_delta_m", 0.0))
    world_after = sync_world_from_belief(
        spec,
        world_after,
        dsl,
        progress_delta_m=progress,
        measured_speed_mps=feedback.get("measured_speed_mps"),
        ego_lane=feedback.get("ego_lane"),
        passed=world_after.passed,
        collision=world_after.collision,
        done=world_after.done,
    )

    dsl, verdict = critique_post_execution(
        dsl, spec, world_before, world_after, dsl.maneuver, execution_feedback=feedback
    )
    maneuver_started = bool(feedback.get("pass_maneuver_started", False))
    maneuver_completed = action == "pass" and _pass_maneuver_complete(world_after, feedback)
    if feedback.get("pass_fsm_active") is False and pass_active and feedback.get("pass_abort_reason"):
        pass_active = False
    if feedback.get("control_failure") and feedback.get("pass_abort_reason"):
        pass_active = False
    if action == "pass" and maneuver_completed:
        pass_active = False
        world_after = replace(world_after, passed=True)

    proposed = state.get("proposed_maneuver", "wait")
    if action == "pass":
        if proposed != "pass":
            execution_outcome = "pass_attempt_blocked_by_critic"
        elif feedback.get("collision"):
            execution_outcome = "pass_attempt_collision"
        elif feedback.get("control_failure"):
            execution_outcome = "pass_attempt_failed_control"
        elif verdict == "replan":
            execution_outcome = "pass_attempt_replan"
        elif maneuver_completed:
            execution_outcome = "pass_attempt_success"
        else:
            execution_outcome = "pass_attempt_in_progress"
    else:
        execution_outcome = "wait_success" if action == "wait" else f"{action}_executed"

    if verdict == "replan":
        fsm_active = bool(feedback.get("pass_fsm_active"))
        recoverable_pass = (
            fsm_active
            and action == "pass"
            and not feedback.get("control_failure")
            and not feedback.get("collision")
        )
        if not recoverable_pass:
            dsl = replace(
                dsl,
                tools_pending=[],
                tools_completed=[],
                maneuver=ManeuverPlan(),
            )
            pass_active = False

    trace = list(state.get("trace", []))
    obs_payload = feedback.get("observation") or {}
    lead_res = obs_payload.get("lead_resolution") or {}
    trace.append(
        {
            "node": "execute",
            "action": feedback.get("action", action),
            "action_semantic": feedback.get("action_semantic", ("follow_lead" if action == "wait" else action)),
            "requested_action": approved,
            "policy": policy,
            "ego_x": round(world_after.ego_x_m, 2),
            "collision": world_after.collision,
            "passed": world_after.passed,
            "post_verdict": verdict,
            "execution_mode": feedback.get("mode"),
            "pass_in_progress": pass_active,
            "pass_maneuver_started": maneuver_started,
            "pass_maneuver_completed": maneuver_completed,
            "pre_execute_belief_front_m": pre_belief_front_m,
            "post_execute_observed_front_m": post_observed_front_m,
            "post_execute_belief_front_m": post_belief_front_m,
            "world_belief_front_m": post_belief_front_m,
            "world_belief_front_valid": dsl.world_belief.front_valid,
            "world_belief_lead_speed_mps": dsl.world_belief.lead_speed_mps,
            "world_belief_lead_speed_valid": dsl.world_belief.front_valid and dsl.world_belief.lead_speed_mps is not None,
            "world_belief_rear_m": dsl.world_belief.rear_gap_m,
            "world_belief_rear_valid": dsl.world_belief.rear_valid,
            "world_belief_oncoming_m": dsl.world_belief.oncoming_gap_m,
            "world_belief_oncoming_valid": dsl.world_belief.oncoming_valid,
            "world_belief_oncoming_available": dsl.world_belief.oncoming_available,
            "world_belief_oncoming_unavailable_reason": dsl.world_belief.oncoming_unavailable_reason,
            "ego_lane_id": feedback.get("ego_lane_id", feedback.get("ego_lane")),
            "target_lane_id": feedback.get("target_lane_id"),
            "lane_center_error_m": feedback.get("lane_center_error_m"),
            "heading_error_deg": feedback.get("heading_error_deg"),
            "steer": feedback.get("steer"),
            "throttle": feedback.get("throttle"),
            "brake": feedback.get("brake"),
            "ego_speed_mps": feedback.get("ego_speed_mps", feedback.get("measured_speed_mps")),
            "insufficient_counts_by_tool": state.get("insufficient_counts_by_tool", {}),
            "last_insufficient_tool": state.get("last_insufficient_tool", ""),
            "perception_retry_count": state.get("perception_retry_count", 0),
            "fallback_reason": state.get("fallback_reason", ""),
            "lead_resolution": lead_res,
            "lead_resolution_reason": lead_res.get("front_resolution_reason"),
            "calibrated_gap_source": lead_res.get("calibrated_gap_source"),
            "used_detection_for_front": lead_res.get("used_detection_for_front"),
            "rear_gap_source": lead_res.get("rear_gap_source"),
            "passing_topology": lead_res.get("passing_topology"),
            "oncoming_required": lead_res.get("oncoming_required"),
            "oncoming_check_reason": lead_res.get("oncoming_check_reason"),
            "execution_outcome": execution_outcome,
            "control_failure": feedback.get("control_failure", False),
            "pass_fsm_phase": feedback.get("pass_fsm_phase"),
            "pass_abort_reason": feedback.get("pass_abort_reason", ""),
            "target_lane_source": feedback.get("target_lane_source"),
            "passing_lane_id": feedback.get("passing_lane_id"),
            "lateral_offset_travel_m": feedback.get("lateral_offset_travel_m"),
            "lateral_offset_passing_m": feedback.get("lateral_offset_passing_m"),
            "lateral_shift_toward_passing_m": feedback.get("lateral_shift_toward_passing_m"),
            "requested_action": feedback.get("requested_action", approved),
            "continue_pass_despite_wait": False,
            "planner_approved_maneuver": approved,
            "decision_oracle_enabled": __import__(
                "autopass.config", fromlist=["decision_oracle_enabled"]
            ).decision_oracle_enabled(),
            "actor_continuity": feedback.get("actor_continuity"),
        }
    )
    pass_id = int(state.get("pass_maneuver_id", 0))
    if maneuver_started:
        pass_id += 1
    phase = "replan" if verdict == "replan" else "plan"
    return {
        "world": asdict(world_after),
        "dsl": dsl_to_dict(dsl),
        "trace": trace,
        "last_verdict": verdict,
        "phase": phase,
        "pass_in_progress": pass_active,
        "pass_maneuver_id": pass_id,
        "approved_maneuver": approved,
    }


def node_evaluate(state: AgenticState) -> Dict[str, Any]:
    from autopass.pass_trace import count_pass_maneuver_starts

    spec = _spec(state)
    world = _world(state)
    trace = state.get("trace", [])
    passes = count_pass_maneuver_starts(trace)
    rejects = sum(1 for t in trace if t.get("verdict") == "reject" or t.get("post_verdict") == "replan")
    route_ok = world.ego_x_m >= spec.route.goal_x_m and not world.collision
    exec_outcomes = [t.get("execution_outcome") for t in trace if t.get("node") == "execute"]
    if world.collision:
        failure = "collision"
    elif any(o == "pass_attempt_collision" for o in exec_outcomes):
        failure = "pass_attempt_collision"
    elif any(o == "pass_attempt_failed_control" for o in exec_outcomes):
        failure = "pass_attempt_failed_control"
    elif any(o == "pass_attempt_success" for o in exec_outcomes):
        failure = "pass_attempt_success" if not route_ok else "none"
    elif passes > 0 or any(o == "pass_attempt_in_progress" for o in exec_outcomes):
        failure = "pass_attempt_incomplete"
    elif any(o == "pass_attempt_blocked_by_critic" for o in exec_outcomes):
        failure = "pass_attempt_blocked_by_critic"
    elif world.t_s > spec.request.deadline_s:
        failure = "deadline_miss"
    elif not route_ok and world.done:
        failure = "timeout"
    elif passes == 0 and not world.collision:
        failure = "wait_success"
    else:
        failure = "none"

    metrics = {
        "scenario_id": spec.scenario_id,
        "policy": state.get("policy", "autopass"),
        "collision": world.collision,
        "route_completed": route_ok,
        "time_to_goal_s": round(world.t_s, 2),
        "deadline_s": spec.request.deadline_s,
        "deadline_pressure": round(deadline_pressure(spec, world), 3),
        "dsl_revision": _dsl(state).revision,
        "planner_rounds": state.get("planner_rounds", 0),
        "approved_passes": passes,
        "critic_rejects": rejects,
        "failure_type": failure,
    }
    from autopass.pass_trace import classify_failure_taxonomy, summarize_agency

    metrics["failure_taxonomy"] = classify_failure_taxonomy(trace, metrics)
    metrics["agency"] = summarize_agency(trace)
    out: Dict[str, Any] = {"metrics": metrics, "phase": "done"}
    if failure in ("collision", "deadline_miss") and state.get("learning_round", 0) < 3:
        mutated = mutate_from_failure(spec, metrics, state.get("learning_round", 0) + 1)
        out["mutated_spec"] = spec_to_dict(mutated)
        out["learning_round"] = state.get("learning_round", 0) + 1
    return out


def route_after_planner(state: AgenticState) -> str:
    phase = state.get("phase", "plan")
    if phase == "tool":
        return "run_tool"
    if phase == "decide":
        return "critique_maneuver"
    if phase == "done":
        return "evaluate"
    return "planner"


def route_after_execute(state: AgenticState) -> str:
    world = _world(state)
    if world.done or len(state.get("trace", [])) >= state.get("max_drive_steps", 90):
        return "evaluate"
    return "planner"


def route_after_critique_maneuver(state: AgenticState) -> str:
    if state.get("phase") == "replan":
        return "planner"
    return "execute"


def build_agentic_graph():
    g = StateGraph(AgenticState)
    g.add_node("init_mission", node_init_mission)
    g.add_node("planner", node_planner)
    g.add_node("run_tool", node_run_tool)
    g.add_node("critique_tool", node_critique_tool)
    g.add_node("critique_maneuver", node_critique_maneuver)
    g.add_node("execute", node_execute)
    g.add_node("evaluate", node_evaluate)

    g.add_edge(START, "init_mission")
    g.add_edge("init_mission", "planner")
    g.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "run_tool": "run_tool",
            "critique_maneuver": "critique_maneuver",
            "evaluate": "evaluate",
            "planner": "planner",
        },
    )
    g.add_edge("run_tool", "critique_tool")
    g.add_edge("critique_tool", "planner")
    g.add_conditional_edges("critique_maneuver", route_after_critique_maneuver, {"planner": "planner", "execute": "execute"})
    g.add_conditional_edges("execute", route_after_execute, {"planner": "planner", "evaluate": "evaluate"})
    g.add_edge("evaluate", END)
    return g.compile()


def run_agentic_episode(
    spec: ScenarioSpec,
    *,
    policy: str = "autopass",
    perception_backend: str | None = None,
    max_drive_steps: int = 90,
    control_mode: str | None = None,
    skip_runtime_check: bool = False,
) -> AgenticState:
    from autopass.config import apply_production_defaults, get_perception_backend, is_test_mode, require_runtime

    apply_production_defaults()
    if not skip_runtime_check and not is_test_mode():
        require_runtime()
    backend = perception_backend or get_perception_backend()
    if control_mode:
        import os

        os.environ["AUTOPASS_CONTROL_MODE"] = control_mode
    app = build_agentic_graph()
    init: AgenticState = {
        "spec": spec_to_dict(spec),
        "world": asdict(initialize_world(spec)),
        "policy": policy,
        "trace": [],
        "metrics": {},
        "max_drive_steps": max_drive_steps,
        "max_planner_rounds": 12,
        "perception_backend": backend,
        "learning_round": 0,
    }
    return app.invoke(init, config={"recursion_limit": 300})
