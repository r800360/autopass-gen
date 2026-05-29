"""Pass maneuver counting and trace helpers for benchmark metrics."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def count_pass_maneuver_starts(trace: List[Dict[str, Any]]) -> int:
    """
    Count pass attempts as maneuver starts (rising edge), not every execute tick.

    A new maneuver starts when execute action==pass and pass_maneuver_started is True,
    or when action==pass after a non-pass execute (fallback for older traces).
    """
    count = 0
    in_maneuver = False
    for entry in trace:
        if entry.get("node") not in ("execute", "baseline"):
            continue
        action = entry.get("action")
        if entry.get("pass_maneuver_completed"):
            in_maneuver = False
            continue
        if action == "pass":
            started = entry.get("pass_maneuver_started")
            if started is True or (started is None and not in_maneuver):
                count += 1
                in_maneuver = True
        elif action in ("wait", "replan", "hold"):
            if entry.get("passed"):
                in_maneuver = False
    return count


def extract_pass_execute_events(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """All execute rows involving pass (for debug), not used for attempt count."""
    return [t for t in trace if t.get("node") == "execute" and t.get("action") == "pass"]


def last_pass_debug(trace: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    events = extract_pass_execute_events(trace)
    return events[-1] if events else None


def summarize_agency(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Metrics proving LLM agency and vision-only belief in production traces."""
    planner = [t for t in trace if t.get("node") == "planner"]
    llm_rounds = sum(1 for t in planner if t.get("agency_source") == "llm")
    oracle_off = sum(1 for t in planner if t.get("decision_oracle_enabled") is False)
    vision_front = 0
    axis_front = 0
    for t in trace:
        if t.get("node") != "execute":
            continue
        lr = t.get("lead_resolution") or {}
        src = lr.get("calibrated_gap_source") or t.get("calibrated_gap_source")
        if lr.get("used_detection_for_front"):
            vision_front += 1
        elif src == "actor_axis":
            axis_front += 1
    execute = [t for t in trace if t.get("node") == "execute"]
    critic_overrides = sum(1 for t in execute if t.get("continue_pass_despite_wait"))
    return {
        "planner_rounds_llm": llm_rounds,
        "planner_rounds_total": len(planner),
        "decision_oracle_disabled_steps": oracle_off,
        "execute_vision_front_steps": vision_front,
        "execute_axis_front_steps": axis_front,
        "critic_execute_mismatches": critic_overrides,
    }


def classify_failure_taxonomy(trace: List[Dict[str, Any]], metrics: Dict[str, Any]) -> str:
    """Episode-level failure bucket for debugging and north-star reporting."""
    ft = metrics.get("failure_type", "none")
    if ft in ("collision", "pass_attempt_collision"):
        return "collision"
    if ft == "pass_attempt_failed_control":
        last = last_pass_debug(trace) or {}
        reason = str(last.get("pass_abort_reason") or last.get("failure_type") or "")
        if "multi_lane" in reason or "lane_departure" in reason:
            return "control_lane_departure"
        if reason:
            return "control_abort"
        return "control_incomplete"
    if ft == "pass_attempt_blocked_by_critic":
        for t in reversed(trace):
            if t.get("node") == "critic_maneuver" and t.get("verdict") == "reject":
                msg = str(t.get("message", "")).lower()
                if "slow" in msg:
                    return "critic_reject_slow_lead"
                if "incomplete" in msg or "evidence" in msg:
                    return "perception_insufficient"
                if "unsafe" in msg or "gap" in msg:
                    return "post_step_unsafe"
                return "critic_reject_other"
        return "critic_reject_other"
    for t in reversed(trace):
        if t.get("node") == "critic_tool" and t.get("verdict") == "insufficient":
            return "perception_insufficient"
        if t.get("node") == "planner" and t.get("can_pass") is False:
            blockers = t.get("pass_blockers") or []
            if blockers and "front gap" in str(blockers[0]).lower():
                return "false_pass_gate_front"
    if ft == "pass_attempt_incomplete":
        return "control_incomplete"
    if ft == "deadline_miss":
        return "deadline_miss"
    if ft == "wait_success":
        return "wait_success"
    if ft == "pass_attempt_success":
        return "pass_success"
    return ft or "none"
