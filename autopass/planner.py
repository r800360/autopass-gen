"""
Planner agent — chooses the next vision tool or final maneuver from measured state.

No default tool queue: each tool is justified from ``perception_summary`` +
``world_belief`` via ``needed_tools``; the critic may reject redundant runs.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from autopass.dsl import PassingDSL
from autopass.perception_state import (
    belief_is_measured,
    deadline_pressure,
    lead_speed_if_available,
    needed_tools,
    pass_evidence_complete,
    slow_lead,
    urgency_level,
)
from autopass.safety import HARD_FLOOR_FRONT_M
from autopass.tools import TOOL_NAMES, perception_summary
from visual_world import ScenarioSpec, WorldState

PlannerAction = Literal["run_tool", "decide_maneuver", "finish"]

# Vision planner gate — must align with safety margins (see estimate_pass_time front+13 m).
MIN_PASS_FRONT_GAP_M = HARD_FLOOR_FRONT_M + 12.0


class PlannerDecision(BaseModel):
    action: PlannerAction
    tool: Optional[str] = Field(default=None, description="Vision tool when action=run_tool")
    maneuver: Optional[str] = Field(default=None, description="pass|wait|replan|abort_pass when action=decide_maneuver")
    reasoning: str = ""


def _pass_preconditions(dsl: PassingDSL, world: WorldState) -> tuple[bool, str]:
    """
  Return (may_propose_pass, wait_reasoning).
  All checks are vision-belief only — critic remains the backstop.
  """
    wb = dsl.world_belief
    gap = float(wb.front_gap_m) if wb.front_gap_m is not None else None
    _, lead_ok = lead_speed_if_available(dsl)

    if gap is not None and gap < MIN_PASS_FRONT_GAP_M and not lead_ok:
        return (
            False,
            (
                f"Front gap is only {gap:.1f}m and lead speed is unavailable; "
                "passing is not justified, so follow/wait while continuing to monitor."
            ),
        )

    blockers: list[str] = []
    if not belief_is_measured(wb) or not wb.front_valid or gap is None:
        blockers.append("front gap is not validated")
    elif gap < MIN_PASS_FRONT_GAP_M:
        blockers.append(
            f"front gap is only {gap:.1f}m (below {MIN_PASS_FRONT_GAP_M:.0f}m passing threshold)"
        )
    if not lead_ok:
        blockers.append("lead speed is unavailable")
    elif not slow_lead(dsl, world):
        blockers.append("slow lead is not confirmed from vision")
    if not pass_evidence_complete(dsl):
        blockers.append("pass perception evidence is incomplete")
    if not wb.rear_valid:
        blockers.append("rear gap is not validated for lane-change safety")
    if wb.oncoming_available and not wb.oncoming_valid:
        blockers.append("oncoming gap is required but not validated for this corridor")
    if wb.oncoming_available is False and wb.oncoming_unavailable_reason:
        pass  # oncoming not applicable — not a pass blocker

    if blockers:
        return False, "; ".join(blockers).capitalize() + " — wait and monitor."
    return True, ""


def _clamp_pass_decision(
    decision: PlannerDecision,
    dsl: PassingDSL,
    world: WorldState,
) -> PlannerDecision:
    if decision.action != "decide_maneuver" or decision.maneuver != "pass":
        return decision
    ok, reason = _pass_preconditions(dsl, world)
    if ok:
        return decision
    return PlannerDecision(action="decide_maneuver", maneuver="wait", reasoning=reason)


def _rule_plan(
    dsl: PassingDSL,
    spec: ScenarioSpec,
    world: WorldState,
    *,
    pass_in_progress: bool = False,
    block_front_measure: bool = False,
) -> PlannerDecision:
    if world.done or world.collision:
        return PlannerDecision(action="finish", reasoning="Episode terminal state reached.")

    if dsl.mission.aggression == "0":
        missing = needed_tools(dsl, spec, world, block_front_measure=block_front_measure)
        if missing:
            return PlannerDecision(action="run_tool", tool=missing[0][0], reasoning=missing[0][1])
        return PlannerDecision(action="decide_maneuver", maneuver="wait", reasoning="No-pass policy.")

    missing = needed_tools(dsl, spec, world, block_front_measure=block_front_measure)
    recent_critic = [n for n in dsl.verification_log[-6:] if n.tool]
    recent_capture_insufficient = sum(
        1 for n in recent_critic if n.tool == "capture_sensors" and n.verdict == "insufficient"
    )
    if missing:
        tool, why = missing[0]
        if tool == "capture_sensors" and recent_capture_insufficient >= 2:
            return PlannerDecision(
                action="decide_maneuver",
                maneuver="wait",
                reasoning="insufficient validated perception; fallback follow_lead until belief recovers",
            )
        return PlannerDecision(action="run_tool", tool=tool, reasoning=why)

    summary = perception_summary(dsl)
    return _decide_maneuver_from_evidence(
        dsl, spec, world, summary, pass_in_progress=pass_in_progress
    )


def plan_next(
    dsl: PassingDSL,
    spec: ScenarioSpec,
    world: WorldState,
    *,
    max_tool_rounds: int = 12,
    pass_in_progress: bool = False,
    block_front_measure: bool = False,
) -> PlannerDecision:
    from agents import llm_agents

    if len(dsl.tools_completed) > max_tool_rounds * 2:
        return PlannerDecision(action="decide_maneuver", maneuver="wait", reasoning="Tool budget exhausted.")

    if llm_agents.use_mock_llm():
        decision = _rule_plan(
            dsl, spec, world, pass_in_progress=pass_in_progress, block_front_measure=block_front_measure
        )
    else:
        decision = _llm_plan(
            dsl, spec, world, pass_in_progress=pass_in_progress, block_front_measure=block_front_measure
        )

    if dsl.mission.aggression == "0" and decision.maneuver == "pass":
        return PlannerDecision(action="decide_maneuver", maneuver="wait", reasoning="No-pass policy.")
    return _clamp_pass_decision(decision, dsl, world)


def _decide_maneuver_from_evidence(
    dsl: PassingDSL,
    spec: ScenarioSpec,
    world: WorldState,
    summary: Dict[str, Any],
    *,
    pass_in_progress: bool = False,
) -> PlannerDecision:
    if world.passed:
        return PlannerDecision(
            action="decide_maneuver",
            maneuver="wait",
            reasoning="Pass completed — cruise in travel lane.",
        )

    if pass_in_progress:
        if not pass_evidence_complete(dsl):
            missing = needed_tools(dsl, spec, world)
            if missing:
                return PlannerDecision(action="run_tool", tool=missing[0][0], reasoning=f"Pass active: {missing[0][1]}")
        rear = summary.get("measure_rear_gap", {})
        oncoming = summary.get("measure_oncoming", {})
        if rear and not rear.get("safe", True):
            return PlannerDecision(action="decide_maneuver", maneuver="abort_pass", reasoning="Rear gap closed during pass.")
        if oncoming and not oncoming.get("safe", True):
            return PlannerDecision(action="decide_maneuver", maneuver="abort_pass", reasoning="Oncoming risk during pass.")
        return PlannerDecision(action="decide_maneuver", maneuver="pass", reasoning="Pass in progress — continue actuation.")

    can_pass, wait_reason = _pass_preconditions(dsl, world)
    if not can_pass:
        missing = needed_tools(dsl, spec, world)
        if missing:
            return PlannerDecision(action="run_tool", tool=missing[0][0], reasoning=missing[0][1])
        return PlannerDecision(action="decide_maneuver", maneuver="wait", reasoning=wait_reason)

    rear = summary.get("measure_rear_gap", {})
    oncoming = summary.get("measure_oncoming", {})
    kin = summary.get("check_kinematics", {})
    traffic = summary.get("assess_traffic", {})

    if rear and not rear.get("safe", True):
        return PlannerDecision(action="decide_maneuver", maneuver="wait", reasoning="Rear gap unsafe (measured).")
    if oncoming and not oncoming.get("safe", True):
        return PlannerDecision(action="decide_maneuver", maneuver="wait", reasoning="Oncoming gap unsafe (measured).")
    if kin and not kin.get("feasible", True):
        if traffic.get("is_real_traffic"):
            return PlannerDecision(action="decide_maneuver", maneuver="replan", reasoning="Dense traffic — replan route.")
        return PlannerDecision(action="decide_maneuver", maneuver="wait", reasoning="Pass not kinematically feasible.")

    u = urgency_level(spec, world)
    if u in ("high", "medium") or dsl.mission.urgency in ("high", "medium"):
        return PlannerDecision(action="decide_maneuver", maneuver="pass", reasoning="Vision checks passed under urgency.")
    return PlannerDecision(action="decide_maneuver", maneuver="wait", reasoning="Safe to wait — low urgency.")


def _llm_plan(
    dsl: PassingDSL,
    spec: ScenarioSpec,
    world: WorldState,
    *,
    pass_in_progress: bool = False,
    block_front_measure: bool = False,
) -> PlannerDecision:
    from agents.llm_agents import structured_invoke

    missing = needed_tools(dsl, spec, world, block_front_measure=block_front_measure)
    summary = perception_summary(dsl)
    wb = summary.get("world_belief", {})
    fallback = _rule_plan(
        dsl, spec, world, pass_in_progress=pass_in_progress, block_front_measure=block_front_measure
    )
    allowed = [t for t, _ in missing] if missing else []
    oncoming_note = ""
    if wb.get("oncoming_available") is False:
        reason = wb.get("oncoming_unavailable_reason") or "not applicable"
        oncoming_note = (
            f"Oncoming is not applicable/unavailable for this corridor ({reason}); "
            "do not claim certainty about absence of opposing traffic.\n"
        )
    prompt = (
        f"Mission: {dsl.mission.text}\n"
        f"Revision: {dsl.revision}\n"
        f"Pass in progress: {pass_in_progress}\n"
        f"Measured belief: {wb}\n"
        f"Tools completed: {dsl.tools_completed}\n"
        f"Needed tools (pick one if any): {allowed}\n"
        f"Summary: {summary}\n"
        f"{oncoming_note}"
        f"PASS GATES (mandatory): propose pass only if front_gap_m >= {MIN_PASS_FRONT_GAP_M:.0f} m, "
        f"lead speed is measured, rear_valid is true, pass evidence tools are complete, and "
        f"oncoming is validated when oncoming_available is true. Otherwise choose wait.\n"
        f"Never describe a gap below {MIN_PASS_FRONT_GAP_M:.0f} m as sufficient for a safe pass.\n"
        f"If no needed tools, choose decide_maneuver (pass|wait|replan|abort_pass).\n"
        f"Valid tools: {TOOL_NAMES}"
    )
    decision = structured_invoke(
        PlannerDecision,
        "Overtaking planner. Pick ONE needed vision tool OR a maneuver. "
        "Never choose a tool not in Needed tools unless replanning after new evidence. "
        "Under deadline pressure, still refuse pass when vision evidence is insufficient.",
        prompt,
        fallback,
    )
    return _clamp_pass_decision(decision, dsl, world)
