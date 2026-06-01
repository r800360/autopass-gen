"""
Planner agent — chooses the next vision tool or final maneuver from measured state.

No default tool queue: each tool is justified from ``perception_summary`` +
``world_belief`` via ``needed_tools``; the critic may reject redundant runs.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from autopass.dsl import PassingDSL
from autopass.pass_gates import evaluate_pass_gates
from autopass.perception_state import (
    needed_tools,
    pass_evidence_complete,
    urgency_level,
)
from autopass.pass_gates import MIN_PASS_FRONT_GAP_M
from autopass.tools import TOOL_NAMES, perception_summary
from visual_world import ScenarioSpec, WorldState

PlannerAction = Literal["run_tool", "decide_maneuver", "finish"]


class PlannerDecision(BaseModel):
    action: PlannerAction
    tool: Optional[str] = Field(default=None, description="Vision tool when action=run_tool")
    maneuver: Optional[str] = Field(default=None, description="pass|wait|replan|abort_pass when action=decide_maneuver")
    reasoning: str = ""


def _pass_preconditions(
    dsl: PassingDSL,
    spec: ScenarioSpec,
    world: WorldState,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Return (may_propose_pass, wait_reasoning, gate_evaluation)."""
    summary = perception_summary(dsl)
    gates = evaluate_pass_gates(dsl, spec, world, summary=summary)
    if gates["can_pass"]:
        return True, "", gates
    blockers = gates["pass_blockers"]
    if blockers:
        return False, "; ".join(blockers).capitalize() + " — wait and monitor.", gates
    return False, "Pass gates not satisfied — wait and monitor.", gates


def _clamp_pass_decision(
    decision: PlannerDecision,
    dsl: PassingDSL,
    spec: ScenarioSpec,
    world: WorldState,
) -> PlannerDecision:
    if decision.action != "decide_maneuver" or decision.maneuver != "pass":
        return decision
    ok, reason, _ = _pass_preconditions(dsl, spec, world)
    if ok:
        return decision
    return PlannerDecision(action="decide_maneuver", maneuver="wait", reasoning=reason)


def _clamp_maneuver_during_pass(
    decision: PlannerDecision,
    *,
    pass_in_progress: bool,
) -> PlannerDecision:
    """Mid-pass: front-gap gate applies before starting pass, not while finishing beside the lead."""
    if not pass_in_progress or decision.action != "decide_maneuver":
        return decision
    if decision.maneuver == "pass":
        return decision
    if decision.maneuver == "abort_pass":
        return PlannerDecision(
            action="decide_maneuver",
            maneuver="pass",
            reasoning=(
                "Pass in progress — closing front gap is expected beside the lead; "
                "finish the lane change and merge-back (do not abort for sub-18m vision alone)."
            ),
        )
    return PlannerDecision(
        action="decide_maneuver",
        maneuver="pass",
        reasoning=(
            "Pass in progress — continue actuation until merge-back completes "
            "(closing front gap is expected; do not wait mid-maneuver)."
        ),
    )


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
        missing = needed_tools(
            dsl, spec, world, block_front_measure=block_front_measure, pass_in_progress=pass_in_progress
        )
        if missing:
            return PlannerDecision(action="run_tool", tool=missing[0][0], reasoning=missing[0][1])
        return PlannerDecision(action="decide_maneuver", maneuver="wait", reasoning="No-pass policy.")

    missing = needed_tools(
        dsl, spec, world, block_front_measure=block_front_measure, pass_in_progress=pass_in_progress
    )
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
    decision = _clamp_maneuver_during_pass(decision, pass_in_progress=pass_in_progress)
    return _clamp_pass_decision(decision, dsl, spec, world)


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
            missing = needed_tools(dsl, spec, world, pass_in_progress=True)
            if missing:
                return PlannerDecision(action="run_tool", tool=missing[0][0], reasoning=f"Pass active: {missing[0][1]}")
        rear = summary.get("measure_rear_gap", {})
        oncoming = summary.get("measure_oncoming", {})
        if rear and not rear.get("safe", True):
            return PlannerDecision(action="decide_maneuver", maneuver="abort_pass", reasoning="Rear gap closed during pass.")
        gates = evaluate_pass_gates(dsl, spec, world, summary=summary)
        if gates["oncoming_required"] and oncoming and not oncoming.get("safe", True):
            return PlannerDecision(action="decide_maneuver", maneuver="abort_pass", reasoning="Oncoming risk during pass.")
        return PlannerDecision(action="decide_maneuver", maneuver="pass", reasoning="Pass in progress — continue actuation.")

    gates = evaluate_pass_gates(dsl, spec, world, summary=summary)
    if not gates["can_pass"]:
        missing = needed_tools(dsl, spec, world, pass_in_progress=pass_in_progress)
        if missing:
            return PlannerDecision(action="run_tool", tool=missing[0][0], reasoning=missing[0][1])
        blockers = gates["pass_blockers"]
        return PlannerDecision(
            action="decide_maneuver",
            maneuver="wait",
            reasoning="; ".join(blockers).capitalize() + " — wait and monitor.",
        )

    traffic = summary.get("assess_traffic", {})
    kin = summary.get("check_kinematics", {})
    if kin and not kin.get("feasible", True):
        if traffic.get("is_real_traffic"):
            return PlannerDecision(action="decide_maneuver", maneuver="replan", reasoning="Dense traffic — replan route.")
        return PlannerDecision(action="decide_maneuver", maneuver="wait", reasoning="Pass not kinematically feasible.")

    motivations = "; ".join(gates["pass_motivation"][:3])
    u = urgency_level(spec, world)
    if u in ("high", "medium") or dsl.mission.urgency in ("high", "medium"):
        return PlannerDecision(
            action="decide_maneuver",
            maneuver="pass",
            reasoning=f"All pass gates satisfied ({motivations}).",
        )
    return PlannerDecision(
        action="decide_maneuver",
        maneuver="wait",
        reasoning="All gates pass but urgency is low — safe to wait.",
    )


def _llm_decide_maneuver(
    dsl: PassingDSL,
    spec: ScenarioSpec,
    world: WorldState,
    summary: Dict[str, Any],
    *,
    pass_in_progress: bool = False,
) -> PlannerDecision:
    """Production: LLM chooses pass|wait|replan|abort_pass; gates clamp unsafe pass."""
    from agents.llm_agents import structured_invoke

    fallback = _decide_maneuver_from_evidence(
        dsl, spec, world, summary, pass_in_progress=pass_in_progress
    )
    gates = evaluate_pass_gates(dsl, spec, world, summary=summary)
    wb = summary.get("world_belief", {})
    allowed = ("pass", "wait", "replan", "abort_pass")
    prompt = (
        f"Mission: {dsl.mission.text}\n"
        f"Urgency: {dsl.mission.urgency}\n"
        f"Deadline pressure: {urgency_level(spec, world)}\n"
        f"Pass in progress: {pass_in_progress}\n"
        f"World belief (vision): {wb}\n"
        f"Pass gates (hard constraints): {gates}\n"
        f"Perception summary: {summary}\n"
        f"Choose action=decide_maneuver and maneuver one of {allowed}. "
        f"Propose pass only when can_pass is true. If pass in progress and rear/oncoming unsafe, abort_pass. "
        f"If pass_in_progress is true, you MUST choose maneuver=pass (never wait/replan) unless aborting. "
        f"Under high urgency prefer pass when gates allow; when can_pass is false and pass not in progress, wait or run tools."
    )
    decision = structured_invoke(
        PlannerDecision,
        "Overtaking planner agent. Decide the next maneuver from vision evidence and deadline pressure. "
        "You are the decision authority; safety gates may override an unsafe pass.",
        prompt,
        fallback,
    )
    if decision.action != "decide_maneuver" or decision.maneuver not in allowed:
        return fallback
    out = PlannerDecision(
        action="decide_maneuver",
        maneuver=decision.maneuver,
        reasoning=decision.reasoning or fallback.reasoning,
    )
    return _clamp_pass_decision(out, dsl, spec, world)


def _llm_plan(
    dsl: PassingDSL,
    spec: ScenarioSpec,
    world: WorldState,
    *,
    pass_in_progress: bool = False,
    block_front_measure: bool = False,
) -> PlannerDecision:
    from agents.llm_agents import structured_invoke

    missing = needed_tools(
        dsl, spec, world, block_front_measure=block_front_measure, pass_in_progress=pass_in_progress
    )
    summary = perception_summary(dsl)
    gates = evaluate_pass_gates(dsl, spec, world, summary=summary)

    if not missing:
        return _llm_decide_maneuver(
            dsl, spec, world, summary, pass_in_progress=pass_in_progress
        )

    fallback = _rule_plan(
        dsl, spec, world, pass_in_progress=pass_in_progress, block_front_measure=block_front_measure
    )
    allowed = [t for t, _ in missing]
    wb = summary.get("world_belief", {})
    oncoming_note = ""
    if not gates["oncoming_required"]:
        reason = gates.get("oncoming_check_reason") or wb.get("oncoming_unavailable_reason") or "same_direction_passing_lane"
        oncoming_note = (
            f"Oncoming is NOT required for this corridor ({reason}). "
            "Do not treat unavailable oncoming as a pass blocker.\n"
        )
    elif wb.get("oncoming_available") is False:
        oncoming_note = (
            f"Oncoming unavailable ({wb.get('oncoming_unavailable_reason', '')}); "
            "only block pass if oncoming_required is true.\n"
        )

    slow_lead_note = (
        "A stationary or very slow lead (lead_speed_mps near 0) is a REASON TO PASS when "
        "front_gap_ok, rear_gap_ok, kinematics_ok, and topology_ok are all true. "
        "Stationary lead alone is NOT unsafe.\n"
    )
    rear_note = (
        "Use measure_rear_gap.safe for rear safety — ignore raw burst rear_closing_mps if "
        "rear_closing_valid is false or magnitude is extreme.\n"
    )
    prompt = (
        f"Mission: {dsl.mission.text}\n"
        f"Revision: {dsl.revision}\n"
        f"Pass in progress: {pass_in_progress}\n"
        f"Measured belief: {wb}\n"
        f"Pass gates: {gates}\n"
        f"Tools completed: {dsl.tools_completed}\n"
        f"Needed tools (pick one if any): {allowed}\n"
        f"Summary: {summary}\n"
        f"{slow_lead_note}"
        f"{rear_note}"
        f"{oncoming_note}"
        f"PASS GATES (mandatory): propose pass only when pass_preconditions are all true in Pass gates. "
        f"front_gap_m must be >= {MIN_PASS_FRONT_GAP_M:.0f} m. "
        f"Never describe a gap below {MIN_PASS_FRONT_GAP_M:.0f} m as sufficient for a safe pass.\n"
        f"If no needed tools, you must NOT choose decide_maneuver — only run_tool from Needed tools.\n"
        f"Valid tools: {TOOL_NAMES}"
    )
    decision = structured_invoke(
        PlannerDecision,
        "Overtaking planner. Pick ONE needed vision tool from Needed tools. "
        "Do not choose decide_maneuver while tools remain in Needed tools. "
        "Under deadline pressure, still refuse pass when vision evidence is insufficient.",
        prompt,
        fallback,
    )
    if decision.action == "decide_maneuver" and allowed:
        return fallback
    if decision.action == "run_tool" and allowed:
        first_tool = allowed[0]
        if decision.tool not in allowed:
            return PlannerDecision(
                action="run_tool",
                tool=first_tool,
                reasoning=f"Planner picked unavailable tool; using required {first_tool}.",
            )
        if decision.tool != first_tool:
            return PlannerDecision(
                action="run_tool",
                tool=first_tool,
                reasoning=f"Required tool order: {first_tool} before {decision.tool}.",
            )
    return _clamp_pass_decision(decision, dsl, spec, world)
