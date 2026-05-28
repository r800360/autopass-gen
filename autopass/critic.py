"""
Critic agent — external verification against vision evidence only.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Tuple

from autopass.dsl import ManeuverPlan, PassingDSL, VerificationNote
from autopass.perception_state import (
    InsufficientPerceptionError,
    pass_evidence_complete,
    slow_lead,
    tool_redundant,
)
from autopass.safety import check_pass_safety
from autopass.tools import perception_summary
from visual_world import ScenarioSpec, WorldState

CriticVerdict = Literal["ok", "insufficient", "reject", "replan"]


def critique_tool_result(
    dsl: PassingDSL,
    tool_name: str,
    payload: Dict[str, Any],
    spec: ScenarioSpec,
    world: WorldState,
) -> Tuple[PassingDSL, CriticVerdict]:
    if payload.get("error_type") == "insufficient_perception":
        dsl = dsl.invalidate_tool(tool_name, payload.get("message", "insufficient perception"))
        note = VerificationNote(
            verdict="insufficient",
            message=(
                f"{tool_name} insufficient perception: "
                f"{payload.get('message', 'missing validated belief')}"
            ),
            tool=tool_name,
        )
        return dsl.append_verification(note), "insufficient"

    redundant = tool_redundant(tool_name, dsl, spec, world)
    if redundant:
        dsl = dsl.invalidate_tool(tool_name, redundant)
        note = VerificationNote(verdict="reject", message=f"Redundant tool: {redundant}", tool=tool_name)
        return dsl.append_verification(note), "reject"

    if tool_name == "capture_sensors" and not payload.get("car_distances"):
        dsl = dsl.invalidate_tool(tool_name, "no vehicles in burst")
        note = VerificationNote(verdict="insufficient", message="No vehicles in burst — retry after motion.", tool=tool_name)
        return dsl.append_verification(note), "insufficient"

    if tool_name == "capture_sensors" and payload.get("front_speed_mps") is None:
        if dsl.world_belief.front_valid:
            note = VerificationNote(
                verdict="ok",
                message="Front gap validated from burst; lead speed unavailable (gap-only).",
                tool=tool_name,
            )
            return dsl.append_verification(note), "ok"
        cars = payload.get("car_distances", [])
        has_front = any(c.get("used_for_front_gap") for c in cars) or any(
            c.get("position") == "front" for c in cars
        )
        if has_front:
            note = VerificationNote(
                verdict="insufficient",
                message="Lead speed not estimated from burst; continue with gap-only evidence.",
                tool=tool_name,
            )
            return dsl.append_verification(note), "insufficient"

    if tool_name == "measure_front_gap":
        gap = payload.get("front_gap_m")
        front_ok = payload.get("front_valid", dsl.world_belief.front_valid)
        if gap is None or not front_ok:
            dsl = dsl.invalidate_tool(tool_name, "front gap not validated")
            note = VerificationNote(
                verdict="insufficient",
                message="Front gap not validated in world_belief.",
                tool=tool_name,
            )
            return dsl.append_verification(note), "insufficient"
        if float(gap) >= 200.0:
            dsl = dsl.invalidate_tool(tool_name, "front gap unreliable")
            note = VerificationNote(verdict="insufficient", message="Front gap reading unreliable.", tool=tool_name)
            return dsl.append_verification(note), "insufficient"
        if not payload.get("lead_speed_valid", False):
            note = VerificationNote(
                verdict="ok",
                message="Front gap verified; lead speed unavailable (gap-only).",
                tool=tool_name,
            )
            return dsl.append_verification(note), "ok"

    if tool_name == "check_kinematics" and not payload.get("feasible") and payload.get("required_time_s", 0) > 8:
        note = VerificationNote(
            verdict="replan",
            message="Pass duration excessive — consider alternate route.",
            tool=tool_name,
            revision_triggered=False,
        )
        return dsl.append_verification(note), "replan"

    note = VerificationNote(verdict="ok", message=f"{tool_name} verified.", tool=tool_name)
    return dsl.append_verification(note), "ok"


def critique_maneuver_proposal(
    dsl: PassingDSL,
    maneuver: str,
    spec: ScenarioSpec,
    world: WorldState,
) -> Tuple[PassingDSL, CriticVerdict, ManeuverPlan]:
    if dsl.mission.aggression == "0" and maneuver in ("pass", "abort_pass"):
        maneuver = "wait"

    if maneuver == "abort_pass":
        plan = ManeuverPlan(kind="wait", reasoning="Critic: abort pass — unsafe mid-maneuver.")
        note = VerificationNote(verdict="ok", message="Pass aborted.", revision_triggered=True)
        return dsl.set_maneuver(plan).append_verification(note), "ok", plan

    if maneuver == "wait":
        plan = ManeuverPlan(kind="wait", reasoning="Critic accepted wait.")
        note = VerificationNote(verdict="ok", message="Wait approved.")
        return dsl.set_maneuver(plan).append_verification(note), "ok", plan

    if maneuver == "replan":
        plan = ManeuverPlan(kind="replan", reasoning="Critic requested route replan.")
        note = VerificationNote(verdict="replan", message="Replan triggered.", revision_triggered=True)
        return dsl.set_maneuver(plan).append_verification(note), "replan", plan

    if maneuver == "pass":
        if world.passed:
            plan = ManeuverPlan(kind="wait", reasoning="Already passed.")
            note = VerificationNote(verdict="ok", message="Pass already done.")
            return dsl.set_maneuver(plan).append_verification(note), "ok", plan

        if not pass_evidence_complete(dsl):
            plan = ManeuverPlan(kind="wait", reasoning="Missing vision tools for pass.")
            note = VerificationNote(verdict="reject", message="Pass rejected: incomplete perception evidence.")
            return dsl.set_maneuver(plan).append_verification(note), "reject", plan

        if not slow_lead(dsl, world):
            plan = ManeuverPlan(kind="wait", reasoning="Vision: no slow lead.")
            note = VerificationNote(verdict="reject", message="Pass rejected: lead not slow (measured).")
            return dsl.set_maneuver(plan).append_verification(note), "reject", plan

        try:
            safety = check_pass_safety(dsl, spec, world)
        except InsufficientPerceptionError as e:
            plan = ManeuverPlan(kind="wait", reasoning=str(e))
            note = VerificationNote(verdict="reject", message=f"Pass rejected: {e}")
            return dsl.set_maneuver(plan).append_verification(note), "reject", plan

        summary = perception_summary(dsl)
        kin = summary.get("check_kinematics", {})
        if not safety.approved:
            plan = ManeuverPlan(kind="wait", reasoning="; ".join(safety.reasons))
            note = VerificationNote(verdict="reject", message="Pass rejected: " + "; ".join(safety.reasons))
            return dsl.set_maneuver(plan).append_verification(note), "reject", plan

        plan = ManeuverPlan(
            kind="pass",
            passing_side="left",
            target_speed_mps=kin.get("target_speed_mps", world.ego_speed_mps + 2),
            required_time_s=kin.get("required_time_s", 4.0),
            reasoning="Critic approved pass from vision evidence.",
        )
        note = VerificationNote(verdict="ok", message=f"Pass approved (TTC {safety.min_ttc_s:.1f}s).")
        return dsl.set_maneuver(plan).append_verification(note), "ok", plan

    plan = ManeuverPlan(kind="hold", reasoning="Unknown maneuver.")
    note = VerificationNote(verdict="reject", message=f"Unknown maneuver {maneuver}")
    return dsl.set_maneuver(plan).append_verification(note), "reject", plan


def critique_post_execution(
    dsl: PassingDSL,
    spec: ScenarioSpec,
    world_before: WorldState,
    world_after: WorldState,
    maneuver: ManeuverPlan,
    *,
    execution_feedback: dict | None = None,
) -> Tuple[PassingDSL, CriticVerdict]:
    if execution_feedback and execution_feedback.get("mode") == "carla_vehicle":
        if execution_feedback.get("collision"):
            note = VerificationNote(
                verdict="replan",
                message=f"CARLA proximity: {execution_feedback.get('collision_detail', 'collision')}",
                revision_triggered=True,
            )
            return dsl.append_verification(note), "replan"
        if execution_feedback.get("near_miss"):
            gap = execution_feedback.get("min_front_gap_m", 0)
            note = VerificationNote(
                verdict="replan",
                message=f"Near-miss: min front gap {gap:.1f}m — replan.",
                revision_triggered=True,
            )
            return dsl.append_verification(note), "replan"
        pq = execution_feedback.get("pass_quality")
        if isinstance(pq, dict) and not pq.get("ok", True):
            note = VerificationNote(
                verdict="replan",
                message="Pass quality poor: " + ", ".join(pq.get("issues", [])[:4]),
                revision_triggered=True,
            )
            return dsl.append_verification(note), "replan"

    if world_after.collision:
        note = VerificationNote(
            verdict="replan",
            message="Collision detected post-execution — replan.",
            revision_triggered=True,
        )
        return dsl.append_verification(note), "replan"

    if maneuver.kind == "pass" and not world_after.passed:
        try:
            safety = check_pass_safety(dsl, spec, world_after)
            if not safety.approved:
                note = VerificationNote(
                    verdict="replan",
                    message="Post-step vision unsafe: " + "; ".join(safety.reasons[:2]),
                    revision_triggered=True,
                )
                return dsl.append_verification(note), "replan"
        except InsufficientPerceptionError:
            pass

    if world_after.ego_x_m >= spec.route.goal_x_m:
        note = VerificationNote(verdict="ok", message="Goal reached.")
        return dsl.append_verification(note), "ok"

    note = VerificationNote(verdict="ok", message="Step executed without anomaly.")
    return dsl.append_verification(note), "ok"
