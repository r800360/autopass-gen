"""Scenario 0 stays wait; demo_07 supports perception-grounded pass when evidence is complete."""
from __future__ import annotations

from dataclasses import replace

import pytest

from autopass.critic import critique_maneuver_proposal, critique_tool_result
from autopass.dsl import PassingDSL, PerceptionRecord, VerificationNote, init_dsl_from_request
from autopass.perception_state import pass_evidence_complete, required_pass_tools
from autopass.planner import MIN_PASS_FRONT_GAP_M, plan_next
from autopass.tools import run_tool
from visual_world import curated_demo_scenarios, initialize_world


def _complete_pass_dsl(
    *,
    front: float,
    lead_speed: float,
    rear: float,
    oncoming_available: bool,
    oncoming_reason: str = "",
) -> PassingDSL:
    spec = curated_demo_scenarios()[6]
    world = initialize_world(spec)
    dsl = init_dsl_from_request(spec.request.text, aggression="high")
    wb = replace(
        dsl.world_belief,
        source="carla_depth",
        front_gap_m=front,
        rear_gap_m=rear,
        oncoming_gap_m=None if not oncoming_available else 280.0,
        front_valid=True,
        rear_valid=True,
        oncoming_valid=oncoming_available,
        oncoming_available=oncoming_available,
        oncoming_unavailable_reason=oncoming_reason,
        lead_speed_mps=lead_speed,
        rear_closing_mps=0.5,
        visibility_m=200.0,
    )
    dsl = dsl.update_belief(wb)
    dsl = dsl.append_perception(
        PerceptionRecord(
            tool="capture_sensors",
            summary="burst",
            data={
                "car_distances": [{"position": "front", "median_depth": front, "used_for_front_gap": True}],
                "front_speed_mps": lead_speed,
                "rear_closing_mps": 0.5,
                "oncoming_available": oncoming_available,
                "oncoming_unavailable_reason": oncoming_reason,
            },
        )
    )
    dsl = dsl.append_verification(VerificationNote(verdict="ok", message="capture ok", tool="capture_sensors"))
    for tool in ("measure_front_gap", "measure_rear_gap", "check_kinematics"):
        dsl, payload = run_tool(tool, dsl, spec, world)
        dsl, verdict = critique_tool_result(dsl, tool, payload, spec, world)
        assert verdict == "ok"
    if oncoming_available:
        dsl, payload = run_tool("measure_oncoming", dsl, spec, world)
        dsl, verdict = critique_tool_result(dsl, "measure_oncoming", payload, spec, world)
        assert verdict == "ok"
    else:
        dsl, payload = run_tool("measure_oncoming", dsl, spec, world)
        dsl, verdict = critique_tool_result(dsl, "measure_oncoming", payload, spec, world)
        assert verdict == "ok"
        assert payload.get("not_applicable") is True
    return dsl


def test_scenario_0_index_unchanged_negative_wait():
    from autopass.critic import critique_tool_result

    spec = curated_demo_scenarios()[0]
    assert spec.scenario_id == "demo_01_clear_urgent_safe_pass"
    world = initialize_world(spec)
    dsl = init_dsl_from_request(spec.request.text, aggression="high")
    wb = replace(
        dsl.world_belief,
        source="carla_depth",
        front_gap_m=12.0,
        front_valid=True,
        lead_speed_mps=None,
        rear_valid=False,
        oncoming_available=False,
        oncoming_unavailable_reason="no_opposing_lane_or_actor",
    )
    dsl = dsl.update_belief(wb)
    dsl = dsl.append_perception(
        PerceptionRecord(tool="capture_sensors", summary="burst", data={"car_distances": []})
    )
    dsl = dsl.append_verification(VerificationNote(verdict="ok", message="capture ok", tool="capture_sensors"))
    dsl, payload = run_tool("measure_front_gap", dsl, spec, world)
    dsl, _ = critique_tool_result(dsl, "measure_front_gap", payload, spec, world)
    decision = plan_next(dsl, spec, world)
    assert decision.maneuver == "wait"
    assert 12.0 < MIN_PASS_FRONT_GAP_M


def test_demo_07_exists_and_is_positive_pass_index():
    specs = curated_demo_scenarios()
    assert len(specs) >= 7
    assert specs[6].scenario_id == "demo_07_clear_safe_pass_perception"
    assert specs[6].lead.distance_m > specs[0].lead.distance_m


def test_pass_evidence_omits_oncoming_when_unavailable():
    dsl = _complete_pass_dsl(
        front=32.0,
        lead_speed=5.0,
        rear=55.0,
        oncoming_available=False,
        oncoming_reason="no_opposing_lane_or_actor",
    )
    assert "measure_oncoming" not in required_pass_tools(dsl)
    assert pass_evidence_complete(dsl)


def test_demo_07_spawn_profile_is_axis_not_waypoint_only():
    from perception.carla_scenario import CarlaScenarioSession

    spec = curated_demo_scenarios()[6]
    profile = CarlaScenarioSession._spawn_profile(spec)
    assert profile["axis_spawn"] is True
    assert float(profile["lead_gap_m"]) >= 32.0


def test_planner_and_critic_approve_pass_for_demo_07_evidence():
    spec = curated_demo_scenarios()[6]
    world = initialize_world(spec)
    dsl = _complete_pass_dsl(
        front=32.0,
        lead_speed=5.0,
        rear=55.0,
        oncoming_available=False,
        oncoming_reason="no_opposing_lane_or_actor",
    )
    decision = plan_next(dsl, spec, world)
    assert decision.maneuver == "pass"
    _, verdict, plan = critique_maneuver_proposal(dsl, "pass", spec, world)
    assert verdict == "ok"
    assert plan.kind == "pass"
