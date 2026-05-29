"""Pass gate evaluation — stationary lead motivates pass when evidence complete."""
from __future__ import annotations

from dataclasses import replace

from autopass.critic import critique_maneuver_proposal, critique_tool_result
from autopass.dsl import PassingDSL, PerceptionRecord, VerificationNote, init_dsl_from_request
from autopass.pass_gates import evaluate_pass_gates, sanitize_burst_rear_closing
from autopass.planner import plan_next
from autopass.tools import run_tool
from visual_world import curated_demo_scenarios, initialize_world


def _demo07_complete_stationary() -> tuple:
    spec = curated_demo_scenarios()[6]
    world = initialize_world(spec)
    dsl = init_dsl_from_request(spec.request.text, aggression="high", urgency="high")
    wb = replace(
        dsl.world_belief,
        source="carla_depth",
        front_gap_m=32.0,
        rear_gap_m=18.0,
        oncoming_gap_m=None,
        front_valid=True,
        rear_valid=True,
        oncoming_valid=False,
        oncoming_available=False,
        oncoming_unavailable_reason="same_direction_passing_lane",
        lead_speed_mps=0.0,
        visibility_m=200.0,
    )
    dsl = dsl.update_belief(wb)
    close_raw, close_valid, _ = sanitize_burst_rear_closing(-78.0)
    assert close_valid is False
    dsl = dsl.append_perception(
        PerceptionRecord(
            tool="capture_sensors",
            summary="burst",
            data={
                "car_distances": [],
                "front_speed_mps": 0.0,
                "rear_closing_mps": close_raw,
                "rear_closing_valid": close_valid,
                "rear_closing_source": "burst_artifact_rejected",
                "hazard": False,
                "passing_topology": "same_direction_adjacent_lane",
                "oncoming_required": False,
                "oncoming_check_reason": "rear_spawned_on_adjacent_passing_lane",
                "oncoming_available": False,
                "oncoming_unavailable_reason": "same_direction_passing_lane",
            },
        )
    )
    dsl = dsl.append_verification(VerificationNote(verdict="ok", message="ok", tool="capture_sensors"))
    for tool in ("measure_front_gap", "measure_rear_gap", "measure_oncoming", "check_kinematics"):
        dsl, payload = run_tool(tool, dsl, spec, world)
        dsl, verdict = critique_tool_result(dsl, tool, payload, spec, world)
        assert verdict == "ok", tool
    return spec, world, dsl


def test_stationary_lead_complete_evidence_can_pass():
    spec, world, dsl = _demo07_complete_stationary()
    gates = evaluate_pass_gates(dsl, spec, world)
    assert gates["pass_preconditions"]["slow_lead_ok"] is True
    assert gates["pass_preconditions"]["oncoming_ok"] is True
    assert gates["oncoming_required"] is False
    assert gates["can_pass"] is True
    decision = plan_next(dsl, spec, world)
    assert decision.maneuver == "pass"
    _, verdict, plan = critique_maneuver_proposal(dsl, "pass", spec, world)
    assert verdict == "ok"
    assert plan.kind == "pass"
