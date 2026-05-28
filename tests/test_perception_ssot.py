"""Decisions must follow vision-derived state, not scenario VehicleSpec priors."""
from __future__ import annotations

from dataclasses import replace

import pytest

from autopass.critic import critique_maneuver_proposal
from autopass.dsl import PassingDSL, PerceptionRecord, dsl_from_dict, init_dsl_from_request
from autopass.perception_state import (
    InsufficientPerceptionError,
    measured_gaps,
    patch_belief_from_capture,
    slow_lead,
)
from autopass.planner import plan_next
from autopass.tools import run_tool
from visual_world import ScenarioSpec, VehicleSpec, curated_demo_scenarios, initialize_world, render_sensor_frame


def _dsl_with_vision_burst(spec, world) -> PassingDSL:
    dsl = init_dsl_from_request(spec.request.text, aggression="high")
    dsl, _ = run_tool("capture_sensors", dsl, spec, world)
    return dsl


def _set_belief_gaps(dsl: PassingDSL, *, front: float, rear: float, oncoming: float, lead_speed: float) -> PassingDSL:
    from dataclasses import replace as r

    wb = r(
        dsl.world_belief,
        source="visual_depth",
        front_gap_m=front,
        rear_gap_m=rear,
        oncoming_gap_m=oncoming,
        front_valid=True,
        rear_valid=True,
        oncoming_valid=True,
        oncoming_available=True,
        lead_speed_mps=lead_speed,
        rear_closing_mps=0.5,
    )
    dsl = dsl.update_belief(wb)
    dsl = dsl.append_perception(
        PerceptionRecord(
            tool="capture_sensors",
            summary="test",
            data={
                "front_speed_mps": lead_speed,
                "rear_closing_mps": 0.5,
                "car_distances": [{"position": "front", "median_depth": front}],
            },
        )
    )
    from autopass.dsl import VerificationNote

    dsl = dsl.append_verification(VerificationNote(verdict="ok", message="capture ok", tool="capture_sensors"))
    return dsl


def test_measured_gaps_ignore_wrong_spec_distance():
    spec = curated_demo_scenarios()[0]
    spec_wrong = replace(spec, lead=VehicleSpec(distance_m=999.0, speed_mps=99.0))
    world = initialize_world(spec_wrong)
    dsl = _set_belief_gaps(init_dsl_from_request("test"), front=26.0, rear=100.0, oncoming=280.0, lead_speed=5.5)
    gaps = measured_gaps(dsl)
    assert gaps["front_m"] == pytest.approx(26.0, abs=0.1)
    assert slow_lead(dsl, world)


def test_planner_decision_unchanged_when_spec_lead_distance_wrong():
    from autopass.critic import critique_tool_result

    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = _set_belief_gaps(init_dsl_from_request(spec.request.text, aggression="high"), front=26.0, rear=110.0, oncoming=290.0, lead_speed=5.6)
    for tool in ("measure_front_gap", "measure_rear_gap", "measure_oncoming", "check_kinematics"):
        dsl, payload = run_tool(tool, dsl, spec, world)
        dsl, _ = critique_tool_result(dsl, tool, payload, spec, world)
    baseline = plan_next(dsl, spec, world)
    spec_lying = replace(spec, lead=VehicleSpec(distance_m=5.0, speed_mps=20.0))
    lied = plan_next(dsl, spec_lying, world)
    assert lied.action == baseline.action
    assert lied.tool == baseline.tool
    assert lied.maneuver == baseline.maneuver


def test_critic_rejects_pass_without_vision_tools():
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = init_dsl_from_request(spec.request.text, aggression="high")
    _, verdict, plan = critique_maneuver_proposal(dsl, "pass", spec, world)
    assert verdict == "reject"
    assert plan.kind == "wait"


def test_rendered_frame_gap_matches_layout_not_spec_typo():
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    _, seg, depth, _ = render_sensor_frame(spec, world)
    from visual_world import LABELS, extract_depth_from_frame

    depth_result = extract_depth_from_frame(seg, depth)
    front = min(c["median_depth"] for c in depth_result["car_distances"] if c["position"] == "front")
    assert 15.0 < front < 40.0
    assert abs(front - (world.lead_x_m - world.ego_x_m)) < 3.0


def test_insufficient_belief_raises_for_safety():
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = init_dsl_from_request("test")
    with pytest.raises(InsufficientPerceptionError):
        measured_gaps(dsl)


def test_redundant_tool_rejected_by_critic():
    from autopass.critic import critique_tool_result

    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = _set_belief_gaps(init_dsl_from_request("t"), front=30.0, rear=90.0, oncoming=250.0, lead_speed=6.0)
    dsl, payload = run_tool("measure_front_gap", dsl, spec, world)
    dsl, verdict0 = critique_tool_result(dsl, "measure_front_gap", payload, spec, world)
    assert verdict0 == "ok"
    dsl, payload = run_tool("measure_front_gap", dsl, spec, world)
    dsl2, verdict = critique_tool_result(dsl, "measure_front_gap", payload, spec, world)
    assert verdict == "reject"


def test_rejected_tool_cannot_become_planner_evidence():
    from autopass.critic import critique_tool_result
    from autopass.perception_state import needed_tools

    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = init_dsl_from_request("t", aggression="high")

    # Capture sensors returns no cars -> critic marks insufficient and tool is invalidated.
    dsl, payload = run_tool("capture_sensors", dsl, spec, world)
    payload["car_distances"] = []
    dsl, verdict = critique_tool_result(dsl, "capture_sensors", payload, spec, world)
    assert verdict == "insufficient"
    assert "capture_sensors" not in dsl.tools_completed
    assert all(rec.tool != "capture_sensors" for rec in dsl.perception_log)
    missing = [t for t, _ in needed_tools(dsl, spec, world)]
    assert "capture_sensors" in missing


def test_oncoming_unavailable_not_fake_small_gap():
    from autopass.belief import observe_from_carla_session

    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)

    class _FakeSession:
        ready = True
        actors = {"oncoming": None}
        _opposing_wp = None

        def sync_npc_poses(self, *_args, **_kwargs):
            return None

        def tick(self):
            return None

        def grab_frame(self):
            import numpy as np

            # Empty segmentation/depth => no actors detected.
            seg = np.zeros((16, 16), dtype=np.uint8)
            depth = np.full((16, 16), 200.0, dtype=np.float32)
            rgb = np.zeros((16, 16, 3), dtype=np.uint8)
            return rgb, seg, depth

    fake = _FakeSession()

    import autopass.belief as belief_mod

    orig_get_session = belief_mod.__dict__.get("get_session", None)
    # Patch via perception.carla_scenario module function used inside observe_from_carla_session
    import perception.carla_scenario as cs

    old = cs.get_session
    cs.get_session = lambda: fake
    try:
        belief, payload = observe_from_carla_session(spec, world)
    finally:
        cs.get_session = old
    assert belief is not None
    assert belief.oncoming_available is False
    assert belief.oncoming_gap_m is None
    assert payload.get("oncoming_available") is False
