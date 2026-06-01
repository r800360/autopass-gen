"""Mid-pass perception gates — stationary lead speed must not block actuation."""
from __future__ import annotations

from dataclasses import replace

from autopass.critic import critique_tool_result
from autopass.dsl import PassingDSL, PerceptionRecord, VerificationNote, init_dsl_from_request
from autopass.pass_gates import evaluate_pass_gates
from autopass.perception_state import needed_tools, patch_belief_from_capture
from autopass.tools import run_tool
from visual_world import curated_demo_scenarios, initialize_world


def _demo07_belief_no_lead_speed() -> tuple:
    spec = curated_demo_scenarios()[6]
    world = initialize_world(spec)
    dsl = init_dsl_from_request(spec.request.text, aggression="high", urgency="high")
    wb = replace(
        dsl.world_belief,
        source="carla_depth",
        front_gap_m=19.0,
        rear_gap_m=14.0,
        oncoming_gap_m=None,
        front_valid=True,
        rear_valid=True,
        oncoming_valid=False,
        oncoming_available=False,
        oncoming_unavailable_reason="same_direction_passing_lane",
        lead_speed_mps=None,
        visibility_m=200.0,
    )
    dsl = dsl.update_belief(wb)
    for tool in ("capture_sensors", "measure_front_gap", "measure_rear_gap", "check_kinematics"):
        dsl = dsl.append_verification(VerificationNote(verdict="ok", message="ok", tool=tool))
        dsl = dsl.append_perception(
            PerceptionRecord(tool=tool, summary="ok", data={"front_valid": True, "safe": True, "feasible": True})
        )
    return spec, world, dsl


def test_slow_lead_ok_without_measured_speed_when_gap_shows_slow_lead():
    spec, world, dsl = _demo07_belief_no_lead_speed()
    gates = evaluate_pass_gates(dsl, spec, world)
    assert gates["pass_preconditions"]["slow_lead_ok"] is True


def test_patch_belief_preserves_prior_lead_speed_when_burst_omits_speed():
    spec, world, dsl = _demo07_belief_no_lead_speed()
    prior = replace(dsl.world_belief, lead_speed_mps=0.0, front_gap_m=19.0, front_valid=True)
    patched = patch_belief_from_capture(
        prior,
        {
            "car_distances": [
                {
                    "position": "front",
                    "depth_m": 19.0,
                    "median_depth": 19.0,
                    "bbox": [500, 300, 700, 400],
                    "used_for_front_gap": True,
                }
            ],
            "front_speed_mps": None,
            "image_width": 1280.0,
            "image_height": 720.0,
        },
    )
    assert patched.lead_speed_mps == 0.0


def test_needed_tools_empty_when_pass_in_progress_and_evidence_complete():
    spec, world, dsl = _demo07_belief_no_lead_speed()
    assert needed_tools(dsl, spec, world, pass_in_progress=True) == []


def test_redundant_capture_during_pass_fsm_does_not_invalidate_evidence(monkeypatch):
    from perception.pass_control_fsm import PassControlState, get_pass_control_state

    spec, world, dsl = _demo07_belief_no_lead_speed()
    dsl, payload = run_tool("capture_sensors", dsl, spec, world)

    class _Session:
        ready = True
        actors = {}

    session = _Session()
    st = PassControlState(active=True, phase="overtake", maneuver_started=True)
    monkeypatch.setattr("perception.carla_scenario.get_session", lambda: session)
    monkeypatch.setattr("perception.pass_control_fsm.get_pass_control_state", lambda _s: st)

    dsl, verdict = critique_tool_result(dsl, "capture_sensors", payload, spec, world)
    assert verdict == "ok"
    assert "capture_sensors" in dsl.tools_completed
