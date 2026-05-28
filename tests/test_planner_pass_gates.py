"""Planner must not propose pass without complete, safe vision evidence."""
from __future__ import annotations

from dataclasses import replace

import pytest

from autopass.critic import critique_maneuver_proposal
from autopass.dsl import PassingDSL, PerceptionRecord, VerificationNote, init_dsl_from_request
from autopass.planner import MIN_PASS_FRONT_GAP_M, PlannerDecision, plan_next
from autopass.tools import run_tool
from visual_world import curated_demo_scenarios, initialize_world


def _dsl_gap_only(
    *,
    front: float = 12.0,
    lead_speed: float | None = None,
    rear_valid: bool = False,
    oncoming_available: bool = False,
    oncoming_reason: str = "no_opposing_lane_or_actor",
) -> PassingDSL:
    dsl = init_dsl_from_request("urgent pass mission", aggression="high", urgency="high")
    wb = replace(
        dsl.world_belief,
        source="carla_depth",
        front_gap_m=front,
        rear_gap_m=999.0,
        oncoming_gap_m=None,
        front_valid=True,
        rear_valid=rear_valid,
        oncoming_valid=False,
        oncoming_available=oncoming_available,
        oncoming_unavailable_reason=oncoming_reason,
        lead_speed_mps=lead_speed,
    )
    dsl = dsl.update_belief(wb)
    dsl = dsl.append_perception(
        PerceptionRecord(
            tool="capture_sensors",
            summary="burst",
            data={"car_distances": [], "front_speed_mps": lead_speed},
        )
    )
    dsl = dsl.append_verification(
        VerificationNote(verdict="ok", message="capture ok", tool="capture_sensors")
    )
    from autopass.critic import critique_tool_result

    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl, payload = run_tool("measure_front_gap", dsl, spec, world)
    dsl, _ = critique_tool_result(dsl, "measure_front_gap", payload, spec, world)
    return dsl


def test_planner_does_not_propose_pass_for_12m_front_gap():
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = _dsl_gap_only(front=12.0, lead_speed=None)
    decision = plan_next(dsl, spec, world)
    assert decision.action == "decide_maneuver"
    assert decision.maneuver == "wait"
    assert decision.maneuver != "pass"
    assert "12.0" in decision.reasoning
    assert "sufficient" not in decision.reasoning.lower() or "not justified" in decision.reasoning.lower()


def test_planner_does_not_propose_pass_when_lead_speed_invalid():
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = _dsl_gap_only(front=30.0, lead_speed=None)
    decision = plan_next(dsl, spec, world)
    assert decision.maneuver == "wait"
    assert "lead speed" in decision.reasoning.lower()


def test_planner_reasoning_does_not_claim_safe_pass_with_insufficient_evidence():
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = _dsl_gap_only(front=12.0, lead_speed=None)
    decision = plan_next(dsl, spec, world)
    lower = decision.reasoning.lower()
    assert "sufficient for a safe pass" not in lower
    assert "safe pass" not in lower or "not justified" in lower
    assert any(
        phrase in lower
        for phrase in ("unavailable", "incomplete", "not validated", "below", "not justified")
    )


def test_critic_still_rejects_pass_if_planner_proposes_it():
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = _dsl_gap_only(front=12.0, lead_speed=None)
    _, verdict, plan = critique_maneuver_proposal(dsl, "pass", spec, world)
    assert verdict == "reject"
    assert plan.kind == "wait"


def test_llm_pass_proposal_clamped_to_wait(monkeypatch):
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = _dsl_gap_only(front=12.0, lead_speed=None)

    bad = PlannerDecision(
        action="decide_maneuver",
        maneuver="pass",
        reasoning="The front gap is 12.0 meters, which is sufficient for a safe pass.",
    )

    def fake_structured(model, system, human, mock_value):
        return bad

    monkeypatch.setattr("agents.llm_agents.use_mock_llm", lambda: False)
    monkeypatch.setattr("agents.llm_agents.structured_invoke", fake_structured)

    decision = plan_next(dsl, spec, world)
    assert decision.maneuver == "wait"
    assert "12.0" in decision.reasoning
    assert "sufficient for a safe pass" not in decision.reasoning.lower()


def test_carla_like_scenario_first_maneuver_is_wait():
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = _dsl_gap_only(
        front=12.0,
        lead_speed=None,
        rear_valid=False,
        oncoming_available=False,
        oncoming_reason="no_opposing_lane_or_actor",
    )
    decision = plan_next(dsl, spec, world)
    assert decision.maneuver == "wait"
    assert 12.0 < MIN_PASS_FRONT_GAP_M
    assert "not justified" in decision.reasoning.lower() or "unavailable" in decision.reasoning.lower()
