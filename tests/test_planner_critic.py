from autopass.critic import critique_maneuver_proposal, critique_tool_result
from autopass.dsl import init_dsl_from_request
from autopass.planner import plan_next
from autopass.tools import run_tool
from visual_world import curated_demo_scenarios, initialize_world


def _run_full_vision_chain(spec, world, dsl):
    dsl, _ = run_tool("capture_sensors", dsl, spec, world)
    for tool in ("measure_front_gap", "measure_rear_gap", "measure_oncoming", "check_kinematics"):
        dsl, payload = run_tool(tool, dsl, spec, world)
        dsl, _ = critique_tool_result(dsl, tool, payload, spec, world)
    return dsl


def test_planner_requests_capture_sensors_first():
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = init_dsl_from_request(spec.request.text, aggression="high")
    decision = plan_next(dsl, spec, world)
    assert decision.action == "run_tool"
    assert decision.tool == "capture_sensors"
    assert "burst" in decision.reasoning.lower() or "frame" in decision.reasoning.lower() or "Need" in decision.reasoning


def test_planner_does_not_pass_under_low_urgency():
    spec = curated_demo_scenarios()[3]
    world = initialize_world(spec)
    dsl = init_dsl_from_request(spec.request.text, aggression="low")
    dsl = _run_full_vision_chain(spec, world, dsl)
    decision = plan_next(dsl, spec, world)
    while decision.action == "run_tool":
        dsl, payload = run_tool(decision.tool, dsl, spec, world)
        dsl, _ = critique_tool_result(dsl, decision.tool, payload, spec, world)
        decision = plan_next(dsl, spec, world)
    assert decision.action == "decide_maneuver"
    assert decision.maneuver == "wait"


def test_critic_rejects_unsafe_pass_without_full_evidence():
    spec = curated_demo_scenarios()[1]
    world = initialize_world(spec)
    dsl = init_dsl_from_request(spec.request.text, aggression="high")
    dsl, verdict, plan = critique_maneuver_proposal(dsl, "pass", spec, world)
    assert verdict == "reject"
    assert plan.kind == "wait"


def test_critic_rejects_unsafe_pass_after_vision_chain():
    spec = curated_demo_scenarios()[1]
    world = initialize_world(spec)
    dsl = init_dsl_from_request(spec.request.text, aggression="high")
    dsl = _run_full_vision_chain(spec, world, dsl)
    dsl, verdict, plan = critique_maneuver_proposal(dsl, "pass", spec, world)
    assert verdict == "reject"
    assert plan.kind == "wait"


def test_llm_plan_does_not_recurse(monkeypatch):
    calls = {"n": 0}

    def fake_structured(model, system, human, mock_value):
        calls["n"] += 1
        return mock_value

    monkeypatch.setattr("agents.llm_agents.use_mock_llm", lambda: False)
    monkeypatch.setattr("agents.llm_agents.structured_invoke", fake_structured)

    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = init_dsl_from_request(spec.request.text, aggression="high")
    plan_next(dsl, spec, world)
    assert calls["n"] == 1
