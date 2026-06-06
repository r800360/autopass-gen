"""HUD belief panel for CARLA demo overlays."""
from __future__ import annotations

from dataclasses import replace

from autopass.demo_belief_panel import belief_panel_hud_lines, build_demo_belief_panel
from autopass.dsl import init_dsl_from_request
from autopass.tools import run_tool
from visual_world import curated_demo_scenarios, initialize_world


def test_belief_panel_shows_vision_gaps_and_can_pass():
    spec = curated_demo_scenarios()[6]
    world = initialize_world(spec)
    dsl = init_dsl_from_request(spec.request.text, aggression="high", urgency="high")
    dsl, _ = run_tool("capture_sensors", dsl, spec, world)
    dsl, _ = run_tool("measure_front_gap", dsl, spec, world)
    panel = build_demo_belief_panel(
        spec,
        world,
        dsl,
        node_label="PLANNER",
        trace_tail={"node": "planner", "can_pass": True, "agency_source": "llm", "decision_oracle_enabled": False},
    )
    lines = belief_panel_hud_lines(panel)
    text = "\n".join(lines)
    assert "VISION front" in text
    assert "CAN_PASS" in text
    assert "oracle=OFF" in text
