"""HUD belief / gate panel for CARLA demo video — makes vision-only decisions inspectable on frame."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from autopass.dsl import PassingDSL, dsl_from_dict
from autopass.pass_gates import evaluate_pass_gates
from autopass.perception_state import deadline_pressure, urgency_level
from autopass.tools import perception_summary
from visual_world import ScenarioSpec, WorldState


def build_demo_belief_panel(
    spec: ScenarioSpec,
    world: WorldState,
    dsl: PassingDSL | Dict[str, Any],
    *,
    node_label: str = "",
    trace_tail: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compact dict burned into demo HUD each graph step."""
    if isinstance(dsl, dict):
        dsl = dsl_from_dict(dsl)
    summary = perception_summary(dsl)
    gates = evaluate_pass_gates(dsl, spec, world, summary=summary, pass_in_progress=bool(
        (trace_tail or {}).get("pass_in_progress")
    ))
    wb = dsl.world_belief
    tail = trace_tail or {}
    lr = tail.get("lead_resolution") or {}
    panel: Dict[str, Any] = {
        "node": (node_label or tail.get("node", "")).upper(),
        "maneuver": tail.get("maneuver") or tail.get("action") or gates.get("decision_rule_source"),
        "can_pass": gates.get("can_pass"),
        "front_m": wb.front_gap_m,
        "front_ok": (gates.get("pass_preconditions") or {}).get("front_gap_ok", wb.front_valid),
        "front_src": (wb.car_distances[0].get("calibrated_gap_source") if wb.car_distances else None)
        or lr.get("calibrated_gap_source")
        or tail.get("calibrated_gap_source"),
        "rear_m": wb.rear_gap_m,
        "rear_ok": wb.rear_valid,
        "oncoming_m": wb.oncoming_gap_m if wb.oncoming_available else "n/a",
        "oncoming_ok": wb.oncoming_valid if wb.oncoming_available else "n/a",
        "lead_spd": wb.lead_speed_mps,
        "urgency": dsl.mission.urgency,
        "pressure": round(deadline_pressure(spec, world), 2),
        "urgency_live": urgency_level(spec, world),
        "blockers": (gates.get("pass_blockers") or [])[:2],
        "fsm": tail.get("pass_fsm_phase") or "",
        "vision_front": lr.get("used_detection_for_front") or tail.get("used_detection_for_front"),
        "agency": tail.get("agency_source") or "",
        "oracle": tail.get("decision_oracle_enabled"),
        "belief_src": wb.source,
        "exec_outcome": tail.get("execution_outcome") or "",
    }
    if tail.get("node") == "planner":
        dec = tail.get("decision") or {}
        if isinstance(dec, dict):
            panel["maneuver"] = dec.get("maneuver") or dec.get("action")
            panel["tool"] = dec.get("tool")
        panel["can_pass"] = tail.get("can_pass", panel["can_pass"])
        panel["blockers"] = (tail.get("pass_blockers") or panel["blockers"])[:2]
    if tail.get("node") == "critic_maneuver":
        panel["maneuver"] = tail.get("maneuver")
        panel["verdict"] = tail.get("verdict")
    return panel


def belief_panel_hud_lines(panel: Dict[str, Any]) -> List[str]:
    """Left-column HUD lines (research claim: vision gaps + gates + agency)."""
    lines: List[str] = ["paint ego=BLUE lead=RED rear=YELLOW"]
    node = str(panel.get("node") or "")
    if node:
        lines.append(f"AGENT::{node}")
    maneuver = panel.get("maneuver")
    tool = panel.get("tool")
    if tool:
        lines.append(f"TOOL {tool}")
    elif maneuver:
        lines.append(f"MANEUVER {maneuver}")
    verdict = panel.get("verdict")
    if verdict:
        lines.append(f"CRITIC {verdict}")
    cp = panel.get("can_pass")
    if cp is not None:
        lines.append(f"CAN_PASS {'YES' if cp else 'NO'}")
    blockers = panel.get("blockers") or []
    if blockers:
        lines.append(f"BLOCK {blockers[0][:42]}")
    front = panel.get("front_m")
    if front is not None:
        src = panel.get("front_src") or "?"
        lines.append(f"VISION front {float(front):.1f}m ({src})")
    rear = panel.get("rear_m")
    if rear is not None and panel.get("rear_ok"):
        lines.append(f"VISION rear {float(rear):.1f}m")
    onc = panel.get("oncoming_m")
    if onc != "n/a" and onc is not None:
        lines.append(f"VISION oncoming {float(onc):.0f}m")
    elif panel.get("oncoming_ok") == "n/a":
        lines.append("VISION oncoming n/a (1-lane pass)")
    lead = panel.get("lead_spd")
    if lead is not None:
        lines.append(f"lead {float(lead):.1f} m/s")
    lines.append(
        f"URGENCY {panel.get('urgency')} p={panel.get('pressure')} live={panel.get('urgency_live')}"
    )
    fsm = panel.get("fsm")
    if fsm:
        lines.append(f"PASS_FSM {fsm}")
    if panel.get("vision_front"):
        lines.append("front=DETECTION")
    agency = panel.get("agency")
    if agency:
        lines.append(f"agency={agency} oracle={'on' if panel.get('oracle') else 'OFF'}")
    elif panel.get("oracle") is False:
        lines.append("oracle=OFF (vision-only)")
    outcome = panel.get("exec_outcome")
    if outcome:
        lines.append(str(outcome))
    return lines[:14]
