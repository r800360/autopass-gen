"""Passing-lane topology from CARLA session layout (not ScenarioSpec)."""
from __future__ import annotations

from typing import Any, Dict


def passing_lane_topology(session) -> Dict[str, Any]:
    """
    Classify whether pass uses same-direction adjacent lane vs opposing lane.

    Sets oncoming_required and oncoming_available consistently for belief/tools.
    """
    if session is None or not getattr(session, "ready", False):
        return {
            "passing_topology": "unknown",
            "oncoming_required": True,
            "oncoming_check_reason": "session_not_ready",
            "oncoming_available": True,
            "oncoming_unavailable_reason": "",
        }

    has_passing_wp = getattr(session, "_passing_wp", None) is not None
    has_opposing_wp = bool(getattr(session, "_opposing_wp", None))
    rear_on_passing = bool(getattr(session, "_rear_on_passing_lane", False))
    oncoming_actor = session.actors.get("oncoming") if getattr(session, "actors", None) else None

    if rear_on_passing or (has_passing_wp and not has_opposing_wp):
        return {
            "passing_topology": "same_direction_adjacent_lane",
            "oncoming_required": False,
            "oncoming_check_reason": (
                "rear_spawned_on_adjacent_passing_lane; oncoming not required for same-direction pass"
            ),
            "oncoming_available": False,
            "oncoming_unavailable_reason": "same_direction_passing_lane",
        }

    if has_opposing_wp and oncoming_actor is not None:
        return {
            "passing_topology": "opposing_lane",
            "oncoming_required": True,
            "oncoming_check_reason": "opposing_lane_present_with_oncoming_actor",
            "oncoming_available": True,
            "oncoming_unavailable_reason": "",
        }

    if has_opposing_wp:
        return {
            "passing_topology": "opposing_lane",
            "oncoming_required": True,
            "oncoming_check_reason": "opposing_lane_present_but_no_oncoming_actor_spawned",
            "oncoming_available": False,
            "oncoming_unavailable_reason": "no_oncoming_actor",
        }

    return {
        "passing_topology": "travel_lane_only",
        "oncoming_required": False,
        "oncoming_check_reason": "no_opposing_lane_at_spawn",
        "oncoming_available": False,
        "oncoming_unavailable_reason": "no_opposing_lane_or_actor",
    }


def apply_topology_to_oncoming_belief(belief, topo: Dict[str, Any]):
    """Patch WorldBelief oncoming fields from topology (in-place dataclass replace at call site)."""
    from dataclasses import replace

    if not topo.get("oncoming_required", True):
        return replace(
            belief,
            oncoming_available=False,
            oncoming_valid=False,
            oncoming_gap_m=None,
            oncoming_unavailable_reason=str(topo.get("oncoming_unavailable_reason", "")),
        )
    return belief
