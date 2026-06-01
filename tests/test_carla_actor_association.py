"""Actor association invariants: rear never front; axis fallback; rear gap from geometry."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from perception.carla_actor_association import (
    REASON_AXIS_FALLBACK,
    assert_no_rear_accepted_for_front,
    resolve_lead_front_gap,
)
from perception.passing_topology import passing_lane_topology


def _session_with_lead(
    *,
    lead_gap_m: float = 19.0,
    rear_gap_m: float = 18.0,
    lead_speed: float = 0.0,
    rear_on_passing_lane: bool = True,
):
    session = MagicMock()
    session.ready = True
    session.lead_longitudinal_gap_m.return_value = lead_gap_m
    session.rear_longitudinal_gap_m.return_value = rear_gap_m
    session.signed_gap_from_ego.side_effect = lambda name: {
        "lead": lead_gap_m,
        "rear": -rear_gap_m,
        "oncoming": 999.0,
    }.get(name)
    session.actors = {
        "lead": MagicMock(id=101),
        "rear": MagicMock(id=102),
        "ego": MagicMock(id=100),
    }
    session._travel_wp = MagicMock(lane_id=5)
    session._passing_wp = MagicMock(lane_id=4)
    session._opposing_wp = None
    session._rear_on_passing_lane = rear_on_passing_lane
    session.map = MagicMock()
    wp = MagicMock(lane_id=5, road_id=6)
    session.map.get_waypoint.return_value = wp

    lead = session.actors["lead"]
    lead.get_velocity.return_value = MagicMock(x=0.0, y=lead_speed, z=0.0)

    return session


@pytest.fixture(autouse=True)
def _enable_decision_oracle(monkeypatch):
    monkeypatch.setenv("AUTOPASS_DECISION_ORACLE", "1")


def test_rear_matched_detection_never_accepted_for_front():
    session = _session_with_lead(lead_gap_m=19.0, rear_gap_m=18.0)
    classified = [
        {
            "bbox": [290, 120, 330, 150],
            "median_depth": 15.48,
            "position": "rear",
            "depth_m": 15.48,
        },
        {
            "bbox": [0, 0, 639, 130],
            "median_depth": 36.0,
            "position": "rear_right",
            "depth_m": 36.0,
        },
    ]
    out, gaps, meta = resolve_lead_front_gap(session, classified)
    assert gaps["front_gap_m"] == 19.0
    assert meta["front_resolution_reason"] == REASON_AXIS_FALLBACK
    assert meta["used_detection_for_front"] is False
    assert meta["calibrated_gap_source"] == "actor_axis"
    assert not any(c.get("used_for_front_gap") for c in out)
    assert not any(c.get("accepted_for_front") for c in out)
    assert_no_rear_accepted_for_front(out)


def test_actor_axis_lead_fallback_without_attaching_rear_detection():
    session = _session_with_lead(lead_gap_m=19.024, rear_gap_m=18.001)
    classified = [
        {"bbox": [1, 1, 2, 2], "median_depth": 15.5, "position": "rear", "depth_m": 15.5},
    ]
    _, gaps, meta = resolve_lead_front_gap(session, classified)
    assert gaps["front_gap_m"] == pytest.approx(19.024, rel=1e-3)
    assert meta["used_detection_for_front"] is False
    assert meta.get("matched_actor_for_detection") is None
    assert meta["lead_actor_id"] == 101
    assert meta["lead_axis_gap_m"] == pytest.approx(19.024, rel=1e-3)


def test_rear_gap_prefers_actor_axis_over_far_blob():
    session = _session_with_lead(lead_gap_m=19.0, rear_gap_m=18.001)
    classified = [
        {"bbox": [290, 120, 330, 150], "median_depth": 15.5, "position": "rear", "depth_m": 15.5},
        {"bbox": [0, 0, 639, 130], "median_depth": 36.36, "position": "rear_right", "depth_m": 36.36},
    ]
    _, gaps, meta = resolve_lead_front_gap(session, classified)
    assert gaps["rear_gap_m"] == pytest.approx(18.001, rel=0.05)
    assert gaps["rear_gap_m"] < 25.0
    assert meta["rear_gap_source"] == "actor_axis_rear_actor"


def test_trace_fields_distinguish_label_actor_and_belief_source():
    session = _session_with_lead(lead_gap_m=32.0, rear_gap_m=18.0)
    classified = [
        {"bbox": [100, 200, 200, 300], "median_depth": 12.0, "position": "rear", "depth_m": 12.0},
    ]
    out, _, meta = resolve_lead_front_gap(session, classified)
    det = out[0]
    assert det["raw_position_label"] == "rear"
    assert det.get("used_for_front_gap") is False
    assert meta["used_detection_for_front"] is False
    assert meta["calibrated_gap_source"] == "actor_axis"
    assert meta["front_resolution_reason"] == REASON_AXIS_FALLBACK


def test_passing_topology_same_direction_when_rear_on_passing_lane():
    session = _session_with_lead(rear_on_passing_lane=True)
    topo = passing_lane_topology(session)
    assert topo["passing_topology"] == "same_direction_adjacent_lane"
    assert topo["oncoming_required"] is False
    assert topo["oncoming_available"] is False


def test_empty_pool_front_valid_from_lead_actor_axis_only():
    session = _session_with_lead(lead_gap_m=32.0)
    _, gaps, meta = resolve_lead_front_gap(session, [])
    assert gaps["front_gap_m"] == 32.0
    assert meta["front_resolution_reason"] == REASON_AXIS_FALLBACK
    assert meta["used_detection_for_front"] is False


def test_vision_only_axis_fallback_when_no_forward_detection(monkeypatch):
    monkeypatch.setenv("AUTOPASS_DECISION_ORACLE", "0")
    session = _session_with_lead(lead_gap_m=32.0, rear_gap_m=18.0)
    classified = [
        {"bbox": [290, 120, 330, 150], "median_depth": 15.5, "position": "rear", "depth_m": 15.5},
    ]
    _, gaps, meta = resolve_lead_front_gap(session, classified)
    assert gaps["front_gap_m"] == pytest.approx(32.0, rel=0.05)
    assert meta["used_detection_for_front"] is False
    assert meta["calibrated_gap_source"] == "actor_axis_fallback"
    assert meta.get("decision_oracle") is False


def test_vision_only_front_from_forward_detection(monkeypatch):
    monkeypatch.setenv("AUTOPASS_DECISION_ORACLE", "0")
    session = _session_with_lead(lead_gap_m=32.0, rear_gap_m=18.0)
    classified = [
        {"bbox": [400, 400, 500, 500], "median_depth": 28.0, "position": "front", "depth_m": 28.0},
        {"bbox": [290, 120, 330, 150], "median_depth": 15.5, "position": "rear", "depth_m": 15.5},
    ]
    _, gaps, meta = resolve_lead_front_gap(session, classified)
    assert meta["used_detection_for_front"] is True
    assert gaps["front_gap_m"] == pytest.approx(28.0, rel=0.1)
    assert meta.get("decision_oracle") is False


def test_vision_only_picks_closest_front_not_farthest_blob(monkeypatch):
    """Two forward detections: must not keep stale 28m when 21m is the nearer lead."""
    monkeypatch.setenv("AUTOPASS_DECISION_ORACLE", "0")
    session = _session_with_lead(lead_gap_m=22.0, rear_gap_m=18.0)
    classified = [
        {"bbox": [400, 400, 500, 500], "median_depth": 28.0, "position": "front", "depth_m": 28.0},
        {
            "bbox": [420, 410, 520, 510],
            "median_depth": 21.0,
            "position": "front",
            "depth_m": 21.0,
            "matched_actor": "lead",
        },
        {"bbox": [290, 120, 330, 150], "median_depth": 15.5, "position": "rear", "depth_m": 15.5},
    ]
    _, gaps, meta = resolve_lead_front_gap(session, classified)
    assert meta["used_detection_for_front"] is True
    assert gaps["front_gap_m"] == pytest.approx(21.0, rel=0.05)


def test_axis_fallback_when_ego_on_passing_lane(monkeypatch):
    """Ego on lane 4, lead on travel lane 5 — axis gap must still resolve."""
    monkeypatch.setenv("AUTOPASS_DECISION_ORACLE", "0")
    session = _session_with_lead(lead_gap_m=12.5, rear_gap_m=18.0)
    session._travel_wp = MagicMock(lane_id=5)
    ego_loc = session.actors["ego"].get_location()
    lead_loc = session.actors["lead"].get_location()

    def _wp_for(loc, **kwargs):
        lane = 5 if loc == lead_loc else 4
        return MagicMock(lane_id=lane, road_id=6)

    session.map.get_waypoint.side_effect = _wp_for
    classified = [
        {"bbox": [1, 1, 2, 2], "median_depth": 15.5, "position": "rear", "depth_m": 15.5},
    ]
    _, gaps, meta = resolve_lead_front_gap(session, classified)
    assert gaps["front_gap_m"] == pytest.approx(12.5, rel=0.05)
    assert meta["calibrated_gap_source"] == "actor_axis_fallback"


def test_lead_matched_detection_can_be_used_for_front():
    session = _session_with_lead(lead_gap_m=30.0, rear_gap_m=20.0)
    classified = [
        {"bbox": [400, 400, 500, 500], "median_depth": 29.5, "position": "front", "depth_m": 29.5},
    ]
    out, gaps, meta = resolve_lead_front_gap(session, classified)
    assert meta["used_detection_for_front"] is True
    assert meta["matched_actor_for_detection"] == "lead"
    picks = [c for c in out if c.get("used_for_front_gap")]
    assert len(picks) == 1
    assert picks[0]["matched_actor"] == "lead"
    assert gaps["front_gap_m"] == pytest.approx(29.5, rel=0.1)
