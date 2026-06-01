"""Steering must not lock at ~180° when pass target waypoint is behind ego."""
from __future__ import annotations

from types import SimpleNamespace

from perception.carla_lane_keep import steer_from_lane_errors


def test_steer_from_lane_errors_bounded_heading():
    loc = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    steer, head, lat = steer_from_lane_errors(loc, 0.0, 12.0, 0.4, max_steer=0.2)
    assert abs(head) < 90.0
    assert abs(steer) <= 0.2
