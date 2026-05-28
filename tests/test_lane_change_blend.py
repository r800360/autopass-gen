"""Lane-change blend steering (corridor-anchored lateral shift)."""
from __future__ import annotations

from types import SimpleNamespace

from perception.carla_lane_keep import blend_locations, lane_change_blend_alpha


def test_blend_locations_midpoint():
    t = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    p = SimpleNamespace(x=0.0, y=4.0, z=0.0)
    mid = blend_locations(t, p, 0.5)
    assert mid.y == 2.0


def test_blend_alpha_monotonic_with_shift():
    w = 3.5
    a0 = lane_change_blend_alpha(0.0, w)
    a1 = lane_change_blend_alpha(1.0, w)
    a2 = lane_change_blend_alpha(3.0, w)
    assert a0 < a1 < a2
