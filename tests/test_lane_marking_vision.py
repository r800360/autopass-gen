"""Lane marking vision cues from semantic segmentation."""
from __future__ import annotations

import numpy as np

from perception.carla_lane_keep import lane_change_blend_alpha
from perception.lane_marking_vision import analyze_lane_markings_under_ego


def test_center_line_detected_under_ego():
    seg = np.zeros((200, 320), dtype=np.uint8)
    seg[120:190, 145:175] = 6
    hint = analyze_lane_markings_under_ego(seg, passing_side="left")
    assert hint["center_line_under_ego"]
    assert hint["passing_commit_boost"] > 0.1


def test_merge_back_blend_zero_until_passing_lane_captured():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from perception.carla_scenario import CarlaScenarioSession

    session = CarlaScenarioSession()
    session.lateral_lane_offsets_m = lambda ego: (0.1, 3.4, 3.5)
    assert session.merge_back_blend_alpha(SimpleNamespace()) == 0.0
    session.lateral_lane_offsets_m = lambda ego: (3.2, 0.25, 3.5)
    assert session.merge_back_blend_alpha(SimpleNamespace()) < 0.15


def test_blend_alpha_uses_passing_center_distance():
    w = 3.5
    a_shift = lane_change_blend_alpha(1.75, w)
    a_commit = lane_change_blend_alpha(1.75, w, dist_to_passing_center_m=1.75)
    assert a_commit > a_shift
    assert a_commit >= 0.78
