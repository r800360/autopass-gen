"""Tests for objective pass maneuver quality scoring."""
from __future__ import annotations

from types import SimpleNamespace

from autopass.pass_quality import score_pass_steps


def test_score_pass_steps_ok_when_centered():
    steps = []
    for i in range(10):
        steps.append(
            SimpleNamespace(scripted_phase="lane_change", lane_center_dist_m=0.4 + i * 0.05)
        )
    for i in range(10):
        steps.append(SimpleNamespace(scripted_phase="overtake", lane_center_dist_m=0.3 + i * 0.02))
    for i in range(8):
        steps.append(SimpleNamespace(scripted_phase="merge_back", lane_center_dist_m=0.2 + i * 0.03))
    report = score_pass_steps(steps)
    assert report.ok
    assert report.merge_steps == 8
    assert report.lane_change_p95_m < 1.0


def test_score_pass_steps_fails_without_merge():
    steps = [
        SimpleNamespace(scripted_phase="lane_change", lane_center_dist_m=0.5),
        SimpleNamespace(scripted_phase="overtake", lane_center_dist_m=0.4),
    ]
    report = score_pass_steps(steps)
    assert not report.ok
    assert "no_merge_back_phase" in report.issues
