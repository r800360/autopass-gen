"""Tests for autonomous CARLA control parameter tuning."""
from __future__ import annotations

from autopass.control_tune import (
    ControlParameterTuner,
    ControlProfile,
    control_objective,
    score_maneuver_result,
)
from autopass.pass_quality import PassQualityReport, score_pass_steps
from perception.carla_pass_maneuver import PassManeuverResult, PassStepRecord


def test_control_objective_prefers_complete_pass():
    good = PassQualityReport(
        ok=True,
        issues=[],
        lane_change_p95_m=1.0,
        overtake_p95_m=0.8,
        merge_p95_m=0.6,
        max_center_m=1.2,
        merge_steps=10,
    )
    bad = PassQualityReport(
        ok=False,
        issues=["no_merge_back_phase"],
        lane_change_p95_m=1.8,
        overtake_p95_m=4.5,
        merge_p95_m=999.0,
        max_center_m=5.0,
        merge_steps=0,
    )
    s_good = control_objective(
        quality=good,
        maneuver_ok=True,
        pass_complete=True,
        merged_back=True,
        used_pass_lane=True,
        issues=[],
    )
    s_bad = control_objective(
        quality=bad,
        maneuver_ok=False,
        pass_complete=False,
        merged_back=False,
        used_pass_lane=True,
        issues=["left_corridor_road:41"],
    )
    assert s_good > s_bad


def test_tuner_suggest_from_feedback_raises_steer_when_stuck():
    tuner = ControlParameterTuner(ControlProfile(max_steer=0.14))
    q = PassQualityReport(
        ok=False,
        issues=["never_entered_passing_lane"],
        lane_change_p95_m=0.5,
        overtake_p95_m=999.0,
        merge_p95_m=999.0,
        max_center_m=0.5,
        merge_steps=0,
    )
    adj = tuner.suggest_from_feedback(q, q.issues)
    assert adj.max_steer > 0.14


def test_score_maneuver_result_from_steps():
    steps = [
        PassStepRecord(
            step=0,
            scripted_phase="lane_change",
            control_action="pass",
            pass_phase="lane_change",
            lane_center_dist_m=1.0,
            steer=0.2,
        ),
    ]
    result = PassManeuverResult(
        ok=False,
        issues=["no_merge_back_phase"],
        steps=steps,
        max_lane_center_m=1.0,
        min_edge_clearance_m=1.0,
        merged_back=False,
        pass_lane_used=True,
        pass_complete=False,
        pass_attempts=1,
        oscillation_count=0,
        final_world=None,
    )
    score, quality = score_maneuver_result(result)
    assert score < 50.0
    assert quality.lane_change_p95_m == 1.0
