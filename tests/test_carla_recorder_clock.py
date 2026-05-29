"""HUD display clock for CARLA demo frames."""
from pathlib import Path

from perception.carla_recorder import CarlaRecorder


def test_display_clock_advances_during_planning_burst():
    rec = CarlaRecorder(Path("runs/_test_recorder"), "test_scenario")
    t0 = rec.next_display_t_s(0.0, label="PLANNER")
    t1 = rec.next_display_t_s(0.0, label="RUN_TOOL")
    t2 = rec.next_display_t_s(0.0, label="CRITIQUE_MANEUVER")
    assert t0 < t1 < t2
    assert t2 > 0.0


def test_display_clock_jumps_on_execute():
    rec = CarlaRecorder(Path("runs/_test_recorder"), "test_scenario2")
    rec.next_display_t_s(0.0, label="PLANNER")
    rec.next_display_t_s(0.0, label="PLANNER")
    t_exec = rec.next_display_t_s(0.5, label="EXECUTE")
    assert t_exec == 0.5


def test_video_fps_tracks_sim_span():
    rec = CarlaRecorder(Path("runs/_test_recorder"), "test_fps")
    for i in range(5):
        rec.next_display_t_s(float(i) * 0.2, label="EXECUTE")
        rec.metadata.append({"t_s": float(i) * 0.2})
    # 4 frame intervals over 0.8s sim → 5 fps raw, clamped to production minimum 12
    assert rec._video_fps(5) == 12

    rec2 = CarlaRecorder(Path("runs/_test_recorder"), "test_fps2")
    for i in range(31):
        rec2.metadata.append({"t_s": float(i) * 0.04})
    assert rec2._video_fps(31) == 25


def test_realtime_mode_holds_time_during_planning():
    import os

    os.environ["AUTOPASS_VIDEO_REALTIME"] = "1"
    try:
        rec = CarlaRecorder(Path("runs/_test_recorder"), "test_scenario3")
        t0 = rec.next_display_t_s(0.0, label="PLANNER")
        t1 = rec.next_display_t_s(0.0, label="PLANNER")
        assert t0 == t1 == 0.0
        t_exec = rec.next_display_t_s(1.0, label="EXECUTE")
        assert t_exec == 1.0
    finally:
        os.environ.pop("AUTOPASS_VIDEO_REALTIME", None)
