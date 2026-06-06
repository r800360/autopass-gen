"""Pass-finish latch must survive abort and relax adjacent-lane physics."""
from __future__ import annotations

from types import SimpleNamespace

from perception.carla_validation import _validate_carla_actors
from perception.pass_control_fsm import abort_pass, check_multi_lane_departure, get_pass_control_state
from perception.pass_geometry import axis_ahead_of_lead, pass_finish_active, pass_geometry_exempt, pass_merge_back_due


def test_pass_finish_active_after_abort():
    session = SimpleNamespace(
        ready=True,
        _pass_corridor_committed=True,
        _pass_peak_shift_m=3.2,
        expected_passing_lane_width_m=lambda: 3.5,
        ego_cleared_lead=lambda c: False,
        actors={"ego": SimpleNamespace()},
    )
    st = get_pass_control_state(session)
    st.active = True
    st.maneuver_started = True
    st.phase = "overtake"
    abort_pass(session, "lane_departure_11.0m")
    assert get_pass_control_state(session).phase == "abort"
    assert pass_finish_active(session) is True
    assert pass_geometry_exempt(session) is True


def test_multi_lane_departure_skipped_while_finishing():
    tw = SimpleNamespace(
        lane_id=5,
        road_id=1,
        transform=SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0, z=0.0)),
    )
    pw = SimpleNamespace(
        lane_id=4,
        road_id=1,
        transform=SimpleNamespace(location=SimpleNamespace(x=0.0, y=3.5, z=0.0)),
    )
    session = SimpleNamespace(
        map=None,
        ready=True,
        _travel_wp=tw,
        _passing_wp=pw,
        _pass_corridor_committed=True,
        _pass_peak_shift_m=3.2,
        expected_passing_lane_width_m=lambda: 3.5,
        ego_cleared_lead=lambda c: False,
        lateral_lane_offsets_m=lambda ego: (11.0, 11.0, 3.5),
    )
    ego = SimpleNamespace(
        get_location=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
    )
    departed, reason = check_multi_lane_departure(session, ego)
    assert departed is False
    assert reason == ""


def test_pass_finish_active_during_merge_back_when_axis_ahead():
    session = SimpleNamespace(
        ready=True,
        _pass_corridor_committed=True,
        _pass_peak_shift_m=3.2,
        expected_passing_lane_width_m=lambda: 3.5,
        ego_cleared_lead=lambda c: False,
        actor_travel_s=lambda name: 36.0 if name == "ego" else 32.0,
    )
    st = get_pass_control_state(session)
    st.active = True
    st.maneuver_started = True
    st.phase = "merge_back"
    assert axis_ahead_of_lead(session, margin_m=3.0) is True
    assert pass_finish_active(session) is True
    assert pass_finish_active(session, clear_of_lead=True) is False


def test_pass_finish_active_after_long_clear_while_wide_lateral():
    """Longitudinal clearance must not end merge-back while ego is still off travel lane."""
    ego = SimpleNamespace(get_location=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0))
    session = SimpleNamespace(
        ready=True,
        _pass_corridor_committed=True,
        _pass_peak_shift_m=3.2,
        expected_passing_lane_width_m=lambda: 3.5,
        ego_cleared_lead=lambda c: True,
        actor_travel_s=lambda name: 48.0 if name == "ego" else 32.0,
        actors={"ego": ego},
        lateral_lane_offsets_m=lambda ego: (18.0, 19.0, 3.5),
    )
    st = get_pass_control_state(session)
    st.active = True
    st.maneuver_started = True
    st.phase = "merge_back"
    assert pass_finish_active(session) is True


def test_wide_off_corridor_detects_runaway_merge():
    from perception.pass_geometry import wide_off_corridor

    session = SimpleNamespace(
        ready=True,
        expected_passing_lane_width_m=lambda: 3.5,
        lateral_lane_offsets_m=lambda ego: (130.0, 128.0, 3.5),
    )
    assert wide_off_corridor(session) is True


def test_begin_pass_preserves_finish_latch():
    from perception.pass_control_fsm import begin_pass

    session = SimpleNamespace(
        ready=True,
        _pass_corridor_committed=True,
        _pass_peak_shift_m=3.2,
        _pass_finish_latch=True,
        _passing_wp=SimpleNamespace(lane_id=4, road_id=1),
        _travel_wp=SimpleNamespace(lane_id=5, road_id=1),
        expected_passing_lane_width_m=lambda: 3.5,
        ego_cleared_lead=lambda c: False,
        actor_travel_s=lambda name: 40.0 if name == "ego" else 32.0,
        lateral_lane_offsets_m=lambda ego: (18.0, 19.0, 3.5),
        ego_corridor_lane_offset_m=lambda ego: 18.0,
        map=None,
    )
    st = begin_pass(session)
    assert session._pass_peak_shift_m == 3.2
    assert st.phase == "merge_back"


def test_beside_or_ahead_triggers_merge_back_due():
    from perception.pass_geometry import beside_or_ahead_of_lead, pass_merge_back_due

    session = SimpleNamespace(
        ready=True,
        _pass_corridor_committed=True,
        _pass_peak_shift_m=3.2,
        expected_passing_lane_width_m=lambda: 3.5,
        ego_cleared_lead=lambda c: False,
        actor_travel_s=lambda name: 32.6 if name == "ego" else 32.0,
        lateral_lane_offsets_m=lambda ego: (18.0, 20.0, 3.5),
        ego_corridor_lane_offset_m=lambda ego: 18.0,
    )
    st = get_pass_control_state(session)
    st.active = True
    st.maneuver_started = True
    st.phase = "lane_change"
    assert beside_or_ahead_of_lead(session, margin_m=0.5)
    assert pass_merge_back_due(session) is True


def test_pass_merge_back_due_on_axis_ahead():
    from perception.pass_geometry import pass_merge_back_due

    session = SimpleNamespace(
        ready=True,
        _pass_corridor_committed=True,
        _pass_peak_shift_m=3.2,
        expected_passing_lane_width_m=lambda: 3.5,
        ego_cleared_lead=lambda c: False,
        actor_travel_s=lambda name: 36.0 if name == "ego" else 32.0,
        lateral_lane_offsets_m=lambda ego: (1.0, 1.2, 3.5),
        ego_corridor_lane_offset_m=lambda ego: 1.0,
    )
    st = get_pass_control_state(session)
    st.active = True
    st.maneuver_started = True
    st.phase = "overtake"
    assert pass_merge_back_due(session) is True


def test_pass_merge_back_not_due_while_behind_on_axis():
    from perception.pass_geometry import pass_merge_back_due, safe_to_merge_to_travel

    session = SimpleNamespace(
        ready=True,
        _pass_corridor_committed=True,
        _pass_peak_shift_m=3.2,
        expected_passing_lane_width_m=lambda: 3.5,
        ego_cleared_lead=lambda c: False,
        lead_longitudinal_gap_m=lambda: 5.0,
        actor_travel_s=lambda name: 27.0 if name == "ego" else 32.0,
    )
    st = get_pass_control_state(session)
    st.active = True
    st.maneuver_started = True
    st.phase = "overtake"
    assert safe_to_merge_to_travel(session) is False
    assert pass_merge_back_due(session) is False


def test_axis_ahead_of_lead():
    session = SimpleNamespace(
        ready=True,
        actor_travel_s=lambda name: 36.0 if name == "ego" else 32.0,
    )
    assert axis_ahead_of_lead(session, margin_m=3.0) is True
    assert axis_ahead_of_lead(session, margin_m=5.0) is False


def test_adjacent_lane_lead_not_too_close_when_beside():
    class _Loc:
        def __init__(self, x, y, lane_id=1, road_id=1):
            self.x, self.y, self.z = x, y, 0.0
            self._wp = SimpleNamespace(lane_id=lane_id, road_id=road_id)

    class _Actor:
        def __init__(self, x, y, lane_id=1):
            self._loc = _Loc(x, y, lane_id=lane_id)
            self.id = 1

        def get_location(self):
            return self._loc

        def get_velocity(self):
            return SimpleNamespace(x=0.0, y=0.0, z=0.0)

    class _Map:
        def get_waypoint(self, loc, project_to_road=True):
            return loc._wp

    s = SimpleNamespace(
        ready=True,
        carla=SimpleNamespace(LaneType=SimpleNamespace(Driving=1)),
        map=_Map(),
        _travel_wp=SimpleNamespace(lane_id=5, road_id=1),
        _pass_corridor_committed=True,
        _pass_peak_shift_m=3.0,
        expected_passing_lane_width_m=lambda: 3.5,
        lateral_shift_toward_passing_m=lambda ego: 3.0,
        signed_gap_from_ego=lambda name: 1.9 if name == "lead" else -20.0,
        actors={
            "ego": _Actor(0, 0, lane_id=4),
            "lead": _Actor(0, 3.5, lane_id=5),
            "rear": _Actor(-20, 0, lane_id=4),
        },
    )
    s.actors["ego"].get_location()._wp = _Loc(0, 0, lane_id=4)._wp
    s.actors["lead"].get_location()._wp = _Loc(0, 3.5, lane_id=5)._wp
    s.actors["rear"].get_location()._wp = _Loc(-20, 0, lane_id=4)._wp

    from perception.carla_geometry import actor_debug_record

    def _rec(session, name, ego_xyz):
        other = session.actors[name]
        loc = other.get_location()
        import math

        d = math.hypot(loc.x, loc.y)
        return {"status": "ok", "euclidean_from_ego_m": d}

    import perception.carla_validation as cv

    orig = cv.actor_debug_record
    cv.actor_debug_record = _rec
    try:
        issues = _validate_carla_actors(s)
    finally:
        cv.actor_debug_record = orig

    assert not any("lead_too_close" in x for x in issues)
