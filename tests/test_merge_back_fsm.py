"""Merge-back FSM must not trigger before longitudinal clearance."""
from __future__ import annotations

from types import SimpleNamespace

from perception.pass_control_fsm import advance_pass_fsm, get_pass_control_state


def _ego():
    return SimpleNamespace(
        get_location=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
        get_transform=lambda: SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )


def test_overtake_does_not_merge_back_without_clearance():
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
        _passing_side="left",
        expected_passing_lane_width_m=lambda: 3.5,
        lateral_shift_toward_passing_m=lambda ego: 2.0,
        _travel_lane_anchor_at_ego=lambda ego: tw,
        _adjacent_passing_lane_wp=lambda ego_wp, side: pw,
        ego_cleared_lead=lambda c: False,
        ego_clear_of_lead=lambda c: False,
        lead_longitudinal_gap_m=lambda: 18.0,
    )
    ego = _ego()
    st = get_pass_control_state(session)
    st.active = True
    st.phase = "overtake"
    st.maneuver_started = True
    st.ticks_in_phase = 5
    st.passing_lane_id = 4

    session.lateral_lane_offsets_m = lambda ego: (0.5, 1.0, 3.5)

    st = advance_pass_fsm(
        session, ego, front_gap_m=18.0, clear_of_lead=False, speed_mps=8.0
    )
    assert st.phase == "overtake"


def test_overtake_slip_to_travel_recommits_lane_change():
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
        _passing_side="left",
        expected_passing_lane_width_m=lambda: 3.5,
        lateral_shift_toward_passing_m=lambda ego: 0.1,
        _travel_lane_anchor_at_ego=lambda ego: tw,
        _passing_lane_anchor_at_ego=lambda ego: pw,
        _adjacent_passing_lane_wp=lambda ego_wp, side: pw,
        ego_cleared_lead=lambda c: False,
        ego_clear_of_lead=lambda c: False,
        lead_longitudinal_gap_m=lambda: 12.0,
    )
    ego = _ego()
    st = get_pass_control_state(session)
    st.active = True
    st.phase = "overtake"
    st.maneuver_started = True
    st.ticks_in_phase = 3
    session.lateral_lane_offsets_m = lambda ego: (0.4, 2.2, 3.5)

    st = advance_pass_fsm(
        session, ego, front_gap_m=12.0, clear_of_lead=False, speed_mps=6.0
    )
    assert st.phase == "lane_change"


def test_merge_back_holds_when_wide_after_axis_ahead():
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
    far_travel = SimpleNamespace(
        lane_id=5,
        road_id=1,
        transform=SimpleNamespace(location=SimpleNamespace(x=0.0, y=20.0, z=0.0)),
    )
    session = SimpleNamespace(
        map=None,
        ready=True,
        _travel_wp=tw,
        _passing_wp=pw,
        _passing_side="left",
        expected_passing_lane_width_m=lambda: 3.5,
        lateral_shift_toward_passing_m=lambda ego: 0.0,
        _travel_lane_anchor_at_ego=lambda ego: far_travel,
        _passing_lane_anchor_at_ego=lambda ego: pw,
        _adjacent_passing_lane_wp=lambda ego_wp, side: pw,
        ego_cleared_lead=lambda c: False,
        actor_travel_s=lambda name: 36.0 if name == "ego" else 32.0,
        _pass_corridor_committed=True,
        _pass_peak_shift_m=3.1,
        lateral_lane_offsets_m=lambda ego: (2.0, 1.5, 3.5),
        ego_corridor_lane_offset_m=lambda ego: 1.5,
    )
    ego = _ego()
    st = get_pass_control_state(session)
    st.active = True
    st.phase = "merge_back"
    st.maneuver_started = True
    st.ticks_in_phase = 4

    st = advance_pass_fsm(
        session, ego, front_gap_m=0.0, clear_of_lead=False, speed_mps=8.0
    )
    assert st.phase in ("merge_back", "idle")


def test_merge_back_recovers_to_overtake_when_runaway_wide():
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
        _passing_side="left",
        expected_passing_lane_width_m=lambda: 3.5,
        lateral_shift_toward_passing_m=lambda ego: 0.0,
        _travel_lane_anchor_at_ego=lambda ego: tw,
        _passing_lane_anchor_at_ego=lambda ego: pw,
        _adjacent_passing_lane_wp=lambda ego_wp, side: pw,
        ego_cleared_lead=lambda c: False,
        actor_travel_s=lambda name: 36.0 if name == "ego" else 32.0,
        _pass_corridor_committed=True,
        _pass_peak_shift_m=3.1,
        lateral_lane_offsets_m=lambda ego: (19.8, 19.5, 3.5),
        ego_corridor_lane_offset_m=lambda ego: 19.5,
    )
    ego = _ego()
    st = get_pass_control_state(session)
    st.active = True
    st.phase = "merge_back"
    st.maneuver_started = True
    st.ticks_in_phase = 4

    st = advance_pass_fsm(
        session, ego, front_gap_m=0.0, clear_of_lead=False, speed_mps=8.0
    )
    assert st.phase == "overtake"


def test_overtake_merge_back_when_axis_ahead():
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
        _passing_side="left",
        expected_passing_lane_width_m=lambda: 3.5,
        lateral_shift_toward_passing_m=lambda ego: 1.8,
        _travel_lane_anchor_at_ego=lambda ego: tw,
        _passing_lane_anchor_at_ego=lambda ego: pw,
        _adjacent_passing_lane_wp=lambda ego_wp, side: pw,
        ego_cleared_lead=lambda c: False,
        actor_travel_s=lambda name: 36.0 if name == "ego" else 32.0,
        _pass_corridor_committed=True,
        _pass_peak_shift_m=3.1,
        lateral_lane_offsets_m=lambda ego: (1.2, 1.0, 3.5),
        ego_corridor_lane_offset_m=lambda ego: 1.0,
    )
    ego = _ego()
    st = get_pass_control_state(session)
    st.active = True
    st.phase = "overtake"
    st.maneuver_started = True
    st.ticks_in_phase = 4

    st = advance_pass_fsm(
        session, ego, front_gap_m=0.0, clear_of_lead=False, speed_mps=8.0
    )
    assert st.phase == "merge_back"


def test_overtake_merge_back_when_axis_ahead_but_wide():
    """Latched pass that drifted wide should recover via merge_back, not stay in overtake."""
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
        _passing_side="left",
        expected_passing_lane_width_m=lambda: 3.5,
        lateral_shift_toward_passing_m=lambda ego: 0.0,
        _travel_lane_anchor_at_ego=lambda ego: tw,
        _passing_lane_anchor_at_ego=lambda ego: pw,
        _adjacent_passing_lane_wp=lambda ego_wp, side: pw,
        ego_cleared_lead=lambda c: False,
        actor_travel_s=lambda name: 36.0 if name == "ego" else 32.0,
        _pass_corridor_committed=True,
        _pass_peak_shift_m=3.1,
        lateral_lane_offsets_m=lambda ego: (19.8, 19.5, 3.5),
        ego_corridor_lane_offset_m=lambda ego: 19.5,
    )
    ego = _ego()
    st = get_pass_control_state(session)
    st.active = True
    st.phase = "overtake"
    st.maneuver_started = True
    st.ticks_in_phase = 4

    st = advance_pass_fsm(
        session, ego, front_gap_m=0.0, clear_of_lead=False, speed_mps=8.0
    )
    assert st.phase == "merge_back"


def test_overtake_merge_back_when_long_cleared():
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
        _passing_side="left",
        expected_passing_lane_width_m=lambda: 3.5,
        lateral_shift_toward_passing_m=lambda ego: 1.8,
        _travel_lane_anchor_at_ego=lambda ego: tw,
        _passing_lane_anchor_at_ego=lambda ego: pw,
        _adjacent_passing_lane_wp=lambda ego_wp, side: pw,
        ego_cleared_lead=lambda c: True,
        ego_clear_of_lead=lambda c: True,
        lead_longitudinal_gap_m=lambda: 2.0,
    )
    ego = _ego()
    st = get_pass_control_state(session)
    st.active = True
    st.phase = "overtake"
    st.maneuver_started = True
    st.ticks_in_phase = 8
    session.lateral_lane_offsets_m = lambda ego: (2.0, 1.5, 3.5)

    st = advance_pass_fsm(
        session, ego, front_gap_m=20.0, clear_of_lead=True, speed_mps=8.0
    )
    assert st.phase == "merge_back"


def test_corridor_committed_stays_overtake_despite_slip():
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
        _passing_side="left",
        expected_passing_lane_width_m=lambda: 3.5,
        lateral_shift_toward_passing_m=lambda ego: 0.1,
        _travel_lane_anchor_at_ego=lambda ego: tw,
        _passing_lane_anchor_at_ego=lambda ego: pw,
        _adjacent_passing_lane_wp=lambda ego_wp, side: pw,
        ego_cleared_lead=lambda c: False,
        ego_clear_of_lead=lambda c: False,
        lead_longitudinal_gap_m=lambda: 12.0,
        _pass_corridor_committed=True,
        _pass_peak_shift_m=3.2,
        actor_travel_s=lambda name: 25.0,
        _pass_fsm_last_ego_s=24.0,
    )
    ego = _ego()
    st = get_pass_control_state(session)
    st.active = True
    st.phase = "overtake"
    st.maneuver_started = True
    st.ticks_in_phase = 8
    session.lateral_lane_offsets_m = lambda ego: (0.4, 2.2, 3.5)

    st = advance_pass_fsm(
        session, ego, front_gap_m=12.0, clear_of_lead=False, speed_mps=6.0
    )
    assert st.phase == "lane_change"


def test_overtake_latched_wide_departure_recommits_lane_change():
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
        _passing_side="left",
        expected_passing_lane_width_m=lambda: 3.5,
        lateral_shift_toward_passing_m=lambda ego: 0.0,
        _travel_lane_anchor_at_ego=lambda ego: tw,
        _passing_lane_anchor_at_ego=lambda ego: pw,
        _adjacent_passing_lane_wp=lambda ego_wp, side: pw,
        ego_cleared_lead=lambda c: False,
        actor_travel_s=lambda name: 23.0 if name == "ego" else 32.0,
        _pass_corridor_committed=True,
        _pass_peak_shift_m=2.9,
    )
    ego = _ego()
    st = get_pass_control_state(session)
    st.active = True
    st.phase = "overtake"
    st.maneuver_started = True
    st.ticks_in_phase = 3
    session.lateral_lane_offsets_m = lambda ego: (7.4, 7.0, 3.5)

    st = advance_pass_fsm(
        session, ego, front_gap_m=8.8, clear_of_lead=False, speed_mps=7.2
    )
    assert st.phase == "lane_change"


def test_overtake_corridor_departure_recommits_lane_change():
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
        _passing_side="left",
        expected_passing_lane_width_m=lambda: 3.5,
        lateral_shift_toward_passing_m=lambda ego: 0.0,
        _travel_lane_anchor_at_ego=lambda ego: tw,
        _passing_lane_anchor_at_ego=lambda ego: pw,
        _adjacent_passing_lane_wp=lambda ego_wp, side: pw,
        ego_cleared_lead=lambda c: False,
        actor_travel_s=lambda name: 22.5,
        _pass_fsm_last_ego_s=22.5,
    )
    ego = _ego()
    st = get_pass_control_state(session)
    st.active = True
    st.phase = "overtake"
    st.maneuver_started = True
    st.ticks_in_phase = 6
    session.lateral_lane_offsets_m = lambda ego: (6.5, 5.5, 3.5)

    st = advance_pass_fsm(
        session, ego, front_gap_m=10.0, clear_of_lead=False, speed_mps=5.7
    )
    assert st.phase == "lane_change"


def test_overtake_merge_back_when_abreast_on_passing():
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
        _passing_side="left",
        expected_passing_lane_width_m=lambda: 3.5,
        lateral_shift_toward_passing_m=lambda ego: 2.8,
        _travel_lane_anchor_at_ego=lambda ego: tw,
        _passing_lane_anchor_at_ego=lambda ego: pw,
        _adjacent_passing_lane_wp=lambda ego_wp, side: pw,
        ego_cleared_lead=lambda c: False,
        actor_travel_s=lambda name: 31.0 if name == "ego" else 32.0,
        _pass_corridor_committed=True,
        _pass_peak_shift_m=3.1,
        lateral_lane_offsets_m=lambda ego: (3.2, 0.35, 3.5),
        ego_corridor_lane_offset_m=lambda ego: 0.35,
    )
    ego = _ego()
    st = get_pass_control_state(session)
    st.active = True
    st.phase = "overtake"
    st.maneuver_started = True
    st.ticks_in_phase = 4

    st = advance_pass_fsm(
        session, ego, front_gap_m=2.0, clear_of_lead=False, speed_mps=6.5
    )
    assert st.phase == "merge_back"


def test_overtake_holds_passing_lane_while_latched():
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
        _passing_side="left",
        expected_passing_lane_width_m=lambda: 3.5,
        lateral_shift_toward_passing_m=lambda ego: 2.6,
        _travel_lane_anchor_at_ego=lambda ego: tw,
        _passing_lane_anchor_at_ego=lambda ego: pw,
        _adjacent_passing_lane_wp=lambda ego_wp, side: pw,
        ego_cleared_lead=lambda c: False,
        actor_travel_s=lambda name: 18.0 if name == "ego" else 32.0,
        _pass_corridor_committed=True,
        _pass_peak_shift_m=3.1,
        lateral_lane_offsets_m=lambda ego: (3.0, 0.88, 3.5),
    )
    ego = _ego()
    st = get_pass_control_state(session)
    st.active = True
    st.phase = "overtake"
    st.maneuver_started = True
    st.ticks_in_phase = 3

    st = advance_pass_fsm(
        session, ego, front_gap_m=14.0, clear_of_lead=False, speed_mps=5.0
    )
    assert st.phase == "overtake"
