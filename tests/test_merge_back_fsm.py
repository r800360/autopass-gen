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
