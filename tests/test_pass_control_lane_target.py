"""Pass control lane-target invariants (mock session, no CARLA server)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from perception.pass_control_fsm import (
    PassControlState,
    begin_pass,
    get_pass_control_state,
    pass_control_tick,
    resolve_target_lane_ids,
)


def _mock_session(*, travel_lane=5, passing_lane=4, road=1):
    tw = SimpleNamespace(lane_id=travel_lane, road_id=road, transform=SimpleNamespace(location=SimpleNamespace(x=0, y=0, z=0)))
    pw = SimpleNamespace(lane_id=passing_lane, road_id=road, transform=SimpleNamespace(location=SimpleNamespace(x=0, y=3.5, z=0)))
    ego_loc = SimpleNamespace(x=0.0, y=0.0, z=0.0)

    class _Ego:
        def get_location(self):
            return ego_loc

        def get_transform(self):
            return SimpleNamespace(location=ego_loc, rotation=SimpleNamespace(yaw=0.0))

        def get_velocity(self):
            return SimpleNamespace(x=8.0, y=0.0, z=0.0)

    session = SimpleNamespace(
        _travel_wp=tw,
        _passing_wp=pw,
        _passing_side="left",
        map=SimpleNamespace(),
        actors={"ego": _Ego()},
        expected_passing_lane_width_m=lambda: 3.5,
        lateral_shift_toward_passing_m=lambda ego: 0.0,
        ego_on_passing_lane=lambda ego: False,
        _travel_lane_anchor_at_ego=lambda ego: tw,
        _adjacent_passing_lane_wp=lambda ego_wp, side: pw,
    )
    return session


def test_wait_target_lane_is_travel_lane():
    session = _mock_session()
    ids = resolve_target_lane_ids(session, "idle")
    assert ids["target_lane_id"] == 5
    assert ids["target_lane_source"] == "travel_lane"


def test_pass_lane_change_targets_passing_lane_not_travel():
    session = _mock_session(travel_lane=5, passing_lane=4)
    ids = resolve_target_lane_ids(session, "lane_change")
    assert ids["target_lane_id"] == 4
    assert ids["passing_lane_id"] == 4
    assert ids["travel_lane_id"] == 5
    assert ids["target_lane_source"] == "passing_lane"


def test_pass_control_tick_wait_during_early_pass_aborts():
    session = _mock_session()
    begin_pass(session)
    st, action, fail = pass_control_tick(
        session,
        session.actors["ego"],
        requested_action="wait",
        pass_in_progress=True,
        front_gap_m=22.0,
        clear_of_lead=False,
        speed_mps=8.0,
    )
    assert action == "wait"
    assert st.abort_reason
    assert not st.active


def test_pass_control_tick_missing_passing_lane_fails():
    session = _mock_session()
    session._passing_wp = None
    st, action, fail = pass_control_tick(
        session,
        session.actors["ego"],
        requested_action="pass",
        pass_in_progress=False,
        front_gap_m=25.0,
        clear_of_lead=False,
        speed_mps=8.0,
    )
    assert action == "wait"
    assert fail is True


def test_wait_during_committed_pass_continues_pass_action():
    session = _mock_session()
    begin_pass(session)
    st = get_pass_control_state(session)
    st.maneuver_started = True
    st.phase = "overtake"
    st.active = True
    session._pass_control = st
    st2, action, fail = pass_control_tick(
        session,
        session.actors["ego"],
        requested_action="wait",
        pass_in_progress=True,
        front_gap_m=8.0,
        clear_of_lead=False,
        speed_mps=6.0,
    )
    assert action == "pass"
    assert st2.active
    assert not fail


def test_wait_when_pass_not_committed_resets_fsm():
    session = _mock_session()
    begin_pass(session)
    st = get_pass_control_state(session)
    st.maneuver_started = True
    st.phase = "overtake"
    st.active = True
    session._pass_control = st
    st2, action, fail = pass_control_tick(
        session,
        session.actors["ego"],
        requested_action="wait",
        pass_in_progress=False,
        front_gap_m=10.0,
        clear_of_lead=False,
        speed_mps=5.0,
    )
    assert action == "wait"
    assert not st2.active
    assert not fail


def test_merge_back_targets_travel_lane():
    session = _mock_session(travel_lane=5, passing_lane=4)
    ids = resolve_target_lane_ids(session, "merge_back")
    assert ids["target_lane_id"] == 5
    assert ids["target_lane_source"] == "return_lane"
