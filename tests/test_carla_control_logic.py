from autopass.carla_tuning import critical_gap_m, safe_follow_m
from perception.carla_control import acc_speed_target, resolve_pass_phase, speed_to_throttle_brake


def test_acc_brakes_at_critical_gap():
    t, b = speed_to_throttle_brake(12.0, 10.0, front_gap_m=critical_gap_m() - 0.5, steer_abs=0.0)
    assert t == 0.0
    assert b > 0.4


def test_pass_waits_for_lateral_room():
    assert resolve_pass_phase("pass", ego_lane=0, clear_of_lead=False, front_gap_m=10.0) == "approach"
    assert resolve_pass_phase("pass", ego_lane=0, clear_of_lead=False, front_gap_m=20.0) == "lane_change"


def test_merge_only_when_clear():
    assert resolve_pass_phase("pass", ego_lane=1, clear_of_lead=True, front_gap_m=5.0) == "merge"


def test_acc_slows_in_travel_lane_when_close():
    target = acc_speed_target(
        action="wait",
        target_speed_mps=12.0,
        current_speed_mps=12.0,
        front_gap_m=safe_follow_m() - 4.0,
        lead_speed_mps=5.0,
        speed_limit_mps=13.4,
        phase="cruise",
    )
    assert target < 8.0


def test_overtake_ignores_lateral_3d_gap_for_acc():
    target = acc_speed_target(
        action="pass",
        target_speed_mps=8.0,
        current_speed_mps=7.0,
        front_gap_m=critical_gap_m() - 1.0,
        lead_speed_mps=5.0,
        speed_limit_mps=13.4,
        phase="overtake",
        longitudinal_lead_gap_m=5.0,
    )
    assert target >= 10.0


def test_lane_change_skips_critical_gap_hard_brake():
    t, b = speed_to_throttle_brake(
        12.0,
        10.0,
        front_gap_m=critical_gap_m() - 0.5,
        steer_abs=0.0,
        phase="lane_change",
    )
    assert t > 0.0
    assert b < 0.4


def test_overtake_skips_critical_gap_hard_brake():
    t, b = speed_to_throttle_brake(
        12.0,
        10.0,
        front_gap_m=critical_gap_m() - 0.5,
        steer_abs=0.0,
        phase="overtake",
    )
    assert t > 0.0
    assert b < 0.4


def test_wait_follow_uses_cruise_semantics_and_bounded_controls(monkeypatch):
    from types import SimpleNamespace
    from perception.carla_control import build_vehicle_control
    from visual_world import curated_demo_scenarios, initialize_world

    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)

    class _Ego:
        def get_transform(self):
            return SimpleNamespace(
                location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                rotation=SimpleNamespace(yaw=0.0),
            )

        def get_velocity(self):
            return SimpleNamespace(x=0.0, y=0.0, z=0.0)

    wp = SimpleNamespace(
        transform=SimpleNamespace(location=SimpleNamespace(x=12.0, y=0.0, z=0.0)),
        lane_id=1,
        road_id=1,
    )
    session = SimpleNamespace(
        map=SimpleNamespace(),
        _last_steer=0.0,
        _last_control_debug={},
        actors={"lead": None},
        lead_longitudinal_gap_m=lambda: 20.0,
        get_steering_waypoint=lambda ego, phase, passing_side: wp,
    )
    ego = _Ego()
    ctrl = build_vehicle_control(
        "wait",
        world=world,
        spec=spec,
        target_speed_mps=0.0,
        session=session,
        ego=ego,
        measured_speed_mps=5.0,
        front_gap_m=25.0,
        ego_lane=0,
    )
    dbg = session._last_control_debug
    assert dbg["action_semantic"] == "follow_lead"
    assert dbg["phase"] == "cruise"
    assert -0.25 <= ctrl.steer <= 0.25
    assert 0.0 <= ctrl.throttle <= 0.72
    assert 0.0 <= ctrl.brake <= 0.9
