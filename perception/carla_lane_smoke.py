#!/usr/bin/env python3
"""CARLA lane-keeping smoke: ego follows travel lane with no_pass/wait control."""
from __future__ import annotations

import argparse
import os
import sys


def run_lane_smoke(*, duration_s: float = 7.0, fail_threshold_m: float = 1.5) -> int:
    from autopass.carla_tuning import lane_departure_warn_m
    from perception.carla_control import _speed_mps, build_vehicle_control, resolve_pass_phase
    from perception.carla_lane_keep import heading_error_deg, lateral_error_m
    from perception.carla_scenario import bootstrap_carla_scenario, get_session
    from visual_world import curated_demo_scenarios, initialize_world

    os.environ.setdefault("AUTOPASS_CARLA_CURATED_CORRIDOR", "1")
    os.environ.setdefault("AUTOPASS_ENVIRONMENT", "highway")

    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    if not bootstrap_carla_scenario(spec, world, map_name="Town04"):
        session = get_session()
        print(f"FAIL: bootstrap failed: {session.last_error}", flush=True)
        return 1

    session = get_session()
    session.enable_ego_physics(True)
    ego = session.actors.get("ego")
    if ego is None:
        print("FAIL: ego missing after bootstrap", flush=True)
        return 1

    delta = session.fixed_delta_seconds
    n_ticks = max(1, int(round(duration_s / delta)))
    max_lane = 0.0
    fail_step = -1

    print(
        f"3) lane follow {duration_s:.1f}s ({n_ticks} ticks), fail if center > {fail_threshold_m}m ...",
        flush=True,
    )
    print(
        "step speed thr brk steer ego(x,y) yaw tgt(x,y) tgt_yaw lat_err head_err lane_center",
        flush=True,
    )

    for step in range(n_ticks):
        session.update_route_cursor(ego)
        session.tick_npcs_kinematic(spec, delta)
        gaps = session.measure_actor_gaps_3d()
        front = gaps.get("front", 999.0)
        clear_of_lead = session.ego_clear_of_lead()
        ego_lane = session.infer_ego_lane_index()
        v = _speed_mps(ego.get_velocity())
        lane_dist = session.ego_lane_center_distance_m(ego)
        max_lane = max(max_lane, lane_dist)
        if lane_dist > fail_threshold_m and fail_step < 0:
            fail_step = step

        phase = resolve_pass_phase("wait", ego_lane=ego_lane, clear_of_lead=clear_of_lead, front_gap_m=front)
        target_wp = session.get_steering_waypoint(ego, phase, "left")
        ego_tf = ego.get_transform()
        ego_loc = ego_tf.location
        tgt_loc = target_wp.transform.location if target_wp else ego_loc
        tgt_yaw = target_wp.transform.rotation.yaw if target_wp else ego_tf.rotation.yaw
        lat = lateral_error_m(ego_loc, ego_tf.rotation.yaw, tgt_loc)
        head = heading_error_deg(ego_tf.rotation.yaw, tgt_loc, ego_loc)

        ctrl = build_vehicle_control(
            "wait",
            world=world,
            spec=spec,
            target_speed_mps=spec.route.speed_limit_mps * 0.85,
            passing_side="left",
            session=session,
            ego=ego,
            measured_speed_mps=v,
            front_gap_m=front,
            clear_of_lead=clear_of_lead,
            ego_lane=ego_lane,
            recovery=lane_dist > lane_departure_warn_m(),
        )
        ego.apply_control(ctrl)
        session.tick()

        cursor_dbg = session.route_cursor_debug_snapshot(ego)
        print(
            f"{step:3d} {v:5.2f} {ctrl.throttle:4.2f} {ctrl.brake:4.2f} {ctrl.steer:+.3f} "
            f"({ego_loc.x:7.1f},{ego_loc.y:7.1f}) {ego_tf.rotation.yaw:6.1f} "
            f"({tgt_loc.x:7.1f},{tgt_loc.y:7.1f}) {tgt_yaw:6.1f} "
            f"{lat:+5.2f} {head:+5.1f} {lane_dist:5.2f} "
            f"cursor_lane={cursor_dbg.get('route_cursor_lane_id')} "
            f"cursor_dist={cursor_dbg.get('cursor_dist_from_ego_m')}",
            flush=True,
        )

    session.shutdown()
    if max_lane > fail_threshold_m:
        print(
            f"FAIL: max lane center distance {max_lane:.2f}m at step>={fail_step} "
            f"(threshold {fail_threshold_m}m)",
            flush=True,
        )
        return 1
    print(f"PASS: max lane center distance {max_lane:.2f}m", flush=True)
    return 0


def main(argv=None) -> int:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    os.environ.setdefault("AUTOPASS_PERCEPTION_BACKEND", "carla")

    parser = argparse.ArgumentParser(description="CARLA lane-keeping smoke test")
    parser.add_argument("--duration", type=float, default=7.0, help="Follow duration in seconds")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Fail if lane center distance exceeds this (meters)",
    )
    args = parser.parse_args(argv)

    print("1) import carla ...", flush=True)
    try:
        import carla  # noqa: F401
    except ImportError as e:
        print(f"   FAIL: {e}", flush=True)
        return 1
    print("   OK", flush=True)

    print("2) bootstrap Town04 highway scenario ...", flush=True)
    return run_lane_smoke(duration_s=args.duration, fail_threshold_m=args.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
