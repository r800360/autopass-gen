#!/usr/bin/env python3
"""CARLA corridor smoke: scan, validate, spawn, and lane-follow a passing corridor."""
from __future__ import annotations

import argparse
import os
import sys


def _connect_map(map_name: str):
    from perception.carla_scenario import bootstrap_carla_scenario, get_session
    from visual_world import curated_demo_scenarios, initialize_world

    os.environ.setdefault("AUTOPASS_CARLA_CURATED_CORRIDOR", "1")
    os.environ.setdefault("AUTOPASS_ENVIRONMENT", "highway")
    os.environ["AUTOPASS_CARLA_MAP"] = map_name

    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    if not bootstrap_carla_scenario(spec, world, map_name=map_name):
        session = get_session()
        print(f"FAIL: bootstrap failed: {session.last_error}", flush=True)
        return None, None, None
    return get_session(), spec, world


def run_diagnose(*, map_name: str, top_k: int, validation_mode: str) -> int:
    from perception.carla_scenario import get_session

    if _connect_map(map_name)[0] is None:
        return 1
    session = get_session()
    print(f"3) corridor diagnostics map={map_name} mode={validation_mode}", flush=True)
    report_text = session.diagnose_passing_corridors(
        max_candidates=300,
        validation_mode=validation_mode,
        top_k=top_k,
    )
    print(report_text, flush=True)
    diag = getattr(session, "_last_corridor_diagnostics", None)
    if diag is not None and diag.valid_count == 0:
        print(
            "\nNOTE: No strict/presentation-valid corridors found. "
            "Try: python -m perception.carla_corridor_smoke --hero",
            flush=True,
        )
    session.shutdown()
    return 0


def run_corridor_smoke(
    *,
    duration_s: float = 7.0,
    lane_threshold_m: float = 1.5,
    map_name: str = "Town04",
    hero: bool = False,
    validation_mode: str | None = None,
) -> int:
    from autopass.carla_tuning import lane_departure_warn_m
    from perception.carla_control import _speed_mps, build_vehicle_control, resolve_pass_phase
    from perception.carla_corridor import HERO_CORRIDOR_WARNING, validate_passing_corridor
    from perception.carla_scenario import bootstrap_carla_scenario, get_session
    from visual_world import curated_demo_scenarios, initialize_world

    os.environ.setdefault("AUTOPASS_CARLA_CURATED_CORRIDOR", "1")
    os.environ.setdefault("AUTOPASS_ENVIRONMENT", "highway")
    os.environ["AUTOPASS_CARLA_MAP"] = map_name
    if hero:
        os.environ["AUTOPASS_CARLA_HERO_CORRIDOR"] = "1"
        os.environ["AUTOPASS_CARLA_CORRIDOR_MODE"] = "hero"
    elif validation_mode:
        os.environ["AUTOPASS_CARLA_CORRIDOR_MODE"] = validation_mode

    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    if not bootstrap_carla_scenario(spec, world, map_name=map_name):
        session = get_session()
        print(f"FAIL: bootstrap failed: {session.last_error}", flush=True)
        if session.last_error and "No curated passing corridor" in session.last_error:
            session.shutdown()
            return run_diagnose(map_name=map_name, top_k=5, validation_mode=validation_mode or "presentation")
        return 1

    session = get_session()
    report = session._corridor_report
    if report is None:
        print("FAIL: no corridor report after bootstrap", flush=True)
        session.shutdown()
        return 1

    if hero or session._corridor_hero_fallback:
        if not (report.hero_ok or report.ok or report.presentation_ok):
            print("FAIL: hero corridor validation failed", flush=True)
            session.shutdown()
            return 1
        if session._corridor_hero_fallback or hero:
            print(f"WARNING: {HERO_CORRIDOR_WARNING}", flush=True)
    elif not report.ok and not report.presentation_ok:
        print("FAIL: no curated corridor selected after bootstrap", flush=True)
        diag = session.diagnose_passing_corridors(top_k=5)
        print(diag, flush=True)
        session.shutdown()
        return 1

    print("3) corridor scan summary", flush=True)
    mode = validation_mode or ("hero" if hero else "presentation")
    candidates = session.scan_passing_corridors(max_candidates=300, validation_mode=mode)
    print(f"   valid_candidates={len(candidates)} map={session._map_name} mode={mode}", flush=True)
    print(f"   chosen: {report.summary_line()}", flush=True)

    vmode = "hero" if hero or session._corridor_hero_fallback else "presentation"
    live = validate_passing_corridor(
        session._travel_wp,
        carla=session.carla,
        world=session.world,
        validation_mode=vmode,  # type: ignore[arg-type]
        find_passing_lane=session._find_passing_lane_wp,
        find_opposing_lane=session._find_opposing_lane_wp,
    )
    if not live.ok and not (hero or session._corridor_hero_fallback) and not live.presentation_ok:
        print(f"FAIL: active corridor invalid: {', '.join(live.issues)}", flush=True)
        session.shutdown()
        return 1
    if hero and not live.ok:
        print(f"   hero corridor issues (non-fatal): {', '.join(live.issues[:4])}", flush=True)

    session.enable_ego_physics(True)
    ego = session.actors.get("ego")
    if ego is None:
        print("FAIL: ego missing", flush=True)
        session.shutdown()
        return 1

    delta = session.fixed_delta_seconds
    n_ticks = max(1, int(round(duration_s / delta)))
    max_lane = 0.0
    junction_hits = 0

    print(f"4) lane-follow {duration_s:.1f}s on curated corridor ...", flush=True)
    for _step in range(n_ticks):
        session.update_route_cursor(ego)
        session.tick_npcs_kinematic(spec, delta)
        gaps = session.measure_actor_gaps_3d()
        front = gaps.get("front", 999.0)
        clear_of_lead = session.ego_clear_of_lead()
        ego_lane = session.infer_ego_lane_index()
        v = _speed_mps(ego.get_velocity())
        lane_dist = session.ego_lane_center_distance_m(ego)
        max_lane = max(max_lane, lane_dist)

        try:
            ego_wp = session.map.get_waypoint(ego.get_location(), project_to_road=True)
            if getattr(ego_wp, "is_junction", False):
                junction_hits += 1
        except Exception:
            pass

        phase = resolve_pass_phase("wait", ego_lane=ego_lane, clear_of_lead=clear_of_lead, front_gap_m=front)
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

    hero_used = hero or session._corridor_hero_fallback
    session.shutdown()

    if junction_hits > 0:
        print(f"FAIL: ego entered junction {junction_hits} tick(s)", flush=True)
        return 1
    if not hero and live.junction_count_in_horizon > 0:
        print(
            f"FAIL: corridor has junctions in maneuver horizon={live.junction_count_in_horizon} "
            f"lights={live.traffic_light_count} stops={live.stop_control_count}",
            flush=True,
        )
        return 1
    if max_lane > lane_threshold_m:
        print(f"FAIL: max lane center distance {max_lane:.2f}m > {lane_threshold_m}m", flush=True)
        return 1

    label = "hero" if hero_used else "curated"
    print(
        f"PASS: {label} corridor ok, max_lane_center={max_lane:.2f}m, "
        f"fwd={report.forward_length_m:.0f}m back={report.backward_length_m:.0f}m",
        flush=True,
    )
    return 0


def main(argv=None) -> int:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    os.environ.setdefault("AUTOPASS_PERCEPTION_BACKEND", "carla")

    parser = argparse.ArgumentParser(description="CARLA curated passing-corridor smoke test")
    parser.add_argument("--duration", type=float, default=7.0)
    parser.add_argument("--lane-threshold", type=float, default=1.5)
    parser.add_argument("--map", type=str, default=os.environ.get("AUTOPASS_CARLA_MAP", "Town04"))
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print rejection diagnostics without failing immediately",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Near-miss candidates to show with --diagnose")
    parser.add_argument(
        "--hero",
        action="store_true",
        help="Use hero corridor mode (maneuver-horizon validation for final video)",
    )
    parser.add_argument(
        "--curated-corridor",
        action="store_true",
        help="Require presentation-mode curated corridor (default for smoke)",
    )
    parser.add_argument(
        "--mode",
        choices=["strict", "presentation", "hero"],
        default=None,
        help="Validation mode override",
    )
    args = parser.parse_args(argv)

    print("1) import carla ...", flush=True)
    try:
        import carla  # noqa: F401
    except ImportError as e:
        print(f"   FAIL: {e}", flush=True)
        return 1
    print("   OK", flush=True)

    mode = args.mode or ("hero" if args.hero else "presentation")
    if args.curated_corridor:
        os.environ["AUTOPASS_CARLA_CURATED_CORRIDOR"] = "1"
        os.environ.setdefault("AUTOPASS_CARLA_CORRIDOR_MODE", "presentation")

    print(f"2) bootstrap {args.map} with corridor mode={mode} ...", flush=True)

    if args.diagnose:
        from perception.carla_scenario import get_session

        _connect_map(args.map)
        session = get_session()
        print(session.diagnose_passing_corridors(max_candidates=300, validation_mode=mode, top_k=args.top_k), flush=True)
        session.shutdown()
        return 0

    return run_corridor_smoke(
        duration_s=args.duration,
        lane_threshold_m=args.lane_threshold,
        map_name=args.map,
        hero=args.hero,
        validation_mode=mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
