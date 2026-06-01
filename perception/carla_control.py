"""
Low-level CARLA execution: ACC + phased lane change + kinematic NPCs.

The agent chooses *when* to pass; this module decides *how* to actuate safely
between graph steps (gap-based braking, no merge-back until clear of lead).
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, Literal, Tuple

from autopass.carla_tuning import (
    critical_gap_m,
    lane_change_lateral_mult,
    lane_change_steer_cap_mult,
    lane_departure_fail_m,
    lane_departure_warn_m,
    lateral_steer_gain,
    max_steer,
    merge_clear_m,
    near_miss_m,
    pass_lateral_min_m,
    route_lookahead_m,
    safe_follow_m,
    steer_gain,
    steer_smooth,
)
from visual_world import ScenarioSpec, WorldState, advance_world_step

PassPhase = Literal["cruise", "approach", "lane_change", "overtake", "merge"]


def _session_cleared_of_lead(session) -> bool:
    from perception.carla_pass_maneuver import merge_clearance_m

    clearance = merge_clearance_m()
    if hasattr(session, "ego_cleared_lead"):
        return bool(session.ego_cleared_lead(clearance))
    return bool(session.ego_clear_of_lead(clearance))


def _speed_mps(velocity) -> float:
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)


def _near_miss_this_step(
    session,
    min_front: float,
    *,
    collision: bool,
    pass_st,
    ego_lane: int,
    requested_action: str,
) -> bool:
    """Near-miss uses longitudinal corridor gap during lateral pass (not Euclidean)."""
    if collision:
        return False
    threshold = near_miss_m()
    if requested_action == "pass" or getattr(pass_st, "active", False):
        lead = session.actors.get("lead") if getattr(session, "ready", False) else None
        if lead is not None and hasattr(session, "_longitudinal_along_travel"):
            along = session._longitudinal_along_travel(lead)
            if along is not None:
                if along <= 0:
                    return False
                if ego_lane == 1 or getattr(pass_st, "phase", "") in ("lane_change", "overtake", "merge_back"):
                    return along < threshold
    return min_front < threshold


def resolve_pass_phase(
    action: str,
    *,
    ego_lane: int,
    clear_of_lead: bool,
    front_gap_m: float,
    pass_fsm_phase: str | None = None,
) -> PassPhase:
    if action != "pass":
        return "cruise"
    if pass_fsm_phase == "merge_back":
        return "merge"
    if pass_fsm_phase == "overtake":
        return "overtake"
    if pass_fsm_phase in ("prepare_pass", "lane_change"):
        return "lane_change"
    if pass_fsm_phase == "abort":
        return "cruise"
    if clear_of_lead:
        return "merge"
    if ego_lane == 1:
        return "overtake"
    if front_gap_m < pass_lateral_min_m():
        return "lane_change"
    return "lane_change"


def acc_speed_target(
    *,
    action: str,
    target_speed_mps: float,
    current_speed_mps: float,
    front_gap_m: float,
    lead_speed_mps: float,
    speed_limit_mps: float,
    phase: PassPhase,
    longitudinal_lead_gap_m: float | None = None,
    pass_fsm_phase: str | None = None,
    pass_maneuver_started: bool = False,
) -> float:
    """Adaptive cruise: cap speed when closing on lead in travel lane."""
    target = target_speed_mps if target_speed_mps > 0 else current_speed_mps

    if action == "pass":
        along = (
            longitudinal_lead_gap_m
            if longitudinal_lead_gap_m is not None
            else front_gap_m
        )
        if pass_fsm_phase == "prepare_pass" and not pass_maneuver_started:
            target = min(target, lead_speed_mps + 1.0, speed_limit_mps * 0.45, max(4.5, current_speed_mps))
        elif pass_fsm_phase == "lane_change" and not pass_maneuver_started:
            target = min(target, max(lead_speed_mps + 1.5, 5.0), speed_limit_mps * 0.52, max(5.0, current_speed_mps * 0.95))
        else:
            target = min(speed_limit_mps + 1.5, max(target, current_speed_mps + 1.0))
        if pass_fsm_phase in ("prepare_pass", "lane_change", "overtake") and along < pass_lateral_min_m():
            target = min(
                target,
                max(lead_speed_mps + 1.5, 4.5),
                speed_limit_mps * 0.58,
                max(4.5, current_speed_mps * 0.82),
            )
    elif action == "replan":
        target = max(4.0, lead_speed_mps)
    else:
        target = min(current_speed_mps, speed_limit_mps)
        if front_gap_m < safe_follow_m() + 6.0:
            target = min(target, lead_speed_mps + 0.5)

    # Beside lead during overtake only — do not surge speed while still in travel lane.
    if phase == "overtake" and pass_maneuver_started:
        along = (
            longitudinal_lead_gap_m
            if longitudinal_lead_gap_m is not None
            else front_gap_m
        )
        if along > 0.0 and along < 8.0:
            target = max(target, lead_speed_mps + 5.0, speed_limit_mps * 0.75)
        return max(0.0, target)

    if phase in ("cruise", "approach", "merge") and front_gap_m < safe_follow_m():
        # Proportional slowdown as gap closes
        gap_err = front_gap_m - safe_follow_m()
        target = min(target, lead_speed_mps + max(-1.0, gap_err * 0.35))

    if front_gap_m < critical_gap_m():
        target = min(target, max(0.0, lead_speed_mps - 1.5))

    return max(0.0, target)


def speed_to_throttle_brake(
    target_speed_mps: float,
    current_speed_mps: float,
    *,
    front_gap_m: float,
    steer_abs: float,
    no_steer_penalty: bool = False,
    phase: PassPhase | None = None,
) -> Tuple[float, float]:
    if phase not in ("overtake", "lane_change") and front_gap_m < critical_gap_m():
        return 0.0, min(1.0, 0.45 + (critical_gap_m() - front_gap_m) / 4.0)

    err = target_speed_mps - current_speed_mps
    if err > 0.8:
        throttle = min(0.65, 0.18 + err / 12.0)
        brake = 0.0
    elif err < -0.6:
        throttle = 0.0
        brake = min(0.85, 0.15 + abs(err) / 5.0)
    else:
        throttle = 0.22
        brake = 0.0

    if steer_abs > 0.1 and not no_steer_penalty:
        throttle *= 0.8
    return throttle, brake


def _target_driving_waypoint(
    session,
    ego,
    phase: PassPhase,
    passing_side: str,
    *,
    action_semantic: str = "",
):
    if action_semantic == "follow_lead" and hasattr(session, "get_travel_steering_waypoint"):
        return session.get_travel_steering_waypoint(ego)
    if hasattr(session, "get_steering_waypoint"):
        steer_phase = "cruise" if action_semantic == "follow_lead" else phase
        return session.get_steering_waypoint(ego, steer_phase, passing_side)
    carla = session.carla
    wp = session.map.get_waypoint(
        ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
    )
    return wp


def _steer_to_waypoint(
    session,
    ego,
    target_wp,
    *,
    recovery: bool = False,
    max_steer_override: float | None = None,
    lateral_gain_mult: float = 1.0,
    lookahead_mult: float = 1.0,
    smooth_mult: float = 1.0,
    max_delta: float = 0.05,
) -> Tuple[float, float, float]:
    """Pure-pursuit / Stanley-style steer toward target waypoint."""
    from perception.carla_lane_keep import pure_pursuit_steer

    if target_wp is None:
        return 0.0, 0.0, 0.0
    prev = getattr(session, "_last_steer", 0.0)
    ego_tf = ego.get_transform()
    target_loc = target_wp.transform.location
    base_la = route_lookahead_m() * (0.55 if recovery else 1.0)
    lookahead = base_la * lookahead_mult
    ms = max_steer_override if max_steer_override is not None else max_steer()
    steer, head, lat = pure_pursuit_steer(
        ego_tf.location,
        ego_tf.rotation.yaw,
        target_loc,
        lookahead_m=lookahead,
        max_steer=ms,
        steer_gain=steer_gain(),
        lateral_gain=lateral_steer_gain() * (1.4 if recovery else 1.0) * lateral_gain_mult,
        prev_steer=prev,
        smooth=steer_smooth() * smooth_mult,
    )
    delta = steer - prev
    if delta > max_delta:
        steer = prev + max_delta
    elif delta < -max_delta:
        steer = prev - max_delta
    session._last_steer = steer
    return steer, head, lat


def build_vehicle_control(
    action: str,
    *,
    world: WorldState,
    spec: ScenarioSpec,
    target_speed_mps: float = 0.0,
    passing_side: str = "left",
    session=None,
    ego=None,
    measured_speed_mps: float | None = None,
    front_gap_m: float | None = None,
    clear_of_lead: bool = False,
    ego_lane: int | None = None,
    recovery: bool = False,
    scripted_phase: str | None = None,
    pass_fsm_phase: str | None = None,
    pass_maneuver_started: bool = False,
) -> Any:
    import carla

    ctrl = carla.VehicleControl()
    ctrl.hand_brake = False
    ctrl.manual_gear_shift = False

    action_semantic = "follow_lead" if action in ("wait", "follow_lead") else action
    current = measured_speed_mps if measured_speed_mps is not None else world.ego_speed_mps
    front = front_gap_m if front_gap_m is not None else max(0.0, world.lead_x_m - world.ego_x_m)
    lead_along = None
    if session is not None and hasattr(session, "lead_longitudinal_gap_m"):
        lead_along = session.lead_longitudinal_gap_m()
    lane = ego_lane if ego_lane is not None else world.ego_lane
    phase = (
        "cruise"
        if action_semantic == "follow_lead"
        else resolve_pass_phase(
            action,
            ego_lane=lane,
            clear_of_lead=clear_of_lead,
            front_gap_m=front,
            pass_fsm_phase=pass_fsm_phase,
        )
    )
    if scripted_phase == "merge_back" and not clear_of_lead:
        phase = "overtake"
    if pass_fsm_phase == "prepare_pass" and not pass_maneuver_started:
        phase = "lane_change"

    lead_speed_for_acc = world.ego_speed_mps
    if session is not None and hasattr(session, "actors"):
        lead_actor = session.actors.get("lead")
        if lead_actor is not None:
            try:
                lead_speed_for_acc = _speed_mps(lead_actor.get_velocity())
            except Exception:
                pass

    target = acc_speed_target(
        action=action,
        target_speed_mps=target_speed_mps,
        current_speed_mps=current,
        front_gap_m=front,
        lead_speed_mps=lead_speed_for_acc,
        speed_limit_mps=spec.route.speed_limit_mps,
        phase=phase,
        longitudinal_lead_gap_m=lead_along,
        pass_fsm_phase=pass_fsm_phase,
        pass_maneuver_started=pass_maneuver_started,
    )

    if recovery:
        target = min(target, max(2.5, current * 0.65))

    fsm_scripted = None
    if pass_fsm_phase in ("prepare_pass", "lane_change"):
        fsm_scripted = "lane_change"
    elif pass_fsm_phase == "overtake":
        fsm_scripted = "overtake"
    elif pass_fsm_phase == "merge_back":
        fsm_scripted = "merge_back"
    if scripted_phase is None and fsm_scripted is not None:
        scripted_phase = fsm_scripted

    steer_kw: dict = {}
    no_steer_penalty = False
    if scripted_phase == "lane_change":
        cap_mult = lane_change_steer_cap_mult()
        shift = 0.0
        width = 3.5
        if session is not None and ego is not None:
            if hasattr(session, "lateral_shift_toward_passing_m"):
                shift = float(session.lateral_shift_toward_passing_m(ego))
            if hasattr(session, "expected_passing_lane_width_m"):
                width = float(session.expected_passing_lane_width_m())
        progress = min(1.0, max(0.0, shift / max(2.5, width)))
        # Gradual lateral commit — avoid overshooting into oncoming shoulder.
        target = min(target, 5.5 + 2.5 * progress)
        steer_kw = {
            "max_steer_override": min(0.22, max_steer() * min(cap_mult, 1.35)),
            "lateral_gain_mult": 0.85 + 0.35 * progress,
            "lookahead_mult": 0.45 + 0.15 * progress,
            "smooth_mult": 0.55,
            "max_delta": 0.035,
        }
        no_steer_penalty = True
    elif scripted_phase == "merge_back":
        if not clear_of_lead:
            scripted_phase = "overtake"
            target = min(target, max(lead_speed_for_acc + 2.0, 5.0))
            steer_kw = {"lateral_gain_mult": 0.95, "lookahead_mult": 0.7, "max_delta": 0.03}
        else:
            target = max(target, 5.0)
            steer_kw = {
                "max_steer_override": min(0.28, max_steer() * 1.35),
                "lateral_gain_mult": 1.6,
                "lookahead_mult": 0.45,
                "smooth_mult": 0.65,
            }
            no_steer_penalty = True
    elif scripted_phase == "overtake":
        if not clear_of_lead:
            target = min(target, max(lead_speed_for_acc + 3.0, 6.0), spec.route.speed_limit_mps * 0.75)
        else:
            target = max(target, spec.route.speed_limit_mps * 0.78)
        steer_kw = {"lateral_gain_mult": 0.95, "lookahead_mult": 0.72, "max_delta": 0.035}

    head_err = 0.0
    lat_err = 0.0
    target_lane_id = None
    travel_lane_id = None
    if session is not None and ego is not None and session.map is not None:
        target_wp = _target_driving_waypoint(
            session, ego, phase, passing_side, action_semantic=action_semantic
        )
        if recovery and hasattr(session, "get_recovery_travel_waypoint"):
            target_wp = session.get_recovery_travel_waypoint(ego) or target_wp
        if action_semantic == "follow_lead" and session._travel_wp is not None and target_wp is not None:
            tw = session._travel_wp
            travel_lane_id = int(tw.lane_id)
            target_lane_id = int(getattr(target_wp, "lane_id", travel_lane_id))
            if target_lane_id != travel_lane_id:
                target_wp = session.get_travel_steering_waypoint(ego)
                target_lane_id = travel_lane_id
            steer_kw.setdefault("lookahead_mult", 0.85)
            steer_kw.setdefault("lateral_gain_mult", 0.85)
            steer_kw.setdefault("max_delta", 0.03)
            steer_kw.setdefault("max_steer_override", min(0.14, max_steer()))
        steer, head_err, lat_err = _steer_to_waypoint(
            session, ego, target_wp, recovery=recovery, **steer_kw
        )
        unstable_heading = abs(head_err) > 35.0 or abs(lat_err) > 2.8
        if unstable_heading and session is not None and hasattr(session, "get_recovery_travel_waypoint"):
            recovery_wp = session.get_recovery_travel_waypoint(ego)
            if recovery_wp is not None:
                steer, head_err, lat_err = _steer_to_waypoint(
                    session,
                    ego,
                    recovery_wp,
                    recovery=True,
                    max_steer_override=min(0.12, max_steer()),
                    lateral_gain_mult=1.2,
                    lookahead_mult=0.65,
                    smooth_mult=0.35,
                    max_delta=0.025,
                )
                target = min(target, max(2.5, current * 0.55))
        if (
            pass_fsm_phase in ("lane_change", "overtake", "prepare_pass")
            and (abs(head_err) > 40.0 or abs(lat_err) > 3.0)
            and session is not None
            and hasattr(session, "get_passing_lane_steering_waypoint")
        ):
            pass_wp = session.get_passing_lane_steering_waypoint(ego)
            steer, head_err, lat_err = _steer_to_waypoint(
                session,
                ego,
                pass_wp,
                recovery=False,
                max_steer_override=min(0.14, max_steer()),
                lateral_gain_mult=0.75,
                lookahead_mult=0.55,
                smooth_mult=0.45,
                max_delta=0.02,
            )
            target = min(target, max(4.5, current * 0.65))
        ctrl.steer = steer
    else:
        ctrl.steer = 0.0

    ctrl.throttle, ctrl.brake = speed_to_throttle_brake(
        target,
        current,
        front_gap_m=front,
        steer_abs=abs(ctrl.steer),
        no_steer_penalty=no_steer_penalty,
        phase=phase,
    )
    if abs(head_err) > 30.0 or abs(lat_err) > 2.5:
        ctrl.throttle = min(ctrl.throttle, 0.18)
        ctrl.brake = max(ctrl.brake, 0.12 if current > 4.0 else 0.0)
    if action_semantic == "follow_lead" and (abs(head_err) > 20.0 or abs(lat_err) > 2.0):
        ctrl.brake = min(ctrl.brake, 0.25)
        ctrl.throttle = min(ctrl.throttle, 0.35)
    if scripted_phase == "lane_change" and current < 6.5 and pass_maneuver_started:
        ctrl.throttle = max(ctrl.throttle, 0.42)
        ctrl.brake = min(ctrl.brake, 0.05)
    if scripted_phase == "overtake" and current < 7.0:
        ctrl.throttle = max(ctrl.throttle, 0.32)
        ctrl.brake = min(ctrl.brake, 0.06)
    ctrl.throttle = max(0.0, min(0.72, ctrl.throttle))
    ctrl.brake = max(0.0, min(0.9, ctrl.brake))
    if session is not None:
        passing_lane_id = None
        if getattr(session, "_passing_wp", None) is not None:
            passing_lane_id = int(session._passing_wp.lane_id)
        lane_center_dist_m = 0.0
        if hasattr(session, "ego_lane_center_distance_m"):
            try:
                lane_center_dist_m = float(
                    session.ego_lane_center_distance_m(ego, phase=phase or "cruise")
                )
            except Exception:
                lane_center_dist_m = abs(float(lat_err))
        session._last_control_debug = {
            "action_semantic": action_semantic,
            "heading_error_deg": float(head_err),
            "lateral_error_m": float(lat_err),
            "lane_center_error_m": lane_center_dist_m,
            "steer": float(ctrl.steer),
            "throttle": float(ctrl.throttle),
            "brake": float(ctrl.brake),
            "ego_speed_mps": float(current),
            "phase": phase,
            "pass_fsm_phase": pass_fsm_phase,
            "target_lane_id": target_lane_id,
            "travel_lane_id": travel_lane_id,
            "passing_lane_id": passing_lane_id,
            "pass_maneuver_started": pass_maneuver_started,
        }
    return ctrl


def advance_npcs_only(spec: ScenarioSpec, world: WorldState, dt: float) -> WorldState:
    full = advance_world_step(spec, world, action="wait", dt=dt)
    return replace(
        world,
        t_s=full.t_s,
        lead_x_m=full.lead_x_m,
        rear_x_m=full.rear_x_m,
        oncoming_x_m=full.oncoming_x_m,
    )


def execute_vehicle_step(
    spec: ScenarioSpec,
    world: WorldState,
    action: str,
    *,
    target_speed_mps: float = 0.0,
    passing_side: str = "left",
    duration_s: float = 1.0,
) -> Tuple[WorldState, Dict[str, Any]]:
    from perception.carla_scenario import get_session

    session = get_session()
    if not session.ready:
        from autopass.config import AutopassConfigurationError, is_test_mode

        if is_test_mode():
            logical = advance_world_step(spec, world, action=action, dt=duration_s)
            return logical, {"mode": "kinematic_fallback", "reason": "carla_not_ready"}
        raise AutopassConfigurationError(
            "CARLA session not ready for vehicle control. "
            "Call bootstrap_carla_scenario() after CarlaUE4.exe is running."
        )

    ego = session.actors.get("ego")
    if ego is None:
        from autopass.config import AutopassConfigurationError, is_test_mode

        if is_test_mode():
            logical = advance_world_step(spec, world, action=action, dt=duration_s)
            return logical, {"mode": "kinematic_fallback", "reason": "no_ego"}
        raise AutopassConfigurationError("CARLA ego vehicle missing after bootstrap.")

    next_episode_step = int(session._episode_step) + 1

    from perception.pass_control_fsm import (
        get_pass_control_state,
        pass_control_tick,
        fsm_diagnostics,
    )

    pass_in_progress = action == "pass" or bool(getattr(session, "_pass_control", None) and get_pass_control_state(session).active)
    requested_action = action
    pass_st = get_pass_control_state(session)
    pass_st, effective_action, fsm_control_failure = pass_control_tick(
        session,
        ego,
        requested_action=requested_action,
        pass_in_progress=pass_in_progress or pass_st.active,
        front_gap_m=session.measure_actor_gaps_3d().get("front", 999.0),
        clear_of_lead=_session_cleared_of_lead(session),
        speed_mps=world.ego_speed_mps,
    )
    action = effective_action
    action_semantic = "follow_lead" if action in ("wait", "follow_lead") else action

    # First execute: finalize spawn layout before actuation lock (continuity forbids snaps later).
    if next_episode_step == 1 and session.allows_pre_decision_actor_layout():
        try:
            from perception.lead_gap_diagnostics import log_lead_gap_checkpoint

            log_lead_gap_checkpoint(
                session, "E_before_first_execute", note=f"action={action_semantic}"
            )
        except Exception:
            pass
        if hasattr(session, "_ensure_spawn_gaps"):
            session._ensure_spawn_gaps(spec)
        # Lead/rear only — ego pose is established once in enable_ego_physics (below).
        if hasattr(session, "_snap_convoy_to_travel_lane"):
            session._snap_convoy_to_travel_lane(spec, skip_ego=True)
        try:
            session.run_pre_control_sanity(
                for_follow_lead=(action_semantic == "follow_lead"),
                align_ego=False,
                check_spawn_gaps=False,
            )
        except Exception as e:
            from autopass.config import AutopassConfigurationError, is_test_mode

            if is_test_mode():
                print(f"[CARLA] Pre-control sanity skipped in test mode: {e}", flush=True)
            else:
                raise AutopassConfigurationError(str(e)) from e
        for _ in range(max(1, getattr(session, "_spawn_settle_ticks", 3))):
            session.tick()

    session.mark_closed_loop_actuation_begun()
    session._episode_step = next_episode_step
    episode_step = session._episode_step
    in_spawn_grace = episode_step <= 1

    session.enable_ego_physics(True)

    from perception.actor_continuity import snapshot_continuity_baseline

    # Baseline after physics settle so ego creep during enable is not vs pre-physics pose.
    if episode_step == 1:
        for _ in range(max(1, getattr(session, "_spawn_settle_ticks", 3))):
            session.tick()
    snapshot_continuity_baseline(session, window_s=float(duration_s))

    delta = session.fixed_delta_seconds
    n_ticks = max(1, int(round(duration_s / delta)))
    speeds: list[float] = []
    min_front = 999.0
    last_ctrl = None
    clear_of_lead = _session_cleared_of_lead(session)
    ego_lane = session.infer_ego_lane_index()
    control_failure = bool(fsm_control_failure)
    lane_departure = False
    max_lane_dist = 0.0
    initial_front = session.measure_actor_gaps_3d().get("front", 999.0)
    pass_abort_reason = pass_st.abort_reason

    import carla

    for _ in range(n_ticks):
        session.update_route_cursor(ego)
        session.tick_npcs_kinematic(spec, delta)
        gaps = session.measure_actor_gaps_3d()
        front = gaps.get("front", 999.0)
        min_front = min(min_front, front)
        clear_of_lead = _session_cleared_of_lead(session)
        ego_lane = session.infer_ego_lane_index()
        v = _speed_mps(ego.get_velocity())
        speeds.append(v)

        pass_st, effective_action, tick_fail = pass_control_tick(
            session,
            ego,
            requested_action=requested_action,
            pass_in_progress=pass_st.active or requested_action == "pass",
            front_gap_m=front,
            clear_of_lead=clear_of_lead,
            speed_mps=v,
        )
        if tick_fail:
            control_failure = True
        action = effective_action
        action_semantic = "follow_lead" if action in ("wait", "follow_lead") else action

        use_corridor_offset = (
            pass_st.active
            and pass_st.phase in ("lane_change", "overtake", "merge_back")
        ) or pass_st.phase == "abort" or (
            pass_st.maneuver_started and action_semantic == "follow_lead"
        )
        if use_corridor_offset and hasattr(session, "ego_corridor_lane_offset_m"):
            lane_dist = float(session.ego_corridor_lane_offset_m(ego))
        elif action_semantic == "follow_lead":
            lane_dist = (
                session.ego_lane_center_distance_m(ego, phase="cruise")
                if hasattr(session, "ego_lane_center_distance_m")
                else 0.0
            )
        else:
            lane_metric_phase = resolve_pass_phase(
                action,
                ego_lane=ego_lane,
                clear_of_lead=clear_of_lead,
                front_gap_m=front,
                pass_fsm_phase=pass_st.phase if pass_st.active else None,
            )
            lane_dist = (
                session.ego_lane_center_distance_m(ego, phase=lane_metric_phase)
                if hasattr(session, "ego_lane_center_distance_m")
                else 0.0
            )
        max_lane_dist = max(max_lane_dist, lane_dist)

        width = session.expected_passing_lane_width_m() if session is not None else 3.5
        depart_fail_m = lane_departure_fail_m()
        if pass_st.active and pass_st.phase in ("lane_change", "overtake", "merge_back"):
            depart_fail_m = max(depart_fail_m, width * 2.15, 7.5)
        committed_between_lanes = (
            pass_st.maneuver_started
            and pass_st.phase in ("lane_change", "overtake")
            and hasattr(session, "ego_corridor_lane_offset_m")
            and lane_dist < width * 2.05
        )

        ctrl_dbg_pre = getattr(session, "_last_control_debug", {}) if session is not None else {}
        head_abs = abs(float(ctrl_dbg_pre.get("heading_error_deg", 0.0)))
        if lane_dist > lane_departure_warn_m() and (head_abs > 35.0 or lane_dist > depart_fail_m):
            recovery_wp = (
                session.get_recovery_travel_waypoint(ego)
                if hasattr(session, "get_recovery_travel_waypoint")
                else None
            )
            if recovery_wp is not None:
                rec_steer, _, _ = _steer_to_waypoint(
                    session,
                    ego,
                    recovery_wp,
                    recovery=True,
                    max_steer_override=min(0.12, max_steer()),
                    max_delta=0.025,
                )
                rec_ctrl = carla.VehicleControl(
                    throttle=0.12,
                    brake=0.2 if v > 3.5 else 0.05,
                    steer=rec_steer,
                    hand_brake=False,
                )
                ego.apply_control(rec_ctrl)
                last_ctrl = rec_ctrl
                session.tick()
                lane_dist = (
                    session.ego_lane_center_distance_m(ego, phase="cruise")
                    if hasattr(session, "ego_lane_center_distance_m")
                    else lane_dist
                )
                max_lane_dist = max(max_lane_dist, lane_dist)

        if lane_dist > depart_fail_m and not committed_between_lanes:
            control_failure = True
            lane_departure = True
            from perception.pass_control_fsm import abort_pass

            abort_pass(session, f"lane_departure_{lane_dist:.1f}m")
            pass_st = get_pass_control_state(session)
            pass_abort_reason = pass_st.abort_reason
            action = "wait"
            action_semantic = "follow_lead"
            # Continue step with merge-back recovery (no hard stop that leaves ego stranded).
        elif pass_st.phase == "abort" and action_semantic == "follow_lead":
            action = "wait"
            action_semantic = "follow_lead"

        recovery = lane_dist > lane_departure_warn_m() or pass_st.phase == "abort"

        fsm_scripted = None
        if pass_st.phase == "abort" and action_semantic == "follow_lead":
            fsm_scripted = "merge_back"
        elif pass_st.active:
            from perception.pass_control_fsm import _scripted_phase_for_fsm

            fsm_scripted = _scripted_phase_for_fsm(pass_st.phase)

        merge_recovery = pass_st.phase == "abort" and action_semantic == "follow_lead"
        last_ctrl = build_vehicle_control(
            action,
            world=world,
            spec=spec,
            target_speed_mps=target_speed_mps,
            passing_side=passing_side,
            session=session,
            ego=ego,
            measured_speed_mps=v,
            front_gap_m=front,
            clear_of_lead=clear_of_lead,
            ego_lane=ego_lane,
            recovery=recovery or merge_recovery or (episode_step == 1 and len(speeds) <= 3),
            scripted_phase=fsm_scripted,
            pass_fsm_phase=pass_st.phase if pass_st.active or pass_st.phase == "abort" else None,
            pass_maneuver_started=pass_st.maneuver_started,
        )
        if merge_recovery:
            last_ctrl.throttle = min(float(last_ctrl.throttle), 0.28)
            last_ctrl.brake = max(float(last_ctrl.brake), 0.08 if v > 4.0 else 0.0)
        if episode_step == 1 and len(speeds) <= 4:
            last_ctrl.throttle = min(float(last_ctrl.throttle), 0.22)
            last_ctrl.brake = max(float(last_ctrl.brake), 0.08 if v > 1.5 else 0.0)
        ego.apply_control(last_ctrl)
        session._last_vehicle_control = last_ctrl
        session.tick()
        sink = getattr(session, "_demo_frame_sink", None)
        import os

        dense_frames = os.environ.get("AUTOPASS_DEMO_DENSE_FRAMES", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if sink is not None and (
            dense_frames or len(speeds) % 3 == 0 or len(speeds) == n_ticks
        ):
            try:
                partial = session.materialize_logical_world(
                    world,
                    measured_speed_mps=v,
                    duration_s=delta * len(speeds),
                    ego_lane=ego_lane,
                    passed=world.passed,
                    collision=False,
                    done=False,
                )
                sink(partial, f"execute_tick_{len(speeds)}")
            except Exception:
                pass

    ctrl = last_ctrl
    measured_speed = float(sum(speeds) / len(speeds)) if speeds else world.ego_speed_mps
    progress_delta = measured_speed * duration_s
    clear_of_lead = _session_cleared_of_lead(session)
    ego_lane = session.infer_ego_lane_index()
    travel_lane_id = int(session._travel_wp.lane_id) if session._travel_wp else None
    travel_road_id = int(session._travel_wp.road_id) if session._travel_wp else None
    try:
        ego_wp_now = session.map.get_waypoint(ego.get_location(), project_to_road=True) if session.map else None
    except Exception:
        ego_wp_now = None
    from perception.carla_pass_maneuver import is_pass_maneuver_complete

    passed_now = is_pass_maneuver_complete(
        session,
        ego_wp=ego_wp_now,
        ego_lane=ego_lane,
        clear_of_lead=clear_of_lead,
        travel_lane_id=travel_lane_id or 0,
        travel_road_id=travel_road_id or 0,
    )
    passed = world.passed or passed_now

    collision, collision_detail = session.check_actor_proximity(threshold_m=4.2)
    collision_source = ""
    if collision:
        collision_source = "carla_proximity"
        if in_spawn_grace:
            collision = False
            collision_detail = ""
            collision_source = "ignored_spawn_grace"
    near_miss = _near_miss_this_step(
        session,
        min_front,
        collision=collision,
        pass_st=pass_st,
        ego_lane=ego_lane,
        requested_action=requested_action,
    )

    done = (
        collision
        or world.ego_x_m + progress_delta >= spec.route.goal_x_m
        or world.t_s + duration_s >= 160.0
    )

    world_after = session.materialize_logical_world(
        world,
        measured_speed_mps=measured_speed,
        duration_s=duration_s,
        ego_lane=ego_lane,
        passed=passed,
        collision=collision,
        done=done,
    )

    if not collision:
        logical_hit, logical_detail = _logical_collision(
            spec, world, world_after.ego_x_m, ego_lane, world_after, passed, session=session
        )
        if logical_hit:
            collision = True
            collision_detail = logical_detail
            collision_source = "logical_overlap"
            world_after = replace(world_after, collision=True, done=True)

    if collision and collision_source not in ("ignored_spawn_grace", ""):
        session.record_collision_event(collision_source, collision_detail, episode_step)
    session._set_spectator_behind_ego()

    phase = resolve_pass_phase(
        action,
        ego_lane=ego_lane,
        clear_of_lead=clear_of_lead,
        front_gap_m=min_front,
        pass_fsm_phase=pass_st.phase if pass_st.active else None,
    )
    fsm_diag = fsm_diagnostics(session, ego, pass_st)
    target_lane_id = fsm_diag.get("target_lane_id")
    target_road_id = fsm_diag.get("target_road_id")
    travel_lane_id = fsm_diag.get("travel_lane_id")
    travel_road_id = fsm_diag.get("travel_road_id")
    feedback = {
        "mode": "carla_vehicle",
        "action": action,
        "action_semantic": "follow_lead" if action in ("wait", "follow_lead") else ("pass" if action == "pass" else action),
        "ticks": n_ticks,
        "measured_speed_mps": round(measured_speed, 2),
        "progress_delta_m": round(progress_delta, 2),
        "throttle": round(float(ctrl.throttle), 3) if ctrl else 0.0,
        "steer": round(float(ctrl.steer), 3) if ctrl else 0.0,
        "brake": round(float(ctrl.brake), 3) if ctrl else 0.0,
        "ego_lane": ego_lane,
        "ego_lane_id": ego_lane,
        "pass_phase": phase,
        "target_lane_id": target_lane_id,
        "target_road_id": target_road_id,
        "travel_lane_id": travel_lane_id,
        "travel_road_id": travel_road_id,
        "front_gap_m": round(min_front, 2),
        "min_front_gap_m": round(min_front, 2),
        "clear_of_lead": clear_of_lead,
        "near_miss": near_miss,
        "collision": collision,
        "collision_detail": collision_detail,
        "collision_source": collision_source,
        "collision_step": episode_step if collision else None,
        "episode_step": episode_step,
        "control_failure": control_failure,
        "failure_type": "lane_departure" if lane_departure else ("pass_control" if pass_abort_reason else ""),
        "pass_abort_reason": pass_abort_reason,
        "pass_maneuver_started": pass_st.maneuver_started,
        "pass_fsm_phase": pass_st.phase,
        "pass_fsm_active": pass_st.active,
        "requested_action": requested_action,
        "max_lane_center_dist_m": round(max_lane_dist, 2),
        "lane_debug": session.route_cursor_debug_snapshot(ego)
        if hasattr(session, "route_cursor_debug_snapshot")
        else {},
        "carla_geometry": session.geometry_debug_snapshot() if hasattr(session, "geometry_debug_snapshot") else {},
    }
    ctrl_dbg = getattr(session, "_last_control_debug", {}) if session is not None else {}
    feedback.update(
        {
            "lane_center_error_m": float(ctrl_dbg.get("lane_center_error_m", max_lane_dist if max_lane_dist else 0.0)),
            "heading_error_deg": float(ctrl_dbg.get("heading_error_deg", 0.0)),
            "steer": round(float(ctrl_dbg.get("steer", feedback["steer"])), 3),
            "throttle": round(float(ctrl_dbg.get("throttle", feedback["throttle"])), 3),
            "brake": round(float(ctrl_dbg.get("brake", feedback["brake"])), 3),
            "ego_speed_mps": round(float(ctrl_dbg.get("ego_speed_mps", measured_speed)), 3),
        }
    )
    feedback.update(fsm_diag)
    if (
        pass_abort_reason
        and not control_failure
        and action_semantic == "follow_lead"
        and "planner_requested" not in pass_abort_reason
    ):
        feedback["control_failure"] = True
        feedback["failure_type"] = "pass_abort"
    if pass_st.phase == "abort" or (not pass_st.active and requested_action == "pass"):
        try:
            lane_err = session.ego_lane_center_distance_m(ego, phase="cruise")
            head_err = abs(float(getattr(session, "_last_control_debug", {}).get("heading_error_deg", 0.0)))
            # Only hard-snap when truly off-corridor — mild departures recover via steer next step.
            if lane_err > 9.5 or head_err > 85.0:
                session.snap_ego_to_travel_pose(max_lateral_m=6.0)
                session._zero_vehicle_control(ego)
            elif lane_err > 2.5 and hasattr(session, "get_travel_steering_waypoint"):
                try:
                    session.update_route_cursor(ego)
                except Exception:
                    pass
        except Exception:
            pass
    if hasattr(session, "pass_longitudinal_snapshot"):
        feedback.update(session.pass_longitudinal_snapshot())
    feedback["actor_continuity"] = session.longitudinal_continuity_diag(
        pass_in_progress=pass_st.active,
        action=action_semantic,
        context="after_execute",
        check_violations=True,
    )
    session.mark_graph_execute_completed()
    if episode_step == 1:
        try:
            from perception.lead_gap_diagnostics import log_lead_gap_checkpoint

            log_lead_gap_checkpoint(session, "F_after_first_execute", note=f"action={action_semantic}")
        except Exception:
            pass
    return world_after, feedback


def _logical_collision(
    spec: ScenarioSpec,
    world: WorldState,
    ego_x: float,
    ego_lane: int,
    npc: WorldState,
    passed: bool,
    *,
    session=None,
) -> Tuple[bool, str]:
    if session is not None and getattr(session, "ready", False):
        gaps_3d = session.measure_actor_gaps_3d()
        lead_signed = session.signed_gap_from_ego("lead") if hasattr(session, "signed_gap_from_ego") else None
        rear_signed = session.signed_gap_from_ego("rear") if hasattr(session, "signed_gap_from_ego") else None
        on_signed = session.signed_gap_from_ego("oncoming") if hasattr(session, "signed_gap_from_ego") else None

        # In CARLA mode, never infer overlap from missing projection.
        if ego_lane == 0 and not passed and lead_signed is not None and gaps_3d.get("front", 999.0) < 12.0:
            if abs(float(lead_signed)) < 4.0:
                return True, "lead_proximity_logical"
        if ego_lane == 0 and rear_signed is not None and gaps_3d.get("rear", 999.0) < 12.0:
            if abs(float(rear_signed)) < 4.0:
                return True, "rear_proximity_logical"
        if ego_lane == 1 and on_signed is not None and gaps_3d.get("oncoming", 999.0) < 14.0:
            if abs(float(on_signed)) < 6.0:
                return True, "oncoming_proximity_logical"
        if ego_lane == 1 and rear_signed is not None and gaps_3d.get("rear", 999.0) < 12.0:
            if abs(float(rear_signed)) < 5.0:
                return True, "rear_proximity_logical"
        return False, ""

    if not passed and ego_lane == 0 and abs(ego_x - npc.lead_x_m) < 4.0:
        return True, "lead_proximity_logical"
    if ego_lane == 0 and abs(ego_x - npc.rear_x_m) < 4.0:
        return True, "rear_proximity_logical"
    if ego_lane == 1 and abs(ego_x - npc.oncoming_x_m) < 6.0:
        return True, "oncoming_proximity_logical"
    if ego_lane == 1 and abs(ego_x - npc.rear_x_m) < 5.0:
        return True, "rear_proximity_logical"
    return False, ""
