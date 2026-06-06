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
    """Full longitudinal clearance (8 m axis-ahead of lead), not the 3 m merge trigger."""
    from perception.carla_pass_maneuver import merge_clearance_m

    clearance = merge_clearance_m()
    if hasattr(session, "ego_cleared_lead"):
        return bool(session.ego_cleared_lead(clearance))
    return bool(session.ego_clear_of_lead(clearance))


def _sync_pass_fsm_for_merge(session, ego, pass_st) -> None:
    """Ensure FSM reflects merge-back once ego is axis-ahead of the lead."""
    from perception.pass_geometry import pass_merge_back_due

    if pass_st is None or ego is None:
        return
    if pass_st.phase in ("idle", "prepare_pass"):
        return
    if not pass_merge_back_due(session):
        return
    pass_st.active = True
    pass_st.phase = "merge_back"
    pass_st.ticks_in_phase = 0
    pass_st.target_lane_source = "return_lane"
    pass_st.maneuver_started = True


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
            # Approach lead at slightly above lead speed before lane change starts
            target = min(target, lead_speed_mps + 2.5, speed_limit_mps * 0.65, max(6.0, current_speed_mps))
        elif pass_fsm_phase == "lane_change" and not pass_maneuver_started:
            # Slow lane change: reduce lateral momentum to prevent outer-barrier overshoot.
            target = min(target, max(lead_speed_mps + 1.5, 4.5), speed_limit_mps * 0.45, max(4.5, current_speed_mps))
        else:
            target = min(speed_limit_mps + 2.0, max(target, current_speed_mps + 1.5))
        if pass_fsm_phase in ("prepare_pass", "lane_change", "overtake") and along < pass_lateral_min_m():
            target = min(
                target,
                max(lead_speed_mps + 3.0, 6.0),
                speed_limit_mps * 0.72,
                max(6.0, current_speed_mps * 0.90),
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
        if along > 0.0 and along < 22.0:
            target = max(
                target,
                lead_speed_mps + (4.0 if along < 8.0 else 2.5),
                speed_limit_mps * (0.75 if along < 8.0 else 0.62),
                6.5,
            )
        return max(0.0, target)

    if phase in ("cruise", "approach", "merge") and front_gap_m < safe_follow_m():
        # Proportional slowdown as gap closes
        gap_err = front_gap_m - safe_follow_m()
        target = min(target, lead_speed_mps + max(-1.0, gap_err * 0.35))

    if (
        front_gap_m < critical_gap_m()
        and pass_fsm_phase not in ("lane_change", "overtake", "merge_back")
    ):
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
    pass_fsm_phase: str | None = None,
) -> Tuple[float, float]:
    if (
        phase not in ("overtake", "lane_change", "merge")
        and pass_fsm_phase not in ("lane_change", "overtake", "merge_back")
        and front_gap_m < critical_gap_m()
    ):
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
    pass_fsm_phase: str | None = None,
    passing_side: str = "left",
) -> Tuple[float, float, float]:
    """Pure-pursuit / Stanley-style steer toward target waypoint."""
    from perception.carla_lane_keep import lateral_error_m, pure_pursuit_steer, steer_from_lane_errors

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
    if abs(head) > 90.0 and session is not None and pass_fsm_phase in (
        "lane_change",
        "overtake",
        "merge_back",
        "prepare_pass",
    ):
        lane_wp = None
        head_lane = 0.0
        if pass_fsm_phase in ("lane_change", "overtake", "prepare_pass") and hasattr(
            session, "heading_error_to_passing_lane_deg"
        ):
            head_lane = float(session.heading_error_to_passing_lane_deg(ego))
            lane_wp = (
                session._passing_lane_anchor_at_ego(ego)
                if hasattr(session, "_passing_lane_anchor_at_ego")
                else getattr(session, "_passing_wp", None)
            )
        elif hasattr(session, "heading_error_to_travel_lane_deg"):
            head_lane = float(session.heading_error_to_travel_lane_deg(ego))
            lane_wp = (
                session._travel_lane_anchor_at_ego(ego)
                if hasattr(session, "_travel_lane_anchor_at_ego")
                else getattr(session, "_travel_wp", None)
            )
        if lane_wp is not None:
            lat_lane = lateral_error_m(ego_tf.location, ego_tf.rotation.yaw, lane_wp.transform.location)
            steer, head, lat = steer_from_lane_errors(
                ego_tf.location,
                ego_tf.rotation.yaw,
                head_lane,
                lat_lane,
                lookahead_m=lookahead,
                max_steer=ms,
                steer_gain=steer_gain(),
                lateral_gain=lateral_steer_gain() * (1.4 if recovery else 1.0) * lateral_gain_mult,
                prev_steer=prev,
                smooth=steer_smooth() * smooth_mult,
            )
        elif hasattr(session, "_lane_change_steer_waypoint") and pass_fsm_phase in ("lane_change", "overtake"):
            alt_wp = session._lane_change_steer_waypoint(ego, passing_side)
            steer2, head2, lat2 = pure_pursuit_steer(
                ego_tf.location,
                ego_tf.rotation.yaw,
                alt_wp.transform.location,
                lookahead_m=lookahead,
                max_steer=ms,
                steer_gain=steer_gain(),
                lateral_gain=lateral_steer_gain() * (1.4 if recovery else 1.0) * lateral_gain_mult,
                prev_steer=prev,
                smooth=steer_smooth() * smooth_mult,
            )
            if abs(head2) < abs(head):
                steer, head, lat = steer2, head2, lat2
    delta = steer - prev
    if delta > max_delta:
        steer = prev + max_delta
    elif delta < -max_delta:
        steer = prev - max_delta
    # Hard cap: prevent multi-tick accumulation past the intended max_steer_override.
    steer = max(-ms, min(ms, steer))
    session._last_steer = steer
    return steer, head, lat


def _passing_lane_steer_sign(session) -> float:
    """Sign of CARLA steer that moves ego toward the passing lane (-1=left, +1=right)."""
    if session is None:
        return 1.0
    ego = session.actors.get("ego") if hasattr(session, "actors") else None
    try:
        travel = (
            session._travel_lane_anchor_at_ego(ego)
            if ego is not None and hasattr(session, "_travel_lane_anchor_at_ego")
            else getattr(session, "_travel_wp", None)
        )
        passing = (
            session._passing_lane_anchor_at_ego(ego)
            if ego is not None and hasattr(session, "_passing_lane_anchor_at_ego")
            else getattr(session, "_passing_wp", None)
        )
        if travel is not None and passing is not None:
            tw_fwd = travel.transform.get_forward_vector()
            tx, ty = float(tw_fwd.x), float(tw_fwd.y)
            mag = math.hypot(tx, ty)
            if mag > 1e-6:
                tx, ty = tx / mag, ty / mag
                tloc = travel.transform.location
                ploc = passing.transform.location
                lx, ly = float(ploc.x - tloc.x), float(ploc.y - tloc.y)
                left_nx, left_ny = -ty, tx
                pass_left = lx * left_nx + ly * left_ny > 0.0
                passing_side = getattr(session, "_passing_side", None) or "right"
                if passing_side == "right":
                    pass_left = not pass_left
                result = -1.0 if pass_left else 1.0
                # Left-side passing: invert final sign so negative steer → leftward → correct.
                if passing_side == "left":
                    result = -result
                return result
        basis = session._ego_travel_basis()
    except Exception:
        return 1.0
    if basis is None:
        return 1.0
    _ego_xyz, travel, lateral = basis
    tx, ty = float(travel[0]), float(travel[1])
    lx, ly = float(lateral[0]), float(lateral[1])
    if abs(tx) + abs(ty) < 1e-6 or abs(lx) + abs(ly) < 1e-6:
        return 1.0
    left_nx, left_ny = -ty, tx
    pass_left = lx * left_nx + ly * left_ny > 0.0
    passing_side = getattr(session, "_passing_side", None) or "right"
    if passing_side == "right":
        pass_left = not pass_left
    result = -1.0 if pass_left else 1.0
    if passing_side == "left":
        result = -result
    # CARLA VehicleControl: -1 = left, +1 = right.
    return result


def _ego_off_spawn_corridor(session, ego) -> bool:
    """True when ego map-lane is outside spawn travel/passing lanes (parallel-road snap)."""
    tw = getattr(session, "_travel_wp", None)
    pw = getattr(session, "_passing_wp", None)
    if tw is None or ego is None or getattr(session, "map", None) is None:
        return False
    try:
        ego_wp = session.map.get_waypoint(ego.get_location(), project_to_road=True)
        if int(ego_wp.road_id) != int(tw.road_id):
            return True
        allowed = {int(tw.lane_id)}
        if pw is not None:
            allowed.add(int(pw.lane_id))
        return int(ego_wp.lane_id) not in allowed
    except Exception:
        return False


def _clamp_pass_maneuver_steer(
    session,
    ego,
    steer: float,
    *,
    pass_fsm_phase: str | None,
    scripted_phase: str | None,
) -> float:
    """Single-lane pass guard — no multi-lane excursions toward barriers."""
    if session is None or ego is None or pass_fsm_phase not in (
        "prepare_pass",
        "lane_change",
        "overtake",
        "merge_back",
    ):
        return steer
    if not hasattr(session, "lateral_lane_offsets_m"):
        return steer
    d_travel, d_pass, width = session.lateral_lane_offsets_m(ego)
    w = max(2.5, float(width))
    shift = (
        float(session.lateral_shift_toward_passing_m(ego))
        if hasattr(session, "lateral_shift_toward_passing_m")
        else 0.0
    )
    sign = _passing_lane_steer_sign(session)
    corridor_min = min(float(d_travel), float(d_pass))
    median_blocked = bool(
        hasattr(session, "passing_lane_median_blocked") and session.passing_lane_median_blocked()
    )
    outer_blocked = bool(
        hasattr(session, "passing_lane_outer_edge_blocked")
        and session.passing_lane_outer_edge_blocked()
    )
    edge_blocked = median_blocked or outer_blocked
    # Outer edge guard: prevent outward steer AND boost inward correction.
    if edge_blocked and shift >= w * 0.42:
        # Block further outward (toward barrier) steering.
        if sign * steer > 0.02:
            if shift >= w * 0.58 or float(d_pass) > w * 0.42:
                return -sign * min(0.16, max(0.08, abs(steer)))
            return min(steer, 0.05) if sign > 0 else max(steer, -0.05)
        # Boost inward correction when ego is drifting toward the outer edge.
        # This counteracts residual lateral momentum without adding outward force.
        if float(d_pass) > w * 0.28 and shift >= w * 0.52:
            correction = -sign * min(max_steer(), 0.12)
            # Apply if stronger than current steer in the correct (inward) direction.
            if sign * correction < 0 and abs(correction) > abs(steer):
                return correction
    # Straddling between lanes — gentle single-lane change only.
    if corridor_min > w * 0.32 and corridor_min < w * 0.68 and scripted_phase == "lane_change":
        cap = min(0.14, max_steer() * lane_change_steer_cap_mult() * 0.75)
        return max(-cap, min(cap, steer))
    # On passing lane center — hold heading, no extra lateral crank.
    if (
        scripted_phase in ("lane_change", "overtake")
        and float(d_pass) < w * 0.38
        and shift >= w * 0.52
    ):
        hold_cap = 0.06
        return max(-hold_cap, min(hold_cap, steer))
    return steer


def _pass_lane_width_m(session, ego) -> float:
    """Lane spacing at ego longitude (dynamic on curves)."""
    if session is None:
        return 3.5
    if ego is not None and hasattr(session, "local_passing_lane_width_m"):
        try:
            return max(2.5, float(session.local_passing_lane_width_m(ego)))
        except Exception:
            pass
    if hasattr(session, "expected_passing_lane_width_m"):
        try:
            return max(2.5, float(session.expected_passing_lane_width_m()))
        except Exception:
            pass
    return 3.5


def _steer_toward_passing_lane_corridor(
    session,
    ego,
    *,
    max_steer_override: float | None = None,
    lateral_gain_mult: float = 1.0,
    smooth_mult: float = 0.55,
    max_delta: float = 0.05,
) -> Tuple[float, float, float]:
    """Lane-error steer onto the passing-lane center (wide / between-lanes recovery)."""
    from perception.carla_lane_keep import lateral_error_m, steer_from_lane_errors

    if session is None or ego is None:
        return 0.0, 0.0, 0.0
    passing = (
        session._passing_lane_anchor_at_ego(ego)
        if hasattr(session, "_passing_lane_anchor_at_ego")
        else getattr(session, "_passing_wp", None)
    )
    if passing is None:
        return getattr(session, "_last_steer", 0.0), 0.0, 0.0
    prev = getattr(session, "_last_steer", 0.0)
    ego_tf = ego.get_transform()
    head = (
        float(session.heading_error_to_passing_lane_deg(ego))
        if hasattr(session, "heading_error_to_passing_lane_deg")
        else 0.0
    )
    signed_lat = (
        float(session.signed_lateral_error_toward_passing_m(ego))
        if hasattr(session, "signed_lateral_error_toward_passing_m")
        else 0.0
    )
    lat = (
        signed_lat
        if abs(signed_lat) > 0.08
        else lateral_error_m(ego_tf.location, ego_tf.rotation.yaw, passing.transform.location)
    )
    ms = max_steer_override if max_steer_override is not None else max_steer()
    la = route_lookahead_m() * 0.35
    lat_gain = lateral_steer_gain() * lateral_gain_mult * (lane_change_lateral_mult() / 2.4)
    steer, head, lat = steer_from_lane_errors(
        ego_tf.location,
        ego_tf.rotation.yaw,
        head,
        lat,
        lookahead_m=la,
        max_steer=ms,
        steer_gain=steer_gain(),
        lateral_gain=lat_gain,
        prev_steer=prev,
        smooth=steer_smooth() * smooth_mult,
    )
    width = 3.5
    if hasattr(session, "expected_passing_lane_width_m"):
        try:
            width = float(session.expected_passing_lane_width_m())
        except Exception:
            pass
    if ego is not None and hasattr(session, "lateral_lane_offsets_m"):
        _dt, d_pass, _w = session.lateral_lane_offsets_m(ego)
        width = max(2.5, float(_w))
    else:
        d_pass = width
    deficit = max(width * 0.52 - float(signed_lat), (width * 0.52) - (width - float(d_pass)))
    if abs(head) < 28.0 and abs(deficit) > 0.12:
        sign = _passing_lane_steer_sign(session)
        mag = min(ms, 0.06 + abs(deficit) * 0.14)
        if deficit > 0.0:
            # Undershot: push toward passing lane
            pull = mag * sign
            steer = max(steer, pull) if sign > 0 else min(steer, pull)
        else:
            # Overshot past passing lane center: pull back inward (inverted direction)
            pull = -mag * sign
            steer = min(steer, pull) if sign > 0 else max(steer, pull)
    delta = steer - prev
    if delta > max_delta:
        steer = prev + max_delta
    elif delta < -max_delta:
        steer = prev - max_delta
    # Hard cap: smoothing must not accumulate past the intended max over multiple ticks.
    steer = max(-ms, min(ms, steer))
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
    if pass_fsm_phase == "prepare_pass" and not pass_maneuver_started:
        phase = "lane_change"

    lead_speed_for_acc = world.ego_speed_mps
    if session is not None and hasattr(session, "actors"):
        lead_actor = session.actors.get("lead")
        if lead_actor is not None:
            try:
                live_v = _speed_mps(lead_actor.get_velocity())
                lead_speed_for_acc = live_v
                # Kinematic NPCs report ~0 m/s from get_velocity(); use commanded speed.
                # Without this, the overtake floor is set at 5.5 m/s (= max(0+2.5, 5.5))
                # while the lead moves at spec speed, making overtake physically impossible.
                if live_v < 0.5 and hasattr(session, "kinematic_lead_speed_mps"):
                    cmd_v = float(session.kinematic_lead_speed_mps())
                    if cmd_v > live_v:
                        lead_speed_for_acc = cmd_v
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

    merge_pass_active = pass_fsm_phase == "merge_back"
    if session is not None and not merge_pass_active:
        try:
            from perception.pass_geometry import pass_finish_active, pass_merge_back_due

            merge_pass_active = pass_merge_back_due(session) or pass_finish_active(session)
        except Exception:
            pass
    if recovery and not merge_pass_active:
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

    on_passing_corridor = (
        session is not None
        and ego is not None
        and hasattr(session, "ego_on_passing_corridor")
        and session.ego_on_passing_corridor(ego)
    )
    corridor_ok = bool(on_passing_corridor)
    if session is not None and ego is not None and hasattr(session, "lateral_lane_offsets_m"):
        d_travel, d_pass, width = session.lateral_lane_offsets_m(ego)
        w = max(2.5, float(width))
        corridor_ok = corridor_ok and float(d_pass) < w * 0.52 and float(d_travel) < w * 0.65

    steer_kw: dict = {}
    no_steer_penalty = False
    if scripted_phase == "lane_change":
        cap_mult = lane_change_steer_cap_mult()
        shift = 0.0
        width = 3.5
        overshoot_m = 0.0
        if session is not None and ego is not None:
            if hasattr(session, "lateral_shift_toward_passing_m"):
                shift = float(session.lateral_shift_toward_passing_m(ego))
            if hasattr(session, "expected_passing_lane_width_m"):
                width = float(session.expected_passing_lane_width_m())
            if hasattr(session, "lateral_overshoot_past_passing_m"):
                overshoot_m = float(session.lateral_overshoot_past_passing_m(ego))
        if (
            pass_maneuver_started
            and lead_along is not None
            and float(lead_along) < pass_lateral_min_m()
        ):
            target = min(target, max(lead_speed_for_acc + 2.0, 6.0), spec.route.speed_limit_mps * 0.62)
        progress = min(1.0, max(0.0, shift / max(2.5, width)))
        d_pass_frac = 0.0
        d_travel_frac = 0.0
        wide_departure = False
        if session is not None and ego is not None and hasattr(session, "lateral_lane_offsets_m"):
            _dt, d_pass, _w = session.lateral_lane_offsets_m(ego)
            d_pass_frac = float(d_pass) / max(2.5, width)
            d_travel_frac = float(_dt) / max(2.5, width)
            wide_departure = min(float(_dt), float(d_pass)) > float(_w) * 0.55
        moderate_overshoot = (
            overshoot_m > 0.12
            and d_pass_frac < 0.95
            and not wide_departure
        )
        if moderate_overshoot:
            progress = min(progress, 0.35)
        elif d_pass_frac < 0.38:
            progress = max(progress, min(1.0, 1.0 - d_pass_frac / 0.38))
            if shift >= width * 0.45:
                target = min(target, max(lead_speed_for_acc + 1.5, 4.8))
        elif d_pass_frac > 0.55 and not wide_departure:
            progress = min(progress, 0.4)
        elif wide_departure:
            progress = max(progress, 0.62)
        # Cap speed during active lateral movement only (not prepare_pass approach phase).
        # During prepare_pass the ego must keep pace with the lead; slowing below lead speed
        # causes the gap to balloon on repeated pass attempts.
        if not (pass_fsm_phase == "prepare_pass" and not pass_maneuver_started):
            target = min(target, 2.8 + 0.9 * progress)  # max ~3.7 m/s while steering laterally
        lat_mult = (0.9 + 0.38 * progress) * (lane_change_lateral_mult() / 2.4)
        if d_pass_frac < 0.36:
            lat_mult = max(lat_mult, (1.05 + 0.65 * (0.36 - d_pass_frac)) * (lane_change_lateral_mult() / 2.4))
        elif d_pass_frac > 0.52 and not wide_departure:
            lat_mult = min(lat_mult, 1.05 * (lane_change_lateral_mult() / 2.4))
        elif wide_departure:
            lat_mult = max(lat_mult, 1.65 * (lane_change_lateral_mult() / 2.4))
        hint = getattr(session, "_lane_marking_hint", {}) if session is not None else {}
        if (
            hint.get("center_line_under_ego")
            and pass_fsm_phase in ("prepare_pass", "lane_change", "overtake")
            and d_pass_frac > 0.22
        ):
            lat_mult = max(lat_mult, 1.18)
            if hasattr(session, "passing_lane_commit_alpha"):
                progress = max(progress, min(1.0, float(session.passing_lane_commit_alpha(ego))))
        if on_passing_corridor:
            steer_kw = {
                "lateral_gain_mult": 0.86,
                "lookahead_mult": 0.56,
                "smooth_mult": 0.65,
                "max_delta": 0.028,
                "max_steer_override": min(0.11, max_steer()),
            }
            no_steer_penalty = True
        else:
            lc_cap = min(
                max_steer() * min(cap_mult, 1.25) * (0.78 + 0.22 * progress),
                0.26,
            )
            if d_pass_frac > 0.42 or (
                session is not None
                and ego is not None
                and hasattr(session, "lateral_lane_offsets_m")
            ):
                _dt, _dp, _w = session.lateral_lane_offsets_m(ego)
                _w = max(2.5, float(_w))
                if float(_dt) > _w * 0.42 and float(_dp) > _w * 0.42:
                    lc_cap = min(max_steer() * cap_mult * (1.55 if wide_departure else 1.35), 0.38 if wide_departure else 0.32)
                    lat_mult = max(lat_mult, (1.65 if wide_departure else 1.35) * (lane_change_lateral_mult() / 2.4))
            steer_kw = {
                "max_steer_override": lc_cap,
                "lateral_gain_mult": lat_mult,
                "lookahead_mult": 0.46 + 0.12 * progress,
                "smooth_mult": 0.55,
                "max_delta": 0.055 if wide_departure else 0.042,
            }
        if moderate_overshoot and not on_passing_corridor:
            steer_kw["max_steer_override"] = min(0.1, max_steer())
            steer_kw["lateral_gain_mult"] = 0.58
            steer_kw["lookahead_mult"] = 0.62
        no_steer_penalty = True
    elif scripted_phase == "merge_back":
        from perception.pass_geometry import (
            axis_ahead_of_lead,
            lateral_corridor_offset_m,
            pass_finish_active,
            pass_merge_back_due,
            wide_off_corridor,
        )

        latch_w = 3.5
        if session is not None and hasattr(session, "expected_passing_lane_width_m"):
            latch_w = float(session.expected_passing_lane_width_m())
        latched = bool(
            session is not None
            and (
                getattr(session, "_pass_corridor_committed", False)
                or float(getattr(session, "_pass_peak_shift_m", 0.0)) >= latch_w * 0.45
            )
        )
        corridor = lateral_corridor_offset_m(session, ego) if session is not None else 999.0
        w = latch_w
        wide = corridor > w * 0.55
        extreme_wide = session is not None and wide_off_corridor(session, ego, width_frac=1.35)
        merge_active = (
            latched
            and session is not None
            and pass_merge_back_due(session, ego)
        ) or pass_finish_active(session)
        if session is not None and hasattr(session, "approaching_corridor_end"):
            try:
                if session.approaching_corridor_end(ego):
                    target = min(target, 3.0 if extreme_wide else 4.5)
            except Exception:
                pass
        elif wide:
            target = min(target, max(3.5, 6.5 - corridor / max(w, 2.5)))
        elif merge_active and session is not None and axis_ahead_of_lead(session):
            target = min(target, max(lead_speed_for_acc + 1.0, 5.0))
        elif merge_active:
            target = max(target, min(lead_speed_for_acc + 2.0, 9.0))
        else:
            target = max(target, 5.5)
        if not clear_of_lead and not wide:
            target = min(target, max(lead_speed_for_acc + 2.5, 5.5))
        steer_kw = {
            "max_steer_override": min(
                0.42 if extreme_wide else (0.38 if wide else 0.32),
                max_steer() * (2.0 if extreme_wide else (1.65 if wide else 1.45)),
            ),
            "lateral_gain_mult": 2.1 if extreme_wide else (1.75 if wide else (1.45 if merge_active else 1.35)),
            "lookahead_mult": 0.32 if extreme_wide else (0.42 if wide else 0.5),
            "smooth_mult": 0.45 if extreme_wide else (0.55 if wide else 0.62),
            "max_delta": 0.06 if extreme_wide else (0.05 if wide else 0.038),
        }
        no_steer_penalty = True
    elif scripted_phase == "overtake":
        from perception.pass_geometry import pass_merge_back_due

        merge_due = session is not None and pass_merge_back_due(session, ego)
        if merge_due and pass_fsm_phase in ("overtake", "merge_back"):
            scripted_phase = "merge_back"
            pass_fsm_phase = "merge_back"
        latch_w = 3.5
        if session is not None and hasattr(session, "expected_passing_lane_width_m"):
            latch_w = float(session.expected_passing_lane_width_m())
        latched = bool(
            session is not None
            and (
                getattr(session, "_pass_corridor_committed", False)
                or float(getattr(session, "_pass_peak_shift_m", 0.0)) >= latch_w * 0.45
            )
        )
        wide_overtake = False
        on_passing_geom = False
        if session is not None and ego is not None and hasattr(session, "lateral_lane_offsets_m"):
            _dt, d_pass, width = session.lateral_lane_offsets_m(ego)
            w = max(2.5, float(width))
            on_passing_geom = float(d_pass) < w * 0.48
            wide_overtake = (float(d_pass) > w * 0.42 or float(_dt) > w * 0.55) and not on_passing_geom
        # During lane change only: cap speed when settling onto passing lane center.
        # Do NOT apply during overtake — ego must accelerate past lead.
        if on_passing_geom and pass_fsm_phase in ("prepare_pass", "lane_change"):
            target = min(target, max(lead_speed_for_acc + 0.5, 4.0))
        if not clear_of_lead and not wide_overtake:
            floor = max(lead_speed_for_acc + (2.5 if latched else 3.5), 5.5 if latched else 6.5)
            cap = spec.route.speed_limit_mps * (0.62 if latched else 0.72)
            target = max(target, min(floor, cap))
            if latched and session is not None and hasattr(session, "actor_travel_s"):
                try:
                    from autopass.carla_tuning import merge_clear_m

                    ego_s = session.actor_travel_s("ego")
                    lead_s = session.actor_travel_s("lead")
                    if ego_s is not None and lead_s is not None and float(ego_s) < float(lead_s) + merge_clear_m():
                        target = min(target, max(lead_speed_for_acc + 2.0, 6.5))
                except Exception:
                    pass
            # Only cap for proximity during lane change — allow full speed during overtake
            if on_passing_geom and lead_along is not None and float(lead_along) < 14.0 and pass_fsm_phase != "overtake":
                target = min(target, 6.0)
        # Strong overtake floor: ego must go meaningfully faster than lead to actually pass.
        if pass_fsm_phase == "overtake" and on_passing_geom and not clear_of_lead and latched:
            overtake_floor = min(spec.route.speed_limit_mps * 0.82, lead_speed_for_acc + 5.0)
            target = max(target, overtake_floor)
        elif clear_of_lead:
            target = max(target, spec.route.speed_limit_mps * 0.78)
        elif wide_overtake:
            target = min(target, max(4.0, current * 0.65))
        if on_passing_geom or (corridor_ok and latched):
            steer_kw = {
                "lateral_gain_mult": 0.86,
                "lookahead_mult": 0.56,
                "smooth_mult": 0.65,
                "max_delta": 0.028,
                "max_steer_override": min(0.11, max_steer()),
            }
            no_steer_penalty = True
        elif latched and not corridor_ok:
            steer_kw = {
                "max_steer_override": min(0.22, max_steer() * 1.25),
                "lateral_gain_mult": 1.35,
                "lookahead_mult": 0.5,
                "smooth_mult": 0.55,
                "max_delta": 0.04,
            }
            no_steer_penalty = True
        elif corridor_ok:
            steer_kw = {
                "lateral_gain_mult": 0.86,
                "lookahead_mult": 0.56,
                "smooth_mult": 0.65,
                "max_delta": 0.028,
                "max_steer_override": min(0.11, max_steer()),
            }
            no_steer_penalty = True
        else:
            steer_kw = {
                "max_steer_override": min(0.24, max_steer() * 1.35),
                "lateral_gain_mult": 1.42,
                "lookahead_mult": 0.46,
                "smooth_mult": 0.52,
                "max_delta": 0.04,
            }
        if wide_overtake and not merge_due:
            target = min(target, max(4.5, current * 0.72))
            steer_kw = {
                "max_steer_override": min(0.28, max_steer() * 1.4),
                "lateral_gain_mult": 1.55,
                "lookahead_mult": 0.44,
                "smooth_mult": 0.48,
                "max_delta": 0.045,
            }
            no_steer_penalty = True
        if (
            not on_passing_corridor
            and session is not None
            and ego is not None
            and hasattr(session, "lateral_overshoot_past_passing_m")
            and float(session.lateral_overshoot_past_passing_m(ego)) > 0.15
            and merge_due
        ):
            scripted_phase = "merge_back"
            pass_fsm_phase = "merge_back"
            steer_kw = {
                "max_steer_override": min(0.28, max_steer() * 1.35),
                "lateral_gain_mult": 1.35,
                "lookahead_mult": 0.45,
                "max_delta": 0.035,
                "smooth_mult": 0.58,
            }
    head_err = 0.0
    lat_err = 0.0
    target_lane_id = None
    travel_lane_id = None
    steer_pass_kw = {
        "pass_fsm_phase": pass_fsm_phase or scripted_phase,
        "passing_side": passing_side,
    }
    if session is not None and ego is not None and session.map is not None:
        target_wp = _target_driving_waypoint(
            session, ego, phase, passing_side, action_semantic=action_semantic
        )
        if _ego_off_spawn_corridor(session, ego) and pass_fsm_phase in (
            "prepare_pass",
            "lane_change",
            "overtake",
            "merge_back",
        ):
            # Ego overshot outside the travel/passing corridor.
            # When the ego has rotated significantly, the spawn-axis anchor system breaks down.
            # Use CARLA direct waypoint lookup on the target lane instead.
            pw = getattr(session, "_passing_wp", None)
            tw = getattr(session, "_travel_wp", None)
            target_lane_id = int(pw.lane_id) if pw is not None else None
            target_road_id = int(pw.road_id) if pw is not None else None
            if pass_fsm_phase == "merge_back" and tw is not None:
                target_lane_id = int(tw.lane_id)
                target_road_id = int(tw.road_id)
            recovery_wp = None
            if target_lane_id is not None and session.map is not None:
                try:
                    carla = session.carla
                    ego_loc = ego.get_location()
                    # Get the nearest waypoint on the CARLA map — then walk to the target lane.
                    ego_wp = session.map.get_waypoint(ego_loc, project_to_road=True,
                                                      lane_type=carla.LaneType.Driving)
                    # Walk left/right to find the target lane.
                    side = getattr(session, "_passing_side", "right")
                    cand = ego_wp
                    for _ in range(8):
                        if int(cand.lane_id) == target_lane_id and int(cand.road_id) == target_road_id:
                            break
                        nxt_left = cand.get_left_lane()
                        nxt_right = cand.get_right_lane()
                        found = None
                        for nxt in ((nxt_right, nxt_left) if (side == "right" and pass_fsm_phase != "merge_back") else (nxt_left, nxt_right)):
                            if nxt is None or nxt.lane_type != carla.LaneType.Driving:
                                continue
                            if int(nxt.lane_id) * int(cand.lane_id) <= 0:
                                continue
                            found = nxt
                            break
                        if found is None:
                            break
                        cand = found
                    if int(cand.lane_id) == target_lane_id:
                        # Project 4m ahead along this waypoint's forward direction.
                        fwd = cand.transform.get_forward_vector()
                        ahead_loc = carla.Location(
                            float(cand.transform.location.x) + float(fwd.x) * 4.0,
                            float(cand.transform.location.y) + float(fwd.y) * 4.0,
                            float(cand.transform.location.z),
                        )
                        from types import SimpleNamespace
                        recovery_wp = SimpleNamespace(
                            lane_id=cand.lane_id, road_id=cand.road_id,
                            transform=SimpleNamespace(location=ahead_loc, rotation=cand.transform.rotation)
                        )
                except Exception:
                    pass
            if recovery_wp is not None:
                target_wp = recovery_wp
            elif pass_fsm_phase == "merge_back" and hasattr(session, "get_travel_steering_waypoint"):
                target_wp = session.get_travel_steering_waypoint(ego, lookahead_m=5.0) or target_wp
            elif hasattr(session, "get_passing_lane_steering_waypoint"):
                target_wp = session.get_passing_lane_steering_waypoint(ego, lookahead_m=5.0) or target_wp
            steer_kw = {
                "max_steer_override": min(0.14, max_steer()),
                "lateral_gain_mult": 0.95,
                "lookahead_mult": 0.5,
                "smooth_mult": 0.55,
                "max_delta": 0.03,
            }
            target = min(target, max(4.0, current * 0.5))
        pass_lateral_active = pass_fsm_phase in ("prepare_pass", "lane_change", "overtake", "merge_back")
        if (
            recovery
            and not pass_lateral_active
            and hasattr(session, "get_recovery_travel_waypoint")
        ):
            target_wp = session.get_recovery_travel_waypoint(ego) or target_wp
        ctrl_dbg_merge = getattr(session, "_last_control_debug", {}) if session is not None else {}
        if ctrl_dbg_merge and scripted_phase == "merge_back" and hasattr(session, "get_travel_steering_waypoint"):
            from perception.pass_geometry import (
                pass_finish_active,
                pass_merge_back_due,
                safe_to_merge_to_travel,
                wide_off_corridor,
            )

            if wide_off_corridor(session, ego):
                target_wp = (
                    session.get_recovery_travel_waypoint(ego)
                    or session.get_travel_steering_waypoint(ego, lookahead_m=5.0)
                    or target_wp
                )
                steer_kw = {
                    "max_steer_override": min(0.42, max_steer() * 2.0),
                    "lateral_gain_mult": 2.15,
                    "lookahead_mult": 0.3,
                    "smooth_mult": 0.42,
                    "max_delta": 0.06,
                }
            elif safe_to_merge_to_travel(session) and (
                pass_merge_back_due(session) or pass_finish_active(session)
            ):
                target_wp = session.get_travel_steering_waypoint(ego, lookahead_m=9.0) or target_wp
                steer_kw = {
                    "max_steer_override": min(0.36, max_steer() * 1.5),
                    "lateral_gain_mult": 1.5,
                    "lookahead_mult": 0.52,
                    "smooth_mult": 0.6,
                    "max_delta": 0.04,
                }
            elif abs(float(ctrl_dbg_merge.get("heading_error_deg", 0.0))) > 50.0:
                if hasattr(session, "_merge_back_steer_waypoint"):
                    target_wp = session._merge_back_steer_waypoint(ego, passing_side) or target_wp
                else:
                    target_wp = session.get_travel_steering_waypoint(ego) or target_wp
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
            session,
            ego,
            target_wp,
            recovery=recovery and not pass_lateral_active,
            **steer_kw,
            **steer_pass_kw,
        )
        # lat_err > 4.5 to avoid triggering "unstable" on a routine lane change (3.5m wide lanes).
        unstable_heading = abs(head_err) > 35.0 or abs(lat_err) > 4.5
        severe_heading = abs(head_err) > 50.0
        lc_shift_incomplete = False
        lc_width = 3.5
        if (
            scripted_phase == "lane_change"
            and session is not None
            and ego is not None
            and hasattr(session, "lateral_shift_toward_passing_m")
        ):
            lc_shift_incomplete = float(session.lateral_shift_toward_passing_m(ego)) < lc_width * 0.85
            if hasattr(session, "expected_passing_lane_width_m"):
                try:
                    lc_width = float(session.expected_passing_lane_width_m())
                    lc_shift_incomplete = float(session.lateral_shift_toward_passing_m(ego)) < lc_width * 0.85
                except Exception:
                    pass
        if (
            (unstable_heading or abs(head_err) > 90.0)
            and pass_lateral_active
            and not (on_passing_corridor and abs(head_err) < 38.0)
            and session is not None
            and hasattr(session, "_lane_change_steer_waypoint")
        ):
            if scripted_phase == "lane_change" and lc_shift_incomplete:
                mb = bool(
                    hasattr(session, "passing_lane_median_blocked")
                    and session.passing_lane_median_blocked()
                )
                steer_boost, head_err, lat_err = _steer_toward_passing_lane_corridor(
                    session,
                    ego,
                    max_steer_override=min(
                        0.14 if mb else 0.28,
                        max_steer() * lane_change_steer_cap_mult() * (1.0 if mb else 1.4),
                    ),
                    lateral_gain_mult=1.1 if mb else 1.55,
                    smooth_mult=0.62,
                    max_delta=0.04 if mb else 0.05,
                )
                pass_sign = _passing_lane_steer_sign(session)
                if pass_sign * steer_boost > 0.0 and abs(steer_boost) >= max(abs(steer) * 0.85, 0.08):
                    steer = steer_boost
            else:
                blend_wp = target_wp
                if scripted_phase == "merge_back" and hasattr(session, "_merge_back_steer_waypoint"):
                    blend_wp = session._merge_back_steer_waypoint(ego, passing_side) or blend_wp
                else:
                    blend_wp = session._lane_change_steer_waypoint(ego, passing_side) or blend_wp
                steer, head_err, lat_err = _steer_to_waypoint(
                    session,
                    ego,
                    blend_wp,
                    recovery=False,
                    max_steer_override=min(0.22, max_steer()) if severe_heading else min(0.14, max_steer()),
                    lateral_gain_mult=1.05 if severe_heading else 0.88,
                    lookahead_mult=0.42 if severe_heading else 0.55,
                    smooth_mult=0.5,
                    max_delta=0.032 if severe_heading else 0.028,
                    **steer_pass_kw,
                )
            if scripted_phase != "merge_back":
                target = min(
                    target,
                    max(3.5 if severe_heading else 4.5, current * (0.55 if severe_heading else 0.72)),
                )
        elif (
            unstable_heading
            and not pass_lateral_active
            and session is not None
            and hasattr(session, "get_recovery_travel_waypoint")
        ):
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
        overshoot_pull = (
            session is not None
            and ego is not None
            and hasattr(session, "lateral_overshoot_past_passing_m")
            and float(session.lateral_overshoot_past_passing_m(ego)) > 0.18
            and hasattr(session, "get_passing_lane_steering_waypoint")
        )
        if overshoot_pull and not pass_lateral_active:
            pass_wp = session.get_passing_lane_steering_waypoint(ego, lookahead_m=5.0)
            steer, head_err, lat_err = _steer_to_waypoint(
                session,
                ego,
                pass_wp,
                recovery=False,
                max_steer_override=min(0.1, max_steer()),
                lateral_gain_mult=0.58,
                lookahead_mult=0.58,
                smooth_mult=0.5,
                max_delta=0.02,
            )
            target = min(target, max(4.5, current * 0.65))
        if (
            scripted_phase == "lane_change"
            and session is not None
            and ego is not None
            and hasattr(session, "lateral_lane_offsets_m")
            and hasattr(session, "get_passing_lane_steering_waypoint")
        ):
            d_travel, d_pass, width = session.lateral_lane_offsets_m(ego)
            w = max(2.5, float(width))
            shift_lc_wide = (
                float(session.lateral_shift_toward_passing_m(ego))
                if hasattr(session, "lateral_shift_toward_passing_m")
                else 0.0
            )
            between = float(d_travel) > w * 0.42 and float(d_pass) > w * 0.42
            corridor_min = min(float(d_travel), float(d_pass))
            wide_lc = (
                between
                or corridor_min > w * 0.55
                or (
                    float(d_travel) > w * 0.30
                    and shift_lc_wide < w * 0.85
                    and float(d_pass) < w * 0.35
                )
            )
            if wide_lc and not on_passing_corridor:
                extreme_wide = corridor_min > w * 0.85
                cap = min(0.38 if extreme_wide else 0.32, max_steer() * lane_change_steer_cap_mult() * (1.65 if extreme_wide else 1.5))
                if (
                    extreme_wide
                    and float(d_pass) > w * 0.85
                    and float(d_travel) > w * 0.85
                    and hasattr(session, "get_recovery_travel_waypoint")
                ):
                    travel_wp = (
                        session.get_recovery_travel_waypoint(ego)
                        or session.get_travel_steering_waypoint(ego, lookahead_m=5.0)
                    )
                    if travel_wp is not None:
                        steer, head_err, lat_err = _steer_to_waypoint(
                            session,
                            ego,
                            travel_wp,
                            recovery=True,
                            max_steer_override=cap,
                            lateral_gain_mult=1.85,
                            lookahead_mult=0.34,
                            smooth_mult=0.48,
                            max_delta=0.055,
                            **steer_pass_kw,
                        )
                        session._last_steer = steer
                else:
                    steer_boost, head_err, lat_err = _steer_toward_passing_lane_corridor(
                        session,
                        ego,
                        max_steer_override=cap,
                        lateral_gain_mult=1.65 if extreme_wide else 1.55,
                        smooth_mult=0.62,
                        max_delta=0.055 if extreme_wide else 0.05,
                    )
                    pass_sign = _passing_lane_steer_sign(session)
                    if pass_sign * steer_boost > 0.0 and abs(steer_boost) >= max(abs(steer) * 0.55, 0.05):
                        steer = steer_boost
                    elif pass_sign * steer <= 0.0 and pass_sign * steer_boost > 0.0:
                        steer = steer_boost
                    elif pass_sign * steer > 0.0 and abs(steer) >= 0.08:
                        pass
                    elif abs(steer) < 0.12:
                        pass_wp = session.get_passing_lane_steering_waypoint(ego, lookahead_m=5.5)
                        steer_pp, head_err, lat_err = _steer_to_waypoint(
                            session,
                            ego,
                            pass_wp,
                            max_steer_override=min(0.28, max_steer() * 1.5),
                            lateral_gain_mult=1.65,
                            lookahead_mult=0.38,
                            smooth_mult=0.72,
                            max_delta=0.048,
                            **steer_pass_kw,
                        )
                        if abs(steer_pp) > abs(steer):
                            steer = steer_pp
        if (
            scripted_phase == "lane_change"
            and session is not None
            and ego is not None
            and hasattr(session, "signed_lateral_error_toward_passing_m")
            and abs(head_err) < 28.0
        ):
            slat = float(session.signed_lateral_error_toward_passing_m(ego))
            cap = min(0.32, max_steer() * lane_change_steer_cap_mult() * 1.5)
            w = _pass_lane_width_m(session, ego)
            _, d_pass, _w = session.lateral_lane_offsets_m(ego)
            deficit = max(w * 0.52 - slat, (w * 0.52) - (w - float(d_pass)))
            if abs(deficit) > 0.15:
                sign = _passing_lane_steer_sign(session)
                mag = min(cap, 0.08 + abs(deficit) * 0.12)
                if deficit > 0.0:
                    # Undershot: push toward passing lane
                    pull = mag * sign
                    steer = max(steer, pull) if sign > 0 else min(steer, pull)
                else:
                    # Overshot past passing lane: pull back inward
                    pull = -mag * sign
                    steer = min(steer, pull) if sign > 0 else max(steer, pull)
        if scripted_phase == "lane_change" and session is not None and ego is not None:
            sign = _passing_lane_steer_sign(session)
            w = _pass_lane_width_m(session, ego)
            slat = float(session.signed_lateral_error_toward_passing_m(ego)) if hasattr(
                session, "signed_lateral_error_toward_passing_m"
            ) else float(session.lateral_shift_toward_passing_m(ego))
            shift_lc = float(session.lateral_shift_toward_passing_m(ego)) if hasattr(
                session, "lateral_shift_toward_passing_m"
            ) else slat
            _, d_pass_lc, _ = session.lateral_lane_offsets_m(ego)
            off_passing = float(d_pass_lc) > w * 0.42
            if off_passing and float(d_pass_lc) > w * 1.25 and hasattr(
                session, "get_recovery_travel_waypoint"
            ):
                travel_wp = (
                    session.get_recovery_travel_waypoint(ego)
                    or session.get_travel_steering_waypoint(ego, lookahead_m=5.0)
                )
                if travel_wp is not None:
                    steer, head_err, lat_err = _steer_to_waypoint(
                        session,
                        ego,
                        travel_wp,
                        recovery=True,
                        max_steer_override=min(0.32, max_steer() * 1.5),
                        lateral_gain_mult=1.65,
                        lookahead_mult=0.38,
                        smooth_mult=0.48,
                        max_delta=0.05,
                        **steer_pass_kw,
                    )
                    session._last_steer = steer
            elif off_passing:
                steer_cor, head_err, lat_err = _steer_toward_passing_lane_corridor(
                    session,
                    ego,
                    max_steer_override=min(0.34, max_steer() * lane_change_steer_cap_mult() * 1.6),
                    lateral_gain_mult=1.7,
                    smooth_mult=0.58,
                    max_delta=0.055,
                )
                if abs(steer_cor) >= 0.06:
                    steer = steer_cor
                else:
                    off_frac = min(1.0, float(d_pass_lc) / max(w * 0.55, 0.1))
                    floor = min(0.32, max_steer() * lane_change_steer_cap_mult() * 1.35) * max(
                        0.55, off_frac
                    )
                    pull = floor * sign
                    if sign > 0:
                        steer = max(steer, pull)
                    else:
                        steer = min(steer, pull)
            elif shift_lc < w * 0.85:
                floor = min(0.32, max_steer() * lane_change_steer_cap_mult()) * max(
                    0.35,
                    min(1.0, (w * 0.55 - min(shift_lc, slat)) / max(w * 0.55, 0.1)),
                )
                pull = floor * sign
                if sign > 0:
                    steer = max(steer, pull)
                else:
                    steer = min(steer, pull)
            elif (
                abs(head_err) >= 28.0
                and shift_lc >= w * 0.85
                and float(d_pass_lc) < w * 0.35
                and hasattr(session, "get_travel_steering_waypoint")
            ):
                travel_wp = session.get_travel_steering_waypoint(ego, lookahead_m=6.0)
                if travel_wp is not None:
                    steer, head_err, lat_err = _steer_to_waypoint(
                        session,
                        ego,
                        travel_wp,
                        recovery=True,
                        max_steer_override=min(0.22, max_steer() * 1.2),
                        lateral_gain_mult=1.1,
                        lookahead_mult=0.55,
                        smooth_mult=0.45,
                        max_delta=0.04,
                        **steer_pass_kw,
                    )
                    target = min(target, max(3.5, current * 0.55))
        if scripted_phase == "lane_change" and session is not None and ego is not None:
            sign = _passing_lane_steer_sign(session)
            w = _pass_lane_width_m(session, ego)
            shift_lc = float(session.lateral_shift_toward_passing_m(ego)) if hasattr(
                session, "lateral_shift_toward_passing_m"
            ) else 0.0
            slat = float(session.signed_lateral_error_toward_passing_m(ego)) if hasattr(
                session, "signed_lateral_error_toward_passing_m"
            ) else shift_lc
            _, d_pass_lc, _ = session.lateral_lane_offsets_m(ego)
            if shift_lc >= w * 0.72 and float(d_pass_lc) < w * 0.38:
                steer_hold, head_err, lat_err = _steer_toward_passing_lane_corridor(
                    session,
                    ego,
                    max_steer_override=min(0.1, max_steer()),
                    lateral_gain_mult=0.8,
                    smooth_mult=0.65,
                    max_delta=0.022,
                )
                steer = steer_hold
                session._last_steer = steer
            elif shift_lc < w * 0.85 or float(d_pass_lc) > w * 0.42:
                median_blocked = bool(
                    hasattr(session, "passing_lane_median_blocked")
                    and session.passing_lane_median_blocked()
                )
                if not (median_blocked and shift_lc >= w * 0.45):
                    if float(d_pass_lc) > w * 0.42:
                        off_frac = min(1.0, float(d_pass_lc) / max(w * 0.55, 0.1))
                        min_mag = min(0.12, max_steer() * lane_change_steer_cap_mult() * 0.55) * max(
                            0.5, off_frac
                        )
                    else:
                        min_mag = min(0.12, max_steer() * lane_change_steer_cap_mult() * 0.5) * max(
                            0.45,
                            min(1.0, 1.0 - shift_lc / max(w * 0.85, 0.1)),
                        )
                    forced = min_mag * sign
                    if sign * steer <= 0.0 or abs(steer) < abs(forced) * 0.88:
                        steer = forced
                        session._last_steer = steer
        elif (
            pass_fsm_phase == "lane_change"
            and scripted_phase == "merge_back"
            and session is not None
            and ego is not None
            and hasattr(session, "lateral_shift_toward_passing_m")
        ):
            sign = _passing_lane_steer_sign(session)
            w = 3.5
            if hasattr(session, "expected_passing_lane_width_m"):
                try:
                    w = float(session.expected_passing_lane_width_m())
                except Exception:
                    pass
            shift_lc = float(session.lateral_shift_toward_passing_m(ego))
            if shift_lc < w * 0.85:
                min_mag = min(0.32, max_steer() * lane_change_steer_cap_mult()) * max(
                    0.55,
                    min(1.0, 1.0 - shift_lc / max(w * 0.85, 0.1)),
                )
                forced = min_mag * sign
                if sign * steer <= 0.0 or abs(steer) < abs(forced) * 0.88:
                    steer = forced
                    session._last_steer = steer
        elif (
            (scripted_phase == "overtake" or pass_fsm_phase == "overtake")
            and session is not None
            and ego is not None
            and hasattr(session, "lateral_shift_toward_passing_m")
        ):
            from perception.pass_geometry import axis_ahead_of_lead, wide_off_corridor

            sign = _passing_lane_steer_sign(session)
            median_blocked = bool(
                hasattr(session, "passing_lane_median_blocked")
                and session.passing_lane_median_blocked()
            )
            w = 3.5
            if hasattr(session, "expected_passing_lane_width_m"):
                try:
                    w = float(session.expected_passing_lane_width_m())
                except Exception:
                    pass
            shift_ot = float(session.lateral_shift_toward_passing_m(ego))
            ahead = axis_ahead_of_lead(session)
            d_travel_ot, d_pass_ot, _ = session.lateral_lane_offsets_m(ego)
            corridor_min_ot = min(float(d_travel_ot), float(d_pass_ot))
            overshoot_ot = (
                float(session.lateral_overshoot_past_passing_m(ego))
                if hasattr(session, "lateral_overshoot_past_passing_m")
                else max(0.0, float(d_pass_ot) - w * 0.32)
            )
            on_passing_center = float(d_pass_ot) < w * 0.36 and overshoot_ot <= w * 0.08
            still_off_passing = not corridor_ok or float(d_pass_ot) > w * 0.32
            peak_ot = float(getattr(session, "_pass_peak_shift_m", 0.0))
            latched_ot = bool(
                getattr(session, "_pass_corridor_committed", False) or peak_ot >= w * 0.45
            )
            wide_ot = wide_off_corridor(session, ego, width_frac=0.95)
            if (
                corridor_min_ot > w * 0.85
                and overshoot_ot > w * 0.25
                and hasattr(session, "get_recovery_travel_waypoint")
            ):
                travel_wp = (
                    session.get_recovery_travel_waypoint(ego)
                    or session.get_travel_steering_waypoint(ego, lookahead_m=4.0)
                )
                if travel_wp is not None:
                    steer, head_err, lat_err = _steer_to_waypoint(
                        session,
                        ego,
                        travel_wp,
                        recovery=True,
                        max_steer_override=min(0.35, max_steer() * 1.6),
                        lateral_gain_mult=1.75,
                        lookahead_mult=0.36,
                        smooth_mult=0.48,
                        max_delta=0.05,
                        **steer_pass_kw,
                    )
                    session._last_steer = steer
                    target = min(target, max(3.5, current * 0.55))
            elif on_passing_center and not ahead:
                steer_hold, head_err, lat_err = _steer_toward_passing_lane_corridor(
                    session,
                    ego,
                    max_steer_override=min(0.1, max_steer()),
                    lateral_gain_mult=0.78,
                    smooth_mult=0.68,
                    max_delta=0.022,
                )
                steer = steer_hold
                session._last_steer = steer
            elif corridor_min_ot > w * 1.25 and session is not None and ego is not None:
                pass_wp = (
                    session.get_passing_lane_steering_waypoint(ego, lookahead_m=5.0)
                    if hasattr(session, "get_passing_lane_steering_waypoint")
                    else None
                )
                if pass_wp is not None:
                    steer_map, head_err, lat_err = _steer_to_waypoint(
                        session,
                        ego,
                        pass_wp,
                        max_steer_override=min(0.14, max_steer()),
                        lateral_gain_mult=0.95,
                        lookahead_mult=0.5,
                        smooth_mult=0.55,
                        max_delta=0.03,
                        **steer_pass_kw,
                    )
                    steer = steer_map
                    session._last_steer = steer
                    target = min(target, max(4.0, current * 0.62))
            elif overshoot_ot > w * 0.12:
                steer_cor, head_err, lat_err = _steer_toward_passing_lane_corridor(
                    session,
                    ego,
                    max_steer_override=min(0.22, max_steer() * 1.2),
                    lateral_gain_mult=1.25,
                    smooth_mult=0.62,
                    max_delta=0.04,
                )
                steer = steer_cor
                session._last_steer = steer
            elif (
                latched_ot
                and wide_ot
                and float(d_pass_ot) > w * 1.05
                and (ahead or axis_ahead_of_lead(session, margin_m=1.0))
                and hasattr(session, "get_recovery_travel_waypoint")
            ):
                travel_wp = (
                    session.get_recovery_travel_waypoint(ego)
                    or session.get_travel_steering_waypoint(ego, lookahead_m=5.0)
                )
                if travel_wp is not None:
                    steer, head_err, lat_err = _steer_to_waypoint(
                        session,
                        ego,
                        travel_wp,
                        recovery=True,
                        max_steer_override=min(0.38, max_steer() * 1.8),
                        lateral_gain_mult=1.85,
                        lookahead_mult=0.34,
                        smooth_mult=0.45,
                        max_delta=0.055,
                        **steer_pass_kw,
                    )
                    session._last_steer = steer
                    if pass_fsm_phase == "overtake":
                        pass_fsm_phase = "merge_back"
                        scripted_phase = "merge_back"
            elif (
                not ahead
                and still_off_passing
                and float(d_pass_ot) < w * 0.95
                and overshoot_ot <= w * 0.12
            ):
                steer_cor, head_err, lat_err = _steer_toward_passing_lane_corridor(
                    session,
                    ego,
                    max_steer_override=min(0.28, max_steer() * lane_change_steer_cap_mult() * 1.4),
                    lateral_gain_mult=1.5,
                    smooth_mult=0.58,
                    max_delta=0.048,
                )
                if abs(steer_cor) >= 0.05:
                    steer = steer_cor
                    session._last_steer = steer
                elif not (wide_ot and float(d_pass_ot) > w * 0.55):
                    off_frac = min(1.0, float(d_pass_ot) / max(w * 0.55, 0.1))
                    cap = 0.12 if median_blocked else 0.22
                    min_mag = min(cap, max_steer() * lane_change_steer_cap_mult() * 0.85) * max(
                        0.45, off_frac
                    )
                    forced = min_mag * sign
                    if sign * steer <= 0.0 or abs(steer) < abs(forced) * 0.88:
                        steer = forced
                        session._last_steer = steer
            elif (
                scripted_phase in ("merge_back", "overtake")
                and wide_off_corridor(session, ego, width_frac=1.0)
                and (
                    shift_ot >= w * 0.85
                    or peak_ot >= w * 0.45
                    or float(d_pass_ot) > w * 0.55
                )
                and (ahead or axis_ahead_of_lead(session, margin_m=0.5))
                and hasattr(session, "get_recovery_travel_waypoint")
            ):
                travel_wp = (
                    session.get_recovery_travel_waypoint(ego)
                    or session.get_travel_steering_waypoint(ego, lookahead_m=5.0)
                )
                if travel_wp is not None:
                    steer, head_err, lat_err = _steer_to_waypoint(
                        session,
                        ego,
                        travel_wp,
                        recovery=True,
                        max_steer_override=min(0.42, max_steer() * 2.0),
                        lateral_gain_mult=2.0,
                        lookahead_mult=0.32,
                        smooth_mult=0.42,
                        max_delta=0.06,
                        **steer_pass_kw,
                    )
                    session._last_steer = steer
        steer = _clamp_pass_maneuver_steer(
            session,
            ego,
            steer,
            pass_fsm_phase=pass_fsm_phase,
            scripted_phase=scripted_phase,
        )
        if (
            pass_fsm_phase in ("prepare_pass", "lane_change", "overtake")
            and session is not None
            and ego is not None
            and hasattr(session, "lateral_shift_toward_passing_m")
        ):
            w = _pass_lane_width_m(session, ego)
            shift_cap = float(session.lateral_shift_toward_passing_m(ego))
            if shift_cap > w * 0.85:
                sign = _passing_lane_steer_sign(session)
                steer = -sign * min(0.12, max(0.06, abs(steer) * 0.65))
                target = min(target, max(3.5, current * 0.45))
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
        pass_fsm_phase=pass_fsm_phase,
    )
    if not on_passing_corridor and scripted_phase != "merge_back":
        if abs(head_err) > 30.0 or abs(lat_err) > 2.5:
            ctrl.throttle = min(ctrl.throttle, 0.18)
            ctrl.brake = max(ctrl.brake, 0.12 if current > 4.0 else 0.0)
        if scripted_phase in ("lane_change", "overtake") and abs(head_err) > 22.0:
            ctrl.throttle = min(ctrl.throttle, 0.28)
            ctrl.brake = max(ctrl.brake, 0.08 if current > 3.0 else 0.0)
        if scripted_phase in ("lane_change", "overtake") and abs(head_err) > 45.0:
            along = lead_along if lead_along is not None else front
            still_behind_lead = along is not None and float(along) > critical_gap_m() + 2.0
            if scripted_phase == "overtake" and still_behind_lead and not clear_of_lead:
                ctrl.throttle = max(min(ctrl.throttle, 0.42), 0.28)
                ctrl.brake = min(ctrl.brake, 0.05)
            else:
                ctrl.throttle = min(ctrl.throttle, 0.12)
                ctrl.brake = max(ctrl.brake, 0.15 if current > 2.5 else 0.05)
    if action_semantic == "follow_lead" and (abs(head_err) > 20.0 or abs(lat_err) > 2.0):
        ctrl.brake = min(ctrl.brake, 0.25)
        ctrl.throttle = min(ctrl.throttle, 0.35)
    if scripted_phase == "lane_change" and current < 6.5 and pass_maneuver_started:
        ctrl.throttle = max(ctrl.throttle, 0.42)
        ctrl.brake = min(ctrl.brake, 0.05)
    if scripted_phase == "overtake" and current < 7.0 and not on_passing_corridor:
        wide_pass = False
        if session is not None and ego is not None and hasattr(session, "lateral_lane_offsets_m"):
            _dt, d_pass, width = session.lateral_lane_offsets_m(ego)
            w = max(2.5, float(width))
            wide_pass = float(d_pass) > w * 0.42 or float(_dt) > w * 0.55
        if not wide_pass:
            ctrl.throttle = max(ctrl.throttle, 0.32)
            ctrl.brake = min(ctrl.brake, 0.04)
    if scripted_phase == "merge_back" and action_semantic == "pass":
        ctrl.throttle = max(ctrl.throttle, 0.48)
        ctrl.brake = min(ctrl.brake, 0.04)
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
        trace_head = float(head_err)
        travel_head = None
        passing_head = None
        if ego is not None and hasattr(session, "heading_error_to_travel_lane_deg"):
            try:
                travel_head = float(session.heading_error_to_travel_lane_deg(ego))
            except Exception:
                travel_head = None
        if ego is not None and hasattr(session, "heading_error_to_passing_lane_deg"):
            try:
                passing_head = float(session.heading_error_to_passing_lane_deg(ego))
            except Exception:
                passing_head = None
        if pass_fsm_phase == "merge_back" and travel_head is not None:
            trace_head = travel_head
        elif scripted_phase in ("lane_change", "overtake") and not corridor_ok and passing_head is not None:
            trace_head = passing_head
        session._last_control_debug = {
            "action_semantic": action_semantic,
            "heading_error_deg": trace_head,
            "steer_heading_error_deg": float(head_err),
            "travel_heading_error_deg": travel_head,
            "passing_heading_error_deg": passing_head,
            "corridor_ok": corridor_ok,
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

    pass_st = get_pass_control_state(session)
    requested_action = action
    committed_pass = action == "pass"

    def _fsm_pass_active() -> bool:
        latched = bool(getattr(session, "_pass_corridor_committed", False)) or float(
            getattr(session, "_pass_peak_shift_m", 0.0)
        ) >= (session.expected_passing_lane_width_m() * 0.45)
        return (
            committed_pass
            or pass_st.active
            or pass_st.maneuver_started
            or pass_st.phase in ("lane_change", "overtake", "merge_back", "abort")
            or latched
        )

    pass_st, effective_action, fsm_control_failure = pass_control_tick(
        session,
        ego,
        requested_action=requested_action,
        pass_in_progress=_fsm_pass_active(),
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
        if hasattr(session, "refresh_lane_marking_hint"):
            try:
                session.refresh_lane_marking_hint()
            except Exception:
                pass
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
            pass_in_progress=_fsm_pass_active(),
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
        d_travel_m = 0.0
        d_pass_m = 0.0
        if hasattr(session, "lateral_lane_offsets_m"):
            try:
                d_travel_m, d_pass_m, width = session.lateral_lane_offsets_m(ego)
            except Exception:
                pass
        from perception.pass_geometry import (
            beside_or_ahead_of_lead,
            pass_committed_latched,
            pass_finish_active,
            pass_merge_back_due,
        )

        pass_latched = pass_committed_latched(session)
        finish_latched = bool(getattr(session, "_pass_finish_latch", False))
        finishing_overtake = (
            pass_finish_active(session)
            or pass_merge_back_due(session)
            or (finish_latched and pass_latched)
            or (pass_latched and beside_or_ahead_of_lead(session, margin_m=0.5))
        )
        if pass_latched and beside_or_ahead_of_lead(session, margin_m=0.5) and pass_st.phase != "merge_back":
            pass_st.active = True
            pass_st.phase = "merge_back"
            pass_st.abort_reason = ""
            pass_st.target_lane_source = "return_lane"
        if pass_latched and hasattr(session, "lateral_lane_offsets_m"):
            try:
                d_travel_m, d_pass_m, width = session.lateral_lane_offsets_m(ego)
                lane_dist = min(float(d_travel_m), float(d_pass_m))
            except Exception:
                pass
        if finishing_overtake and pass_st.phase == "abort":
            pass_st.active = True
            pass_st.phase = "overtake"
            pass_st.abort_reason = ""

        depart_fail_m = lane_departure_fail_m()
        if action_semantic == "follow_lead" and not pass_st.active:
            depart_fail_m = max(depart_fail_m, width * 1.35, 5.5)
        if pass_st.active and pass_st.phase in ("lane_change", "overtake", "merge_back"):
            depart_fail_m = max(depart_fail_m, width * 2.15, 7.5)
        if pass_st.active and pass_st.phase == "merge_back":
            depart_fail_m = max(depart_fail_m, width * 2.75, 9.5)
        if finishing_overtake:
            depart_fail_m = max(depart_fail_m, width * 4.5, 16.0)

        ctrl_dbg_pre = getattr(session, "_last_control_debug", {}) if session is not None else {}
        head_abs = abs(float(ctrl_dbg_pre.get("heading_error_deg", 0.0)))
        between_lanes_geom = d_travel_m > width * 0.72 and d_pass_m > width * 0.72
        depart_trigger = lane_dist > depart_fail_m
        if pass_st.phase != "merge_back":
            depart_trigger = depart_trigger or (
                lane_dist > lane_departure_warn_m() and head_abs > 35.0
            )
        if finishing_overtake:
            depart_trigger = between_lanes_geom and lane_dist > width * 2.2
        if (
            depart_trigger
            and lane_dist > lane_departure_warn_m()
            and not finishing_overtake
            and not (pass_latched and beside_or_ahead_of_lead(session, margin_m=0.5))
        ):
            use_passing_recovery = pass_st.phase in ("prepare_pass", "lane_change") and not pass_latched
            if use_passing_recovery:
                rec_steer, _, _ = _steer_toward_passing_lane_corridor(
                    session,
                    ego,
                    max_steer_override=min(0.3, max_steer() * lane_change_steer_cap_mult()),
                    lateral_gain_mult=1.45,
                    smooth_mult=0.65,
                    max_delta=0.045,
                )
            else:
                recovery_wp = (
                    session.get_recovery_travel_waypoint(ego)
                    if hasattr(session, "get_recovery_travel_waypoint")
                    else None
                )
                rec_steer = 0.0
                if recovery_wp is not None:
                    rec_steer, _, _ = _steer_to_waypoint(
                        session,
                        ego,
                        recovery_wp,
                        recovery=True,
                        max_steer_override=min(0.12, max_steer()),
                        max_delta=0.025,
                    )
            if abs(rec_steer) > 0.02 or not use_passing_recovery:
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

        if (
            lane_dist > depart_fail_m
            and not finishing_overtake
            and not pass_merge_back_due(session)
            and pass_st.phase != "merge_back"
            and not (pass_latched and beside_or_ahead_of_lead(session, margin_m=0.5))
        ):
            in_pass_maneuver = pass_st.active and pass_st.phase in (
                "prepare_pass",
                "lane_change",
                "overtake",
                "merge_back",
            )
            if in_pass_maneuver:
                control_failure = True
                lane_departure = True
                from perception.pass_control_fsm import abort_pass

                abort_pass(session, f"lane_departure_{lane_dist:.1f}m")
                pass_st = get_pass_control_state(session)
                pass_abort_reason = pass_st.abort_reason
                action = "wait"
                action_semantic = "follow_lead"
            elif action_semantic == "follow_lead" and lane_dist > depart_fail_m * 1.15:
                rec_wp = (
                    session.get_recovery_travel_waypoint(ego)
                    if hasattr(session, "get_recovery_travel_waypoint")
                    else None
                )
                if rec_wp is not None:
                    rec_steer, _, _ = _steer_to_waypoint(
                        session,
                        ego,
                        rec_wp,
                        recovery=True,
                        max_steer_override=min(0.14, max_steer()),
                        max_delta=0.035,
                    )
                    rec_ctrl = carla.VehicleControl(
                        throttle=0.08,
                        brake=0.25 if v > 3.0 else 0.12,
                        steer=rec_steer,
                        hand_brake=False,
                    )
                    ego.apply_control(rec_ctrl)
                    last_ctrl = rec_ctrl
                    session.tick()
        elif pass_st.phase == "abort" and action_semantic == "follow_lead":
            action = "wait"
            action_semantic = "follow_lead"

        along_lead = front
        if hasattr(session, "lead_longitudinal_gap_m"):
            try:
                along_lead = float(session.lead_longitudinal_gap_m())
            except Exception:
                pass
        critical_close = along_lead < critical_gap_m() and not clear_of_lead
        corridor_lost = between_lanes_geom or lane_dist > width * 2.05
        if pass_st.active and pass_st.phase in ("lane_change", "overtake", "merge_back") and (
            pass_latched or pass_st.maneuver_started
        ):
            corridor_lost = False
        if pass_merge_back_due(session):
            corridor_lost = False
            critical_close = False
        if finishing_overtake:
            corridor_lost = False
            critical_close = False

        between_lanes_abort = (
            pass_st.phase == "abort"
            and lane_dist > width * 1.35
            and hasattr(session, "lateral_lane_offsets_m")
            and not pass_finish_active(session)
            and not pass_merge_back_due(session)
        )
        needs_merge_back = pass_merge_back_due(session)
        if not needs_merge_back and pass_st.phase == "merge_back":
            needs_merge_back = pass_finish_active(session) and pass_latched
        if (
            pass_st.phase == "lane_change"
            and session is not None
            and hasattr(session, "lateral_shift_toward_passing_m")
        ):
            w_lc = 3.5
            if hasattr(session, "expected_passing_lane_width_m"):
                try:
                    w_lc = float(session.expected_passing_lane_width_m())
                except Exception:
                    pass
            shift_lc = float(session.lateral_shift_toward_passing_m(ego))
            if shift_lc < w_lc * 0.85:
                needs_merge_back = False
        recovery = (
            (lane_dist > lane_departure_warn_m() and not finishing_overtake)
            or (pass_st.phase == "abort" and not between_lanes_abort)
            or critical_close
            or corridor_lost
        )
        if finishing_overtake and (d_pass_m > width * 0.55 or d_travel_m > width * 0.55):
            recovery = False
        if between_lanes_abort:
            recovery = False
        if needs_merge_back:
            recovery = False
            corridor_lost = False
            critical_close = False
            between_lanes_abort = False

        fsm_scripted = None
        head_abs_pre = abs(float(ctrl_dbg_pre.get("heading_error_deg", 0.0)))
        steer_merge_back = (
            needs_merge_back
            or pass_st.phase == "merge_back"
            or (
                pass_st.phase == "abort"
                and pass_latched
                and action_semantic in ("follow_lead", "pass")
            )
        )
        if steer_merge_back:
            fsm_scripted = "merge_back"
        elif pass_st.active:
            from perception.pass_control_fsm import _scripted_phase_for_fsm

            fsm_scripted = _scripted_phase_for_fsm(pass_st.phase)

        if critical_close or corridor_lost:
            import carla

            stop = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0, hand_brake=False)
            ego.apply_control(stop)
            last_ctrl = stop
            session.tick()
            speeds.append(_speed_mps(ego.get_velocity()))
            continue
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
            recovery=recovery or (episode_step == 1 and len(speeds) <= 3),
            scripted_phase=fsm_scripted,
            pass_fsm_phase=pass_st.phase if pass_st.active or pass_st.phase == "abort" else None,
            pass_maneuver_started=pass_st.maneuver_started,
        )
        if steer_merge_back and action == "pass":
            corridor_w = 3.5
            if hasattr(session, "expected_passing_lane_width_m"):
                try:
                    corridor_w = float(session.expected_passing_lane_width_m())
                except Exception:
                    pass
            from perception.pass_geometry import lateral_corridor_offset_m

            wide = lateral_corridor_offset_m(session, ego) > corridor_w * 0.55
            if not wide:
                last_ctrl.throttle = max(float(last_ctrl.throttle), 0.48)
                last_ctrl.brake = min(float(last_ctrl.brake), 0.03)
            else:
                last_ctrl.throttle = min(float(last_ctrl.throttle), 0.35)
                last_ctrl.brake = max(float(last_ctrl.brake), 0.08 if v > 4.0 else 0.02)
        elif steer_merge_back and action_semantic == "follow_lead" and pass_latched:
            last_ctrl.throttle = max(float(last_ctrl.throttle), 0.42)
            last_ctrl.brake = min(float(last_ctrl.brake), 0.05)
        elif (
            along_lead < safe_follow_m()
            and not clear_of_lead
            and pass_st.phase not in ("lane_change", "overtake", "merge_back")
        ):
            last_ctrl.throttle = min(float(last_ctrl.throttle), 0.15)
            last_ctrl.brake = max(float(last_ctrl.brake), 0.25 if v > 2.0 else 0.1)
        elif (
            pass_st.phase == "overtake"
            and not clear_of_lead
            and pass_latched
        ):
            last_ctrl.throttle = max(float(last_ctrl.throttle), 0.52)
            last_ctrl.brake = min(float(last_ctrl.brake), 0.04)
        if episode_step == 1 and len(speeds) <= 4 and action_semantic != "pass":
            last_ctrl.throttle = min(float(last_ctrl.throttle), 0.22)
            last_ctrl.brake = max(float(last_ctrl.brake), 0.08 if v > 1.5 else 0.0)
        elif episode_step == 1 and action_semantic == "pass" and len(speeds) <= 6:
            last_ctrl.throttle = max(float(last_ctrl.throttle), 0.38)
            last_ctrl.brake = min(float(last_ctrl.brake), 0.04)
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
    _sync_pass_fsm_for_merge(session, ego, pass_st)
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

    from perception.pass_geometry import axis_ahead_of_lead, pass_finish_active

    collision, collision_detail = session.check_actor_proximity(threshold_m=4.2)
    collision_source = ""
    if collision:
        collision_source = "carla_proximity"
        if in_spawn_grace:
            collision = False
            collision_detail = ""
            collision_source = "ignored_spawn_grace"
        elif collision_detail.startswith("lead_") and (
            pass_finish_active(session)
            or pass_st.phase == "merge_back"
            or axis_ahead_of_lead(session)
        ):
            collision = False
            collision_detail = ""
            collision_source = ""
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
        "duration_s": round(float(duration_s), 3),
        "control_ticks": n_ticks,
        "fixed_delta_s": round(float(delta), 4),
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
            "travel_heading_error_deg": ctrl_dbg.get("travel_heading_error_deg"),
            "passing_heading_error_deg": ctrl_dbg.get("passing_heading_error_deg"),
            "corridor_ok": ctrl_dbg.get("corridor_ok"),
            "steer": round(float(ctrl_dbg.get("steer", feedback["steer"])), 3),
            "throttle": round(float(ctrl_dbg.get("throttle", feedback["throttle"])), 3),
            "brake": round(float(ctrl_dbg.get("brake", feedback["brake"])), 3),
            "ego_speed_mps": round(float(ctrl_dbg.get("ego_speed_mps", measured_speed)), 3),
        }
    )
    feedback.update(fsm_diag)
    if session is not None:
        if hasattr(session, "lead_longitudinal_gap_m"):
            try:
                feedback["lead_longitudinal_gap_m"] = round(
                    float(session.lead_longitudinal_gap_m()), 2
                )
            except Exception:
                pass
        snap = (
            session.pass_longitudinal_snapshot()
            if hasattr(session, "pass_longitudinal_snapshot")
            else {}
        )
        if snap:
            feedback["pass_longitudinal"] = snap
        hint = getattr(session, "_lane_marking_hint", {}) or {}
        feedback["lane_marking_center_under_ego"] = bool(hint.get("center_line_under_ego"))
        feedback["passing_lane_commit_alpha"] = round(
            float(session.passing_lane_commit_alpha(ego))
            if hasattr(session, "passing_lane_commit_alpha") and ego is not None
            else 0.0,
            3,
        )
        if hint:
            feedback["lane_marking_center_frac"] = hint.get("center_line_frac")
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
        from perception.pass_geometry import axis_ahead_of_lead, pass_finish_active

        gaps_3d = session.measure_actor_gaps_3d()
        lead_signed = session.signed_gap_from_ego("lead") if hasattr(session, "signed_gap_from_ego") else None
        rear_signed = session.signed_gap_from_ego("rear") if hasattr(session, "signed_gap_from_ego") else None
        on_signed = session.signed_gap_from_ego("oncoming") if hasattr(session, "signed_gap_from_ego") else None
        pass_exempt = pass_finish_active(session) or axis_ahead_of_lead(session)

        # In CARLA mode, never infer overlap from missing projection.
        if (
            not pass_exempt
            and ego_lane == 0
            and not passed
            and lead_signed is not None
            and gaps_3d.get("front", 999.0) < 12.0
        ):
            if abs(float(lead_signed)) < 4.0:
                return True, "lead_proximity_logical"
        if (
            not pass_exempt
            and ego_lane == 0
            and rear_signed is not None
            and gaps_3d.get("rear", 999.0) < 12.0
        ):
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
