"""Scripted CARLA pass maneuver for corridor validation, smoke tests, and hero demo."""

from __future__ import annotations



import math

from dataclasses import dataclass, field

from typing import Any, Callable, Dict, List, Optional, Tuple



from autopass.carla_tuning import (

    corridor_merge_horizon_m,

    lane_departure_fail_m,

    lane_departure_warn_m,

    merge_clear_m,

    pass_lateral_min_m,

)



SCRIPTED_PASS_SEQUENCE: Tuple[str, ...] = ("follow", "lane_change", "overtake", "merge_back", "done")



DEFAULT_PHASE_MAX_S = {

    "follow": 6.0,

    "lane_change": 12.0,

    "overtake": 14.0,

    "merge_back": 10.0,

}

# Stay on passing lane long enough to pass lead before merge-back steering.
MIN_OVERTAKE_ELAPSED_S = 2.5
# Lateral shift onto passing lane before leaving lane_change (map lane_id lags physics).
PASSING_LANE_SHIFT_MIN_M = 2.0
# Signed longitudinal margin (travel axis) before merge-back — hero pass uses 8–12 m.
MERGE_CLEARANCE_M = 10.0
# Consecutive ticks required for phase transitions / pass completion.
PHASE_STABLE_TICKS = 5
LANE_CENTER_STABLE_M = 0.55
MIN_LANE_CHANGE_SPEED_MPS = 2.0
MERGE_START_STABLE_TICKS = 3

DEFAULT_TOTAL_MAX_S = 45.0

MIN_EDGE_CLEARANCE_M = 0.85





@dataclass

class PassStepRecord:

    step: int

    scripted_phase: str

    control_action: str

    pass_phase: str

    ego_lane_id: Optional[int] = None

    ego_road_id: Optional[int] = None

    target_lane_id: Optional[int] = None

    target_road_id: Optional[int] = None

    travel_lane_id: Optional[int] = None

    travel_road_id: Optional[int] = None

    lateral_error_m: float = 0.0

    lane_center_dist_m: float = 0.0

    edge_clearance_m: Optional[float] = None

    lead_gap_m: float = 999.0

    rear_gap_m: float = 999.0

    speed_mps: float = 0.0

    steer: float = 0.0

    ok: bool = True

    note: str = ""

    pass_started: bool = False

    pass_in_progress: bool = False

    monitor_ok: bool = True

    abort_reason: str = ""

    pass_completed: bool = False

    cleared_lead: bool = False

    ego_s_m: Optional[float] = None

    lead_s_m: Optional[float] = None

    rear_s_m: Optional[float] = None





@dataclass

class PassManeuverResult:

    ok: bool

    issues: List[str] = field(default_factory=list)

    steps: List[PassStepRecord] = field(default_factory=list)

    max_lane_center_m: float = 0.0

    min_edge_clearance_m: float = 999.0

    merged_back: bool = False

    pass_lane_used: bool = False

    pass_complete: bool = False

    pass_attempts: int = 0

    oscillation_count: int = 0

    final_world: Any = None





@dataclass

class NoPassFollowResult:

    ok: bool

    issues: List[str] = field(default_factory=list)

    steps: List[PassStepRecord] = field(default_factory=list)

    final_world: Any = None





def _speed_mps(velocity) -> float:

    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)





def estimate_edge_clearance_m(session, ego_wp) -> Optional[float]:

    """Approximate distance from lane center to nearest non-driving boundary."""

    if ego_wp is None or session is None or session.carla is None:

        return None

    carla = session.carla

    try:

        loc = ego_wp.transform.location

        width = float(getattr(ego_wp, "lane_width", 3.5) or 3.5)

        half = width * 0.5

        best = half

        for getter in (ego_wp.get_left_lane, ego_wp.get_right_lane):

            try:

                adj = getter()

            except Exception:

                adj = None

            if adj is None:

                best = max(best, half + 0.8)

                continue

            if adj.lane_type != carla.LaneType.Driving:

                d = loc.distance(adj.transform.location)

                best = min(best, max(0.2, d))

        return best

    except Exception:

        return None





def action_for_scripted_phase(phase: str) -> str:

    if phase in ("follow", "done"):

        return "wait"

    return "pass"





def resolve_scripted_pass_phase(

    scripted_phase: str,

    *,

    ego_lane: int,

    clear_of_lead: bool,

    front_gap_m: float,

) -> str:

    from perception.carla_control import resolve_pass_phase



    action = action_for_scripted_phase(scripted_phase)

    if scripted_phase == "merge_back":

        return "merge"

    if scripted_phase == "lane_change":

        return "lane_change"

    if scripted_phase == "overtake":

        return "overtake"

    if scripted_phase == "done":

        return "cruise"

    return resolve_pass_phase(action, ego_lane=ego_lane, clear_of_lead=clear_of_lead, front_gap_m=front_gap_m)





def _on_passing_lane(session, ego_wp) -> bool:

    if ego_wp is None or session._passing_wp is None:

        return False

    return ego_wp.lane_id == session._passing_wp.lane_id and ego_wp.road_id == session._passing_wp.road_id





def _on_travel_lane(session, ego_wp, travel_lane, travel_road) -> bool:

    if ego_wp is None:

        return False

    return ego_wp.lane_id == travel_lane and ego_wp.road_id == travel_road


def _off_curated_corridor(session, ego_wp, travel_road_id: int) -> bool:
    pw = session._passing_wp
    if ego_wp is None or pw is None:
        return False
    return int(ego_wp.road_id) not in (int(travel_road_id), int(pw.road_id))


def merge_clearance_m() -> float:
    from autopass.carla_tuning import merge_clear_m

    return max(MERGE_CLEARANCE_M, merge_clear_m())


def _cleared_lead(session, fallback_clear: bool) -> bool:
    if hasattr(session, "ego_cleared_lead"):
        return bool(session.ego_cleared_lead(merge_clearance_m()))
    return fallback_clear


@dataclass
class PhaseStability:
    on_passing_lane_ticks: int = 0
    lane_change_ready_ticks: int = 0
    cleared_lead_ticks: int = 0
    travel_lane_ticks: int = 0
    merge_centered_ticks: int = 0


def pass_maneuver_status(
    session,
    *,
    scripted_phase: str,
    ego_wp,
    lane_center_m: float,
    cleared_lead: bool,
    travel_lane_id: int,
    travel_road_id: int,
    monitor_ok: bool = True,
    abort_reason: str = "",
    pass_attempts: int = 0,
) -> Dict[str, Any]:
    long_snap = (
        session.pass_longitudinal_snapshot()
        if hasattr(session, "pass_longitudinal_snapshot")
        else {}
    )
    pw = session._passing_wp
    on_travel = _on_travel_lane(session, ego_wp, travel_lane_id, travel_road_id)
    in_progress = scripted_phase in ("lane_change", "overtake", "merge_back")
    return {
        "pass_started": pass_attempts >= 1 and scripted_phase != "follow",
        "pass_in_progress": in_progress,
        "phase": scripted_phase,
        "monitor_ok": monitor_ok,
        "abort_reason": abort_reason,
        "pass_completed": scripted_phase == "done" and on_travel and cleared_lead,
        "cleared_lead": cleared_lead,
        "ego_s_m": long_snap.get("ego_s_m"),
        "lead_s_m": long_snap.get("lead_s_m"),
        "rear_s_m": long_snap.get("rear_s_m"),
        "ego_road_id": getattr(ego_wp, "road_id", None),
        "target_road_id": travel_road_id,
        "ego_lane_id": getattr(ego_wp, "lane_id", None),
        "target_lane_id": travel_lane_id if scripted_phase == "merge_back" else getattr(pw, "lane_id", None),
        "passing_lane_id": getattr(pw, "lane_id", None),
        "passing_road_id": getattr(pw, "road_id", None),
    }





def is_pass_maneuver_complete(
    session,
    *,
    ego_wp,
    ego_lane: int,
    clear_of_lead: bool,
    travel_lane_id: int,
    travel_road_id: int,
    lane_center_m: float = 0.0,
    travel_stable_ticks: int = 0,
    min_stable_ticks: int = 0,
) -> bool:
    """Pass is complete when ego cleared lead and is stably centered on the travel lane."""
    if not clear_of_lead:
        return False
    if ego_wp is not None and _off_curated_corridor(session, ego_wp, travel_road_id):
        return False
    if not _on_travel_lane(session, ego_wp, travel_lane_id, travel_road_id):
        return False
    if min_stable_ticks > 0 and travel_stable_ticks < min_stable_ticks:
        return False
    if min_stable_ticks > 0 and lane_center_m > LANE_CENTER_STABLE_M:
        return False
    return True





def _advance_scripted_phase(
    phase: str,
    *,
    session,
    ego,
    ego_wp,
    ego_lane: int,
    clear_of_lead: bool,
    front_gap_m: float,
    travel_lane_id: int,
    travel_road_id: int,
    phase_elapsed_s: float,
    phase_max_s: float,
    lane_center_m: float,
    speed_mps: float,
    stability: PhaseStability,
) -> str:
    cleared = _cleared_lead(session, clear_of_lead)
    on_corridor = ego_wp is None or not _off_curated_corridor(session, ego_wp, travel_road_id)

    if phase == "follow":
        if front_gap_m >= pass_lateral_min_m() * 0.85 or phase_elapsed_s >= phase_max_s:
            return "lane_change"
        return phase

    if phase == "lane_change":
        shift = 0.0
        if ego is not None and hasattr(session, "lateral_shift_toward_passing_m"):
            shift = session.lateral_shift_toward_passing_m(ego)
        on_pass = _on_passing_lane(session, ego_wp)
        lane_ready = (
            on_corridor
            and speed_mps >= MIN_LANE_CHANGE_SPEED_MPS
            and (
                (on_pass and lane_center_m <= LANE_CENTER_STABLE_M)
                or (shift >= PASSING_LANE_SHIFT_MIN_M and lane_center_m <= LANE_CENTER_STABLE_M * 1.5)
            )
        )
        if lane_ready:
            stability.lane_change_ready_ticks += 1
        else:
            stability.lane_change_ready_ticks = 0
        if stability.lane_change_ready_ticks >= PHASE_STABLE_TICKS:
            return "overtake"
        if phase_elapsed_s >= phase_max_s and on_pass:
            return "overtake"
        return phase

    if phase == "overtake":
        if phase_elapsed_s < MIN_OVERTAKE_ELAPSED_S:
            stability.cleared_lead_ticks = 0
            return phase
        if cleared and on_corridor:
            stability.cleared_lead_ticks += 1
        else:
            stability.cleared_lead_ticks = 0
        if stability.cleared_lead_ticks >= MERGE_START_STABLE_TICKS:
            return "merge_back"
        merge_horizon = corridor_merge_horizon_m()
        corridor_ending = False
        if ego is not None and hasattr(session, "approaching_corridor_end"):
            corridor_ending = session.approaching_corridor_end(ego, min_horizon_m=merge_horizon)
        pw = session._passing_wp
        if pw is not None and ego is not None:
            left = session.remaining_lane_horizon_m(ego, pw.lane_id, pw.road_id)
            corridor_ending = corridor_ending or left < max(10.0, merge_horizon - 6.0)
        if corridor_ending and cleared and phase_elapsed_s >= MIN_OVERTAKE_ELAPSED_S:
            return "merge_back"
        if phase_elapsed_s >= phase_max_s and cleared:
            return "merge_back"
        return phase

    if phase == "merge_back":
        on_travel = _on_travel_lane(session, ego_wp, travel_lane_id, travel_road_id)
        if (
            on_travel
            and on_corridor
            and cleared
            and lane_center_m <= LANE_CENTER_STABLE_M
        ):
            stability.merge_centered_ticks += 1
        else:
            stability.merge_centered_ticks = 0
        if (
            is_pass_maneuver_complete(
                session,
                ego_wp=ego_wp,
                ego_lane=ego_lane,
                clear_of_lead=cleared,
                travel_lane_id=travel_lane_id,
                travel_road_id=travel_road_id,
                lane_center_m=lane_center_m,
                travel_stable_ticks=stability.merge_centered_ticks,
                min_stable_ticks=PHASE_STABLE_TICKS,
            )
            or phase_elapsed_s >= phase_max_s
        ):
            return "done"
        return phase

    return phase





def _step_diagnostics(

    session,

    spec,

    world,

    ego,

    *,

    scripted: str,

    step_i: int,

    lane_fail: float,

    min_edge: float,

) -> PassStepRecord:

    from perception.carla_control import build_vehicle_control

    from perception.carla_lane_keep import heading_error_deg, lateral_error_m



    delta = session.fixed_delta_seconds
    clearance_m = merge_clearance_m()

    session.update_route_cursor(ego)

    session.tick_npcs_kinematic(spec, delta)

    gaps = session.measure_actor_gaps_3d()

    front = gaps.get("front", 999.0)

    rear = gaps.get("rear", 999.0)

    clear_of_lead = _cleared_lead(session, session.ego_clear_of_lead(clearance_m))

    ego_lane = session.infer_ego_lane_index()

    v = _speed_mps(ego.get_velocity())

    lane_dist = session.ego_lane_center_distance_m(ego, phase=scripted)



    try:

        ego_wp = session.map.get_waypoint(ego.get_location(), project_to_road=True)

    except Exception:

        ego_wp = None



    edge = estimate_edge_clearance_m(session, ego_wp)

    pass_phase = resolve_scripted_pass_phase(

        scripted,

        ego_lane=ego_lane,

        clear_of_lead=clear_of_lead,

        front_gap_m=front,

    )

    passing_side = getattr(session, "_passing_side", "left") or "left"
    target_wp = session.get_steering_waypoint(ego, pass_phase, passing_side)

    ego_tf = ego.get_transform()

    tgt_loc = target_wp.transform.location if target_wp else ego_tf.location

    lat = lateral_error_m(ego_tf.location, ego_tf.rotation.yaw, tgt_loc)



    travel_lane = session._travel_wp.lane_id if session._travel_wp else None

    travel_road = session._travel_wp.road_id if session._travel_wp else None



    recovery = lane_dist > lane_departure_warn_m()
    if ego_wp is not None and travel_road is not None and session._passing_wp is not None:
        allowed_roads = {travel_road, session._passing_wp.road_id}
        if ego_wp.road_id not in allowed_roads:
            recovery = True

    action = action_for_scripted_phase(scripted)

    ctrl = build_vehicle_control(
        action,
        world=world,
        spec=spec,
        target_speed_mps=spec.route.speed_limit_mps * (0.80 if scripted == "follow" else 0.90),
        passing_side=passing_side,
        session=session,
        ego=ego,
        measured_speed_mps=v,
        front_gap_m=front,
        clear_of_lead=clear_of_lead,
        ego_lane=ego_lane,
        recovery=recovery,
        scripted_phase=scripted if scripted in ("lane_change", "overtake", "merge_back") else None,
    )

    ego.apply_control(ctrl)

    session.tick()



    step_ok = lane_dist <= lane_fail

    note = ""

    if edge is not None and edge < min_edge:

        step_ok = False

        note = f"edge_clearance_{edge:.2f}m"

    if lane_dist > lane_fail:

        note = f"lane_center_{lane_dist:.2f}m"



    long_snap = (
        session.pass_longitudinal_snapshot()
        if hasattr(session, "pass_longitudinal_snapshot")
        else {}
    )
    off_corridor = (
        ego_wp is not None
        and travel_road is not None
        and _off_curated_corridor(session, ego_wp, travel_road)
    )
    monitor_ok = not off_corridor
    abort_reason = f"left_corridor_road:{ego_wp.road_id}" if off_corridor else ""

    return PassStepRecord(
        step=step_i,
        scripted_phase=scripted,
        control_action=action,
        pass_phase=pass_phase,
        ego_lane_id=getattr(ego_wp, "lane_id", None),
        ego_road_id=getattr(ego_wp, "road_id", None),
        target_lane_id=getattr(target_wp, "lane_id", None) if target_wp else None,
        target_road_id=getattr(target_wp, "road_id", None) if target_wp else None,
        travel_lane_id=travel_lane,
        travel_road_id=travel_road,
        lateral_error_m=round(lat, 3),
        lane_center_dist_m=round(lane_dist, 3),
        edge_clearance_m=round(edge, 3) if edge is not None else None,
        lead_gap_m=round(front, 2),
        rear_gap_m=round(rear, 2),
        speed_mps=round(v, 2),
        steer=round(float(ctrl.steer), 3),
        ok=step_ok,
        note=note,
        pass_started=scripted != "follow",
        pass_in_progress=scripted in ("lane_change", "overtake", "merge_back"),
        monitor_ok=monitor_ok,
        abort_reason=abort_reason,
        pass_completed=False,
        cleared_lead=clear_of_lead,
        ego_s_m=long_snap.get("ego_s_m"),
        lead_s_m=long_snap.get("lead_s_m"),
        rear_s_m=long_snap.get("rear_s_m"),
    )





def run_scripted_pass_maneuver(

    session,

    spec,

    world,

    *,

    phase_max_s: Optional[Dict[str, float]] = None,

    total_max_s: float = DEFAULT_TOTAL_MAX_S,

    lane_fail_m: Optional[float] = None,

    min_edge_clearance_m: float = MIN_EDGE_CLEARANCE_M,

    verbose: bool = True,

    use_state_machine: bool = True,

    on_step: Optional[Callable[[PassStepRecord, Any], None]] = None,

) -> PassManeuverResult:

    from dataclasses import replace



    phase_limits = {**DEFAULT_PHASE_MAX_S, **(phase_max_s or {})}

    lane_fail = lane_fail_m if lane_fail_m is not None else lane_departure_fail_m()

    issues: List[str] = []

    steps: List[PassStepRecord] = []

    max_lane = 0.0

    min_edge = 999.0

    merged_back = False

    used_pass_lane = False

    pass_complete = False

    lane_flip_count = 0

    prev_on_pass = False



    session.enable_ego_physics(True)

    ego = session.actors.get("ego")

    if ego is None or session._travel_wp is None:

        return PassManeuverResult(ok=False, issues=["ego_or_travel_wp_missing"], pass_attempts=0)



    travel_lane = session._travel_wp.lane_id

    travel_road = session._travel_wp.road_id

    if session._passing_wp is None:

        issues.append("no_passing_lane_at_spawn")



    delta = session.fixed_delta_seconds
    clearance_m = merge_clearance_m()

    step_i = 0

    total_elapsed = 0.0

    scripted = "follow"

    phase_elapsed = 0.0

    pass_attempts = 1  # single maneuver start for scripted pass
    stability = PhaseStability()



    while scripted != "done" and total_elapsed < total_max_s:

        rec = _step_diagnostics(

            session,

            spec,

            world,

            ego,

            scripted=scripted,

            step_i=step_i,

            lane_fail=lane_fail,

            min_edge=min_edge_clearance_m,

        )

        steps.append(rec)

        max_lane = max(max_lane, rec.lane_center_dist_m)

        if rec.edge_clearance_m is not None:

            min_edge = min(min_edge, rec.edge_clearance_m)



        on_pass = (
            rec.ego_lane_id is not None
            and session._passing_wp is not None
            and rec.ego_lane_id == session._passing_wp.lane_id
            and rec.ego_road_id == session._passing_wp.road_id
        )

        if on_pass:

            used_pass_lane = True

        if on_pass != prev_on_pass:

            lane_flip_count += 1

        prev_on_pass = on_pass



        if scripted == "merge_back" and rec.ego_lane_id == travel_lane and rec.ego_road_id == travel_road:

            merged_back = True



        if verbose:

            print(

                f"{step_i:3d} {scripted:12s} {rec.pass_phase:10s} "

                f"ego_lane={rec.ego_lane_id}/{rec.ego_road_id} "

                f"tgt={rec.target_lane_id}/{rec.target_road_id} "

                f"lat={rec.lateral_error_m:+.2f} center={rec.lane_center_dist_m:.2f} "

                f"edge={rec.edge_clearance_m} lead={rec.lead_gap_m:.1f} "

                f"rear={rec.rear_gap_m:.1f} v={rec.speed_mps:.1f} steer={rec.steer:+.3f} "

                f"{rec.note}",

                flush=True,

            )



        if on_step is not None:

            on_step(rec, world)



        if not rec.ok:

            issues.append(f"{scripted}_step_{step_i}:{rec.note or 'lane_departure'}")

        if (
            scripted in ("lane_change", "overtake", "merge_back")
            and rec.ego_road_id is not None
            and not rec.monitor_ok
        ):
            issues.append(rec.abort_reason or f"left_corridor_road:{rec.ego_road_id}")
            break



        world = session.materialize_logical_world(

            world,

            measured_speed_mps=rec.speed_mps,

            duration_s=delta,

            ego_lane=session.infer_ego_lane_index(),

            passed=is_pass_maneuver_complete(
                session,
                ego_wp=session.map.get_waypoint(ego.get_location(), project_to_road=True) if session.map else None,
                ego_lane=session.infer_ego_lane_index(),
                clear_of_lead=_cleared_lead(session, session.ego_clear_of_lead(clearance_m)),
                travel_lane_id=travel_lane,
                travel_road_id=travel_road,
                lane_center_m=rec.lane_center_dist_m,
                travel_stable_ticks=stability.merge_centered_ticks,
                min_stable_ticks=PHASE_STABLE_TICKS if scripted == "merge_back" else 0,
            ),

            collision=False,

            done=False,

        )



        step_i += 1

        total_elapsed += delta

        phase_elapsed += delta



        if use_state_machine:

            try:

                ego_wp = session.map.get_waypoint(ego.get_location(), project_to_road=True)

            except Exception:

                ego_wp = None

            clear_of_lead = _cleared_lead(session, session.ego_clear_of_lead(clearance_m))

            next_phase = _advance_scripted_phase(
                scripted,
                session=session,
                ego=ego,
                ego_wp=ego_wp,
                ego_lane=session.infer_ego_lane_index(),
                clear_of_lead=clear_of_lead,
                front_gap_m=rec.lead_gap_m,
                travel_lane_id=travel_lane,
                travel_road_id=travel_road,
                phase_elapsed_s=phase_elapsed,
                phase_max_s=float(phase_limits.get(scripted, 8.0)),
                lane_center_m=rec.lane_center_dist_m,
                speed_mps=rec.speed_mps,
                stability=stability,
            )

            if next_phase != scripted:

                scripted = next_phase

                phase_elapsed = 0.0

        else:

            # Legacy fixed-duration mode (corridor validation during bootstrap)

            break



    if not use_state_machine:

        return _run_fixed_duration_pass(

            session, spec, world, phase_limits, lane_fail, min_edge_clearance_m, verbose

        )



    try:
        final_wp = session.map.get_waypoint(ego.get_location(), project_to_road=True) if session.map else None
    except Exception:
        final_wp = None
    final_lane_dist = session.ego_lane_center_distance_m(ego, phase="merge_back") if ego else 999.0
    pass_complete = is_pass_maneuver_complete(
        session,
        ego_wp=final_wp,
        ego_lane=session.infer_ego_lane_index(),
        clear_of_lead=_cleared_lead(session, session.ego_clear_of_lead(clearance_m)),
        travel_lane_id=travel_lane,
        travel_road_id=travel_road,
        lane_center_m=final_lane_dist,
        travel_stable_ticks=stability.merge_centered_ticks,
        min_stable_ticks=PHASE_STABLE_TICKS,
    )
    if steps:
        steps[-1] = replace(
            steps[-1],
            pass_completed=pass_complete and scripted == "done",
        )

    world = replace(world, passed=pass_complete)



    oscillation_count = max(0, lane_flip_count - 2)

    if oscillation_count > 2:

        issues.append(f"pass_lane_oscillation:{oscillation_count}")

    if not merged_back:

        issues.append("merge_back_not_on_travel_lane")

    if not used_pass_lane:

        issues.append("never_entered_passing_lane")

    if not pass_complete:

        issues.append("pass_not_complete")

    if min_edge < min_edge_clearance_m:

        issues.append(f"insufficient_edge_clearance:{min_edge:.2f}m")

    if total_elapsed >= total_max_s and scripted != "done":

        issues.append(f"pass_timeout:{total_elapsed:.1f}s")

    if pass_attempts > 1:

        issues.append(f"multiple_pass_attempts:{pass_attempts}")



    ok = len(issues) == 0

    return PassManeuverResult(

        ok=ok,

        issues=issues,

        steps=steps,

        max_lane_center_m=max_lane,

        min_edge_clearance_m=min_edge,

        merged_back=merged_back,

        pass_lane_used=used_pass_lane,

        pass_complete=pass_complete,

        pass_attempts=pass_attempts,

        oscillation_count=oscillation_count,

        final_world=world,

    )





def _run_fixed_duration_pass(

    session,

    spec,

    world,

    durations: Dict[str, float],

    lane_fail: float,

    min_edge_clearance_m: float,

    verbose: bool,

) -> PassManeuverResult:

    """Fixed-duration pass for corridor bootstrap validation (no video)."""

    issues: List[str] = []

    steps: List[PassStepRecord] = []

    max_lane = 0.0

    min_edge = 999.0

    merged_back = False

    used_pass_lane = False

    ego = session.actors.get("ego")

    travel_lane = session._travel_wp.lane_id

    travel_road = session._travel_wp.road_id

    step_i = 0



    for scripted in ("follow", "lane_change", "overtake", "merge_back"):

        dur = float(durations.get(scripted, 4.0))

        n_ticks = max(1, int(round(dur / session.fixed_delta_seconds)))

        for _ in range(n_ticks):

            rec = _step_diagnostics(

                session, spec, world, ego,

                scripted=scripted, step_i=step_i, lane_fail=lane_fail, min_edge=min_edge_clearance_m,

            )

            steps.append(rec)

            max_lane = max(max_lane, rec.lane_center_dist_m)

            if rec.edge_clearance_m is not None:

                min_edge = min(min_edge, rec.edge_clearance_m)

            if (
                rec.ego_lane_id == session._passing_wp.lane_id
                and rec.ego_road_id == session._passing_wp.road_id
            ):

                used_pass_lane = True

            if rec.ego_lane_id == travel_lane and rec.ego_road_id == travel_road and scripted == "merge_back":

                merged_back = True

            if verbose:

                print(

                    f"{step_i:3d} {scripted:12s} center={rec.lane_center_dist_m:.2f} "

                    f"lead={rec.lead_gap_m:.1f} v={rec.speed_mps:.1f}",

                    flush=True,

                )

            if not rec.ok:

                issues.append(f"{scripted}_step_{step_i}:{rec.note}")

            step_i += 1



    if not merged_back:

        issues.append("merge_back_not_on_travel_lane")

    if not used_pass_lane:

        issues.append("never_entered_passing_lane")

    if min_edge < min_edge_clearance_m:

        issues.append(f"insufficient_edge_clearance:{min_edge:.2f}m")



    return PassManeuverResult(

        ok=len(issues) == 0,

        issues=issues,

        steps=steps,

        max_lane_center_m=max_lane,

        min_edge_clearance_m=min_edge,

        merged_back=merged_back,

        pass_lane_used=used_pass_lane,

        pass_complete=merged_back and used_pass_lane,

        pass_attempts=1,

    )





def run_no_pass_follow(

    session,

    spec,

    world,

    *,

    duration_s: float = 28.0,

    on_step: Optional[Callable[[PassStepRecord, Any], None]] = None,

) -> NoPassFollowResult:

    """Hold behind lead — no lane change (no-pass baseline for hero video)."""

    issues: List[str] = []

    steps: List[PassStepRecord] = []

    session.enable_ego_physics(True)

    ego = session.actors.get("ego")

    if ego is None:

        return NoPassFollowResult(ok=False, issues=["ego_missing"])



    delta = session.fixed_delta_seconds

    n_ticks = max(1, int(round(duration_s / delta)))

    lane_fail = lane_departure_fail_m()



    for step_i in range(n_ticks):

        rec = _step_diagnostics(

            session, spec, world, ego,

            scripted="follow", step_i=step_i, lane_fail=lane_fail, min_edge=MIN_EDGE_CLEARANCE_M,

        )

        steps.append(rec)

        if on_step:

            on_step(rec, world)

        world = session.materialize_logical_world(

            world,

            measured_speed_mps=rec.speed_mps,

            duration_s=delta,

            ego_lane=0,

            passed=False,

            collision=False,

            done=False,

        )

        if not rec.ok:

            issues.append(f"follow_step_{step_i}:{rec.note}")



    ok = len(issues) == 0 and rec.lead_gap_m < 35.0

    if rec.lead_gap_m >= 35.0:

        issues.append(f"drifted_from_lead:{rec.lead_gap_m:.1f}m")

    return NoPassFollowResult(ok=ok, issues=issues, steps=steps, final_world=world)

