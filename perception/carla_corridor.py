"""CARLA passing-corridor validation: presentation-safe and hero modes for benchmark video."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

ValidationMode = Literal["strict", "presentation", "hero"]

NOT_CURATED_CORRIDOR_MSG = (
    "Selected CARLA route is not a curated passing corridor; "
    "use highway curated mode or choose a different spawn."
)

HERO_CORRIDOR_WARNING = (
    "Using hero corridor: validated for the passing maneuver horizon, "
    "not for full-map autonomous driving."
)

DEFAULT_LOOKAHEAD_M = 120.0
DEFAULT_BEHIND_M = 40.0
DEFAULT_MAX_YAW_DELTA_DEG = 12.0
TRAFFIC_CONTROL_RADIUS_M = 28.0

MANEUVER_HORIZON_PRESENTATION_M = 100.0
MANEUVER_HORIZON_HERO_M = 90.0

# Hand-curated spawn indices per map (tried before automatic scan when scan fails).
# Add indices discovered via: python -m perception.carla_corridor_smoke --diagnose --top-k 10
CURATED_CORRIDOR_CANDIDATES: Dict[str, List[int]] = {
    # Town04 road 6: spawn 141 (lane 5) — 200m forward straight, validated passing corridor.
    "Town04": [141, 140],
    # Town01 road 4 / road 10: long straight two-lane roads with passing lane.
    "Town01": [2, 4, 9, 12, 16, 19],
    # Town02 road 12: good straight segments.
    "Town02": [19, 21, 27, 29, 34],
    # Town03: limited options, road 67.
    "Town03": [170],
    # Town05: multi-lane highway-style roads.
    "Town05": [144, 155, 159, 161, 179],
}

SCAN_MAP_PRIORITY: Tuple[str, ...] = ("Town04", "Town01", "Town05", "Town02", "Town03")

_MODE_PARAMS: Dict[str, Dict[str, float]] = {
    "strict": {
        "lookahead_m": 120.0,
        "behind_m": 40.0,
        "min_forward_m": 80.0,
        "min_behind_m": 30.0,
        "maneuver_horizon_m": 120.0,
        "max_yaw_delta_deg": 12.0,
        "lock_lane_identity": 1.0,
        "require_opposing_lane": 1.0,
    },
    "presentation": {
        "lookahead_m": 100.0,
        "behind_m": 35.0,
        "min_forward_m": 60.0,
        "min_behind_m": 25.0,
        "maneuver_horizon_m": 65.0,
        "max_yaw_delta_deg": 28.0,
        "lock_lane_identity": 0.0,
        "require_opposing_lane": 0.0,
    },
    "hero": {
        "lookahead_m": 80.0,
        "behind_m": 30.0,
        "min_forward_m": 50.0,
        "min_behind_m": 20.0,
        "maneuver_horizon_m": 60.0,
        "max_yaw_delta_deg": 35.0,
        "lock_lane_identity": 0.0,
        "require_opposing_lane": 0.0,
    },
}


@dataclass
class CorridorReport:
    ok: bool
    issues: List[str] = field(default_factory=list)
    validation_mode: str = "strict"
    presentation_ok: bool = False
    hero_ok: bool = False
    lookahead_m: float = DEFAULT_LOOKAHEAD_M
    behind_m: float = DEFAULT_BEHIND_M
    maneuver_horizon_m: float = DEFAULT_LOOKAHEAD_M
    junction_count: int = 0
    junction_count_in_horizon: int = 0
    traffic_light_count: int = 0
    stop_control_count: int = 0
    road_id: Optional[int] = None
    lane_id: Optional[int] = None
    start_x: float = 0.0
    start_y: float = 0.0
    start_yaw: float = 0.0
    forward_length_m: float = 0.0
    backward_length_m: float = 0.0
    maneuver_forward_length_m: float = 0.0
    has_passing_lane: bool = False
    has_opposing_lane: bool = False
    max_yaw_delta_deg: float = 0.0
    heading_change_deg: float = 0.0
    lane_change_count: int = 0
    waypoint_count: int = 0
    spawn_index: Optional[int] = None
    map_name: Optional[str] = None

    def summary_line(self) -> str:
        return (
            f"mode={self.validation_mode} road={self.road_id} lane={self.lane_id} "
            f"start=({self.start_x:.1f},{self.start_y:.1f}) yaw={self.start_yaw:.1f} "
            f"fwd={self.forward_length_m:.0f}m back={self.backward_length_m:.0f}m "
            f"horizon_fwd={self.maneuver_forward_length_m:.0f}m "
            f"junctions={self.junction_count} (in_horizon={self.junction_count_in_horizon}) "
            f"lights={self.traffic_light_count} stops={self.stop_control_count} "
            f"passing={self.has_passing_lane} opposing={self.has_opposing_lane} "
            f"yaw={self.heading_change_deg:.1f}deg lane_changes={self.lane_change_count}"
        )


@dataclass
class CorridorCandidateRecord:
    """One spawn evaluation with fields for diagnostics and near-miss ranking."""

    spawn_index: int
    map_name: str
    ok: bool
    primary_rejection: str
    issues: List[str]
    report: CorridorReport
    transform: Optional[Dict[str, float]] = None
    near_miss_score: float = 0.0

    @property
    def road_id(self) -> Optional[int]:
        return self.report.road_id

    @property
    def lane_id(self) -> Optional[int]:
        return self.report.lane_id

    @property
    def straight_length_m(self) -> float:
        return self.report.maneuver_forward_length_m

    @property
    def junction_count(self) -> int:
        return self.report.junction_count_in_horizon

    @property
    def heading_change_deg(self) -> float:
        return self.report.heading_change_deg

    @property
    def lane_change_count(self) -> int:
        return self.report.lane_change_count

    @property
    def opposing_lane_found(self) -> bool:
        return self.report.has_opposing_lane

    @property
    def passing_lane_found(self) -> bool:
        return self.report.has_passing_lane

    @property
    def traffic_light_count(self) -> int:
        return self.report.traffic_light_count

    @property
    def stop_prop_count(self) -> int:
        return self.report.stop_control_count


@dataclass
class CorridorScanDiagnostics:
    map_name: str
    validation_mode: str
    total_scanned: int
    valid_count: int
    rejection_counts: Dict[str, int] = field(default_factory=dict)
    near_misses: List[CorridorCandidateRecord] = field(default_factory=list)

    def recommendation(self) -> str:
        if self.valid_count > 0:
            best = max(
                (r for r in self.near_misses if r.ok),
                key=lambda r: r.near_miss_score,
                default=None,
            )
            if best is not None:
                return (
                    f"Use spawn_index={best.spawn_index} on {self.map_name} "
                    f"({best.report.summary_line()})"
                )
        if not self.near_misses:
            return "No candidates scanned; check CARLA map load and spawn points."
        top = self.near_misses[0]
        curated = CURATED_CORRIDOR_CANDIDATES.get(self.map_name, [])
        hint = (
            f"Add spawn_index={top.spawn_index} to CURATED_CORRIDOR_CANDIDATES['{self.map_name}'] "
            "or try --hero / another map in SCAN_MAP_PRIORITY."
        )
        if curated:
            hint = f"Try manual indices {curated} or " + hint
        return (
            f"Closest near-miss: spawn_index={top.spawn_index} "
            f"reason={top.primary_rejection} score={top.near_miss_score:.1f}. {hint}"
        )


def _normalize_yaw_delta(yaw0: float, yaw1: float) -> float:
    d = yaw1 - yaw0
    while d > 180.0:
        d -= 360.0
    while d < -180.0:
        d += 360.0
    return abs(d)


def _default_find_passing_lane(wp, carla) -> Any:
    for getter in (wp.get_left_lane, wp.get_right_lane):
        try:
            adj = getter()
        except Exception:
            adj = None
        if adj is None or adj.lane_type != carla.LaneType.Driving:
            continue
        if adj.lane_id * wp.lane_id > 0:
            return adj
    return None


def _default_find_opposing_lane(wp, carla) -> Any:
    for getter in (wp.get_left_lane, wp.get_right_lane):
        try:
            adj = getter()
        except Exception:
            adj = None
        if adj is None or adj.lane_type != carla.LaneType.Driving:
            continue
        if adj.lane_id * wp.lane_id < 0:
            return adj
    return None


def _walk_corridor(
    start_wp,
    distance_m: float,
    *,
    direction: str,
    lane_id: int,
    road_id: int,
    yaw0: float,
    max_yaw_delta_deg: float,
    lock_lane_identity: bool,
    maneuver_horizon_m: float,
) -> Tuple[List[Any], float, int, int, int, List[str], float, float]:
    """Walk along corridor; return points, walked, junction counts, lane changes, issues, yaw stats."""
    issues: List[str] = []
    points: List[Any] = [start_wp]
    walked = 0.0
    junction_count = 0
    junction_in_horizon = 0
    lane_change_count = 0
    max_yaw = 0.0
    max_yaw_horizon = 0.0
    cur = start_wp
    remaining = float(distance_m)
    min_forward = 4.0
    cur_lane = lane_id
    cur_road = road_id

    while remaining > 0.5:
        step = min(min_forward, remaining)
        in_horizon = walked < maneuver_horizon_m
        if direction == "forward":
            nxt = cur.next(step)
        else:
            nxt = cur.previous(step)
        if not nxt:
            tag = f"corridor_{direction}_ends_early_at_{walked:.0f}m"
            if in_horizon:
                issues.append(tag)
            break
        if len(nxt) > 1:
            tag = f"corridor_{direction}_branch_at_{walked:.0f}m"
            if in_horizon:
                issues.append(tag)
            break
        cand = nxt[0]
        at_dist = walked + step
        if getattr(cand, "is_junction", False):
            junction_count += 1
            if at_dist <= maneuver_horizon_m:
                junction_in_horizon += 1
                issues.append(f"corridor_{direction}_junction_at_{at_dist:.0f}m")
            break
        if lock_lane_identity:
            if cand.lane_id != lane_id or cand.road_id != road_id:
                issues.append(
                    f"corridor_{direction}_lane_discontinuity_at_{at_dist:.0f}m "
                    f"(expected road={road_id} lane={lane_id}, got road={cand.road_id} lane={cand.lane_id})"
                )
                break
        elif cand.lane_id != cur_lane or cand.road_id != cur_road:
            lane_change_count += 1
            cur_lane = cand.lane_id
            cur_road = cand.road_id
        yaw_delta = _normalize_yaw_delta(yaw0, cand.transform.rotation.yaw)
        max_yaw = max(max_yaw, yaw_delta)
        if at_dist <= maneuver_horizon_m:
            max_yaw_horizon = max(max_yaw_horizon, yaw_delta)
        if in_horizon and yaw_delta > max_yaw_delta_deg:
            issues.append(f"corridor_{direction}_turn_at_{at_dist:.0f}m (yaw_delta={yaw_delta:.1f}deg)")
            break
        try:
            loc0 = cur.transform.location
            loc1 = cand.transform.location
            dx = loc1.x - loc0.x
            dy = loc1.y - loc0.y
            seg_len = math.sqrt(dx * dx + dy * dy)
            if seg_len > 0.5:
                yaw_rad = math.radians(cand.transform.rotation.yaw)
                lat = abs(-dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad))
                if lat > 2.5:
                    tag = f"corridor_{direction}_center_jump_at_{at_dist:.0f}m"
                    if in_horizon:
                        issues.append(tag)
                    break
        except Exception:
            pass
        cur = cand
        points.append(cur)
        walked += step
        remaining -= step

    return (
        points,
        walked,
        junction_count,
        junction_in_horizon,
        lane_change_count,
        issues,
        max_yaw,
        max_yaw_horizon,
    )


def _nearby_traffic_controls(
    world,
    corridor_points: List[Any],
    *,
    max_path_distance_m: Optional[float] = None,
    radius_m: float = TRAFFIC_CONTROL_RADIUS_M,
) -> Tuple[int, int]:
    if world is None or not corridor_points:
        return 0, 0
    lights = 0
    stops = 0
    origin = corridor_points[0].transform.location
    try:
        patterns = (
            ("traffic.traffic_light", "light"),
            ("traffic.stop", "stop"),
            ("static.prop.stop", "stop"),
            ("static.prop.trafficwarning", "stop"),
        )
        for pattern, kind in patterns:
            try:
                actors = world.get_actors().filter(pattern)
            except Exception:
                continue
            for actor in actors:
                try:
                    loc = actor.get_location()
                except Exception:
                    continue
                near = False
                for wp in corridor_points:
                    try:
                        wloc = wp.transform.location
                        if max_path_distance_m is not None:
                            along = math.sqrt(
                                (wloc.x - origin.x) ** 2 + (wloc.y - origin.y) ** 2
                            )
                            if along > max_path_distance_m:
                                continue
                        if loc.distance(wloc) <= radius_m:
                            near = True
                            break
                    except Exception:
                        continue
                if not near:
                    continue
                if kind == "light":
                    lights += 1
                else:
                    stops += 1
    except Exception:
        pass
    return lights, stops


def primary_rejection_reason(issues: Sequence[str]) -> str:
    if not issues:
        return "ok"
    priority = (
        "spawn_not_driving_lane",
        "spawn_in_junction",
        "spawn_lane_or_road_unknown",
        "no_opposing_lane",
        "no_passing_lane",
        "insufficient_forward_length",
        "insufficient_backward_length",
        "traffic_light_near_corridor",
        "stop_control_near_corridor",
        "corridor_forward_junction",
        "corridor_backward_junction",
        "corridor_forward_turn",
        "corridor_forward_branch",
        "corridor_forward_lane_discontinuity",
        "corridor_forward_center_jump",
    )
    for key in priority:
        for issue in issues:
            if issue.startswith(key) or key in issue:
                return key
    return issues[0].split(":")[0].split("_at_")[0]


def near_miss_score(report: CorridorReport, issues: Sequence[str]) -> float:
    """Higher is closer to passing (for ranking near-misses)."""
    score = report.forward_length_m + report.backward_length_m * 0.5
    score += report.maneuver_forward_length_m * 0.3
    if report.has_opposing_lane:
        score += 10.0
    if report.has_passing_lane:
        score += 12.0
    else:
        score -= 35.0
    score -= report.lane_change_count * 4.0
    score -= report.junction_count_in_horizon * 25.0
    score -= report.traffic_light_count * 15.0
    score -= report.stop_control_count * 12.0
    # Penalize curves harder — lane-change geometry is unstable on tight bends.
    yaw_pen = 1.2 if report.validation_mode in ("presentation", "hero") else 0.4
    score -= report.heading_change_deg * yaw_pen
    score -= len(issues) * 3.0
    return score


def validate_passing_corridor(
    spawn_wp,
    *,
    lookahead_m: float = DEFAULT_LOOKAHEAD_M,
    behind_m: float = DEFAULT_BEHIND_M,
    carla=None,
    world=None,
    require_passing_lane: bool = False,
    require_opposing_lane: bool = True,
    max_yaw_delta_deg: float = DEFAULT_MAX_YAW_DELTA_DEG,
    min_forward_m: float = 80.0,
    min_behind_m: float = 30.0,
    maneuver_horizon_m: Optional[float] = None,
    lock_lane_identity: bool = True,
    validation_mode: ValidationMode = "strict",
    find_passing_lane: Optional[Callable] = None,
    find_opposing_lane: Optional[Callable] = None,
    spawn_index: Optional[int] = None,
    map_name: Optional[str] = None,
    _skip_derived_modes: bool = False,
) -> CorridorReport:
    """
    Validate a CARLA spawn waypoint as a passing corridor.

    Modes:
      strict — stable road/lane for full lookahead; full-map traffic scan.
      presentation — maneuver-horizon junction/control checks; lane continuity may change.
      hero — shortest horizon for final video; same relaxed geometry as presentation.
    """
    params = _MODE_PARAMS.get(validation_mode, _MODE_PARAMS["strict"])
    if validation_mode != "strict":
        lookahead_m = float(params["lookahead_m"])
        behind_m = float(params["behind_m"])
        min_forward_m = float(params["min_forward_m"])
        min_behind_m = float(params["min_behind_m"])
        max_yaw_delta_deg = float(params["max_yaw_delta_deg"])
        lock_lane_identity = bool(params["lock_lane_identity"])
        require_opposing_lane = bool(params["require_opposing_lane"])
        if maneuver_horizon_m is None:
            maneuver_horizon_m = float(params["maneuver_horizon_m"])
    if maneuver_horizon_m is None:
        maneuver_horizon_m = max(lookahead_m, behind_m)

    issues: List[str] = []
    if spawn_wp is None:
        return CorridorReport(ok=False, issues=["spawn_waypoint_missing"], validation_mode=validation_mode)

    try:
        loc = spawn_wp.transform.location
        start_x, start_y = float(loc.x), float(loc.y)
        start_yaw = float(spawn_wp.transform.rotation.yaw)
    except Exception:
        start_x = start_y = start_yaw = 0.0

    lane_id = getattr(spawn_wp, "lane_id", None)
    road_id = getattr(spawn_wp, "road_id", None)

    if carla is not None and getattr(spawn_wp, "lane_type", None) != carla.LaneType.Driving:
        issues.append("spawn_not_driving_lane")
    if getattr(spawn_wp, "is_junction", False):
        issues.append("spawn_in_junction")
    if lane_id is None or road_id is None:
        issues.append("spawn_lane_or_road_unknown")

    find_pass = find_passing_lane or (lambda wp: _default_find_passing_lane(wp, carla) if carla else None)
    find_opp = find_opposing_lane or (lambda wp: _default_find_opposing_lane(wp, carla) if carla else None)

    has_passing = find_pass(spawn_wp) is not None if carla else False
    has_opposing = find_opp(spawn_wp) is not None if carla else False
    if require_passing_lane and not has_passing:
        issues.append("no_passing_lane")
    if require_opposing_lane and not has_opposing:
        issues.append("no_opposing_lane")

    yaw0 = start_yaw
    (
        fwd_points,
        fwd_m,
        fwd_junctions,
        fwd_junc_horizon,
        fwd_lane_changes,
        fwd_issues,
        fwd_yaw,
        fwd_yaw_horizon,
    ) = _walk_corridor(
        spawn_wp,
        lookahead_m,
        direction="forward",
        lane_id=lane_id,
        road_id=road_id,
        yaw0=yaw0,
        max_yaw_delta_deg=max_yaw_delta_deg,
        lock_lane_identity=lock_lane_identity,
        maneuver_horizon_m=maneuver_horizon_m,
    )
    (
        back_points,
        back_m,
        back_junctions,
        back_junc_horizon,
        back_lane_changes,
        back_issues,
        back_yaw,
        _back_yaw_horizon,
    ) = _walk_corridor(
        spawn_wp,
        behind_m,
        direction="backward",
        lane_id=lane_id,
        road_id=road_id,
        yaw0=yaw0,
        max_yaw_delta_deg=max_yaw_delta_deg,
        lock_lane_identity=lock_lane_identity,
        maneuver_horizon_m=maneuver_horizon_m,
    )

    issues.extend(fwd_issues)
    issues.extend(back_issues)
    junction_count = fwd_junctions + back_junctions
    junction_in_horizon = fwd_junc_horizon + back_junc_horizon
    lane_change_count = fwd_lane_changes + back_lane_changes

    issues = [
        i
        for i in issues
        if not (
            (i.startswith("corridor_forward_ends_early") and fwd_m >= min_forward_m)
            or (i.startswith("corridor_backward_ends_early") and back_m >= min_behind_m)
        )
    ]

    if fwd_m < min_forward_m:
        issues.append(f"insufficient_forward_length:{fwd_m:.0f}m<{min_forward_m:.0f}m")
    if back_m < min_behind_m:
        issues.append(f"insufficient_backward_length:{back_m:.0f}m<{min_behind_m:.0f}m")

    maneuver_fwd = min(fwd_m, maneuver_horizon_m)
    horizon_points = [p for p in fwd_points if True]
    if len(fwd_points) > 1:
        origin = spawn_wp.transform.location
        horizon_points = []
        acc = 0.0
        prev = spawn_wp
        horizon_points.append(spawn_wp)
        for p in fwd_points[1:]:
            try:
                seg = prev.transform.location.distance(p.transform.location)
            except Exception:
                seg = 4.0
            acc += seg
            horizon_points.append(p)
            prev = p
            if acc >= maneuver_horizon_m:
                break

    lights, stops = _nearby_traffic_controls(
        world,
        horizon_points if validation_mode != "strict" else fwd_points + back_points,
        max_path_distance_m=maneuver_horizon_m if validation_mode != "strict" else None,
    )
    if lights > 0:
        issues.append(f"traffic_light_near_corridor:{lights}")
    if stops > 0:
        issues.append(f"stop_control_near_corridor:{stops}")

    ok = len(issues) == 0

    presentation_ok = ok if validation_mode == "presentation" else False
    hero_ok = ok if validation_mode == "hero" else False
    if not _skip_derived_modes:
        if validation_mode != "presentation":
            presentation_ok = validate_passing_corridor(
                spawn_wp,
                carla=carla,
                world=world,
                validation_mode="presentation",
                find_passing_lane=find_passing_lane,
                find_opposing_lane=find_opposing_lane,
                spawn_index=spawn_index,
                map_name=map_name,
                _skip_derived_modes=True,
            ).ok
        if validation_mode != "hero":
            hero_ok = validate_passing_corridor(
                spawn_wp,
                carla=carla,
                world=world,
                validation_mode="hero",
                find_passing_lane=find_passing_lane,
                find_opposing_lane=find_opposing_lane,
                spawn_index=spawn_index,
                map_name=map_name,
                _skip_derived_modes=True,
            ).ok

    return CorridorReport(
        ok=ok,
        issues=issues,
        validation_mode=validation_mode,
        presentation_ok=presentation_ok,
        hero_ok=hero_ok,
        lookahead_m=lookahead_m,
        behind_m=behind_m,
        maneuver_horizon_m=maneuver_horizon_m,
        junction_count=junction_count,
        junction_count_in_horizon=junction_in_horizon,
        traffic_light_count=lights,
        stop_control_count=stops,
        road_id=road_id,
        lane_id=lane_id,
        start_x=start_x,
        start_y=start_y,
        start_yaw=start_yaw,
        forward_length_m=fwd_m,
        backward_length_m=back_m,
        maneuver_forward_length_m=maneuver_fwd,
        has_passing_lane=has_passing,
        has_opposing_lane=has_opposing,
        max_yaw_delta_deg=max(fwd_yaw, back_yaw),
        heading_change_deg=max(fwd_yaw_horizon, back_yaw),
        lane_change_count=lane_change_count,
        waypoint_count=len(fwd_points) + len(back_points),
        spawn_index=spawn_index,
        map_name=map_name,
    )


def _transform_dict(spawn) -> Optional[Dict[str, float]]:
    try:
        t = spawn.location
        r = spawn.rotation
        return {"x": float(t.x), "y": float(t.y), "z": float(t.z), "yaw": float(r.yaw)}
    except Exception:
        return None


def evaluate_spawn_candidate(
    spawn_index: int,
    spawn_wp,
    *,
    map_name: str,
    validation_mode: ValidationMode,
    carla=None,
    world=None,
    spawn=None,
    find_passing_lane: Optional[Callable] = None,
    find_opposing_lane: Optional[Callable] = None,
) -> CorridorCandidateRecord:
    report = validate_passing_corridor(
        spawn_wp,
        carla=carla,
        world=world,
        validation_mode=validation_mode,
        find_passing_lane=find_passing_lane,
        find_opposing_lane=find_opposing_lane,
        spawn_index=spawn_index,
        map_name=map_name,
    )
    primary = primary_rejection_reason(report.issues)
    score = near_miss_score(report, report.issues)
    return CorridorCandidateRecord(
        spawn_index=spawn_index,
        map_name=map_name,
        ok=report.ok,
        primary_rejection=primary,
        issues=list(report.issues),
        report=report,
        transform=_transform_dict(spawn) if spawn is not None else None,
        near_miss_score=score,
    )


def scan_spawn_candidates(
    spawns: Sequence,
    *,
    map_name: str,
    map_obj,
    validation_mode: ValidationMode,
    carla=None,
    world=None,
    max_candidates: int = 300,
    indices: Optional[Sequence[int]] = None,
    find_passing_lane: Optional[Callable] = None,
    find_opposing_lane: Optional[Callable] = None,
) -> List[CorridorCandidateRecord]:
    records: List[CorridorCandidateRecord] = []
    if indices is None:
        indices = range(min(max_candidates, len(spawns)))
    for idx in indices:
        if idx < 0 or idx >= len(spawns):
            continue
        sp = spawns[idx]
        wp = map_obj.get_waypoint(sp.location, project_to_road=True)
        records.append(
            evaluate_spawn_candidate(
                idx,
                wp,
                map_name=map_name,
                validation_mode=validation_mode,
                carla=carla,
                world=world,
                spawn=sp,
                find_passing_lane=find_passing_lane,
                find_opposing_lane=find_opposing_lane,
            )
        )
    return records


def build_scan_diagnostics(
    records: Sequence[CorridorCandidateRecord],
    *,
    map_name: str,
    validation_mode: ValidationMode,
) -> CorridorScanDiagnostics:
    rejection_counts: Dict[str, int] = {}
    for rec in records:
        if rec.ok:
            continue
        rejection_counts[rec.primary_rejection] = rejection_counts.get(rec.primary_rejection, 0) + 1
    ranked = sorted(records, key=lambda r: r.near_miss_score, reverse=True)
    return CorridorScanDiagnostics(
        map_name=map_name,
        validation_mode=validation_mode,
        total_scanned=len(records),
        valid_count=sum(1 for r in records if r.ok),
        rejection_counts=rejection_counts,
        near_misses=ranked,
    )


def format_diagnostics_report(diag: CorridorScanDiagnostics, *, top_k: int = 5) -> str:
    lines = [
        f"map={diag.map_name} mode={diag.validation_mode}",
        f"total_candidates_scanned={diag.total_scanned}",
        f"valid_candidates={diag.valid_count}",
        "rejection_reasons:",
    ]
    for reason, count in sorted(diag.rejection_counts.items(), key=lambda x: -x[1])[:15]:
        lines.append(f"  {reason}: {count}")
    lines.append(f"top_{top_k}_near_misses:")
    for rec in diag.near_misses[:top_k]:
        tf = rec.transform or {}
        lines.append(
            f"  spawn_index={rec.spawn_index} score={rec.near_miss_score:.1f} "
            f"reason={rec.primary_rejection} "
            f"transform=({tf.get('x', 0):.1f},{tf.get('y', 0):.1f}) yaw={tf.get('yaw', 0):.1f} "
            f"road_id={rec.road_id} lane_id={rec.lane_id} "
            f"straight_length={rec.straight_length_m:.0f}m "
            f"junction_count={rec.junction_count} "
            f"heading_change_deg={rec.heading_change_deg:.1f} "
            f"lane_change_count={rec.lane_change_count} "
            f"opposing_lane_found={rec.opposing_lane_found} "
            f"passing_lane_found={rec.passing_lane_found} "
            f"traffic_light_count={rec.traffic_light_count} "
            f"stop_prop_count={rec.stop_prop_count}"
        )
    lines.append(f"recommendation: {diag.recommendation()}")
    return "\n".join(lines)


def pick_best_candidate(
    records: Sequence[CorridorCandidateRecord],
) -> Optional[CorridorCandidateRecord]:
    valid = [r for r in records if r.ok]
    if not valid:
        return None
    return max(valid, key=lambda r: r.near_miss_score)


def pick_hero_candidate(
    records: Sequence[CorridorCandidateRecord],
) -> Optional[CorridorCandidateRecord]:
    """Best record that passes hero validation (may fail strict/presentation)."""
    hero_valid = [r for r in records if r.report.hero_ok or (r.report.validation_mode == "hero" and r.ok)]
    if not hero_valid:
        return None
    return max(hero_valid, key=lambda r: r.near_miss_score)


def corridor_accepted_for_production(report: CorridorReport) -> Tuple[bool, bool]:
    """
    Return (accepted, used_hero_fallback).
    Strict/presentation ok → accepted without hero; hero_ok alone → accepted with warning.
    """
    if report.ok or report.presentation_ok:
        return True, False
    if report.hero_ok:
        return True, True
    return False, False
