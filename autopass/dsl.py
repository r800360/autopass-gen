"""
Domain-specific language for AutoPass missions.

The DSL is the shared representation planners, critics, and executors read/write.
It is patched iteratively during plan → tool → verify → replan cycles (not frozen
at graph start).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Literal, Optional


ManeuverKind = Literal["pass", "wait", "replan", "hold"]
RoadKind = Literal["highway", "urban", "local"]
UrgencyKind = Literal["low", "medium", "high"]


@dataclass
class MissionSpec:
    text: str
    start: str
    goal: str
    deadline_s: float
    urgency: UrgencyKind = "medium"
    aggression: str = "low"  # 0 | low | high


@dataclass
class RouteWaypoint:
    street: str
    action: str = "drive"
    speed_mps: float = 15.0


@dataclass
class RoutePlan:
    description: str = ""
    waypoints: List[RouteWaypoint] = field(default_factory=list)
    eta_s: float = 60.0
    road_type: RoadKind = "highway"


@dataclass
class PerceptionRecord:
    """One tool invocation result appended during planning."""
    tool: str
    summary: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationNote:
    """Critic feedback after a tool run or maneuver."""
    verdict: Literal["ok", "insufficient", "reject", "replan"]
    message: str
    tool: Optional[str] = None
    revision_triggered: bool = False


@dataclass
class ManeuverPlan:
    kind: ManeuverKind = "hold"
    passing_side: str = ""
    target_speed_mps: float = 0.0
    required_time_s: float = 0.0
    reasoning: str = ""


@dataclass
class ExecutionRecord:
    """Feedback from CARLA VehicleControl or kinematic step — observed by critic."""
    action: str
    mode: str
    summary: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActorBelief:
    exists: bool = False
    distance_m: Optional[float] = None
    speed_mps: Optional[float] = None
    position_label: str = ""


@dataclass
class WorldBelief:
    """Mutable world estimate — updated after each execute from CARLA/visual depth."""

    source: str = "initial"
    t_s: float = 0.0
    ego_lane: int = 0
    ego_speed_mps: float = 0.0
    progress_m: float = 0.0
    front_gap_m: Optional[float] = None
    rear_gap_m: Optional[float] = None
    oncoming_gap_m: Optional[float] = None
    front_valid: bool = False
    rear_valid: bool = False
    oncoming_valid: bool = False
    oncoming_available: bool = True
    oncoming_unavailable_reason: str = ""
    visibility_m: Optional[float] = None
    lead_speed_mps: Optional[float] = None
    rear_closing_mps: Optional[float] = None
    oncoming_approach_mps: Optional[float] = None
    depth_confidence: float = 0.0
    car_distances: List[Dict[str, Any]] = field(default_factory=list)
    actors: Dict[str, ActorBelief] = field(default_factory=dict)
    physics_valid: bool = True
    physics_issues: List[str] = field(default_factory=list)


def update_world_belief(
    belief: WorldBelief,
    *,
    t_s: float,
    ego_lane: int,
    ego_speed_mps: float,
    progress_m: float,
) -> WorldBelief:
    return replace(
        belief,
        t_s=t_s,
        ego_lane=ego_lane,
        ego_speed_mps=ego_speed_mps,
        progress_m=progress_m,
    )


@dataclass
class PassingDSL:
    """Living plan document — revision increments on each replan."""

    mission: MissionSpec
    route: RoutePlan = field(default_factory=RoutePlan)
    maneuver: ManeuverPlan = field(default_factory=ManeuverPlan)
    perception_log: List[PerceptionRecord] = field(default_factory=list)
    execution_log: List[ExecutionRecord] = field(default_factory=list)
    verification_log: List[VerificationNote] = field(default_factory=list)
    tools_completed: List[str] = field(default_factory=list)
    tools_pending: List[str] = field(default_factory=list)
    revision: int = 0
    world_belief: WorldBelief = field(default_factory=WorldBelief)

    def update_belief(self, belief: WorldBelief) -> "PassingDSL":
        return replace(self, world_belief=belief)

    def patch_mission(self, **kwargs: Any) -> "PassingDSL":
        return replace(self, mission=replace(self.mission, **kwargs))

    def patch_route(self, **kwargs: Any) -> "PassingDSL":
        return replace(self, route=replace(self.route, **kwargs))

    def append_execution(self, record: ExecutionRecord) -> "PassingDSL":
        log = list(self.execution_log) + [record]
        return replace(self, execution_log=log)

    def append_perception(self, record: PerceptionRecord) -> "PassingDSL":
        done = list(self.tools_completed)
        if record.tool not in done:
            done.append(record.tool)
        pending = [t for t in self.tools_pending if t != record.tool]
        log = list(self.perception_log) + [record]
        return replace(self, perception_log=log, tools_completed=done, tools_pending=pending)

    def invalidate_tool(self, tool: str, reason: str = "") -> "PassingDSL":
        """Remove invalidated tool evidence so planner cannot rely on rejected payloads."""
        log = [r for r in self.perception_log if r.tool != tool]
        done = [t for t in self.tools_completed if t != tool]
        pending = [t for t in self.tools_pending if t != tool]
        return replace(self, perception_log=log, tools_completed=done, tools_pending=pending)

    def append_verification(self, note: VerificationNote) -> "PassingDSL":
        log = list(self.verification_log) + [note]
        out = replace(self, verification_log=log)
        if note.revision_triggered:
            out = replace(
                out,
                revision=out.revision + 1,
                tools_pending=[],
                tools_completed=[],
                maneuver=ManeuverPlan(),
            )
        return out

    def set_maneuver(self, maneuver: ManeuverPlan) -> "PassingDSL":
        return replace(self, maneuver=maneuver)


def init_dsl_from_request(
    text: str,
    *,
    start: str = "A",
    goal: str = "B",
    deadline_s: float = 90.0,
    urgency: UrgencyKind = "medium",
    aggression: str = "low",
    road_type: RoadKind = "highway",
) -> PassingDSL:
    mission = MissionSpec(
        text=text,
        start=start,
        goal=goal,
        deadline_s=deadline_s,
        urgency=urgency,
        aggression=aggression,
    )
    route = RoutePlan(
        description=f"Route {start} → {goal}",
        waypoints=[RouteWaypoint(street="Main", speed_mps=13.4)],
        road_type=road_type,
    )
    return PassingDSL(mission=mission, route=route, tools_pending=[], tools_completed=[])


def dsl_to_dict(dsl: PassingDSL) -> Dict[str, Any]:
    return asdict(dsl)


def dsl_from_dict(d: Dict[str, Any]) -> PassingDSL:
    mission = MissionSpec(**d["mission"])
    route_d = d.get("route", {})
    wps = [RouteWaypoint(**w) for w in route_d.get("waypoints", [])]
    route = RoutePlan(
        description=route_d.get("description", ""),
        waypoints=wps,
        eta_s=route_d.get("eta_s", 60.0),
        road_type=route_d.get("road_type", "highway"),
    )
    man_d = d.get("maneuver", {})
    maneuver = ManeuverPlan(**man_d) if man_d else ManeuverPlan()
    perception_log = [PerceptionRecord(**p) for p in d.get("perception_log", [])]
    verification_log = [VerificationNote(**v) for v in d.get("verification_log", [])]
    wb_d = d.get("world_belief", {})
    actors_raw = wb_d.get("actors", {})
    actors = {k: ActorBelief(**v) if isinstance(v, dict) else v for k, v in actors_raw.items()}
    world_belief = WorldBelief(
        source=wb_d.get("source", "initial"),
        t_s=wb_d.get("t_s", 0.0),
        ego_lane=wb_d.get("ego_lane", 0),
        ego_speed_mps=wb_d.get("ego_speed_mps", 0.0),
        progress_m=wb_d.get("progress_m", 0.0),
        front_gap_m=wb_d.get("front_gap_m"),
        rear_gap_m=wb_d.get("rear_gap_m"),
        oncoming_gap_m=wb_d.get("oncoming_gap_m"),
        front_valid=wb_d.get("front_valid", False),
        rear_valid=wb_d.get("rear_valid", False),
        oncoming_valid=wb_d.get("oncoming_valid", False),
        oncoming_available=wb_d.get("oncoming_available", True),
        oncoming_unavailable_reason=wb_d.get("oncoming_unavailable_reason", ""),
        visibility_m=wb_d.get("visibility_m"),
        lead_speed_mps=wb_d.get("lead_speed_mps"),
        rear_closing_mps=wb_d.get("rear_closing_mps"),
        oncoming_approach_mps=wb_d.get("oncoming_approach_mps"),
        depth_confidence=wb_d.get("depth_confidence", 0.0),
        car_distances=wb_d.get("car_distances", []),
        actors=actors,
        physics_valid=wb_d.get("physics_valid", True),
        physics_issues=wb_d.get("physics_issues", []),
    )
    return PassingDSL(
        mission=mission,
        route=route,
        maneuver=maneuver,
        perception_log=perception_log,
        execution_log=[ExecutionRecord(**e) for e in d.get("execution_log", [])],
        verification_log=verification_log,
        tools_completed=d.get("tools_completed", []),
        tools_pending=d.get("tools_pending", []),
        revision=d.get("revision", 0),
        world_belief=world_belief,
    )
