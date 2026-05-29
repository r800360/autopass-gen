"""Safety physics — vision gaps and measured speeds only."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from autopass.dsl import PassingDSL
from autopass.perception_state import measured_gaps, measured_speeds, urgency_margin_scale
from visual_world import ScenarioSpec, WorldState

HARD_FLOOR_REAR_M = 12.0
HARD_FLOOR_ONCOMING_M = 10.0
HARD_FLOOR_VISIBILITY_M = 25.0
HARD_FLOOR_FRONT_M = 6.0
HARD_FLOOR_TTC_S = 2.5


@dataclass
class SafetyResult:
    approved: bool
    reasons: List[str] = field(default_factory=list)
    min_ttc_s: float = math.inf
    risk_score: float = 0.0


def estimate_pass_time(front_m: float, ego_mps: float, lead_mps: float) -> float:
    relative_gain = max(1.5, ego_mps - lead_mps + 1.0)
    return min(12.0, max(3.0, (front_m + 13.0) / relative_gain))


def check_pass_safety(
    dsl: PassingDSL,
    spec: ScenarioSpec,
    world: WorldState,
    *,
    visibility_m: Optional[float] = None,
) -> SafetyResult:
    gaps = measured_gaps(dsl)
    speeds = measured_speeds(dsl, world)
    front_m = gaps["front_m"]
    rear_m = gaps["rear_m"]
    oncoming_m = gaps["oncoming_m"]
    vis = visibility_m if visibility_m is not None else gaps["visibility_m"]

    scale = urgency_margin_scale(spec, world)
    reasons: List[str] = []
    t_pass = estimate_pass_time(front_m, speeds["ego_mps"], speeds["lead_mps"])
    buffer_m = 12.0 * scale

    from autopass.pass_gates import rear_closing_from_log

    rear_closing, rear_closing_trusted, _ = rear_closing_from_log(dsl)
    if not rear_closing_trusted:
        rear_closing = 0.0
    rear_ttc = math.inf if rear_closing <= 1e-6 else rear_m / rear_closing

    wb = dsl.world_belief
    oncoming_closing = speeds["oncoming_closing_mps"]
    if wb.oncoming_available is False or wb.oncoming_gap_m is None:
        oncoming_ttc = math.inf
        required_oncoming = 0.0
    else:
        oncoming_ttc = oncoming_m / max(1e-6, oncoming_closing)
        required_oncoming = max(HARD_FLOOR_ONCOMING_M, oncoming_closing * t_pass + buffer_m)
    min_ttc = min(rear_ttc, oncoming_ttc)

    required_visibility = max(HARD_FLOOR_VISIBILITY_M, speeds["ego_mps"] * t_pass + buffer_m)
    required_rear = max(HARD_FLOOR_REAR_M, 16.0 * scale + rear_closing * 2.0 * scale)

    if rear_m < required_rear:
        reasons.append(f"rear gap too small: {rear_m:.1f} < {required_rear:.1f} m")
    if wb.oncoming_available is not False and wb.oncoming_gap_m is not None:
        if oncoming_m < required_oncoming:
            reasons.append(f"oncoming gap too small: {oncoming_m:.1f} < {required_oncoming:.1f} m")
    if vis < required_visibility:
        reasons.append(f"visibility too low: {vis:.1f} < {required_visibility:.1f} m")
    if front_m < HARD_FLOOR_FRONT_M:
        reasons.append(f"front distance too small: {front_m:.1f} m")
    if min_ttc < HARD_FLOOR_TTC_S:
        reasons.append(f"TTC too low: {min_ttc:.2f} s")

    risk = 0.0
    risk += min(2.0, required_oncoming / max(1e-6, oncoming_m))
    risk += min(2.0, required_visibility / max(1e-6, vis))
    risk += min(2.0, required_rear / max(1e-6, rear_m))
    risk /= 6.0
    return SafetyResult(approved=not reasons, reasons=reasons, min_ttc_s=min_ttc, risk_score=risk)


def check_pass_safety_legacy(
    spec: ScenarioSpec,
    world: WorldState,
    *,
    front_m: float,
    rear_m: float,
    oncoming_m: float,
    visibility_m: float = 200.0,
    lead_mps: float,
    rear_closing_mps: float,
    oncoming_closing_mps: float,
) -> SafetyResult:
    """Test helper with explicit measured inputs (no DSL)."""
    from autopass.dsl import PassingDSL, init_dsl_from_request
    from autopass.dsl import PerceptionRecord

    dsl = init_dsl_from_request("test")
    dsl = dsl.append_perception(
        PerceptionRecord(
            tool="capture_sensors",
            summary="test",
            data={"front_speed_mps": lead_mps, "rear_closing_mps": rear_closing_mps},
        )
    )
    from autopass.belief import gaps_from_car_distances

    gaps = {"front_gap_m": front_m, "rear_gap_m": rear_m, "oncoming_gap_m": oncoming_m}
    from dataclasses import replace

    dsl = dsl.update_belief(
        replace(
            dsl.world_belief,
            source="visual_depth",
            front_gap_m=front_m,
            rear_gap_m=rear_m,
            oncoming_gap_m=oncoming_m,
            front_valid=front_m < 200.0,
            rear_valid=rear_m < 200.0,
            oncoming_valid=oncoming_m < 200.0,
            oncoming_available=True,
            visibility_m=visibility_m,
            lead_speed_mps=lead_mps,
            rear_closing_mps=rear_closing_mps,
            oncoming_approach_mps=max(0.0, oncoming_closing_mps - world.ego_speed_mps),
        )
    )
    return check_pass_safety(dsl, spec, world, visibility_m=visibility_m)
