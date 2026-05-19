"""LLM agents for decisions that are hard to hard-code; deterministic mocks for demo/tests."""
from __future__ import annotations

import os
from typing import Any, Dict, Type, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class TravelRequest(BaseModel):
    starting_point: str
    goal: str
    aggressive_level: str


class RearPassingTimeEstimate(BaseModel):
    required_lane_change_time_s: float = Field(description="Seconds needed to complete lane change safely")
    reasoning: str = ""


class TargetVelocityDecision(BaseModel):
    target_speed_mps: float = Field(description="Recommended passing speed in m/s")
    reasoning: str = ""


class TrafficCheckDecision(BaseModel):
    needs_traffic_check: bool
    is_real_traffic: bool
    reasoning: str = ""


class RoadConditionAssessment(BaseModel):
    road_type: str = Field(description="highway, urban, or suburban")
    lane_density: float = Field(description="Approximate cars per 100m in passing lane")
    reasoning: str = ""


class ReplanDecision(BaseModel):
    should_replan: bool
    route_description: str = ""
    estimated_time_s: float = 0.0
    waypoints: list = Field(default_factory=list)
    reasoning: str = ""


def use_mock_llm() -> bool:
    return os.environ.get("AUTOPASS_MOCK_LLM", "1").strip() not in ("0", "false", "False")


def structured_invoke(model: Type[T], system: str, human: str, mock_value: T) -> T:
    if use_mock_llm():
        return mock_value
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=os.environ.get("AUTOPASS_LLM_MODEL", "gpt-4o-mini"), temperature=0)
    return llm.with_structured_output(model).invoke([SystemMessage(content=system), HumanMessage(content=human)])


def parse_travel_request(text: str) -> TravelRequest:
    lower = text.lower()
    aggressive = "high" if any(w in lower for w in ("hurry", "rush", "asap", "quick", "10 min", "5 min")) else "low"
    if any(w in lower for w in ("no pass", "stay in lane", "safe mode")):
        aggressive = "0"
    goal = "Airport" if "airport" in lower else "Destination"
    start = "unknown"
    if "from downtown" in lower:
        start = "Downtown Mall"
    return structured_invoke(
        TravelRequest,
        "Parse autonomous-vehicle travel requests.",
        text,
        TravelRequest(starting_point=start, goal=goal, aggressive_level=aggressive),
    )


def estimate_rear_passing_time(
    back_distance_m: float,
    back_closing_rate_mps: float,
    ego_speed_mps: float,
    passing_side: str,
) -> RearPassingTimeEstimate:
    base = 3.5 + max(0.0, 8.0 - back_distance_m / 15.0) + max(0.0, back_closing_rate_mps - 1.0)
    mock = RearPassingTimeEstimate(
        required_lane_change_time_s=round(min(8.0, max(2.5, base)), 2),
        reasoning=f"Lane change on {passing_side} with rear gap {back_distance_m:.0f}m, closing {back_closing_rate_mps:.1f}m/s",
    )
    return structured_invoke(
        RearPassingTimeEstimate,
        "Estimate seconds required for a safe lane change given rear traffic.",
        f"ego_speed={ego_speed_mps}, back_distance={back_distance_m}, closing_rate={back_closing_rate_mps}, side={passing_side}",
        mock,
    )


def decide_target_velocity(
    front_speed_mps: float,
    speed_limit_mps: float,
    road_type: str,
    ego_speed_mps: float,
) -> TargetVelocityDecision:
    if road_type == "highway":
        cap = speed_limit_mps
        target = min(cap, max(front_speed_mps + 4.0, front_speed_mps * 1.15))
    elif road_type == "urban":
        target = min(speed_limit_mps, front_speed_mps + 2.5)
    else:
        target = min(speed_limit_mps, front_speed_mps + 3.5)
    target = max(ego_speed_mps, min(target, speed_limit_mps))
    mock = TargetVelocityDecision(target_speed_mps=round(target, 2), reasoning=f"{road_type} road cap {speed_limit_mps:.1f} m/s")
    return structured_invoke(
        TargetVelocityDecision,
        "Choose a safe passing target speed (m/s) using front car speed, speed limit, and road type.",
        f"front={front_speed_mps}, limit={speed_limit_mps}, road={road_type}, ego={ego_speed_mps}",
        mock,
    )


def traffic_check(
    move_but_not_pass_count: int,
    ego_speed_mps: float,
    speed_limit_mps: float,
    road_type: str,
    lane_density: float,
) -> TrafficCheckDecision:
    needs = move_but_not_pass_count >= 2 or (road_type == "urban" and move_but_not_pass_count >= 1)
    is_traffic = lane_density >= 0.35 or (needs and ego_speed_mps < 0.55 * speed_limit_mps)
    mock = TrafficCheckDecision(needs_traffic_check=needs, is_real_traffic=is_traffic, reasoning=f"density={lane_density:.2f}")
    return structured_invoke(
        TrafficCheckDecision,
        "Decide if repeated move-but-not-pass is real traffic vs road layout.",
        f"count={move_but_not_pass_count}, ego={ego_speed_mps}, limit={speed_limit_mps}, road={road_type}, density={lane_density}",
        mock,
    )


def assess_road(lane_density: float, road_type: str) -> RoadConditionAssessment:
    mock = RoadConditionAssessment(road_type=road_type, lane_density=lane_density, reasoning="segmentation density")
    return structured_invoke(
        RoadConditionAssessment,
        "Summarize road conditions for replanning.",
        f"road={road_type}, density={lane_density}",
        mock,
    )


def replan_route(goal: str, current_plan: list, lane_density: float, trip_eta: float) -> ReplanDecision:
    should = lane_density >= 0.35
    wps = current_plan if not should else [
        {"street": "Alternate Route", "action": "merge", "speed": 22.0},
        {"street": "Harbor Drive", "action": "drive", "speed": 20.0},
    ]
    mock = ReplanDecision(
        should_replan=should,
        route_description="Bypass congested segment" if should else "Keep current plan",
        estimated_time_s=trip_eta * (0.92 if should else 1.0),
        waypoints=wps,
        reasoning="Traffic density triggered alternate route" if should else "No replan needed",
    )
    return structured_invoke(
        ReplanDecision,
        "Decide whether to replan given traffic; output waypoints if yes.",
        f"goal={goal}, density={lane_density}, eta={trip_eta}",
        mock,
    )


def plans_are_same(plan_a: list, plan_b: list) -> bool:
    if len(plan_a) != len(plan_b):
        return False
    for a, b in zip(plan_a, plan_b):
        if isinstance(a, dict) and isinstance(b, dict):
            if a.get("street") != b.get("street") or a.get("action") != b.get("action"):
                return False
        elif a != b:
            return False
    return True
