"""
AutoPass-Gen LangGraph visual closed-loop prototype.

What this implements:
    ScenarioSpec -> Request/Urgency Interpreter -> Visual Renderer
    -> RGB + semantic segmentation + depth map -> Perception/Map extraction
    -> Passing Planning Agent -> Safety Checker -> Execution -> Evaluator
    -> Feedback/mutation loop.

This is intentionally simulator-independent. The perception module does not read
privileged distances directly. It renders a concrete top-down visual scene, a
semantic mask, and a depth map, then extracts front/rear/oncoming/visibility
estimates from those rendered sensor products. CARLA can replace the renderer
without changing the LangGraph agent interface.

Run:
    pip install langgraph matplotlib numpy pillow pytest
    python autopass_langgraph_demo_final.py --mode demo --out-dir runs/demo
    python autopass_langgraph_demo_final.py --mode closed_loop --rounds 3 --n 8 --out-dir runs/closed_loop
    pytest -q test_autopass_langgraph_demo_final.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

import numpy as np

Action = Literal["pass", "wait", "replan"]
PolicyName = Literal["autopass", "no_pass", "aggressive"]
LABELS = {"background": 0, "ego": 1, "lead": 2, "rear": 3, "oncoming": 4, "occlusion": 5, "road": 6}
LABEL_NAMES = {v: k for k, v in LABELS.items()}


@dataclass(frozen=True)
class RequestSpec:
    text: str
    start: str = "A"
    goal: str = "B"
    deadline_s: float = 90.0


@dataclass(frozen=True)
class VehicleSpec:
    distance_m: float
    speed_mps: float
    accel_mps2: float = 0.0


@dataclass(frozen=True)
class OcclusionSpec:
    kind: str = "none"
    severity: float = 0.0
    sight_distance_m: float = 140.0


@dataclass(frozen=True)
class WeatherSpec:
    rain: float = 0.0
    fog: float = 0.0
    sun_angle_deg: float = 45.0


@dataclass(frozen=True)
class SensorSpec:
    mode: Literal["rgb_depth_seg"] = "rgb_depth_seg"
    noise_std_m: float = 0.15


@dataclass(frozen=True)
class RouteSpec:
    town: str = "SyntheticTown"
    start_x_m: float = 0.0
    goal_x_m: float = 180.0
    lane_width_m: float = 3.6
    speed_limit_mps: float = 13.4


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    route: RouteSpec
    request: RequestSpec
    ego_speed_mps: float
    lead: VehicleSpec
    rear: VehicleSpec
    oncoming: VehicleSpec
    occlusion: OcclusionSpec
    weather: WeatherSpec
    sensor: SensorSpec


@dataclass
class WorldState:
    t_s: float = 0.0
    ego_x_m: float = 0.0
    ego_lane: int = 0
    ego_speed_mps: float = 10.0
    lead_x_m: float = 28.0
    rear_x_m: float = -80.0
    oncoming_x_m: float = 180.0
    passed: bool = False
    collision: bool = False
    done: bool = False


@dataclass
class UrgencyState:
    urgency_level: Literal["low", "medium", "high"]
    delay_cost: float
    deadline_pressure: float


@dataclass
class PassState:
    front_distance_m: float
    rear_distance_m: float
    oncoming_distance_m: float
    ego_speed_mps: float
    lead_speed_mps: float
    rear_speed_mps: float
    oncoming_speed_mps: float
    visibility_m: float
    deadline_pressure: float
    urgency_level: str
    time_to_goal_s: float
    lane: int


@dataclass
class SafetyResult:
    approved: bool
    reasons: List[str] = field(default_factory=list)
    min_ttc_s: float = math.inf
    risk_score: float = 0.0


class GraphState(TypedDict, total=False):
    spec: Dict[str, Any]
    policy: PolicyName
    world: Dict[str, Any]
    urgency: Dict[str, Any]
    perception: Dict[str, Any]
    pass_state: Dict[str, Any]
    proposed_action: Action
    approved_action: Action
    safety: Dict[str, Any]
    trace: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    out_dir: str


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def finite_or_none(x: float) -> Optional[float]:
    return None if math.isinf(x) or math.isnan(x) else round(float(x), 3)


def dict_to_spec(d: Dict[str, Any]) -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id=d["scenario_id"],
        route=RouteSpec(**d["route"]),
        request=RequestSpec(**d["request"]),
        ego_speed_mps=d["ego_speed_mps"],
        lead=VehicleSpec(**d["lead"]),
        rear=VehicleSpec(**d["rear"]),
        oncoming=VehicleSpec(**d["oncoming"]),
        occlusion=OcclusionSpec(**d["occlusion"]),
        weather=WeatherSpec(**d["weather"]),
        sensor=SensorSpec(**d["sensor"]),
    )


def initialize_world(spec: ScenarioSpec) -> WorldState:
    return WorldState(
        ego_x_m=spec.route.start_x_m,
        ego_lane=0,
        ego_speed_mps=spec.ego_speed_mps,
        lead_x_m=spec.route.start_x_m + spec.lead.distance_m,
        rear_x_m=spec.route.start_x_m - spec.rear.distance_m,
        oncoming_x_m=spec.route.start_x_m + spec.oncoming.distance_m,
    )


def generate_scenario(i: int, seed: int = 0, difficulty: float = 0.50) -> ScenarioSpec:
    """General stochastic generator.

    difficulty in [0, 1] narrows gaps, tightens deadlines, and increases occlusion.
    The demo mode uses curated cases below, while closed_loop uses this plus mutation.
    """
    rng = random.Random(seed * 100_000 + i)
    d = clamp(difficulty, 0.0, 1.0)
    deadline = rng.uniform(42.0 - 24.0 * d, 55.0 - 22.0 * d)
    lead_speed = rng.uniform(5.0, 7.8)
    ego_speed = rng.uniform(11.0, 13.2)
    front = rng.uniform(22.0, 36.0)
    rear_dist = rng.uniform(95.0 - 45.0 * d, 130.0 - 55.0 * d)
    rear_speed = rng.uniform(8.0, 13.5 + 2.0 * d)
    oncoming_dist = rng.uniform(250.0 - 145.0 * d, 320.0 - 160.0 * d)
    oncoming_speed = rng.uniform(8.0, 13.5 + 2.0 * d)
    severity = clamp(rng.choice([0.0, 0.2, 0.4, 0.65]) + 0.25 * d, 0.0, 0.9)
    sight = rng.uniform(150.0 - 80.0 * d, 210.0 - 100.0 * d) * (1.0 - 0.35 * severity)
    return ScenarioSpec(
        scenario_id=f"scenario_{i:03d}_d{int(100*d):02d}",
        route=RouteSpec(goal_x_m=180.0),
        request=RequestSpec(text=f"I need to go from A to B within {deadline:.0f} seconds.", deadline_s=deadline),
        ego_speed_mps=ego_speed,
        lead=VehicleSpec(distance_m=front, speed_mps=lead_speed),
        rear=VehicleSpec(distance_m=rear_dist, speed_mps=rear_speed),
        oncoming=VehicleSpec(distance_m=oncoming_dist, speed_mps=oncoming_speed),
        occlusion=OcclusionSpec(kind="parked_cars" if severity > 0.05 else "none", severity=severity, sight_distance_m=sight),
        weather=WeatherSpec(rain=rng.uniform(0.0, 0.35 + 0.25 * d), fog=rng.uniform(0.0, 0.35 + 0.30 * d)),
        sensor=SensorSpec(noise_std_m=0.10 + 0.50 * severity),
    )


def curated_demo_scenarios() -> List[ScenarioSpec]:
    """Small deterministic set that demonstrates easy, hard, and boundary cases.

    This avoids the demo depending on lucky random seeds while still exercising the
    exact same LangGraph/perception/safety/evaluation pipeline.
    """
    base_route = RouteSpec(goal_x_m=180.0, speed_limit_mps=13.4)
    return [
        ScenarioSpec(
            scenario_id="demo_01_clear_urgent_safe_pass",
            route=base_route,
            request=RequestSpec(text="I need to reach B very soon.", deadline_s=18.0),
            ego_speed_mps=12.8,
            lead=VehicleSpec(distance_m=24.0, speed_mps=5.6),
            rear=VehicleSpec(distance_m=120.0, speed_mps=9.0),
            oncoming=VehicleSpec(distance_m=290.0, speed_mps=9.5),
            occlusion=OcclusionSpec(kind="none", severity=0.0, sight_distance_m=210.0),
            weather=WeatherSpec(),
            sensor=SensorSpec(noise_std_m=0.05),
        ),
        ScenarioSpec(
            scenario_id="demo_02_unsafe_oncoming_rejected",
            route=base_route,
            request=RequestSpec(text="I am late, but do not drive unsafely.", deadline_s=20.0),
            ego_speed_mps=12.5,
            lead=VehicleSpec(distance_m=24.0, speed_mps=5.8),
            rear=VehicleSpec(distance_m=115.0, speed_mps=9.0),
            oncoming=VehicleSpec(distance_m=74.0, speed_mps=13.0),
            occlusion=OcclusionSpec(kind="none", severity=0.0, sight_distance_m=200.0),
            weather=WeatherSpec(),
            sensor=SensorSpec(noise_std_m=0.05),
        ),
        ScenarioSpec(
            scenario_id="demo_03_occluded_boundary_replan",
            route=base_route,
            request=RequestSpec(text="Need to arrive quickly through a visually blocked road.", deadline_s=24.0),
            ego_speed_mps=12.0,
            lead=VehicleSpec(distance_m=28.0, speed_mps=6.0),
            rear=VehicleSpec(distance_m=95.0, speed_mps=10.0),
            oncoming=VehicleSpec(distance_m=230.0, speed_mps=10.0),
            occlusion=OcclusionSpec(kind="parked_cars", severity=0.75, sight_distance_m=58.0),
            weather=WeatherSpec(rain=0.25, fog=0.20),
            sensor=SensorSpec(noise_std_m=0.15),
        ),
        ScenarioSpec(
            scenario_id="demo_04_low_urgency_wait_is_ok",
            route=base_route,
            request=RequestSpec(text="Please drive to B when convenient.", deadline_s=52.0),
            ego_speed_mps=12.0,
            lead=VehicleSpec(distance_m=30.0, speed_mps=7.0),
            rear=VehicleSpec(distance_m=100.0, speed_mps=10.0),
            oncoming=VehicleSpec(distance_m=250.0, speed_mps=10.0),
            occlusion=OcclusionSpec(kind="none", severity=0.0, sight_distance_m=190.0),
            weather=WeatherSpec(),
            sensor=SensorSpec(noise_std_m=0.05),
        ),
        ScenarioSpec(
            scenario_id="demo_05_fast_rear_rejected",
            route=base_route,
            request=RequestSpec(text="I need to arrive quickly, but rear traffic is closing.", deadline_s=22.0),
            ego_speed_mps=12.2,
            lead=VehicleSpec(distance_m=25.0, speed_mps=5.8),
            rear=VehicleSpec(distance_m=20.0, speed_mps=17.0),
            oncoming=VehicleSpec(distance_m=270.0, speed_mps=9.0),
            occlusion=OcclusionSpec(kind="none", severity=0.0, sight_distance_m=210.0),
            weather=WeatherSpec(),
            sensor=SensorSpec(noise_std_m=0.05),
        ),
        ScenarioSpec(
            scenario_id="demo_06_medium_safe_selective_pass",
            route=base_route,
            request=RequestSpec(text="Try to arrive on time if there is a safe opening.", deadline_s=25.0),
            ego_speed_mps=12.3,
            lead=VehicleSpec(distance_m=32.0, speed_mps=6.2),
            rear=VehicleSpec(distance_m=105.0, speed_mps=10.0),
            oncoming=VehicleSpec(distance_m=245.0, speed_mps=10.0),
            occlusion=OcclusionSpec(kind="none", severity=0.0, sight_distance_m=185.0),
            weather=WeatherSpec(rain=0.05, fog=0.05),
            sensor=SensorSpec(noise_std_m=0.08),
        ),
    ]


# ------------------------ Visual sensor backend ------------------------

def _world_to_pixel(x_rel: float, lane: int, x_min: float, x_max: float, width: int, height: int) -> Tuple[int, int]:
    col = int(round((x_rel - x_min) / (x_max - x_min) * (width - 1)))
    y_norm = 0.26 if lane == 0 else 0.72
    row = int(round((1.0 - y_norm) * (height - 1)))
    return col, row


def _draw_rect(rgb: np.ndarray, seg: np.ndarray, depth: np.ndarray, cx: int, cy: int, w: int, h: int, label: int, d_m: float) -> None:
    h_img, w_img = seg.shape
    x0, x1 = max(0, cx - w // 2), min(w_img, cx + w // 2 + 1)
    y0, y1 = max(0, cy - h // 2), min(h_img, cy + h // 2 + 1)
    if x0 >= x1 or y0 >= y1:
        return
    seg[y0:y1, x0:x1] = label
    depth[y0:y1, x0:x1] = d_m
    # Keep RGB meaningful without relying on external image assets.
    shade = 60 + 25 * label
    rgb[y0:y1, x0:x1, :] = np.array([shade, max(20, 210 - 20 * label), 80 + 10 * label], dtype=np.uint8)


def render_sensor_frame(spec: ScenarioSpec, world: WorldState, width: int = 640, height: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Render RGB, semantic segmentation, and metric depth maps.

    The planner will extract distances from these arrays. This is the key swap
    point for CARLA RGB/depth/semantic cameras.
    """
    x_min, x_max = -110.0, 330.0
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, :] = np.array([35, 35, 35], dtype=np.uint8)
    seg = np.zeros((height, width), dtype=np.uint8)
    depth = np.full((height, width), np.inf, dtype=np.float32)

    # Road lanes.
    row_own0, row_own1 = int(height * 0.58), int(height * 0.93)
    row_pass0, row_pass1 = int(height * 0.13), int(height * 0.48)
    for r0, r1 in [(row_own0, row_own1), (row_pass0, row_pass1)]:
        rgb[r0:r1, :, :] = np.array([70, 70, 70], dtype=np.uint8)
        seg[r0:r1, :] = LABELS["road"]
        # Approximate road depth by column distance from ego.
        xs = np.linspace(x_min, x_max, width, dtype=np.float32)
        depth[r0:r1, :] = np.abs(xs)[None, :]

    # Visibility/occlusion limit from geometry + weather.
    visibility = min(spec.occlusion.sight_distance_m, 210.0)
    visibility *= (1.0 - 0.25 * spec.weather.fog - 0.15 * spec.weather.rain)
    visibility = max(20.0, visibility)
    vis_col, _ = _world_to_pixel(visibility, 1, x_min, x_max, width, height)
    if 0 <= vis_col < width:
        seg[:, vis_col : min(width, vis_col + 5)] = LABELS["occlusion"]
        depth[:, vis_col : min(width, vis_col + 5)] = visibility
        rgb[:, vis_col : min(width, vis_col + 5), :] = np.array([130, 90, 60], dtype=np.uint8)
        if spec.occlusion.severity > 0:
            rgb[:, vis_col:, :] = (rgb[:, vis_col:, :].astype(np.float32) * (1.0 - 0.25 * spec.occlusion.severity)).astype(np.uint8)

    vehicles = [
        ("ego", 0.0, world.ego_lane, LABELS["ego"]),
        ("lead", world.lead_x_m - world.ego_x_m, 0, LABELS["lead"]),
        ("rear", world.rear_x_m - world.ego_x_m, 1, LABELS["rear"]),
        ("oncoming", world.oncoming_x_m - world.ego_x_m, 1, LABELS["oncoming"]),
    ]
    for _, x_rel, lane, label in vehicles:
        cx, cy = _world_to_pixel(x_rel, lane, x_min, x_max, width, height)
        _draw_rect(rgb, seg, depth, cx, cy, 22, 13, label, abs(float(x_rel)))

    # Simple weather degradation on RGB only. Seg/depth remain rendered products.
    if spec.weather.fog > 0 or spec.weather.rain > 0:
        alpha = clamp(0.22 * spec.weather.fog + 0.10 * spec.weather.rain, 0.0, 0.45)
        rgb = (rgb.astype(np.float32) * (1 - alpha) + 190 * alpha).astype(np.uint8)

    meta = {"x_min_m": x_min, "x_max_m": x_max, "width": width, "height": height, "visibility_truth_m": visibility}
    return rgb, seg, depth, meta


def _median_depth_for_label(seg: np.ndarray, depth: np.ndarray, label: int, fallback: float) -> float:
    vals = depth[(seg == label) & np.isfinite(depth)]
    if vals.size == 0:
        return fallback
    return float(np.median(vals))


def extract_pass_state_from_sensors(spec: ScenarioSpec, world: WorldState, urgency: UrgencyState, rgb: np.ndarray, seg: np.ndarray, depth: np.ndarray) -> Tuple[Dict[str, Any], PassState]:
    noise = spec.sensor.noise_std_m
    # Deterministic pseudo-noise from scenario/time so tests are stable.
    rng = random.Random(hash((spec.scenario_id, round(world.t_s, 2))) & 0xFFFFFFFF)

    def noisy(x: float) -> float:
        return max(0.0, x + rng.gauss(0.0, noise))

    front = noisy(_median_depth_for_label(seg, depth, LABELS["lead"], 999.0))
    rear = noisy(_median_depth_for_label(seg, depth, LABELS["rear"], 999.0))
    oncoming = noisy(_median_depth_for_label(seg, depth, LABELS["oncoming"], 999.0))
    visibility = noisy(_median_depth_for_label(seg, depth, LABELS["occlusion"], 210.0))

    perception = {
        "sensor_backend": "rendered_rgb_segmentation_depth",
        "rgb_shape": list(rgb.shape),
        "segmentation_labels_present": sorted([LABEL_NAMES.get(int(v), str(int(v))) for v in np.unique(seg)]),
        "depth": {
            "front_m": round(front, 3),
            "rear_m": round(rear, 3),
            "oncoming_m": round(oncoming, 3),
            "visibility_m": round(visibility, 3),
        },
        "segmentation": {
            "ego_pixels": int(np.sum(seg == LABELS["ego"])),
            "lead_pixels": int(np.sum(seg == LABELS["lead"])),
            "rear_pixels": int(np.sum(seg == LABELS["rear"])),
            "oncoming_pixels": int(np.sum(seg == LABELS["oncoming"])),
            "occlusion_pixels": int(np.sum(seg == LABELS["occlusion"])),
        },
    }
    ps = PassState(
        front_distance_m=front,
        rear_distance_m=rear,
        oncoming_distance_m=oncoming,
        ego_speed_mps=world.ego_speed_mps,
        lead_speed_mps=spec.lead.speed_mps,
        rear_speed_mps=spec.rear.speed_mps,
        oncoming_speed_mps=spec.oncoming.speed_mps,
        visibility_m=visibility,
        deadline_pressure=urgency.deadline_pressure,
        urgency_level=urgency.urgency_level,
        time_to_goal_s=max(0.0, spec.route.goal_x_m - world.ego_x_m) / max(1e-6, world.ego_speed_mps),
        lane=world.ego_lane,
    )
    return perception, ps


def visual_perception(spec: ScenarioSpec, world: WorldState, urgency: UrgencyState) -> Tuple[Dict[str, Any], PassState, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rgb, seg, depth, _ = render_sensor_frame(spec, world)
    perception, ps = extract_pass_state_from_sensors(spec, world, urgency, rgb, seg, depth)
    return perception, ps, (rgb, seg, depth)


# Backward-compatible helper for tests and quick inspection.
def synthetic_perception(spec: ScenarioSpec, world: WorldState) -> Tuple[Dict[str, Any], PassState]:
    urgency = UrgencyState(**node_interpret_request({"spec": asdict(spec), "world": asdict(world)})["urgency"])
    perception, ps, _ = visual_perception(spec, world, urgency)
    return perception, ps


# ------------------------ LangGraph nodes ------------------------

def node_interpret_request(state: GraphState) -> Dict[str, Any]:
    spec = dict_to_spec(state["spec"])
    world = WorldState(**state.get("world", asdict(initialize_world(spec))))
    remaining = max(1e-6, spec.request.deadline_s - world.t_s)
    nominal_time = max(0.0, spec.route.goal_x_m - world.ego_x_m) / max(1e-6, spec.route.speed_limit_mps)
    pressure = clamp(nominal_time / remaining, 0.0, 2.5)
    if pressure > 0.90:
        level: Literal["low", "medium", "high"] = "high"
    elif pressure > 0.62:
        level = "medium"
    else:
        level = "low"
    return {"urgency": asdict(UrgencyState(level, delay_cost=pressure * 10.0, deadline_pressure=pressure))}


def node_perception_map(state: GraphState) -> Dict[str, Any]:
    spec = dict_to_spec(state["spec"])
    world = WorldState(**state["world"])
    urgency = UrgencyState(**state["urgency"])
    perception, ps, _ = visual_perception(spec, world, urgency)
    return {"perception": perception, "pass_state": asdict(ps)}


def estimate_pass_time(ps: PassState) -> float:
    relative_gain = max(1.5, ps.ego_speed_mps - ps.lead_speed_mps + 1.0)
    distance_to_clear = ps.front_distance_m + 13.0
    return clamp(distance_to_clear / relative_gain, 3.0, 12.0)


def node_planning_agent(state: GraphState) -> Dict[str, Any]:
    policy = state.get("policy", "autopass")
    ps = PassState(**state["pass_state"])
    slow_lead = ps.front_distance_m < 48.0 and ps.lead_speed_mps < 0.88 * ps.ego_speed_mps

    if policy == "no_pass":
        return {"proposed_action": "wait"}
    if policy == "aggressive":
        return {"proposed_action": "pass" if slow_lead else "wait"}

    urgent_enough = ps.urgency_level in {"medium", "high"} or ps.deadline_pressure > 0.65
    progress_gain = estimate_pass_time(ps) < max(4.0, ps.time_to_goal_s * 0.75)
    if slow_lead and urgent_enough and progress_gain:
        return {"proposed_action": "pass"}
    if ps.deadline_pressure > 1.20 and not slow_lead:
        return {"proposed_action": "replan"}
    return {"proposed_action": "wait"}


def check_pass_safety(spec: ScenarioSpec, ps: PassState) -> SafetyResult:
    reasons: List[str] = []
    t_pass = estimate_pass_time(ps)
    buffer_m = 12.0

    rear_closing_speed = max(0.0, ps.rear_speed_mps - ps.ego_speed_mps)
    rear_ttc = math.inf if rear_closing_speed <= 1e-6 else ps.rear_distance_m / rear_closing_speed
    oncoming_closing_speed = ps.ego_speed_mps + ps.oncoming_speed_mps
    oncoming_ttc = ps.oncoming_distance_m / max(1e-6, oncoming_closing_speed)
    min_ttc = min(rear_ttc, oncoming_ttc)

    required_oncoming_gap = oncoming_closing_speed * t_pass + buffer_m
    required_visibility = ps.ego_speed_mps * t_pass + buffer_m
    required_rear_gap = 16.0 + rear_closing_speed * 2.0

    if ps.rear_distance_m < required_rear_gap:
        reasons.append(f"rear gap too small: {ps.rear_distance_m:.1f} < {required_rear_gap:.1f} m")
    if ps.oncoming_distance_m < required_oncoming_gap:
        reasons.append(f"oncoming gap too small: {ps.oncoming_distance_m:.1f} < {required_oncoming_gap:.1f} m")
    if ps.visibility_m < required_visibility:
        reasons.append(f"visibility too low: {ps.visibility_m:.1f} < {required_visibility:.1f} m")
    if ps.front_distance_m < 8.0:
        reasons.append("front distance too small for lane change")
    if min_ttc < 3.0:
        reasons.append(f"TTC too low: {min_ttc:.2f} s")

    risk = 0.0
    risk += clamp(required_oncoming_gap / max(1e-6, ps.oncoming_distance_m), 0, 2)
    risk += clamp(required_visibility / max(1e-6, ps.visibility_m), 0, 2)
    risk += clamp(required_rear_gap / max(1e-6, ps.rear_distance_m), 0, 2)
    risk /= 6.0
    return SafetyResult(approved=not reasons, reasons=reasons, min_ttc_s=min_ttc, risk_score=risk)


def node_safety_checker(state: GraphState) -> Dict[str, Any]:
    spec = dict_to_spec(state["spec"])
    ps = PassState(**state["pass_state"])
    proposed = state["proposed_action"]
    if proposed != "pass":
        safety = SafetyResult(approved=True, reasons=[f"{proposed} does not require pass clearance"], min_ttc_s=math.inf)
        return {"safety": asdict(safety), "approved_action": proposed}
    safety = check_pass_safety(spec, ps)
    approved_action: Action = "pass" if safety.approved else ("replan" if ps.deadline_pressure > 1.05 else "wait")
    return {"safety": asdict(safety), "approved_action": approved_action}


def node_execute(state: GraphState) -> Dict[str, Any]:
    spec = dict_to_spec(state["spec"])
    world = WorldState(**state["world"])
    action = state["approved_action"]
    dt = 1.0

    lead_x = world.lead_x_m + spec.lead.speed_mps * dt + 0.5 * spec.lead.accel_mps2 * dt * dt
    rear_x = world.rear_x_m + spec.rear.speed_mps * dt
    oncoming_x = world.oncoming_x_m - spec.oncoming.speed_mps * dt

    if action == "pass":
        ego_lane = 1
        ego_speed = min(spec.route.speed_limit_mps + 2.0, world.ego_speed_mps + 1.35)
    elif action == "replan":
        ego_lane = 0
        # Replan means abandon pass and follow safely, but try not to stop abruptly.
        front = max(0.0, world.lead_x_m - world.ego_x_m)
        target = spec.lead.speed_mps + (1.0 if front > 22.0 else 0.0)
        ego_speed = max(4.5, min(world.ego_speed_mps, target))
    else:
        ego_lane = 0
        front = max(0.0, world.lead_x_m - world.ego_x_m)
        target = spec.lead.speed_mps if front < 22.0 else min(world.ego_speed_mps, spec.route.speed_limit_mps)
        ego_speed = max(0.0, min(world.ego_speed_mps + 0.4, target))

    ego_x = world.ego_x_m + ego_speed * dt
    passed = world.passed or ego_x > lead_x + 9.0
    if passed:
        ego_lane = 0
        ego_speed = min(spec.route.speed_limit_mps, ego_speed)

    collision = False
    if ego_lane == 0 and abs(ego_x - lead_x) < 4.0:
        collision = True
    if ego_lane == 1 and abs(ego_x - oncoming_x) < 6.0:
        collision = True
    if ego_lane == 1 and abs(ego_x - rear_x) < 5.0:
        collision = True

    done = collision or ego_x >= spec.route.goal_x_m or world.t_s + dt >= 160.0
    new_world = WorldState(
        t_s=world.t_s + dt,
        ego_x_m=ego_x,
        ego_lane=ego_lane,
        ego_speed_mps=ego_speed,
        lead_x_m=lead_x,
        rear_x_m=rear_x,
        oncoming_x_m=oncoming_x,
        passed=passed,
        collision=collision,
        done=done,
    )

    trace = list(state.get("trace", []))
    trace.append(
        {
            "t_s": round(new_world.t_s, 2),
            "ego_x_m": round(new_world.ego_x_m, 2),
            "ego_lane": new_world.ego_lane,
            "proposed_action": state["proposed_action"],
            "approved_action": action,
            "safety_approved": bool(state["safety"]["approved"]),
            "safety_reasons": state["safety"].get("reasons", []),
            "risk_score": round(float(state["safety"].get("risk_score", 0.0)), 3),
            "min_ttc_s": finite_or_none(float(state["safety"].get("min_ttc_s", math.inf))),
            "urgency": state["urgency"],
            "pass_state": state["pass_state"],
            "perception": state["perception"],
        }
    )
    return {"world": asdict(new_world), "trace": trace}


def node_evaluate(state: GraphState) -> Dict[str, Any]:
    spec = dict_to_spec(state["spec"])
    world = WorldState(**state["world"])
    trace = state.get("trace", [])
    proposed_passes = sum(1 for x in trace if x.get("proposed_action") == "pass")
    approved_passes = sum(1 for x in trace if x.get("approved_action") == "pass")
    rejected_passes = sum(1 for x in trace if x.get("proposed_action") == "pass" and not x.get("safety_approved", False))
    min_ttc_vals = [x.get("min_ttc_s") for x in trace if x.get("min_ttc_s") is not None]
    min_ttc = min(min_ttc_vals) if min_ttc_vals else math.inf
    route_completed = world.ego_x_m >= spec.route.goal_x_m and not world.collision

    no_pass_counterfactual = spec.route.goal_x_m / max(1e-6, min(spec.ego_speed_mps, spec.lead.speed_mps + 0.3))
    autopass_missed_safe = state.get("policy") == "autopass" and approved_passes == 0 and no_pass_counterfactual > spec.request.deadline_s and route_completed

    if world.collision:
        failure_type = "collision"
    elif rejected_passes > 0 and state.get("policy") == "aggressive":
        failure_type = "unsafe_pass_attempt_rejected"
    elif autopass_missed_safe:
        failure_type = "missed_safe_pass"
    elif world.t_s > spec.request.deadline_s + 1e-9:
        failure_type = "over_conservative_delay"
    elif not route_completed:
        failure_type = "timeout"
    else:
        failure_type = "none"

    metrics = {
        "scenario_id": spec.scenario_id,
        "policy": state.get("policy", "autopass"),
        "collision": world.collision,
        "route_completed": route_completed,
        "time_to_goal_s": round(world.t_s, 2),
        "deadline_s": round(spec.request.deadline_s, 2),
        "proposed_passes": proposed_passes,
        "approved_passes": approved_passes,
        "unsafe_passes": rejected_passes,
        "min_ttc_s": finite_or_none(float(min_ttc)),
        "failure_type": failure_type,
    }
    return {"metrics": metrics}


def should_continue(state: GraphState) -> Literal["continue", "finish"]:
    world = WorldState(**state["world"])
    if world.done or len(state.get("trace", [])) >= 90:
        return "finish"
    return "continue"


def build_graph():
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception as e:  # pragma: no cover
        raise RuntimeError("LangGraph is required. Install with: pip install langgraph") from e

    g = StateGraph(GraphState)
    g.add_node("request_urgency_interpreter", node_interpret_request)
    g.add_node("visual_perception_map_tools", node_perception_map)
    g.add_node("passing_planning_agent", node_planning_agent)
    g.add_node("safety_checker", node_safety_checker)
    g.add_node("execution", node_execute)
    g.add_node("evaluator", node_evaluate)

    g.add_edge(START, "request_urgency_interpreter")
    g.add_edge("request_urgency_interpreter", "visual_perception_map_tools")
    g.add_edge("visual_perception_map_tools", "passing_planning_agent")
    g.add_edge("passing_planning_agent", "safety_checker")
    g.add_edge("safety_checker", "execution")
    g.add_conditional_edges("execution", should_continue, {"continue": "request_urgency_interpreter", "finish": "evaluator"})
    g.add_edge("evaluator", END)
    return g.compile()


# ------------------------ Evaluation and feedback ------------------------

def mutate_from_failure(spec: ScenarioSpec, metrics: Dict[str, Any], round_idx: int) -> ScenarioSpec:
    """Closed-loop generator update from evaluator output."""
    sid = f"{spec.scenario_id}_mut{round_idx}"
    ft = metrics.get("failure_type", "none")
    if ft == "none":
        return replace(
            spec,
            scenario_id=sid,
            request=replace(spec.request, deadline_s=max(16.0, spec.request.deadline_s - 2.0)),
            oncoming=replace(spec.oncoming, distance_m=max(85.0, spec.oncoming.distance_m - 18.0)),
            occlusion=replace(spec.occlusion, severity=clamp(spec.occlusion.severity + 0.08, 0.0, 0.9), sight_distance_m=max(45.0, spec.occlusion.sight_distance_m - 12.0)),
        )
    if ft in {"over_conservative_delay", "missed_safe_pass"}:
        return replace(
            spec,
            scenario_id=sid,
            request=replace(spec.request, deadline_s=max(15.0, spec.request.deadline_s - 2.5)),
            lead=replace(spec.lead, speed_mps=max(4.8, spec.lead.speed_mps - 0.4)),
            oncoming=replace(spec.oncoming, distance_m=spec.oncoming.distance_m + 25.0),
            rear=replace(spec.rear, distance_m=spec.rear.distance_m + 10.0),
            occlusion=replace(spec.occlusion, sight_distance_m=spec.occlusion.sight_distance_m + 12.0),
        )
    if ft in {"unsafe_pass_attempt_rejected", "collision"}:
        return replace(
            spec,
            scenario_id=sid,
            request=replace(spec.request, deadline_s=spec.request.deadline_s + 3.0),
            oncoming=replace(spec.oncoming, distance_m=spec.oncoming.distance_m + 35.0),
            rear=replace(spec.rear, distance_m=spec.rear.distance_m + 12.0),
            occlusion=replace(spec.occlusion, severity=clamp(spec.occlusion.severity - 0.08, 0.0, 0.9), sight_distance_m=spec.occlusion.sight_distance_m + 16.0),
        )
    return replace(spec, scenario_id=sid)


def run_one(spec: ScenarioSpec, policy: PolicyName, out_dir: Optional[Path] = None) -> GraphState:
    app = build_graph()
    init: GraphState = {"spec": asdict(spec), "policy": policy, "world": asdict(initialize_world(spec)), "trace": [], "out_dir": str(out_dir or "runs/demo")}
    result = app.invoke(init)
    if out_dir is not None:
        save_outputs(result, out_dir)
    return result


def run_batch(specs: List[ScenarioSpec], policies: List[PolicyName], out_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        (out_dir / "scenarios").mkdir(exist_ok=True)
        (out_dir / "scenarios" / f"{spec.scenario_id}.json").write_text(json.dumps(asdict(spec), indent=2), encoding="utf-8")
        for policy in policies:
            result = run_one(spec, policy, out_dir)
            rows.append(result["metrics"])
    write_metrics_csv(rows, out_dir / "metrics.csv")
    return rows


def run_closed_loop(rounds: int, n: int, seed: int, policies: List[PolicyName], out_dir: Path) -> List[Dict[str, Any]]:
    """Run adaptive generation. The feedback loop mutates AutoPass failures and boundary cases."""
    all_rows: List[Dict[str, Any]] = []
    specs = [generate_scenario(i, seed=seed, difficulty=0.35) for i in range(n)]
    for r in range(rounds):
        round_dir = out_dir / f"round_{r:02d}"
        rows = run_batch(specs, policies, round_dir)
        for row in rows:
            row["round"] = r
        all_rows.extend(rows)
        autopass_by_id = {row["scenario_id"]: row for row in rows if row["policy"] == "autopass"}
        new_specs: List[ScenarioSpec] = []
        for spec in specs:
            new_specs.append(mutate_from_failure(spec, autopass_by_id[spec.scenario_id], r + 1))
        # Add a little fresh mass each round so the generator is not only local search.
        specs = new_specs[: max(1, n - 1)] + [generate_scenario(10_000 + r, seed=seed, difficulty=clamp(0.45 + 0.15 * r, 0.0, 0.9))]
    write_metrics_csv(all_rows, out_dir / "closed_loop_metrics.csv")
    return all_rows


def write_metrics_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------ Visual outputs ------------------------

def _save_array_images(spec: ScenarioSpec, world: WorldState, out_prefix: Path) -> Dict[str, str]:
    from PIL import Image
    rgb, seg, depth, _ = render_sensor_frame(spec, world)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rgb_path = out_prefix.with_suffix(".rgb.png")
    seg_path = out_prefix.with_suffix(".seg.png")
    depth_path = out_prefix.with_suffix(".depth.png")
    Image.fromarray(rgb).save(rgb_path)
    # Scale labels for visibility.
    Image.fromarray((seg.astype(np.uint8) * 35)).save(seg_path)
    finite = depth[np.isfinite(depth)]
    max_depth = float(np.percentile(finite, 95)) if finite.size else 1.0
    depth_img = np.where(np.isfinite(depth), np.clip(depth / max_depth * 255, 0, 255), 255).astype(np.uint8)
    Image.fromarray(depth_img).save(depth_path)
    return {"rgb": str(rgb_path), "segmentation": str(seg_path), "depth": str(depth_path)}


def render_scene(spec: ScenarioSpec, world: WorldState, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    perception, _ = synthetic_perception(spec, world)
    depth_summary = perception["depth"]
    rgb, seg, depth, _ = render_sensor_frame(spec, world)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(2, 2, 1)
    ax_rgb = fig.add_subplot(2, 2, 2)
    ax_seg = fig.add_subplot(2, 2, 3)
    ax_depth = fig.add_subplot(2, 2, 4)

    x_min, x_max = world.ego_x_m - 45, world.ego_x_m + 150
    for y in [0, spec.route.lane_width_m, 2 * spec.route.lane_width_m]:
        ax.plot([x_min, x_max], [y, y], linewidth=1)
    ax.plot([x_min, x_max], [spec.route.lane_width_m, spec.route.lane_width_m], linestyle="--", linewidth=1)

    def car(axis, x: float, lane: int, label: str, fill: bool = False) -> None:
        y = lane * spec.route.lane_width_m + spec.route.lane_width_m / 2
        axis.add_patch(plt.Rectangle((x - 2.2, y - 0.8), 4.4, 1.6, fill=fill, alpha=0.30, linewidth=2))
        axis.text(x, y + 1.1, label, ha="center", va="bottom", fontsize=8)

    car(ax, world.ego_x_m, world.ego_lane, "ego")
    car(ax, world.lead_x_m, 0, "lead")
    car(ax, world.rear_x_m, 1, "rear")
    car(ax, world.oncoming_x_m, 1, "oncoming")
    vis_end = world.ego_x_m + depth_summary["visibility_m"]
    ax.axvspan(vis_end, x_max, alpha=0.08)
    ax.text(vis_end, spec.route.lane_width_m * 1.7, "visibility/depth limit", rotation=90, fontsize=8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, spec.route.lane_width_m * 2 + 0.8)
    ax.set_yticks([spec.route.lane_width_m / 2, spec.route.lane_width_m * 1.5])
    ax.set_yticklabels(["own lane", "passing lane"])
    ax.set_title("world scene")
    ax.set_xlabel("x position (m)")

    ax_rgb.imshow(rgb)
    ax_rgb.set_title("rendered RGB camera")
    ax_rgb.axis("off")
    ax_seg.imshow(seg, vmin=0, vmax=6)
    ax_seg.set_title("semantic segmentation mask")
    ax_seg.axis("off")
    finite = depth[np.isfinite(depth)]
    vmax = float(np.percentile(finite, 95)) if finite.size else 1.0
    ax_depth.imshow(np.where(np.isfinite(depth), depth, vmax), vmin=0, vmax=vmax)
    ax_depth.set_title(f"depth map / extracted: F={depth_summary['front_m']:.1f}, R={depth_summary['rear_m']:.1f}, O={depth_summary['oncoming_m']:.1f}, V={depth_summary['visibility_m']:.1f}m")
    ax_depth.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_outputs(result: GraphState, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = dict_to_spec(result["spec"])
    policy = result["policy"]
    trace = result.get("trace", [])
    (out_dir / "traces").mkdir(exist_ok=True)
    (out_dir / "metrics_json").mkdir(exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)
    trace_path = out_dir / "traces" / f"{spec.scenario_id}_{policy}_trace.json"
    metrics_path = out_dir / "metrics_json" / f"{spec.scenario_id}_{policy}_metrics.json"
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(result.get("metrics", {}), indent=2), encoding="utf-8")

    # Save first and final rendered sensor products for demo evidence.
    initial_world = initialize_world(spec)
    final_world = WorldState(**result["world"])
    _save_array_images(spec, initial_world, out_dir / "frames" / f"{spec.scenario_id}_{policy}_initial")
    _save_array_images(spec, final_world, out_dir / "frames" / f"{spec.scenario_id}_{policy}_final")
    render_scene(spec, initial_world, out_dir / "frames" / f"{spec.scenario_id}_{policy}_initial_panel.png", f"{spec.scenario_id} / {policy} / initial visual perception")
    render_scene(spec, final_world, out_dir / "frames" / f"{spec.scenario_id}_{policy}_final_panel.png", f"{spec.scenario_id} / {policy} / {result['metrics']['failure_type']}")


def print_summary(rows: List[Dict[str, Any]], policies: List[PolicyName]) -> None:
    print("\n=== AutoPass LangGraph Visual Closed-Loop Demo Summary ===")
    for policy in policies:
        subset = [r for r in rows if r["policy"] == policy]
        if not subset:
            continue
        failures = sum(1 for r in subset if r["failure_type"] != "none")
        avg_t = sum(float(r["time_to_goal_s"]) for r in subset) / len(subset)
        proposed = sum(int(r["proposed_passes"]) for r in subset)
        approved = sum(int(r["approved_passes"]) for r in subset)
        unsafe = sum(int(r["unsafe_passes"]) for r in subset)
        by_type: Dict[str, int] = {}
        for r in subset:
            by_type[r["failure_type"]] = by_type.get(r["failure_type"], 0) + 1
        print(f"{policy:10s} failures={failures}/{len(subset)} avg_time={avg_t:.2f}s proposed_passes={proposed} approved_passes={approved} unsafe/rejected={unsafe} types={by_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "batch", "closed_loop"], default="demo")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=float, default=0.45)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/demo"))
    parser.add_argument("--policies", default="autopass,no_pass,aggressive")
    args = parser.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    if not all(p in {"autopass", "no_pass", "aggressive"} for p in policies):
        raise ValueError("policies must be drawn from: autopass,no_pass,aggressive")
    typed_policies = [p for p in policies]  # type: ignore[list-item]

    if args.mode == "demo":
        specs = curated_demo_scenarios()[: args.n]
        rows = run_batch(specs, typed_policies, args.out_dir)
    elif args.mode == "batch":
        specs = [generate_scenario(i, seed=args.seed, difficulty=args.difficulty) for i in range(args.n)]
        rows = run_batch(specs, typed_policies, args.out_dir)
    else:
        rows = run_closed_loop(args.rounds, args.n, args.seed, typed_policies, args.out_dir)

    print_summary(rows, typed_policies)
    print(f"\nWrote outputs to: {args.out_dir}")
    print("Key demo evidence: metrics.csv or closed_loop_metrics.csv, traces/*.json, frames/*rgb.png, frames/*seg.png, frames/*depth.png, frames/*panel.png")


if __name__ == "__main__":
    main()
