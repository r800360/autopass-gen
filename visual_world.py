"""Visual driving world: scenario specs, RGB/segmentation/depth rendering, metric extraction."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np

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


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def initialize_world(spec: ScenarioSpec) -> WorldState:
    return WorldState(
        ego_x_m=spec.route.start_x_m,
        ego_lane=0,
        ego_speed_mps=spec.ego_speed_mps,
        lead_x_m=spec.route.start_x_m + spec.lead.distance_m,
        rear_x_m=spec.route.start_x_m - spec.rear.distance_m,
        oncoming_x_m=spec.route.start_x_m + spec.oncoming.distance_m,
    )


def curated_demo_scenarios() -> List[ScenarioSpec]:
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
    shade = 60 + 25 * label
    rgb[y0:y1, x0:x1, :] = np.array([shade, max(20, 210 - 20 * label), 80 + 10 * label], dtype=np.uint8)


def render_sensor_frame(spec: ScenarioSpec, world: WorldState, width: int = 640, height: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    x_min, x_max = -110.0, 330.0
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, :] = np.array([35, 35, 35], dtype=np.uint8)
    seg = np.zeros((height, width), dtype=np.uint8)
    depth = np.full((height, width), np.inf, dtype=np.float32)

    row_own0, row_own1 = int(height * 0.58), int(height * 0.93)
    row_pass0, row_pass1 = int(height * 0.13), int(height * 0.48)
    for r0, r1 in [(row_own0, row_own1), (row_pass0, row_pass1)]:
        rgb[r0:r1, :, :] = np.array([70, 70, 70], dtype=np.uint8)
        seg[r0:r1, :] = LABELS["road"]
        xs = np.linspace(x_min, x_max, width, dtype=np.float32)
        depth[r0:r1, :] = np.abs(xs)[None, :]

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


def label_to_car_position(label: int, lane_row: int, height: int) -> str:
    """Map semantic label + image row to ego-relative position string."""
    if label == LABELS["lead"]:
        return "front"
    if label == LABELS["oncoming"]:
        return "front_left"
    if label == LABELS["rear"]:
        return "rear_left" if lane_row > height // 2 else "rear_right"
    return "front"


def bbox_for_label(seg: np.ndarray, label: int) -> List[int]:
    ys, xs = np.where(seg == label)
    if ys.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def extract_depth_from_frame(seg: np.ndarray, depth: np.ndarray) -> Dict:
    """Build depth-estimation dict from real segmentation + depth arrays."""
    car_labels = [LABELS["lead"], LABELS["rear"], LABELS["oncoming"]]
    car_distances = []
    h = seg.shape[0]
    for label in car_labels:
        if np.sum(seg == label) == 0:
            continue
        d = _median_depth_for_label(seg, depth, label, 999.0)
        cy = int(np.where(seg == label)[0].mean())
        car_distances.append({
            "bbox": bbox_for_label(seg, label),
            "median_depth": round(d, 2),
            "min_depth": round(d * 0.92, 2),
            "position": label_to_car_position(label, cy, h),
        })
    finite = depth[np.isfinite(depth)]
    return {
        "depth_map": depth.astype(np.float32),
        "min_depth": float(np.min(finite)) if finite.size else 1.0,
        "max_depth": float(np.max(finite)) if finite.size else 200.0,
        "car_distances": car_distances,
    }


def extract_segmentation_from_frame(seg: np.ndarray, rgb: np.ndarray) -> Dict:
    """Build segmentation dict from semantic mask (instance = connected label region)."""
    h, w = seg.shape
    car_masks = []
    for name in ("lead", "rear", "oncoming"):
        label = LABELS[name]
        if np.sum(seg == label) == 0:
            continue
        mask = (seg == label).astype(np.uint8)
        car_masks.append({
            "bbox": bbox_for_label(seg, label),
            "mask": mask,
            "confidence": 0.95,
            "label": "car",
        })
    hazards = []
    if np.sum(seg == LABELS["occlusion"]) > 50:
        hazards.append({"bbox": bbox_for_label(seg, LABELS["occlusion"]), "label": "occlusion", "confidence": 0.8})
    return {
        "car_masks": car_masks,
        "lane_lines": [
            {"points": [[w // 3, h], [w // 3, h // 2]], "type": "dashed", "side": "left"},
            {"points": [[2 * w // 3, h], [2 * w // 3, h // 2]], "type": "solid", "side": "right"},
        ],
        "hazards": hazards,
        "drivable_area": (seg == LABELS["road"]).astype(np.uint8),
    }
