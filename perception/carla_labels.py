"""Map CARLA Cityscapes semantic IDs to vehicle regions for depth extraction."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

VEHICLE_LABELS = {10, 12, 13, 14, 15, 16, 18, 19, 20}


def carla_seg_to_car_distances(seg: np.ndarray, depth_m: np.ndarray) -> List[Dict]:
    cars = []
    h, w = seg.shape
    for label in VEHICLE_LABELS:
        mask = seg == label
        if np.sum(mask) < 30:
            continue
        ys, xs = np.where(mask)
        d_vals = depth_m[mask]
        d_vals = d_vals[np.isfinite(d_vals) & (d_vals > 1.0) & (d_vals < 500.0)]
        if d_vals.size == 0:
            continue
        median_d = float(np.median(d_vals))
        cy = float(ys.mean())
        cx = float(xs.mean())
        if cy > h * 0.50 and abs(cx - w * 0.5) < w * 0.22:
            position = "front"
        elif cy > h * 0.55:
            position = "front"
        elif cy < h * 0.62 and median_d < 55.0:
            if cx < w * 0.44:
                position = "rear_left"
            elif cx > w * 0.56:
                position = "rear_right"
            else:
                position = "rear"
        elif cx < w * 0.45:
            position = "front_left"
        elif cx > w * 0.55:
            position = "front_right"
        else:
            position = "front"
        cars.append({
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "median_depth": round(median_d, 2),
            "min_depth": round(float(np.min(d_vals)), 2),
            "position": position,
            "cy_mean": cy,
            "cx_mean": cx,
        })
    return sorted(cars, key=lambda c: c["median_depth"])


def carla_frame_to_perception(rgb: np.ndarray, seg: np.ndarray, depth_m: np.ndarray) -> tuple:
    car_distances = carla_seg_to_car_distances(seg, depth_m)
    finite = depth_m[np.isfinite(depth_m)]
    depth_result = {
        "depth_map": depth_m.astype(np.float32),
        "min_depth": float(np.min(finite)) if finite.size else 1.0,
        "max_depth": float(np.max(finite)) if finite.size else 200.0,
        "car_distances": car_distances,
    }
    return rgb, seg, depth_result
