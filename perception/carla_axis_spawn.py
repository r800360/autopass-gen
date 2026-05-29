"""World-space axis spawn math (ego origin + travel/lateral basis). No CARLA imports."""
from __future__ import annotations

import math
from typing import Tuple


def normalize3(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    mag = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if mag < 1e-9:
        return (0.0, 0.0, 0.0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def world_location_from_ego_offset(
    ego_xyz: Tuple[float, float, float],
    travel_dir: Tuple[float, float, float],
    lateral_dir: Tuple[float, float, float],
    *,
    longitudinal_m: float,
    lateral_m: float = 0.0,
) -> Tuple[float, float, float]:
    """
    loc = ego + longitudinal_m * travel_dir + lateral_m * lateral_dir
    All direction vectors should be normalized; lateral_dir may be zero.
    """
    t = normalize3(travel_dir)
    lat = lateral_dir
    return (
        ego_xyz[0] + longitudinal_m * t[0] + lateral_m * lat[0],
        ego_xyz[1] + longitudinal_m * t[1] + lateral_m * lat[1],
        ego_xyz[2] + longitudinal_m * t[2] + lateral_m * lat[2],
    )


def projected_distance_m(
    ego_xyz: Tuple[float, float, float],
    other_xyz: Tuple[float, float, float],
    travel_dir: Tuple[float, float, float],
) -> float:
    """Signed distance of ``other`` along ``travel_dir`` from ``ego`` (+ = ahead)."""
    t = normalize3(travel_dir)
    dx = other_xyz[0] - ego_xyz[0]
    dy = other_xyz[1] - ego_xyz[1]
    dz = other_xyz[2] - ego_xyz[2]
    return dx * t[0] + dy * t[1] + dz * t[2]


def euclidean_distance_m(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)
