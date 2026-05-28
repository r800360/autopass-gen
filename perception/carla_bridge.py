"""Optional live CARLA sensor bridge (requires CARLA 0.9.15 Python 3.7 egg + running simulator)."""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np

_carla = None
_world = None
_vehicle = None
_rgb = None
_depth = None
_seg = None


def _bootstrap_carla():
    global _carla
    if _carla is not None:
        return _carla
    # egg = os.environ.get(
    #     "CARLA_EGG",
    #     r"C:\carla\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg",
    # )
    # if egg and os.path.isfile(egg) and egg not in sys.path:
    #     sys.path.insert(0, egg)
    import carla as carla_mod

    _carla = carla_mod
    return carla_mod


def connect(host: str = "127.0.0.1", port: int = 2000, timeout_s: float = 5.0) -> bool:
    global _world, _vehicle, _rgb, _depth, _seg
    try:
        carla = _bootstrap_carla()
        client = carla.Client(host, port)
        client.set_timeout(timeout_s)
        _world = client.get_world()
        blueprint = _world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
        spawn = _world.get_map().get_spawn_points()[0]
        _vehicle = _world.spawn_actor(blueprint, spawn)
        cam_bp = _world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "640")
        cam_bp.set_attribute("image_size_y", "256")
        depth_bp = _world.get_blueprint_library().find("sensor.camera.depth")
        depth_bp.set_attribute("image_size_x", "640")
        depth_bp.set_attribute("image_size_y", "256")
        seg_bp = _world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
        seg_bp.set_attribute("image_size_x", "640")
        seg_bp.set_attribute("image_size_y", "256")
        transform = carla.Transform(carla.Location(x=1.6, z=1.4))
        _rgb = _world.spawn_actor(cam_bp, transform, attach_to=_vehicle)
        _depth = _world.spawn_actor(depth_bp, transform, attach_to=_vehicle)
        _seg = _world.spawn_actor(seg_bp, transform, attach_to=_vehicle)
        return True
    except Exception:
        return False


def grab_carla_frame() -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return (rgb, semantic_ids, depth_m) from latest CARLA sensors."""
    if _rgb is None:
        return None
    carla = _bootstrap_carla()
    rgb_raw = None
    depth_raw = None
    seg_raw = None

    def _save_rgb(img):
        nonlocal rgb_raw
        rgb_raw = img

    def _save_depth(img):
        nonlocal depth_raw
        depth_raw = img

    def _save_seg(img):
        nonlocal seg_raw
        seg_raw = img

    _rgb.listen(_save_rgb)
    _depth.listen(_save_depth)
    _seg.listen(_save_seg)
    _world.tick()
    if rgb_raw is None:
        return None
    array = np.frombuffer(rgb_raw.raw_data, dtype=np.uint8)
    array = array.reshape((rgb_raw.height, rgb_raw.width, 4))[:, :, :3]
    seg_array = np.frombuffer(seg_raw.raw_data, dtype=np.uint8).reshape((seg_raw.height, seg_raw.width, 4))[:, :, 2]
    depth_array = np.frombuffer(depth_raw.raw_data, dtype=np.uint8)
    depth_array = depth_array.reshape((depth_raw.height, depth_raw.width, 4))
    depth_m = (depth_array[:, :, 2] + depth_array[:, :, 1] * 256 + depth_array[:, :, 0] * 256 * 256)
    depth_m = depth_m.astype(np.float32) / (256**3 - 1) * 1000.0
    return array.copy(), seg_array.copy(), depth_m.copy()
