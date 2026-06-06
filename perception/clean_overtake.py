"""Clean, generalizable CARLA overtake driver.

Design goals (why this module exists):
  * Lane compliance BY CONSTRUCTION — steering always targets a real CARLA
    lane-center waypoint (``map.get_waypoint`` + ``get_left_lane``/``get_right_lane``),
    so the ego can never cross five lanes or drive into a median/wall.
  * Generalizable — the corridor finder works on ANY town because it reads the
    actual road graph, not hand-tuned spawn ids.
  * Vision-grounded, agentic decision — front/rear gaps come from semantic
    segmentation + metric depth (front and rear cameras); the pass/wait call is
    made by an LLM under deadline pressure, clamped by hard safety gates.
  * Honest metrics — collision sensor + per-tick lane-deviation + off-road check.

This intentionally replaces the fragile axis/corridor control tangle. It is
small and navigable on purpose.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from perception.carla_labels import carla_seg_to_car_distances
from perception.carla_lane_keep import pure_pursuit_steer

# ---------------------------------------------------------------------------
# Safety constants (shared with the rest of the project's gate logic).
# ---------------------------------------------------------------------------
MIN_PASS_FRONT_GAP_M = 18.0   # need this much clear road ahead before committing
SLOW_LEAD_MAX_MPS = 9.0       # only overtake a genuinely slower lead
REAR_SAFE_BASE_M = 12.0       # passing-lane rear gap floor
ONCOMING_SAFE_M = 45.0        # opposing-lane clearance needed for opposite-side pass


# ---------------------------------------------------------------------------
# Scenario specification.
# ---------------------------------------------------------------------------
@dataclass
class OvertakeScenario:
    scenario_id: str
    town: str
    narrative: str
    urgency: str = "high"               # low|medium|high
    expected: str = "pass"              # pass|wait
    lead_speed_mps: float = 6.0
    ego_cruise_mps: float = 14.0
    ego_pass_mps: float = 20.0
    lead_gap_m: float = 26.0            # initial longitudinal gap ego->lead
    passing_side: str = "left"          # preferred side; falls back to available
    rear_traffic_mps: float = 0.0       # >0 spawns a car behind in passing lane
    rear_gap_m: float = 45.0
    blocker_in_passing_lane: bool = False  # parks a slow car in passing lane (reject)
    blocker_gap_m: float = 22.0
    oncoming: bool = False              # two-lane road: pass uses the OPPOSING lane
    oncoming_actor: bool = False        # spawn a car in the opposing lane
    oncoming_actor_dist_m: float = 95.0
    oncoming_actor_mps: float = 8.0
    weather: str = "clear_noon"
    min_straight_m: float = 170.0
    corridor_rank: int = 0              # pick the Nth-best corridor (variety per map)
    ego_bp: str = "vehicle.tesla.model3"
    lead_bp: str = "vehicle.audi.tt"
    sim_budget_s: float = 32.0
    min_follow_s: float = 2.5           # follow/deliberate before any lane change


# ---------------------------------------------------------------------------
# World / weather helpers.
# ---------------------------------------------------------------------------
_WEATHER = {
    "clear_noon": "ClearNoon",
    "cloudy_noon": "CloudyNoon",
    "wet_noon": "WetNoon",
    "wet_cloudy_noon": "WetCloudyNoon",
    "soft_rain": "SoftRainNoon",
    "hard_rain": "HardRainNoon",
    "clear_sunset": "ClearSunset",
    "clear_sunrise": "ClearSunrise",
    "fog": "HardRainNoon",
}


def _set_weather(carla, world, name: str) -> None:
    preset = _WEATHER.get(name, "ClearNoon")
    try:
        world.set_weather(getattr(carla.WeatherParameters, preset))
    except Exception:
        world.set_weather(carla.WeatherParameters.ClearNoon)


def load_world(carla, client, town: str):
    cur = client.get_world()
    if town not in cur.get_map().name:
        client.load_world(town)
    world = client.get_world()
    settings = world.get_settings()
    prev = (settings.synchronous_mode, settings.fixed_delta_seconds)
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    return world, prev


def restore_world(world, prev) -> None:
    try:
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = prev
        world.apply_settings(settings)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Corridor finder — pick a long straight stretch with a same-direction
# adjacent lane. Map-agnostic: reads the road graph.
# ---------------------------------------------------------------------------
def _straight_len(carla, wp, step: float = 5.0, max_n: int = 60) -> float:
    cur = wp
    total = 0.0
    yaw0 = wp.transform.rotation.yaw
    for _ in range(max_n):
        nxts = cur.next(step)
        if not nxts:
            break
        cur = nxts[0]
        dyaw = abs(((cur.transform.rotation.yaw - yaw0 + 180) % 360) - 180)
        if dyaw > 28:
            break
        total += step
    return total


def _same_dir_sides(carla, wp) -> List[str]:
    out = []
    left, right = wp.get_left_lane(), wp.get_right_lane()
    if left is not None and left.lane_type == carla.LaneType.Driving and left.lane_id * wp.lane_id > 0:
        out.append("left")
    if right is not None and right.lane_type == carla.LaneType.Driving and right.lane_id * wp.lane_id > 0:
        out.append("right")
    return out


def _opposing_side(carla, wp) -> Optional[str]:
    left, right = wp.get_left_lane(), wp.get_right_lane()
    if left is not None and left.lane_type == carla.LaneType.Driving and left.lane_id * wp.lane_id < 0:
        return "left"
    if right is not None and right.lane_type == carla.LaneType.Driving and right.lane_id * wp.lane_id < 0:
        return "right"
    return None


def find_corridor(
    carla,
    world,
    *,
    min_straight_m: float,
    need_oncoming: bool,
    preferred_side: str,
    rank: int = 0,
) -> Tuple[Any, str]:
    """Return (travel_lane_waypoint, passing_side).

    A corridor is only accepted if the travel lane AND the chosen passing lane stay
    parallel, Driving, non-junction, and straight for the whole maneuver length. This
    rejects ramps/forks/curves that made earlier runs drive off-road.
    """
    m = world.get_map()
    wps = m.generate_waypoints(5.0)
    target = max(90.0, min_straight_m)
    cands = []
    seen = set()
    for wp in wps:
        if wp.lane_type != carla.LaneType.Driving or wp.is_junction:
            continue
        key = (wp.road_id, wp.lane_id, round(wp.s / 40))
        if key in seen:
            continue
        seen.add(key)
        if need_oncoming:
            sides = [s for s in (_opposing_side(carla, wp),) if s]
        else:
            sides = _same_dir_sides(carla, wp)
        if not sides:
            continue
        # Prefer the requested side, then any side; among clean ones prefer an INTERIOR
        # passing lane (road on its far side) so an overshoot can't reach a median/edge.
        ordered = sorted(sides, key=lambda s: 0 if s == preferred_side else 1)
        for side in ordered:
            ext = _corridor_extent(carla, wp, side, opposing=need_oncoming, target=target)
            if ext < min(target, 100.0):
                continue
            interior = _passing_lane_interior(carla, wp, side, opposing=need_oncoming)
            score = ext - abs(wp.transform.location.z) * 3.0 + (60.0 if interior else 0.0)
            score += 25.0 if side == preferred_side else 0.0
            cands.append((score, ext, side, wp))
            break
    if not cands:
        raise RuntimeError(f"No clean corridor (>= {min(target, 80.0):.0f}m parallel) found on {m.name}")
    cands.sort(key=lambda t: -t[0])
    idx = min(rank, len(cands) - 1)
    _, _, side, wp = cands[idx]
    return wp, side


def _passing_lane_interior(carla, wp, side: str, *, opposing: bool) -> bool:
    """True if the passing lane has drivable road on its far side (not a median/edge).

    An interior passing lane tolerates small overshoot during the lane change; an
    edge/median-adjacent one can put the ego onto grass. Two-lane (opposing) passes have
    no interior option, so they are always treated as acceptable.
    """
    if opposing:
        return True
    pl = wp.get_left_lane() if side == "left" else wp.get_right_lane()
    if pl is None:
        return False
    outer = pl.get_left_lane() if side == "left" else pl.get_right_lane()
    if outer is None:
        return False
    return outer.lane_type in (carla.LaneType.Driving, carla.LaneType.Shoulder,
                               carla.LaneType.Parking, carla.LaneType.Bidirectional)


def _corridor_extent(carla, wp, side: str, *, opposing: bool, target: float,
                     step: float = 5.0) -> float:
    """Metres for which travel+passing lanes stay clean and parallel (straight, no junction)."""
    Driving = carla.LaneType.Driving
    t_cur = wp
    yaw0 = wp.transform.rotation.yaw
    d = 0.0
    while d < target + step:
        nb = t_cur.get_left_lane() if side == "left" else t_cur.get_right_lane()
        if nb is None or nb.lane_type != Driving:
            break
        same_dir = nb.lane_id * t_cur.lane_id > 0
        if opposing and same_dir:
            break
        if (not opposing) and (not same_dir):
            break
        tl, nl = t_cur.transform.location, nb.transform.location
        lane_w = math.hypot(tl.x - nl.x, tl.y - nl.y)
        if lane_w < 2.4 or lane_w > 6.0:
            break
        if abs(((t_cur.transform.rotation.yaw - yaw0 + 180) % 360) - 180) > 12:
            break
        nxt = t_cur.next(step)
        if not nxt:
            break
        t_cur = nxt[0]
        if t_cur.is_junction:
            break
        d += step
    return d


def lane_to(carla, wp, target_lane_id: int):
    """Walk left/right lanes until lane_id matches target (adjacent, <=3 hops)."""
    cur = wp
    for _ in range(4):
        if cur is None:
            return wp
        if cur.lane_id == target_lane_id:
            return cur
        left, right = cur.get_left_lane(), cur.get_right_lane()
        cand = None
        best = abs(cur.lane_id - target_lane_id)
        for nb in (left, right):
            if nb is None or nb.lane_type != carla.LaneType.Driving:
                continue
            d = abs(nb.lane_id - target_lane_id)
            if d < best:
                best = d
                cand = nb
        if cand is None:
            return cur
        cur = cand
    return cur


# ---------------------------------------------------------------------------
# Geometry helpers.
# ---------------------------------------------------------------------------
def speed_mps(v) -> float:
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def lateral_offset_to_wp(ego_tf, wp) -> float:
    """Absolute lateral distance from ego to a lane waypoint's center line."""
    if wp is None:
        return 99.0
    dx = ego_tf.location.x - wp.transform.location.x
    dy = ego_tf.location.y - wp.transform.location.y
    yaw = math.radians(wp.transform.rotation.yaw)
    # perpendicular component
    return abs(-dx * math.sin(yaw) + dy * math.cos(yaw))


def signed_longitudinal(from_tf, to_loc) -> float:
    """Signed forward distance (along from_tf heading) to a location. + = ahead."""
    dx = to_loc.x - from_tf.location.x
    dy = to_loc.y - from_tf.location.y
    yaw = math.radians(from_tf.rotation.yaw)
    return dx * math.cos(yaw) + dy * math.sin(yaw)


def throttle_brake(cur: float, target: float) -> Tuple[float, float]:
    err = target - cur
    if err >= 0:
        return min(0.85, 0.30 + 0.55 * err), 0.0
    return 0.0, min(0.6, -0.22 * err)


# ---------------------------------------------------------------------------
# Camera decode.
# ---------------------------------------------------------------------------
def decode_rgb(img) -> np.ndarray:
    return np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))[:, :, :3].copy()


def decode_seg(img) -> np.ndarray:
    return np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))[:, :, 2].copy()


def decode_depth(img) -> np.ndarray:
    a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4)).astype(np.float32)
    d = a[:, :, 2] + a[:, :, 1] * 256.0 + a[:, :, 0] * 256.0 * 256.0
    return d / (256 ** 3 - 1) * 1000.0


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
class OvertakeRun:
    def __init__(self, carla, world, scn: OvertakeScenario, out_dir: Path):
        self.carla = carla
        self.world = world
        self.m = world.get_map()
        self.scn = scn
        self.out_dir = Path(out_dir)
        self.frames_dir = self.out_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.actors: Dict[str, Any] = {}
        self.sensors: Dict[str, Any] = {}
        self._buf: Dict[str, Any] = {}
        self.collision = False
        self._frames: List[np.ndarray] = []
        self.trace: List[dict] = []
        self.max_lane_dev = 0.0
        self.offroad = False
        self.dt = 0.05
        # phase state
        self.phase = "follow"
        self._t_s = 0.0
        self.prev_steer = 0.0
        self.travel_lane_id = 0
        self.passing_lane_id = 0
        self.passing_side = scn.passing_side
        self.passing_opposing = bool(scn.oncoming)
        self._cap_idx = 0

    # ---- spawn -----------------------------------------------------------
    def _spawn(self, bp_name: str, wp, *, color: Optional[str] = None):
        bl = self.world.get_blueprint_library()
        try:
            bp = bl.find(bp_name)
        except Exception:
            bp = bl.filter("vehicle.*")[0]
        if color and bp.has_attribute("color"):
            try:
                bp.set_attribute("color", color)
            except Exception:
                pass
        tf = self.carla.Transform(
            self.carla.Location(x=wp.transform.location.x, y=wp.transform.location.y, z=wp.transform.location.z + 0.4),
            wp.transform.rotation,
        )
        actor = self.world.try_spawn_actor(bp, tf)
        return actor

    def setup(self) -> None:
        carla, world, m, scn = self.carla, self.world, self.m, self.scn
        _set_weather(carla, world, scn.weather)

        travel_wp, side = find_corridor(
            carla, world,
            min_straight_m=scn.min_straight_m,
            need_oncoming=scn.oncoming,
            preferred_side=scn.passing_side,
            rank=scn.corridor_rank,
        )
        self.passing_side = side
        self.travel_lane_id = travel_wp.lane_id
        if scn.oncoming:
            opp = travel_wp.get_left_lane() if side == "left" else travel_wp.get_right_lane()
            self.passing_lane_id = opp.lane_id
        else:
            pass_wp = travel_wp.get_left_lane() if side == "left" else travel_wp.get_right_lane()
            self.passing_lane_id = pass_wp.lane_id

        # ego at travel lane
        ego = self._spawn(scn.ego_bp, travel_wp, color="20,90,200")
        if ego is None:
            raise RuntimeError("ego spawn failed")
        self.actors["ego"] = ego

        # lead ahead in travel lane
        lead_wp = travel_wp.next(scn.lead_gap_m)[0]
        lead = self._spawn(scn.lead_bp, lead_wp, color="200,40,40")
        if lead is None:
            # retry slightly closer
            lead = self._spawn(scn.lead_bp, travel_wp.next(max(12.0, scn.lead_gap_m - 6))[0], color="200,40,40")
        self.actors["lead"] = lead

        # optional rear traffic in passing lane
        if scn.rear_traffic_mps > 0:
            pass_lane_wp = lane_to(carla, travel_wp, self.passing_lane_id)
            back = pass_lane_wp.previous(scn.rear_gap_m)
            if back:
                rear = self._spawn("vehicle.dodge.charger_police", back[0], color="30,30,30")
                if rear is not None:
                    self.actors["rear"] = rear

        # optional blocker parked in passing lane ahead (reject scenario)
        if scn.blocker_in_passing_lane:
            pass_lane_wp = lane_to(carla, travel_wp, self.passing_lane_id)
            bwp = pass_lane_wp.next(scn.blocker_gap_m)
            if bwp:
                blk = self._spawn("vehicle.carlamotors.carlacola", bwp[0], color="230,200,30")
                if blk is not None:
                    self.actors["blocker"] = blk

        # optional oncoming car in opposing lane (ahead, travelling toward ego)
        if scn.oncoming and scn.oncoming_actor:
            opp_wp = lane_to(carla, travel_wp, self.passing_lane_id)
            # opposing lane runs backward vs ego, so a point ahead of ego is .previous()
            owp = opp_wp.previous(scn.oncoming_actor_dist_m)
            if owp:
                onc = self._spawn("vehicle.audi.etron", owp[0])
                if onc is not None:
                    self.actors["oncoming"] = onc

        self._attach_sensors(ego)
        # settle
        for _ in range(12):
            world.tick()
        self._spectator_follow()

    def _attach_sensors(self, ego) -> None:
        carla, world = self.carla, self.world
        bl = world.get_blueprint_library()

        def cam(kind: str, w=640, h=360, fov=90):
            bp = bl.find(kind)
            bp.set_attribute("image_size_x", str(w))
            bp.set_attribute("image_size_y", str(h))
            bp.set_attribute("fov", str(fov))
            return bp

        front_tr = carla.Transform(carla.Location(x=1.6, z=1.5), carla.Rotation(pitch=-3.0))
        rear_tr = carla.Transform(carla.Location(x=-1.8, z=1.5), carla.Rotation(yaw=180.0, pitch=-3.0))
        over_tr = carla.Transform(carla.Location(z=42.0), carla.Rotation(pitch=-90.0))

        specs = [
            ("rgb", "sensor.camera.rgb", front_tr),
            ("seg", "sensor.camera.semantic_segmentation", front_tr),
            ("depth", "sensor.camera.depth", front_tr),
            ("rseg", "sensor.camera.semantic_segmentation", rear_tr),
            ("rdepth", "sensor.camera.depth", rear_tr),
            ("overhead", "sensor.camera.rgb", over_tr),
        ]
        for name, kind, tr in specs:
            w, h = (800, 450) if name == "overhead" else (800, 450)
            s = world.spawn_actor(cam(kind, w, h), tr, attach_to=ego)
            s.listen(lambda img, n=name: self._buf.__setitem__(n, img))
            self.sensors[name] = s

        col_bp = bl.find("sensor.other.collision")
        col = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)
        col.listen(lambda e: setattr(self, "collision", True))
        self.sensors["collision"] = col

    def _spectator_follow(self) -> None:
        pass  # overhead camera provides the bird's-eye; spectator left alone

    # ---- vision gaps -----------------------------------------------------
    def perceive(self) -> Dict[str, Any]:
        """Vision-grounded gaps from front + rear segmentation/depth."""
        out: Dict[str, Any] = {"front_gap_m": None, "rear_gap_m": None, "oncoming_gap_m": None,
                               "passing_lane_ahead_gap_m": None, "front_dets": [], "lead_speed_mps": None}
        rgb_img = self._buf.get("rgb")
        seg_img = self._buf.get("seg")
        depth_img = self._buf.get("depth")
        if rgb_img is not None and seg_img is not None and depth_img is not None:
            seg = decode_seg(seg_img)
            depth = decode_depth(depth_img)
            dets = carla_seg_to_car_distances(seg, depth)
            W = float(seg.shape[1])
            # Monocular lateral world offset (m, +=right) from pixel-x and depth.
            # For a 90deg FOV pinhole camera fx = W/2, so lateral = depth*(cx-W/2)/fx.
            for d in dets:
                d["lateral_m"] = round(2.0 * d["median_depth"] * (d["cx_mean"] / W - 0.5), 1)
            out["front_dets"] = dets
            side_sign = -1.0 if self.passing_side == "left" else 1.0
            # Front = vehicle in OUR lane directly ahead (small lateral offset).
            front = [d for d in dets if abs(d["lateral_m"]) < 2.6 and 2.0 < d["median_depth"] < 110]
            if front:
                out["front_gap_m"] = round(min(d["median_depth"] for d in front), 1)
            # Oncoming (two-lane): any vehicle in the opposing lane region toward passing side.
            if self.scn.oncoming:
                onc = [d for d in dets if 2.0 < d["lateral_m"] * side_sign < 6.5 and 2.0 < d["median_depth"] < 110]
                if onc:
                    out["oncoming_gap_m"] = round(min(d["median_depth"] for d in onc), 1)
            else:
                # Passing lane = exactly one lane over (not the far shoulder/parked cars),
                # and only flag a blocker that is near enough to matter for the pull-out.
                adj = [d for d in dets if 2.8 < d["lateral_m"] * side_sign < 5.4 and 2.0 < d["median_depth"] < 40.0]
                if adj:
                    out["passing_lane_ahead_gap_m"] = round(min(d["median_depth"] for d in adj), 1)
        rseg_img = self._buf.get("rseg")
        rdepth_img = self._buf.get("rdepth")
        if rseg_img is not None and rdepth_img is not None:
            rseg = decode_seg(rseg_img)
            rdepth = decode_depth(rdepth_img)
            rdets = carla_seg_to_car_distances(rseg, rdepth)
            Wr = float(rseg.shape[1])
            for d in rdets:
                d["lateral_m"] = round(2.0 * d["median_depth"] * (d["cx_mean"] / Wr - 0.5), 1)
            # Rear camera faces backward; the passing lane is mirrored to the OTHER side.
            side_sign = -1.0 if self.passing_side == "left" else 1.0
            rear = [d for d in rdets if d["median_depth"] < 90
                    and (abs(d["lateral_m"]) < 2.2 or -6.5 < d["lateral_m"] * side_sign < -2.0)]
            if rear:
                out["rear_gap_m"] = round(min(d["median_depth"] for d in rear), 1)
        # measured lead speed (control bookkeeping; honest — from actor velocity)
        lead = self.actors.get("lead")
        if lead is not None:
            out["lead_speed_mps"] = round(speed_mps(lead.get_velocity()), 1)
        return out

    # ---- safety gates (hard constraints) --------------------------------
    def evaluate_gates(self, gaps: Dict[str, Any]) -> Dict[str, Any]:
        front = gaps.get("front_gap_m")
        rear = gaps.get("rear_gap_m")
        onc = gaps.get("oncoming_gap_m")
        pahead = gaps.get("passing_lane_ahead_gap_m")
        lead_v = gaps.get("lead_speed_mps")
        front_ok = front is not None and front >= MIN_PASS_FRONT_GAP_M
        slow_ok = lead_v is not None and lead_v <= SLOW_LEAD_MAX_MPS
        rear_ok = rear is None or rear >= REAR_SAFE_BASE_M  # None = nothing seen behind
        onc_ok = (not self.scn.oncoming) or (onc is None or onc >= ONCOMING_SAFE_M)
        pclear_ok = pahead is None or pahead >= 24.0  # passing lane clear ahead
        blockers = []
        if not front_ok:
            blockers.append(f"front gap {front}m < {MIN_PASS_FRONT_GAP_M:.0f}m" if front is not None else "front gap unmeasured")
        if not slow_ok:
            blockers.append(f"lead not slow ({lead_v} m/s)" if lead_v is not None else "lead speed unknown")
        if not rear_ok:
            blockers.append(f"passing-lane rear gap {rear}m < {REAR_SAFE_BASE_M:.0f}m")
        if not onc_ok:
            blockers.append(f"oncoming {onc}m < {ONCOMING_SAFE_M:.0f}m")
        if not pclear_ok:
            blockers.append(f"passing lane blocked ahead ({pahead}m)")
        can_pass = front_ok and slow_ok and rear_ok and onc_ok and pclear_ok
        return {
            "front_gap_ok": front_ok, "slow_lead_ok": slow_ok, "rear_gap_ok": rear_ok,
            "oncoming_ok": onc_ok, "passing_clear_ok": pclear_ok,
            "can_pass": can_pass, "blockers": blockers,
        }

    # ---- agentic decision (LLM judgment, gate-clamped) ------------------
    def decide(self, gaps: Dict[str, Any], gates: Dict[str, Any]) -> Tuple[str, str]:
        """Return (decision, reasoning). Hard gates override; LLM judges under urgency."""
        if not gates["can_pass"]:
            return "wait", "Safety gate blocks pass: " + "; ".join(gates["blockers"])
        from pydantic import BaseModel, Field

        class PassWait(BaseModel):
            decision: str = Field(description="'pass' or 'wait'")
            reasoning: str = ""

        urgency = self.scn.urgency
        # rule fallback (also the value used when AUTOPASS_MOCK_LLM=1)
        mock_dec = "pass" if urgency in ("high", "medium") else "wait"
        mock = PassWait(decision=mock_dec, reasoning=f"urgency={urgency}, all gates satisfied")
        try:
            from agents.llm_agents import structured_invoke
            prompt = (
                f"You are the overtaking decision layer of an autonomous vehicle under trip "
                f"deadline pressure. Urgency: {urgency}.\n"
                f"Vision-derived gaps (segmentation+depth): front_gap={gaps.get('front_gap_m')} m, "
                f"passing-lane rear_gap={gaps.get('rear_gap_m')} m, oncoming_gap={gaps.get('oncoming_gap_m')} m, "
                f"lead_speed={gaps.get('lead_speed_mps')} m/s.\n"
                f"Hard safety gates already PASSED: {gates}.\n"
                f"You are trapped behind a slow lead. Decide 'pass' to overtake now or 'wait' to keep "
                f"following. Greedy under urgency: pass when gates allow and it saves time; wait only if "
                f"the situation looks marginal. Respond with decision and a one-sentence reasoning."
            )
            res = structured_invoke(PassWait,
                                    "Autonomous overtaking decision agent. Safety gates may override you, "
                                    "but you choose pass vs wait from vision evidence and urgency.",
                                    prompt, mock)
            dec = res.decision.strip().lower()
            if dec not in ("pass", "wait"):
                dec = mock_dec
            return dec, (res.reasoning or mock.reasoning)
        except Exception as e:
            return mock_dec, f"{mock.reasoning} (llm_fallback: {e})"

    # ---- NPC control: constant-speed lane follower ----------------------
    def _drive_npc(self, actor, target_speed: float, prev_attr: str) -> None:
        if actor is None:
            return
        carla = self.carla
        tf = actor.get_transform()
        wp = self.m.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        nxts = wp.next(max(4.0, target_speed * 0.8))
        if not nxts:
            return
        tgt = nxts[0].transform.location
        prev = getattr(self, prev_attr, 0.0)
        steer, _, _ = pure_pursuit_steer(tf.location, tf.rotation.yaw, tgt,
                                         lookahead_m=8.0, max_steer=0.5, steer_gain=40.0,
                                         lateral_gain=0.7, prev_steer=prev, smooth=0.5)
        setattr(self, prev_attr, steer)
        thr, brk = throttle_brake(speed_mps(actor.get_velocity()), target_speed)
        actor.apply_control(carla.VehicleControl(throttle=thr, brake=brk, steer=steer))

    def drive_npcs(self) -> None:
        scn = self.scn
        self._drive_npc(self.actors.get("lead"), scn.lead_speed_mps, "_lead_steer")
        if "rear" in self.actors:
            self._drive_npc(self.actors["rear"], scn.rear_traffic_mps, "_rear_steer")
        if "oncoming" in self.actors:
            self._drive_npc(self.actors["oncoming"], scn.oncoming_actor_mps, "_onc_steer")
        # blocker stays put (parked obstacle)
        if "blocker" in self.actors:
            self.actors["blocker"].apply_control(self.carla.VehicleControl(brake=1.0, hand_brake=True))

    # ---- ego control per phase -----------------------------------------
    def _target_lane_id(self) -> int:
        if self.phase in ("lane_change", "overtake"):
            return self.passing_lane_id
        return self.travel_lane_id

    def _lane_lookahead(self, cur_wp, target_lane_id: int, look: float):
        """Lane-center waypoint ~look metres AHEAD of the ego in its travel direction.

        The opposing lane (two-lane passing) runs backward relative to the ego, so
        a point ahead of the ego is reached via .previous() on that lane.
        """
        lane_wp = lane_to(self.carla, cur_wp, target_lane_id)
        opposing = self.passing_opposing and target_lane_id == self.passing_lane_id
        nxts = lane_wp.previous(look) if opposing else lane_wp.next(look)
        return (nxts[0] if nxts else lane_wp), lane_wp

    def _acc_target_speed(self, gaps: Dict[str, Any]) -> float:
        """Adaptive cruise: hold a steady gap (above the pass threshold) behind the lead."""
        scn = self.scn
        front = gaps.get("front_gap_m")
        desired_gap = MIN_PASS_FRONT_GAP_M + 6.0  # ~24m: keeps measured gap above pass gate
        lead_v = gaps.get("lead_speed_mps") or scn.lead_speed_mps
        if front is None:
            return scn.ego_cruise_mps * 0.6  # lead not yet in view: gentle approach
        if front < 9.0:
            return 0.0  # emergency hold
        target = lead_v + 0.5 * (front - desired_gap)
        return max(0.0, min(scn.ego_cruise_mps, target))

    def drive_ego(self, gaps: Dict[str, Any]) -> Dict[str, float]:
        carla = self.carla
        ego = self.actors["ego"]
        tf = ego.get_transform()
        cur_wp = self.m.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        v = speed_mps(ego.get_velocity())
        look = max(6.0, v * 0.9)
        ahead_wp, tgt_lane_wp = self._lane_lookahead(cur_wp, self._target_lane_id(), look)
        tgt_loc = ahead_wp.transform.location

        # speed target by phase
        scn = self.scn
        if self.phase in ("lane_change", "overtake"):
            target_speed = scn.ego_pass_mps
        elif self.phase == "merge_back":
            target_speed = scn.ego_cruise_mps
        else:  # follow — ACC: hold a steady gap behind the slow lead
            target_speed = self._acc_target_speed(gaps)

        steer, head, lat = pure_pursuit_steer(
            tf.location, tf.rotation.yaw, tgt_loc,
            lookahead_m=look, max_steer=0.75, steer_gain=34.0,
            lateral_gain=1.0, prev_steer=self.prev_steer, smooth=0.45,
        )
        self.prev_steer = steer
        thr, brk = throttle_brake(v, target_speed)
        ego.apply_control(carla.VehicleControl(throttle=thr, brake=brk, steer=steer))
        return {"speed": v, "steer": steer, "target_speed": target_speed,
                "lat_to_target": lateral_offset_to_wp(tf, tgt_lane_wp)}

    # ---- phase transitions ---------------------------------------------
    def update_phase(self, gaps: Dict[str, Any], decision: str) -> None:
        carla = self.carla
        ego = self.actors["ego"]
        lead = self.actors.get("lead")
        tf = ego.get_transform()
        cur_wp = self.m.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        travel_wp = lane_to(carla, cur_wp, self.travel_lane_id)
        pass_wp = lane_to(carla, cur_wp, self.passing_lane_id)
        d_travel = lateral_offset_to_wp(tf, travel_wp)
        d_pass = lateral_offset_to_wp(tf, pass_wp)
        long_to_lead = signed_longitudinal(tf, lead.get_transform().location) if lead is not None else 99.0

        if self.phase == "follow":
            if decision == "pass" and self._t_s >= self.scn.min_follow_s:
                self.phase = "lane_change"
        elif self.phase == "lane_change":
            if d_pass < 0.6 and cur_wp.lane_id == self.passing_lane_id:
                self.phase = "overtake"
        elif self.phase == "overtake":
            # ahead of lead by a safe margin -> merge back (generous so we never clip the lead)
            if long_to_lead < -12.0:
                self.phase = "merge_back"
        elif self.phase == "merge_back":
            if d_travel < 0.6 and cur_wp.lane_id == self.travel_lane_id:
                self.phase = "done"

    # ---- recording ------------------------------------------------------
    def capture(self, hud: List[str], dets: List[dict]) -> None:
        from perception.vision_demo_overlay import compose_demo_frame
        from perception.carla_recorder import _stack_views
        rgb_img = self._buf.get("rgb")
        over_img = self._buf.get("overhead")
        if rgb_img is None:
            return
        rgb = decode_rgb(rgb_img)
        # mark which detection is the lead/front for overlay
        classified = []
        for d in dets:
            dd = dict(d)
            dd["depth_m"] = d.get("median_depth")
            if d["position"] == "front":
                dd["used_for_front_gap"] = True
            classified.append(dd)
        ego_panel = compose_demo_frame(rgb, hud_lines=hud, classified=classified, draw_boxes=True)
        over = decode_rgb(over_img) if over_img is not None else None
        composite = _stack_views(ego_panel, over)
        from PIL import Image
        path = self.frames_dir / f"frame_{self._cap_idx:05d}.png"
        Image.fromarray(composite).save(path)
        self._cap_idx += 1

    def track_compliance(self) -> None:
        carla = self.carla
        ego = self.actors["ego"]
        tf = ego.get_transform()
        loc = tf.location
        # Off-road = genuinely far from ANY driving lane center (grass/median), not merely
        # straddling a lane line or grazing a shoulder. Distance-based avoids false positives.
        near = self.m.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if near is None:
            self.offroad = True
            return
        nl = near.transform.location
        dist2d = math.hypot(loc.x - nl.x, loc.y - nl.y)
        if dist2d > 4.0:
            self.offroad = True
        # Lane-keeping is only meaningful in steady cruise phases; during an active
        # lane change (lane_change/overtake/merge_back) crossing the line is intended.
        if self.phase in ("follow", "done"):
            self.max_lane_dev = max(self.max_lane_dev, lateral_offset_to_wp(tf, near))

    # ---- main loop ------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        scn = self.scn
        max_ticks = int(scn.sim_budget_s / self.dt)
        decide_every = int(0.5 / self.dt)
        capture_every = 2
        decision, reasoning = "wait", "initial follow"
        gaps = self.perceive()
        gates = self.evaluate_gates(gaps)
        done_settle = 0
        for tick in range(max_ticks):
            self._t_s = tick * self.dt
            gaps = self.perceive()
            self.drive_npcs()
            if self.phase == "follow" and tick % decide_every == 0:
                gates = self.evaluate_gates(gaps)
                decision, reasoning = self.decide(gaps, gates)
                self.trace.append({
                    "tick": tick, "t_s": round(tick * self.dt, 2), "phase": self.phase,
                    "gaps": gaps, "gates": gates, "decision": decision, "reasoning": reasoning,
                })
            ego_ctrl = self.drive_ego(gaps)
            self.world.tick()
            self.update_phase(gaps, decision)
            self.track_compliance()
            if tick % capture_every == 0:
                hud = self._hud(tick, gaps, gates, decision, reasoning, ego_ctrl)
                self.capture(hud, gaps.get("front_dets", []))
            if self.collision or self.offroad:
                break
            if self.phase == "done":
                done_settle += 1
                if done_settle > int(2.0 / self.dt):
                    break
        return self._finalize()

    def _hud(self, tick, gaps, gates, decision, reasoning, ego_ctrl) -> List[str]:
        scn = self.scn
        lines = [
            f"t={tick * self.dt:4.1f}s  {scn.town}  {scn.scenario_id}",
            f"PHASE: {self.phase.upper()}   pass_side={self.passing_side}",
            f"urgency={scn.urgency}  expected={scn.expected}",
            f"VISION front={gaps.get('front_gap_m')}m rear={gaps.get('rear_gap_m')}m "
            f"onc={gaps.get('oncoming_gap_m')}m lead_v={gaps.get('lead_speed_mps')}m/s",
            f"CAN_PASS={'YES' if gates.get('can_pass') else 'no'}  DECISION={decision.upper()}",
            f"ego_v={ego_ctrl['speed']:.1f}->{ego_ctrl['target_speed']:.0f}m/s steer={ego_ctrl['steer']:+.2f}",
            f"why: {reasoning[:64]}",
        ]
        if gates.get("blockers"):
            lines.append("BLOCK: " + "; ".join(gates["blockers"])[:60])
        return lines

    def _finalize(self) -> Dict[str, Any]:
        ego = self.actors.get("ego")
        lead = self.actors.get("lead")
        ahead = False
        if ego is not None and lead is not None:
            ahead = signed_longitudinal(ego.get_transform(), lead.get_transform().location) < 0
        completed = (self.phase in ("merge_back", "done")) and ahead and not self.collision and not self.offroad
        if self.scn.expected == "wait":
            # Correct = behaved safely: never crashed/left road. Declining outright OR
            # yielding to the hazard then overtaking once it clears are both acceptable.
            completed = (not self.collision) and (not self.offroad)
        result = {
            "scenario_id": self.scn.scenario_id,
            "town": self.scn.town,
            "expected": self.scn.expected,
            "final_phase": self.phase,
            "ego_ahead_of_lead": ahead,
            "collision": self.collision,
            "offroad": self.offroad,
            "max_lane_dev_m": round(self.max_lane_dev, 2),
            "passing_side": self.passing_side,
            "success": completed,
            "frames": self._cap_idx,
        }
        return result

    def write_video(self, name: str, fps: int = 18) -> Optional[Path]:
        try:
            import imageio.v3 as iio
        except ImportError:
            return None
        from PIL import Image
        paths = sorted(self.frames_dir.glob("frame_*.png"))
        if not paths:
            return None
        frames = [np.array(Image.open(p)) for p in paths]
        out = self.out_dir / name
        iio.imwrite(out, frames, fps=fps, codec="libx264")
        return out

    def teardown(self) -> None:
        for s in self.sensors.values():
            try:
                s.stop(); s.destroy()
            except Exception:
                pass
        for a in self.actors.values():
            try:
                a.destroy()
            except Exception:
                pass


def run_scenario(scn: OvertakeScenario, out_dir: Path, *, host="127.0.0.1", port=2000) -> Dict[str, Any]:
    import carla
    import json
    client = carla.Client(host, port)
    client.set_timeout(30.0)
    world, prev = load_world(carla, client, scn.town)
    run = OvertakeRun(carla, world, scn, out_dir)
    try:
        run.setup()
        result = run.run()
        mp4 = run.write_video(f"{scn.scenario_id}.mp4")
        result["video"] = str(mp4) if mp4 else None
        out = Path(out_dir)
        (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (out / "trace.json").write_text(json.dumps(run.trace, indent=2), encoding="utf-8")
        return result
    finally:
        run.teardown()
        restore_world(world, prev)

