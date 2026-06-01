"""
Spawn curated AutoPass scenarios inside CARLA so you see vehicles in the simulator.

Requires:
  - CarlaUE4.exe running (0.9.16 recommended, matching pip install carla==0.9.16)
  - import carla works in your venv (Python 3.10–3.12)

Usage:
  from perception.carla_scenario import bootstrap_carla_scenario
  bootstrap_carla_scenario(spec, world, map_name="Town04")
"""
from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import numpy as np

from autopass.carla_tuning import rear_follow_min_m, route_lookahead_m, safe_follow_m

from visual_world import ScenarioSpec, WorldState

_session: Optional["CarlaScenarioSession"] = None


def _normalize_map_name(map_name: str) -> str:
    """Short CARLA map token (e.g. Town04) for stable session reuse."""
    if not map_name:
        return "Town04"
    for token in ("Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10"):
        if token in map_name:
            return token
    return map_name.split("/")[-1].replace(".umap", "") or map_name


def reset_carla_session_for_tests() -> None:
    """Clear module singleton between orchestration tests."""
    global _session
    if _session is not None:
        _session.shutdown()
    _session = None


class CarlaScenarioSession:
    def __init__(self) -> None:
        self.client = None
        self.world = None
        self.map = None
        self.carla = None
        self.actors: Dict[str, object] = {}
        self.sensors: Dict[str, object] = {}
        self.anchor_wp = None
        self._travel_wp = None
        self._route_cursor = None
        self._passing_wp = None
        self._passing_side: str = "left"
        self._opposing_wp = None
        self._pass_maneuver_validated = False
        self._pass_validation_in_progress = False
        self._rgb_buf = None
        self._depth_buf = None
        self._seg_buf = None
        self._overhead_buf = None
        self.ready = False
        self.last_error: Optional[str] = None
        self._spawn_lead_m = 20.0
        self._spawn_rear_m = 25.0
        self._spawn_on_m = 40.0
        self._ego_physics = False
        self.fixed_delta_seconds = 0.05
        self._map_name: Optional[str] = None
        self._scenario_id: Optional[str] = None
        self._spawn_ego_s: Optional[float] = None
        self._spawn_logical_x: Optional[float] = None
        self._physical_key: Optional[str] = None
        self.map_load_count: int = 0
        self.last_bootstrap_action: str = "none"
        self._episode_step: int = 0
        self._collision_events: list = []
        self._spawn_settle_ticks: int = 5
        self._sensor_frame_counts: Dict[str, int] = {"rgb": 0, "depth": 0, "seg": 0, "overhead": 0}
        self._sensor_last_frame: Dict[str, int] = {"rgb": -1, "depth": -1, "seg": -1, "overhead": -1}
        self._sensor_listeners: Dict[str, object] = {}
        self._sensor_callback_errors: list = []
        self._sensor_warmup_ticks: int = 0
        self._sensor_warmup_method: str = "none"
        self._last_steer: float = 0.0
        self._lane_departure_stopped: bool = False
        self._corridor_report = None
        self._corridor_hero_fallback = False
        self._last_corridor_diagnostics = None
        self._corridor_pick_pool: list = []
        self._rear_on_passing_lane: bool = False
        self._axis_ego_xyz: Optional[Tuple[float, float, float]] = None
        self._axis_travel_dir: Optional[Tuple[float, float, float]] = None
        self._axis_lateral_dir: Optional[Tuple[float, float, float]] = None
        self._axis_spawn_active: bool = False
        from perception.actor_continuity import reset_continuity_state

        reset_continuity_state(self)

    def allows_pre_decision_actor_layout(self) -> bool:
        from perception.actor_continuity import allows_pre_decision_actor_layout

        return allows_pre_decision_actor_layout(self)

    def mark_closed_loop_actuation_begun(self) -> None:
        from perception.actor_continuity import mark_closed_loop_actuation_begun

        mark_closed_loop_actuation_begun(self)

    def mark_graph_execute_completed(self) -> None:
        from perception.actor_continuity import mark_graph_execute_completed

        mark_graph_execute_completed(self)

    def longitudinal_continuity_diag(self, **kwargs):
        from perception.actor_continuity import longitudinal_continuity_diag

        return longitudinal_continuity_diag(self, **kwargs)

    def _travel_axis(self):
        """
        Authoritative travel axis for perception gaps (ego-relative).

        CARLA actor poses update on world.tick() in synchronous mode; gap math uses
        live ego position when available so it matches axis spawn placement.
        Direction comes from the spawn travel waypoint forward vector.
        """
        fwd_tuple: Optional[Tuple[float, float, float]] = None
        if self._travel_wp is not None:
            try:
                fwd = self._travel_wp.transform.get_forward_vector()
                mag = math.sqrt(fwd.x * fwd.x + fwd.y * fwd.y + fwd.z * fwd.z)
                if mag > 1e-6:
                    fwd_tuple = (fwd.x / mag, fwd.y / mag, fwd.z / mag)
            except Exception:
                pass
        if fwd_tuple is None:
            cached = getattr(self, "_axis_travel_dir", None)
            if cached is not None:
                fwd_tuple = cached
        if fwd_tuple is None:
            return None
        origin = None
        ego = self.actors.get("ego")
        if ego is not None:
            try:
                loc = ego.get_location()
                if abs(float(loc.x)) + abs(float(loc.y)) > 1.0:
                    origin = (float(loc.x), float(loc.y), float(loc.z))
            except Exception:
                pass
        if origin is None:
            cached_ego = getattr(self, "_axis_ego_xyz", None)
            if cached_ego is not None:
                origin = cached_ego
        if origin is None and self._travel_wp is not None:
            try:
                loc = self._travel_wp.transform.location
                origin = (float(loc.x), float(loc.y), float(loc.z))
            except Exception:
                return None
        if origin is None:
            return None
        return origin, fwd_tuple

    def _layout_tick_sync(self) -> None:
        """Apply pending transforms before reading actor locations (sync mode)."""
        if self.is_synchronous_mode() and self.world is not None:
            self.world.tick()

    def project_actor_along_travel_axis(self, actor_or_name) -> Optional[float]:
        """Signed scalar s from travel-origin along travel-direction; None on failure."""
        axis = self._travel_axis()
        if axis is None:
            return None
        origin, fwd = axis
        actor = actor_or_name
        if isinstance(actor_or_name, str):
            actor = self.actors.get(actor_or_name)
        if actor is None:
            return None
        try:
            loc = actor.get_location()
        except Exception:
            return None
        ox, oy, oz = self._xyz_components(origin)
        return (loc.x - ox) * fwd[0] + (loc.y - oy) * fwd[1] + (loc.z - oz) * fwd[2]

    def longitudinal_gap(self, reference_actor, target_actor) -> Optional[float]:
        s_ref = self.project_actor_along_travel_axis(reference_actor)
        s_tgt = self.project_actor_along_travel_axis(target_actor)
        if s_ref is None or s_tgt is None:
            return None
        return s_tgt - s_ref

    def signed_gap_from_ego(self, actor_name: str) -> Optional[float]:
        ego = self.actors.get("ego")
        actor = self.actors.get(actor_name)
        if ego is None or actor is None:
            return None
        return self.longitudinal_gap(ego, actor)

    def lane_identity(self, actor_name: str) -> Optional[Dict[str, int]]:
        if self.map is None:
            return None
        actor = self.actors.get(actor_name)
        if actor is None:
            return None
        try:
            wp = self.map.get_waypoint(actor.get_location(), project_to_road=True)
            return {"lane_id": int(wp.lane_id), "road_id": int(wp.road_id)}
        except Exception:
            return None

    def init_logical_anchor(self, logical_ego_x: float) -> None:
        """Bind logical 1D ego_x_m to CARLA travel-axis position at spawn."""
        self._spawn_logical_x = logical_ego_x
        self._spawn_ego_s = self._ego_travel_s()
        self._last_logical_t_s = 0.0

    def graph_logical_t_s(self) -> float:
        """Authoritative closed-loop sim time (updated on each execute materialize)."""
        return float(getattr(self, "_last_logical_t_s", 0.0))

    def _ego_travel_s(self) -> float:
        s = self.project_actor_along_travel_axis("ego")
        return float(s) if s is not None else 0.0

    def materialize_logical_world(
        self,
        world_before: WorldState,
        *,
        measured_speed_mps: float,
        duration_s: float,
        ego_lane: int,
        passed: bool,
        collision: bool,
        done: bool,
    ) -> WorldState:
        """Project CARLA actor layout into the 1D WorldState used by the graph."""
        min_rear = 3.5
        min_lead = 2.5
        ego_x = world_before.ego_x_m + measured_speed_mps * duration_s
        if self._travel_wp is not None and self.actors.get("ego"):
            ego_s = self._ego_travel_s()
            if self._spawn_ego_s is None or self._spawn_logical_x is None:
                self.init_logical_anchor(world_before.ego_x_m)
                ego_s = self._ego_travel_s()
            ego_x = self._spawn_logical_x + (ego_s - self._spawn_ego_s)

        if self.actors.get("lead"):
            lead_signed = self.signed_gap_from_ego("lead")
            if lead_signed is None:
                lead_x = world_before.lead_x_m + (ego_x - world_before.ego_x_m)
            else:
                lead_x = ego_x + max(min_lead, lead_signed)
        else:
            lead_x = world_before.lead_x_m + (ego_x - world_before.ego_x_m)
        if self.actors.get("rear"):
            rear_signed = self.signed_gap_from_ego("rear")
            if rear_signed is None:
                rear_x = world_before.rear_x_m + (ego_x - world_before.ego_x_m)
            else:
                rear_x = ego_x - max(min_rear, -rear_signed)
        else:
            rear_x = world_before.rear_x_m + (ego_x - world_before.ego_x_m)
        rear_x = min(rear_x, ego_x - min_rear)
        if passed:
            lead_x = min(lead_x, ego_x - min_lead - 2.0)

        delta = ego_x - world_before.ego_x_m
        oncoming_x = world_before.oncoming_x_m + delta
        oncoming = self.actors.get("oncoming")
        if oncoming is not None:
            along = self.signed_gap_from_ego("oncoming")
            if along is not None and along > 5.0:
                oncoming_x = ego_x + along

        out = WorldState(
            t_s=world_before.t_s + duration_s,
            ego_x_m=ego_x,
            ego_lane=ego_lane,
            ego_speed_mps=measured_speed_mps,
            lead_x_m=lead_x,
            rear_x_m=rear_x,
            oncoming_x_m=oncoming_x,
            passed=passed,
            collision=collision,
            done=done,
        )
        self._last_logical_t_s = float(out.t_s)
        return out

    def _zero_actor_kinematics(self, actor) -> None:
        """Clear linear/angular velocity before toggling physics (prevents spawn spin)."""
        if actor is None or self.carla is None:
            return
        carla = self.carla
        try:
            actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        except Exception:
            pass

    def snap_ego_to_travel_pose(self, *, max_lateral_m: float = 4.0) -> bool:
        """Snap ego position and yaw to the spawn travel lane at current longitudinal s."""
        ego = self.actors.get("ego")
        tw = self._travel_wp
        if ego is None or tw is None or self.map is None or self.carla is None:
            return False
        anchor = self._travel_lane_anchor_at_ego(ego) or tw
        from perception.carla_lane_keep import lane_center_distance_m

        loc = ego.get_location()
        if lane_center_distance_m(loc, anchor) > max_lateral_m:
            return False
        carla = self.carla
        wp_loc = anchor.transform.location
        rot = anchor.transform.rotation
        was_phys = bool(getattr(self, "_ego_physics", False))
        if was_phys:
            ego.set_simulate_physics(False)
        ego.set_transform(
            carla.Transform(
                carla.Location(float(wp_loc.x), float(wp_loc.y), float(wp_loc.z) + 0.25),
                carla.Rotation(float(rot.pitch), float(rot.yaw), float(rot.roll)),
            )
        )
        self._zero_vehicle_control(ego)
        self._zero_actor_kinematics(ego)
        self._last_steer = 0.0
        if was_phys:
            ego.set_simulate_physics(True)
            self._zero_actor_kinematics(ego)
        return True

    def apply_inter_step_cruise(self, spec: ScenarioSpec, world: WorldState) -> None:
        """Hold travel-lane cruise between graph nodes (avoid stale pass steer spinning ego)."""
        ego = self.actors.get("ego")
        if ego is None or not getattr(self, "_ego_physics", False):
            return
        from perception.carla_control import build_vehicle_control

        gaps = self.measure_actor_gaps_3d()
        ctrl = build_vehicle_control(
            "wait",
            world=world,
            spec=spec,
            session=self,
            ego=ego,
            measured_speed_mps=self._ego_speed_mps(),
            front_gap_m=gaps.get("front", 999.0),
            clear_of_lead=self.ego_clear_of_lead(),
            ego_lane=self.infer_ego_lane_index(),
            recovery=False,
        )
        ego.apply_control(ctrl)
        self._last_vehicle_control = ctrl
        self._last_steer = float(ctrl.steer)

    def enable_ego_physics(self, enabled: bool = True) -> None:
        """Enable Unreal physics on ego only; NPCs stay kinematic."""
        from autopass.config import AutopassConfigurationError

        ego = self.actors.get("ego")
        if self.world is None or ego is None:
            raise AutopassConfigurationError(
                "CARLA ego vehicle missing. Bootstrap the scenario before execute_vehicle_step."
            )
        if not self.ready and not getattr(self, "_pass_validation_in_progress", False):
            # Normal execute path should go through bootstrap_carla_scenario() first.
            if self.client is None:
                raise AutopassConfigurationError(
                    "CARLA client not connected. Start CarlaUE4.exe, then bootstrap."
                )
        self._ego_physics = enabled
        if enabled:
            self._ensure_spawn_gaps()
            ego.set_autopilot(False)
            self.align_ego_to_travel_lane(max_lateral_m=3.5)
            self.snap_ego_to_travel_pose(max_lateral_m=4.0)
            self._zero_vehicle_control(ego)
            ego.set_simulate_physics(False)
            if self.world is not None:
                self.tick()
            self._zero_actor_kinematics(ego)
            ego.set_simulate_physics(True)
            self._zero_actor_kinematics(ego)
            self._zero_vehicle_control(ego)
            if self.world is not None:
                for _ in range(2):
                    self.tick()
                    self._zero_actor_kinematics(ego)
        else:
            ego.set_simulate_physics(False)

    def measure_actor_gaps_3d(self) -> Dict[str, float]:
        from perception.carla_geometry import actor_location_tuple, euclidean_m

        ego = self.actors.get("ego")
        if ego is None:
            return {"front": 999.0, "rear": 999.0, "oncoming": 999.0}
        ego_xyz = actor_location_tuple(ego)
        if ego_xyz is None:
            return {"front": 999.0, "rear": 999.0, "oncoming": 999.0}
        ego_loc = type("L", (), {"x": ego_xyz[0], "y": ego_xyz[1], "z": ego_xyz[2]})()
        out = {"front": 999.0, "rear": 999.0, "oncoming": 999.0}
        for name, key in (("lead", "front"), ("rear", "rear"), ("oncoming", "oncoming")):
            actor = self.actors.get(name)
            if actor is None:
                continue
            other_xyz = actor_location_tuple(actor)
            if other_xyz is None:
                continue
            other_loc = type("L", (), {"x": other_xyz[0], "y": other_xyz[1], "z": other_xyz[2]})()
            out[key] = euclidean_m(ego_loc, other_loc)
        return out

    def actor_travel_s(self, actor_name: str) -> Optional[float]:
        """Signed progress of an actor along the curated travel axis (meters)."""
        s = self.project_actor_along_travel_axis(actor_name)
        return float(s) if s is not None else None

    def ego_clear_of_lead(self, merge_clear_m: float = 8.0) -> bool:
        return self.ego_cleared_lead(merge_clear_m)

    def ego_cleared_lead(self, clearance_m: float = 8.0) -> bool:
        """True when ego is at least ``clearance_m`` ahead of lead along the travel corridor."""
        ego_s = self.actor_travel_s("ego")
        lead_s = self.actor_travel_s("lead")
        if ego_s is not None and lead_s is not None:
            return ego_s > lead_s + float(clearance_m)
        ego = self.actors.get("ego")
        lead = self.actors.get("lead")
        if ego is None or lead is None:
            return False
        along = self.longitudinal_gap(ego, lead)
        if along is not None:
            return along < -float(clearance_m)
        if self.map is None:
            return False
        try:
            ego_wp = self.map.get_waypoint(ego.get_location(), project_to_road=True)
            fwd = ego_wp.transform.get_forward_vector()
            ego_loc = ego.get_location()
            lead_loc = lead.get_location()
            fallback_along = (lead_loc.x - ego_loc.x) * fwd.x + (lead_loc.y - ego_loc.y) * fwd.y
            return fallback_along < -float(clearance_m)
        except Exception:
            return False

    def pass_longitudinal_snapshot(self) -> Dict[str, Optional[float]]:
        """Signed travel-axis positions for pass maneuver diagnostics."""
        return {
            "ego_s_m": self.actor_travel_s("ego"),
            "lead_s_m": self.actor_travel_s("lead"),
            "rear_s_m": self.actor_travel_s("rear"),
        }

    def _ego_speed_mps(self) -> float:
        ego = self.actors.get("ego")
        if ego is None:
            return 0.0
        v = ego.get_velocity()
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    def _travel_forward_vector(self):
        axis = self._travel_axis()
        if axis is not None:
            _, fwd = axis
            return type("V", (), {"x": fwd[0], "y": fwd[1], "z": fwd[2]})()
        ego = self.actors.get("ego")
        if ego is None or self.map is None:
            return None
        wp = self.map.get_waypoint(ego.get_location(), project_to_road=True)
        return wp.transform.get_forward_vector()

    def _longitudinal_along_travel(self, other) -> Optional[float]:
        """Offset along the scenario travel axis (+ = ahead of ego, - = behind)."""
        ego = self.actors.get("ego")
        if ego is None:
            return None
        return self.longitudinal_gap(ego, other)

    def rear_longitudinal_gap_m(self) -> float:
        rear = self.actors.get("rear")
        if rear is None:
            return 999.0
        along = self._longitudinal_along_travel(rear)
        if along is None:
            return 999.0
        return max(0.0, -self._coerce_gap_m(along, default=999.0))

    @staticmethod
    def _coerce_gap_m(value: object, *, default: float = 999.0) -> float:
        """Return a finite gap in meters; non-numeric values → default (mock actors, pre-tick)."""
        if value is None:
            return default
        try:
            gap = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(gap):
            return default
        return gap

    def lead_longitudinal_gap_m(self) -> float:
        lead = self.actors.get("lead")
        if lead is None:
            return 999.0
        along = self._longitudinal_along_travel(lead)
        if along is None:
            return 999.0
        return max(0.0, self._coerce_gap_m(along, default=999.0))

    def ego_convoy_misaligned(self) -> bool:
        """True when ego has left the NPC convoy axis (e.g. turning at a junction)."""
        ego = self.actors.get("ego")
        lead = self.actors.get("lead")
        if ego is None or lead is None:
            return False
        if ego.get_location().distance(lead.get_location()) < 6.0:
            return False
        return self.lead_longitudinal_gap_m() < 3.0

    def update_route_cursor(self, ego) -> None:
        """Keep cruise reference on the spawn travel lane slightly ahead of ego."""
        if self._travel_wp is None or ego is None:
            return
        anchor = self._travel_lane_anchor_at_ego(ego)
        if anchor is None:
            return
        tw = self._travel_wp
        self._route_cursor = self._wp_on_lane_ahead(anchor, 2.0, tw.lane_id, tw.road_id)

    def _wp_on_lane_ahead(
        self,
        wp,
        distance_m: float,
        lane_id=None,
        road_id=None,
        *,
        same_carriageway: bool = False,
    ):
        """Advance along the same lane/road; stop at junctions and lane changes."""
        if wp is None or distance_m < 0.5:
            return wp
        if lane_id is None:
            lane_id = wp.lane_id
        if road_id is None:
            road_id = wp.road_id
        cur = wp
        remaining = float(distance_m)
        while remaining > 0.5:
            step = min(4.0, remaining)
            nxt = cur.next(step)
            cand = self._pick_next_waypoint(nxt, lane_id, road_id, same_carriageway=same_carriageway)
            if cand is None:
                break
            if getattr(cand, "is_junction", False):
                break
            if int(cand.road_id) != int(road_id):
                break
            if same_carriageway:
                if int(cand.lane_id) * int(lane_id) <= 0:
                    break
            elif int(cand.lane_id) != int(lane_id):
                break
            cur = cand
            remaining -= step
        return cur

    def _pick_next_waypoint(self, nxt, lane_id, road_id, *, same_carriageway: bool = False):
        """When CARLA returns multiple next waypoints, stay on the curated corridor."""
        if not nxt:
            return None
        if len(nxt) == 1:
            return nxt[0]
        for cand in nxt:
            if int(cand.road_id) != int(road_id):
                continue
            if getattr(cand, "is_junction", False):
                continue
            if same_carriageway:
                if int(cand.lane_id) * int(lane_id) <= 0:
                    continue
            elif int(cand.lane_id) != int(lane_id):
                continue
            return cand
        return None

    def _corridor_actor_transform(
        self,
        ahead_m: float,
        *,
        lane: str = "travel",
    ):
        """Place actor on curated corridor waypoints (never raw world-axis offset)."""
        carla = self.carla
        tw = self._travel_wp
        if tw is None or carla is None:
            return None
        dist = float(ahead_m)
        if lane == "passing" and self._passing_wp is not None:
            base = self._passing_wp
            lane_id = base.lane_id
            road_id = base.road_id
        else:
            base = tw
            lane_id = tw.lane_id
            road_id = tw.road_id
        if dist >= 0.0:
            wp = self._wp_on_lane_ahead(
                base, dist, lane_id, road_id, same_carriageway=True
            )
        else:
            wp = self._wp_behind(base, abs(dist))
        if wp is None:
            return None
        if lane != "passing" and not self._same_carriageway(wp, tw):
            wp = self._wp_on_lane_ahead(tw, abs(dist), tw.lane_id, tw.road_id, same_carriageway=True)
            if wp is None:
                return None
        loc = wp.transform.location
        rot = wp.transform.rotation
        return carla.Transform(
            carla.Location(float(loc.x), float(loc.y), float(loc.z) + 0.3),
            carla.Rotation(rot.pitch, rot.yaw, rot.roll),
        )

    def _travel_lane_anchor_at_ego(self, ego):
        """Waypoint on the spawn travel lane at ego's longitudinal position."""
        if self._travel_wp is None or self.map is None or ego is None:
            return self._travel_wp
        try:
            ego_wp = self.map.get_waypoint(ego.get_location(), project_to_road=True)
        except Exception:
            return self._travel_wp
        tw = self._travel_wp
        if ego_wp.road_id == tw.road_id and ego_wp.lane_id == tw.lane_id:
            return ego_wp
        s_ego = self.project_actor_along_travel_axis(ego)
        if s_ego is None or s_ego <= 0.5:
            return tw
        return self._wp_on_lane_ahead(tw, s_ego, tw.lane_id, tw.road_id)

    def expected_passing_lane_width_m(self) -> float:
        """Lateral spacing between spawn travel and passing lane centers."""
        tw = self._travel_wp
        pw = self._passing_wp
        if tw is None or pw is None:
            return 3.5
        try:
            return max(2.5, float(tw.transform.location.distance(pw.transform.location)))
        except Exception:
            return 3.5

    def lateral_shift_toward_passing_m(self, ego) -> float:
        """Meters ego has shifted from travel lane toward the passing lane (0 = on travel)."""
        from perception.carla_lane_keep import lane_center_distance_m

        travel = self._travel_lane_anchor_at_ego(ego)
        if ego is None or travel is None or self._passing_wp is None:
            return 0.0
        passing = self._adjacent_passing_lane_wp(travel, self._passing_side or "left")
        if passing is None:
            return 0.0
        loc = ego.get_location()
        d_travel = lane_center_distance_m(loc, travel)
        d_pass = lane_center_distance_m(loc, passing)
        return max(0.0, float(d_travel) - float(d_pass))

    def lane_change_blend_alpha(self, ego) -> float:
        from perception.carla_lane_keep import lane_change_blend_alpha

        return lane_change_blend_alpha(
            self.lateral_shift_toward_passing_m(ego),
            self.expected_passing_lane_width_m(),
        )

    def _lane_change_blend_at(
        self, travel, passing, *, alpha: float, lookahead_m: float | None = None
    ):
        from perception.carla_lane_keep import blend_locations

        if travel is None or passing is None:
            return None
        if lookahead_m is not None and lookahead_m > 0.0:
            la = float(lookahead_m)
            travel_pt = self._wp_on_lane_ahead(
                travel, la, travel.lane_id, travel.road_id
            ) or travel
            pass_pt = self._wp_on_lane_ahead(
                passing, la, passing.lane_id, passing.road_id
            ) or passing
            t_loc = travel_pt.transform.location
            p_loc = pass_pt.transform.location
        else:
            t_loc = travel.transform.location
            p_loc = passing.transform.location
        return blend_locations(t_loc, p_loc, alpha)

    def get_lane_change_path_center_at_ego(self, ego):
        """Blend lane centers at ego longitude (for metrics — not the steer lookahead)."""
        travel = self._travel_lane_anchor_at_ego(ego) or self._travel_wp
        if travel is None or self._passing_wp is None or ego is None:
            return None
        passing = self._adjacent_passing_lane_wp(travel, self._passing_side or "left")
        if passing is None:
            return None
        width = self.expected_passing_lane_width_m()
        shift = self.lateral_shift_toward_passing_m(ego)
        alpha = min(1.0, max(0.0, float(shift) / width))
        return self._lane_change_blend_at(travel, passing, alpha=alpha, lookahead_m=None)

    def get_lane_change_steer_target_location(self, ego):
        """
        Short-lookahead point between travel and passing lanes on the curated corridor.

        Avoids steering toward a distant adjacent-lane waypoint (wide shallow arc).
        """
        travel = self._travel_lane_anchor_at_ego(ego) or self._travel_wp
        if travel is None or self._passing_wp is None or ego is None:
            return None
        passing = self._adjacent_passing_lane_wp(travel, self._passing_side or "left")
        if passing is None:
            return None
        la = min(6.0, max(3.0, route_lookahead_m() * 0.32))
        alpha = self.lane_change_blend_alpha(ego)
        return self._lane_change_blend_at(travel, passing, alpha=alpha, lookahead_m=la)

    def get_passing_lane_steering_waypoint(self, ego, *, lookahead_m: float | None = None):
        """Lookahead on the spawn adjacent passing lane centerline (not travel lane)."""
        from types import SimpleNamespace

        from autopass.carla_tuning import route_lookahead_m

        tw = self._travel_wp
        pw = self._passing_wp
        if pw is None or ego is None or tw is None:
            return pw
        la = float(lookahead_m) if lookahead_m is not None else min(8.0, route_lookahead_m() * 0.55)
        travel = self._travel_lane_anchor_at_ego(ego) or tw
        passing = self._adjacent_passing_lane_wp(travel, self._passing_side or "left") or pw
        ahead = self._wp_on_lane_ahead(passing, la, int(pw.lane_id), int(pw.road_id))
        ref = ahead or passing
        return SimpleNamespace(
            lane_id=int(pw.lane_id),
            road_id=int(pw.road_id),
            transform=ref.transform,
        )

    def _lane_change_steer_waypoint(self, ego, passing_side: str = "left"):
        """Steering reference on corridor road_id/lane_id with blended lateral target."""
        from types import SimpleNamespace

        pw = self._passing_wp
        travel = self._travel_lane_anchor_at_ego(ego) or self._travel_wp
        loc = self.get_lane_change_steer_target_location(ego)
        if travel is None or loc is None:
            return self._travel_wp
        alpha = self.lane_change_blend_alpha(ego) if ego is not None else 0.0
        passing = self._adjacent_passing_lane_wp(travel, self._passing_side or passing_side)
        ref_meta = passing if (pw is not None and alpha >= 0.1 and passing is not None) else travel
        if pw is not None and alpha >= 0.1:
            ref_meta = pw
        return SimpleNamespace(
            lane_id=int(ref_meta.lane_id),
            road_id=int(ref_meta.road_id),
            transform=SimpleNamespace(
                location=loc,
                rotation=travel.transform.rotation,
            ),
        )

    def ego_on_passing_lane(self, ego) -> bool:
        if ego is None or self.map is None or self._passing_wp is None:
            return False
        try:
            wp = self.map.get_waypoint(ego.get_location(), project_to_road=True)
        except Exception:
            return False
        pw = self._passing_wp
        return int(wp.road_id) == int(pw.road_id) and int(wp.lane_id) == int(pw.lane_id)

    def passing_lane_horizon_from_spawn_m(self, *, probe_m: float = 80.0) -> float:
        """Forward meters on the passing lane from spawn before road/lane change."""
        pw = self._passing_wp
        tw = self._travel_wp
        if pw is None or tw is None:
            return 0.0
        start = self._adjacent_passing_lane_wp(tw, self._passing_side or "left") or pw
        end = self._wp_on_lane_ahead(start, probe_m, pw.lane_id, pw.road_id)
        if end is None or start is None:
            return 0.0
        try:
            return float(start.transform.location.distance(end.transform.location))
        except Exception:
            return 0.0

    def _adjacent_passing_lane_wp(self, ego_wp, passing_side: str = "left"):
        if self._passing_wp is None or ego_wp is None:
            return self._passing_wp
        carla = self.carla
        target_id = self._passing_wp.lane_id
        target_road = self._passing_wp.road_id
        if ego_wp.lane_id == target_id and ego_wp.road_id == target_road:
            return ego_wp
        if not hasattr(ego_wp, "get_left_lane") or not hasattr(ego_wp, "get_right_lane"):
            return self._passing_wp
        side = self._passing_side or passing_side
        adj = ego_wp.get_left_lane() if side == "left" else ego_wp.get_right_lane()
        if adj and adj.lane_type == carla.LaneType.Driving and adj.lane_id * ego_wp.lane_id > 0:
            if adj.lane_id == target_id and adj.road_id == target_road:
                return adj
        return self._passing_wp

    def _ego_lane_anchor_waypoint(self, ego):
        """Lane-center reference: compare travel vs passing anchor, pick nearer lane."""
        from perception.carla_lane_keep import lane_center_distance_m

        if ego is None or self.map is None or self._travel_wp is None:
            return None
        loc = ego.get_location()
        travel = self._travel_lane_anchor_at_ego(ego) or self._travel_wp
        d_travel = lane_center_distance_m(loc, travel)
        pw = self._passing_wp
        if pw is None:
            return travel
        passing = self._adjacent_passing_lane_wp(travel, self._passing_side or "left")
        if passing is None:
            return travel
        d_pass = lane_center_distance_m(loc, passing)
        if d_pass + 0.3 < d_travel and d_pass < 2.5:
            return passing
        return travel

    def remaining_lane_horizon_m(
        self, ego, lane_id: int, road_id: int, *, max_probe: float = 35.0
    ) -> float:
        """Meters of same-lane road ahead before junction / road-id change."""
        if ego is None or self._travel_wp is None:
            return 0.0
        anchor = self._travel_lane_anchor_at_ego(ego) or self._travel_wp
        if anchor is None:
            return 0.0
        if anchor.lane_id != lane_id or anchor.road_id != road_id:
            anchor = self._wp_on_lane_ahead(self._travel_wp, 0.0, lane_id, road_id) or anchor
        end = self._wp_on_lane_ahead(anchor, max_probe, lane_id, road_id)
        if end is None:
            return 0.0
        try:
            return float(anchor.transform.location.distance(end.transform.location))
        except Exception:
            return 0.0

    def approaching_corridor_end(self, ego, *, min_horizon_m: float = 14.0) -> bool:
        """True when curated travel or passing lane runs out ahead of ego."""
        pw = self._passing_wp
        tw = self._travel_wp
        if ego is None or tw is None:
            return False
        travel_left = self.remaining_lane_horizon_m(ego, tw.lane_id, tw.road_id)
        if travel_left < min_horizon_m:
            return True
        if pw is not None:
            passing_left = self.remaining_lane_horizon_m(ego, pw.lane_id, pw.road_id)
            if passing_left < min_horizon_m:
                return True
        return False

    def get_travel_steering_waypoint(self, ego, *, lookahead_m: float | None = None):
        """
        Cruise / wait steering target on the spawn travel lane at ego longitude.

        Uses the curated travel axis (not a long wp.next chain) so ego on an adjacent
        projected lane still steers toward the correct centerline.
        """
        from types import SimpleNamespace

        from autopass.carla_tuning import route_lookahead_m

        tw = self._travel_wp
        if tw is None or ego is None:
            return tw
        la = float(lookahead_m) if lookahead_m is not None else min(8.0, route_lookahead_m() * 0.5)
        anchor = self._travel_lane_anchor_at_ego(ego) or tw
        axis = self._travel_axis()
        if axis is not None:
            _, fwd = axis
            loc = anchor.transform.location
            carla = self.carla
            target_loc = carla.Location(
                float(loc.x) + float(fwd[0]) * la,
                float(loc.y) + float(fwd[1]) * la,
                float(loc.z) + float(fwd[2]) * la,
            )
            return SimpleNamespace(
                lane_id=int(tw.lane_id),
                road_id=int(tw.road_id),
                transform=SimpleNamespace(
                    location=target_loc,
                    rotation=anchor.transform.rotation,
                ),
            )
        return self._wp_on_lane_ahead(anchor, la, tw.lane_id, tw.road_id)

    def align_ego_to_travel_lane(self, *, max_lateral_m: float = 2.5) -> bool:
        """Snap ego onto spawn travel lane center if map projection put it on an adjacent lane."""
        ego = self.actors.get("ego")
        tw = self._travel_wp
        if ego is None or tw is None or self.map is None:
            return False
        try:
            ego_wp = self.map.get_waypoint(ego.get_location(), project_to_road=True)
        except Exception:
            return False
        if ego_wp.road_id == tw.road_id and ego_wp.lane_id == tw.lane_id:
            return False
        anchor = self._travel_lane_anchor_at_ego(ego)
        if anchor is None:
            return False
        from perception.carla_lane_keep import lane_center_distance_m

        if lane_center_distance_m(ego.get_location(), anchor) > max_lateral_m:
            return False
        carla = self.carla
        loc = anchor.transform.location
        rot = anchor.transform.rotation
        ego.set_transform(
            carla.Transform(
                carla.Location(float(loc.x), float(loc.y), float(loc.z) + 0.25),
                carla.Rotation(float(rot.pitch), float(rot.yaw), float(rot.roll)),
            )
        )
        print(
            f"[CARLA] Aligned ego to travel lane {tw.road_id}/{tw.lane_id} "
            f"(was {ego_wp.road_id}/{ego_wp.lane_id})",
            flush=True,
        )
        return True

    def _pass_lead_gap_floor_m(self, spec: ScenarioSpec | None = None) -> float:
        """Minimum lead gap for pre-control sanity (axis pass demos need >=26m)."""
        if spec is not None:
            profile = self._spawn_profile(spec)
            if profile.get("axis_spawn"):
                return max(26.0, float(profile.get("lead_floor_m", 26.0)))
            return float(profile.get("lead_floor_m", 10.0))
        return 26.0

    def _restore_axis_spawn_layout(self, spec: ScenarioSpec) -> None:
        """Re-apply axis longitudinal layout from live ego (after corridor/tick side effects)."""
        if not self._uses_axis_spawn(spec):
            return
        profile = self._spawn_profile(spec)
        self._spawn_lead_m = max(
            float(self._spawn_lead_m),
            float(profile.get("lead_gap_m", self._spawn_lead_m)),
        )
        rear_m = float(profile.get("rear_spawn_m", self._spawn_rear_m))
        self._spawn_rear_m = min(float(self._spawn_rear_m), rear_m + 5.0)
        self.refresh_axis_ego_from_live()
        self._place_actor_longitudinal("lead", self._spawn_lead_m, lane="travel", spec=spec)
        if self._rear_on_passing_lane and self.actors.get("rear") is not None:
            self._hold_rear_on_passing_lane(rear_m)
        self._layout_tick_sync()

    def _finalize_spawn_layout(self, spec: ScenarioSpec) -> None:
        """Ensure convoy gaps and lane coherence before pre-control checks."""
        profile = self._spawn_profile(spec)
        floor_m = max(float(self._spawn_lead_m), self._pass_lead_gap_floor_m(spec))
        if profile.get("axis_spawn"):
            floor_m = max(floor_m, float(profile.get("lead_gap_m", floor_m)))
        self._spawn_lead_m = floor_m
        if profile.get("axis_spawn"):
            rear_cap = float(profile.get("rear_spawn_m", 18.0)) + 5.0
            self._spawn_rear_m = min(float(self._spawn_rear_m), rear_cap)
        self._snap_convoy_to_travel_lane(spec)
        self._layout_tick_sync()
        self._extend_lead_to_target_gap(self._spawn_lead_m, spec=spec)
        if profile.get("axis_spawn"):
            self._restore_axis_spawn_layout(spec)
        elif self._rear_on_passing_lane and self.actors.get("rear") is not None:
            tf = self._corridor_actor_transform(-float(self._spawn_rear_m), lane="passing")
            if tf is not None:
                self.actors["rear"].set_transform(tf)
            self._layout_tick_sync()
        self.assert_convoy_on_travel_corridor(spec)

    def run_pre_control_sanity(
        self,
        *,
        for_follow_lead: bool = True,
        align_ego: bool = True,
        check_spawn_gaps: bool = True,
        spec: ScenarioSpec | None = None,
    ) -> Dict[str, object]:
        """Log diagnostics; optionally align ego; fail fast unless test mode."""
        from autopass.config import is_test_mode
        from perception.carla_pre_control import assert_pre_control_sanity, log_pre_control_diagnostic

        if spec is None:
            spec = getattr(self, "_bootstrap_spec", None)
        if spec is not None and self._uses_axis_spawn(spec):
            self._restore_axis_spawn_layout(spec)
        elif self.is_synchronous_mode():
            self._layout_tick_sync()
        if align_ego:
            self.align_ego_to_travel_lane()
        log_pre_control_diagnostic(self, for_follow_lead=for_follow_lead)
        lead_min = self._pass_lead_gap_floor_m(spec)
        lead_max = 40.0
        if spec is not None:
            profile = self._spawn_profile(spec)
            lead_min = max(lead_min, float(profile.get("lead_floor_m", lead_min)))
            lead_max = max(lead_min + 2.0, float(profile.get("lead_cap_m", lead_max)))
        ok, issues, diag = assert_pre_control_sanity(
            self,
            for_follow_lead=for_follow_lead,
            check_spawn_gaps=check_spawn_gaps,
            lead_gap_min_m=lead_min,
            lead_gap_max_m=lead_max,
            raise_on_fail=not is_test_mode(),
        )
        if not ok:
            print("[CARLA] Pre-control sanity issues: " + "; ".join(issues), flush=True)
        return diag

    def ego_lane_center_distance_m(self, ego, phase: str | None = None) -> float:
        """Distance to lane center for the lane ego is actually occupying."""
        from perception.carla_lane_keep import lane_center_distance_m

        if ego is None or self.map is None:
            return 999.0
        loc = ego.get_location()
        tw = self._travel_wp
        pw = self._passing_wp

        try:
            ego_wp = self.map.get_waypoint(loc, project_to_road=True)
        except Exception:
            ego_wp = None

        if phase in ("cruise", "travel") and tw is not None:
            anchor = self._travel_lane_anchor_at_ego(ego) or tw
            return lane_center_distance_m(loc, anchor)

        if phase == "merge_back" and tw is not None:
            anchor = self._travel_lane_anchor_at_ego(ego) or tw
            return lane_center_distance_m(loc, anchor)

        if phase == "lane_change" and pw is not None:
            path_loc = self.get_lane_change_path_center_at_ego(ego)
            if path_loc is not None:
                from types import SimpleNamespace

                ref = SimpleNamespace(
                    transform=SimpleNamespace(location=path_loc),
                )
                return lane_center_distance_m(loc, ref)
        if phase in ("lane_change", "overtake") and pw is not None and ego_wp is not None:
            on_passing = ego_wp.road_id == pw.road_id and ego_wp.lane_id == pw.lane_id
            if on_passing:
                return lane_center_distance_m(loc, ego_wp)
            if tw is not None:
                anchor = self._travel_lane_anchor_at_ego(ego) or tw
                return lane_center_distance_m(loc, anchor)

        anchor = self._ego_lane_anchor_waypoint(ego)
        if anchor is None:
            return 999.0
        return lane_center_distance_m(loc, anchor)

    def get_recovery_travel_waypoint(self, ego):
        """Nearest valid travel-lane point ahead for emergency lane-center recovery."""
        anchor = self._travel_lane_anchor_at_ego(ego) or self._travel_wp
        if anchor is None or self._travel_wp is None:
            return anchor
        tw = self._travel_wp
        return self._wp_on_lane_ahead(
            anchor, max(6.0, route_lookahead_m() * 0.5), tw.lane_id, tw.road_id
        )

    def route_cursor_debug_snapshot(self, ego) -> dict:
        out: dict = {
            "route_cursor_lane_id": None,
            "route_cursor_road_id": None,
            "cursor_dist_from_ego_m": None,
            "travel_lane_id": None,
            "travel_road_id": None,
        }
        if self._travel_wp is not None:
            out["travel_lane_id"] = self._travel_wp.lane_id
            out["travel_road_id"] = self._travel_wp.road_id
        if self._route_cursor is not None:
            out["route_cursor_lane_id"] = self._route_cursor.lane_id
            out["route_cursor_road_id"] = self._route_cursor.road_id
        if ego is not None and self._route_cursor is not None:
            try:
                out["cursor_dist_from_ego_m"] = round(
                    ego.get_location().distance(self._route_cursor.transform.location), 2
                )
            except Exception:
                pass
        return out

    def get_steering_waypoint(self, ego, phase: str, passing_side: str = "left"):
        """Follow travel lane for cruise/wait/merge; passing lane only during pass phases."""
        if self.map is None or ego is None:
            return self._travel_wp
        tw = self._travel_wp
        side = self._passing_side or passing_side
        if phase in ("cruise", "travel"):
            return self.get_travel_steering_waypoint(ego)
        if phase == "merge":
            return self.get_travel_steering_waypoint(ego)
        if phase in ("approach", "prepare_pass", "lane_change") and self._passing_wp is not None:
            return self._lane_change_steer_waypoint(ego, side)
        if phase == "overtake" and self._passing_wp is not None:
            return self.get_passing_lane_steering_waypoint(ego)
        return self.get_travel_steering_waypoint(ego)

    def _snap_convoy_to_travel_lane(self, spec: ScenarioSpec | None = None) -> None:
        """Keep ego/lead/rear on the spawn travel lane (avoid parallel-road projection)."""
        if not self.allows_pre_decision_actor_layout():
            return
        tw = self._travel_wp
        if tw is None or self.map is None:
            return
        if spec is not None and self._uses_axis_spawn(spec):
            for name, role, dist, rear_pass in (
                ("ego", "travel", 0.0, False),
                ("lead", "lead", self._spawn_lead_m, False),
                ("rear", "rear", self._spawn_rear_m, self._rear_on_passing_lane),
            ):
                actor = self.actors.get(name)
                if actor is None:
                    continue
                rear_lane = "passing" if rear_pass else "travel"
                tf = self._layout_transform(name, dist, lane=rear_lane, spec=spec)
                actor.set_transform(tf)
            self.refresh_axis_ego_from_live()
            self._layout_tick_sync()
            return
        for name, role, dist in (
            ("ego", "travel", 0.0),
            ("lead", "lead", self._spawn_lead_m),
            ("rear", "rear", self._spawn_rear_m),
        ):
            actor = self.actors.get(name)
            if actor is None:
                continue
            tf = self._role_transform(role, dist)
            actor.set_transform(tf)
            try:
                wp = self.map.get_waypoint(actor.get_location(), project_to_road=True)
                if wp.road_id != tw.road_id or wp.lane_id != tw.lane_id:
                    actor.set_transform(self._role_transform(role, dist))
            except Exception:
                pass

    def _ensure_spawn_gaps(self, spec: ScenarioSpec | None = None) -> None:
        """Rear must start behind ego on the travel lane with safe longitudinal gap."""
        if not self.allows_pre_decision_actor_layout():
            return
        if not self.ready and (spec is None or not self._uses_axis_spawn(spec)):
            return
        self._layout_tick_sync()
        profile = self._spawn_profile(spec) if spec is not None else {}
        axis = spec is not None and self._uses_axis_spawn(spec)
        min_rear = rear_follow_min_m() + 4.0
        min_lead = max(10.0, safe_follow_m() - 2.0)
        if axis:
            min_lead = max(min_lead, float(profile.get("lead_floor_m", min_lead)))
            rear_cap = float(profile.get("rear_spawn_m", 18.0)) + 5.0
            self._spawn_rear_m = min(max(float(self._spawn_rear_m), min_rear), rear_cap)
        else:
            self._spawn_rear_m = max(self._spawn_rear_m, min_rear)
        self._spawn_lead_m = max(self._spawn_lead_m, min_lead)
        rear = self.actors.get("rear")
        if rear is not None and not axis:
            for _ in range(6):
                gap = self.rear_longitudinal_gap_m()
                if gap >= rear_follow_min_m():
                    break
                extra = rear_follow_min_m() - gap + 5.0
                self._spawn_rear_m += extra
                rear.set_transform(
                    self._role_transform(
                        "rear", self._spawn_rear_m, rear_on_passing_lane=self._rear_on_passing_lane
                    )
                )
        elif rear is not None and axis and self._rear_on_passing_lane:
            self._hold_rear_on_passing_lane(float(profile.get("rear_spawn_m", self._spawn_rear_m)))
            self._layout_tick_sync()
        lead = self.actors.get("lead")
        if lead is not None and self.lead_longitudinal_gap_m() < min_lead:
            self._spawn_lead_m = max(self._spawn_lead_m, min_lead)
            if spec is not None and self._uses_axis_spawn(spec):
                self._place_actor_longitudinal("lead", self._spawn_lead_m, lane="travel", spec=spec)
            else:
                lead.set_transform(self._role_transform("lead", self._spawn_lead_m))

    def tick_npcs_kinematic(self, spec: ScenarioSpec, dt: float) -> None:
        if not self.ready:
            return
        profile = self._spawn_profile(spec)
        lead_speed = float(profile.get("lead_speed_mps", spec.lead.speed_mps))
        self._kinematic_lead_speed_mps = lead_speed
        self._step_lead_npc(spec, dt, speed_mps=lead_speed)
        self._step_rear_npc(spec, dt)
        if self.actors.get("oncoming") is not None:
            self._step_actor_forward("oncoming", spec.oncoming.speed_mps, dt)

    def _step_actor_along_travel_axis(self, name: str, speed_mps: float, dt: float) -> None:
        """Small forward step along cached travel axis (safe for axis-spawn demos)."""
        from perception.carla_axis_spawn import normalize3
        from perception.carla_geometry import actor_location_tuple

        actor = self.actors.get(name)
        basis = self._ego_travel_basis()
        if actor is None or basis is None:
            return
        _ego_xyz, travel, _lat = basis
        travel = normalize3(travel)
        loc = actor_location_tuple(actor)
        if loc is None:
            return
        step_m = max(0.05, float(speed_mps) * float(dt))
        try:
            tf = actor.get_transform()
            carla = self.carla
            new_loc = carla.Location(
                loc[0] + travel[0] * step_m,
                loc[1] + travel[1] * step_m,
                loc[2] + travel[2] * step_m,
            )
            actor.set_transform(carla.Transform(new_loc, tf.rotation))
        except Exception:
            pass

    def _step_lead_npc(self, spec: ScenarioSpec, dt: float, *, speed_mps: float | None = None) -> None:
        if self.lead_longitudinal_gap_m() < 5.0:
            return
        spd = float(speed_mps if speed_mps is not None else spec.lead.speed_mps)
        # Always follow spawn travel-lane centerline — never ego-forward axis (yaw drift veers lead off road).
        self._step_npc_on_travel_lane("lead", spd, dt)

    def _step_rear_npc(self, spec: ScenarioSpec, dt: float) -> None:
        """Rear follows ego — never rams a slow/stopped ego (common on first physics step)."""
        if getattr(self, "_rear_on_passing_lane", False) and self._uses_axis_spawn(spec):
            follow_gap = max(float(self._spawn_rear_m), rear_follow_min_m())
            self._hold_rear_on_passing_lane(follow_gap)
            return
        gap = self.rear_longitudinal_gap_m()
        min_gap = rear_follow_min_m()
        crit = 7.0
        ego_sp = self._ego_speed_mps()
        desired = spec.rear.speed_mps
        if gap <= crit:
            desired = max(0.0, ego_sp - 2.5)
        elif gap <= min_gap:
            desired = min(desired, ego_sp)
        else:
            desired = min(desired, ego_sp + max(0.0, (gap - min_gap) * 1.2))
        if gap < min_gap and desired > ego_sp:
            desired = ego_sp
        # During ego pass / lane change, hold rear back so it does not close to <8m.
        if gap < 28.0 and ego_sp > 2.0:
            desired = min(desired, max(0.0, ego_sp - 1.5))
        self._kinematic_rear_speed_mps = float(desired)
        if getattr(self, "_rear_on_passing_lane", False) and self._uses_axis_spawn(spec):
            self._hold_rear_on_passing_lane(max(float(self._spawn_rear_m), rear_follow_min_m()))
        else:
            self._step_npc_on_travel_lane("rear", desired, dt)

    def _step_npc_on_travel_lane(self, name: str, speed_mps: float, dt: float) -> None:
        """Advance NPC along spawn travel lane/road — never drift to parallel segments."""
        actor = self.actors.get(name)
        tw = self._travel_wp
        if actor is None or self.map is None or tw is None:
            return
        step_m = max(0.0, speed_mps * dt)
        if step_m < 1e-6:
            return
        try:
            s = self.project_actor_along_travel_axis(actor)
            if s is None:
                if name == "lead":
                    s = float(self._spawn_lead_m)
                elif name == "rear":
                    s = -float(self._spawn_rear_m)
                else:
                    s = 0.0
            nxt = self._wp_on_lane_ahead(tw, float(s) + step_m, tw.lane_id, tw.road_id)
            if nxt is None:
                return
            loc = nxt.transform.location
            loc.z += 0.3
            actor.set_transform(self.carla.Transform(loc, nxt.transform.rotation))
        except Exception:
            pass

    def _hold_rear_on_passing_lane(self, gap_m: float) -> None:
        """Keep rear on the passing lane at a fixed travel-axis gap behind ego."""
        rear = self.actors.get("rear")
        tw = self._travel_wp
        pw = self._passing_wp
        ego = self.actors.get("ego")
        if rear is None or tw is None or pw is None or ego is None or self.carla is None:
            return
        anchor = self._travel_lane_anchor_at_ego(ego) or tw
        rear_travel = self._wp_behind(anchor, float(gap_m))
        passing = self._adjacent_passing_lane_wp(rear_travel, self._passing_side or "left") or pw
        loc = passing.transform.location
        loc.z += 0.3
        rear.set_transform(
            self.carla.Transform(loc, passing.transform.rotation)
        )

    def _same_carriageway(self, wp_a, wp_b) -> bool:
        if wp_a is None or wp_b is None:
            return False
        try:
            if int(wp_a.road_id) != int(wp_b.road_id):
                return False
            return int(wp_a.lane_id) * int(wp_b.lane_id) > 0
        except Exception:
            return False

    def _actor_on_travel_corridor(
        self,
        actor,
        tw,
        spec: ScenarioSpec | None = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Whether an actor is on the spawn travel lane for convoy checks.

        CARLA map.get_waypoint(project_to_road=True) can assign a parallel road_id
        while the actor remains on the curated lane index (common after axis spawn).
        For axis-spawn demos we accept matching lane_id + small lateral error to travel.
        """
        if actor is None or tw is None or self.map is None:
            return True, None
        try:
            wp = self.map.get_waypoint(actor.get_location(), project_to_road=True)
        except Exception as e:
            return False, f"waypoint: {e}"
        try:
            lane_id = int(wp.lane_id)
            travel_lane = int(tw.lane_id)
        except Exception:
            return False, "lane_id_unavailable"
        if lane_id != travel_lane:
            return False, f"off_travel_lane lane={lane_id} (expected {travel_lane})"
        if lane_id * travel_lane <= 0:
            return False, f"opposite_lane lane={lane_id}"
        if self._same_carriageway(wp, tw):
            return True, None
        axis_spawn = (
            (spec is not None and self._uses_axis_spawn(spec))
            or bool(getattr(self, "_axis_spawn_active", False))
        )
        if axis_spawn:
            from perception.carla_lane_keep import lane_center_distance_m

            ego = self.actors.get("ego")
            lead = self.actors.get("lead")
            # Axis-placed lead: same lane index + target gap along travel axis is enough.
            # get_waypoint may report a parallel road_id; wp.next cannot reach 32m here.
            if lead is not None and actor is lead and ego is not None:
                along = self.longitudinal_gap(ego, actor)
                if along is not None and along >= 8.0:
                    return True, None
            if ego is not None and actor is ego:
                anchor = self._travel_lane_anchor_at_ego(ego) or tw
                if lane_center_distance_m(actor.get_location(), anchor) < 4.0:
                    return True, None
        return (
            False,
            f"wrong_carriageway lane={lane_id} road={wp.road_id} "
            f"(travel lane={travel_lane} road={tw.road_id})",
        )

    def assert_convoy_on_travel_corridor(self, spec: ScenarioSpec | None = None) -> None:
        """Fail fast if ego/lead are not on the curated travel carriageway."""
        from autopass.config import AutopassConfigurationError, is_test_mode

        tw = self._travel_wp
        if tw is None or self.map is None:
            return
        if spec is None:
            spec = getattr(self, "_bootstrap_spec", None)

        def _collect_issues() -> list[str]:
            issues: list[str] = []
            for name in ("ego", "lead"):
                actor = self.actors.get(name)
                if actor is None:
                    continue
                ok, detail = self._actor_on_travel_corridor(actor, tw, spec)
                if not ok and detail:
                    issues.append(f"{name}_{detail}")
            return issues

        issues = _collect_issues()
        if issues:
            ego = self.actors.get("ego")
            lead = self.actors.get("lead")
            spec_hint = spec
            if spec_hint is None and self._scenario_id and "clear_safe_pass_perception" in self._scenario_id:
                from visual_world import curated_demo_scenarios

                for s in curated_demo_scenarios():
                    if s.scenario_id == self._scenario_id:
                        spec_hint = s
                        break
            if ego is not None:
                ego.set_transform(self._role_transform("travel", 0.0))
            if lead is not None:
                if spec_hint is not None and self._uses_axis_spawn(spec_hint):
                    self.refresh_axis_ego_from_live()
                    tf = self._axis_actor_transform(float(self._spawn_lead_m), lane="travel")
                    if tf is not None:
                        lead.set_transform(tf)
                    else:
                        self._place_actor_longitudinal(
                            "lead", float(self._spawn_lead_m), lane="travel", spec=spec_hint
                        )
                else:
                    lead.set_transform(self._corridor_actor_transform(float(self._spawn_lead_m), lane="travel"))
            self._layout_tick_sync()
            issues = _collect_issues()
        if issues and not is_test_mode():
            raise AutopassConfigurationError(
                "CARLA convoy spawn incoherent — ego/lead must share travel lane:\n  - "
                + "\n  - ".join(issues)
            )

    def advance_perception_burst_frame(self, spec: ScenarioSpec, dt: float) -> None:
        """Advance simulator during capture burst so depth-derived speeds are measurable."""
        if not self.ready:
            return
        if getattr(self, "_closed_loop_actuation_begun", False):
            ego = self.actors.get("ego")
            self._apply_last_vehicle_control(ego)
            self.tick_npcs_kinematic(spec, dt)
            self.tick()
            return
        profile = self._spawn_profile(spec)
        lead_speed = float(profile.get("lead_speed_mps", spec.lead.speed_mps))
        # Ego stays put; axis-spawn scenarios restore lead after burst — do not waypoint-step lead
        # (wp.next can jump backward toward ego and defeat restore).
        if not self._uses_axis_spawn(spec):
            self._step_npc_on_travel_lane("lead", lead_speed, dt)
        if self.actors.get("rear") is not None:
            if getattr(self, "_rear_on_passing_lane", False) and self._uses_axis_spawn(spec):
                self._hold_rear_on_passing_lane(float(self._spawn_rear_m))
            else:
                self._step_npc_on_travel_lane("rear", min(spec.rear.speed_mps, lead_speed + 1.0), dt)
        self.tick()

    @staticmethod
    def _spawn_profile(spec: ScenarioSpec) -> Dict[str, float | bool]:
        """CARLA layout knobs keyed by scenario_id (ScenarioSpec distances are orchestration only)."""
        sid = spec.scenario_id
        if "clear_safe_pass_perception" in sid:
            return {
                "axis_spawn": True,
                "lead_cap_m": 38.0,
                "lead_floor_m": 32.0,
                "lead_gap_m": 32.0,
                "rear_passing_lane": True,
                "rear_spawn_m": 18.0,
                "lead_speed_mps": float(spec.lead.speed_mps),
            }
        return {
            "axis_spawn": False,
            "lead_cap_m": 22.0,
            "lead_floor_m": 12.0,
            "lead_gap_m": 0.0,
            "rear_passing_lane": False,
            "rear_spawn_m": 0.0,
            "lead_speed_mps": float(spec.lead.speed_mps),
        }

    def _uses_axis_spawn(self, spec: ScenarioSpec) -> bool:
        return bool(self._spawn_profile(spec).get("axis_spawn"))

    def _uses_axis_longitudinal_layout(self, spec: ScenarioSpec | None = None) -> bool:
        """True when lead/rear should be placed via world-axis offset, not waypoint.next."""
        if spec is not None and self._uses_axis_spawn(spec):
            return True
        if getattr(self, "_axis_spawn_active", False):
            return True
        return getattr(self, "_axis_ego_xyz", None) is not None and getattr(
            self, "_axis_travel_dir", None
        ) is not None

    def _forward_unit_from_rotation(self, rotation) -> Tuple[float, float, float]:
        """Unreal/CARLA yaw-pitch forward vector (world space)."""
        from perception.carla_axis_spawn import normalize3

        yaw = math.radians(float(rotation.yaw))
        pitch = math.radians(float(rotation.pitch))
        x = math.cos(pitch) * math.cos(yaw)
        y = math.cos(pitch) * math.sin(yaw)
        z = math.sin(pitch)
        return normalize3((x, y, z))

    def _cache_axis_basis_from_ego_transform(self, ego_tf) -> None:
        """Cache spawn-time ego pose — get_location() is often (0,0,0) until the next tick."""
        from perception.carla_axis_spawn import normalize3

        loc = ego_tf.location
        self._axis_ego_xyz = (float(loc.x), float(loc.y), float(loc.z))
        travel = self._forward_unit_from_rotation(ego_tf.rotation)
        if self._travel_wp is not None:
            try:
                tw_fwd = self._travel_wp.transform.get_forward_vector()
                travel = normalize3((float(tw_fwd.x), float(tw_fwd.y), float(tw_fwd.z)))
            except Exception:
                pass
        self._axis_travel_dir = travel
        lateral = (0.0, 0.0, 0.0)
        if self.map is not None and self._passing_wp is not None:
            try:
                ego_wp = self.map.get_waypoint(loc, project_to_road=True)
                passing = self._adjacent_passing_lane_wp(ego_wp, self._passing_side or "left")
                if passing is not None:
                    tloc = ego_wp.transform.location
                    ploc = passing.transform.location
                    lateral = normalize3(
                        (float(ploc.x - tloc.x), float(ploc.y - tloc.y), float(ploc.z - tloc.z))
                    )
            except Exception:
                pass
        self._axis_lateral_dir = lateral

    def _refresh_axis_basis_from_actors(self) -> None:
        """After a tick, actor locations are trustworthy — refresh cached basis."""
        ego = self.actors.get("ego")
        if ego is None:
            return
        try:
            loc = ego.get_location()
            if abs(loc.x) + abs(loc.y) > 1.0:
                if getattr(self, "_closed_loop_actuation_begun", False):
                    self._axis_ego_xyz = (float(loc.x), float(loc.y), float(loc.z))
                    return
                self._cache_axis_basis_from_ego_transform(ego.get_transform())
        except Exception:
            pass

    def refresh_axis_ego_from_live(self) -> None:
        """Re-cache travel basis from current ego pose (before longitudinal placement)."""
        self._refresh_axis_basis_from_actors()

    def restore_lead_spawn_longitudinal_gap(self, spec: ScenarioSpec) -> None:
        """Re-place lead at spawn_lead_m from live ego (after perception burst stepping)."""
        self._restore_lead_called_this_step = True
        if not self.ready or not self._uses_axis_spawn(spec):
            return
        if not self.allows_pre_decision_actor_layout():
            return
        self.refresh_axis_ego_from_live()
        self._place_actor_longitudinal("lead", float(self._spawn_lead_m), lane="travel", spec=spec)
        if self.is_synchronous_mode():
            self.tick()

    def _ego_travel_basis(self) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        World-space basis: (ego_xyz, travel_unit, lateral_unit toward passing lane).
        Uses cached spawn transform when CARLA has not synced actor locations yet.
        """
        axis_ego = getattr(self, "_axis_ego_xyz", None)
        axis_travel = getattr(self, "_axis_travel_dir", None)
        if axis_ego is not None and axis_travel is not None:
            lat = getattr(self, "_axis_lateral_dir", None) or (0.0, 0.0, 0.0)
            return axis_ego, axis_travel, lat
        ego = self.actors.get("ego")
        if ego is None:
            return None
        self._cache_axis_basis_from_ego_transform(ego.get_transform())
        return self._ego_travel_basis()

    def _passing_lane_lateral_offset_m(self) -> float:
        """Meters from travel-lane center to passing-lane center at spawn corridor."""
        tw = self._travel_wp
        pw = self._passing_wp
        if tw is None or pw is None:
            return 3.5
        try:
            tloc = tw.transform.location
            ploc = pw.transform.location
            return float(
                math.sqrt(
                    (ploc.x - tloc.x) ** 2 + (ploc.y - tloc.y) ** 2 + (ploc.z - tloc.z) ** 2
                )
            )
        except Exception:
            return 3.5

    def _role_transform_for_actor(self, name: str, ahead_m: float, *, lane: str = "travel") -> object:
        """Waypoint-based spawn transform on the curated corridor (never world-axis offset)."""
        rear_pass = bool(getattr(self, "_rear_on_passing_lane", False))
        if name == "lead":
            return self._role_transform("lead", float(ahead_m))
        if name == "rear":
            return self._role_transform(
                "rear",
                abs(float(ahead_m)),
                rear_on_passing_lane=rear_pass or lane == "passing",
            )
        if name == "ego":
            return self._role_transform("travel", 0.0)
        return self._role_transform("travel", 0.0)

    def _axis_actor_transform(
        self,
        ahead_m: float,
        *,
        lane: str = "travel",
    ):
        """Place actor at a fixed offset from live ego along the travel axis (map-independent)."""
        from perception.carla_axis_spawn import world_location_from_ego_offset

        carla = self.carla
        basis = self._ego_travel_basis()
        if basis is None or carla is None:
            return None
        ego_xyz, travel, lateral = basis
        lat_m = 0.0
        if lane == "passing" and self._passing_wp is not None:
            lat_m = self._passing_lane_lateral_offset_m()
            if (self._passing_side or "left") == "right":
                lat_m = -lat_m
        xyz = world_location_from_ego_offset(
            ego_xyz,
            travel,
            lateral,
            longitudinal_m=float(ahead_m),
            lateral_m=lat_m,
        )
        rot = None
        if lane == "travel" and self._travel_wp is not None:
            rot = self._travel_wp.transform.rotation
        elif self.map is not None and lane != "travel":
            try:
                wp = self.map.get_waypoint(
                    carla.Location(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                    project_to_road=True,
                )
                rot = wp.transform.rotation
            except Exception:
                rot = None
        if rot is None:
            ego = self.actors.get("ego")
            if ego is not None:
                try:
                    rot = ego.get_transform().rotation
                except Exception:
                    rot = None
        if rot is None and self._travel_wp is not None:
            rot = self._travel_wp.transform.rotation
        if rot is None:
            return None
        return carla.Transform(
            carla.Location(float(xyz[0]), float(xyz[1]), float(xyz[2]) + 0.3),
            carla.Rotation(float(rot.pitch), float(rot.yaw), float(rot.roll)),
        )

    def _transform_from_ego_longitudinal(
        self,
        ahead_m: float,
        *,
        lane: str = "travel",
        spec: ScenarioSpec | None = None,
    ):
        """Layout placement: axis offset for pass demos, else corridor waypoints."""
        if self._uses_axis_longitudinal_layout(spec):
            tf = self._axis_actor_transform(float(ahead_m), lane=lane)
            if tf is not None:
                return tf
        return self._corridor_actor_transform(float(ahead_m), lane=lane)

    def _axis_spawn_gap_metrics(self) -> Dict[str, float]:
        from perception.carla_axis_spawn import euclidean_distance_m, projected_distance_m
        from perception.carla_geometry import actor_location_tuple

        basis = self._ego_travel_basis()
        out: Dict[str, float] = {}
        if basis is None:
            return out
        cached_ego, travel, _ = basis
        ego_xyz = actor_location_tuple(self.actors.get("ego"))
        if ego_xyz is None or (abs(ego_xyz[0]) + abs(ego_xyz[1]) < 1.0):
            ego_xyz = cached_ego
        for name, req_key in (("lead", "lead"), ("rear", "rear")):
            other = actor_location_tuple(self.actors.get(name))
            if other is None:
                continue
            proj = projected_distance_m(ego_xyz, other, travel)
            if name == "rear":
                proj = max(0.0, -float(proj))
            else:
                proj = max(0.0, float(proj))
            out[f"actual_projected_{req_key}_gap_m"] = round(proj, 2)
            out[f"actual_euclidean_{req_key}_dist_m"] = round(float(euclidean_distance_m(ego_xyz, other)), 2)
        return out

    def _log_axis_spawn_layout(self, spec: ScenarioSpec) -> None:
        from perception.carla_geometry import actor_location_tuple

        basis = self._ego_travel_basis()
        ego_xyz = actor_location_tuple(self.actors.get("ego"))
        if basis is None or ego_xyz is None:
            print("[CARLA] Axis spawn layout: basis unavailable", flush=True)
            return
        _, travel, lateral = basis
        metrics = self._axis_spawn_gap_metrics()
        print(
            "[CARLA] Axis spawn layout:\n"
            f"  ego_location=({ego_xyz[0]:.1f}, {ego_xyz[1]:.1f}, {ego_xyz[2]:.1f})\n"
            f"  travel_direction=({travel[0]:.4f}, {travel[1]:.4f}, {travel[2]:.4f})\n"
            f"  lateral_direction=({lateral[0]:.4f}, {lateral[1]:.4f}, {lateral[2]:.4f})\n"
            f"  requested_lead_gap_m={self._spawn_lead_m:.1f}\n"
            f"  actual_projected_lead_gap_m={metrics.get('actual_projected_lead_gap_m', -1):.1f}\n"
            f"  actual_euclidean_lead_dist_m={metrics.get('actual_euclidean_lead_dist_m', -1):.1f}\n"
            f"  requested_rear_gap_m={self._spawn_rear_m:.1f}\n"
            f"  actual_projected_rear_gap_m={metrics.get('actual_projected_rear_gap_m', -1):.1f}\n"
            f"  actual_euclidean_rear_dist_m={metrics.get('actual_euclidean_rear_dist_m', -1):.1f}",
            flush=True,
        )
        lead_proj = metrics.get("actual_projected_lead_gap_m", 0.0)
        if lead_proj < 26.0:
            print(
                f"[CARLA] WARNING: projected lead gap {lead_proj:.1f}m < 26m at spawn_index="
                f"{getattr(self._corridor_report, 'spawn_index', '?')}",
                flush=True,
            )

    def _layout_transform(
        self,
        name: str,
        ahead_m: float,
        *,
        lane: str = "travel",
        spec: ScenarioSpec | None = None,
    ):
        """Spawn/layout transform: axis offset for pass demos, corridor otherwise."""
        use_axis = self._uses_axis_longitudinal_layout(spec)
        if name == "lead":
            if use_axis:
                tf = self._axis_actor_transform(float(ahead_m), lane="travel")
                if tf is not None:
                    return tf
            return self._corridor_actor_transform(float(ahead_m), lane="travel")
        if name == "rear":
            rear_lane = "passing" if lane == "passing" or self._rear_on_passing_lane else "travel"
            dist = -abs(float(ahead_m))
            if use_axis:
                tf = self._axis_actor_transform(dist, lane=rear_lane)
                if tf is not None:
                    return tf
            return self._corridor_actor_transform(dist, lane=rear_lane)
        if name in ("ego", "travel"):
            return self._role_transform("travel", 0.0)
        if use_axis:
            tf = self._axis_actor_transform(float(ahead_m), lane=lane)
            if tf is not None:
                return tf
        return self._corridor_actor_transform(float(ahead_m), lane=lane)

    def _place_actor_longitudinal(
        self,
        name: str,
        ahead_m: float,
        *,
        lane: str = "travel",
        layout_snap: bool = True,
        spec: ScenarioSpec | None = None,
    ) -> bool:
        actor = self.actors.get(name)
        if actor is None:
            return False
        tf = self._layout_transform(name, ahead_m, lane=lane, spec=spec)
        if tf is None:
            return False
        if not layout_snap:
            actor.set_transform(tf)
            return True
        from perception.actor_continuity import apply_layout_transform

        if name == "rear":
            self._restore_rear_called_this_step = True
        return apply_layout_transform(
            self,
            actor,
            tf,
            reason=f"place_{name}_longitudinal_{ahead_m:.1f}m_{lane}",
        )

    def _step_actor_forward(self, name: str, speed_mps: float, dt: float) -> None:
        actor = self.actors.get(name)
        if actor is None or self.map is None:
            return
        step_m = max(0.05, speed_mps * dt)
        try:
            wp = self.map.get_waypoint(actor.get_location(), project_to_road=True)
            nxt = wp.next(step_m)
            if not nxt:
                return
            target = nxt[0]
            loc = target.transform.location
            loc.z += 0.3
            actor.set_transform(self.carla.Transform(loc, target.transform.rotation))
        except Exception:
            pass

    def sync_npc_poses(self, spec: ScenarioSpec, world: WorldState) -> None:
        if not self.ready or self._ego_physics:
            return
        if self._uses_axis_spawn(spec) or not self.allows_pre_decision_actor_layout():
            return
        scale = self._spawn_lead_m / max(1.0, spec.lead.distance_m)
        gap_m = max(3.0, world.lead_x_m - world.ego_x_m) if not world.passed else world.lead_x_m - world.ego_x_m
        lead_d = max(10.0, gap_m * scale)
        rear_d = max(rear_follow_min_m() + 2.0, (world.ego_x_m - world.rear_x_m) * scale)
        on_d = max(18.0, (world.oncoming_x_m - world.ego_x_m) * scale)
        if self.actors.get("lead"):
            self.actors["lead"].set_transform(self._role_transform("lead", min(lead_d, self._spawn_lead_m + 5)))
        if self.actors.get("rear"):
            self.actors["rear"].set_transform(
                self._role_transform("rear", max(self._spawn_rear_m, min(rear_d, self._spawn_rear_m + 8)))
            )
        if self.actors.get("oncoming"):
            self.actors["oncoming"].set_transform(
                self._role_transform("oncoming", min(on_d, self._spawn_on_m + 10))
            )
        if self.actors.get("ego") and not self._ego_physics:
            lane = "passing" if world.ego_lane == 1 else "travel"
            self.actors["ego"].set_transform(self._role_transform(lane, 0.0))

    def infer_ego_lane_index(self) -> int:
        ego = self.actors.get("ego")
        tw = self._travel_wp
        pw = self._passing_wp
        if ego is None or pw is None or tw is None or self.map is None:
            return 0
        try:
            ego_wp = self.map.get_waypoint(ego.get_location(), project_to_road=True)
            if ego_wp.road_id == pw.road_id and ego_wp.lane_id == pw.lane_id:
                return 1
            if ego_wp.road_id == tw.road_id and ego_wp.lane_id == tw.lane_id:
                return 0
        except Exception:
            pass
        return 0

    def check_actor_proximity(self, threshold_m: float = 4.5) -> Tuple[bool, str]:
        from perception.carla_geometry import actor_location_tuple, euclidean_m

        ego = self.actors.get("ego")
        if ego is None or self.map is None:
            return False, ""
        ego_xyz = actor_location_tuple(ego)
        if ego_xyz is None:
            return False, ""
        ego_loc = type("L", (), {"x": ego_xyz[0], "y": ego_xyz[1], "z": ego_xyz[2]})()
        try:
            ego_wp = self.map.get_waypoint(ego.get_location(), project_to_road=True)
        except Exception:
            ego_wp = None
        for name, actor in self.actors.items():
            if name == "ego" or actor is None:
                continue
            other_xyz = actor_location_tuple(actor)
            if other_xyz is None:
                continue
            other_loc = type("L", (), {"x": other_xyz[0], "y": other_xyz[1], "z": other_xyz[2]})()
            d = euclidean_m(ego_loc, other_loc)
            if d >= threshold_m:
                continue
            if ego_wp is not None and name == "oncoming":
                try:
                    actor_wp = self.map.get_waypoint(
                        actor.get_location(), project_to_road=True
                    )
                    if ego_wp.lane_id * actor_wp.lane_id < 0 and d > 2.8:
                        continue
                except Exception:
                    pass
            if name == "rear" and self.rear_longitudinal_gap_m() >= rear_follow_min_m() - 1.0:
                continue
            if name == "lead" and self.lead_longitudinal_gap_m() >= 5.0:
                continue
            return True, f"{name}_within_{d:.1f}m"
        return False, ""

    def _zero_vehicle_control(self, actor) -> None:
        if actor is None or self.carla is None:
            return
        ctrl = self.carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0, hand_brake=False)
        actor.apply_control(ctrl)

    def reset_episode_state(self, *, settle: bool = True) -> None:
        """Reset per-episode counters, steering reference, and collision history."""
        self._episode_step = 0
        from perception.actor_continuity import reset_continuity_state

        reset_continuity_state(self)
        self._collision_events = []
        self._route_cursor = self._travel_wp
        self._last_steer = 0.0
        self._lane_departure_stopped = False
        if not self.ready or self.world is None:
            return
        self._ego_physics = False
        for name in ("ego", "lead", "rear", "oncoming"):
            actor = self.actors.get(name)
            if actor is not None:
                try:
                    actor.set_simulate_physics(False)
                    self._zero_vehicle_control(actor)
                except Exception:
                    pass
        if settle:
            for _ in range(self._spawn_settle_ticks):
                self.tick()
            try:
                from perception.lead_gap_diagnostics import log_lead_gap_checkpoint

                log_lead_gap_checkpoint(self, "B_after_post_spawn_settle")
            except Exception:
                pass

    def end_episode(self) -> None:
        """Park actors kinematically before the next benchmark row respawns."""
        self.reset_episode_state(settle=False)
        if self.world is not None:
            self.tick()

    def record_collision_event(self, source: str, detail: str, step: int) -> None:
        self._collision_events.append({"source": source, "detail": detail, "step": step})

    def actor_distance_snapshot(self) -> Dict[str, float]:
        gaps = self.measure_actor_gaps_3d()
        lead_signed = self.signed_gap_from_ego("lead")
        rear_signed = self.signed_gap_from_ego("rear")
        on_signed = self.signed_gap_from_ego("oncoming")
        return {
            "front_m": gaps.get("front", 999.0),
            "rear_m": gaps.get("rear", 999.0),
            "oncoming_m": gaps.get("oncoming", 999.0),
            "lead_long_m": float(lead_signed) if lead_signed is not None else 999.0,
            "rear_long_m": float(-rear_signed) if rear_signed is not None else 999.0,
            "oncoming_long_m": float(on_signed) if on_signed is not None else 999.0,
        }

    @staticmethod
    def _xyz_components(loc) -> Tuple[float, float, float]:
        """CARLA Location or (x, y, z) tuple from _travel_axis / axis cache."""
        if loc is None:
            return 0.0, 0.0, 0.0
        if isinstance(loc, (tuple, list)) and len(loc) >= 3:
            return float(loc[0]), float(loc[1]), float(loc[2])
        return float(loc.x), float(loc.y), float(loc.z)

    def geometry_debug_snapshot(self) -> Dict[str, object]:
        axis = self._travel_axis()
        origin = None
        direction = None
        if axis is not None:
            org, fwd = axis
            ox, oy, oz = self._xyz_components(org)
            origin = {"x": round(ox, 2), "y": round(oy, 2), "z": round(oz, 2)}
            direction = {"x": round(fwd[0], 4), "y": round(fwd[1], 4), "z": round(fwd[2], 4)}
        projected_s: Dict[str, Optional[float]] = {}
        lanes: Dict[str, Optional[Dict[str, int]]] = {}
        for name in ("ego", "lead", "rear", "oncoming"):
            s = self.project_actor_along_travel_axis(name)
            projected_s[name] = None if s is None else round(float(s), 3)
            lanes[name] = self.lane_identity(name)
        return {
            "travel_origin": origin,
            "travel_direction": direction,
            "projected_s": projected_s,
            "signed_gaps_from_ego": {
                "lead": None if self.signed_gap_from_ego("lead") is None else round(float(self.signed_gap_from_ego("lead")), 3),
                "rear": None if self.signed_gap_from_ego("rear") is None else round(float(self.signed_gap_from_ego("rear")), 3),
                "oncoming": None if self.signed_gap_from_ego("oncoming") is None else round(float(self.signed_gap_from_ego("oncoming")), 3),
            },
            "euclidean_distances": self.measure_actor_gaps_3d(),
            "lane_identity": lanes,
        }

    def actor_transform_snapshot(self) -> Dict[str, Dict[str, float]]:
        from perception.carla_geometry import actor_debug_record, actor_location_tuple

        ego_xyz = actor_location_tuple(self.actors.get("ego"))
        out: Dict[str, Dict[str, float]] = {}
        for name in ("ego", "lead", "rear", "oncoming"):
            rec = actor_debug_record(self, name, ego_xyz if name != "ego" else None)
            if rec.get("transform"):
                out[name] = rec["transform"]
        return out

    def actor_validation_snapshot(self) -> Dict[str, object]:
        from perception.carla_geometry import actor_debug_record, actor_location_tuple

        ego_xyz = actor_location_tuple(self.actors.get("ego"))
        rows = [actor_debug_record(self, name, ego_xyz if name != "ego" else None) for name in ("ego", "lead", "rear", "oncoming")]
        return {"actors": rows, "ego_xyz": ego_xyz}

    def _destroy_sensor_actors(self) -> None:
        for s in self.sensors.values():
            if s is None:
                continue
            try:
                s.stop()
                s.destroy()
            except Exception:
                pass
        self.sensors.clear()
        self._sensor_listeners.clear()

    def _destroy_actors_and_sensors(self) -> None:
        self._destroy_sensor_actors()
        for a in self.actors.values():
            if a is None:
                continue
            try:
                a.destroy()
            except Exception:
                pass
        self.sensors.clear()
        self.actors.clear()
        self._sensor_listeners.clear()
        self._sensor_callback_errors.clear()
        self._ego_physics = False
        self._episode_step = 0
        self._collision_events = []
        self._sensor_frame_counts = {"rgb": 0, "depth": 0, "seg": 0, "overhead": 0}
        self._sensor_last_frame = {"rgb": -1, "depth": -1, "seg": -1, "overhead": -1}

    def respawn_episode(
        self,
        spec: ScenarioSpec,
        world: WorldState,
        *,
        physical_key: str,
        same_physical: bool = False,
    ) -> bool:
        """Same map — destroy actors and spawn a fresh episode (new policy/urgency row)."""
        if not self.ready or self.world is None or self.map is None:
            self.last_error = "session not ready for respawn"
            return False
        try:
            self.end_episode()
            self._destroy_actors_and_sensors()
            repick = not same_physical or self._travel_wp is None
            self._spawn_scenario(spec, world, repick_spawn=repick)
            self._attach_sensors()
            self._set_spectator_behind_ego()
            self._scenario_id = spec.scenario_id
            self._physical_key = physical_key
            self._spawn_ego_s = None
            self._spawn_logical_x = None
            self.init_logical_anchor(world.ego_x_m)
            self._ensure_spawn_gaps(spec)
            self._bootstrap_spec = spec
            self._finalize_spawn_layout(spec)
            self.reset_episode_state(settle=True)
            try:
                self.run_pre_control_sanity(for_follow_lead=True, align_ego=True, spec=spec)
            except Exception as e:
                self.last_error = str(e)
                print(f"[CARLA] Respawn failed pre-control: {e}", flush=True)
                return False
            if not self.wait_for_sensor_frames(max_ticks=50, timeout_s=10.0):
                self.last_error = "sensor warmup failed after respawn"
                return False
            env_kind = os.environ.get("AUTOPASS_ENVIRONMENT", "highway")
            try:
                self.assert_curated_corridor_or_raise(env_kind)
            except Exception as e:
                self.last_error = str(e)
                print(f"[CARLA] Respawn failed: {e}")
                return False
            if same_physical:
                self.last_bootstrap_action = "reuse_map_same_physical"
                print(
                    f"[CARLA] Reset actors on {self._map_name} "
                    f"(same physical layout, scenario={spec.scenario_id})"
                )
            else:
                self.last_bootstrap_action = "reuse_map_respawn"
                print(
                    f"[CARLA] Respawned actors on {self._map_name} "
                    f"(physical={physical_key}, scenario={spec.scenario_id})"
                )
            return True
        except Exception as e:
            self.last_error = str(e)
            print(f"[CARLA] Respawn failed: {e}")
            return False

    def bootstrap(self, spec: ScenarioSpec, world: WorldState, map_name: str) -> bool:
        try:
            import carla as carla_mod
        except ImportError as e:
            self.last_error = f"import carla failed: {e}"
            print(f"[CARLA] {self.last_error}")
            return False

        self.carla = carla_mod
        host = os.environ.get("CARLA_HOST", "127.0.0.1")
        port = int(os.environ.get("CARLA_PORT", "2000"))
        short = _normalize_map_name(map_name)

        try:
            self.client = carla_mod.Client(host, port)
            self.client.set_timeout(15.0)
            available = self.client.get_available_maps()
            if self._map_name == short and self.world is not None:
                print(f"[CARLA] Reusing map {short} (no reload)")
                self.last_bootstrap_action = "reuse_map"
            elif not any(short in m for m in available):
                print(f"[CARLA] Map {short} not in {available[:5]}... using current world.")
                self.world = self.client.get_world()
                self._map_name = short
                self.map_load_count += 1
                self.last_bootstrap_action = "load_map"
            else:
                print(f"[CARLA] Loading map {short} ...")
                self.world = self.client.load_world(short)
                self._map_name = short
                self.map_load_count += 1
                self.last_bootstrap_action = "load_map"
            self.map = self.world.get_map()

            self._apply_world_sync_settings()
            self._spawn_scenario(spec, world, repick_spawn=True)
            self._bootstrap_spec = spec
            self._extend_lead_to_target_gap(float(self._spawn_lead_m), spec=spec)
            max_repick = int(os.environ.get("AUTOPASS_CARLA_MAX_CORRIDOR_REPICK", "5"))
            if (
                self._uses_axis_spawn(spec)
                and self.lead_longitudinal_gap_m() < 26.0
                and max_repick > 0
            ):
                if self._try_repick_corridor_for_pass(spec, world):
                    self._extend_lead_to_target_gap(float(self._spawn_lead_m), spec=spec)
                    print(
                        f"[CARLA] Repicked corridor for axis pass demo; lead_gap="
                        f"{self.lead_longitudinal_gap_m():.1f}m",
                        flush=True,
                    )
                elif self.lead_longitudinal_gap_m() < 26.0:
                    print(
                        "[CARLA] WARNING: no corridor repick achieved >=26m lead gap; "
                        "demo_07 may not reach pass gates on this map spawn.",
                        flush=True,
                    )
            if self._require_pass_maneuver_validation() and not self._validate_spawn_pass_maneuver(spec, world):
                if max_repick > 0 and self._try_repick_corridor_for_pass(spec, world):
                    self._extend_lead_to_target_gap(float(self._spawn_lead_m), spec=spec)
                else:
                    self.last_error = "pass_maneuver_validation_failed"
                    print(f"[CARLA] Bootstrap failed: {self.last_error}")
                    return False
            self._assert_required_actors()
            self._attach_sensors()
            self._set_spectator_behind_ego()
            self.init_logical_anchor(world.ego_x_m)
            self._ensure_spawn_gaps(spec)
            self._finalize_spawn_layout(spec)
            self.reset_episode_state(settle=True)
            try:
                self.run_pre_control_sanity(for_follow_lead=True, align_ego=True, spec=spec)
            except Exception as e:
                self.last_error = str(e)
                print(f"[CARLA] Bootstrap failed pre-control: {e}", flush=True)
                return False
            if not self.wait_for_sensor_frames(max_ticks=50, timeout_s=10.0):
                self.last_error = "sensor warmup failed after bootstrap"
                return False
            self.ready = True
            self._scenario_id = spec.scenario_id
            env_kind = os.environ.get("AUTOPASS_ENVIRONMENT", "highway")
            try:
                self.assert_curated_corridor_or_raise(env_kind)
            except Exception as e:
                self.last_error = str(e)
                print(f"[CARLA] Bootstrap failed: {e}")
                return False
            if self.last_bootstrap_action == "load_map":
                pass  # _map_name already set
            elif not self._map_name:
                self._map_name = short

            from perception.carla_validation import _validate_carla_actors

            issues = _validate_carla_actors(self)
            if issues:
                print(f"[CARLA] Layout warnings after spawn: {', '.join(issues)}")
            print(
                "[CARLA] Scenario live — ego/lead/rear on travel lane; "
                "oncoming on opposing lane (no same-lane head-on spawn)."
            )
            return True
        except Exception as e:
            self.last_error = str(e)
            print(f"[CARLA] Bootstrap failed: {e}")
            return False

    def _lane_edge_clearance_m(self, lane_wp) -> float:
        from perception.carla_pass_maneuver import estimate_edge_clearance_m

        return estimate_edge_clearance_m(self, lane_wp) or 0.0

    def _find_passing_lane_wp(self, wp):
        """Same-direction adjacent lane with best edge clearance (avoid shoulder/wall side)."""
        carla = self.carla
        if wp is None:
            return None
        best = None
        best_clear = -1.0
        best_side = "left"
        for side, getter in (("left", wp.get_left_lane), ("right", wp.get_right_lane)):
            try:
                adj = getter()
            except Exception:
                adj = None
            if adj is None or adj.lane_type != carla.LaneType.Driving:
                continue
            if adj.lane_id * wp.lane_id <= 0:
                continue
            clearance = self._lane_edge_clearance_m(adj)
            if clearance > best_clear:
                best_clear = clearance
                best = adj
                best_side = side
        if best is not None:
            self._passing_side = best_side
        return best

    def _require_pass_maneuver_validation(self) -> bool:
        import os

        from autopass.config import hero_corridor_enabled

        # Pass-smoke and hero demo run the real maneuver; skip expensive boot dry-run + repick loop.
        if os.environ.get("AUTOPASS_CARLA_PASS_SMOKE", "").strip() in ("1", "true", "True"):
            return False
        if os.environ.get("AUTOPASS_CARLA_SKIP_PASS_BOOT_VALIDATE", "").strip() in ("1", "true", "True"):
            return False
        if os.environ.get("AUTOPASS_CARLA_VALIDATE_PASS_ON_BOOT", "").strip() not in ("1", "true", "True"):
            return False
        return hero_corridor_enabled()

    def _validate_spawn_pass_maneuver(self, spec: ScenarioSpec, world: WorldState) -> bool:
        from perception.carla_pass_maneuver import run_scripted_pass_maneuver

        if self._passing_wp is None:
            print("[CARLA] Pass validation: no passing lane at spawn")
            return False
        min_pass_horizon = float(os.environ.get("AUTOPASS_CARLA_MIN_PASSING_HORIZON_M", "48"))
        pass_horizon = self.passing_lane_horizon_from_spawn_m()
        if pass_horizon < min_pass_horizon:
            print(
                f"[CARLA] Pass validation: passing lane too short ({pass_horizon:.0f}m "
                f"< {min_pass_horizon:.0f}m) — need another spawn",
                flush=True,
            )
            return False
        self._pass_validation_in_progress = True
        try:
            result = run_scripted_pass_maneuver(
                self, spec, world, verbose=False, total_max_s=40.0, use_state_machine=True
            )
        finally:
            self._pass_validation_in_progress = False
        self._pass_maneuver_validated = result.ok
        # Restore spawn layout after dry-run validation maneuver
        if self.actors.get("ego"):
            self.actors["ego"].set_transform(self._layout_transform("ego", 0.0, spec=spec))
        if self.actors.get("lead"):
            self.actors["lead"].set_transform(
                self._layout_transform("lead", self._spawn_lead_m, spec=spec)
            )
        if self.actors.get("rear"):
            rear_lane = "passing" if self._rear_on_passing_lane else "travel"
            self.actors["rear"].set_transform(
                self._layout_transform("rear", self._spawn_rear_m, lane=rear_lane, spec=spec)
            )
        if self.actors.get("oncoming"):
            self.actors["oncoming"].set_transform(self._role_transform("oncoming", self._spawn_on_m))
        self.reset_episode_state(settle=True)
        if result.ok:
            print(
                f"[CARLA] Pass maneuver validation ok "
                f"(max_lane={result.max_lane_center_m:.2f}m edge={result.min_edge_clearance_m:.2f}m)",
                flush=True,
            )
        else:
            print(
                f"[CARLA] Pass maneuver validation failed: {', '.join(result.issues[:6])}",
                flush=True,
            )
        return result.ok

    def _apply_corridor_pick(self, spawns, rec) -> None:
        best_wp = self.map.get_waypoint(spawns[rec.spawn_index].location, project_to_road=True)
        self._travel_wp = best_wp
        self._route_cursor = best_wp
        self.anchor_wp = best_wp
        self._passing_wp = self._find_passing_lane_wp(best_wp)
        self._opposing_wp = self._find_opposing_lane_wp(best_wp)
        self._corridor_report = rec.report
        print(
            f"[CARLA] Corridor pick spawn_index={rec.spawn_index} "
            f"{rec.report.summary_line()}",
            flush=True,
        )

    def _try_repick_corridor_for_pass(self, spec: ScenarioSpec, world: WorldState) -> bool:
        pool = list(self._corridor_pick_pool or [])
        spawns = self.map.get_spawn_points()
        max_attempts = int(os.environ.get("AUTOPASS_CARLA_MAX_CORRIDOR_REPICK", "5"))
        attempts = 0
        for rec in pool:
            if rec.spawn_index == getattr(self._corridor_report, "spawn_index", None):
                continue
            if not rec.report.has_passing_lane:
                continue
            attempts += 1
            if attempts > max_attempts:
                print(
                    f"[CARLA] Stopping corridor repick after {max_attempts} attempts "
                    f"(set AUTOPASS_CARLA_MAX_CORRIDOR_REPICK to raise limit)",
                    flush=True,
                )
                break
            print(f"[CARLA] Retrying corridor spawn_index={rec.spawn_index} for pass maneuver ...", flush=True)
            self._destroy_actors_and_sensors()
            self._apply_corridor_pick(spawns, rec)
            self._spawn_scenario(spec, world, repick_spawn=False)
            self._ensure_spawn_gaps(spec)
            self.reset_episode_state(settle=True)
            if self._validate_spawn_pass_maneuver(spec, world):
                return True
        return False

    def _straight_segment_score(self, wp) -> float:
        """Higher when the lane stays straight ahead (reject T-junction branches)."""
        if getattr(wp, "is_junction", False):
            return -1.0
        yaw0 = wp.transform.rotation.yaw
        cur = wp
        score = 10.0
        for step_m in (8.0, 8.0, 12.0, 12.0):
            nxt = cur.next(step_m)
            if not nxt or len(nxt) > 1:
                return -1.0
            cur = nxt[0]
            if getattr(cur, "is_junction", False):
                return -1.0
            if abs(cur.transform.rotation.yaw - yaw0) > 12.0:
                return -1.0
            score += 2.0
        return score

    def _find_opposing_lane_wp(self, wp):
        carla = self.carla
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

    def _pick_highway_spawn(self, *, require_curated: bool | None = None):
        from autopass.config import corridor_validation_mode, curated_corridor_enabled, hero_corridor_enabled
        from perception.carla_corridor import (
            CURATED_CORRIDOR_CANDIDATES,
            HERO_CORRIDOR_WARNING,
            build_scan_diagnostics,
            corridor_accepted_for_production,
            evaluate_spawn_candidate,
            format_diagnostics_report,
            pick_best_candidate,
            pick_hero_candidate,
            scan_spawn_candidates,
            validate_passing_corridor,
        )

        carla = self.carla
        if require_curated is None:
            require_curated = curated_corridor_enabled()
        spawns = self.map.get_spawn_points()
        if not spawns:
            raise RuntimeError("No spawn points on map")

        mode = corridor_validation_mode()
        if hero_corridor_enabled():
            mode = "hero"

        map_name = self._map_name or "Town04"
        max_scan = 300

        def _evaluate_indices(indices, vmode):
            return scan_spawn_candidates(
                spawns,
                map_name=map_name,
                map_obj=self.map,
                validation_mode=vmode,
                carla=carla,
                world=self.world,
                max_candidates=max_scan,
                indices=indices,
                find_passing_lane=self._find_passing_lane_wp,
                find_opposing_lane=self._find_opposing_lane_wp,
            )

        auto_indices = list(range(min(max_scan, len(spawns))))
        records = _evaluate_indices(auto_indices, mode)
        ranked = sorted(records, key=lambda r: r.near_miss_score, reverse=True)
        if mode == "hero":
            hero_pool = [r for r in ranked if r.report.hero_ok or r.ok]
            hero_pool.sort(
                key=lambda r: (
                    r.report.has_passing_lane,
                    r.report.has_opposing_lane,
                    r.near_miss_score,
                ),
                reverse=True,
            )
            self._corridor_pick_pool = hero_pool[:20]
            chosen = hero_pool[0] if hero_pool else None
        else:
            self._corridor_pick_pool = [r for r in ranked if r.ok][:20]
            chosen = pick_best_candidate(records)

        if chosen is None and mode != "hero":
            hero_records = _evaluate_indices(auto_indices, "hero")
            hero_pool = sorted(
                [r for r in hero_records if r.report.hero_ok],
                key=lambda r: (r.report.has_passing_lane, r.near_miss_score),
                reverse=True,
            )
            self._corridor_pick_pool = hero_pool[:20]
            chosen = hero_pool[0] if hero_pool else None

        manual_indices = CURATED_CORRIDOR_CANDIDATES.get(map_name, [])
        if chosen is None and manual_indices:
            print(f"[CARLA] Trying {len(manual_indices)} manual curated spawn indices on {map_name} ...")
            manual_records = _evaluate_indices(manual_indices, mode)
            chosen = pick_best_candidate(manual_records) if mode != "hero" else pick_hero_candidate(manual_records)
            if chosen is None:
                hero_manual = _evaluate_indices(manual_indices, "hero")
                chosen = pick_hero_candidate(hero_manual)
            if chosen is not None:
                records = manual_records

        self._last_corridor_diagnostics = build_scan_diagnostics(
            records, map_name=map_name, validation_mode=mode
        )

        if chosen is None:
            diag_text = format_diagnostics_report(self._last_corridor_diagnostics, top_k=5)
            msg = (
                f"No curated passing corridor on {map_name} "
                f"(scanned {len(records)} spawn points, mode={mode})."
            )
            if require_curated:
                raise RuntimeError(msg)
            print(f"[CARLA] {msg}\n{diag_text}\nUsing best-effort spawn (not validated).")
            best_wp = self.map.get_waypoint(spawns[0].location, project_to_road=True)
            best_report = validate_passing_corridor(
                best_wp,
                carla=carla,
                world=self.world,
                validation_mode="hero",
                require_opposing_lane=False,
                find_passing_lane=self._find_passing_lane_wp,
                find_opposing_lane=self._find_opposing_lane_wp,
            )
            self._travel_wp = best_wp
            self._route_cursor = best_wp
            self.anchor_wp = best_wp
            self._passing_wp = self._find_passing_lane_wp(best_wp)
            self._opposing_wp = self._find_opposing_lane_wp(best_wp)
            self._corridor_report = best_report
            self._corridor_hero_fallback = True
            return best_wp

        best_report = chosen.report
        accepted, used_hero = corridor_accepted_for_production(best_report)
        if not accepted and require_curated:
            from perception.carla_corridor import NOT_CURATED_CORRIDOR_MSG

            raise RuntimeError(NOT_CURATED_CORRIDOR_MSG)

        chosen.report.spawn_index = chosen.spawn_index
        self._apply_corridor_pick(spawns, chosen)
        best_report = chosen.report
        self._corridor_hero_fallback = used_hero or (
            hero_corridor_enabled() and best_report.hero_ok and not best_report.ok
        )
        if best_report.ok or best_report.presentation_ok:
            print(f"[CARLA] Curated corridor: {best_report.summary_line()}")
        elif self._corridor_hero_fallback:
            print(f"[CARLA] WARNING: {HERO_CORRIDOR_WARNING}")
            print(f"[CARLA] Hero corridor: {best_report.summary_line()}")
        elif require_curated:
            from perception.carla_corridor import NOT_CURATED_CORRIDOR_MSG

            raise RuntimeError(NOT_CURATED_CORRIDOR_MSG)
        if self._opposing_wp is None:
            print("[CARLA] WARNING: no opposing-direction lane near spawn — oncoming omitted.")
        if self._passing_wp is None:
            print("[CARLA] WARNING: no passing lane at spawn — pass maneuvers will fail.")
        return self._travel_wp

    def scan_passing_corridors(self, *, max_candidates: int = 300, validation_mode: str | None = None) -> list:
        """Return (CorridorReport, waypoint) for each valid spawn candidate."""
        from autopass.config import corridor_validation_mode
        from perception.carla_corridor import scan_spawn_candidates

        out = []
        if self.map is None or self.carla is None:
            return out
        mode = validation_mode or corridor_validation_mode()
        spawns = self.map.get_spawn_points()
        records = scan_spawn_candidates(
            spawns,
            map_name=self._map_name or "Town04",
            map_obj=self.map,
            validation_mode=mode,  # type: ignore[arg-type]
            carla=self.carla,
            world=self.world,
            max_candidates=max_candidates,
            find_passing_lane=self._find_passing_lane_wp,
            find_opposing_lane=self._find_opposing_lane_wp,
        )
        for rec in records:
            if rec.ok:
                wp = self.map.get_waypoint(spawns[rec.spawn_index].location, project_to_road=True)
                out.append((rec.report, wp))
        out.sort(key=lambda item: item[0].forward_length_m + item[0].backward_length_m, reverse=True)
        return out

    def diagnose_passing_corridors(
        self,
        *,
        max_candidates: int = 300,
        validation_mode: str | None = None,
        top_k: int = 5,
    ) -> str:
        from autopass.config import corridor_validation_mode
        from perception.carla_corridor import build_scan_diagnostics, format_diagnostics_report, scan_spawn_candidates

        if self.map is None or self.carla is None:
            return "session map/carla not ready"
        mode = validation_mode or corridor_validation_mode()
        spawns = self.map.get_spawn_points()
        records = scan_spawn_candidates(
            spawns,
            map_name=self._map_name or "Town04",
            map_obj=self.map,
            validation_mode=mode,  # type: ignore[arg-type]
            carla=self.carla,
            world=self.world,
            max_candidates=max_candidates,
            find_passing_lane=self._find_passing_lane_wp,
            find_opposing_lane=self._find_opposing_lane_wp,
        )
        diag = build_scan_diagnostics(records, map_name=self._map_name or "Town04", validation_mode=mode)
        self._last_corridor_diagnostics = diag
        return format_diagnostics_report(diag, top_k=top_k)

    def assert_curated_corridor_or_raise(self, environment: str = "highway") -> None:
        from autopass.config import AutopassConfigurationError, curated_corridor_enabled, hero_corridor_enabled
        from perception.carla_corridor import (
            HERO_CORRIDOR_WARNING,
            NOT_CURATED_CORRIDOR_MSG,
            corridor_accepted_for_production,
            validate_passing_corridor,
        )

        if not self.ready or self._travel_wp is None:
            return
        report = self._corridor_report
        if report is None or not (report.ok or report.presentation_ok or report.hero_ok):
            report = validate_passing_corridor(
                self._travel_wp,
                carla=self.carla,
                world=self.world,
                validation_mode="presentation",
                find_passing_lane=self._find_passing_lane_wp,
                find_opposing_lane=self._find_opposing_lane_wp,
            )
            self._corridor_report = report

        accepted, used_hero = corridor_accepted_for_production(report)
        if accepted:
            if used_hero or self._corridor_hero_fallback:
                if hero_corridor_enabled() or used_hero:
                    print(f"[CARLA] WARNING: {HERO_CORRIDOR_WARNING}", flush=True)
                    return
            return

        if curated_corridor_enabled() or environment in ("highway", "synthetic"):
            if hero_corridor_enabled() and report.hero_ok:
                print(f"[CARLA] WARNING: {HERO_CORRIDOR_WARNING}", flush=True)
                self._corridor_hero_fallback = True
                return
            detail = ", ".join(report.issues[:6]) if report.issues else "unknown"
            raise AutopassConfigurationError(f"{NOT_CURATED_CORRIDOR_MSG} ({detail})")

    def _wp_ahead(self, base_wp, distance_m: float, *, same_carriageway: bool = False):
        if distance_m < 0.5:
            return base_wp
        tw = self._travel_wp
        if tw is not None and base_wp is not None and int(base_wp.road_id) == int(tw.road_id):
            if same_carriageway or int(base_wp.lane_id) == int(tw.lane_id):
                locked = self._wp_on_lane_ahead(
                    base_wp,
                    distance_m,
                    tw.lane_id,
                    tw.road_id,
                    same_carriageway=same_carriageway,
                )
                if locked is not None:
                    return locked
        nxt = base_wp.next(distance_m)
        return nxt[0] if nxt else base_wp

    def _wp_behind(self, base_wp, distance_m: float):
        if distance_m < 0.5:
            return base_wp
        tw = self._travel_wp
        if tw is not None and base_wp is not None:
            if base_wp.road_id == tw.road_id and base_wp.lane_id == tw.lane_id:
                cur = base_wp
                remaining = float(distance_m)
                while remaining > 0.5:
                    step = min(4.0, remaining)
                    prv = cur.previous(step)
                    if not prv or len(prv) > 1:
                        break
                    cand = prv[0]
                    if getattr(cand, "is_junction", False):
                        break
                    if cand.lane_id != tw.lane_id or cand.road_id != tw.road_id:
                        break
                    cur = cand
                    remaining -= step
                return cur
        prv = base_wp.previous(distance_m)
        return prv[0] if prv else base_wp

    def _opposing_wp_ahead(self, distance_m: float):
        if self._opposing_wp is None or self._travel_wp is None:
            return self._travel_wp
        try:
            seed = self.map.get_waypoint(self._travel_wp.transform.location, project_to_road=True)
            for getter in (seed.get_left_lane, seed.get_right_lane):
                try:
                    adj = getter()
                except Exception:
                    adj = None
                if adj is None or adj.lane_type != self.carla.LaneType.Driving:
                    continue
                if adj.lane_id * seed.lane_id < 0:
                    return self._wp_ahead(adj, distance_m)
        except Exception:
            pass
        return self._wp_ahead(self._opposing_wp, distance_m)

    def _extend_lead_to_target_gap(self, target_m: float, spec: ScenarioSpec | None = None) -> None:
        """Move lead forward until projected gap reaches target_m on the travel corridor."""
        lead = self.actors.get("lead")
        if lead is None:
            return
        target_m = float(target_m)
        use_axis = self._uses_axis_longitudinal_layout(spec)
        for _ in range(24):
            self._layout_tick_sync()
            gap = self.lead_longitudinal_gap_m()
            if gap >= target_m - 0.75:
                tw = self._travel_wp
                if tw is not None and self.map is not None:
                    try:
                        wp = self.map.get_waypoint(lead.get_location(), project_to_road=True)
                        on_corridor, _ = self._actor_on_travel_corridor(lead, tw, spec)
                        if on_corridor:
                            if not use_axis or gap >= target_m - 0.5:
                                return
                    except Exception:
                        if not use_axis:
                            return
                elif not use_axis:
                    return
                elif gap >= target_m - 0.5:
                    return
            placed = False
            if use_axis:
                self.refresh_axis_ego_from_live()
                tf = self._axis_actor_transform(target_m, lane="travel")
                if tf is not None:
                    lead.set_transform(tf)
                    placed = True
            if not placed:
                tf = self._corridor_actor_transform(target_m, lane="travel")
                if tf is not None:
                    lead.set_transform(tf)
                    placed = True
            if placed:
                self._layout_tick_sync()
                continue
            tw = self._travel_wp
            if tw is None or self.map is None or self.carla is None:
                return
            extra = max(2.0, target_m - gap)
            try:
                lead_wp = self.map.get_waypoint(lead.get_location(), project_to_road=True)
            except Exception:
                return
            nwp = self._wp_on_lane_ahead(
                lead_wp,
                extra,
                tw.lane_id,
                tw.road_id,
                same_carriageway=True,
            )
            if nwp is None:
                return
            loc = nwp.transform.location
            lead.set_transform(
                self.carla.Transform(
                    self.carla.Location(float(loc.x), float(loc.y), float(loc.z) + 0.3),
                    nwp.transform.rotation,
                )
            )
            if self.is_synchronous_mode():
                self.tick()

    def _assert_required_actors(self) -> None:
        from autopass.config import AutopassConfigurationError, is_test_mode

        missing = [n for n in ("ego", "lead") if self.actors.get(n) is None]
        if missing and not is_test_mode():
            raise AutopassConfigurationError(
                f"CARLA bootstrap missing required actors: {', '.join(missing)}"
            )

    def _role_transform(self, role: str, distance_m: float, *, rear_on_passing_lane: bool = False):
        carla = self.carla
        if role == "lead":
            wp = self._wp_ahead(self._travel_wp, distance_m, same_carriageway=True)
        elif role == "rear":
            base = self._passing_wp if rear_on_passing_lane and self._passing_wp is not None else self._travel_wp
            wp = self._wp_behind(base, distance_m)
        elif role == "oncoming":
            wp = self._opposing_wp_ahead(distance_m)
        elif role == "passing":
            base = self._passing_wp if self._passing_wp is not None else self._travel_wp
            wp = self._wp_ahead(base, 0.0) if distance_m < 0.5 else self._wp_ahead(base, distance_m)
        else:
            wp = self._travel_wp
        base_loc = wp.transform.location
        loc = carla.Location(float(base_loc.x), float(base_loc.y), float(base_loc.z) + 0.3)
        rot = wp.transform.rotation
        return carla.Transform(loc, carla.Rotation(rot.pitch, rot.yaw, rot.roll))

    def _apply_world_sync_settings(self) -> None:
        """
        CARLA-recommended sync + fixed delta for deterministic sensors and physics.

        See: https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/
        """
        if self.world is None:
            return
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        # Physics substepping: fixed_delta (0.05) <= max_substep_delta * max_substeps
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        self.world.apply_settings(settings)
        self.world.tick()

    def _release_world_sync_settings(self) -> None:
        """Return world to async mode on shutdown (CARLA best practice)."""
        if self.world is None:
            return
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        except Exception:
            pass

    def _spawn_one(self, bp, tf, name: str):
        """Spawn with CARLA try_spawn_actor; retry along forward vector on collision."""
        carla = self.carla
        if self.world is None or carla is None:
            print(f"[CARLA]   FAILED {name}: world not ready")
            return None
        actor = self.world.try_spawn_actor(bp, tf)
        if actor is not None:
            actor.set_simulate_physics(False)
            print(f"[CARLA]   spawned {name} at ({tf.location.x:.0f}, {tf.location.y:.0f})")
            return actor
        yaw_rad = math.radians(float(getattr(tf.rotation, "yaw", 0.0)))
        fwd_x = math.cos(yaw_rad)
        fwd_y = math.sin(yaw_rad)
        for step_m in (1.5, 3.0, 5.0, -1.5, -3.0):
            loc = tf.location
            retry_tf = carla.Transform(
                carla.Location(
                    float(loc.x) + fwd_x * step_m,
                    float(loc.y) + fwd_y * step_m,
                    float(loc.z),
                ),
                tf.rotation,
            )
            actor = self.world.try_spawn_actor(bp, retry_tf)
            if actor is not None:
                actor.set_simulate_physics(False)
                print(
                    f"[CARLA]   spawned {name} at ({retry_tf.location.x:.0f}, "
                    f"{retry_tf.location.y:.0f}) after {step_m:+.1f}m retry"
                )
                return actor
        try:
            actor = self.world.spawn_actor(bp, tf)
            actor.set_simulate_physics(False)
            print(f"[CARLA]   spawned {name} at ({tf.location.x:.0f}, {tf.location.y:.0f})")
            return actor
        except Exception as e:
            print(f"[CARLA]   FAILED {name}: {e}")
            return None

    def _spawn_scenario(self, spec: ScenarioSpec, world: WorldState, *, repick_spawn: bool = True) -> None:
        self._scenario_id = spec.scenario_id
        if repick_spawn or self._travel_wp is None:
            self._pick_highway_spawn()
        bp_lib = self.world.get_blueprint_library()
        ego_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        lead_bp = bp_lib.filter("vehicle.audi.tt")[0]
        rear_bp = bp_lib.filter("vehicle.nissan.micra")[0]
        on_bp = bp_lib.filter("vehicle.volkswagen.t2")[0]

        profile = self._spawn_profile(spec)
        lead_cap = float(profile["lead_cap_m"])
        lead_floor = float(profile["lead_floor_m"])
        axis_spawn = bool(profile.get("axis_spawn"))
        self._axis_spawn_active = axis_spawn
        if axis_spawn and float(profile.get("lead_gap_m", 0.0)) > 0:
            self._spawn_lead_m = float(profile["lead_gap_m"])
        else:
            self._spawn_lead_m = max(min(spec.lead.distance_m, lead_cap), lead_floor)
        rear_on_passing = bool(profile.get("rear_passing_lane"))
        self._rear_on_passing_lane = rear_on_passing
        if rear_on_passing:
            self._spawn_rear_m = max(float(profile.get("rear_spawn_m", 14.0)), rear_follow_min_m() + 2.0)
        else:
            self._spawn_rear_m = max(
                min(max(8.0, world.ego_x_m - world.rear_x_m), 30.0),
                rear_follow_min_m() + 4.0,
            )
        self._spawn_on_m = min(max(18.0, world.oncoming_x_m - world.ego_x_m), 50.0)

        print("[CARLA] Spawning actors:")
        ego_tf = self._role_transform("travel", 0.0)
        self.actors["ego"] = self._spawn_one(ego_bp, ego_tf, "ego")
        if axis_spawn:
            self._cache_axis_basis_from_ego_transform(ego_tf)
            self.actors["lead"] = self._spawn_one(
                lead_bp,
                self._layout_transform("lead", self._spawn_lead_m, spec=spec),
                "lead",
            )
            rear_lane = "passing" if rear_on_passing else "travel"
            self.actors["rear"] = self._spawn_one(
                rear_bp,
                self._layout_transform("rear", self._spawn_rear_m, lane=rear_lane, spec=spec),
                "rear",
            )
        else:
            self.actors["lead"] = self._spawn_one(
                lead_bp, self._role_transform("lead", self._spawn_lead_m), "lead"
            )
            self.actors["rear"] = self._spawn_one(
                rear_bp,
                self._role_transform("rear", self._spawn_rear_m, rear_on_passing_lane=rear_on_passing),
                "rear",
            )
        self._ensure_spawn_gaps(spec)
        self._snap_convoy_to_travel_lane(spec)
        self._extend_lead_to_target_gap(float(self._spawn_lead_m), spec=spec)
        if axis_spawn:
            if self.is_synchronous_mode():
                self.tick()
            ego = self.actors.get("ego")
            if ego is not None:
                self._cache_axis_basis_from_ego_transform(ego.get_transform())
            self._log_axis_spawn_layout(spec)
            try:
                from perception.lead_gap_diagnostics import log_lead_gap_checkpoint

                log_lead_gap_checkpoint(self, "A_after_actor_spawn", note=spec.scenario_id)
            except Exception:
                pass
        if self._opposing_wp is not None:
            self.actors["oncoming"] = self._spawn_one(
                on_bp, self._role_transform("oncoming", self._spawn_on_m), "oncoming"
            )
        else:
            self.actors["oncoming"] = None
            print("[CARLA]   skipped oncoming (no opposing lane at spawn)")
        n = sum(1 for v in self.actors.values() if v is not None)
        print(f"[CARLA] {n}/4 actors in world")
        self._spawn_ego_s = None
        self._spawn_logical_x = None
        self.init_logical_anchor(world.ego_x_m)

    def apply_world_state(self, spec: ScenarioSpec, world: WorldState, *, sync_poses: bool = False) -> None:
        if not self.ready:
            return
        if sync_poses:
            self.sync_npc_poses(spec, world)
        self._set_spectator_behind_ego()

    def _apply_last_vehicle_control(self, ego) -> None:
        ctrl = getattr(self, "_last_vehicle_control", None)
        if ctrl is None or ego is None:
            return
        try:
            ego.apply_control(ctrl)
        except Exception:
            pass

    def animate_steps(
        self,
        spec: ScenarioSpec,
        world: WorldState,
        steps: int = 8,
        *,
        on_tick=None,
    ) -> None:
        """Advance CARLA for recording/sensors; keep physics running between graph steps."""
        if steps <= 0:
            return
        ego = self.actors.get("ego")
        dt = float(getattr(self, "fixed_delta_seconds", 0.05) or 0.05)
        pass_active = False
        try:
            from perception.pass_control_fsm import get_pass_control_state

            pst = get_pass_control_state(self)
            pass_active = bool(
                pst.active and pst.phase in ("lane_change", "overtake", "merge_back", "prepare_pass")
            )
        except Exception:
            pass_active = False
        for tick_i in range(steps):
            self.apply_world_state(spec, world)
            if getattr(self, "_ego_physics", False) and ego is not None:
                if pass_active:
                    self._apply_last_vehicle_control(ego)
                else:
                    self.apply_inter_step_cruise(spec, world)
            self.tick_npcs_kinematic(spec, dt)
            self.tick()
            if on_tick is None or ego is None:
                continue
            try:
                v = math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2 + ego.get_velocity().z ** 2)
                ego_lane = self.infer_ego_lane_index()
                partial = self.materialize_logical_world(
                    world,
                    measured_speed_mps=float(v),
                    duration_s=dt * (tick_i + 1),
                    ego_lane=ego_lane,
                    passed=world.passed,
                    collision=False,
                    done=False,
                )
                on_tick(partial, tick_i + 1)
            except Exception:
                pass

    def _configure_camera_bp(self, bp, *, width: int = 640, height: int = 256, fov: float = 90.0):
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        return bp

    def _make_sensor_listener(self, name: str):
        def _listener(image) -> None:
            try:
                self._on_sensor_frame(name, image)
            except Exception as e:
                self._sensor_callback_errors.append(f"{name}: {e!r}")

        return _listener

    def _on_sensor_frame(self, name: str, img) -> None:
        if name == "rgb":
            self._rgb_buf = img
        elif name == "depth":
            self._depth_buf = img
        elif name == "seg":
            self._seg_buf = img
        elif name == "overhead":
            self._overhead_buf = img
        self._sensor_frame_counts[name] = self._sensor_frame_counts.get(name, 0) + 1
        try:
            self._sensor_last_frame[name] = int(getattr(img, "frame", -1))
        except Exception:
            pass

    def _attach_sensors(self) -> None:
        carla = self.carla
        ego = self.actors.get("ego")
        if ego is None:
            raise RuntimeError("Cannot attach sensors: ego actor missing")
        bp_lib = self.world.get_blueprint_library()
        cam = self._configure_camera_bp(bp_lib.find("sensor.camera.rgb"))
        depth = self._configure_camera_bp(bp_lib.find("sensor.camera.depth"))
        seg = self._configure_camera_bp(bp_lib.find("sensor.camera.semantic_segmentation"))
        tr = carla.Transform(carla.Location(x=1.6, z=1.4), carla.Rotation(pitch=0.0))
        self._rgb_buf = None
        self._depth_buf = None
        self._seg_buf = None
        self._overhead_buf = None
        self._sensor_frame_counts = {"rgb": 0, "depth": 0, "seg": 0, "overhead": 0}
        self._sensor_last_frame = {"rgb": -1, "depth": -1, "seg": -1, "overhead": -1}
        self._sensor_callback_errors = []
        self._sensor_listeners.clear()
        self._destroy_sensor_actors()

        self.sensors["rgb"] = self.world.spawn_actor(cam, tr, attach_to=ego)
        self.sensors["depth"] = self.world.spawn_actor(depth, tr, attach_to=ego)
        self.sensors["seg"] = self.world.spawn_actor(seg, tr, attach_to=ego)
        for name in ("rgb", "depth", "seg"):
            listener = self._make_sensor_listener(name)
            self._sensor_listeners[name] = listener
            self.sensors[name].listen(listener)

        over_tr = carla.Transform(carla.Location(z=38.0), carla.Rotation(pitch=-90.0))
        self.sensors["overhead"] = self.world.spawn_actor(cam, over_tr, attach_to=ego)
        oh_listener = self._make_sensor_listener("overhead")
        self._sensor_listeners["overhead"] = oh_listener
        self.sensors["overhead"].listen(oh_listener)

        # One tick so sensor actors register before warmup loop.
        if self.is_synchronous_mode():
            self.world.tick()

    def is_synchronous_mode(self) -> bool:
        if self.world is None:
            return False
        try:
            return bool(self.world.get_settings().synchronous_mode)
        except Exception:
            return False

    def world_settings_snapshot(self) -> Dict[str, object]:
        if self.world is None:
            return {"synchronous_mode": None, "fixed_delta_seconds": None}
        try:
            s = self.world.get_settings()
            return {
                "synchronous_mode": bool(s.synchronous_mode),
                "fixed_delta_seconds": float(s.fixed_delta_seconds),
            }
        except Exception:
            return {"synchronous_mode": None, "fixed_delta_seconds": None}

    def _sensor_channels_ready(self) -> Dict[str, bool]:
        needed = ("rgb", "depth", "seg")
        return {
            k: self._sensor_frame_counts.get(k, 0) > 0 and getattr(self, f"_{k}_buf", None) is not None
            for k in needed
        }

    def sensor_status(self) -> Dict[str, object]:
        ws = self.world_settings_snapshot()
        return {
            "connected": self.client is not None,
            "map": self._map_name,
            "ready": self.ready,
            "ego_exists": self.actors.get("ego") is not None,
            "sensors": {k: v is not None for k, v in self.sensors.items()},
            "listeners_registered": {k: k in self._sensor_listeners for k in ("rgb", "depth", "seg", "overhead")},
            "frame_counts": dict(self._sensor_frame_counts),
            "last_frame_ids": dict(self._sensor_last_frame),
            "buffers_ready": self._sensor_channels_ready(),
            "sync_mode": ws.get("synchronous_mode"),
            "fixed_delta_seconds": ws.get("fixed_delta_seconds"),
            "warmup_ticks_attempted": self._sensor_warmup_ticks,
            "warmup_tick_method": self._sensor_warmup_method,
            "callback_errors": list(self._sensor_callback_errors),
        }

    def sensor_actor_debug(self) -> Dict[str, object]:
        out: Dict[str, object] = {"ego": None, "sensors": {}}
        ego = self.actors.get("ego")
        if ego is not None:
            try:
                tf = ego.get_transform()
                loc = tf.location
                out["ego"] = {
                    "id": int(ego.id),
                    "type_id": str(ego.type_id),
                    "transform": {"x": round(loc.x, 2), "y": round(loc.y, 2), "z": round(loc.z, 2)},
                }
            except Exception as e:
                out["ego"] = {"error": repr(e)}
        for name in ("rgb", "depth", "seg", "overhead"):
            sensor = self.sensors.get(name)
            if sensor is None:
                out["sensors"][name] = {"exists": False}
                continue
            rec: Dict[str, object] = {
                "exists": True,
                "listen_registered": name in self._sensor_listeners,
                "frames": self._sensor_frame_counts.get(name, 0),
                "last_frame_id": self._sensor_last_frame.get(name, -1),
            }
            try:
                rec["id"] = int(sensor.id)
                rec["type_id"] = str(sensor.type_id)
                parent = sensor.parent
                rec["parent_id"] = int(parent.id) if parent is not None else None
                stf = sensor.get_transform()
                sloc = stf.location
                rec["transform"] = {"x": round(sloc.x, 2), "y": round(sloc.y, 2), "z": round(sloc.z, 2)}
            except Exception as e:
                rec["error"] = repr(e)
            out["sensors"][name] = rec
        return out

    def sensor_full_diagnostic(self) -> str:
        st = self.sensor_status()
        actors = self.sensor_actor_debug()
        lines = [
            f"client_connected={st['connected']}",
            f"map={st['map']}",
            f"session_ready={st['ready']}",
            f"synchronous_mode={st['sync_mode']}",
            f"fixed_delta_seconds={st['fixed_delta_seconds']}",
            f"ego={actors.get('ego')}",
            f"sensors_exist={st['sensors']}",
            f"listeners_registered={st['listeners_registered']}",
            f"frame_counts={st['frame_counts']}",
            f"last_frame_ids={st['last_frame_ids']}",
            f"buffers_ready={st['buffers_ready']}",
            f"warmup_ticks_attempted={st['warmup_ticks_attempted']}",
            f"warmup_tick_method={st['warmup_tick_method']}",
            f"callback_errors={st['callback_errors']}",
            f"sensor_actors={actors.get('sensors')}",
        ]
        return "\n".join(lines)

    def sensor_frame_diagnostic(self) -> str:
        return self.sensor_full_diagnostic()

    def wait_for_sensor_frames(
        self,
        *,
        max_ticks: int = 50,
        timeout_s: float = 10.0,
        verbose: bool = False,
    ) -> bool:
        """Tick until RGB+depth+seg have each received at least one frame."""
        import time

        if self.world is None:
            return False
        needed = ("rgb", "depth", "seg")
        missing_sensors = [k for k in needed if self.sensors.get(k) is None]
        if missing_sensors:
            self._sensor_callback_errors.append(f"missing_sensor_actors:{','.join(missing_sensors)}")
            return False

        t0 = time.perf_counter()
        self._sensor_warmup_ticks = 0
        sync = self.is_synchronous_mode()
        self._sensor_warmup_method = "world.tick" if sync else "world.wait_for_tick"

        for tick_i in range(1, max_ticks + 1):
            ready = self._sensor_channels_ready()
            if all(ready.values()):
                if verbose:
                    print(
                        f"   tick {tick_i}: rgb={self._sensor_frame_counts['rgb']} "
                        f"depth={self._sensor_frame_counts['depth']} "
                        f"seg={self._sensor_frame_counts['seg']}",
                        flush=True,
                    )
                return True
            self._advance_world_one_step(sync)
            self._sensor_warmup_ticks = tick_i
            if verbose and (tick_i <= 3 or tick_i % 5 == 0):
                print(
                    f"   tick {tick_i}: rgb={self._sensor_frame_counts['rgb']} "
                    f"depth={self._sensor_frame_counts['depth']} "
                    f"seg={self._sensor_frame_counts['seg']}",
                    flush=True,
                )
            if time.perf_counter() - t0 > timeout_s:
                break
        return all(self._sensor_channels_ready().values())

    def _advance_world_one_step(self, sync: bool) -> None:
        if self.world is None:
            return
        if sync:
            self.world.tick()
            return
        try:
            self.world.wait_for_tick(timeout=1.0)
        except Exception:
            try:
                self.world.tick()
            except Exception as e:
                self._sensor_callback_errors.append(f"tick_failed:{e!r}")

    def _set_spectator_behind_ego(self) -> None:
        ego = self.actors.get("ego")
        if not ego:
            return
        carla = self.carla
        ego_tf = ego.get_transform()
        spectator = self.world.get_spectator()
        back = ego_tf.get_forward_vector() * -10.0
        loc = ego_tf.location + back + carla.Location(z=5.0)
        spectator.set_transform(
            carla.Transform(loc, carla.Rotation(pitch=-15.0, yaw=ego_tf.rotation.yaw))
        )

    def tick(self) -> None:
        if self.world:
            self._advance_world_one_step(self.is_synchronous_mode())

    def grab_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if not self.ready or self._rgb_buf is None:
            return None
        rgb_raw = self._rgb_buf
        depth_raw = self._depth_buf
        seg_raw = self._seg_buf
        if depth_raw is None or seg_raw is None:
            return None
        rgb = np.frombuffer(rgb_raw.raw_data, dtype=np.uint8).reshape(
            (rgb_raw.height, rgb_raw.width, 4)
        )[:, :, :3]
        seg = np.frombuffer(seg_raw.raw_data, dtype=np.uint8).reshape(
            (seg_raw.height, seg_raw.width, 4)
        )[:, :, 2]
        depth_a = np.frombuffer(depth_raw.raw_data, dtype=np.uint8).reshape(
            (depth_raw.height, depth_raw.width, 4)
        )
        depth_m = (
            depth_a[:, :, 2].astype(np.float32)
            + depth_a[:, :, 1].astype(np.float32) * 256
            + depth_a[:, :, 0].astype(np.float32) * 256 * 256
        )
        depth_m = depth_m / (256**3 - 1) * 1000.0
        return rgb.copy(), seg.copy(), depth_m.copy()

    def grab_overhead_rgb(self) -> Optional[np.ndarray]:
        if not self.ready or self._overhead_buf is None:
            return None
        raw = self._overhead_buf
        return np.frombuffer(raw.raw_data, dtype=np.uint8).reshape((raw.height, raw.width, 4))[:, :, :3].copy()

    def grab_frame_pair(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        frame = self.grab_frame()
        if frame is None:
            return None
        rgb, seg, depth = frame
        return rgb, seg, depth, self.grab_overhead_rgb()

    def shutdown(self) -> None:
        self._destroy_actors_and_sensors()
        self._release_world_sync_settings()
        self.ready = False
        self._map_name = None
        self._scenario_id = None
        self._physical_key = None
        self._spawn_ego_s = None
        self._spawn_logical_x = None
        self._travel_wp = None
        self._route_cursor = None
        self._passing_wp = None
        self._opposing_wp = None
        self._corridor_report = None
        self._corridor_hero_fallback = False
        self._last_corridor_diagnostics = None
        self._episode_step = 0
        self._collision_events = []
        self._sensor_frame_counts = {"rgb": 0, "depth": 0, "seg": 0, "overhead": 0}
        self._sensor_last_frame = {"rgb": -1, "depth": -1, "seg": -1, "overhead": -1}
        self.last_bootstrap_action = "shutdown"


def bootstrap_minimal_ego_sensors(map_name: str = "Town04") -> bool:
    """Minimal path: load map, spawn one ego, attach sensors, warm frames."""
    from visual_world import curated_demo_scenarios, initialize_world

    session = get_session()
    if session.ready:
        session.shutdown()
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    return session.bootstrap(spec, world, map_name)


def run_sensor_smoke(*, minimal: bool = False, verbose: bool = True) -> int:
    """Run sensor smoke; return 0 on success."""
    from visual_world import curated_demo_scenarios, initialize_world

    session = get_session()
    if minimal:
        ok = bootstrap_minimal_ego_sensors("Town04")
    else:
        # Perception / pass demo layout (axis spawn, 32m lead) — not demo_01 waypoint cap.
        spec = curated_demo_scenarios()[6]
        world = initialize_world(spec)
        ok = bootstrap_carla_scenario(spec, world, map_name="Town04")

    if not ok:
        print(f"FAIL: bootstrap failed: {session.last_error}", flush=True)
        print(session.sensor_full_diagnostic(), flush=True)
        return 1

    print("3) warmup rgb/depth/seg ...", flush=True)
    if not session.wait_for_sensor_frames(max_ticks=50, timeout_s=10.0, verbose=verbose):
        print("FAIL: sensor warmup timed out", flush=True)
        print(session.sensor_full_diagnostic(), flush=True)
        return 1

    counts = session._sensor_frame_counts
    print(f"rgb frames: {counts.get('rgb', 0)}", flush=True)
    print(f"depth frames: {counts.get('depth', 0)}", flush=True)
    print(f"seg frames: {counts.get('seg', 0)}", flush=True)

    frame = session.grab_frame()
    if frame is None:
        print("FAIL: grab_frame returned None", flush=True)
        print(session.sensor_full_diagnostic(), flush=True)
        return 1
    rgb, seg, depth = frame
    print(f"rgb shape={rgb.shape} dtype={rgb.dtype}", flush=True)
    print(f"seg shape={seg.shape} dtype={seg.dtype}", flush=True)
    print(f"depth shape={depth.shape} dtype={depth.dtype}", flush=True)
    if counts.get("rgb", 0) >= 1 and counts.get("depth", 0) >= 1 and counts.get("seg", 0) >= 1:
        print("PASS: all sensor channels received frames", flush=True)
        return 0
    print("FAIL: one or more channels missing frames", flush=True)
    return 1


def get_session() -> CarlaScenarioSession:
    global _session
    if _session is None:
        _session = CarlaScenarioSession()
    return _session


def bootstrap_carla_scenario(
    spec: ScenarioSpec,
    world: WorldState,
    map_name: Optional[str] = None,
    *,
    physical_key: Optional[str] = None,
) -> bool:
    map_name = _normalize_map_name(map_name or spec.route.town or os.environ.get("AUTOPASS_CARLA_MAP", "Town04"))
    session = get_session()
    physical_key = physical_key or f"{map_name}|{spec.scenario_id}"

    if session.ready and _normalize_map_name(session._map_name or "") != map_name:
        session.shutdown()

    if session.ready:
        same_physical = session._physical_key == physical_key
        return session.respawn_episode(
            spec, world, physical_key=physical_key, same_physical=same_physical
        )

    ok = session.bootstrap(spec, world, map_name)
    if ok:
        session._physical_key = physical_key
        session._map_name = map_name
    return ok


def acquire_carla_frame(spec: ScenarioSpec, world: WorldState) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    session = get_session()
    if not session.ready:
        map_name = _normalize_map_name(spec.route.town or os.environ.get("AUTOPASS_CARLA_MAP", "Town04"))
        if not session.bootstrap(spec, world, map_name):
            return None
    if not session.wait_for_sensor_frames(max_ticks=10, timeout_s=2.0):
        return None
    return session.grab_frame()


def run_carla_preflight(*, require_frames: bool = True) -> None:
    """Verify actors are separated and sensors produce frames before benchmark rows."""
    from autopass.config import AutopassConfigurationError
    from autopass.physics import _validate_carla_actors

    session = get_session()
    if not session.ready:
        raise AutopassConfigurationError("CARLA preflight: session not ready")

    snap = session.actor_validation_snapshot()
    print(f"[CARLA] Preflight actors: {snap}", flush=True)

    for row in snap.get("actors", []):
        if row.get("status") == "actor_missing" and row.get("name") in ("ego", "lead", "rear"):
            raise AutopassConfigurationError(f"CARLA preflight: actor_missing:{row['name']}")

    issues = _validate_carla_actors(session)
    if issues:
        raise AutopassConfigurationError(
            "CARLA preflight actor validation failed:\n  - " + "\n  - ".join(issues)
        )

    from perception.carla_validation import validate_session_corridor

    corridor_issues = validate_session_corridor(session)
    if corridor_issues:
        raise AutopassConfigurationError(
            "CARLA preflight corridor validation failed:\n  - " + "\n  - ".join(corridor_issues)
        )

    report = getattr(session, "_corridor_report", None)
    if report is not None and report.ok:
        print(f"[CARLA] Preflight corridor ok: {report.summary_line()}", flush=True)

    if require_frames:
        if not session.wait_for_sensor_frames(max_ticks=20, timeout_s=5.0):
            raise AutopassConfigurationError(
                "CARLA preflight: sensors did not produce RGB/depth/seg frames.\n"
                + session.sensor_full_diagnostic()
            )
        frame = session.grab_frame()
        if frame is None:
            raise AutopassConfigurationError(
                "CARLA preflight: grab_frame returned None after warmup.\n"
                + session.sensor_full_diagnostic()
            )
        rgb, seg, depth = frame
        print(
            f"[CARLA] Preflight frames ok rgb={rgb.shape}/{rgb.dtype} "
            f"seg={seg.shape}/{seg.dtype} depth={depth.shape}/{depth.dtype}",
            flush=True,
        )
