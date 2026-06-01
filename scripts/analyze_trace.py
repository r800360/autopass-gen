"""Quick trace summarizer for carla_watch runs."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/carla_watch/demo_07_clear_safe_pass_perception_agentic_trace.json")
t = json.loads(p.read_text())
print("entries:", len(t))
nodes: dict[str, int] = {}
for e in t:
    n = e.get("node", e.get("tool", "?"))
    nodes[n] = nodes.get(n, 0) + 1
print("by node:", nodes)
execs = [e for e in t if e.get("node") == "execute"]
print("executes:", len(execs))
for i, e in enumerate(execs):
    fb = e.get("feedback") or e
    ac = fb.get("actor_continuity", {})
    print(
        f"exec{i} action={e.get('action', fb.get('action'))} "
        f"step={fb.get('episode_step')} "
        f"dt={fb.get('duration_s')} "
        f"ticks={fb.get('control_ticks')} "
        f"head={fb.get('heading_error_deg')} "
        f"lane_center={fb.get('lane_center_error_m')} "
        f"lat_steer={fb.get('lateral_error_m')} "
        f"max_lane={fb.get('max_lane_center_dist_m')} "
        f"prog={fb.get('progress_delta_m')} "
        f"ego_spd={fb.get('ego_speed_mps')} "
        f"viol={ac.get('continuity_violations')} "
        f"ego_d={ac.get('delta_ego_world_m')} "
        f"lead_d={ac.get('delta_lead_world_m')} "
        f"fsm={fb.get('pass_fsm_phase')} "
        f"d_pass={fb.get('lateral_offset_passing_m')} "
        f"shift={fb.get('lateral_shift_toward_passing_m')}"
    )
print("\n--- tools with lead restore/reset ---")
for e in t:
    if e.get("node") != "tool":
        continue
    ac = e.get("actor_continuity_after") or {}
    if ac.get("restore_lead_called") or ac.get("any_actor_transform_reset"):
        print(
            e.get("tool"),
            "reason=",
            ac.get("actor_transform_reset_reason"),
            "restore=",
            ac.get("restore_lead_called"),
            "lead_world_d=",
            ac.get("delta_lead_world_m"),
        )
