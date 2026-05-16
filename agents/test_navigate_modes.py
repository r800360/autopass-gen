"""
Test Navigate Modes — Jupyter Notebook Cells
=============================================
Copy each "# %% [markdown]" and "# %%" block as a separate cell.
Make sure map_server.py is running before testing Mode 1:
    python map_server.py
"""

# %% [markdown]
# # Testing Navigate Three Modes
# We test each mode by calling the `navigate` function directly with
# a crafted state dict. This isolates each mode without running the
# full graph (no interrupt, no checkpointer needed).

# %% Cell 1 — Imports
import json
import random
import numpy as np
from autopassing import (
    navigate,
    analyze_current_lane,
    AutoPassingState,
    SensorData,
    capture_multi_frame_perception,
)

# %% [markdown]
# ## Mode 1: Plan Route (first visit)
# Triggers when `navigation_plan` is empty and `passing_signal` is empty.
# Pulls the city map from the server, calls the LLM to plan a route,
# runs multi-frame perception, and initializes all trip tracking fields.
#
# **Requires:** map_server.py running on port 8100
#
# **passing_available** is now probability-based:
# - `"0"` → 0% chance → never triggers passing
# - `"low"` → 50% chance → triggers roughly half the time
# - `"high"` → 90% chance → triggers most of the time

# %% Cell 2 — Test Mode 1
mode1_state = {
    "travel_request": "Take me to the airport, I'm in a hurry",
    "starting_point": "Downtown Mall",
    "goal": "Airport",
    "aggressive_level": "high",
    "original_aggressive_level": "high",
    # Empty — triggers Mode 1
    "navigation_plan": [],
    "passing_signal": "",
    "messages": [],
}

print("=" * 60)
print("MODE 1: Plan Route")
print("=" * 60)

result1 = navigate(mode1_state)

print(f"\n--- Output fields ---")
print(f"navigation_plan: {len(result1['navigation_plan'])} waypoints")
for i, wp in enumerate(result1["navigation_plan"]):
    print(f"  [{i}] {wp}")
print(f"current_position: {result1['current_position']}")
print(f"current_waypoint_index: {result1['current_waypoint_index']}")
print(f"trip_eta: {result1['trip_eta']:.1f}s")
print(f"trip_elapsed_time: {result1['trip_elapsed_time']}")
print(f"depth_check_interval: {result1['depth_check_interval']:.2f}s")
print(f"arrived: {result1['arrived']}")
print(f"passing_available: {result1['passing_available']}")
print(f"\n--- Sensor data (from multi-frame perception) ---")
sensor = result1["sensor_data"]
print(f"  front_car_distance: {sensor['front_car_distance']:.1f}m")
print(f"  front_car_speed: {sensor['front_car_speed']} m/s")
print(f"  front_car_length: {sensor['front_car_length']} m")
print(f"  back_car_distance: {sensor['back_car_distance']:.1f}m")
print(f"  back_car_closing_rate: {sensor['back_car_closing_rate']} m/s")
print(f"  hazard_detected: {sensor['hazard_detected']}")
print(f"\n--- Message ---")
print(result1["messages"][0].content)

# %% [markdown]
# ## Mode 2: Generate Passing Plan (after passing decision)
# Triggers when `passing_signal` is `"pass"` or `"no_pass"`.
# We test both sub-cases.
#
# **Important:** Mode 2 now reads pre-computed physics from state
# (`passing_target_velocity`, `passing_acceleration`, `passing_required_time`).
# These would normally be computed by `analyze_current_lane` (Node 6).

# %% Cell 3 — Test Mode 2 with passing_signal = "pass"
# Build a state that looks like it came from the passing subgraph
# with an approved pass. We need: sensor_data, navigation_plan,
# passing_side, goal, plus the pre-computed physics fields.

# First, get a real sensor_data from multi-frame perception
perception = capture_multi_frame_perception(num_frames=5, interval_s=0.4)
depth_result = perception["depth_result"]
front_cars = [c for c in depth_result["car_distances"] if c["position"] == "front"]
front_dist = min(front_cars, key=lambda c: c["median_depth"])["median_depth"] if front_cars else 50.0

mock_sensor = SensorData(
    front_car_distance=front_dist,
    front_car_speed=perception["front_car_speed"],
    front_car_length=perception["front_car_length"],
    back_car_distance=100.0,
    back_car_closing_rate=perception["back_car_closing_rate"],
    ego_speed=15.0,
    speed_limit=30.0,
)

# Pre-computed physics (normally set by analyze_current_lane)
target_vel = min(1.5 * mock_sensor.front_car_speed, mock_sensor.speed_limit)
ego_avg = 0.5 * (mock_sensor.ego_speed + target_vel)
denom = ego_avg - mock_sensor.front_car_speed
pass_dist = front_dist + mock_sensor.front_car_length
req_time = pass_dist / denom if denom > 0 else 99.0
accel = (target_vel - mock_sensor.ego_speed) / req_time if req_time > 0 else 0.0

mode2_pass_state = {
    "travel_request": "Take me to the airport",
    "starting_point": "Downtown Mall",
    "goal": "Airport",
    "aggressive_level": "high",
    "original_aggressive_level": "high",
    "navigation_plan": [
        {"street": "Main Street", "action": "drive", "speed": 15.0},
        {"street": "Highway 5", "action": "merge", "speed": 25.0},
        {"street": "Harbor Drive", "action": "drive", "speed": 20.0},
    ],
    "current_waypoint_index": 0,
    "current_position": "Main Street",
    "sensor_data": mock_sensor.model_dump(),
    "passing_available": False,
    # This triggers Mode 2
    "passing_signal": "pass",
    "passing_side": "left",
    # Pre-computed physics from analyze_current_lane
    "passing_target_velocity": round(target_vel, 2),
    "passing_acceleration": round(accel, 2),
    "passing_required_time": round(req_time, 2),
    "messages": [],
}

print("=" * 60)
print("MODE 2a: Passing Signal = PASS")
print("=" * 60)
print(f"  Pre-computed physics:")
print(f"    target_velocity: {target_vel:.1f} m/s")
print(f"    acceleration: {accel:.2f} m/s²")
print(f"    required_time: {req_time:.1f}s")

result2a = navigate(mode2_pass_state)

print(f"\n--- Output fields ---")
print(f"passing_instructions:")
instructions = result2a["passing_instructions"]
print(f"  overtake_maneuver: {json.dumps(instructions['overtake_maneuver'], indent=4)}")
print(f"  merge_back_maneuver: {json.dumps(instructions['merge_back_maneuver'], indent=4)}")
print(f"navigation_plan (replanned): {len(result2a['navigation_plan'])} waypoints")
for i, wp in enumerate(result2a["navigation_plan"]):
    print(f"  [{i}] {wp}")
print(f"current_position: {result2a['current_position']}")
print(f"passing_available: {result2a['passing_available']}")
print(f"\n--- Message ---")
print(result2a["messages"][0].content)


# %% Cell 4 — Test Mode 2 with passing_signal = "no_pass"
mode2_nopass_state = {
    **mode2_pass_state,
    "passing_signal": "no_pass",
}

print("=" * 60)
print("MODE 2b: Passing Signal = NO PASS")
print("=" * 60)

result2b = navigate(mode2_nopass_state)

print(f"\n--- Output fields ---")
print(f"passing_instructions: {result2b['passing_instructions']}")
print(f"passing_available: {result2b['passing_available']}")
print(f"\n--- Message ---")
print(result2b["messages"][0].content)


# %% [markdown]
# ## Mode 3: Continue Driving (loop-back from executor)
# Triggers when `navigation_plan` exists and `passing_signal` is empty.
# Advances waypoints, runs multi-frame perception, and checks whether
# to trigger passing based on the depth check interval.
#
# **passing_available** uses the same probability roll as Mode 1:
# - At each depth check interval, if a front car is detected,
#   roll `random.random() < probability` to decide

# %% Cell 5 — Test Mode 3: single step
mode3_state = {
    "travel_request": "Take me to the airport",
    "starting_point": "Downtown Mall",
    "goal": "Airport",
    "aggressive_level": "high",
    "original_aggressive_level": "high",
    "navigation_plan": [
        {"street": "Main Street", "action": "drive", "speed": 15.0},
        {"street": "Oak Boulevard", "action": "turn_right", "speed": 11.0},
        {"street": "Highway 5", "action": "merge", "speed": 25.0},
        {"street": "Harbor Drive", "action": "turn_left", "speed": 20.0},
        {"street": "Harbor Drive", "action": "drive", "speed": 20.0},
    ],
    "current_waypoint_index": 0,
    "current_position": "Main Street",
    "sensor_data": mock_sensor.model_dump(),
    "passing_available": False,
    # Empty → triggers Mode 3 (plan already exists)
    "passing_signal": "",
    "passing_instructions": {"overtake_maneuver": [], "merge_back_maneuver": []},
    "trip_elapsed_time": 0.0,
    "trip_eta": 45.0,
    "depth_check_interval": 1.11,   # high urgency
    "last_depth_check_time": 0.0,
    "arrived": False,
    "messages": [],
}

print("=" * 60)
print("MODE 3: Continue Driving (single step)")
print("=" * 60)

result3 = navigate(mode3_state)

print(f"\n--- Output fields ---")
print(f"current_waypoint_index: {result3['current_waypoint_index']}")
print(f"current_position: {result3['current_position']}")
print(f"trip_elapsed_time: {result3['trip_elapsed_time']:.1f}s")
print(f"arrived: {result3['arrived']}")
print(f"passing_available: {result3['passing_available']}")
print(f"\n--- Sensor data (from multi-frame perception) ---")
s3 = result3["sensor_data"]
print(f"  front_car_distance: {s3['front_car_distance']:.1f}m")
print(f"  front_car_speed: {s3['front_car_speed']} m/s")
print(f"  front_car_length: {s3['front_car_length']} m")
print(f"  back_car_closing_rate: {s3['back_car_closing_rate']} m/s")
print(f"  hazard_detected: {s3['hazard_detected']}")
print(f"\n--- Message ---")
print(result3["messages"][0].content)


# %% Cell 6 — Test Mode 3: full driving loop until arrival
print("=" * 60)
print("MODE 3: Full Driving Loop (simulate until arrival)")
print("=" * 60)

# Start from the same state
loop_state = dict(mode3_state)
step = 0

while True:
    step += 1
    result = navigate(loop_state)

    # Print a compact summary for each step
    arrived = result.get("arrived", False)
    wp_idx = result.get("current_waypoint_index", "?")
    pos = result.get("current_position", "?")
    elapsed = result.get("trip_elapsed_time", 0)
    passing = result.get("passing_available", False)

    total_wps = len(loop_state["navigation_plan"])
    print(f"  Step {step}: waypoint {wp_idx}/{total_wps}, "
          f"pos={pos}, elapsed={elapsed:.1f}s, "
          f"passing_available={passing}, arrived={arrived}")

    if arrived:
        print(f"\n  >>> ARRIVED after {step} steps, {elapsed:.1f}s elapsed")
        break

    # Merge result back into state for next iteration
    # (simulates what the graph would do)
    loop_state.update(result)
    # Reset passing_signal so we stay in Mode 3
    loop_state["passing_signal"] = ""

    # Safety: prevent infinite loops in case of a bug
    if step > 20:
        print("  >>> Safety stop: too many steps")
        break


# %% Cell 7 — Test Mode 3: post-pass resume
print("=" * 60)
print("MODE 3: Post-Pass Resume")
print("=" * 60)
print("Simulates returning from a pass at waypoint 1, should skip ahead.\n")

post_pass_state = {
    **mode3_state,
    "current_position": "post_pass_position",   # signals we just finished a pass
    "current_waypoint_index": 1,                 # was at waypoint 1 when pass started
}

result_resume = navigate(post_pass_state)

print(f"Before: waypoint_index=1, position='post_pass_position'")
print(f"After:  waypoint_index={result_resume['current_waypoint_index']}, "
      f"position='{result_resume['current_position']}'")
print(f"  (Should have advanced by 2: +1 for post-pass skip, +1 for normal driving)")
print(f"\n--- Message ---")
print(result_resume["messages"][0].content)


# %% Cell 8 — Test Mode 3 with urgency=0 (never triggers passing)
print("=" * 60)
print("MODE 3: Urgency=0 (should never trigger passing)")
print("=" * 60)

safe_state = {
    **mode3_state,
    "aggressive_level": "0",
    "depth_check_interval": 0.0,   # 0 urgency → never check
}

result_safe = navigate(safe_state)

print(f"aggressive_level: 0")
print(f"depth_check_interval: 0.0 (never)")
print(f"passing_available: {result_safe['passing_available']}")
print(f"  (Should always be False regardless of what depth estimation sees)")
print(f"\n--- Message ---")
print(result_safe["messages"][0].content)


# %% [markdown]
# ## Test analyze_current_lane (Node 6) — Physics Checks
# This node makes the final pass/no-pass decision using:
# 1. Multi-frame perception (obstacles, distances, speeds)
# 2. Road availability (pass_distance <= available_distance)
# 3. Kinematics (target_velocity, required_time <= 5s)
#
# We use `unittest.mock.patch` to override `capture_multi_frame_perception`
# so the node uses our controlled values instead of random dummy data.

# %% Cell 9 — Helper: build a fake perception result
from unittest.mock import patch

def make_fake_perception(
    front_car_distance=30.0,
    front_car_speed=12.0,
    front_car_length=4.5,
    next_hazard_distance=150.0,
    hazard_detected=False,
    num_cars=2,
):
    """Build a dict that looks exactly like capture_multi_frame_perception output,
    but with controlled values so we can test the physics deterministically."""
    return {
        "depth_result": {
            "car_distances": [
                # Closest front car (the one we want to pass)
                {"position": "front", "median_depth": front_car_distance, "bbox": [400, 300, 600, 500]},
                # Next car ahead (used as next_hazard_distance)
                {"position": "front", "median_depth": next_hazard_distance, "bbox": [500, 350, 550, 400]},
            ][:num_cars],
        },
        "seg_result": {
            "car_masks": [None] * num_cars,
            "hazards": [{"label": "cone"}] if hazard_detected else [],
        },
        "front_car_speed": front_car_speed,
        "front_car_length": front_car_length,
        "back_car_closing_rate": 0.5,
        "hazard_detected": hazard_detected,
        "num_frames": 5,
    }


# %% Cell 10 — Test analyze_current_lane: normal scenario (passing take too much time)
print("=" * 60)
print("ANALYZE CURRENT LANE: Normal Scenario")
print("=" * 60)

node6_state = {
    "sensor_data": SensorData(
        front_car_distance=30.0,       # 30m ahead
        front_car_speed=12.0,          # front car going 12 m/s
        front_car_length=4.5,          # 4.5m long car
        back_car_distance=80.0,        # plenty of room behind
        back_car_closing_rate=0.5,     # barely closing
        ego_speed=15.0,                # we're going 15 m/s
        speed_limit=30.0,              # limit is 30 m/s
    ).model_dump(),
    "front_approval": True,
    "back_approval": True,
    "checker_result": "approved",
    "passing_side": "left",
    "messages": [],
}

# Patch perception to return our controlled values
fake_perception = make_fake_perception(
    front_car_distance=30.0,
    front_car_speed=12.0,
    front_car_length=4.5,
    next_hazard_distance=150.0,   # plenty of road
    hazard_detected=False,
)

with patch("autopassing.capture_multi_frame_perception", return_value=fake_perception):
    result_cl = analyze_current_lane(node6_state)

print(f"Result: {result_cl['current_lane_result']}")
print(f"  (Expected: approve)")
print(f"Physics:")
print(f"  target_velocity: {result_cl['passing_target_velocity']} m/s")
print(f"  acceleration: {result_cl['passing_acceleration']} m/s²")
print(f"  required_time: {result_cl['passing_required_time']}s")
print(f"\n--- Reasoning ---")
print(result_cl["messages"][0].content)


# %% Cell 11 — Test analyze_current_lane: front car too fast (ego can't overtake)
print("=" * 60)
print("ANALYZE CURRENT LANE: Front Car Too Fast")
print("=" * 60)

fast_front_state = {
    "sensor_data": SensorData(
        front_car_distance=25.0,
        front_car_speed=28.0,          # front car almost at speed limit
        front_car_length=4.5,
        back_car_distance=80.0,
        back_car_closing_rate=0.0,
        ego_speed=15.0,
        speed_limit=30.0,              # target = min(1.5*28, 30) = 30
        # ego_avg = 0.5*(15+30) = 22.5
        # denominator = 22.5 - 28 = -5.5 → FAIL (can't overtake)
    ).model_dump(),
    "front_approval": True,
    "back_approval": True,
    "checker_result": "approved",
    "passing_side": "left",
    "messages": [],
}

fake_fast = make_fake_perception(
    front_car_distance=25.0,
    front_car_speed=28.0,
    front_car_length=4.5,
    next_hazard_distance=150.0,
)

with patch("autopassing.capture_multi_frame_perception", return_value=fake_fast):
    result_fast = analyze_current_lane(fast_front_state)

print(f"Result: {result_fast['current_lane_result']}")
print(f"  (Expected: disapprove — ego avg speed < front car speed)")
print(f"\n--- Reasoning ---")
print(result_fast["messages"][0].content)


# %% Cell 12 — Test analyze_current_lane: not enough road
print("=" * 60)
print("ANALYZE CURRENT LANE: Not Enough Road")
print("=" * 60)

short_road_state = {
    "sensor_data": SensorData(
        front_car_distance=30.0,
        front_car_speed=10.0,
        front_car_length=5.0,
        back_car_distance=80.0,
        back_car_closing_rate=0.0,
        ego_speed=15.0,
        speed_limit=30.0,
        next_front_car_distance=40.0,  # next hazard only 40m away!
        # pass_distance = 30 + 5 = 35m
        # available_distance = 40 - 20 = 20m
        # 35 > 20 → FAIL (not enough road)
    ).model_dump(),
    "front_approval": True,
    "back_approval": True,
    "checker_result": "approved",
    "passing_side": "left",
    "messages": [],
}

fake_short = make_fake_perception(
    front_car_distance=30.0,
    front_car_speed=10.0,
    front_car_length=5.0,
    next_hazard_distance=40.0,    # very close next hazard
)

with patch("autopassing.capture_multi_frame_perception", return_value=fake_short):
    result_short = analyze_current_lane(short_road_state)

print(f"Result: {result_short['current_lane_result']}")
print(f"  (Expected: disapprove — pass_distance > available_distance)")
print(f"\n--- Reasoning ---")
print(result_short["messages"][0].content)


# %% Cell 13 — Test analyze_current_lane: hazard detected
print("=" * 60)
print("ANALYZE CURRENT LANE: Hazard Detected")
print("=" * 60)

hazard_state = {
    "sensor_data": SensorData(
        front_car_distance=30.0,
        front_car_speed=10.0,
        front_car_length=4.5,
        back_car_distance=80.0,
        back_car_closing_rate=0.0,
        ego_speed=15.0,
        speed_limit=30.0,
    ).model_dump(),
    "front_approval": True,
    "back_approval": True,
    "checker_result": "approved",
    "passing_side": "left",
    "messages": [],
}

fake_hazard = make_fake_perception(
    front_car_distance=30.0,
    front_car_speed=10.0,
    front_car_length=4.5,
    hazard_detected=True,   # obstacle in path!
)

with patch("autopassing.capture_multi_frame_perception", return_value=fake_hazard):
    result_hazard = analyze_current_lane(hazard_state)

print(f"Result: {result_hazard['current_lane_result']}")
print(f"  (Expected: disapprove — hazard/obstacle detected)")
print(f"\n--- Reasoning ---")
print(result_hazard["messages"][0].content)
