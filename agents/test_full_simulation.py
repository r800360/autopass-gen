"""
Full End-to-End Simulation — Jupyter Notebook Cells
====================================================
Simulates the ego vehicle visiting Navigate three times
(Mode 1 → Mode 2 → Mode 3), calling every node in the graph
manually with mocked perception data.

Scenario:
  1. Mode 1: Plan route, detect slow front car → passing_available=True
  2. Passing subgraph: front ✓, back ✓, checker ✓, current lane ✓ → pass
  3. Mode 2: Generate passing instructions, replan route
  4. Executor: execute pass, reset fields, loop back
  5. Mode 3: Continue driving, detect another front car, passing_available=True
  6. Passing subgraph: front ✓, back ✓, checker ✓, current lane ✗ (not enough road) → no_pass
  7. Mode 2: Handle no_pass, continue on original plan

Copy each "# %%" block as a separate Jupyter cell.
Make sure map_server.py is running: python map_server.py
"""

# %% Cell 1 — Imports
import json
import time
import random
import numpy as np
from unittest.mock import patch
from autopassing import (
    navigate,
    check_passing_lane_front,
    check_passing_lane_back,
    checker,
    analyze_current_lane,
    send_passing_signal,
    carla_executor,
    farewell,
    route_after_navigation,
    route_after_checker,
    route_after_executor,
    AutoPassingState,
    SensorData,
)


# %% Cell 2 — Mock builders
def make_fake_depth_result(
    front_car_distance=35.0,
    passing_lane_front_distance=120.0,
    passing_lane_front_side="front_left",
    passing_lane_rear_distance=None,
    passing_lane_rear_side="rear_left",
):
    """Build a fake run_depth_estimation return value.
    Used by check_passing_lane_front and check_passing_lane_back."""
    cars = [
        {
            "bbox": [565, 310, 715, 410],
            "median_depth": front_car_distance,
            "min_depth": front_car_distance - 5,
            "position": "front",
        },
    ]
    # Car in passing lane ahead (for front checker)
    if passing_lane_front_distance is not None:
        cars.append({
            "bbox": [260, 320, 380, 400],
            "median_depth": passing_lane_front_distance,
            "min_depth": passing_lane_front_distance - 5,
            "position": passing_lane_front_side,
        })
    # Car in passing lane behind (for back checker)
    if passing_lane_rear_distance is not None:
        cars.append({
            "bbox": [260, 420, 380, 500],
            "median_depth": passing_lane_rear_distance,
            "min_depth": passing_lane_rear_distance - 5,
            "position": passing_lane_rear_side,
        })
    return {
        "depth_map": np.zeros((720, 1280), dtype=np.float32),
        "min_depth": 5.0,
        "max_depth": 200.0,
        "car_distances": cars,
    }


def make_fake_perception(
    front_car_distance=35.0,
    front_car_speed=10.0,
    front_car_length=4.5,
    next_hazard_distance=150.0,
    hazard_detected=False,
    num_front_cars=2,
):
    """Build a fake capture_multi_frame_perception return value.
    Used by Navigate (Modes 1 and 3) and analyze_current_lane."""
    cars = []
    if num_front_cars >= 1:
        cars.append({"position": "front", "median_depth": front_car_distance, "bbox": [400, 300, 600, 500]})
    if num_front_cars >= 2:
        cars.append({"position": "front", "median_depth": next_hazard_distance, "bbox": [500, 350, 550, 400]})
    return {
        "depth_result": {"car_distances": cars},
        "seg_result": {
            "car_masks": [None] * num_front_cars,
            "hazards": [{"label": "cone"}] if hazard_detected else [],
        },
        "front_car_speed": front_car_speed,
        "front_car_length": front_car_length,
        "back_car_closing_rate": 0.5,
        "hazard_detected": hazard_detected,
        "num_frames": 5,
    }


def print_step(step_num, title, msg_content):
    """Pretty-print a step in the simulation."""
    print(f"\n{'='*70}")
    print(f"  STEP {step_num}: {title}")
    print(f"{'='*70}")
    print(msg_content)


# %% [markdown]
# # Full Simulation
# We maintain a single `state` dict and update it after each node call,
# exactly as LangGraph would. Each step prints the node's output message
# and key state changes.

# %% Cell 3 — Initialize state (travel request already extracted)
# Skip extract_travel_request since it needs user interrupt.
# We start with a pre-extracted request.
state = {
    "travel_request": "Take me to the airport, I'm in a hurry",
    "starting_point": "Downtown Mall",
    "goal": "Airport",
    "aggressive_level": "high",
    "original_aggressive_level": "high",
    # Empty — triggers Mode 1
    "navigation_plan": [],
    "passing_signal": "",
    "passing_instructions": {"overtake_maneuver": [], "merge_back_maneuver": []},
    "passing_available": False,
    "front_approval": False,
    "back_approval": False,
    "passing_side": "",
    "checker_result": "",
    "current_lane_result": "",
    "passing_target_velocity": 0.0,
    "passing_acceleration": 0.0,
    "passing_required_time": 0.0,
    "consecutive_disapprovals": 0,
    "arrived": False,
    "messages": [],
    "sensor_data": SensorData(
    front_car_distance=20.0,
    front_car_speed=10.0,
    front_car_length=4.5,
    back_car_distance=90.0,
    back_car_closing_rate=0.5,
    ego_speed=20.0,
    speed_limit=30.0,
    next_front_car_distance=150.0,
    ).model_dump(),
}

print("Initial state created.")
print(f"  Trip: {state['starting_point']} → {state['goal']}")
print(f"  Aggressive level: {state['aggressive_level']}")
print(f"  Navigation plan: {'empty' if not state['navigation_plan'] else 'exists'}")


# %% Cell 4 — STEP 1: Navigate Mode 1 (Plan Route)
# Perception: slow front car at 35m, speed 10 m/s
# Force passing_available=True by seeding random so probability check passes
fake_perception_pass1 = make_fake_perception(
    front_car_distance=20.0,
    front_car_speed=10.0,
    front_car_length=4.5,
    next_hazard_distance=150.0,
    hazard_detected=False,
)

# Seed random so the probability roll (0.9 for "high") passes
random.seed(42)  # random.random() with seed 42 gives 0.639... < 0.9 → True

with patch("autopassing.capture_multi_frame_perception", return_value=fake_perception_mode1):
    result1 = navigate(state)

state.update(result1)

print_step(1, "NAVIGATE MODE 1 — Plan Route", result1["messages"][0].content)
print(f"\n  Key outputs:")
print(f"    navigation_plan: {len(state['navigation_plan'])} waypoints")
for i, wp in enumerate(state["navigation_plan"]):
    print(f"      [{i}] {wp}")
print(f"    passing_available: {state['passing_available']}")
print(f"    depth_check_interval: {state.get('depth_check_interval', 'N/A')}")
print(f"    trip_eta: {state.get('trip_eta', 'N/A')}")

# Check routing
route = route_after_navigation(state)
print(f"\n  Routing decision: → {route}")
assert state["passing_available"] == True, "Expected passing_available=True for this test!"
assert route == "check_passing", "Expected route to passing checks!"


# %% Cell 5 — STEP 2: Passing Subgraph (FIRST PASS — should APPROVE)
# 2a: check_passing_lane_front
#   Left side: car at 120m > 50m threshold, trend constant → APPROVE left
fake_depth_pass1 = make_fake_depth_result(
    front_car_distance=35.0,
    passing_lane_front_distance=200.0,   # matches sensor from Mode 1 (no passing lane car in perception)
    passing_lane_front_side="front_left",
    passing_lane_rear_distance=200.0,    # matches sensor (no rear passing lane car in perception)
    passing_lane_rear_side="rear_left",
)

with patch("autopassing.run_depth_estimation", return_value=fake_depth_pass1):
    result_front = check_passing_lane_front(state)

state.update(result_front)
print_step("2a", "CHECK PASSING LANE FRONT", result_front["messages"][0].content)
print(f"    front_approval: {state['front_approval']}")
print(f"    passing_side: {state['passing_side']}")

# 2b: check_passing_lane_back
#   Left side: car at 90m > 30m threshold, closing rate low → APPROVE
with patch("autopassing.run_depth_estimation", return_value=fake_depth_pass1):
    result_back = check_passing_lane_back(state)

state.update(result_back)
print_step("2b", "CHECK PASSING LANE BACK", result_back["messages"][0].content)
print(f"    back_approval: {state['back_approval']}")

# 2c: Checker
result_checker = checker(state)
state.update(result_checker)
print_step("2c", "CHECKER", result_checker["messages"][0].content)
print(f"    checker_result: {state['checker_result']}")

route_ck = route_after_checker(state)
print(f"    Routing: → {route_ck}")
assert state["checker_result"] == "approved", "Expected checker approved!"
assert route_ck == "analyze_current_lane"

# 2d: analyze_current_lane
#   To make this APPROVE, we need a faster ego or shorter pass_distance.
#   Let's use: front_car_distance=20m, ego_speed=20, front_car_speed=10
#   target_velocity = min(15, 30) = 15, ego_avg = 0.5*(20+10) = 15
#   denom = 17.5 - 10 = 7.5, pass_dist = 20+4.5 = 24.5
#   required_time = 24.5 / 7.5 = 3.27s < 5s ✓

with patch("autopassing.capture_multi_frame_perception", return_value=fake_perception_pass1):
    result_cl = analyze_current_lane(state)

state.update(result_cl)
print_step("2d", "ANALYZE CURRENT LANE", result_cl["messages"][0].content)
print(f"    current_lane_result: {state['current_lane_result']}")
print(f"    target_velocity: {state['passing_target_velocity']} m/s")
print(f"    acceleration: {state['passing_acceleration']} m/s²")
print(f"    required_time: {state['passing_required_time']}s")

assert state["current_lane_result"] == "approve", (
    f"Expected approve but got {state['current_lane_result']}. "
    f"Check the physics: the scenario should be passable."
)

# 2e: send_passing_signal
result_signal = send_passing_signal(state)
state.update(result_signal)
print_step("2e", "SEND PASSING SIGNAL", result_signal["messages"][0].content)
print(f"    passing_signal: {state['passing_signal']}")
assert state["passing_signal"] == "pass"


# %% Cell 6 — STEP 3: Navigate Mode 2 (Generate Passing Plan)
# passing_signal="pass" triggers Mode 2
# Navigate reads the pre-computed physics from state and asks
# the LLM to plan the overtake maneuver + replan route.
with patch("autopassing.capture_multi_frame_perception", return_value=fake_perception_pass1):
    result2 = navigate(state)

state.update(result2)

print_step(3, "NAVIGATE MODE 2 — Generate Passing Plan", result2["messages"][0].content)
print(f"\n  Key outputs:")
print(f"    passing_instructions:")
inst = state["passing_instructions"]
print(f"      overtake: {json.dumps(inst['overtake_maneuver'], indent=6)}")
print(f"      merge_back: {json.dumps(inst['merge_back_maneuver'], indent=6)}")
print(f"    replanned navigation: {len(state['navigation_plan'])} waypoints")
for i, wp in enumerate(state["navigation_plan"]):
    print(f"      [{i}] {wp}")

route2 = route_after_navigation(state)
print(f"\n  Routing decision: → {route2}")
# After Mode 2, passing_available is reset to False → goes to executor
assert route2 == "carla_executor", "Expected route to executor after Mode 2!"


# %% Cell 7 — STEP 4: Carla Executor (Execute Pass, Reset, Loop Back)
result_exec = carla_executor(state)
state.update(result_exec)

print_step(4, "CARLA EXECUTOR — Execute & Reset", result_exec["messages"][0].content)
print(f"\n  Reset fields:")
print(f"    passing_signal: '{state['passing_signal']}' (empty → Mode 3 next)")
print(f"    passing_instructions: {state['passing_instructions']}")
print(f"    front_approval: {state['front_approval']}")
print(f"    back_approval: {state['back_approval']}")
print(f"    arrived: {state.get('arrived', False)}")

route_exec = route_after_executor(state)
print(f"\n  Routing decision: → {route_exec}")
assert route_exec == "navigate", "Expected loop back to Navigate!"


# %% Cell 8 — STEP 5: Navigate Mode 3 (Continue Driving — Second Front Car)
# State has: navigation_plan (exists), passing_signal="" → Mode 3
# This time we detect another front car but the next hazard is very close
# (not enough road to pass).
#
# We need passing_available=True so it enters the passing subgraph again.
# Force the probability roll to succeed.
random.seed(42)

fake_perception_mode3 = make_fake_perception(
    front_car_distance=25.0,     # new front car at 25m
    front_car_speed=12.0,        # going 12 m/s
    front_car_length=5.0,        # slightly longer car
    next_hazard_distance=45.0,   # next hazard very close! (not enough road)
)

with patch("autopassing.capture_multi_frame_perception", return_value=fake_perception_mode3):
    result3 = navigate(state)

state.update(result3)

print_step(5, "NAVIGATE MODE 3 — Continue Driving", result3["messages"][0].content)
print(f"\n  Key outputs:")
print(f"    current_waypoint_index: {state.get('current_waypoint_index')}")
print(f"    current_position: {state.get('current_position')}")
print(f"    trip_elapsed_time: {state.get('trip_elapsed_time', 0):.1f}s")
print(f"    passing_available: {state['passing_available']}")
s3 = state["sensor_data"]
print(f"    sensor: front_car={s3['front_car_distance']:.1f}m, "
      f"speed={s3['front_car_speed']}m/s, length={s3['front_car_length']}m")

route3 = route_after_navigation(state)
print(f"\n  Routing decision: → {route3}")
assert state["passing_available"] == True, (
    "Expected passing_available=True! The probability roll may have failed. "
    "Try a different random seed."
)
assert route3 == "check_passing", "Expected route to passing checks!"


# %% Cell 9 — STEP 6: Passing Subgraph (SECOND PASS — should FAIL at current lane)
# Front and back checkers will approve (passing lane is clear),
# but analyze_current_lane will reject because not enough road.
#
# pass_distance = 25 + 5 = 30m
# available_distance = 45 - 20 = 25m
# 30 > 25 → NOT ENOUGH ROAD → disapprove

fake_depth_pass2 = make_fake_depth_result(
    front_car_distance=25.0,
    passing_lane_front_distance=200.0,   # matches sensor (no passing lane car seen in Mode 3)
    passing_lane_front_side="front_left",
    passing_lane_rear_distance=200.0,    # matches sensor (no rear passing lane car seen)
    passing_lane_rear_side="rear_left",
)

# 6a: Front checker
with patch("autopassing.run_depth_estimation", return_value=fake_depth_pass2):
    result_front2 = check_passing_lane_front(state)
state.update(result_front2)
print_step("6a", "CHECK PASSING LANE FRONT", result_front2["messages"][0].content)
print(f"    front_approval: {state['front_approval']}")
print(f"    passing_side: {state['passing_side']}")

# 6b: Back checker
with patch("autopassing.run_depth_estimation", return_value=fake_depth_pass2):
    result_back2 = check_passing_lane_back(state)
state.update(result_back2)
print_step("6b", "CHECK PASSING LANE BACK", result_back2["messages"][0].content)
print(f"    back_approval: {state['back_approval']}")

# 6c: Checker
result_checker2 = checker(state)
state.update(result_checker2)
print_step("6c", "CHECKER", result_checker2["messages"][0].content)
print(f"    checker_result: {state['checker_result']}")

route_ck2 = route_after_checker(state)
print(f"    Routing: → {route_ck2}")

# 6d: analyze_current_lane — THIS SHOULD FAIL (not enough road)
# Update sensor_data to match the scenario
state["sensor_data"] = SensorData(
    front_car_distance=25.0,
    front_car_speed=12.0,
    front_car_length=5.0,
    back_car_distance=80.0,
    back_car_closing_rate=0.5,
    ego_speed=15.0,
    speed_limit=30.0,
    next_front_car_distance=45.0,   # next hazard very close!
).model_dump()

fake_perception_pass2 = make_fake_perception(
    front_car_distance=25.0,
    front_car_speed=12.0,
    front_car_length=5.0,
    next_hazard_distance=45.0,   # only 45m to next hazard
    hazard_detected=False,
)

with patch("autopassing.capture_multi_frame_perception", return_value=fake_perception_pass2):
    result_cl2 = analyze_current_lane(state)

state.update(result_cl2)
print_step("6d", "ANALYZE CURRENT LANE — NOT ENOUGH ROAD", result_cl2["messages"][0].content)
print(f"    current_lane_result: {state['current_lane_result']}")
print(f"\n    Expected: disapprove (pass_distance=30m > available_distance=25m)")

assert state["current_lane_result"] == "disapprove", (
    f"Expected disapprove but got {state['current_lane_result']}!"
)

# 6e: send_passing_signal → no_pass
result_signal2 = send_passing_signal(state)
state.update(result_signal2)
print_step("6e", "SEND PASSING SIGNAL", result_signal2["messages"][0].content)
print(f"    passing_signal: {state['passing_signal']}")
assert state["passing_signal"] == "no_pass"


# %% Cell 10 — STEP 7: Navigate Mode 2 (Handle No-Pass)
# passing_signal="no_pass" triggers Mode 2 again, but this time
# it just acknowledges and continues on the current plan.
with patch("autopassing.capture_multi_frame_perception", return_value=fake_perception_mode3):
    result_nopass = navigate(state)

state.update(result_nopass)

print_step(7, "NAVIGATE MODE 2 — Handle No-Pass", result_nopass["messages"][0].content)
print(f"\n  Key outputs:")
print(f"    passing_available: {state['passing_available']}")
print(f"    passing_instructions: {state['passing_instructions']}")

route_final = route_after_navigation(state)
print(f"\n  Routing decision: → {route_final}")
assert route_final == "carla_executor", "Expected route to executor after no_pass!"


# %% Cell 11 — STEP 8: Executor after No-Pass
result_exec2 = carla_executor(state)
state.update(result_exec2)

print_step(8, "CARLA EXECUTOR — After No-Pass", result_exec2["messages"][0].content)
print(f"\n  Reset fields:")
print(f"    passing_signal: '{state['passing_signal']}'")
print(f"    arrived: {state.get('arrived', False)}")

route_exec2 = route_after_executor(state)
print(f"\n  Routing decision: → {route_exec2}")
assert route_exec2 == "navigate", "Expected loop back to Navigate!"


# %% Cell 12 — STEP 9: Mode 3 Driving Loop Until Arrival
# Now we loop: Navigate Mode 3 → Executor → Navigate Mode 3 → ...
# until the vehicle arrives at the destination.
#
# We use a perception with NO front car so passing never triggers,
# letting the vehicle drive through remaining waypoints cleanly.

print(f"\n{'='*70}")
print(f"  STEP 9+: DRIVING LOOP (Mode 3 → Executor → ... → END)")
print(f"{'='*70}")
print(f"  Starting at waypoint {state.get('current_waypoint_index', '?')}")
print(f"  Plan has {len(state.get('navigation_plan', []))} waypoints total\n")

# No front car → passing never triggers, clean driving
fake_perception_clear = make_fake_perception(
    front_car_distance=200.0,    # no front car nearby
    front_car_speed=0.0,
    front_car_length=0.0,
    next_hazard_distance=500.0,
    num_front_cars=0,            # empty road
)

step_counter = 9
loop_iteration = 0
MAX_ITERATIONS = 30  # safety limit

while loop_iteration < MAX_ITERATIONS:
    loop_iteration += 1

    # --- Navigate Mode 3 ---
    with patch("autopassing.capture_multi_frame_perception", return_value=fake_perception_clear):
        result_nav = navigate(state)
    state.update(result_nav)

    arrived = state.get("arrived", False)
    wp_idx = state.get("current_waypoint_index", "?")
    pos = state.get("current_position", "?")
    elapsed = state.get("trip_elapsed_time", 0)
    passing = state.get("passing_available", False)
    total_wps = len(state.get("navigation_plan", []))

    print(f"  Loop {loop_iteration}: Navigate Mode 3 → "
          f"waypoint {wp_idx}/{total_wps}, pos={pos}, "
          f"elapsed={elapsed:.1f}s, passing={passing}")

    if arrived:
        print(f"\n  >>> ARRIVED at {state['goal']}! (elapsed={elapsed:.1f}s)")
        route_end = route_after_executor(state)
        print(f"  >>> route_after_executor → {route_end}")

        # Call farewell node (final node before END)
        result_farewell = farewell(state)
        state.update(result_farewell)
        print(f"\n  >>> {result_farewell['messages'][0].content}")
        break

    # Route check
    route_loop = route_after_navigation(state)
    if route_loop == "check_passing":
        # Shouldn't happen with empty road, but handle gracefully
        print(f"    (passing triggered unexpectedly, skipping for clean test)")
        state["passing_available"] = False

    # --- Executor ---
    result_exec_loop = carla_executor(state)
    state.update(result_exec_loop)

    route_exec_loop = route_after_executor(state)
    if route_exec_loop == "end":
        print(f"\n  >>> Executor says END (arrived={state.get('arrived', False)})")
        break

else:
    print(f"\n  >>> Safety stop after {MAX_ITERATIONS} iterations")
