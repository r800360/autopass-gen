# Vision-grounded agentic loop

## Underlying goal

Under trip deadline pressure, decide **when** an AV may overtake using **only vision-derived gaps** (segmentation + depth), with inspectable multi-step reasoning, hard safety gates, selective LLM judgment, and **closed-loop re-sensing after each actuation step**.

## Single entry points

| Command | Purpose |
|---------|---------|
| `pytest -q` | Offline proof (perception SSOT, planner agency, graph) |
| `python demo.py --mode multi_agent` | Agentic episode + traces (`runs/demo/`) |
| `python demo_carla_watch.py --hero-pass --scenario clear_safe_pass` | **Canonical CARLA closed loop** + video |
| `python scripts/run_carla_scenario_gallery.py --fast --demos 0,4,6,7` | Multi-scenario CARLA videos + `gallery_summary.json` |
| `python -m autopass.benchmark --policies no_pass,aggressive,autopass` | Pareto time–safety comparison |

## Architecture

- **Planner** — picks the next **needed** vision tool from `needed_tools(perception_summary, world_belief)` or proposes `pass|wait|replan|abort_pass`. No default tool queue.
- **Critic** — rejects redundant tools and incomplete evidence; `check_pass_safety(dsl, …)` uses measured gaps/speeds only.
- **Executor** — CARLA `VehicleControl` or kinematic step; then **post-step depth → `world_belief`**.
- **Learning** — `mutate_from_failure` on collision/deadline miss (see `autopass/learning.py`).

## Vision SSOT

`autopass/perception_state.py` — decisions must not read `spec.lead.distance_m` or `spec.lead.speed_mps`. Spawn/orchestration may still use `ScenarioSpec` in CARLA NPC stepping.

## Tests that guard the claim

- `pytest -q` — 199+ offline tests (SSOT, planner/critic, pass control lane targets).
- `tests/test_perception_ssot.py` — wrong spec distances do not change pass/wait.
- `tests/test_planner_critic.py` — agency and safety gates.
- `tests/test_agentic_graph.py` — belief updates after execute.
- `tests/test_pass_control_lane_target.py` — pass steers toward passing lane, not travel lane.

## Production defaults

| Variable | Production default | Purpose |
|----------|-------------------|---------|
| `AUTOPASS_MOCK_LLM` | `0` | Real OpenAI planner |
| `AUTOPASS_DECISION_ORACLE` | `0` | No simulator axis gap fallbacks |
| `AUTOPASS_LLM_TEMPERATURE` | `0.4` | Non-deterministic agency |
| `AUTOPASS_PERCEPTION_BACKEND` | `carla` | Live sensors |
| `AUTOPASS_CONTROL_MODE` | `vehicle` | CARLA VehicleControl |

Control gains (`MAX_STEER`, lane-change blend) ship with **demo-safe defaults in code** — you should not need to tune env vars for hero videos. Optional `AUTOPASS_CARLA_*` overrides exist for ablations only.

Trace field `metrics.agency` reports LLM rounds, vision-front steps, and critic/execute alignment.

## Demo video (inspectable claim on frame)

`demo_carla_watch.py --hero-pass` burns in each frame:

- **Depth boxes** on ego RGB (green LEAD, cyan rear, orange oncoming)
- **Belief panel**: `CAN_PASS`, vision front/rear gaps, urgency pressure, pass FSM phase, `oracle=OFF`, agency source

Disable boxes with `vision_overlay: false` in recorder `extra` (off by default outside hero demo).

## North-star metric (CARLA hero corridors)

On hero scenarios (`demo_07_clear_safe_pass_perception`, etc.):

`route_completed AND pass_attempt_success AND NOT collision`

Report per urgency; trace field `failure_taxonomy` classifies episodes (`control_lane_departure`, `control_incomplete`, `perception_insufficient`, …).

## Pass execution (control)

`perception/pass_control_fsm.py` + `perception/carla_control.py` run a bounded pass FSM:

- **Wait / follow:** `target_lane_id == travel_lane_id`
- **Pass / lane change:** `target_lane_id == passing_lane_id` (adjacent lane from spawn topology)
- **Abort** on multi-lane departure or planner `wait` during early lane change (no silent forced pass)
