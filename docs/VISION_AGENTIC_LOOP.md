# Vision-grounded agentic loop

## Underlying goal

Under trip deadline pressure, decide **when** an AV may overtake using **only vision-derived gaps** (segmentation + depth), with inspectable multi-step reasoning, hard safety gates, selective LLM judgment, and **closed-loop re-sensing after each actuation step**.

## Single entry points

| Command | Purpose |
|---------|---------|
| `pytest -q` | Offline proof (perception SSOT, planner agency, graph) |
| `python demo.py --mode multi_agent` | Agentic episode + traces (`runs/demo/`) |
| `python demo_carla_watch.py --hero-pass --scenario clear_safe_pass` | **Canonical CARLA closed loop** + video |
| `python -m autopass.benchmark --policies no_pass,aggressive,autopass` | Pareto time–safety comparison |

## Architecture

- **Planner** — picks the next **needed** vision tool from `needed_tools(perception_summary, world_belief)` or proposes `pass|wait|replan|abort_pass`. No default tool queue.
- **Critic** — rejects redundant tools and incomplete evidence; `check_pass_safety(dsl, …)` uses measured gaps/speeds only.
- **Executor** — CARLA `VehicleControl` or kinematic step; then **post-step depth → `world_belief`**.
- **Learning** — `mutate_from_failure` on collision/deadline miss (see `autopass/learning.py`).

## Vision SSOT

`autopass/perception_state.py` — decisions must not read `spec.lead.distance_m` or `spec.lead.speed_mps`. Spawn/orchestration may still use `ScenarioSpec` in CARLA NPC stepping.

## Tests that guard the claim

- `tests/test_perception_ssot.py` — wrong spec distances do not change pass/wait.
- `tests/test_planner_critic.py` — agency and safety gates.
- `tests/test_agentic_graph.py` — belief updates after execute.
