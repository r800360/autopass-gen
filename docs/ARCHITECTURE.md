# AutoPass-Gen Architecture (agentic)

## One system, one story

All Python entry points (`demo.py`, `autopass_langgraph_demo.py`, `agents/autopassing.py`, `demo_carla_watch.py`) use the same **agentic** LangGraph in `autopass/graph.py`.

**Production defaults** (see `autopass/config.py`, `docs/LLM_AND_CONTROL.md`):

- Real OpenAI LLM (`AUTOPASS_MOCK_LLM=0`)
- CARLA perception (`AUTOPASS_PERCEPTION_BACKEND=carla`)
- Vehicle physics control (`AUTOPASS_CONTROL_MODE=vehicle`)
- Fail-fast `AutopassConfigurationError` if API key or CARLA server is missing

## Agents (course definition)

| Agent | Role |
|-------|------|
| **Planner** | Chooses the *next* vision tool from `needed_tools()` + `world_belief`, or `pass` / `wait` / `replan` / `abort_pass`. **No default tool queue.** |
| **Tool executor** | Runs one planner-selected tool; appends results to the **DSL** `perception_log`. |
| **Critic** | External verification after each tool and after execution; triggers **DSL revision** on replan. |
| **Executor** | CARLA `VehicleControl` (production) or kinematic step (test mode); updates **`world_belief`** from post-step depth. |

## DSL (`autopass/dsl.py`)

`PassingDSL` is updated every cycle:

- `mission`, `route`, `maneuver`
- `perception_log` — one entry per tool call
- `execution_log` — CARLA / kinematic feedback
- `verification_log` — critic notes
- `world_belief` — **post-step measured gaps** (`front_gap_m`, `rear_gap_m`, `oncoming_gap_m`)
- `tools_completed` — tools run this cycle (replan clears; no prefilled queue)
- `revision` — increments on replan

## Closed-loop belief (`autopass/belief.py`)

After each `execute` node:

1. Ego cameras capture depth (CARLA or visual test renderer)
2. `observe_post_step` → `PassingDSL.world_belief`
3. Next planner tools read `belief_gaps()` before stale `perception_log` entries

## Vision tools (`autopass/tools.py`)

High-level tools the planner may invoke:

- `capture_sensors` — multi-frame RGB / seg / depth burst
- `measure_front_gap`, `measure_rear_gap`, `measure_oncoming` — prefer `world_belief` when fresh
- `check_kinematics`, `assess_traffic`

## Physical validation (`autopass/physics.py`)

- Logical 1D layout: no lead/rear overlap, oncoming ahead in passing lane
- CARLA session: actor separation, oncoming faces ego, driving lanes
- `assert_physical_or_raise` after each execute step (production)

## CARLA

- **Maps:** `highway` → Town04, `town` → Town03, `local` → Town01 (`autopass/scenarios.py`)
- **18 scenarios:** 6 demos × 3 environments (`all_comprehensive_scenarios()`)
- **Video:** ego camera + overhead camera, composited in `perception/carla_recorder.py`
- **Ego control:** `perception/carla_control.py` (throttle/steer/brake); feedback in `DSL.execution_log`
- **NPC traffic:** kinematic along waypoints from `WorldState` (reproducible scenarios)

## Commands

```powershell
.\.venv\Scripts\activate
pytest -q                                    # offline (AUTOPASS_TEST_MODE=1)
$env:OPENAI_API_KEY = "sk-..."
# Start C:\CARLA_0.9.16\CarlaUE4.exe
python demo_carla_watch.py --environment highway --scenario 0
python demo.py --carla --mode multi_agent
```
