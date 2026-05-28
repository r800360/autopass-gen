# Runtime configuration (production defaults)

Production entry points (`demo.py`, `demo_carla_watch.py`) use:

| Variable | Production default | Test mode (`pytest`) |
|----------|-------------------|----------------------|
| `AUTOPASS_MOCK_LLM` | `0` (real OpenAI) | `1` |
| `AUTOPASS_PERCEPTION_BACKEND` | `carla` | `visual` |
| `AUTOPASS_CONTROL_MODE` | `vehicle` | `kinematic` |
| `AUTOPASS_TEST_MODE` | unset | `1` (set by `conftest.py`) |

## Fail-fast

Missing configuration raises `AutopassConfigurationError` with a fix hint:

- No `OPENAI_API_KEY` when mock LLM is off
- `import carla` fails (wrong Python / missing wheel)
- CARLA server not reachable on `CARLA_HOST:CARLA_PORT` (default `127.0.0.1:2000`)

## Closed-loop belief

After each `execute` step, ego cameras capture depth → `PassingDSL.world_belief` (`front_gap_m`, `rear_gap_m`, `oncoming_gap_m`). Planner tools read `world_belief` before re-querying sensors.

## Production run

```powershell
$env:OPENAI_API_KEY = "sk-..."
# Start C:\CARLA_0.9.16\CarlaUE4.exe
.\.venv\Scripts\activate
python demo_carla_watch.py --environment highway --scenario 0
```

## Offline tests

```powershell
pytest -q   # AUTOPASS_TEST_MODE=1 automatically
```
