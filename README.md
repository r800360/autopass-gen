# AutoPass-Gen

A live, vision-grounded **agentic** system for urgency-aware overtaking decisions in
[CARLA](https://carla.org) 0.9.16. Every half second an **LLM planner** chooses which perception or
verification tool to call next, a **deterministic critic** verifies any proposed pass against hard
safety gates and freshly-sensed evidence, and a **waypoint controller** that steers only to real
CARLA lane centers executes the maneuver, so the vehicle is lane-compliant by construction. When the
critic rejects a proposal the agent remembers why, re-senses, and re-plans, producing emergent
behaviors like yielding to a fast follower and then overtaking once it clears.

The LLM owns **process** (what to look at, when to propose) while deterministic code owns **safety** (the
critic) and **motion** (the controller). The LLM can never approve its own pass or steer the car off
its lane.

## How it works

- **Tools (planner action space):** `sense_front`, `sense_passing_lane`, `sense_rear` (segmentation +
  metric depth → lane-associated gaps), `check_corridor` (road-graph lookahead), `propose_pass`,
  `hold`. The number and order of tool calls vary per scene which shows *agentic* control over process.
- **Hard safety gates (relative warrant):** front gap ≥ 18 m; the lead must be at least **4 m/s slower
  than the ego's desired cruise speed** (road-relative, not a fixed cutoff); passing-lane rear ≥ 12 m;
  oncoming ≥ 45 m (two-lane); passing lane clear ≥ 24 m. Gates can only *block* a pass, never force one.
- **Lane compliance by construction:** steering always targets a real lane-center waypoint from the
  CARLA map graph, so the ego cannot drift off-road or cross multiple lanes (measured max deviation ≤ 0.6 m).

## Key files

| Path | Role |
|---|---|
| `perception/clean_overtake.py` | Controller, perception, hard gates, recording, benchmark metrics |
| `perception/overtake_agent.py` | LLM planner, deterministic critic, mutable `ScenarioDSL` |
| `agents/llm_agents.py` | `structured_invoke` (GPT-4o-mini, typed Pydantic schemas) |
| `scripts/run_overtake.py` | 23-scenario catalog + single-run CLI |
| `scripts/run_benchmark.py` | 3-policy benchmark (never-pass / autopass / always-pass) |
| `scripts/carla_agent_bridge.py`, `scripts/carla_agent_exec.py` | Localhost bridge into the CARLA process |

## Setup

- CARLA 0.9.16 server running locally.
- `pip install -r requirements.txt`
- Provide an OpenAI key for the live agent: set `OPENAI_API_KEY`, or put it in a gitignored
  `.openai_key` file at the repo root.

Live runs are dispatched through the bridge (`carla_agent_exec.py`), since the agent shell cannot open
a CARLA RPC socket directly.

## Run

Single scenario (live agent, with video):
```bash
python scripts/carla_agent_exec.py --timeout 900 -- \
  "<venv>/python.exe" scripts/run_overtake.py --scenario s01_t04_highway_safe_pass --out-dir runs/demo
```
Full 23-scenario campaign: add `--all`. Use `--mock` for fast control-only iteration (no LLM),
`--hires` for 720p.

Three-policy benchmark (one town per job for CARLA stability):
```bash
python scripts/carla_agent_exec.py --timeout 3000 -- \
  "<venv>/python.exe" scripts/run_benchmark.py --town t04 --policy autopass --out-dir runs/bench
python scripts/run_benchmark.py --aggregate-only --out-dir runs/bench   # → benchmark_summary.json
```

Each run writes a two-panel video (ego RGB + HUD beside bird's-eye), the raw `frames/`, a per-cycle
`trace.json` (tool sequence, planner rationales, critic verdict, DSL snapshot), and a `result.json`
of honest metrics.

## Results

23-scenario live campaign across four towns: **23/23 correct, 0 collisions, 0 off-road, ≤ 0.6 m lane
deviation.** Controlled three-policy benchmark (`runs/benchmark_live_v2`, 69 live runs):

| Policy | Overtakes | Collisions | Unsafe | Unwarranted | Speed (m/s) |
|---|---|---|---|---|---|
| Never-pass | 0 / 18 | 0 / 23 | 0 | 0 | 3.9 |
| **AutoPass-Gen (ours)** | **18 / 18** | **0 / 23** | **0** | **0** | **9.7** |
| Always-pass | 18 / 18 | 2 / 23 | 2 | 3 | 9.7 |

AutoPass-Gen completes every warranted overtake with zero collisions; always-pass collides on the
hazard scenarios and makes unwarranted passes; never-pass crawls at a third of the speed.

## License

MIT - see [LICENSE](LICENSE). Rohan Sachdeva, Xinwei Mai, Pranav Prabu, Rathang Pandit (UC San Diego).
