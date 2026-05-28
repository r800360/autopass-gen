# CARLA control tuning

## What “CARLA control tuning” means

The **agentic** layer (planner / critic / DSL) decides **when** to pass, wait, or replan.

The **executor** (`perception/carla_control.py`) decides **how** to actuate in the simulator between graph steps: throttle, brake, steer, lane target.

**Tuning** = the numeric parameters in that executor. They are **not** learned weights and **not** the LLM. They behave like cruise-control and lane-keep gains in a real stack.

## Architecture (agentic improvement, not RL)

```
Planner → Critic approves pass → Execute (ACC + phased lane change)
                                      ↓
                              min_front_gap, near_miss, pass_phase
                                      ↓
                              Critic post-check → DSL revision / replan
                                      ↓
                              world_belief from depth → next Planner cycle
```

This is **iterative improvement in the agentic sense**: bad execution updates `verification_log`, may increment `revision`, and the planner re-senses with fresh `world_belief`. It is **not** reinforcement learning (no policy gradient, no reward model training).

RL would be a separate research track; for the course demo, **closed-loop replan from measured gaps** is the right story for Chandraker.

## Pass phases (executor state machine)

| Phase | Behavior |
|-------|----------|
| `approach` | Stay in travel lane; ACC slows if too close to lead |
| `lane_change` | Steer toward left passing lane (only if gap ≥ `PASS_LATERAL_MIN_M`) |
| `overtake` | Stay in passing lane until clear of lead |
| `merge` | Merge right only when lead is **behind** ego along road (3D check) |
| `cruise` | Lane-keep + ACC on wait/replan |

This fixes the “steer left then immediately steer right into the lead” bug: merge-back was tied to logical `passed`, not physical clearance.

## Tunable parameters (environment variables)

| Variable | Default | Effect |
|----------|---------|--------|
| `AUTOPASS_CARLA_SAFE_FOLLOW_M` | 14 | ACC comfort gap behind lead |
| `AUTOPASS_CARLA_CRITICAL_GAP_M` | 5.5 | Hard brake threshold |
| `AUTOPASS_CARLA_PASS_LATERAL_MIN_M` | 16 | Min gap before starting lane change |
| `AUTOPASS_CARLA_MAX_STEER` | 0.18 | Steering cap (lower = gentler) |
| `AUTOPASS_CARLA_STEER_GAIN` | 60 | Lane tracking gain (higher = gentler) |
| `AUTOPASS_CARLA_NEAR_MISS_M` | 7 | Critic replan if min gap during step drops below |
| `AUTOPASS_CARLA_MERGE_CLEAR_M` | 8 | Longitudinal clearance before merge-back |

### Autonomous tuning (preferred over manual env vars)

The executor learns knobs from **pass_quality** scores (lane center p95 per phase, merge completed, corridor exit):

```powershell
python -m perception.carla_pass_tune --trials 8
python -m perception.carla_pass_smoke
```

`tune` hill-climbs `max_steer`, lateral gains, merge horizon, etc. and writes `.autopass/carla_control_profile.json`. Smoke loads that profile automatically. The critic also nudges the profile when `pass_quality` fails during agentic runs.

Manual overrides remain available but are a fallback:

```powershell
$env:AUTOPASS_CARLA_USE_PROFILE = "0"
$env:AUTOPASS_CARLA_MAX_STEER = "0.16"
```

## What to look for in CARLA / trace

- **`EXECUTE` + `wait`**: ego should follow lead without rear-ending (ACC braking).
- **`EXECUTE` + `pass`**: approach in-lane first, then one smooth lane change, overtake, merge.
- **Trace / DSL** after execute:
  - `execution_log[-1].data.pass_phase`
  - `min_front_gap_m`, `near_miss`, `clear_of_lead`
- **Replan**: `dsl.revision` increments when critic sees `near_miss` or collision.

## Limits (honest)

- NPCs are kinematic on waypoints, not Traffic Manager — good for reproducible demos, not full urban realism.
- One execute step ≈ 1 s of motion; a full pass takes **multiple** agent cycles (by design).
- Perfect driving on all 18 maps would need map-specific tuning or a full local planner — out of scope for a 252D agentic stack demo.

The project vision stays: **vision-grounded, urgency-aware, critic-gated overtaking with a living DSL** — with a **safe enough** CARLA executor to make the loop credible on video.
