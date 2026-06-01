# AutoPass-Gen — System Analysis (First Principles)

This document reconstructs what the codebase is trying to do, whether it is agentic or deterministic, why it is failing its underlying goal today, and strategic recommendations to achieve a research claim that distinguishes it from mainstream autonomous driving work.

---

## 1. Underlying goal (the real claim)

**Surface goal:** Decide when a ride-hailing AV may overtake a slow lead vehicle under trip deadline pressure.

**Underlying goal (what would be unique):** Close the loop on **vision-grounded overtaking as an inspectable multi-agent decision process** — where every pass/wait/replan is justified by **measured pixel evidence** (segmentation + depth), re-sensed after each actuation step, with hard safety gates and selective LLM judgment only where geometry is insufficient.

That is **not** “train an end-to-end policy in CARLA.” It is:

> **Separate measurement (code + vision) from judgment (LLM), orchestrate both in a replanning graph, and prove the stack can pass safely under urgency using only what cameras see — then act in physics.**

The north-star metric (from `docs/VISION_AGENTIC_LOOP.md`):

`route_completed AND pass_attempt_success AND NOT collision`

on hero corridors, with trace fields explaining *why* each decision was made.

**Why this matters vs. prior AD research:** Most stacks either (a) use privileged simulator state for gaps, (b) learn a monolithic policy without auditable pass gates, or (c) treat planning as a single optimizer. AutoPass-Gen’s claim is **hybrid agency**: LangGraph orchestration + vision SSOT + deterministic safety authority + closed-loop belief refresh.

---

## 2. Problem decomposition (first principles)

### 2.1 The decision problem

Overtaking under deadline pressure is a **sequential, partially observable** problem:

| Observable (vision) | Latent / contextual |
|---------------------|---------------------|
| Front gap in travel lane | Whether lead is “slow enough” to bother |
| Rear gap / closing rate in passing lane | Traffic density interpretation |
| Oncoming gap in opposing lane | Urgency vs. conservative wait |
| Road length / kinematics feasibility | Target pass speed |

A fixed rule (“pass if front gap > 30 m”) fails because urgency, rear closing, oncoming, and kinematics interact. A pure LLM fails because distances must be **measured**, not hallucinated.

### 2.2 How the code solves it

```
Mission + deadline
       ↓
Planner (LLM or rules) — picks NEXT vision tool OR maneuver proposal
       ↓
Tool executor — capture_sensors, measure_*_gap, check_kinematics, assess_traffic
       ↓
Critic — verifies tool output; blocks redundant/insufficient evidence
       ↓
Critic (maneuver) — pass_gates + check_pass_safety on world_belief only
       ↓
Executor — CARLA VehicleControl (prod) or kinematic step (test)
       ↓
observe_post_step — depth/seg → world_belief
       ↓
(replan loop or evaluate)
```

**Key modules:**

| Layer | Files | Role |
|-------|-------|------|
| Orchestration | `autopass/graph.py` | LangGraph loop |
| Agency | `autopass/planner.py` | Tool/maneuver selection |
| SSOT | `autopass/perception_state.py`, `autopass/dsl.py` | Decisions read `world_belief`, not spawn distances |
| Safety authority | `autopass/critic.py`, `autopass/pass_gates.py`, `autopass/safety.py` | Deterministic veto |
| Actuation | `autopass/executor.py`, `perception/carla_control.py`, `perception/pass_control_fsm.py` | Physics step |
| Simulation | `perception/carla_scenario.py` | Spawn, sync tick, sensors, NPC kinematics |
| Perception | `perception/pipeline.py`, `perception/carla_labels.py` | RGB/seg/depth → gaps |

---

## 3. Is the system agentic?

**Yes, in architecture — but hybrid, not fully autonomous.**

| Component | Agentic? | Mechanism |
|-----------|----------|-----------|
| Planner tool order | **Yes** (prod) | LLM chooses from `needed_tools()`; no default queue |
| Maneuver pass/wait/replan | **Partially** | LLM proposes; `_clamp_pass_decision`, critic, pass_gates override |
| Critic / safety | **No** | Deterministic rules on measured gaps |
| Executor / control FSM | **No** | Pure pursuit of approved maneuver |
| Perception | **No** | Seg + depth geometry |
| `assess_traffic`, target speed | **Hybrid** | LLM + measured gaps |

**Test mode (`AUTOPASS_TEST_MODE=1`):** Rule planner (`AUTOPASS_MOCK_LLM=1`) — graph structure preserved, agency simulated.

**Hero demo (`autopass/hero_demo.py`):** Scripted pass FSM with *synthetic* trace entries — **not** the full agentic loop. Do not use as evidence of agency.

**Verdict:** The system is **agentic at the planning/orchestration layer** with a **deterministic safety shell**. That is intentional and defensible for a research claim about *inspectable* overtaking — not about emergent end-to-end behavior.

---

## 4. Is the system deterministic?

**Mixed — by design.**

| Mode | Deterministic? |
|------|----------------|
| Offline tests | Largely yes (kinematic world, mock LLM) |
| Pass gates, critic, safety | Yes |
| CARLA sync + fixed Δt=0.05 | Physics/sensors repeatable *if* spawn and tick order fixed |
| Production LLM (T=0.4) | **No** — stochastic tool/maneuver choice |
| NPC kinematic stepping | Yes (scripted speeds from ScenarioSpec) |
| Corridor scan / spawn index | Can vary if map geometry or scan order changes |

**CARLA determinism requirements** (per [CARLA synchrony docs](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/)): synchronous mode + fixed delta + consistent tick cadence + reload for exact replay. The codebase enables sync/fixed delta in `CarlaScenarioSession._apply_world_sync_settings()`.

**Verdict:** **Decision logic is intended to be reproducible given the same belief state.** End-to-end episodes are **not** fully deterministic in production due to LLM sampling and spawn/corridor variability.

---

## 5. Why it is failing the underlying goal today

The architecture is sound; **integration gaps** block the closed-loop claim:

### 5.1 Vision → action loop not reliably closed in CARLA

- **Spawn fragility:** Dual spawn systems (waypoint chain vs. world-axis offset), stale `(0,0,0)` actor locations before tick, `wp.next` jumping to parallel roads → wrong gaps, failed pass gates, or spawn collisions.
- **Perception scale bug (fixed):** Pipeline used 1280 px focal length while CARLA cameras are 640×256 → wrong car length / speed estimates.
- **Burst vs. layout:** Multi-frame capture moves NPCs then restores lead — axis spawn path needed to avoid gap collapse.

### 5.2 Control vs. decision still partially decoupled

- `hero_demo.py` runs **scripted** pass maneuvers, not planner-driven execute steps.
- When spawn fails or belief is empty, fallbacks (`wait`, oracle gaps if enabled) mask perception failure instead of proving vision works.

### 5.3 Evidence chain breaks under real simulator conditions

Pass gates require `front_gap_ok`, `rear_gap_ok`, `oncoming_ok`, `kinematics_ok`, `evidence_ok`. Any of these fail when:

- Corridor has no validated passing lane
- Lead gap < 26 m on axis hero scenario
- Seg/depth does not associate detections to lead/rear actors
- Ego departs travel lane during control (failure taxonomy: `control_lane_departure`)

### 5.4 Research claim vs. implementation mismatch

| Claim | Current gap |
|-------|-------------|
| Vision-only decisions | SSOT enforced in code/tests; CARLA spawn distances still orchestrate NPC layout |
| Closed-loop re-sensing | Implemented in `belief.observe_post_step`; undermined if execute doesn't advance physics reliably |
| Multi-agent inspectability | Trace + DSL exist; hero video may show scripted control |
| Beats naive baselines | Benchmark harness exists; CARLA rows fail when bootstrap fails |

---

## 6. Strategic recommendations (priority order)

### P0 — Make CARLA the standard path (spawn + sync + sensors)

1. **Single spawn authority:** `carla_scenario.py` only; corridor scan → axis layout for hero; `try_spawn_actor` with forward retries (CARLA API).
2. **Sync discipline:** `synchronous_mode=True`, `fixed_delta_seconds=0.05`, physics substepping; one client ticks; disable sync on shutdown.
3. **Sensor contract:** Persistent `listen()` callbacks; warmup via tick loop; RGB + depth + semantic at 640×256 FOV 90°; focal length from actual width.
4. **Pre-flight gate:** `python -m perception.carla_sensor_smoke` and `carla_corridor_smoke --diagnose` before any demo/benchmark.

### P1 — Close the agentic loop in CARLA (not scripted hero)

1. Run **`demo_carla_watch.py --hero-pass`** / full `run_agentic_episode` with `AUTOPASS_CONTROL_MODE=vehicle` — every execute calls `execute_vehicle_step`, then `observe_post_step`.
2. Remove synthetic trace from hero path or label it clearly as **control-only** ablation.
3. Metric dashboard: `failure_taxonomy` per scenario × urgency on every CARLA run.

### P2 — Harden vision SSOT under sim noise

1. Actor association (`carla_actor_association.py`) + gap calibration (`carla_gap_calibrate.py`) as required steps before pass gates flip.
2. Never enable `AUTOPASS_DECISION_ORACLE` in production demos (axis gap fallback defeats the claim).
3. Add regression tests: wrong `spec.lead.distance_m` must not change pass decision (`test_perception_ssot.py` pattern).

### P3 — Research differentiation (what to write / measure)

1. **Pareto frontier:** `python -m autopass.benchmark --policies no_pass,aggressive,autopass` — time vs. collision vs. missed safe pass.
2. **Agency ablation:** LLM planner vs. rule planner vs. aggressive baseline on **same** vision belief.
3. **Failure taxonomy paper figure:** Show *why* passes fail (perception vs. control vs. gates) — unique vs. end-to-end AD papers.

### P4 — What NOT to add (preserves claim)

- **No LiDAR/radar in the decision path** — would contradict vision-grounded SSOT unless explicitly labeled as ablation.
- **No Traffic Manager random traffic** for hero scenarios — breaks reproducibility.
- **No privileged gap injection** in planner/critic paths.

---

## 7. CARLA API alignment (implemented / standard patterns)

| Pattern | Location | Notes |
|---------|----------|-------|
| `Client` + `set_timeout` | `bootstrap()` | 15 s timeout |
| `load_world` / reuse map | `bootstrap()` | Avoid reload when same Town |
| Sync + fixed Δt + substepping | `_apply_world_sync_settings()` | Per CARLA docs |
| `try_spawn_actor` + retry | `_spawn_one()` | Collision-safe spawn |
| `set_simulate_physics(False)` on NPCs | `_spawn_one()` | Kinematic convoy |
| Ego physics enable dance | `enable_ego_physics()` | Disable→zero→enable |
| Sensor BPs + `listen()` | `_attach_sensors()` | Persistent callbacks |
| `world.tick()` in sync mode | `tick()`, warmup | Client-driven stepping |
| Async release on shutdown | `_release_world_sync_settings()` | Clean teardown |

**Sensors used:** RGB, depth, semantic segmentation (Cityscapes IDs). Overhead RGB for recording only. No LiDAR/radar in decision loop — intentional for vision claim.

**Legacy bridge:** `perception/carla_bridge.py` now delegates to `CarlaScenarioSession` instead of spawning a duplicate ego without sync.

---

## 8. Commands to validate progress

```powershell
# Offline proof (217 tests)
pytest -q

# CARLA health (simulator must be running)
python -m perception.carla_sensor_smoke
python -m perception.carla_corridor_smoke --diagnose

# Full agentic CARLA closed loop
python demo_carla_watch.py --hero-pass --scenario clear_safe_pass
```

---

## 9. Summary table

| Question | Answer |
|----------|--------|
| What problem? | Urgent, safe, vision-grounded overtaking with inspectable reasoning |
| How? | LangGraph + vision tools + deterministic critic/gates + CARLA execute + belief refresh |
| Agentic? | **Hybrid** — LLM orchestrates; safety and control are deterministic |
| Deterministic? | **Partial** — gates/control yes; LLM and spawn variability no |
| Why failing? | Spawn/perception/control integration, not the graph design |
| Unique claim? | Auditable multi-agent pass pipeline on **pixels only**, under deadline pressure |
| Next milestone? | Reliable CARLA hero pass via full graph loop, not scripted FSM |
