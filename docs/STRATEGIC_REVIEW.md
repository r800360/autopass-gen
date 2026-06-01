# AutoPass-Gen — Strategic Review (May 2026)

This document reconstructs the system from first principles, assesses agenticity and determinism, explains why the underlying goal is not yet achieved in CARLA, and recommends engineering changes to **prove the research claim faster**.

---

## 1. Underlying goal (north star)

> **Under trip deadline pressure, decide when an AV may overtake a slow lead vehicle using only vision-derived gaps (segmentation + depth), with inspectable multi-step reasoning, hard safety gates, and closed-loop re-planning after each actuation step.**

**Success criterion (CARLA hero corridors):**

`route_completed AND pass_attempt_success AND NOT collision`

…with trace fields explaining every pass / wait / reject, and `metrics.agency` showing vision-grounded belief (not simulator oracle gaps).

**What would be distinctive vs. mainstream AD research:**

| Typical AD research | AutoPass-Gen claim |
|---------------------|-------------------|
| End-to-end imitation / RL | Explicit **decision layer** (plan → sense → verify → act) |
| Fixed TTC / never-pass stacks | **Urgency-conditioned** pass vs wait |
| Privileged simulator state | **Pixel-grounded** front / rear / oncoming gaps |
| Monolithic planner | **Shared DSL** + tool use + critic replan |
| Opaque neural policy | **Traces** for every pass / wait / reject |

Every line of code should defend this table. Anything that bypasses vision for decisions (axis-gap oracle, spawn-distance gates, scripted pass without replan) **weakens the claim**.

---

## 2. Problem reconstruction (first principles)

### 2.1 The decision problem

Overtaking under deadline pressure is **sequential and partially observable**:

| Observable (vision) | Contextual (judgment) |
|---------------------|-------------------------|
| Front gap in travel lane | Is lead slow enough to bother? |
| Rear gap / closing in passing lane | Traffic density interpretation |
| Oncoming gap (when topology requires it) | Urgency vs conservative wait |
| Road length / kinematics | Target pass speed, replan need |

A fixed rule fails because urgency, rear closing, oncoming, and kinematics interact. A pure LLM fails because distances must be **measured**, not hallucinated.

### 2.2 Intended solution architecture

```
Mission + deadline → urgency
        ↓
Planner (LLM or rules) — next vision tool OR pass|wait|replan|abort_pass
        ↓
Tool executor — capture_sensors, measure_*_gap, check_kinematics, assess_traffic
        ↓
Critic (tools) — reject redundant / insufficient evidence
        ↓
Critic (maneuver) — pass_gates + check_pass_safety on world_belief ONLY
        ↓
Executor — CARLA VehicleControl (prod) or kinematic step (test)
        ↓
observe_post_step — depth/seg → world_belief
        ↓
(replan loop or evaluate)
```

**Module map:**

| Layer | Primary files | Role |
|-------|---------------|------|
| Orchestration | `autopass/graph.py` | LangGraph closed loop |
| Agency | `autopass/planner.py`, `agents/llm_agents.py` | Tool / maneuver selection |
| SSOT | `autopass/perception_state.py`, `autopass/dsl.py` | Decisions read `world_belief`, not `spec.lead.distance_m` |
| Safety authority | `autopass/critic.py`, `autopass/pass_gates.py`, `autopass/safety.py` | Deterministic veto |
| Actuation | `autopass/executor.py`, `perception/carla_control.py`, `perception/pass_control_fsm.py` | Physics step |
| Simulation | `perception/carla_scenario.py`, `visual_world.py` | CARLA + synthetic renderer |
| Perception | `perception/pipeline.py`, `perception/carla_labels.py`, `perception/carla_actor_association.py` | RGB/seg/depth → gaps |

**Single graph:** All entry points (`demo.py`, `demo_carla_watch.py`, `agents/autopassing.py`, `autopass/hero_demo.py`) converge on `autopass/graph.py`. There is no separate “paper pipeline” anymore — good for claim coherence.

---

## 3. Is the system agentic?

**Verdict: hybrid agentic by design — not fully autonomous, and that is defensible.**

| Component | Agentic? | Mechanism |
|-----------|----------|-----------|
| Planner tool order | **Yes** (prod) | LLM picks from `needed_tools()`; graph enforces required order |
| Maneuver pass/wait/replan | **Partially** | LLM proposes; `_clamp_pass_decision`, critic, pass_gates override |
| Critic / safety / pass_gates | **No** | Deterministic rules on measured gaps |
| Executor / pass FSM | **No** | Pure pursuit of approved maneuver |
| Perception | **No** | Seg + depth geometry |
| `assess_traffic`, target speed | **Hybrid** | LLM + measured gaps |

**Test mode (`AUTOPASS_TEST_MODE=1`):** Rule planner simulates agency; graph topology preserved. **216/217 pytest cases pass** — the **decision architecture is proven offline**.

**What “agentic” means for the paper:** The LLM is the **orchestrator** of evidence gathering and contextual judgment; code owns **measurement and veto**. That is the research claim — not emergent end-to-end driving.

---

## 4. Is the system deterministic?

**Verdict: intentionally mixed.**

| Subsystem | Deterministic? |
|-----------|----------------|
| pass_gates, critic, safety | Yes, given same `world_belief` + summary |
| Rule planner (mock LLM) | Yes |
| Production LLM (T=0.4) | No — stochastic tool/maneuver choice |
| CARLA sync + fixed Δt=0.05 | Repeatable *if* spawn and tick order fixed |
| Corridor scan / spawn index | Can vary with map geometry |
| Offline kinematic world | Largely yes |

**Implication:** Reproducibility for demos should use **frozen spawn + mock LLM or T=0** for the video; agency ablation uses **same belief, different planner** (strongest controlled experiment).

---

## 5. Why the underlying goal is not achieved today

The **architecture matches the claim**. Failure is **integration and proof surface**, not graph design.

### 5.1 Root cause ranking

| Priority | Blocker | Symptom |
|----------|---------|---------|
| **P0** | CARLA bootstrap / spawn fragility | Empty belief, wrong gaps, spawn collisions, corridor scan misses passing lane |
| **P0** | Vision → gate chain breaks in sim | `front_valid=false`, wrong lead association, seg on 640×256, passing-lane rear wins front cone |
| **P1** | Dual-world sync (`WorldState` ↔ CARLA) | Logical progress vs physics diverge; NPC kinematic stepping vs ego physics |
| **P1** | Control complexity | Lane departure, incomplete pass FSM, `pass_attempt_failed_control` |
| **P2** | SSOT leaks | Axis-gap fallback in `carla_actor_association.py`; critic accepts axis gap when speed missing; `AUTOPASS_DECISION_ORACLE` in test |
| **P2** | Engineering surface area | ~3k lines in `carla_scenario.py` alone; days spent on spawn not on claim |
| **P3** | Stale docs | `docs/PROJECT_STORY.md` still says CARLA does not use VehicleControl — outdated |

### 5.2 The closed loop (what must work once)

```text
capture_sensors → world_belief.front_gap_m validated
→ pass_gates.can_pass == true (under high urgency)
→ critic approves pass
→ execute_vehicle_step (pass FSM completes)
→ observe_post_step refreshes belief
→ route_completed, no collision
```

Any break in this chain produces `wait_success`, `perception_insufficient`, or `control_lane_departure` — which is what you are seeing in CARLA runs.

### 5.3 What offline tests already prove

- Vision SSOT: wrong `spec.lead.distance_m` does not change pass/wait (`test_perception_ssot.py`)
- Planner agency and gate clamping (`test_planner_critic.py`, `test_agentic_graph.py`)
- Pass control targets passing lane (`test_pass_control_lane_target.py`)
- Graph belief refresh after execute (`test_agentic_graph.py`)

**The research stack works in synthetic space.** CARLA is the missing **validation layer**, not the missing **design**.

---

## 6. CARLA: ideal or wrong tool?

### 6.1 Why CARLA fits the claim

- Open-source **RGB + semantic segmentation + depth** on ego camera
- **VehicleControl** API for closed-loop actuation
- Maps with multi-lane highways (Town04) for overtaking topology
- Already integrated; course investment sunk

### 6.2 Why CARLA is blocking you

- **Spawn and corridor logic** dominate engineering time (`carla_scenario.py`, axis spawn, curated corridor scan)
- **Non-determinism** without frozen layout
- **Windows/Python 3.10** wheel friction
- **Semantic quality** and **640×256** resolution make association hard
- **No native “overtake benchmark”** — you built one from scratch

### 6.3 Alternatives (honest comparison)

| Simulator | Fit for claim | Speed to demo | Notes |
|-----------|---------------|---------------|-------|
| **visual_world (in-repo)** | Proves decision + traces; weak “on road” | **Fastest** | Already 216 tests green |
| **CARLA (frozen corridor)** | Full pixel + physics claim | Medium | Keep — but freeze one scene |
| **MetaDrive** | RL-friendly, weaker seg/depth story | Fast loops | Harder to defend “seg+depth only” |
| **Isaac Sim** | High fidelity | Slow setup | Overkill for course timeline |
| **nuPlan** | Planning metrics | No low-level pass control | Wrong layer |
| **Real vehicle / rosbag** | Ultimate claim | Not feasible now | Future work |

### 6.4 Recommendation

**Do not abandon CARLA** — it is the right long-term validator for the pixel-grounded claim.

**Do decouple proof from CARLA for the next 72 hours:**

1. **Claim proof layer** = `visual_world` + LangGraph traces + benchmark Pareto (`autopass/benchmark.py`)
2. **Physics validation layer** = one **hardcoded** Town04 corridor (no scan), rule planner first, then LLM

Trying to prove the full claim only through live CARLA corridor discovery is why progress feels zero despite a working graph.

---

## 7. Engineering choices to prove the claim faster

### 7.1 Shrink the integration surface (highest ROI)

| Change | Effort | Impact on claim |
|--------|--------|-----------------|
| **Hardcode hero spawn** (transform JSON for Town04 passing lane) | 1 day | Eliminates corridor scan / axis spawn bugs |
| **Single demo command** only: `demo_carla_watch.py --hero-pass` | hours | Stops path confusion |
| **Preflight gate**: no demo unless `carla_sensor_smoke` + `carla_corridor_smoke --diagnose` pass | hours | Fail fast, not mid-episode |
| **Remove axis-gap from production belief path**; keep in diagnostics only | 1 day | SSOT integrity |
| **Trace burn-in on video** (gates, belief, maneuver on HUD) | 1–2 days | Best advisor/demo artifact |
| **Rule planner CARLA pass first** (`AUTOPASS_MOCK_LLM=1`), then LLM | hours | Separates control from agency bugs |

### 7.2 Two-track development

```text
Track A — Proof (synthetic):  demo.py --mode multi_agent + trace JSON + side-by-side panels
Track B — Validation (CARLA):   frozen spawn → green smoke → one successful hero pass on video
```

Merge tracks only after Track B step 1 (spawn + perception stable) is green.

### 7.3 What NOT to build (preserves uniqueness)

- No LiDAR/radar in decision path (ablation only)
- No Traffic Manager random traffic on hero scenarios
- No privileged gap injection in planner/critic (`AUTOPASS_DECISION_ORACLE=0` in all demos)
- No scripted pass FSM bypassing planner/critic (hero_demo already uses full graph — keep it that way)

### 7.4 Demo package that “shows off” the claim

Minimum credible demo set:

1. **Side-by-side video**: RGB | seg | depth with gap annotations
2. **Trace timeline**: planner → tool → critic → execute with `pass_preconditions` / `pass_blockers`
3. **Urgency ablation**: same scenario, low urgency → wait; high urgency + gates → pass
4. **Policy Pareto**: `no_pass` vs `aggressive` vs `autopass` on benchmark CSV
5. **Agency ablation**: rule planner vs LLM on **identical** `world_belief` snapshots (offline replay)

Items 1–4 can ship from **visual_world** before CARLA is perfect. Item 5 is the strongest “we are agentic but safe” figure.

---

## 8. Strategic roadmap (ordered)

### Phase 0 — Today (unblock perception)

- Run smokes: `python -m perception.carla_sensor_smoke`, `python -m perception.carla_corridor_smoke --diagnose`
- Fix the 1 failing control test (`test_carla_control_logic.py`) — control semantics drift
- Confirm `AUTOPASS_DECISION_ORACLE=0`, focal length uses actual camera width (`pipeline.camera_focal_px`)

### Phase 1 — Proof without CARLA (2–3 days)

- Record `demo.py --mode multi_agent` for `clear_safe_pass` × low/high urgency
- Generate trace overlay panel in `carla_recorder` or post-process script
- Run `python -m autopass.benchmark --policies no_pass,aggressive,autopass` in test mode → Pareto figure for paper

### Phase 2 — Frozen CARLA hero (3–5 days)

- Replace corridor scan for hero with **fixed transforms** on Town04
- `AUTOPASS_MOCK_LLM=1` until one clean pass video
- Metric: `failure_taxonomy == none` or `pass_attempt_success`

### Phase 3 — Full claim (1 week)

- `AUTOPASS_MOCK_LLM=0` on same frozen scene
- Report `metrics.agency`: `execute_vision_front_steps` > 0, `execute_axis_front_steps` == 0
- North-star on hero: route + pass + no collision

### Phase 4 — Differentiation (paper)

- Failure taxonomy figure (why passes fail — unique vs end-to-end AD)
- Agency ablation on frozen belief log
- Urgency-conditioned pass rate vs baselines

---

## 9. Summary answers

| Question | Answer |
|----------|--------|
| What problem? | Urgent, safe, vision-grounded overtaking with inspectable reasoning |
| How? | LangGraph + vision tools + deterministic critic/gates + execute + belief refresh |
| Agentic? | **Hybrid** — LLM orchestrates; safety and control are deterministic |
| Deterministic? | **Partial** — gates yes; LLM and spawn variability no |
| Why failing in CARLA? | Spawn/perception/control integration, not graph design |
| Is architecture right? | **Yes** — 216 offline tests support it |
| Is CARLA ideal? | **Yes for final claim**, **no as the only proof path** |
| Fastest path to demo? | Synthetic traces + frozen CARLA corridor + HUD burn-in |
| Unique claim achievable? | **Yes**, if SSOT holds in CARLA and one hero pass closes the loop on video |

---

## 10. Commands

```powershell
# Offline proof (216+ tests)
py -3.10 -m pytest -q

# Synthetic agentic demo + traces
py -3.10 demo.py --mode multi_agent

# CARLA health (simulator must be running)
py -3.10 -m perception.carla_sensor_smoke
py -3.10 -m perception.carla_corridor_smoke --diagnose

# Canonical CARLA closed loop (production defaults)
py -3.10 demo_carla_watch.py --hero-pass --scenario clear_safe_pass --policy autopass --urgency high

# Benchmark Pareto
py -3.10 -m autopass.benchmark --policies no_pass,aggressive,autopass
```
