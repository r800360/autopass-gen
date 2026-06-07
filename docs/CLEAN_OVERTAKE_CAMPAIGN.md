# Clean Overtake Campaign — vision-grounded agentic passing in CARLA

This is the current, supported path for generating presentation-grade overtake videos. It
replaces the older `visual_campaign` / `carla_control.build_vehicle_control` path, whose
custom "axis/corridor" steering drove the ego off-road across maps.

## What it does

Under trip-deadline pressure, the ego decides when to overtake a slow lead using **only
vision-derived gaps** (semantic segmentation + metric depth), with an **LLM judgment** that is
**hard-gated by safety checks**, and a **waypoint-based controller** that is lane-compliant by
construction. Works on any CARLA town because it reads the real road graph.

## Files

| Area | File |
|------|------|
| Driver (controller + perception + recording, the safety floor) | `perception/clean_overtake.py` |
| Agentic layer (planner + critic + mutable DSL + memory) | `perception/overtake_agent.py` |
| Scenario catalog + CLI | `scripts/run_overtake.py` |
| Review montage (6 frames -> 1 image) | `scripts/make_montage.py` |
| Presentation reel (all clips + title cards) | `scripts/build_reel.py` |

## How it works (the research claim, made true)

1. **Perception (pixel-grounded).** Front + rear RGB/seg/depth cameras. `carla_seg_to_car_distances`
   gives vehicle masks + median depth. Each detection is lane-associated with a **monocular lateral
   offset** `lateral_m = 2*depth*(cx/W - 0.5)` (fov=90): |lat|<2.6 m = our lane (the lead),
   one lane over toward the passing side = adjacent/oncoming. This separates the lead from
   roadside / oncoming / parked vehicles far better than raw pixel-x.
2. **Hard safety gates** (`evaluate_gates`): front gap >= 18 m, lead slow (<= 9 m/s), passing-lane
   rear gap >= 12 m, oncoming >= 45 m (two-lane only), passing lane clear ahead. These can only
   *block* a pass; they never force one.
3. **Agentic decision layer** (`overtake_agent.py`, `AUTOPASS_AGENTIC=1`, default on). This is the
   "real claim": agency over *process*, not just wording. It is **not** a fixed pipeline.
   - **Planner (LLM, tool choice).** Each 0.5 s deliberation cycle, a real LLM
     (`structured_invoke`, gpt-4o-mini, temp 0.4 — non-deterministic) chooses the *next tool* to
     call from `{sense_front, sense_passing_lane, sense_rear, check_corridor, propose_pass, hold}`.
     The number and order of tool calls **vary per scene** (see any `trace.json`).
   - **Mutable DSL** (`ScenarioDSL`): a live belief + plan + memory state, updated iteratively as
     tools run. It tracks each belief's freshness, accumulates **memory** (critic denials, rejection
     reasons, full tool history), and is revision-stamped — not just an input scenario.
   - **Critic (deterministic verifier).** `propose_pass` is checked by a separate critic that
     verifies the hard gates **and** that the gating evidence was freshly consulted this cycle and
     the corridor geometry is verified. The LLM **cannot approve its own action**; a rejection is
     remembered and forces a **re-sense / replan** next cycle (this is what produces the
     yield-to-rear-traffic-then-overtake and wait-for-oncoming-then-overtake behaviors).
   - **Greedy under urgency.** The deterministic critic is the safety authority: when it approves on
     fresh evidence under high/medium urgency, the agent takes the pass (missing a safe pass is a
     failure in this project's thesis). Safety is always the critic's; only the *process* is the
     LLM's.
   - The full per-cycle tool sequence, critic verdict, denial count and DSL revision are written to
     the trace **and drawn on the video HUD** (`PLAN: ... > ...`, `CRITIC: ...`).
4. **Control (lane-compliant by construction, the reflexive floor).** Pure-pursuit steering toward a
   real CARLA lane-center waypoint; FSM phases follow -> lane_change -> overtake -> merge_back ->
   done. ACC holds a steady gap behind the lead so the front-gap gate can open without rear-ending.
   Two-lane passes use the opposing lane (lookahead via `.previous()`). The overtake -> merge_back
   transition is **vision-triggered**: the rear camera detects the just-overtaken lead behind the
   ego (with a sim-position safety fallback so the ego can never overtake forever). `merge_trigger`
   in `result.json` records which fired.
5. **Corridor finder** (`find_corridor`): reads the road graph for a straight stretch where the
   travel AND passing lanes stay parallel/driving/non-junction for the whole maneuver, preferring
   an **interior** passing lane (road on both sides) so a small overshoot can't reach a median/edge.

## Run it (through the CARLA bridge)

```powershell
# one scenario, real LLM
python scripts/carla_agent_exec.py --timeout 900 -- "<venv>/python.exe" `
  scripts/run_overtake.py --scenario s01_t04_highway_safe_pass --out-dir runs/campaign_llm_v1

# all 20
python scripts/carla_agent_exec.py --timeout 3600 -- "<venv>/python.exe" `
  scripts/run_overtake.py --all --out-dir runs/campaign_llm_v1

# fast control-only iteration (no LLM latency); deterministic mock planner still
# exercises the full tool/critic/DSL machinery
... scripts/run_overtake.py --scenario <id> --out-dir runs/<v> --mock

# big-screen render (1280x720, HUD font scales with resolution); or --render 1920x1080
... scripts/run_overtake.py --all --hires --out-dir runs/<v>
```

Toggle the agentic layer with `AUTOPASS_AGENTIC=0` (falls back to the single-call `decide()` for
A/B comparison). The waypoint controller + hard gates are identical in both modes.

Each scenario writes `<id>.mp4` (ego-RGB+BEV two-panel with overlays), `frames/`,
`result.json`, `trace.json`. The batch writes `summary.json`. Build the reel with
`python scripts/build_reel.py runs/campaign_llm_v1`.

## Scenario catalog (23)

Across 4 towns (Town04 highway/mountain, Town05 multi-lane arterial, Town03 urban, Town01 rural
two-lane), varying lead speed, urgency, passing side, weather (clear/cloudy/wet/hard-rain/sunset),
and road type (same-direction multilane vs opposing-lane two-lane):

- **s01–s15**: clean overtakes (incl. right-side pass, stalled lead, wet/hard-rain/sunset, slow farm
  vehicle on a country lane via the oncoming lane).
- **s16, s18, s20**: correct declines — lead is not actually slow, keep lane.
- **s17, s19**: dynamic safety passes — yield to fast rear traffic / wait for oncoming, then overtake
  once the hazard clears (driven by the critic reject -> re-sense -> approve loop).
- **s21, s22, s23**: agency / realism showcases — live ambient Traffic-Manager traffic (s21, s22) and
  a slow heavy truck lead (s23).

**Ambient traffic** (`ambient=N` on a scenario) spawns N Traffic-Manager vehicles kept out of the
ego's corridor (a 160 m x 9 m box along its heading) so the world is lively and realistic while the
staged overtake stays crash-free. Reliable on Town04's straight highway; Town05's curved,
junction-laced corridors make dense ambient unsafe, so ambient showcases run on Town04.

> Town02 and Town10HD were evaluated and dropped: Town02's short two-lane stretches collide at
> junctions / block on roadside clutter; Town10HD's downtown is lined with parked-car props that
> clutter the passing lane. Neither is a controller bug — they are just unsuitable venues.

## Results (agentic, real LLM, hi-res — `runs/campaign_agentic_v2`)

**23/23 behaved correctly — 0 collisions, 0 off-road, max lane deviation <= 0.6 m everywhere.**
Every pass/wait was produced by the live agentic loop (planner tool-choice + critic + mutable DSL,
gpt-4o-mini @ temp 0.4, non-deterministic). The campaign logged **1142 LLM tool-calls total**, and
the per-scenario deliberation depth varies with the scene — clean passes settle in ~6 cycles, a
two-lane wait-for-oncoming took 17, and fast-lead declines deliberate the full clip (~32) — i.e. the
tool sequence is **not** fixed.

- **15 clean overtakes** (s01–s15): Town04 highway (safe, stalled lead, wet, right-side, sunset,
  coastal-overcast), Town05 multi-lane arterial (×3), Town03 urban (×2), Town01 rural two-lane (×3
  incl. hard rain + slow farm vehicle on the oncoming lane).
- **2 dynamic safety passes** (s17, s19): yield to fast rear traffic / wait for oncoming, then
  overtake once the hazard clears — driven by the critic reject -> re-sense -> approve loop.
- **3 correct declines** (s16, s18, s20): lead is not actually slow -> keep lane safely.
- **3 agency / realism showcases** (s21–s23): live ambient Traffic-Manager traffic (incl. a busy
  ~22-vehicle highway) and a slow heavy-truck lead, all overtaken cleanly.

Earlier `runs/campaign_llm_v2` (20/20) is the pre-agentic single-call version (still valid; the
committed baseline you liked). The agentic v2 supersedes it.

Artifacts: per-scenario hi-res `<id>.mp4` (ego-RGB + BEV with gap boxes + PLAN/CRITIC HUD) +
`frames/` + `trace.json` (per-cycle tool sequence, critic verdict, DSL revisions) + `result.json`;
batch `summary.json`; combined `presentation_reel.mp4` (title card per scenario). Review any clip
with `python scripts/make_montage.py runs/campaign_agentic_v2/<id>/frames out.png`.
