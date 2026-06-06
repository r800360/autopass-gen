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
| Driver (controller + perception + agency + recording) | `perception/clean_overtake.py` |
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
3. **Agentic decision** (`decide`): when gates allow, a real LLM (`structured_invoke`, gpt-4o-mini,
   temperature 0.4 — non-deterministic) chooses pass vs wait under the stated urgency, with a
   one-sentence rationale logged to the trace.
4. **Control (lane-compliant by construction).** Pure-pursuit steering toward a real CARLA
   lane-center waypoint; FSM phases follow -> lane_change -> overtake -> merge_back -> done. ACC
   holds a steady gap behind the lead so the front-gap gate can open without rear-ending. Two-lane
   passes use the opposing lane (lookahead via `.previous()`).
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

# fast control-only iteration (no LLM latency)
... scripts/run_overtake.py --scenario <id> --out-dir runs/<v> --mock
```

Each scenario writes `<id>.mp4` (ego-RGB+BEV two-panel with overlays), `frames/`,
`result.json`, `trace.json`. The batch writes `summary.json`. Build the reel with
`python scripts/build_reel.py runs/campaign_llm_v1`.

## Scenario catalog (20)

14 overtakes + 6 safety holds across 5 towns (Town04 highway/mountain, Town05 multi-lane arterial,
Town03 urban, Town01 & Town02 two-lane rural), varying lead speed, urgency, passing side, weather
(clear/cloudy/wet/hard-rain/sunset), and road type (same-direction multilane vs opposing-lane
two-lane). Reject cases are driven by the safety gates (lead not slow, fast rear traffic in the
passing lane, oncoming traffic on a two-lane road).

> Town10HD was evaluated and dropped from the clean-overtake set: its downtown is lined with parked-car
> props that clutter the passing lane and its corridors are curvy, which is unsuitable for a clean
> overtake demo (not a controller bug).

## Results (real LLM, `runs/campaign_llm_v2`)

**20/20 behaved correctly — 0 collisions, 0 off-road, max lane deviation <= 0.6 m everywhere.**
Decisions made by the live LLM (gpt-4o-mini, temperature 0.4, non-deterministic) with reasoning
logged per cycle in each `trace.json`.

- 14 clean overtakes: Town04 highway (safe pass, stalled lead, wet, right-side, sunset, coastal
  overcast), Town05 multi-lane arterial (×3 incl. wet), Town03 urban (×2), Town01 rural two-lane
  (×3 incl. hard rain + slow farm vehicle).
- 2 dynamic safety passes: yield to a fast vehicle in the passing lane, then overtake once clear
  (Town04); wait for oncoming traffic on a two-lane road, then overtake (Town01).
- 4 correct declines: lead is not actually slow, so overtaking is not warranted (Town04 ×1,
  Town05 ×1, Town03 ×1, Town04 rear-traffic case ×1) — ego keeps its lane safely.

Artifacts: per-scenario `<id>.mp4` + `frames/` + `trace.json` + `result.json`; batch `summary.json`;
combined `presentation_reel.mp4` (~3.4 min, title card per scenario). Review any clip quickly with
`python scripts/make_montage.py runs/campaign_llm_v2/<id>/frames out.png`.
