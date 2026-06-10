# AutoPass-Gen — internal checkpoint (final production pass)

_Snapshot of the live agentic CARLA system as of this pass. Companion to
`docs/CLEAN_OVERTAKE_CAMPAIGN.md` and `docs/HANDOFF_FINAL.md`._

## Research question
Under trip-deadline pressure, **when may an autonomous vehicle overtake a slow lead** using
**only vision-derived gaps** (semantic segmentation + metric depth), with **inspectable
multi-step reasoning**, **hard safety gates**, and **closed-loop re-planning** after each
actuation step? We study the decision layer of one maneuver (overtake vs. wait/yield), not a
full driving stack.

## System architecture (current, live)
Single per-tick closed loop in CARLA 0.9.16 (synchronous, 20 Hz / dt=0.05):
1. **Perception** (`perception/clean_overtake.py::perceive`): front + rear RGB / semantic-seg /
   depth cameras on the ego. `carla_seg_to_car_distances` -> vehicle masks + median depth. Each
   detection gets a **monocular lateral offset** `lat = 2*depth*(cx/W - 0.5)` (fov 90) to
   lane-associate it: |lat|<2.6 m = our lane (the lead); one lane over toward the passing side =
   adjacent / oncoming; rear camera mirrored for the passing-lane rear gap.
2. **Agentic decision layer** (`perception/overtake_agent.py`, `AUTOPASS_AGENTIC=1`): every 0.5 s
   while in `follow`, an LLM **planner** chooses the next tool; a deterministic **critic** verifies
   any proposed pass; a mutable **DSL** carries belief + freshness + memory. (Details below.)
3. **Hard safety gates** (`evaluate_gates`): front gap >= 18 m, lead slow <= 9 m/s, passing-lane
   rear gap >= 12 m, oncoming >= 45 m (two-lane only), passing lane clear ahead >= 24 m. Gates can
   only **block** a pass; never force one.
4. **Controller** (lane-compliant by construction): pure-pursuit steering to real CARLA
   lane-center waypoints (`get_waypoint` + `get_left_lane`/`get_right_lane`); FSM
   `follow -> lane_change -> overtake -> merge_back -> done`. ACC holds a steady gap behind the
   lead so the front gate can open without rear-ending. Two-lane passes use the opposing lane
   (lookahead via `.previous()`). Merge-back is vision-triggered (rear cam sees the lead behind)
   with a deterministic "ego is genuinely past the lead" precondition (`long_to_lead < -7`).
5. **Corridor finder** (`find_corridor`): reads the road graph for a straight stretch where the
   travel AND passing lanes stay parallel / driving / non-junction for the whole maneuver,
   preferring an interior passing lane. Map-agnostic.

## What makes it truly agentic now (vs. an augmented single LLM call)
- **Agency over tools, not just wording.** The planner picks the next tool from
  `{sense_front, sense_passing_lane, sense_rear, check_corridor, propose_pass, hold}`; the order
  and number of tool calls **vary per scene** (logged per cycle in `trace.json`).
- **External verification.** A separate deterministic **critic** checks the hard gates + that the
  gating evidence was freshly sensed this cycle + corridor geometry. The LLM cannot approve its own
  action; a rejection is remembered and forces re-sense / replan next cycle.
- **Iterative DSL mutation + memory.** The DSL is live shared state, not just an input scenario;
  it tracks per-field freshness, accumulates denials and full tool history, and is
  revision-stamped.
- **External feedback / closed loop.** CARLA advances the world; the next cycle re-senses and the
  belief is updated. This produces the yield-to-rear-traffic-then-overtake and
  wait-for-oncoming-then-overtake behaviors.
- **Greedy under urgency, safe under code.** The critic is the safety authority: when it approves
  on fresh evidence under high/medium urgency, the agent takes the pass (missing a safe pass is a
  failure in this project's thesis); otherwise it waits and re-plans.

## DSL / world-belief representation (`overtake_agent.py::ScenarioDSL`)
Live mutable record: front/rear/oncoming/passing-lane gaps + lead speed (vision-derived), each
with a freshness/age stamp; `two_lane`, `urgency`; memory = `denials`, `tool_history`, last critic
rejection reason; `intent`/`plan`; `revision` counter. `as_gaps()` projects it to the gate input.

## Agent roles
- **Planner (LLM):** chooses the next tool each cycle (gather missing/stale evidence; propose a
  pass when fresh; hold only on a clear hazard).
- **Tools:** `sense_front`, `sense_rear`, `sense_passing_lane` (seg+depth gaps), `check_corridor`
  (road-graph lookahead validity), `propose_pass`, `hold`.
- **Critic (deterministic):** verifies hard gates + evidence freshness + corridor; approve -> act,
  reject -> remember + re-sense/replan.
- **Controller (code):** waypoint lane-keep + lane-change FSM; lane-compliant by construction.

## Safety gates (deterministic floor)
front >= 18 m; lead <= 9 m/s ("slow enough to be worth overtaking"); passing-lane rear >= 12 m;
oncoming >= 45 m (two-lane); passing lane clear >= 24 m. **Note:** front/slow gates are about
*warrant/feasibility*; rear/oncoming/clear gates are about *external hazard*. The benchmark now
separates a hazard-gate violation (genuinely unsafe) from a slow-gate-only violation (unwarranted).

## CARLA bridge / live execution
Agent shell cannot reach CARLA RPC; dispatch through `scripts/carla_agent_exec.py` to the bridge
(`scripts/carla_agent_bridge.py`) running in the user's terminal with `CarlaUE4.exe`. Live LLM key
is loaded by `run_benchmark.py` from gitignored `./.openai_key` if the bridge env lacks
`OPENAI_API_KEY`. CARLA 0.9.16 crashes (ThreadGroup assertion) after ~3 `load_world` calls -> run
ONE town per job; `load_world` skips the reload when already on that town.

## Major engineering changes during this work
- Replaced the fragile axis/corridor control tangle (`carla_control.build_vehicle_control`, the old
  4.5k-line scenario file) with the small waypoint-based `clean_overtake.py` (lane-compliant by
  construction; generalizes to any town).
- Added the agentic layer `overtake_agent.py` (planner + critic + mutable DSL + memory + greedy
  backstop) on top of the unchanged controller + gates.
- Vision-triggered merge-back with a deterministic "past the lead" safety precondition.
- Ambient Traffic-Manager traffic kept out of the ego corridor box (Town04 showcases).
- Hi-res render option + resolution-scaled HUD font; big-screen demo clip with caption banners.
- Benchmark: policy modes (`AUTOPASS_POLICY` = no_pass / autopass / aggressive) + metrics
  (mean_speed, overtake_completed, unsafe vs unwarranted pass, lane dev).

## Known bugs / uncertainties / false pos-neg
- **Metric nuance (fixed this pass):** "unsafe pass attempt" previously counted any pass committed
  while `can_pass=False`, which lumped *unwarranted* (lead not slow, e.g. s16) together with
  *unsafe* (rear/oncoming hazard). Now split: unsafe = hazard-gate violation only.
- **Slow-lead threshold (9 m/s)** is a design parameter; a human might overtake a 9-12 m/s lead.
  The system declines those as "not worth it" (efficiency, not safety). Documented as a limitation,
  not changed to chase numbers.
- **Phase bookkeeping** ("ego ahead of lead", merge trigger) uses sim positions as a safety floor;
  the *decision* gaps are vision-derived.
- **CARLA reload instability** forces one-town-per-job benchmarking.
- Scenarios are hand-authored single-lead overtakes (+ optional ambient TM), not random traffic.
- Lead-speed overlay can lag the commanded speed by a frame; not used for gating.
