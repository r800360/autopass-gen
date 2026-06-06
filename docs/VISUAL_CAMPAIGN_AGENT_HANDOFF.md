# AutoPass-Gen visual campaign — handoff (SUPERSEDED)

**This older handoff (the v50 / `visual_campaign` / `carla_control.build_vehicle_control` path) is
superseded.** That control path steered the ego from custom "axis/corridor" geometry and drove
off-road across maps (v45–v65). It has been replaced by a clean, generalizable, waypoint-based
driver.

## Use this instead

See **`docs/CLEAN_OVERTAKE_CAMPAIGN.md`**. Key files:
- `perception/clean_overtake.py` — controller + vision + agentic decision + recording (small, navigable)
- `scripts/run_overtake.py` — 20-scenario catalog + CLI (`--all`, `--scenario id[,id]`, `--mock`)
- `scripts/make_montage.py` — 6-frame review montage
- `scripts/build_reel.py` — stitch all clips into one presentation reel

Run through the CARLA bridge (unchanged — see below):
```powershell
python scripts/carla_agent_exec.py --timeout 3600 -- "<venv>/python.exe" `
  scripts/run_overtake.py --all --out-dir runs/campaign_llm_v2
```

Latest validated results: `runs/campaign_llm_v2/summary.json` (+ per-scenario mp4/frames/trace).

## Why the rewrite

The clean driver steers only toward real CARLA lane-center waypoints (`get_waypoint` +
`get_left_lane`/`get_right_lane`), so it is lane-compliant by construction and works on any town.
Gaps are vision-derived (front+rear semantic-seg + depth, monocular lateral lane association); the
pass/wait call is a real LLM under urgency, hard-gated by safety checks; honest metrics (collision
sensor + lane-deviation + off-road). This achieves the project's north star (below) without the
fragile control tangle.

## CARLA bridge (still mandatory)

User keeps in their integrated terminal: `python scripts/carla_agent_bridge.py` (+ CarlaUE4.exe).
Dispatch from the agent shell via `scripts/carla_agent_exec.py`. Installed maps: Town01–05, Town10HD
(Town06/Town07 not installed). Town10HD is unsuitable for clean overtakes (parked-car clutter).

## North star (unchanged)

Under trip deadline pressure, decide when an AV may overtake a slow lead using only vision-derived
gaps (segmentation + depth), with inspectable multi-step reasoning, hard safety gates, and
closed-loop re-planning after each actuation step. Deliverable = MP4 + frames + overlays.
