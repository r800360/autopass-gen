# AutoPass-Gen — final handoff

Final state of the live, vision-grounded agentic overtaking system, its production benchmark, and
the paper. Pairs with `docs/CHECKPOINT.md` (architecture detail) and
`docs/CLEAN_OVERTAKE_CAMPAIGN.md` (campaign how-to).

## 1. Final results

### Live full-campaign benchmark (`runs/benchmark_live/`, 69 live CARLA runs)
autopass = live agentic LLM (planner + critic + tool loop, no mock, no oracle); no_pass and
aggressive are deterministic policies (no LLM). Aggregated in
`runs/benchmark_live/benchmark_summary.json`.

| Policy | Overtakes completed | Collisions | Unsafe passes | Unwarranted passes | Mean speed (m/s) | Max lane dev (m) |
|--------|--------------------|-----------|---------------|--------------------|------------------|------------------|
| Never-pass          | 0 / 18  | 0 / 23 | 0 | 0 | 3.9 | 0.24 |
| **AutoPass-Gen (ours)** | **18 / 18** | **0 / 23** | **0** | **0** | **9.7** | 0.60 |
| Always-pass         | 18 / 18 | 2 / 23 | 2 | 3 | 9.7 | 0.60 |

- "Overtakes completed" is over the 18 pass-intended (slow-lead) scenarios; the other 5 are hazards.
- "Unsafe" = a pass committed into a rear/oncoming/blocked-lane hazard. "Unwarranted" = a pass of a
  lead that was not actually slow (clear lane, little time saved).

### Per-scenario, the decisive cases
- **Always-pass collisions (2):** `s17_t04_reject_rear_traffic` (cuts in front of a fast closing
  car) and `s19_t01_rural_oncoming_reject` (head-on into oncoming).
- **Always-pass unwarranted (3):** `s16/s18/s20_reject_fast_lead` (overtakes a not-slow lead; no
  collision, just unnecessary).
- **AutoPass-Gen on the 5 hazards (all collision-free):** declines the fast leads
  (s16, s18, s20; no pass attempted), yields-then-overtakes the rear-traffic case (s17), and waits
  out the oncoming case (s19).

### Campaign (qualitative video evidence, `runs/campaign_agentic_v2/`)
23/23 scenarios correct, 0 collisions, 0 off-road, lane dev <= 0.6 m, 1142 live LLM tool-calls.
Strongest qualitative clips for talks/figures: **s17** (yield to fast rear traffic, then pass) and
**s19** (wait for oncoming, then pass) — behaviors a fixed trajectory cannot produce. Combined
demo: `presentation_assets/demo_hero.mp4`; reel: `runs/campaign_agentic_v2/presentation_reel.mp4`.

## 2. Files

| Area | Path |
|------|------|
| Controller + perception + recording | `perception/clean_overtake.py` |
| Agentic layer (planner/critic/DSL/memory) | `perception/overtake_agent.py` |
| Scenario catalog + single-run CLI | `scripts/run_overtake.py` |
| Benchmark (3 policies, metrics) | `scripts/run_benchmark.py` |
| CARLA bridge | `scripts/carla_agent_bridge.py`, `scripts/carla_agent_exec.py` |
| Paper (CVPR, manual bib) | `paper/paper.tex`, `paper/refs.bib`, `paper/figs/*.png` |
| Figures | `presentation_assets/fig_architecture.png`, `fig_problem.png`, `fig_results.png` |
| Slides + backup video | `10_Autopass_Gen.pptx`, `10_Autopass_Gen.mp4` |
| Live API key (gitignored) | `.openai_key` |

## 3. Exact commands used

Verify bridge (from agent shell):
```
.\.venv\Scripts\python.exe scripts\carla_agent_exec.py -c "import carla; print(carla.Client('127.0.0.1',2000).get_server_version())"
```
Run the live benchmark, ONE town per job (CARLA 0.9.16 crashes after ~24 actor cycles or ~3 world
reloads per process; one town per job stays safe). Key auto-loads from `.openai_key`:
```
.\.venv\Scripts\python.exe scripts\carla_agent_exec.py --timeout 1800 -- ^
  ".venv\Scripts\python.exe" scripts\run_benchmark.py --town t04 --policy autopass --out-dir runs\benchmark_live
# repeat for {t04,t05,t03,t01} x {no_pass,autopass,aggressive}; then aggregate:
.\.venv\Scripts\python.exe scripts\run_benchmark.py --aggregate-only --out-dir runs\benchmark_live
```
Single scenario / campaign (live): `scripts\run_overtake.py --scenario <id>` or `--all`
(`--hires` for 720p, `--mock` for fast no-LLM control checks).

## 4. Remaining risks / open items
- **CARLA 0.9.16 stability:** the bridge/server can crash on long multi-town, multi-policy runs
  (ThreadGroup assertion). Mitigation in place: one town + one policy per process, gap-fill
  re-runs, disk-based aggregation. If a run dies, re-dispatch the missing (town, policy).
- **Slide-4 table in `10_Autopass_Gen.pptx`** still shows the earlier 6-scenario benchmark; the
  live full-campaign numbers above supersede it. Re-run `build_deck.py` (with the table numbers
  updated) to sync if you present from the deck again.
- **Paper compile:** `paper/paper.tex` needs `cvpr.sty` (your CVPR/Overleaf template). It uses a
  manual `thebibliography` (no `.bst` needed). All 13 cite keys resolve; braces balanced. Figures
  are in `paper/figs/`.
- **Slow-lead threshold (9 m/s)** is a design parameter (declines marginally-slow leads); documented
  as a limitation, intentionally not tuned to chase numbers.
- **LLM non-determinism:** decisions vary run-to-run (temp 0.4); the deterministic critic + gates
  bound this so it affects \emph{when} evidence is gathered, not \emph{whether} an unsafe pass is
  allowed. e.g. on s19 a given run may wait the whole clip rather than yield-then-pass; both are
  safe.

## 5. To continue in a new conversation
1. Start `CarlaUE4.exe` + `python scripts/carla_agent_bridge.py` (with the key, or rely on
   `.openai_key`).
2. `runs/benchmark_live/benchmark_summary.json` is the source of truth for the benchmark table.
3. `paper/paper.tex` is the current paper; fill/adjust only with re-verified numbers.
4. Read `docs/CHECKPOINT.md` for the architecture and `clean-overtake-rewrite` memory for history.
