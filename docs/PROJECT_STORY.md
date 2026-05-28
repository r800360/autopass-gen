# AutoPass-Gen — project story for advisors / TAs

Use this when explaining the project in meetings, slides, or email. It ties **research questions** to **what you built** and **what is still simulated**.

---

## One-sentence pitch

**AutoPass-Gen** is a LangGraph multi-agent system that decides **when** an autonomous ride-hailing vehicle may overtake a slow lead car, using **RGB + semantic segmentation + depth** (not simulator cheat distances), with **code agents for measurement** and **LLMs only where judgment is needed**.

---

## The problem (why anyone cares)

Ride-hailing AVs get a **goal + time pressure** (“airport in 20 minutes”). They must:

1. **Make progress** — sitting behind a slow vehicle misses deadlines.
2. **Stay safe** — a bad pass causes collisions; never passing causes unhappy users and late arrivals.

Passing is **sequential and visual**: you must read gaps in the **passing lane** (front and rear), road length ahead, oncoming traffic, and whether physics allows finishing the maneuver in time. That is not a single “if distance < X” rule.

---

## Research question (say this explicitly)

> **How can we orchestrate overtaking as a multi-agent decision process that is grounded in visual perception, with clear separation between measurable checks (code) and contextual judgment (LLMs)?**

Sub-questions you are exploring:

| Sub-question | What you built to probe it |
|--------------|----------------------------|
| Can perception support passing without privileged world coordinates? | Seg + depth → lane masks → front/rear/oncoming distances |
| Where do LLMs help vs hurt? | LLM: rear closing time, target speed, traffic/replan; Code: front gap, checker, kinematics |
| Can LangGraph express the paper’s passing pipeline? | `agents/autopassing.py` — navigate modes, passing subgraph, executor loop |
| Can the stack improve over scenarios? | Visual closed-loop + mutation (`autopass_langgraph_demo.py`); curated pass/fail scenarios |

---

## Hypothesis (testable claim)

**A hybrid stack** — LangGraph orchestration + vision-based distances + code safety gates + selective LLM judgment — **rejects unsafe passes and approves safe ones** more reliably than fixed thresholds or naive speed rules (e.g. “always 1.5× lead speed”), while still responding to urgency.

**Evidence you can show today:** 16 pytest cases; scenario traces (`demo_01` safe pass vs `demo_02` oncoming reject); RGB/seg/depth panels; optional CARLA video.

---

## What is *not* the research claim (be honest with your TA)

| Built | Limitation |
|-------|------------|
| CARLA spawn + cameras | Ego motion in watch demos is **scenario-driven** (logical `WorldState` → actor poses), not full vehicle control API |
| `carla_executor` node | Still **logical** drive step; does not send throttle/steer yet |
| Map server `:8100` | **Optional** fake city map for navigation LLM — not CARLA |
| Mock LLMs default | Reproducible demo; real OpenAI optional |

Framing: *“We validated the **decision architecture** and **vision interface**; low-level CARLA control is the next integration step.”*

---

## Two pipelines (stop conflating them)

Your repo has **two complementary demos**. Name them separately when you talk:

### 1. Visual closed-loop (`autopass_langgraph_demo.py` / `demo_carla_watch.py --pipeline visual`)

**Story:** End-to-end **your** research prototype — request → urgency → **perceive from pixels** → plan → safety → execute → evaluate → (optional) mutation.

**Best artifact:** `demo_01_*_visual_loop.mp4` — clean kinematics, passing/wait/replan.

### 2. Paper + redesign multi-agent (`agents/autopassing.py` / `demo_carla_watch.py --pipeline multi_agent`)

**Story:** Friend’s **reference architecture** (navigate / passing subgraph / checker / current lane) plus **redesigned LLM agents** (rear time, target velocity, traffic tiers).

**Best artifact:** message trace under `runs/demo/multi_agent/*_messages.txt` — shows *why* pass/no_pass.

---

## Architecture in 30 seconds (for a whiteboard)

```
User request + deadline
        ↓
   Navigate (route + urgency)
        ↓
   Passing needed? ──no──→ drive / CARLA executor loop
        │
       yes
        ↓
   Perception (RGB, seg, depth) → distances
        ↓
   Code: front gap, rear gap, checker
        ↓
   LLM: rear lane-change time, target speed (when needed)
        ↓
   Code: road length + kinematics (current lane)
        ↓
   pass | wait | move_but_not_pass | replan
        ↓
   Execute (visual world or CARLA watch animation)
```

---

## Contribution vs prior work (course paper baseline)

| Baseline (paper flow) | Your extension |
|----------------------|----------------|
| Passing pipeline as monolithic logic | **LangGraph** nodes + explicit state |
| Fixed / naive rules | **LLM + code split** per redesign |
| Simulator distances | **Segmentation + depth** extraction |
| Single demo | **Curated scenario suite** + tests + CARLA optional |

---

## Demo script for a skeptical TA (3 minutes)

1. **Problem:** “Urgent trip, slow lead — pass or wait?”
2. **Show one panel:** RGB + seg + depth + extracted front gap (`runs/presentation/visual/frames/...`).
3. **Show trace:** 4–6 lines from `demo_01_*_messages.txt` — front approved, back approved, pass signal.
4. **Show test:** `pytest -q` → 16 passed.
5. **Optional CARLA:** `demo_carla_watch.py --pipeline visual` — “simulator validates the **interface**; control is staged.”

---

## FAQ your TA might ask

**Q: Is this just a chatbot driving a car?**  
A: No. Most nodes are **deterministic code** on structured perception. LLMs only estimate rear closing time, target speed, and traffic/replan judgments.

**Q: Does CARLA prove the planner works?**  
A: CARLA proves we can **feed real sensor data** and **visualize** decisions. Full physics control is follow-on work.

**Q: What’s the evaluation metric?**  
A: Scenario outcomes: collision, deadline miss, unnecessary pass, correct reject (e.g. oncoming). Closed-loop mutation tracks improvement over rounds on synthetic/visual worlds.

**Q: What’s the long-term vision?**  
A: Closed-loop **learning from pass outcomes** (your mutation loop) with **multi-agent passing** in sim then on-road — same graph, richer executor.

---

## Suggested email / meeting opener (copy-edit)

> We’re building **AutoPass-Gen**: a vision-grounded LangGraph stack for **safe overtaking under deadline pressure**. Our research question is how to split **measurement** (segmentation/depth + code checkers) from **judgment** (LLMs for rear time and speed context). We have a working visual closed-loop, the paper’s multi-agent graph with redesign nodes, tests, and optional CARLA video. We’d like feedback on whether our **evaluation scenarios** match that question, and what to prioritize next: CARLA vehicle control vs learning loop data collection.

---

## Team talking points (align before the TA meeting)

1. Lead with **research question**, not LangGraph jargon.
2. Show **one scenario end-to-end** before listing files.
3. Separate **visual loop** (your prototype) vs **autopassing graph** (paper + redesign).
4. Acknowledge **simulated executor** in CARLA; point to visual MP4 for correct motion.
5. Agree on **next milestone** (e.g. “wire executor to CARLA VehicleControl” or “run mutation batch on 6 scenarios”).
