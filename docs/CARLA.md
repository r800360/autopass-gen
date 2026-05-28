# CARLA + AutoPass-Gen

## Do you need the map server?

**No.** The map server (`agents/map_server.py`) is only for the **navigation LLM’s fake city map** (SimCity streets/landmarks). It has nothing to do with CARLA’s Unreal map.

| Component | What it is |
|-----------|------------|
| **CARLA** (`CarlaUE4.exe`) | 3D simulator + sensors (RGB, depth, semantic) |
| **Map server** (`:8100/map`) | Optional JSON map for route-planning demos |

You can run CARLA without the map server. You will still see the map-server warning unless you start Flask on port 8100.

---

## Python versions: why the egg matters

Your install only includes:

`C:\carla\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg`

That file is **compiled for Python 3.7’s ABI**. It will **not** import in a 3.10–3.14 venv:

```text
ImportError: ... egg is not a supported wheel on this platform
```

| Approach | Works? |
|----------|--------|
| Use py3.7 egg inside py3.10 `.venv` | **No** |
| Skip egg, `pip install carla` on 3.10 | **No** for 0.9.15 (no matching pip wheel) |
| **Sidecar** (py3.7 talks to CARLA, py3.10+ reads HTTP) | **Yes — recommended** |
| Run entire project on py3.7 only | **No** — LangGraph/modern deps need 3.10+ |

**You cannot choose “no egg” and still use CARLA 0.9.15** unless you replace it with a newer CARLA build that ships eggs/wheels for your Python version (different install).

---

## Recommended setup: CARLA sidecar (3 terminals)

### Terminal 1 — CARLA simulator

```powershell
C:\carla\CarlaUE4.exe
```

Wait until the town is loaded.

### Terminal 2 — Capture server (Python 3.7)

Install Python 3.7 if needed, then:

```powershell
py -3.7 -m pip install numpy
set CARLA_EGG=C:\carla\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg
py -3.7 scripts\carla_capture_server.py
```

You should see: `Connected. GET /frame for rgb+seg+depth npz`

### Terminal 3 — AutoPass (your `.venv`, Python 3.10+)

```powershell
cd C:\Users\bsach\Documents\autopass-gen
.\.venv\Scripts\Activate.ps1
$env:AUTOPASS_PERCEPTION_BACKEND = "carla"
$env:AUTOPASS_CARLA_SIDECAR = "http://127.0.0.1:8201"
$env:AUTOPASS_MOCK_LLM = "1"
python demo.py --carla --mode multi_agent
```

Health check:

```powershell
curl http://127.0.0.1:8201/health
```

---

## Visible closed loop + MP4 (recommended for demos)

`demo.py --carla` spawns vehicles and runs the LangGraph **once** (logical state only). It does **not** step the simulator each planner tick or write video.

Use **`demo_carla_watch.py`** with CarlaUE4 already running:

```powershell
pip install imageio imageio-ffmpeg
python demo_carla_watch.py --pipeline visual --scenario 0
python demo_carla_watch.py --pipeline multi_agent --scenario 0 --steps 40 --carla-map Town04
```

- Spawns ego + lead + rear + oncoming (distances capped so all four stay on-screen).
- Each planner/graph step: updates actor poses, ticks CARLA, records ego RGB with HUD overlay.
- Writes `runs/carla_watch/<scenario>_visual_loop.mp4` or `*_multi_agent.mp4` plus PNG frames.

Chase the CARLA window while it runs; open the MP4 afterward to review passing decisions.

---

## What changes when `backend=carla`

- `perception/pipeline.py` calls the sidecar (or in-process egg on py3.7 only).
- **No** `visual_world.render_sensor_frame` fallback if the sidecar returns a frame.
- Segmentation/depth use **CARLA Cityscapes IDs** via `perception/carla_labels.py`.
- LangGraph logic in `agents/autopassing.py` is unchanged.

`carla_executor` in the graph is still a **simulated** drive step (not full vehicle control). Perception is real; low-level control can be extended later.

---

## Without sidecar (single Python 3.7 only — not recommended)

Only if you run **everything** under `py -3.7` and install old LangGraph deps (fragile). Your project targets 3.10+; use the sidecar instead.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` on 2000 | Start `CarlaUE4.exe` first |
| `Connection refused` on 8201 | Start `carla_capture_server.py` with py3.7 |
| `import carla` fails on 3.10 | Expected — use sidecar |
| Empty `car_distances` | Drive/spawn traffic in CARLA; ego may be alone |
| Falls back to visual renderer | Sidecar down or `/frame` failed — check Terminal 2 |

---

## Optional: map server + CARLA together

```powershell
# Terminal A
python agents\map_server.py

# Terminals 1–3 as above
```

Navigation gets the SimCity JSON map; perception uses CARLA cameras. Independent systems.
