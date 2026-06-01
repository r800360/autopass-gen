#!/usr/bin/env python3

"""

Watch the agentic closed loop in CARLA with ego + overhead video.



Hero pass mode (recommended for final presentation):

  python demo_carla_watch.py --hero-pass --scenario clear_safe_pass --policy autopass --urgency high

  python demo_carla_watch.py --hero-pass --scenario clear_safe_pass --policy no_pass

"""

from __future__ import annotations



import argparse

import os

import sys

from dataclasses import asdict

from pathlib import Path



ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "agents"))



from autopass.config import apply_production_defaults, require_runtime



apply_production_defaults()

os.environ.setdefault("AUTOPASS_MOCK_LLM", "0")
os.environ.setdefault("AUTOPASS_DECISION_ORACLE", "0")
os.environ.setdefault("AUTOPASS_LLM_TEMPERATURE", "0.4")

require_runtime(need_carla=True, need_openai=True)





def _ensure_curated_corridor() -> None:

    os.environ.setdefault("AUTOPASS_CARLA_CURATED_CORRIDOR", "1")

    os.environ.setdefault("AUTOPASS_ENVIRONMENT", "highway")





def _display_t_s_for_frame(recorder, world, label: str) -> tuple[float, float]:
    """HUD time (monotonic) and raw graph world.t_s for metadata."""
    from perception.carla_scenario import get_session

    graph_t = float(world.t_s)
    session = get_session()
    if session.ready:
        graph_t = max(graph_t, session.graph_logical_t_s())
    display_t = recorder.next_display_t_s(graph_t, label=label)
    return display_t, graph_t


def _record_session(
    recorder,
    spec,
    world,
    label: str,
    extra: dict | None = None,
    *,
    graph_step: int | None = None,
    animate_steps: int = 6,
) -> None:

    from perception.carla_scenario import get_session



    session = get_session()

    session.animate_steps(spec, world, steps=animate_steps)
    session.wait_for_sensor_frames(max_ticks=8, timeout_s=2.0)

    pair = session.grab_frame_pair()

    if pair is None:

        frame = session.grab_frame()

        if frame is None:

            return

        rgb, _, _ = frame

        overhead = None

    else:

        rgb, _, _, overhead = pair

    display_t, graph_t = _display_t_s_for_frame(recorder, world, label)
    meta = dict(extra or {})
    meta["graph_t_s"] = round(graph_t, 3)
    recorder.capture(
        rgb,
        t_s=display_t,
        label=label,
        extra=meta,
        overhead=overhead,
        graph_step=graph_step,
    )





def run_agentic_carla_loop(
    spec,
    world,
    out_dir: Path,
    ticks_per_step: int,
    max_steps: int,
    *,
    policy: str = "autopass",
) -> dict:

    from visual_world import WorldState, spec_to_dict



    from autopass.graph import build_agentic_graph

    from autopass.scenarios import assert_carla_environment_allowed, showcase_map_for_environment

    from perception.carla_recorder import CarlaRecorder

    from perception.carla_scenario import bootstrap_carla_scenario, get_session

    from perception.context import set_context



    env_kind = os.environ.get("AUTOPASS_ENVIRONMENT", "highway")

    map_name = showcase_map_for_environment(env_kind)

    os.environ["AUTOPASS_CARLA_MAP"] = map_name

    assert_carla_environment_allowed(env_kind)

    set_context(spec, world, "carla")

    if not bootstrap_carla_scenario(spec, world, map_name=map_name):

        from autopass.config import AutopassConfigurationError



        raise AutopassConfigurationError(

            "CARLA bootstrap failed. Start CarlaUE4.exe, verify pip install carla==0.9.16, "

            "then run: python carla_smoke.py"

        )



    recorder = CarlaRecorder(out_dir, spec.scenario_id)

    sim_world = world

    session = get_session()

    def _demo_sink(partial_world, tick_label: str) -> None:
        _record_session(
            recorder,
            spec,
            partial_world,
            f"EXECUTE {tick_label}",
            {"subframe": tick_label},
            animate_steps=0,
        )

    session._demo_frame_sink = _demo_sink

    def _recording_animate(world_state: WorldState, label: str, steps: int, *, graph_step: int | None = None, extra: dict | None = None) -> None:
        if steps <= 0:
            return

        def _on_tick(partial_world, tick_i: int) -> None:
            _record_session(
                recorder,
                spec,
                partial_world,
                f"{label} tick{tick_i}",
                {**(extra or {}), "subframe": tick_i},
                graph_step=graph_step,
                animate_steps=0,
            )

        session.animate_steps(spec, world_state, steps=steps, on_tick=_on_tick)

    # Settle sensors/actors before first frame so spawn layout is stable on video.
    session.wait_for_sensor_frames(max_ticks=12, timeout_s=4.0)
    session.animate_steps(spec, sim_world, steps=4)

    _record_session(recorder, spec, sim_world, "AGENTIC START", {"environment": env_kind, "map": map_name})



    app = build_agentic_graph()

    init = {

        "spec": spec_to_dict(spec),

        "world": asdict(sim_world),

        "policy": policy,

        "trace": [],

        "max_drive_steps": max_steps,

        "max_planner_rounds": 12,

        "perception_backend": "carla",

        "control_mode": os.environ.get("AUTOPASS_CONTROL_MODE", "vehicle"),

    }

    os.environ.setdefault("AUTOPASS_VIDEO_REALTIME", "1")

    record_nodes = frozenset({"execute", "critique_maneuver", "planner", "run_tool"})

    frame_idx = 0

    final_state = init

    for event in app.stream(init, config={"recursion_limit": 300}):

        for node_name, update in event.items():

            final_state = {**final_state, **update}

            if "world" in update:

                sim_world = WorldState(**update["world"])

                set_context(spec, sim_world, "carla")

            if node_name not in record_nodes:

                continue

            extra = {}

            if node_name == "planner" and update.get("last_tool"):

                extra["tool"] = update["last_tool"]

            if node_name == "execute":

                extra["action"] = update.get("trace", [{}])[-1].get("action") if update.get("trace") else ""

            animate = ticks_per_step

            if node_name == "execute":

                animate = 0

            elif node_name in ("planner", "run_tool"):

                animate = 0

            if animate > 0:

                _recording_animate(

                    sim_world,

                    node_name.upper(),

                    animate,

                    graph_step=frame_idx,

                    extra=extra,

                )

            else:

                _record_session(

                    recorder,

                    spec,

                    sim_world,

                    node_name.upper(),

                    extra,

                    graph_step=frame_idx,

                    animate_steps=0,

                )

            frame_idx += 1

            if node_name == "execute":

                _recording_animate(
                    sim_world,
                    "EXECUTE_TAIL",
                    min(2, ticks_per_step),
                    graph_step=frame_idx,
                    extra=extra,
                )



    if "world" in final_state:
        sim_world = WorldState(**final_state["world"])

    _record_session(

        recorder,

        spec,

        sim_world,

        f"DONE {final_state.get('metrics', {}).get('failure_type', 'pending')}",

        {"dsl_revision": final_state.get("dsl", {}).get("revision", 0)},

    )

    mp4 = recorder.write_video(f"{spec.scenario_id}_{env_kind}_agentic.mp4")

    session._demo_frame_sink = None
    get_session().shutdown()

    import json

    trace_path = out_dir / f"{spec.scenario_id}_agentic_trace.json"
    trace_path.write_text(json.dumps(final_state.get("trace", []), indent=2), encoding="utf-8")
    metrics_path = out_dir / f"{spec.scenario_id}_agentic_metrics.json"
    metrics_path.write_text(json.dumps(final_state.get("metrics", {}), indent=2), encoding="utf-8")
    print(f"[CARLA] Trace: {trace_path}")
    print(f"[CARLA] Metrics: {metrics_path}")

    if mp4:

        print(f"[CARLA] Video: {mp4}")

    print(f"[CARLA] Frames: {recorder.frames_dir}")

    return final_state


def main() -> None:

    parser = argparse.ArgumentParser(description="CARLA agentic loop with video")

    parser.add_argument("--scenario", type=str, default="0", help="Scenario index or family name (e.g. clear_safe_pass)")

    parser.add_argument("--environment", choices=["highway", "town", "local"], default="highway")

    parser.add_argument(

        "--curated-corridor",

        action="store_true",

        help="Require curated straight passing corridor (recommended for final video)",

    )

    parser.add_argument(

        "--hero",

        action="store_true",

        help="Use hero corridor mode (maneuver-horizon validation for final video)",

    )

    parser.add_argument(

        "--hero-pass",

        action="store_true",

        help="Run canonical closed-loop demo (agentic graph + CARLA video)",

    )

    parser.add_argument(

        "--policy",

        choices=["autopass", "no_pass"],

        default="autopass",

        help="Hero pass policy (default: autopass)",

    )

    parser.add_argument(

        "--urgency",

        choices=["low", "medium", "high"],

        default="high",

        help="Request urgency for hero pass (default: high)",

    )

    parser.add_argument(

        "--allow-unvalidated",

        action="store_true",

        help="Allow town/local without corridor validation (not for final presentation)",

    )

    parser.add_argument("--out-dir", type=Path, default=Path("runs/carla_watch"))

    parser.add_argument("--steps", type=int, default=40)

    parser.add_argument("--ticks", type=int, default=10)

    args = parser.parse_args()



    os.environ["AUTOPASS_ENVIRONMENT"] = args.environment

    if args.curated_corridor or args.environment == "highway" or args.hero_pass:

        os.environ["AUTOPASS_CARLA_CURATED_CORRIDOR"] = "1"

    if args.hero or args.hero_pass:

        os.environ["AUTOPASS_CARLA_HERO_CORRIDOR"] = "1"

        os.environ["AUTOPASS_CARLA_CORRIDOR_MODE"] = "hero"

    elif args.curated_corridor:

        os.environ.setdefault("AUTOPASS_CARLA_CORRIDOR_MODE", "presentation")

    if args.allow_unvalidated:

        os.environ["AUTOPASS_CARLA_ALLOW_UNVALIDATED"] = "1"

    _ensure_curated_corridor()



    if args.hero_pass:
        from dataclasses import replace

        from autopass.scenarios import showcase_map_for_environment
        from visual_world import curated_demo_scenarios, initialize_world

        family = args.scenario
        map_name = showcase_map_for_environment(args.environment)
        os.environ["AUTOPASS_CARLA_MAP"] = map_name

        use_axis_demo = family in ("6", "clear_safe_pass", "clear_safe_pass_perception")
        if family.isdigit() and int(family) == 6:
            use_axis_demo = True

        if use_axis_demo:
            os.environ.pop("AUTOPASS_CARLA_HERO_CORRIDOR", None)
            os.environ["AUTOPASS_CARLA_CORRIDOR_MODE"] = "presentation"
            os.environ.setdefault("AUTOPASS_CARLA_SKIP_PASS_BOOT_VALIDATE", "1")
            os.environ.setdefault("AUTOPASS_CARLA_MAX_CORRIDOR_REPICK", "0")
            spec = curated_demo_scenarios()[6]
            spec = replace(spec, route=replace(spec.route, town=map_name))
            family = "clear_safe_pass_perception"
        else:
            from autopass.benchmark_catalog import FAMILY_TO_DEMO_ID
            from autopass.hero_demo import resolve_hero_scenario

            if family.isdigit():
                demos = curated_demo_scenarios()
                idx = int(family)
                family = next(
                    (f for f, did in FAMILY_TO_DEMO_ID.items() if did == demos[idx].scenario_id),
                    "clear_safe_pass",
                )
            spec, _case = resolve_hero_scenario(family, urgency=args.urgency, environment=args.environment)

        world = initialize_world(spec)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        print("AutoPass CARLA closed-loop (agentic — vision-grounded pass/wait)")
        print(f"  Family: {family} -> {spec.scenario_id}")
        print(f"  Policy: {args.policy}  Urgency: {args.urgency}")
        print(f"  Map: {spec.route.town}")
        print(f"  Output: {args.out_dir}\n")
        os.environ["AUTOPASS_CONTROL_MODE"] = "vehicle"
        os.environ.setdefault("AUTOPASS_EXECUTE_DT_S", "0.35")
        os.environ.setdefault("AUTOPASS_VIDEO_REALTIME", "1")
        os.environ.setdefault("AUTOPASS_DEMO_DENSE_FRAMES", "1")
        run_agentic_carla_loop(spec, world, args.out_dir, args.ticks, args.steps, policy=args.policy)
        return



    from autopass.scenarios import assert_carla_environment_allowed, curated_scenarios_by_environment, showcase_map_for_environment

    from visual_world import initialize_world



    os.environ["AUTOPASS_CARLA_MAP"] = showcase_map_for_environment(args.environment)

    assert_carla_environment_allowed(args.environment)



    env_specs = dict(curated_scenarios_by_environment())

    spec = env_specs.get(args.environment) or list(env_specs.values())[0]

    from visual_world import curated_demo_scenarios



    if str(args.scenario).isdigit() and int(args.scenario) < len(curated_demo_scenarios()):

        spec = curated_demo_scenarios()[int(args.scenario)]

        from dataclasses import replace



        spec = replace(spec, route=replace(spec.route, town=os.environ["AUTOPASS_CARLA_MAP"]))



    world = initialize_world(spec)

    args.out_dir.mkdir(parents=True, exist_ok=True)



    print("AutoPass CARLA Watch (agentic)")

    print(f"  Environment: {args.environment} -> {os.environ['AUTOPASS_CARLA_MAP']}")

    print(f"  Scenario: {spec.scenario_id}")

    print(f"  Output: {args.out_dir}\n")



    run_agentic_carla_loop(spec, world, args.out_dir, args.ticks, args.steps)





if __name__ == "__main__":

    main()

