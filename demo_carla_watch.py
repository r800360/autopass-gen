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

require_runtime(need_carla=True, need_openai=True)





def _ensure_curated_corridor() -> None:

    os.environ.setdefault("AUTOPASS_CARLA_CURATED_CORRIDOR", "1")

    os.environ.setdefault("AUTOPASS_ENVIRONMENT", "highway")





def _record_session(recorder, spec, world, label: str, extra: dict | None = None) -> None:

    from perception.carla_scenario import get_session



    session = get_session()

    session.animate_steps(spec, world, steps=6)

    pair = session.grab_frame_pair()

    if pair is None:

        frame = session.grab_frame()

        if frame is None:

            return

        rgb, _, _ = frame

        overhead = None

    else:

        rgb, _, _, overhead = pair

    recorder.capture(rgb, t_s=world.t_s, label=label, extra=extra, overhead=overhead)





def run_agentic_carla_loop(spec, world, out_dir: Path, ticks_per_step: int, max_steps: int) -> None:

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

    _record_session(recorder, spec, sim_world, "AGENTIC START", {"environment": env_kind, "map": map_name})



    app = build_agentic_graph()

    init = {

        "spec": spec_to_dict(spec),

        "world": asdict(sim_world),

        "policy": "autopass",

        "trace": [],

        "max_drive_steps": max_steps,

        "perception_backend": "carla",

        "control_mode": os.environ.get("AUTOPASS_CONTROL_MODE", "vehicle"),

    }

    record_nodes = frozenset({"execute", "critique_maneuver", "planner", "run_tool"})

    step = 0

    final_state = init

    for event in app.stream(init, config={"recursion_limit": 250}):

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

            _record_session(recorder, spec, sim_world, node_name.upper(), extra)

            get_session().animate_steps(spec, sim_world, steps=ticks_per_step)

            step += 1

            if step >= max_steps:

                break

        if step >= max_steps:

            break



    _record_session(

        recorder,

        spec,

        sim_world,

        f"DONE {final_state.get('metrics', {}).get('failure_type', 'pending')}",

        {"dsl_revision": final_state.get("dsl", {}).get("revision", 0)},

    )

    mp4 = recorder.write_video(f"{spec.scenario_id}_{env_kind}_agentic.mp4")

    get_session().shutdown()

    if mp4:

        print(f"[CARLA] Video: {mp4}")

    print(f"[CARLA] Frames: {recorder.frames_dir}")





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
        from autopass.benchmark_catalog import FAMILY_TO_DEMO_ID
        from autopass.hero_demo import resolve_hero_scenario

        family = args.scenario
        if family.isdigit():
            from visual_world import curated_demo_scenarios

            demos = curated_demo_scenarios()
            idx = int(family)
            family = next(
                (f for f, did in FAMILY_TO_DEMO_ID.items() if did == demos[idx].scenario_id),
                "clear_safe_pass",
            )
        from visual_world import initialize_world

        spec, _case = resolve_hero_scenario(family, urgency=args.urgency, environment=args.environment)
        world = initialize_world(spec)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        print("AutoPass CARLA closed-loop (agentic — vision-grounded pass/wait)")
        print(f"  Family: {family} → {spec.scenario_id}")
        print(f"  Policy: {args.policy}  Urgency: {args.urgency}")
        print(f"  Map: {spec.route.town}")
        print(f"  Output: {args.out_dir}\n")
        os.environ["AUTOPASS_CONTROL_MODE"] = "vehicle"
        run_agentic_carla_loop(spec, world, args.out_dir, args.ticks, args.steps)
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

    print(f"  Environment: {args.environment} → {os.environ['AUTOPASS_CARLA_MAP']}")

    print(f"  Scenario: {spec.scenario_id}")

    print(f"  Output: {args.out_dir}\n")



    run_agentic_carla_loop(spec, world, args.out_dir, args.ticks, args.steps)





if __name__ == "__main__":

    main()

