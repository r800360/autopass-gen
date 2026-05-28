#!/usr/bin/env python3
"""Quick check: CARLA server + Python API + optional scenario spawn."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def main() -> None:
    parser = argparse.ArgumentParser(description="CARLA connectivity and spawn smoke test")
    parser.add_argument("--sensors", action="store_true", help="Also verify RGB/depth/seg sensor frames")
    parser.add_argument("--lane-follow", action="store_true", help="Run lane-keeping smoke (ego follow travel lane)")
    parser.add_argument("--corridor", action="store_true", help="Run curated passing-corridor smoke")
    parser.add_argument("--pass-maneuver", action="store_true", help="Run scripted pass-maneuver smoke")
    args = parser.parse_args()

    if args.pass_maneuver:
        from perception.carla_pass_smoke import main as pass_main

        raise SystemExit(pass_main())

    if args.corridor:
        from perception.carla_corridor_smoke import main as corridor_main

        raise SystemExit(corridor_main())

    if args.lane_follow:
        from perception.carla_lane_smoke import main as lane_main

        raise SystemExit(lane_main())

    if args.sensors:
        from perception.carla_sensor_smoke import main as sensor_main

        raise SystemExit(sensor_main())

    print("1) Testing import carla ...")
    try:
        import carla
    except ImportError as e:
        print(f"   FAIL: {e}")
        print("   Fix: pip install carla==0.9.16  (Python 3.10–3.12 only)")
        sys.exit(1)
    print("   OK")

    print("2) Connecting to CarlaUE4 (127.0.0.1:2000) ...")
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    try:
        ver = client.get_server_version()
    except Exception as e:
        print(f"   FAIL: {e}")
        print("   Fix: start CarlaUE4.exe and wait for the town to load.")
        sys.exit(1)
    print(f"   OK — server version: {ver}")

    print("3) Loading Town04 and spawning ego ...")
    from visual_world import curated_demo_scenarios, initialize_world
    from perception.carla_scenario import bootstrap_carla_scenario

    os.environ.setdefault("AUTOPASS_CARLA_CURATED_CORRIDOR", "1")
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    os.environ["AUTOPASS_PERCEPTION_BACKEND"] = "carla"
    if bootstrap_carla_scenario(spec, world, map_name="Town04"):
        print("   OK — you should SEE vehicles in the CARLA window.")
        print("   Press Ctrl+C when done; actors will be destroyed.")
        try:
            import time
            for _ in range(300):
                from perception.carla_scenario import get_session
                get_session().tick()
                time.sleep(0.05)
        except KeyboardInterrupt:
            from perception.carla_scenario import get_session
            get_session().shutdown()
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
