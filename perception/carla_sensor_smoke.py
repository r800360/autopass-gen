#!/usr/bin/env python3
"""CARLA sensor smoke test: verify RGB/depth/seg produce frames after spawn."""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    os.environ.setdefault("AUTOPASS_PERCEPTION_BACKEND", "carla")

    parser = argparse.ArgumentParser(description="CARLA RGB/depth/seg sensor smoke test")
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Use minimal ego-only bootstrap (isolates sensors from full scenario)",
    )
    args = parser.parse_args(argv)

    print("1) import carla ...", flush=True)
    try:
        import carla  # noqa: F401
    except ImportError as e:
        print(f"   FAIL: {e}", flush=True)
        return 1
    print("   OK", flush=True)

    from perception.carla_scenario import get_session, run_sensor_smoke

    if args.minimal:
        print("2) minimal bootstrap (Town04 + ego + sensors) ...", flush=True)
    else:
        print("2) bootstrap Town04 scenario ...", flush=True)

    code = run_sensor_smoke(minimal=args.minimal, verbose=True)
    if code != 0:
        session = get_session()
        print("\n--- SENSOR DIAGNOSTICS ---", flush=True)
        print(session.sensor_full_diagnostic(), flush=True)
        print("--- END DIAGNOSTICS ---", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
