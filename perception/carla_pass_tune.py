#!/usr/bin/env python3
"""Autonomously tune CARLA pass control from pass_quality objective (requires Carla)."""
from __future__ import annotations

import argparse
import os
import sys

from autopass.control_tune import (
    ControlParameterTuner,
    apply_control_profile,
    clear_control_profile_env,
    save_control_profile,
    score_maneuver_result,
)
from autopass.carla_tuning import apply_saved_profile_if_enabled


def _run_one_trial(*, lane_threshold_m: float, total_max_s: float):
    from autopass.hero_demo import resolve_hero_scenario
    from perception.carla_pass_maneuver import run_scripted_pass_maneuver
    from perception.carla_scenario import bootstrap_carla_scenario, get_session
    from visual_world import initialize_world

    os.environ.setdefault("AUTOPASS_CARLA_CURATED_CORRIDOR", "1")
    os.environ.setdefault("AUTOPASS_CARLA_HERO_CORRIDOR", "1")
    os.environ.setdefault("AUTOPASS_CARLA_CORRIDOR_MODE", "hero")
    os.environ.setdefault("AUTOPASS_ENVIRONMENT", "highway")
    os.environ.setdefault("AUTOPASS_CARLA_SKIP_PASS_BOOT_VALIDATE", "1")
    os.environ["AUTOPASS_CARLA_PASS_SMOKE"] = "1"
    os.environ["AUTOPASS_CARLA_USE_PROFILE"] = "0"

    spec, _case = resolve_hero_scenario("clear_safe_pass", urgency="high", environment="highway")
    world = initialize_world(spec)
    map_name = spec.route.town or os.environ.get("AUTOPASS_CARLA_MAP", "Town04")
    if not bootstrap_carla_scenario(spec, world, map_name=map_name):
        session = get_session()
        raise RuntimeError(session.last_error or "bootstrap failed")

    session = get_session()
    result = run_scripted_pass_maneuver(
        session,
        spec,
        world,
        lane_fail_m=lane_threshold_m,
        total_max_s=total_max_s,
        use_state_machine=True,
        verbose=False,
    )
    session.shutdown()
    return result


def run_tune(*, trials: int, lane_threshold_m: float, total_max_s: float) -> int:
    try:
        import carla  # noqa: F401
    except ImportError:
        print("FAIL: carla package not installed in this interpreter.", flush=True)
        return 1

    tuner = ControlParameterTuner()
    print(f"Tuning {trials} trials (objective from pass_quality) ...", flush=True)
    print(f"  baseline profile: {tuner.best_profile}", flush=True)

    for i in range(trials):
        clear_control_profile_env()
        profile = tuner.suggest()
        apply_control_profile(profile)
        print(f"\n--- trial {i + 1}/{trials} ---", flush=True)
        print(f"  profile: max_steer={profile.max_steer} lc_lat={profile.lane_change_lateral_mult}", flush=True)
        try:
            result = _run_one_trial(lane_threshold_m=lane_threshold_m, total_max_s=total_max_s)
        except Exception as exc:
            print(f"  trial error: {exc}", flush=True)
            from autopass.pass_quality import PassQualityReport

            tuner.observe(
                profile,
                -100.0,
                PassQualityReport(False, ["trial_error"], 999.0, 999.0, 999.0, 999.0, 0),
            )
            continue
        score, quality = score_maneuver_result(result)
        tuner.observe(profile, score, quality)
        print(f"  score={score:.1f} quality_ok={quality.ok} issues={quality.issues}", flush=True)
        print(
            f"  merged={result.merged_back} complete={result.pass_complete} "
            f"pass_lane={result.pass_lane_used}",
            flush=True,
        )

    print(f"\nBest score={tuner.best_score:.1f}", flush=True)
    if tuner.best_score <= 0:
        print(
            "No successful profile (score <= 0). Parameter search cannot fix spawn geometry "
            "or maneuver logic — fix those first. Do not set AUTOPASS_CARLA_USE_PROFILE=1.",
            flush=True,
        )
        return 1
    path = save_control_profile(tuner.best_profile)
    print(f"Saved profile → {path.resolve()}", flush=True)
    print("Re-run: $env:AUTOPASS_CARLA_USE_PROFILE='1'; python -m perception.carla_pass_smoke", flush=True)
    return 0


def main(argv=None) -> int:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    os.environ.setdefault("AUTOPASS_PERCEPTION_BACKEND", "carla")

    parser = argparse.ArgumentParser(description="Autotune CARLA pass executor from pass_quality")
    parser.add_argument("--trials", type=int, default=6, help="Coordinate-search trials in CARLA")
    parser.add_argument("--lane-threshold", type=float, default=1.5)
    parser.add_argument("--total-max-s", type=float, default=45.0)
    args = parser.parse_args(argv)
    return run_tune(
        trials=max(2, args.trials),
        lane_threshold_m=args.lane_threshold,
        total_max_s=args.total_max_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
