#!/usr/bin/env python3
"""CARLA scripted pass-maneuver smoke on the hero corridor."""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict


def _import_carla_or_exit() -> int:
    """Import carla from the active interpreter; print a venv hint on failure."""
    try:
        import carla  # noqa: F401
    except ImportError as e:
        exe = sys.executable
        print(f"   FAIL: {e}", flush=True)
        print(
            f"\nUse the Python where you installed carla==0.9.16 (your .venv), not a bare py launcher.\n"
            f"  Current interpreter: {exe}\n"
            f"  Works:    python -m perception.carla_pass_smoke\n"
            f"  Fails if: py -3.10 -m ... points at a different install without the carla wheel.\n"
            f"  Install:  pip install carla==0.9.16  (or the cp310 wheel from your CARLA folder)\n",
            flush=True,
        )
        return 1
    print("   OK", flush=True)
    return 0


def run_pass_smoke(*, lane_threshold_m: float = 1.5, total_max_s: float = 45.0) -> int:
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

    from autopass.carla_tuning import apply_saved_profile_if_enabled
    from autopass.control_tune import load_saved_profile

    os.environ.setdefault("AUTOPASS_CARLA_USE_PROFILE", "0")
    if apply_saved_profile_if_enabled():
        prof = load_saved_profile()
        print(
            f"   tuned profile: max_steer={prof.max_steer} "
            f"lc_lat_mult={prof.lane_change_lateral_mult} "
            f"merge_horizon={prof.corridor_merge_horizon_m}m",
            flush=True,
        )

    spec, _case = resolve_hero_scenario("clear_safe_pass", urgency="high", environment="highway")
    world = initialize_world(spec)
    map_name = spec.route.town or os.environ.get("AUTOPASS_CARLA_MAP", "Town04")
    if not bootstrap_carla_scenario(spec, world, map_name=map_name):
        session = get_session()
        err = session.last_error or "unknown"
        print(f"FAIL: bootstrap failed: {err}", flush=True)
        if "pass_maneuver" in err:
            print(
                "Hint: corridor may be unsuitable for the pass maneuver; "
                "try another spawn or re-run corridor smoke.",
                flush=True,
            )
        return 1

    session = get_session()
    report = session._corridor_report
    if report is not None:
        print(f"   corridor: {report.summary_line()}", flush=True)
        if report.spawn_index is not None:
            print(f"   spawn_index={report.spawn_index}", flush=True)

    print("3) scripted pass maneuver (follow → lane_change → overtake → merge_back) ...", flush=True)
    print(
        "   step  phase         pass_ph    lane_id  tgt_lane  lat_err  center  edge   lead   rear   v     steer",
        flush=True,
    )
    result = run_scripted_pass_maneuver(
        session,
        spec,
        world,
        lane_fail_m=lane_threshold_m,
        total_max_s=total_max_s,
        use_state_machine=True,
        verbose=True,
    )
    session.shutdown()

    phase_center: dict[str, list[float]] = defaultdict(list)
    phase_steer: dict[str, list[float]] = defaultdict(list)
    for rec in result.steps:
        phase_center[rec.scripted_phase].append(float(rec.lane_center_dist_m))
        phase_steer[rec.scripted_phase].append(abs(float(rec.steer)))

    def _phase_line(phase: str) -> str:
        vals = phase_center.get(phase, [])
        if not vals:
            return f"{phase}: n=0"
        vals_sorted = sorted(vals)
        p95_idx = min(len(vals_sorted) - 1, int(0.95 * (len(vals_sorted) - 1)))
        p95 = vals_sorted[p95_idx]
        peak = max(vals_sorted)
        mean = sum(vals_sorted) / len(vals_sorted)
        mean_abs_steer = (
            sum(phase_steer.get(phase, [])) / max(1, len(phase_steer.get(phase, [])))
        )
        return (
            f"{phase}: n={len(vals_sorted)} mean_center={mean:.2f}m "
            f"p95_center={p95:.2f}m peak_center={peak:.2f}m mean_abs_steer={mean_abs_steer:.3f}"
        )

    if result.pass_attempts != 1:
        print(f"FAIL: pass_attempts={result.pass_attempts} (expected 1)", flush=True)
        return 1
    from autopass.pass_quality import score_pass_steps

    quality = score_pass_steps(result.steps)
    print(f"SCORE: ok={quality.ok} issues={quality.issues}", flush=True)

    if not result.ok:
        print(f"FAIL: pass maneuver: {', '.join(result.issues)}", flush=True)
        print("QUALITY:", _phase_line("lane_change"), flush=True)
        print("QUALITY:", _phase_line("overtake"), flush=True)
        print("QUALITY:", _phase_line("merge_back"), flush=True)
        return 1
    print(
        f"PASS: scripted pass ok attempts={result.pass_attempts} "
        f"max_lane={result.max_lane_center_m:.2f}m "
        f"min_edge={result.min_edge_clearance_m:.2f}m "
        f"merged_back={result.merged_back} complete={result.pass_complete}",
        flush=True,
    )
    print("QUALITY:", _phase_line("lane_change"), flush=True)
    print("QUALITY:", _phase_line("overtake"), flush=True)
    print("QUALITY:", _phase_line("merge_back"), flush=True)
    if not quality.ok:
        print(f"WARN: pass quality score failed: {', '.join(quality.issues)}", flush=True)
    return 0


def main(argv=None) -> int:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    os.environ.setdefault("AUTOPASS_PERCEPTION_BACKEND", "carla")

    parser = argparse.ArgumentParser(description="CARLA scripted pass-maneuver smoke test")
    parser.add_argument("--lane-threshold", type=float, default=1.5)
    parser.add_argument("--total-max-s", type=float, default=45.0)
    args = parser.parse_args(argv)

    print("1) import carla ...", flush=True)
    if _import_carla_or_exit() != 0:
        return 1

    print("2) bootstrap hero corridor with pass validation ...", flush=True)
    return run_pass_smoke(lane_threshold_m=args.lane_threshold, total_max_s=args.total_max_s)


if __name__ == "__main__":
    raise SystemExit(main())
