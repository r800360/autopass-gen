"""
Urgency-conditioned benchmark harness for AutoPass-Gen.

Usage:
  python -m autopass.benchmark --out-dir runs/benchmark_urgency --n 50 \\
      --policies no_pass,aggressive,ttc_only,autopass --urgencies low,medium,high
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from autopass.benchmark_baselines import run_baseline_episode
from autopass.benchmark_catalog import BenchmarkCase, UrgencyLevel, benchmark_cases, carla_physical_key
from autopass.benchmark_metrics import derive_run_metrics
from autopass.config import apply_production_defaults, get_perception_backend, is_test_mode, require_runtime
from autopass.scenarios import assert_carla_environment_allowed
from visual_world import initialize_world, spec_to_dict


def _parse_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def expand_benchmark_work(
    policies: Sequence[str],
    *,
    urgencies: Optional[Sequence[UrgencyLevel]] = None,
    families: Optional[Sequence[str]] = None,
    environments: Optional[Sequence[str]] = None,
    n: Optional[int] = None,
) -> List[Tuple[BenchmarkCase, str]]:
    """Deterministic (case × policy) list; --n caps total rows after expansion."""
    cases = benchmark_cases(
        families=list(families) if families else None,
        urgencies=list(urgencies) if urgencies else None,
        environments=list(environments) if environments else None,
    )
    work: List[Tuple[BenchmarkCase, str]] = []
    for case in cases:
        for policy in policies:
            work.append((case, policy))
    work.sort(key=lambda item: (item[0].environment, item[0].scenario_family, item[0].urgency, item[1]))
    if n is not None:
        work = work[: max(0, n)]
    return work


def _map_name_for_case(case: BenchmarkCase) -> str:
    from autopass.scenarios import carla_map_for_kind, showcase_map_for_environment

    map_name = case.spec.route.town
    if not map_name or map_name == "SyntheticTown":
        map_name = showcase_map_for_environment(
            case.environment if case.environment != "synthetic" else "highway"
        )
    elif case.environment == "highway":
        map_name = showcase_map_for_environment("highway")
    return map_name


def _guard_carla_environments(environments: Sequence[str]) -> None:
    for env in environments:
        assert_carla_environment_allowed(env)


def _prepare_carla_for_agentic(case: BenchmarkCase) -> str:
    """Bootstrap or reuse CARLA session; returns bootstrap action label for logging."""
    from autopass.config import AutopassConfigurationError, get_perception_backend
    from perception.carla_scenario import bootstrap_carla_scenario, get_session, run_carla_preflight
    from perception.context import set_context

    if get_perception_backend() != "carla":
        return "skipped"
    world = initialize_world(case.spec)
    map_name = _map_name_for_case(case)
    physical_key = carla_physical_key(map_name, case)
    if not bootstrap_carla_scenario(case.spec, world, map_name=map_name, physical_key=physical_key):
        err = get_session().last_error or "unknown error"
        raise AutopassConfigurationError(
            f"CARLA bootstrap failed for map {map_name}: {err}. "
            "Start CarlaUE4.exe, verify pip install carla==0.9.16, then: python carla_control_smoke.py"
        )
    run_carla_preflight(require_frames=True)
    set_context(case.spec, world, "carla")
    return get_session().last_bootstrap_action


def _timeout_result(case: BenchmarkCase, policy: str) -> Dict[str, Any]:
    world = initialize_world(case.spec)
    return {
        "spec": spec_to_dict(case.spec),
        "world": asdict(world),
        "policy": policy,
        "trace": [{"node": "benchmark", "event": "timeout"}],
        "metrics": {
            "scenario_id": case.scenario_id,
            "policy": policy,
            "collision": False,
            "route_completed": False,
            "time_to_goal_s": world.t_s,
            "failure_type": "timeout",
        },
    }


def _log_bench(msg: str) -> None:
    print(f"[BENCH] {msg}", flush=True)


def run_single(
    case: BenchmarkCase,
    policy: str,
    *,
    max_steps: int = 60,
    seed: int = 0,
    skip_runtime_check: bool = False,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    if policy in ("aggressive", "ttc_only"):
        return run_baseline_episode(
            case.spec,
            policy,  # type: ignore[arg-type]
            max_steps=max_steps,
            fixed_urgency=case.urgency,
        )

    from autopass.graph import run_agentic_episode

    _prepare_carla_for_agentic(case)
    return run_agentic_episode(
        case.spec,
        policy=policy,
        max_drive_steps=max_steps,
        skip_runtime_check=skip_runtime_check,
    )


def run_single_timed(
    case: BenchmarkCase,
    policy: str,
    *,
    max_steps: int = 60,
    seed: int = 0,
    skip_runtime_check: bool = False,
    timeout_s: Optional[float] = None,
) -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    if timeout_s is None or timeout_s <= 0:
        result = run_single(
            case, policy, max_steps=max_steps, seed=seed, skip_runtime_check=skip_runtime_check
        )
        return result, time.perf_counter() - t0

    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(
        run_single,
        case,
        policy,
        max_steps=max_steps,
        seed=seed,
        skip_runtime_check=skip_runtime_check,
    )
    try:
        result = fut.result(timeout=timeout_s)
    except FuturesTimeoutError:
        result = _timeout_result(case, policy)
    pool.shutdown(wait=False, cancel_futures=True)
    return result, time.perf_counter() - t0


def _finalize_carla_after_row() -> None:
    from autopass.config import get_perception_backend

    if get_perception_backend() != "carla":
        return
    try:
        from perception.carla_scenario import get_session

        if get_session().ready:
            get_session().end_episode()
    except Exception:
        pass


def _print_collision_debug(
    *,
    run_index: int,
    case: BenchmarkCase,
    policy: str,
    result: Dict[str, Any],
    row: Dict[str, Any],
) -> None:
    from autopass.config import get_perception_backend

    print(f"[BENCH] DEBUG collision on row {run_index}", flush=True)
    print(f"  policy={policy} scenario={case.scenario_id} urgency={case.urgency}", flush=True)
    print(
        f"  collision_source={row.get('collision_source')} reason={row.get('collision_reason')} "
        f"actor={row.get('collision_actor')} step={row.get('collision_step')}",
        flush=True,
    )
    trace = result.get("trace", [])
    prev_action = "none"
    cur_action = row.get("final_action", "none")
    for t in reversed(trace):
        if t.get("node") == "execute":
            cur_action = t.get("action", cur_action)
            break
    for t in trace:
        if t.get("node") == "execute":
            prev_action = t.get("action", prev_action)
            if t.get("collision"):
                break
    print(f"  previous_action={prev_action} current_action={cur_action}", flush=True)
    if get_perception_backend() == "carla":
        try:
            from perception.carla_scenario import get_session

            session = get_session()
            if session.ready:
                print(f"  transforms={session.actor_transform_snapshot()}", flush=True)
                print(f"  distances={session.actor_distance_snapshot()}", flush=True)
                if hasattr(session, "geometry_debug_snapshot"):
                    print(f"  geometry={session.geometry_debug_snapshot()}", flush=True)
        except Exception as exc:
            print(f"  carla_snapshot_error={exc}", flush=True)


def run_benchmark_batch(
    *,
    out_dir: Path,
    policies: Sequence[str],
    urgencies: Optional[Sequence[UrgencyLevel]] = None,
    families: Optional[Sequence[str]] = None,
    environments: Optional[Sequence[str]] = None,
    n: Optional[int] = None,
    max_steps: int = 60,
    seed: int = 42,
    skip_runtime_check: bool = False,
    timeout_s: Optional[float] = None,
    debug_first_collision: bool = False,
) -> List[Dict[str, Any]]:
    env_list = list(environments) if environments else (["synthetic"] if skip_runtime_check else ["highway"])
    if not skip_runtime_check and get_perception_backend() == "carla":
        _guard_carla_environments(env_list)

    work = expand_benchmark_work(
        policies,
        urgencies=urgencies,
        families=families,
        environments=environments,
        n=n,
    )
    backend = get_perception_backend() if not skip_runtime_check else "visual"
    total = len(work)

    out_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for i, (case, policy) in enumerate(work, start=1):
        run_seed = seed + i * 9973
        run_id = f"{i:03d}_{policy}_{case.scenario_id}_{case.urgency}"
        initial_world = asdict(initialize_world(case.spec))
        carla_action = "n/a"
        if backend == "carla" and policy not in ("aggressive", "ttc_only"):
            try:
                from perception.carla_scenario import get_session

                carla_action = (
                    get_session().last_bootstrap_action if get_session().ready else "will_bootstrap"
                )
            except Exception:
                carla_action = "unknown"

        _log_bench(
            f"{i}/{total} start policy={policy} scenario={case.scenario_id} "
            f"family={case.scenario_family} urgency={case.urgency} env={case.environment} "
            f"backend={backend} carla={carla_action}"
        )

        result, duration_s = run_single_timed(
            case,
            policy,
            max_steps=max_steps,
            seed=run_seed,
            skip_runtime_check=skip_runtime_check,
            timeout_s=timeout_s,
        )

        if backend == "carla" and policy not in ("aggressive", "ttc_only"):
            try:
                from perception.carla_scenario import get_session

                carla_action = get_session().last_bootstrap_action
            except Exception:
                pass

        row = derive_run_metrics(case, policy, result)
        if result.get("metrics", {}).get("failure_type") == "timeout":
            row["failure_type"] = "timeout"
            row["trace_complete"] = False
        row["seed"] = run_seed
        row["run_id"] = run_id
        row["run_duration_s"] = round(duration_s, 2)
        rows.append(row)

        _log_bench(
            f"{i}/{total} done collision={row.get('collision')} time={row.get('time_to_goal_s')}s "
            f"pass_attempts={row.get('pass_attempts')} failure={row.get('failure_type', 'none')} "
            f"duration={duration_s:.1f}s carla={carla_action}"
        )

        if row.get("collision"):
            _print_collision_debug(run_index=i, case=case, policy=policy, result=result, row=row)

        trace_doc = {
            "run_id": run_id,
            "run_index": i,
            "policy_name": policy,
            "scenario_id": case.scenario_id,
            "scenario_family": case.scenario_family,
            "urgency": case.urgency,
            "environment": case.environment,
            "spec": spec_to_dict(case.spec),
            "initial_world": initial_world,
            "initial_metrics": {
                "policy_name": policy,
                "scenario_id": case.scenario_id,
                "urgency": case.urgency,
                "time_to_goal_s": 0.0,
                "collision": False,
                "pass_attempts": 0,
            },
            "metrics": row,
            "final_metrics": row,
            "trace": result.get("trace", []),
            "dsl": result.get("dsl"),
            "perception_log": (result.get("dsl") or {}).get("perception_log", []),
            "verification_log": (result.get("dsl") or {}).get("verification_log", []),
            "execution_log": (result.get("dsl") or {}).get("execution_log", []),
            "world_belief": (result.get("dsl") or {}).get("world_belief"),
            "collision_events": [
                e
                for e in ((result.get("dsl") or {}).get("execution_log") or [])
                if (e.get("data") or {}).get("collision")
            ],
        }
        trace_name = f"{i:03d}_{policy}_{case.scenario_id}_{case.urgency}.json"
        (traces_dir / trace_name).write_text(json.dumps(trace_doc, indent=2), encoding="utf-8")

        _finalize_carla_after_row()

        if debug_first_collision and row.get("collision"):
            _finalize_carla_after_row()
            break

    csv_path = out_dir / "runs.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AutoPass urgency benchmark batch runner")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/benchmark_urgency"))
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Cap total benchmark rows after expanding cases × policies (deterministic order)",
    )
    parser.add_argument("--policies", type=str, default="no_pass,aggressive,ttc_only,autopass")
    parser.add_argument("--urgencies", type=str, default="low,medium,high")
    parser.add_argument("--families", type=str, default="", help="Comma-separated subset of scenario families")
    parser.add_argument(
        "--environments",
        type=str,
        default=None,
        help="synthetic,highway,town,local (production default: highway)",
    )
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--timeout-s", type=float, default=None, help="Wall-clock timeout per row (seconds)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug-first-collision",
        action="store_true",
        help="Stop after first collision row and print CARLA diagnostic snapshot",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force AUTOPASS_TEST_MODE=1 (visual + mock LLM, no CARLA/API required)",
    )
    parser.add_argument(
        "--curated-corridor",
        action="store_true",
        help="Require curated straight CARLA passing corridor (sets AUTOPASS_CARLA_CURATED_CORRIDOR=1)",
    )
    parser.add_argument(
        "--hero-corridor",
        action="store_true",
        help="Use hero corridor mode for CARLA benchmark (maneuver-horizon validation)",
    )
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="Allow town/local CARLA environments without corridor validation (presentation not recommended)",
    )
    args = parser.parse_args(argv)

    if args.offline:
        import os

        os.environ["AUTOPASS_TEST_MODE"] = "1"
    if args.curated_corridor:
        import os

        os.environ["AUTOPASS_CARLA_CURATED_CORRIDOR"] = "1"
        os.environ.setdefault("AUTOPASS_CARLA_CORRIDOR_MODE", "presentation")
    if args.hero_corridor:
        import os

        os.environ["AUTOPASS_CARLA_HERO_CORRIDOR"] = "1"
        os.environ["AUTOPASS_CARLA_CORRIDOR_MODE"] = "hero"
        os.environ["AUTOPASS_CARLA_CURATED_CORRIDOR"] = "1"
    if args.allow_unvalidated:
        import os

        os.environ["AUTOPASS_CARLA_ALLOW_UNVALIDATED"] = "1"
    apply_production_defaults()
    if not is_test_mode():
        require_runtime()

    policies = _parse_list(args.policies)
    urgencies = _parse_list(args.urgencies)  # type: ignore[assignment]
    families = _parse_list(args.families) or None
    if args.environments:
        environments = _parse_list(args.environments)
    else:
        environments = ["synthetic"] if is_test_mode() else ["highway"]

    timeout_s = args.timeout_s
    if timeout_s is None and not is_test_mode():
        timeout_s = 240.0

    rows = run_benchmark_batch(
        out_dir=args.out_dir,
        policies=policies,
        urgencies=urgencies,  # type: ignore[arg-type]
        families=families,
        environments=environments,
        n=args.n,
        max_steps=args.max_steps,
        seed=args.seed,
        skip_runtime_check=is_test_mode(),
        timeout_s=timeout_s,
        debug_first_collision=args.debug_first_collision,
    )
    print(f"Wrote {len(rows)} runs to {args.out_dir / 'runs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
