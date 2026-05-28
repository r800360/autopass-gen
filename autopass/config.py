"""
Runtime configuration: production defaults and fail-fast validation.

Set AUTOPASS_TEST_MODE=1 (pytest does this automatically) for offline mock/visual tests.
"""
from __future__ import annotations

import os
import sys


class AutopassConfigurationError(Exception):
    """Missing API key, CARLA server, or incompatible install — fix and retry."""


def is_test_mode() -> bool:
    return os.environ.get("AUTOPASS_TEST_MODE", "").strip() in ("1", "true", "True")


def mock_llm_enabled() -> bool:
    if is_test_mode():
        return os.environ.get("AUTOPASS_MOCK_LLM", "1").strip() in ("1", "true", "True")
    return os.environ.get("AUTOPASS_MOCK_LLM", "0").strip() in ("1", "true", "True")


def perception_backend() -> str:
    if is_test_mode():
        return os.environ.get("AUTOPASS_PERCEPTION_BACKEND", "visual")
    return os.environ.get("AUTOPASS_PERCEPTION_BACKEND", "carla")


get_perception_backend = perception_backend


def control_mode() -> str:
    if is_test_mode():
        return os.environ.get("AUTOPASS_CONTROL_MODE", "kinematic")
    return os.environ.get("AUTOPASS_CONTROL_MODE", "vehicle")


def apply_production_defaults() -> None:
    """Production: real LLM + CARLA perception + vehicle physics."""
    if is_test_mode():
        os.environ.setdefault("AUTOPASS_MOCK_LLM", "1")
        os.environ.setdefault("AUTOPASS_PERCEPTION_BACKEND", "visual")
        os.environ.setdefault("AUTOPASS_CONTROL_MODE", "kinematic")
        return
    os.environ["AUTOPASS_MOCK_LLM"] = os.environ.get("AUTOPASS_MOCK_LLM", "0")
    os.environ["AUTOPASS_PERCEPTION_BACKEND"] = os.environ.get("AUTOPASS_PERCEPTION_BACKEND", "carla")
    os.environ["AUTOPASS_CONTROL_MODE"] = os.environ.get("AUTOPASS_CONTROL_MODE", "vehicle")


def require_openai() -> None:
    if is_test_mode() or mock_llm_enabled():
        return
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise AutopassConfigurationError(
            "OPENAI_API_KEY is required when AUTOPASS_MOCK_LLM=0. "
            "Export your course API key, or set AUTOPASS_TEST_MODE=1 for offline pytest."
        )


def require_carla_package() -> None:
    if is_test_mode() or perception_backend() != "carla":
        return
    try:
        import carla  # noqa: F401
    except ImportError as e:
        py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        raise AutopassConfigurationError(
            f"Cannot import carla: {e}. "
            f"Current Python {py}. Install: pip install carla==0.9.16 "
            f"(wheel must match cp{sys.version_info.major}{sys.version_info.minor}, e.g. "
            f"C:\\CARLA_0.9.16\\PythonAPI\\carla\\dist\\carla-0.9.16-cp310-cp310-win_amd64.whl)."
        ) from e


def require_carla_server() -> None:
    if is_test_mode() or perception_backend() != "carla":
        return
    require_carla_package()
    import carla

    host = os.environ.get("CARLA_HOST", "127.0.0.1")
    port = int(os.environ.get("CARLA_PORT", "2000"))
    try:
        client = carla.Client(host, port)
        client.set_timeout(8.0)
        ver = client.get_server_version()
        if not ver:
            raise RuntimeError("empty server version")
    except Exception as e:
        raise AutopassConfigurationError(
            f"Cannot connect to CARLA at {host}:{port}: {e}. "
            "Start CarlaUE4.exe and wait for the town to load."
        ) from e


def require_runtime(*, need_carla: bool = True, need_openai: bool = True) -> None:
    """Call at demo/episode entry; raises AutopassConfigurationError on misconfiguration."""
    apply_production_defaults()
    if need_openai:
        require_openai()
    if need_carla and perception_backend() == "carla":
        require_carla_server()


def curated_corridor_enabled() -> bool:
    """When true, CARLA spawns must pass straight non-junction corridor validation."""
    if is_test_mode():
        return os.environ.get("AUTOPASS_CARLA_CURATED_CORRIDOR", "0").strip() in ("1", "true", "True")
    return os.environ.get("AUTOPASS_CARLA_CURATED_CORRIDOR", "1").strip() in ("1", "true", "True")


def allow_unvalidated_carla_environments() -> bool:
    """Explicit opt-in for town/local CARLA maps without corridor validation."""
    return os.environ.get("AUTOPASS_CARLA_ALLOW_UNVALIDATED", "").strip() in ("1", "true", "True")


def hero_corridor_enabled() -> bool:
    """When true, accept hero-validated corridors for demo/benchmark video."""
    return os.environ.get("AUTOPASS_CARLA_HERO_CORRIDOR", "0").strip() in ("1", "true", "True")


def corridor_validation_mode() -> str:
    """strict | presentation | hero — default presentation for automatic scan."""
    mode = os.environ.get("AUTOPASS_CARLA_CORRIDOR_MODE", "presentation").strip().lower()
    if mode not in ("strict", "presentation", "hero"):
        return "presentation"
    return mode
