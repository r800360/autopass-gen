import os

import pytest

from autopass.config import AutopassConfigurationError, apply_production_defaults, is_test_mode, require_openai


def test_test_mode_is_active_under_pytest():
    assert is_test_mode()


def test_production_defaults_in_test_mode_stay_offline():
    apply_production_defaults()
    assert os.environ["AUTOPASS_MOCK_LLM"] == "1"
    assert os.environ["AUTOPASS_PERCEPTION_BACKEND"] == "visual"


def test_require_openai_raises_without_key_when_not_mock(monkeypatch):
    monkeypatch.delenv("AUTOPASS_TEST_MODE", raising=False)
    monkeypatch.setenv("AUTOPASS_MOCK_LLM", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(AutopassConfigurationError, match="OPENAI_API_KEY"):
        require_openai()
