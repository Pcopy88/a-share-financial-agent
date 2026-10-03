import pytest

from annual_report_agent.config import Settings
from annual_report_agent.generation.profiles import create_client, public_profiles


def test_default_profile_uses_explicit_environment_model_setting(monkeypatch):
    captured = {}
    monkeypatch.setattr("annual_report_agent.generation.profiles.DeepSeekClient",
                        lambda **kwargs: captured.update(kwargs) or object())
    settings = Settings(_env_file=None, mode="live", deepseek_api_key="test-only-key",
                        deepseek_model="configured-model")
    create_client(settings)
    assert captured["model"] == "configured-model"
    assert public_profiles(settings)["profiles"]["deepseek_flash"]["model_id"] == "configured-model"
    assert "test-only-key" not in str(public_profiles(settings))


def test_mock_mode_rejects_unknown_profile():
    with pytest.raises(ValueError, match="unknown model profile"):
        create_client(Settings(_env_file=None, mode="mock"), "unknown")


@pytest.mark.parametrize("field,value", [("temperature", 0.2), ("max_output_tokens", 99),
    ("timeout_seconds", 60), ("context_token_budget", 6000), ("thinking", "enabled"),
    ("output_format", "text"), ("provider", "unsupported")])
def test_profile_rejects_options_that_adapter_does_not_apply(tmp_path, field, value):
    import yaml
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml.safe_dump({"default": "custom", "profiles": {"custom": {
        "provider": "deepseek", "model_id": "model", field: value}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        create_client(Settings(_env_file=None, profiles_file=path, mode="mock"))


def test_effective_profile_records_actual_stage_settings_and_secret_free_hash():
    from annual_report_agent.generation.profiles import effective_profile
    settings = Settings(_env_file=None, mode="live", deepseek_api_key="not-a-real-secret",
                        deepseek_model="chosen-model", request_timeout=13, context_token_budget=7000)
    actual = effective_profile(settings)
    assert actual["model_id"] == "chosen-model"
    assert actual["stage_max_tokens"] == {"planner": 3072, "composer": 4096}
    assert actual["timeout_seconds"] == {"default": 90, "connect": 10}
    assert actual["context_token_budget"] == 7000
    assert actual["thinking"] == "disabled"
    assert actual["mode"] == "live"
    assert "not-a-real-secret" not in str(actual)
    assert len(actual["sha256"]) == 64
    assert effective_profile(settings.model_copy(update={"deepseek_model": "other"}))["sha256"] != actual["sha256"]
    assert effective_profile(settings.model_copy(update={"mode": "mock"}))["sha256"] != actual["sha256"]
    assert effective_profile(settings.model_copy(update={"request_timeout": 22}))["sha256"] == actual["sha256"]
