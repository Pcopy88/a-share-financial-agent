"""Locally maintained model allowlist; UI receives profile names, never secrets."""

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from annual_report_agent.generation.cloud_client import DeepSeekClient

_PROFILE_FIELDS = {"provider", "model_id", "model_env", "base_url_env", "api_key_env",
                   "thinking", "output_format", "snapshot_id", "region", "pricing", "note"}


def load_profiles(settings):
    path = Path(settings.profiles_file)
    if not path.is_file():
        return {"default": "deepseek_flash", "profiles": {"deepseek_flash": {
            "provider": "deepseek", "model_id": settings.deepseek_model,
            "base_url_env": "DEEPSEEK_BASE_URL", "api_key_env": "DEEPSEEK_API_KEY",
            "thinking": "disabled", "output_format": "json_object"}}}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("default") not in data.get("profiles", {}):
        raise ValueError("invalid model profile registry")
    for profile in data["profiles"].values():
        if not isinstance(profile, dict) or not profile.get("model_id"):
            raise ValueError("profile must define model_id")
        if any(key in profile for key in ("api_key", "token", "password", "authorization")):
            raise ValueError("profiles must reference environment names, never literal secrets")
        if set(profile) - _PROFILE_FIELDS:
            raise ValueError("unsupported model profile parameter")
        for key, fixed in (("provider", "deepseek"), ("thinking", "disabled"),
                           ("output_format", "json_object")):
            if profile.get(key, fixed) != fixed:
                raise ValueError(f"unsupported model profile {key}")
        if profile.get("model_env"):
            if profile["model_env"] != "DEEPSEEK_MODEL":
                raise ValueError("only the public DEEPSEEK_MODEL setting can override model_id")
            profile["model_id"] = settings.deepseek_model
    return data


def effective_profile(settings, name="default"):
    """Public, actual runtime contract. Fixed stage limits are not YAML knobs.

    planner/composer set max_tokens=3072/4096; DeepSeekClient sets a 90 s
    timeout (10 s connect), disabled thinking and JSON output per request.
    request_timeout applies to other adapters, not these generation requests.
    Response model/system_fingerprint remain per-response usage observations.
    """
    data = load_profiles(settings)
    selected = data["default"] if name == "default" else name
    if selected not in data["profiles"]:
        raise ValueError("unknown model profile")
    profile = data["profiles"][selected]
    env_url = profile.get("base_url_env", "DEEPSEEK_BASE_URL")
    url = settings.deepseek_base_url if env_url == "DEEPSEEK_BASE_URL" else os.environ.get(env_url, "")
    if url:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("unsupported model endpoint; requires a public HTTPS API root")
    live = settings.mode == "live"
    actual = {"mode": settings.mode, "provider": "deepseek" if live else "local_mock",
        "model_id": profile["model_id"] if live else "mock-rule-based",
        "requested_model_id": profile["model_id"], "endpoint": url.rstrip("/") if live else None,
        "thinking": "disabled" if live else None, "output_format": "json_object" if live else None,
        "stream": False, "stage_max_tokens": {"planner": 3072, "composer": 4096} if live else {},
        "timeout_seconds": {"default": 90, "connect": 10} if live else {},
        "transport_max_retries": 2 if live else 0, "planner_max_schema_repairs": 1 if live else 0,
        "context_token_budget": settings.context_token_budget,
        "temperature": "provider_default_not_set" if live else None,
        "seed": "not_set" if live else None,
        "snapshot_policy": "observe response model and system_fingerprint; alias is not a pinned snapshot" if live else "simulation"}
    digest = hashlib.sha256(json.dumps(actual, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode()).hexdigest()
    return {"profile_name": selected, **actual, "sha256": digest}


def public_profiles(settings):
    data = load_profiles(settings)
    allowed = {"provider", "model_id", "thinking", "output_format", "region", "snapshot_id", "note"}
    return {"default": data["default"], "profiles": {
        name: {key: value for key, value in profile.items() if key in allowed}
        for name, profile in data["profiles"].items()}}


def create_client(settings, name="default"):
    data = load_profiles(settings)
    name = data["default"] if name == "default" else name
    if name not in data["profiles"]:
        raise ValueError("unknown model profile")
    if settings.mode == "mock":
        return None
    profile = data["profiles"][name]
    env_key = profile.get("api_key_env", "DEEPSEEK_API_KEY")
    key = (settings.deepseek_api_key.get_secret_value() if env_key == "DEEPSEEK_API_KEY"
           else os.environ.get(env_key, ""))
    env_url = profile.get("base_url_env", "DEEPSEEK_BASE_URL")
    url = settings.deepseek_base_url if env_url == "DEEPSEEK_BASE_URL" else os.environ.get(env_url, "")
    if not key or not url:
        raise ValueError(f"model configuration missing: {env_key} or {env_url}")
    return DeepSeekClient(api_key=key, base_url=url, model=profile["model_id"])
