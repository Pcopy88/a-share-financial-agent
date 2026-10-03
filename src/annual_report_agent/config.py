"""Configuration with explicit cloud/simulation separation."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    mode: Literal["mock", "live"] = Field("mock", validation_alias="ARA_MODE")
    data_dir: Path = Field(Path("data"), validation_alias="ARA_DATA_DIR")
    output_dir: Path = Field(Path("outputs"), validation_alias="ARA_OUTPUT_DIR")
    profiles_file: Path = Field(Path("model_profiles.yaml"), validation_alias="ARA_PROFILES_FILE")
    deepseek_api_key: SecretStr = Field(SecretStr(""), validation_alias="DEEPSEEK_API_KEY", repr=False)
    dashscope_api_key: SecretStr = Field(SecretStr(""), validation_alias="DASHSCOPE_API_KEY", repr=False)
    deepseek_base_url: str = Field("https://api.deepseek.com", validation_alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field("deepseek-flash", validation_alias="DEEPSEEK_MODEL")
    embedding_url: str = Field("", validation_alias="DASHSCOPE_EMBEDDING_URL")
    embedding_model: str = Field("text-embedding-v3", validation_alias="ARA_EMBEDDING_MODEL")
    embedding_dimension: int = Field(1024, validation_alias="ARA_EMBEDDING_DIMENSION", ge=1)
    retrieval_mode: Literal["dense", "bm25", "hybrid"] = Field(
        "dense", validation_alias="ARA_RETRIEVAL_MODE"
    )
    rerank_enabled: bool = Field(False, validation_alias="ARA_RERANK_ENABLED")
    rerank_device: str = Field("auto", validation_alias="ARA_RERANK_DEVICE")
    rerank_batch_size: int = Field(2, validation_alias="ARA_RERANK_BATCH_SIZE", ge=1, le=32)
    rerank_max_length: int = Field(1024, validation_alias="ARA_RERANK_MAX_LENGTH", ge=512, le=8192)
    api_host: str = Field("127.0.0.1", validation_alias="ARA_API_HOST")
    api_port: int = Field(8000, validation_alias="ARA_API_PORT", ge=1, le=65535)
    ui_port: int = Field(8501, validation_alias="ARA_UI_PORT", ge=1, le=65535)
    api_url: str = Field("http://127.0.0.1:8000", validation_alias="ARA_API_URL")
    max_upload_mb: int = Field(100, validation_alias="ARA_MAX_UPLOAD_MB", ge=1, le=1024)
    context_token_budget: int = Field(6000, validation_alias="ARA_CONTEXT_TOKEN_BUDGET", ge=500)
    max_run_calls: int = Field(24, validation_alias="ARA_MAX_RUN_CALLS", ge=1, le=100)
    max_run_tokens: int = Field(80000, validation_alias="ARA_MAX_RUN_TOKENS", ge=1000)
    request_timeout: float = Field(60, validation_alias="ARA_REQUEST_TIMEOUT", gt=0)

    @property
    def runtime_dir(self) -> Path:
        return self.data_dir.resolve() / self.mode

    @property
    def results_dir(self) -> Path:
        return self.output_dir.resolve() / self.mode

    def missing_cloud_config(self) -> list[str]:
        values = {
            "DEEPSEEK_API_KEY": self.deepseek_api_key.get_secret_value(),
            "DASHSCOPE_API_KEY": self.dashscope_api_key.get_secret_value(),
            "DASHSCOPE_EMBEDDING_URL": self.embedding_url,
        }
        return [key for key, value in values.items() if not value.strip()]

    def public_dict(self) -> dict:
        return self.model_dump(mode="json", exclude={"deepseek_api_key", "dashscope_api_key"})
