from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.log_config import configure_logging


configure_logging()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: SecretStr | None = None
    anthropic_base_url: str | None = None
    anthropic_model: str = "claude-sonnet-4-5"
    model_temperature: float = 0
    model_max_tokens: int | None = 4096
    model_transient_retry_count: int = 3
    model_transient_retry_initial_seconds: float = 1.0
    model_transient_retry_max_seconds: float = 12.0
    context_token_threshold: int = 120_000
    reactive_compact_retry_count: int = 1
    subagent_default_max_turns: int = 5
    subagent_max_turns: int = 10
    checkpoint_db_path: str = "data/agent_checkpoints.sqlite"
    store_db_path: str = "data/agent_store.sqlite"
    thread_id_path: str = "data/current_thread_id.txt"
    mcp_config_path: str = "mcp.json"
    mcp_discovery_timeout_seconds: float = 10.0
    mcp_tool_timeout_seconds: float = 60.0
    user_id: str = "local-user"
    project_id: str = "learn_claude_code2"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
