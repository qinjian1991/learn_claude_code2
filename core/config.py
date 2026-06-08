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
    context_token_threshold: int = 120_000
    reactive_compact_retry_count: int = 1
    checkpoint_db_path: str = "data/agent_checkpoints.sqlite"
    store_db_path: str = "data/agent_store.sqlite"
    thread_id_path: str = "data/current_thread_id.txt"
    user_id: str = "local-user"
    project_id: str = "learn_claude_code2"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
