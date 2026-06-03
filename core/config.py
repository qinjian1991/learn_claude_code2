from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
