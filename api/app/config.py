"""
Application configuration, loaded from environment variables (and .env
locally). Kept separate from schemas.py deliberately -- these are
runtime settings, not request/response contracts.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    redis_url: str = "redis://localhost:6379"
    rate_limit: str = "10/minute"


settings = Settings()