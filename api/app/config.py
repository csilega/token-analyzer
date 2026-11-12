"""
Application configuration, loaded from environment variables (and .env
locally). Kept separate from schemas.py deliberately -- these are
runtime settings, not request/response contracts.
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _split_origins(raw: str) -> list[str]:
    return [o.strip() for o in raw.split(",") if o.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    redis_url: str = "redis://localhost:6379"
    rate_limit: str = "10/minute"

    # REQUIRED -- no safe default exists (fail fast, not fail open).
    # Comma-separated browser origins, e.g.
    #   CORS_ALLOW_ORIGINS=https://app.example.com,http://localhost:5173
    cors_allow_origins: str

    @field_validator("cors_allow_origins")
    @classmethod
    def _validate_origins(cls, v: str) -> str:
        origins = _split_origins(v)
        if not origins:
            raise ValueError("CORS_ALLOW_ORIGINS must list at least one origin")
        for o in origins:
            if o == "*":
                raise ValueError("'*' is not allowed: credentials are enabled")
            if not o.startswith(("http://", "https://")):
                raise ValueError(f"origin {o!r} must start with http:// or https://")
            if o.endswith("/"):
                raise ValueError(f"origin {o!r} must not end with '/'")
        return v

    @property
    def cors_origins(self) -> list[str]:
        return _split_origins(self.cors_allow_origins)


settings = Settings()