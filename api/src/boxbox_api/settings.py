"""API settings from the environment (``BOXBOX_*``)."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from boxbox.config import Profile
from boxbox.paths import artifacts_root


class Settings(BaseSettings):
    """Where artifacts live and which profile answers requests without ``profile``."""

    model_config = SettingsConfigDict(env_prefix="BOXBOX_", extra="ignore")

    artifacts_dir: Path = artifacts_root()
    default_profile: Profile = "full"
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
