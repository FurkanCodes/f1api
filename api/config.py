from dataclasses import dataclass
import os


@dataclass
class Settings:
    fastf1_cache_dir: str = os.getenv("FASTF1_CACHE_DIR", "./cache")
    session_cache_max: int = int(os.getenv("SESSION_CACHE_MAX", "3"))
    enable_gzip: bool = os.getenv("ENABLE_GZIP", "0").lower() in {"1", "true", "yes"}
    # CORS
    allowed_origins: list[str] = tuple(os.getenv("ALLOWED_ORIGINS", "*").split(","))  # type: ignore
    allow_credentials: bool = os.getenv("ALLOW_CREDENTIALS", "false").lower() in {"1", "true", "yes"}


def get_settings() -> Settings:
    return Settings()
