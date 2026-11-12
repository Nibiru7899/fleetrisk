from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FLEETRISK_", env_file=".env", extra="ignore")

    database_url: str = (
        "postgresql+psycopg://fleetrisk:local-development-only@127.0.0.1:5432/fleetrisk"
    )
    source_root: Path = Path("data/raw")
    artifact_root: Path = Path("artifacts")
    processed_root: Path = Path("data/processed")
    api_key: str = ""
    model_id: str = ""
    allow_synthetic: bool = False
    mlflow_uri: str = "sqlite:///mlruns/tracking.db"
    max_source_bytes: int = Field(default=8 * 1024**3, gt=0)


def contained_path(root: Path, relative: str) -> Path:
    base = root.resolve()
    path = (base / relative).resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError("Path must identify a file within the configured root")
    return path
