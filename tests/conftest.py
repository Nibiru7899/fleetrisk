import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from fleetrisk.config import Settings
from fleetrisk.data.db import make_engine
from fleetrisk.data.fixtures import write_ltfs_fixture


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        source_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        artifact_root=tmp_path / "artifacts",
        allow_synthetic=True,
        mlflow_uri=f"sqlite:///{tmp_path.as_posix()}/tracking.db",
    )


@pytest.fixture
def source(settings: Settings) -> Path:
    return write_ltfs_fixture(settings.source_root / "ltfs.csv", 300)


@pytest.fixture
def postgres(settings: Settings, monkeypatch: pytest.MonkeyPatch):
    url = os.environ.get("FLEETRISK_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Real PostgreSQL integration gate blocked: FLEETRISK_TEST_DATABASE_URL unset")
    from sqlalchemy.engine import make_url

    name = make_url(url).database or ""
    if not name.endswith("_test"):
        pytest.fail("Destructive test isolation requires a dedicated database ending in _test")
    settings.database_url = url
    monkeypatch.setenv("FLEETRISK_DATABASE_URL", url)
    engine = make_engine(settings)
    command.upgrade(Config("alembic.ini"), "head")
    with engine.begin() as connection:
        connection.execute(
            text("""TRUNCATE audit_events, financial_assumptions, predictions, model_versions,
            experiments, feature_snapshots, quality_results, historical_credit_rows, loan_observations,
            applications, ingestions, dataset_snapshots, data_sources RESTART IDENTITY CASCADE""")
        )
    yield engine
    engine.dispose()
