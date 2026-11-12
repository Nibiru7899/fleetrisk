from typing import Any

from sqlalchemy import Connection, Engine, create_engine, text

from fleetrisk.config import Settings

REVISION = "0001"


def make_engine(settings: Settings) -> Engine:
    if not settings.database_url.startswith("postgresql+psycopg://"):
        raise ValueError("FleetRisk requires PostgreSQL with psycopg; SQLite is not supported")
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        connect_args={"connect_timeout": 5},
    )


def rows(connection: Connection, sql: str, **parameters: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(text(sql), parameters).mappings()]


def one(connection: Connection, sql: str, **parameters: Any) -> dict[str, Any] | None:
    result = connection.execute(text(sql), parameters).mappings().first()
    return dict(result) if result is not None else None


def ready(engine: Engine) -> bool:
    with engine.connect() as connection:
        return (
            connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
            == REVISION
        )
