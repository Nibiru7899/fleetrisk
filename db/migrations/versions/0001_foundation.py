"""Initial PostgreSQL foundation. The adjacent 0001.sql is immutable migration input."""

from pathlib import Path

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    sql = Path(__file__).resolve().parents[1] / "0001.sql"
    op.get_bind().exec_driver_sql(sql.read_text(encoding="utf-8"))


def downgrade() -> None:
    for table in (
        "audit_events",
        "financial_assumptions",
        "predictions",
        "model_versions",
        "experiments",
        "feature_snapshots",
        "quality_results",
        "historical_credit_rows",
        "loan_observations",
        "applications",
    ):
        op.drop_table(table)
    op.drop_constraint("ingestion_snapshot_fk", "ingestions", type_="foreignkey")
    op.drop_table("dataset_snapshots")
    op.drop_table("ingestions")
    op.drop_table("data_sources")
