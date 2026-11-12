from pathlib import Path

import polars as pl
import pytest

from fleetrisk.config import contained_path
from fleetrisk.data.validation import validate


def test_fixture_quality(source: Path) -> None:
    result = validate(source, "ltfs")
    assert result["valid"] and result["row_count"] == 300
    assert result["labels"]["positive"] + result["labels"]["negative"] == 300
    assert result["duplicate_keys"] == 0


@pytest.mark.parametrize(
    "damage,expected",
    [
        ("missing", "missing_required_columns"),
        ("duplicate", "duplicate_source_keys"),
        ("numeric", "invalid_numeric_values"),
        ("target", "invalid_target"),
        ("negative", "negative_amount_or_ratio::asset_cost"),
        ("ltv", "ltv_outside_percent_range"),
    ],
)
def test_invalid_records_not_discarded(source: Path, damage: str, expected: str) -> None:
    frame = pl.read_csv(source)
    if damage == "missing":
        frame = frame.drop("asset_cost")
    elif damage == "duplicate":
        frame = pl.concat([frame, frame.head(1)])
    elif damage == "numeric":
        frame = frame.with_columns(pl.lit("broken").alias("asset_cost"))
    elif damage == "target":
        frame = frame.with_columns(pl.lit(0.5).alias("loan_default"))
    elif damage == "negative":
        frame = frame.with_columns(pl.lit(-1).alias("asset_cost"))
    else:
        frame = frame.with_columns(pl.lit(101).alias("ltv"))
    frame.write_csv(source)
    result = validate(source, "ltfs")
    assert not result["valid"] and expected in result["errors"]


@pytest.mark.parametrize("relative", ["../escape.csv", "..\\escape.csv", "C:\\Windows\\win.ini"])
def test_path_traversal(tmp_path: Path, relative: str) -> None:
    with pytest.raises(ValueError):
        contained_path(tmp_path, relative)
