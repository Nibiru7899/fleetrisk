"""Deterministic artificial source shapes, never empirical credit-risk evidence."""

import csv
from pathlib import Path

import numpy as np


def write_ltfs_fixture(path: Path, count: int = 600, seed: int = 42) -> Path:
    if count < 100:
        raise ValueError("Modeling fixture requires at least 100 rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "UniqueID",
                "disbursed_amount",
                "asset_cost",
                "ltv",
                "PERFORM_CNS.SCORE",
                "Employment.Type",
                "loan_default",
            ],
        )
        writer.writeheader()
        for i in range(count):
            asset = int(rng.integers(40000, 150000))
            ltv = float(rng.uniform(40, 95))
            score = int(rng.integers(300, 850))
            probability = 1 / (1 + np.exp(-(-1 + (ltv - 70) / 30 - (score - 550) / 200)))
            writer.writerow(
                {
                    "UniqueID": f"SYNTHETIC-{i:06}",
                    "disbursed_amount": round(asset * ltv / 100, 2),
                    "asset_cost": asset,
                    "ltv": round(ltv, 2),
                    "PERFORM_CNS.SCORE": score,
                    "Employment.Type": "Salaried" if i % 2 else "Self employed",
                    "loan_default": int(rng.random() < probability),
                }
            )
    return path
