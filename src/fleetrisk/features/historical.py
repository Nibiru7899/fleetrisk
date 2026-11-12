"""Application-relative Home Credit aggregates; not current-loan outcome histories."""

import polars as pl


def bureau_as_of(bureau: pl.DataFrame, cutoff_day: int = 0) -> pl.DataFrame:
    # DAYS_CREDIT timestamps creation only. Updated balances/status have no verified
    # historical availability, so intentionally do not aggregate those attributes.
    return (
        bureau.filter(pl.col("DAYS_CREDIT").is_not_null() & (pl.col("DAYS_CREDIT") <= cutoff_day))
        .group_by("SK_ID_CURR")
        .agg(
            pl.len().alias("prior_credit_count"),
            pl.col("DAYS_CREDIT").max().alias("latest_credit_day"),
        )
        .sort("SK_ID_CURR")
    )


def installments_as_of(installments: pl.DataFrame, cutoff_day: int = 0) -> pl.DataFrame:
    return (
        installments.filter(
            pl.col("DAYS_ENTRY_PAYMENT").is_not_null()
            & (pl.col("DAYS_ENTRY_PAYMENT") <= cutoff_day)
            & (pl.col("DAYS_INSTALMENT") <= cutoff_day)
        )
        .group_by("SK_ID_CURR")
        .agg(
            pl.len().alias("observed_payment_rows"),
            pl.col("AMT_PAYMENT").sum().alias("historical_payment_amount"),
        )
        .sort("SK_ID_CURR")
    )
