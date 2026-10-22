"""EDA summary tables used by the notebooks (pure functions on the lap/feature tables)."""

import numpy as np
import pandas as pd

from boxbox.schema import STINT_KEYS


def race_overview(laps: pd.DataFrame) -> pd.DataFrame:
    """Per race: laps, drivers, clean share, stops, data sources."""
    return (
        laps.groupby("race_id")
        .agg(
            season=("season", "first"),
            laps=("lap", "size"),
            drivers=("driver_id", "nunique"),
            clean_share=("is_clean", "mean"),
            stops=("pit_in", "sum"),
            tyre_source=("tyre_source", "first"),
            era=("era", "first"),
        )
        .reset_index()
    )


def exclusion_table(laps: pd.DataFrame) -> pd.DataFrame:
    """Rows per exclusion reason per race (wide)."""
    reasons = laps["exclusion_reason"].fillna("clean")
    return pd.crosstab(laps["race_id"], reasons)


def stint_table(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per stint: compound, length, source and start/end laps."""
    return (
        laps.groupby(STINT_KEYS)
        .agg(
            compound=("compound", "first"),
            start_lap=("lap", "min"),
            end_lap=("lap", "max"),
            length=("lap", "size"),
            start_tyre_age=("tyre_age", "min"),
            stint_source=("stint_source", "first"),
        )
        .reset_index()
    )


def degradation_curve_table(
    features: pd.DataFrame, value: str = "y_s", max_age: int = 40
) -> pd.DataFrame:
    """Median/IQR of ``value`` minus driver offset by compound and tyre age (clean rows)."""
    rows = features[features["is_training_row"] & (features["tyre_age"] <= max_age)]
    adjusted = rows[value] - rows["driver_offset_s"].fillna(0.0)
    grouped = rows.assign(v=adjusted).groupby(["compound", "tyre_age"])["v"]
    table = grouped.quantile(np.array([0.25, 0.5, 0.75])).unstack()
    table.columns = ["q25_s", "median_s", "q75_s"]
    return table.assign(n=grouped.size()).reset_index()


def raw_vs_corrected(features: pd.DataFrame) -> pd.DataFrame:
    """Median raw, fuel-corrected and field-relative pace by stint lap (confounder view)."""
    rows = features[features["is_training_row"]]
    return (
        rows.groupby("stint_lap")
        .agg(
            raw_s=("lap_time_s", "median"),
            fuel_corrected_s=("lap_fc_s", "median"),
            relative_s=("y_s", "median"),
            n=("lap", "size"),
        )
        .reset_index()
    )
