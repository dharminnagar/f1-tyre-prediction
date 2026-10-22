"""Clean-lap flags: every row is kept, with ``is_clean`` and the first ``exclusion_reason``."""

import numpy as np
import pandas as pd

from boxbox.config import Config
from boxbox.preprocessing.tyre_layer import neutralised
from boxbox.schema import EXCLUSION_REASONS


def field_slow_laps(laps: pd.DataFrame, neutralised_pct: float) -> pd.Series:
    """Heuristic SC/VSC/red detection: the field's median lap is abnormally slow.

    The reference is the field's median best qualifying lap, known before the race, so the flag
    at lap L never depends on later laps. Used for every race (the only neutralisation signal
    before 2018 and for the Kaggle export, which lacks TrackStatus).
    """
    racing = laps[(laps["lap"] > 1) & ~laps["pit_in"] & ~laps["pit_out"]]
    per_lap = racing.groupby(["race_id", "lap"])["lap_time_s"].median()
    reference = laps.groupby("race_id")["quali_best_s"].median()
    race_reference = per_lap.index.get_level_values("race_id").map(reference.to_dict())
    ratio: pd.Series = per_lap / race_reference.to_numpy(dtype=np.float64)
    keys: set[tuple[str, int]] = set(ratio[ratio > neutralised_pct].index)
    return pd.Series(
        [k in keys for k in zip(laps["race_id"], laps["lap"], strict=True)],
        index=laps.index,
        dtype=bool,
    )


def _rolling_field_median(laps: pd.DataFrame, eligible: pd.Series, window: int) -> pd.Series:
    per_lap = (
        laps[eligible].groupby(["race_id", "lap"])["lap_time_s"].median().rename("field_median")
    )
    rolled = (
        per_lap.groupby(level="race_id")
        .rolling(window, min_periods=1)
        .median()
        .droplevel(0)
        .rename("rolling_median")
    )
    lookup = laps[["race_id", "lap"]].merge(rolled.reset_index(), on=["race_id", "lap"], how="left")
    return pd.Series(lookup["rolling_median"].to_numpy(), index=laps.index)


def flag_laps(laps: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add ``track_neutralised``, ``is_clean`` and ``exclusion_reason`` (no rows dropped).

    Rules (first match wins, in ``EXCLUSION_REASONS`` order): missing time, lap 1, pit in/out,
    SC/VSC/red track status, field-slow heuristic, the lap after a neutralisation, inaccurate
    timing, wet (rain or intermediate/wet tyres), and > ``outlier_pct`` of the causal rolling
    field median of otherwise-clean laps.
    """
    df = laps.copy()
    status_neutral = neutralised(df["track_status"])
    field_slow = field_slow_laps(df, cfg.neutralised_field_pct)
    lap_neutral = (status_neutral | field_slow).groupby([df["race_id"], df["lap"]]).transform("max")
    neutral_by_lap = lap_neutral.groupby([df["race_id"], df["lap"]]).first()
    previous_neutral = pd.Series(
        [
            bool(neutral_by_lap.get((r, lap - 1), False))
            for r, lap in zip(df["race_id"], df["lap"], strict=True)
        ],
        index=df.index,
    )
    wet = df["rainfall"].astype(bool) | df["compound_raw"].astype("string").str.upper().isin(
        ["INTERMEDIATE", "WET"]
    ).fillna(value=False)

    conditions: dict[str, pd.Series] = {
        "missing_time": df["lap_time_s"].isna(),
        "lap_one": df["lap"] <= 1,
        "pit_in": df["pit_in"].astype(bool),
        "pit_out": df["pit_out"].astype(bool),
        "track_status": status_neutral,
        "field_slow": field_slow,
        "post_neutralisation": previous_neutral & ~lap_neutral.astype(bool),
        "inaccurate": ~df["is_accurate"].astype(bool),
        "wet": wet.astype(bool),
    }
    excluded_so_far = pd.concat(conditions.values(), axis=1).any(axis=1)
    rolling = _rolling_field_median(df, ~excluded_so_far, cfg.outlier_window_laps)
    conditions["outlier"] = df["lap_time_s"] > cfg.outlier_pct * rolling

    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    for name in reversed(EXCLUSION_REASONS):
        reason = reason.mask(conditions[name].fillna(value=False).astype(bool), name)
    df["exclusion_reason"] = reason
    df["is_clean"] = reason.isna()
    df["track_neutralised"] = lap_neutral.astype(bool)
    return df


def clean_lap_summary(laps: pd.DataFrame) -> pd.DataFrame:
    """Counts of rows per exclusion reason per race (``clean`` for kept laps)."""
    labels = laps["exclusion_reason"].fillna("clean")
    return (
        laps.assign(reason=labels)
        .groupby(["race_id", "reason"])
        .size()
        .rename("n")
        .reset_index()
        .astype({"n": np.int64})
    )
