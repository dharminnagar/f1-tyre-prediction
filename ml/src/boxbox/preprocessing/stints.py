"""Stint reconstruction from pit stops and verification against observed tyre changes."""

import numpy as np
import pandas as pd

from boxbox.preprocessing.laps import as_bool
from boxbox.schema import DRIVER_KEYS, STINT_KEYS


def infer_pit_laps(laps: pd.DataFrame, excess_threshold_s: float) -> pd.Series:
    """Infer in-laps from lap times when no pit-stop table exists.

    A stop costs time across the in-lap and out-lap, so the pair excess over the field's
    median for that lap (minus the driver's usual offset) is compared with a threshold;
    the first lap of each qualifying local maximum is flagged as the in-lap.
    """
    field_median = laps.groupby(["race_id", "lap"])["lap_time_s"].transform("median")
    relative = laps["lap_time_s"] - field_median
    offset = relative.groupby([laps["race_id"], laps["driver_id"]]).transform("median")
    excess = (relative - offset).fillna(0.0)
    grouped = excess.groupby([laps["race_id"], laps["driver_id"]])
    pair = excess + grouped.shift(-1).fillna(0.0)
    pair_grouped = pair.groupby([laps["race_id"], laps["driver_id"]])
    is_peak = (pair >= pair_grouped.shift(1).fillna(-np.inf)) & (
        pair > pair_grouped.shift(-1).fillna(-np.inf)
    )
    return (is_peak & (pair > excess_threshold_s) & (laps["lap"] > 1)).astype(bool)


def apply_inferred_pit_laps(laps: pd.DataFrame, pit_in: pd.Series) -> pd.DataFrame:
    """Set ``pit_in``/``pit_out`` from inferred in-laps."""
    pit_out = pit_in.groupby([laps["race_id"], laps["driver_id"]]).shift(1)
    return laps.assign(pit_in=pit_in, pit_out=as_bool(pit_out))


def reconstruct_stints(
    laps: pd.DataFrame, tyre_change_allowed: bool, has_pit_table: bool
) -> pd.DataFrame:
    """Add ``stint_id``, ``stint_lap``, ``tyre_age``, per-stint ``compound`` and ``stint_source``.

    ``stint_id`` = 1 + number of pit stops completed (out-laps seen). ``stint_lap`` counts the
    out-lap as 1. ``tyre_age`` follows observed tyre life when available (offset per stint so it
    advances exactly one per lap), otherwise ``stint_lap`` (labelled inferred). When tyre changes
    were not allowed (2005) stops are fuel-only and the whole race is one stint.
    """
    df = laps.sort_values([*DRIVER_KEYS, "lap"]).copy()
    pit_out = df["pit_out"].astype(bool) & (df["lap"] > 1)
    if tyre_change_allowed:
        df["stint_id"] = pit_out.astype(int).groupby([df["race_id"], df["driver_id"]]).cumsum() + 1
    else:
        df["stint_id"] = 1
    stint_start = df.groupby(STINT_KEYS)["lap"].transform("min")
    df["stint_lap"] = df["lap"] - stint_start + 1

    life_offset = (
        (df["tyre_life"] - df["stint_lap"]).groupby([df[k] for k in STINT_KEYS]).transform("median")
    )
    observed = life_offset.notna()
    df["tyre_age"] = np.where(observed, df["stint_lap"] + life_offset.round(), df["stint_lap"])
    df["tyre_age"] = df["tyre_age"].clip(lower=1).astype(float)

    compound_mode = (
        df.dropna(subset=["compound"])
        .groupby(STINT_KEYS)["compound"]
        .agg(lambda s: s.mode().iloc[0])
        .rename("stint_compound")
    )
    df = df.merge(compound_mode, left_on=STINT_KEYS, right_index=True, how="left")
    df["compound"] = df["stint_compound"].fillna("UNKNOWN").astype(str)
    df = df.drop(columns=["stint_compound"])
    fallback = "inferred_pitstops" if has_pit_table else "inferred_laptimes"
    df["stint_source"] = np.where(observed, "observed_tyre", fallback)
    return df.reset_index(drop=True)


def verify_stints(laps: pd.DataFrame, pit_stops: pd.DataFrame) -> pd.DataFrame:
    """Check each Ergast pit stop against an observed tyre change on the following lap.

    Uses the raw tyre layer (``compound_raw``/``tyre_life``), independent of the pit flags used
    for reconstruction. Returns one row per stop with ``matched``.
    """
    observed = laps.dropna(subset=["tyre_life"]).sort_values([*DRIVER_KEYS, "lap"])
    grouped = observed.groupby(DRIVER_KEYS)
    changed = (observed["compound_raw"] != grouped["compound_raw"].shift(1)) | (
        observed["tyre_life"] < grouped["tyre_life"].shift(1)
    )
    change_keys = set(
        zip(
            observed.loc[changed, "race_id"],
            observed.loc[changed, "driver_id"],
            observed.loc[changed, "lap"] - 1,
            strict=True,
        )
    )
    observed_drivers = set(zip(observed["race_id"], observed["driver_id"], strict=True))
    stops = pit_stops[
        [
            (r, d) in observed_drivers
            for r, d in zip(pit_stops["race_id"], pit_stops["driver_id"], strict=True)
        ]
    ]
    matched = [
        (r, d, lap) in change_keys
        for r, d, lap in zip(stops["race_id"], stops["driver_id"], stops["lap"], strict=True)
    ]
    return stops.assign(matched=pd.Series(matched, index=stops.index, dtype=bool)).reset_index(
        drop=True
    )
