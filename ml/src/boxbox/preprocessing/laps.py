"""Canonical one-row-per-driver-per-lap table from the timing backbone plus the tyre layer."""

import numpy as np
import pandas as pd

from boxbox.ingestion.tables import RaceTables
from boxbox.preprocessing.tyre_layer import normalise_compound
from boxbox.schema import CANONICAL_LAP_COLUMNS, race_id_for


def as_bool(series: pd.Series) -> pd.Series:
    """Nullable/object flags -> plain bool with missing treated as False."""
    return series.astype("boolean").fillna(value=False).astype(bool)


def _flag_laps(laps: pd.DataFrame, stops: pd.DataFrame, lap_offset: int) -> pd.Series:
    keys = set(zip(stops["driver_id"], stops["lap"] + lap_offset, strict=True))
    return pd.Series(
        [k in keys for k in zip(laps["driver_id"], laps["lap"], strict=True)],
        index=laps.index,
        dtype=bool,
    )


def to_canonical_laps(
    tables: RaceTables, tyre_layer: pd.DataFrame | None, event_name: str, source: str
) -> pd.DataFrame:
    """Join Ergast lap times/positions, pit stops, results, qualifying and the tyre layer.

    Lap times always come from the Ergast backbone (official classification timing) so every
    source shares one definition; the tyre layer adds sectors, tyres, track status and weather.
    Pit flags are the union of Ergast stops and tyre-layer flags.
    """
    backbone = tables.laps.dropna(subset=["driver_id"]).sort_values(["driver_id", "lap"])
    backbone = backbone.assign(
        lap=backbone["lap"].astype(int),
        lap_end_time_s=backbone.groupby("driver_id")["lap_time_s"].cumsum(),
    )
    backbone["pit_in"] = _flag_laps(backbone, tables.pit_stops, 0)
    backbone["pit_out"] = _flag_laps(backbone, tables.pit_stops, 1)

    if tyre_layer is not None and not tyre_layer.empty:
        layer = tyre_layer.drop(columns=["lap_time_s"]).assign(lap=tyre_layer["lap"].astype(int))
        merged = backbone.merge(layer, on=["driver_id", "lap"], how="left")
        merged["pit_in"] = merged["pit_in"] | as_bool(merged["pit_in_flag"])
        merged["pit_out"] = merged["pit_out"] | (
            as_bool(merged["pit_out_flag"]) & (merged["lap"] > 1)
        )
        merged["is_accurate"] = as_bool(merged["is_accurate"])
        merged["compound"] = normalise_compound(merged["compound_raw"])
        merged["track_status"] = merged["track_status"].fillna("").astype(str)
        merged["rainfall"] = as_bool(merged["rainfall"])
    else:
        merged = backbone.assign(
            sector_1_s=np.nan,
            sector_2_s=np.nan,
            sector_3_s=np.nan,
            compound_raw=pd.Series(pd.NA, index=backbone.index, dtype="string"),
            compound=pd.Series(pd.NA, index=backbone.index, dtype="string"),
            tyre_life=np.nan,
            track_status="",
            is_accurate=True,
            air_temp=np.nan,
            track_temp=np.nan,
            humidity=np.nan,
            rainfall=False,
        )

    results = tables.results.drop_duplicates(subset=["driver_id"]).set_index("driver_id")
    quali = tables.quali.drop_duplicates(subset=["driver_id"]).set_index("driver_id")
    canonical = merged.assign(
        race_id=race_id_for(tables.season, tables.round_number),
        season=tables.season,
        round=tables.round_number,
        event_name=event_name,
        driver_code=merged["driver_id"].map(results["driver_code"]).fillna(""),
        driver_name=merged["driver_id"].map(results["driver_name"]).fillna(merged["driver_id"]),
        constructor_id=merged["driver_id"].map(results["constructor_id"]).fillna("unknown"),
        team=merged["driver_id"].map(results["team"]).fillna(""),
        fastf1_stint=np.nan,
        total_laps=int(tables.results["laps_completed"].max()),
        quali_best_s=merged["driver_id"].map(quali["quali_best_s"]),
        source=source,
    )
    canonical["position"] = canonical["position"].astype(float)
    canonical = canonical.drop_duplicates(subset=["driver_id", "lap"]).sort_values(
        ["driver_id", "lap"]
    )
    return canonical[[*CANONICAL_LAP_COLUMNS, "compound_raw"]].reset_index(drop=True)


def add_gaps(laps: pd.DataFrame, traffic_gap_s: float) -> pd.DataFrame:
    """Gap to the car ahead/behind on the same lap from cumulative race time, plus traffic flag."""
    ordered = laps.sort_values(["race_id", "lap", "lap_end_time_s"])
    by_lap = ordered.groupby(["race_id", "lap"])["lap_end_time_s"]
    ordered = ordered.assign(
        gap_ahead_s=by_lap.diff(),
        gap_behind_s=-by_lap.diff(-1),
    )
    ordered["in_traffic"] = (ordered["gap_ahead_s"] < traffic_gap_s).astype(float)
    return ordered.sort_values(["race_id", "driver_id", "lap"]).reset_index(drop=True)
