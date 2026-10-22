"""Build the clean lap-level dataset for a profile from interim tables."""

import hashlib
from dataclasses import dataclass

import pandas as pd

from boxbox.config import Config
from boxbox.ingestion.download import read_tables, season_sources
from boxbox.paths import DatasetPaths, dataset_paths
from boxbox.preprocessing.cleaning import flag_laps
from boxbox.preprocessing.laps import add_gaps, to_canonical_laps
from boxbox.preprocessing.stints import (
    apply_inferred_pit_laps,
    infer_pit_laps,
    reconstruct_stints,
    verify_stints,
)
from boxbox.schema import era_for_season, race_id_for


@dataclass(frozen=True)
class LapDataset:
    """Clean lap table plus the stint verification table."""

    laps: pd.DataFrame
    stint_verification: pd.DataFrame
    pit_stops: pd.DataFrame


def build_season(cfg: Config, paths: DatasetPaths, season: int) -> LapDataset:
    """Canonical, stint-annotated and flagged laps for one season's race."""
    sources = season_sources(season)
    tables = read_tables(paths, season, sources.round_number, cfg.event_name)
    tyre_path = paths.interim_table(season, "tyre_layer")
    tyre_layer = pd.read_parquet(tyre_path) if tyre_path.exists() else None
    label = f"{sources.tables_source}+{sources.tyre_source or 'none'}"
    laps = to_canonical_laps(tables, tyre_layer, cfg.event_name, label)
    has_pit_table = not tables.pit_stops.empty
    if not has_pit_table:
        laps = apply_inferred_pit_laps(laps, infer_pit_laps(laps, cfg.pit_detection_excess_s))
    laps = reconstruct_stints(
        laps,
        tyre_change_allowed=season not in cfg.no_tyre_change_seasons,
        has_pit_table=has_pit_table,
    )
    laps = add_gaps(laps, cfg.traffic_gap_s)
    laps["era"] = era_for_season(season)
    laps["tyre_source"] = sources.tyre_source or "none"
    pit_stops = tables.pit_stops.assign(race_id=race_id_for(season, sources.round_number))
    verification = verify_stints(laps, pit_stops) if tyre_layer is not None else pit_stops.iloc[:0]
    return LapDataset(laps, verification, pit_stops)


def build_lap_dataset(cfg: Config) -> LapDataset:
    """All seasons of the profile, flagged; refuel-era seasons dropped if configured."""
    paths = dataset_paths()
    seasons = [
        s
        for s in cfg.seasons
        if not (cfg.exclude_refuel_era and era_for_season(s) == "refuel_2005_2009")
    ]
    parts = [build_season(cfg, paths, s) for s in seasons]
    laps = pd.concat([p.laps for p in parts], ignore_index=True)
    laps = flag_laps(laps, cfg)
    laps = laps.sort_values(["race_id", "driver_id", "lap"]).reset_index(drop=True)
    verification = pd.concat(
        [p.stint_verification for p in parts if not p.stint_verification.empty], ignore_index=True
    ).astype({"matched": bool})
    pit_stops = pd.concat([p.pit_stops for p in parts], ignore_index=True)
    return LapDataset(laps, verification, pit_stops)


def dataset_hash(laps: pd.DataFrame) -> str:
    """Deterministic content hash of a lap table (row order independent of build order)."""
    ordered = laps.sort_values(["race_id", "driver_id", "lap"]).reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(ordered.astype(str), index=False)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()[:16]


def write_lap_dataset(cfg: Config, dataset: LapDataset) -> None:
    """Persist laps_clean / stint_verification / pit_stops parquet for the profile."""
    paths = dataset_paths()
    target = paths.processed_file(cfg.profile, "laps_clean")
    target.parent.mkdir(parents=True, exist_ok=True)
    dataset.laps.to_parquet(target, index=False)
    dataset.stint_verification.to_parquet(
        paths.processed_file(cfg.profile, "stint_verification"), index=False
    )
    dataset.pit_stops.to_parquet(paths.processed_file(cfg.profile, "pit_stops"), index=False)


def read_lap_dataset(cfg: Config) -> LapDataset:
    """Load the processed dataset written by :func:`write_lap_dataset`."""
    paths = dataset_paths()
    return LapDataset(
        pd.read_parquet(paths.processed_file(cfg.profile, "laps_clean")),
        pd.read_parquet(paths.processed_file(cfg.profile, "stint_verification")),
        pd.read_parquet(paths.processed_file(cfg.profile, "pit_stops")),
    )
