"""Download orchestration: historical tables (Kaggle -> Ergast fallback) + 2018+ tyre layer."""

import json
import logging
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from boxbox.config import Config, TyreSource
from boxbox.ingestion.ergast_client import ErgastClient
from boxbox.ingestion.f1db_client import F1DBClient
from boxbox.ingestion.fastf1_client import FastF1Client
from boxbox.ingestion.jtrotman import JtrotmanDump, load_jtrotman, race_tables_from_jtrotman
from boxbox.ingestion.kaggle_client import KaggleClient, KaggleUnavailableError
from boxbox.ingestion.openf1_client import (
    OPENF1_MIN_SEASON,
    OpenF1Client,
    read_openf1_endpoint,
)
from boxbox.ingestion.tables import TABLE_NAMES, RaceTables
from boxbox.paths import PROJECT_ROOT, DatasetPaths, dataset_paths
from boxbox.preprocessing.tyre_layer import (
    tyre_layer_from_fastf1,
    tyre_layer_from_fastf1_export,
    tyre_layer_from_openf1,
)

logger = logging.getLogger(__name__)
HistoricalSource = str


class SeasonSources(BaseModel):
    """Which source actually supplied each layer of one season's race (recorded in manifest)."""

    season: int
    round_number: int
    tables_source: HistoricalSource
    tyre_source: TyreSource | None
    fallbacks: list[str] = Field(default_factory=list)


def _kaggle_client() -> KaggleClient:
    return KaggleClient(PROJECT_ROOT / ".env", PROJECT_ROOT / ".venv" / "bin" / "kaggle")


def _ensure_kaggle_dataset(dataset_ref: str, target: Path) -> Path:
    if not any(target.glob("*.csv")):
        _kaggle_client().download_dataset(dataset_ref, target)
    return target


def _write_tables(paths: DatasetPaths, tables: RaceTables) -> None:
    paths.interim.mkdir(parents=True, exist_ok=True)
    for name in TABLE_NAMES:
        tables.table(name).to_parquet(paths.interim_table(tables.season, name), index=False)


def read_tables(paths: DatasetPaths, season: int, round_number: int, race_name: str) -> RaceTables:
    """Read one season's normalised historical tables from interim parquet."""
    frames = {name: pd.read_parquet(paths.interim_table(season, name)) for name in TABLE_NAMES}
    return RaceTables(
        season=season,
        round_number=round_number,
        race_name=race_name,
        laps=frames["laps"],
        pit_stops=frames["pit_stops"],
        results=frames["results"],
        quali=frames["quali"],
    )


def _historical_tables(cfg: Config, paths: DatasetPaths, season: int) -> tuple[RaceTables, str]:
    fallbacks: list[str] = []
    if cfg.historical_source == "kaggle":
        try:
            dump = _jtrotman_dump(cfg, paths)
            return race_tables_from_jtrotman(dump, season, cfg.event_name), "kaggle_jtrotman"
        except (KaggleUnavailableError, LookupError, FileNotFoundError) as error:
            fallbacks.append(f"kaggle_jtrotman: {error}")
            logger.warning(
                "season %s: Kaggle unavailable (%s); falling back to Ergast", season, error
            )
    tables = ErgastClient(paths.fastf1_cache).race_tables(season, cfg.event_name)
    return tables, "ergast" if not fallbacks else "ergast (kaggle fallback)"


_JTROTMAN_CACHE: dict[Path, JtrotmanDump] = {}


def _jtrotman_dump(cfg: Config, paths: DatasetPaths) -> JtrotmanDump:
    directory = _ensure_kaggle_dataset(cfg.sources.jtrotman_dataset, paths.kaggle_race_data)
    if directory not in _JTROTMAN_CACHE:
        _JTROTMAN_CACHE[directory] = load_jtrotman(directory)
    return _JTROTMAN_CACHE[directory]


def _tyre_layer(
    cfg: Config, paths: DatasetPaths, tables: RaceTables, fallbacks: list[str]
) -> tuple[pd.DataFrame, TyreSource] | None:
    season = tables.season
    total_laps = int(tables.results["laps_completed"].max())
    for source in cfg.sources.tyre_source_order:
        try:
            if source == "fastf1":
                raw = FastF1Client(paths.fastf1_cache).load_race_weekend(season, cfg.event_name)
                return tyre_layer_from_fastf1(raw), source
            if source == "openf1" and season >= OPENF1_MIN_SEASON:
                season_dir = OpenF1Client(paths.openf1).download_race(
                    season, cfg.sources.openf1_circuit_short_name
                )
                frames = {
                    e: pd.DataFrame(read_openf1_endpoint(season_dir, e))
                    for e in ("laps", "stints", "pit", "race_control", "weather")
                }
                layer = tyre_layer_from_openf1(
                    frames["laps"],
                    frames["stints"],
                    frames["pit"],
                    frames["race_control"],
                    frames["weather"],
                    tables.results,
                    total_laps,
                )
                return layer, source
            if source == "kaggle_fastf1_export":
                directory = _ensure_kaggle_dataset(
                    cfg.sources.fastf1_export_dataset, paths.kaggle_fastf1_export
                )
                export = pd.read_csv(next(directory.glob("*.csv")), low_memory=False)
                layer = tyre_layer_from_fastf1_export(
                    export, cfg.sources.fastf1_export_gp_name, season, tables.results
                )
                if not layer.empty:
                    return layer, source
                fallbacks.append(f"{source}: no rows for {season}")
        # Any connector failure (HTTP 401 on the F1 archive, missing season, ...) -> next source.
        except Exception as error:
            fallbacks.append(f"{source}: {type(error).__name__}: {str(error)[:160]}")
            logger.warning("season %s: tyre source %s failed (%s)", season, source, error)
    return None


def download_season(cfg: Config, season: int, force: bool = False) -> SeasonSources:
    """Fetch one season's race into ``dataset/interim`` (no-op when already present)."""
    paths = dataset_paths()
    record = _read_sources(paths).get(str(season))
    tyre_path = paths.interim_table(season, "tyre_layer")
    if record is not None and not force:
        has_tables = all(paths.interim_table(season, n).exists() for n in TABLE_NAMES)
        needs_tyres = season >= cfg.fastf1_min_season
        if has_tables and (tyre_path.exists() or not needs_tyres):
            return record
    tables, tables_source = _historical_tables(cfg, paths, season)
    _write_tables(paths, tables)
    fallbacks: list[str] = []
    tyre_source: TyreSource | None = None
    if season >= cfg.fastf1_min_season:
        layer = _tyre_layer(cfg, paths, tables, fallbacks)
        if layer is not None:
            layer[0].to_parquet(tyre_path, index=False)
            tyre_source = layer[1]
    sources = SeasonSources(
        season=season,
        round_number=tables.round_number,
        tables_source=tables_source,
        tyre_source=tyre_source,
        fallbacks=fallbacks,
    )
    _write_source(paths, sources)
    logger.info("season %s: tables=%s tyres=%s", season, tables_source, tyre_source)
    return sources


def download(cfg: Config, force: bool = False, with_f1db: bool = False) -> list[SeasonSources]:
    """Download every season of the profile; optionally the F1DB metadata release."""
    records = [download_season(cfg, season, force) for season in cfg.seasons]
    if with_f1db:
        try:
            F1DBClient(dataset_paths().f1db).download()
        except Exception as error:
            logger.warning("f1db download skipped: %s", error)
    return records


def _read_sources(paths: DatasetPaths) -> dict[str, SeasonSources]:
    if not paths.sources_record.exists():
        return {}
    payload: object = json.loads(paths.sources_record.read_text())
    if not isinstance(payload, dict):
        return {}
    return {str(k): SeasonSources.model_validate(v) for k, v in payload.items()}


def _write_source(paths: DatasetPaths, record: SeasonSources) -> None:
    records = _read_sources(paths)
    records[str(record.season)] = record
    paths.sources_record.write_text(
        json.dumps({k: v.model_dump() for k, v in sorted(records.items())}, indent=2)
    )


def season_sources(season: int) -> SeasonSources:
    """The recorded sources of a downloaded season."""
    record = _read_sources(dataset_paths()).get(str(season))
    if record is None:
        raise FileNotFoundError(f"season {season} not downloaded; run `boxbox download`")
    return record
