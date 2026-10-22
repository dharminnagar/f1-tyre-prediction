"""Ergast (jolpica mirror) connector via ``fastf1.ergast``: credential-free historical tables."""

import datetime as dt
from collections.abc import Callable
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.ergast import Ergast

from boxbox.ingestion.tables import (
    LAP_TABLE_COLUMNS,
    PIT_TABLE_COLUMNS,
    QUALI_TABLE_COLUMNS,
    RESULT_TABLE_COLUMNS,
    RaceTables,
)

PAGE_LIMIT = 100
JsonRecord = dict[str, object]


def _seconds(value: object) -> float:
    if isinstance(value, dt.timedelta):
        return value.total_seconds()
    return float("nan")


def _records(value: object) -> list[JsonRecord]:
    if not isinstance(value, list):
        return []
    return [{str(k): v for k, v in item.items()} for item in value if isinstance(item, dict)]


def _record(value: object) -> JsonRecord:
    return {str(k): v for k, v in value.items()} if isinstance(value, dict) else {}


class ErgastClient:
    """Typed wrapper over ``fastf1.ergast.Ergast`` returning normalised :class:`RaceTables`."""

    def __init__(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
        self._api = Ergast(result_type="raw", auto_cast=True, limit=PAGE_LIMIT)

    def _paged(self, fetch: Callable[[int], object], child_key: str) -> list[JsonRecord]:
        rows: list[JsonRecord] = []
        offset = 0
        while True:
            response = fetch(offset)
            races = _records(response)
            for race in races:
                rows.extend(_records(race.get(child_key)))
            total = int(getattr(response, "total_results", 0))
            offset += PAGE_LIMIT
            if offset >= total or not races:
                return rows

    def find_round(self, season: int, race_name: str) -> int | None:
        """Round number of ``race_name`` in ``season`` (None when not held)."""
        schedule = _records(self._api.get_race_schedule(season))
        for race in schedule:
            if race.get("raceName") == race_name:
                return int(str(race["round"]))
        return None

    def lap_table(self, season: int, round_number: int) -> pd.DataFrame:
        """Per-driver per-lap position and lap time (seconds)."""
        laps = self._paged(
            lambda offset: self._api.get_lap_times(season, round_number, offset=offset), "Laps"
        )
        rows = [
            {
                "driver_id": str(timing["driverId"]),
                "lap": int(str(lap["number"])),
                "position": int(str(timing["position"])),
                "lap_time_s": _seconds(timing.get("time")),
            }
            for lap in laps
            for timing in _records(lap.get("Timings"))
        ]
        return pd.DataFrame(rows, columns=LAP_TABLE_COLUMNS)

    def pit_table(self, season: int, round_number: int) -> pd.DataFrame:
        """Pit stops (in-lap number and stationary+lane duration)."""
        stops = self._paged(
            lambda offset: self._api.get_pit_stops(season, round_number, offset=offset),
            "PitStops",
        )
        rows = [
            {
                "driver_id": str(s["driverId"]),
                "stop": int(str(s["stop"])),
                "lap": int(str(s["lap"])),
                "duration_s": _seconds(s.get("duration")),
            }
            for s in stops
        ]
        return pd.DataFrame(rows, columns=PIT_TABLE_COLUMNS)

    def result_table(self, season: int, round_number: int) -> pd.DataFrame:
        """Race classification with driver and constructor identifiers."""
        results = self._paged(
            lambda offset: self._api.get_race_results(season, round_number, offset=offset),
            "Results",
        )
        rows = []
        for r in results:
            driver, constructor = _record(r.get("Driver")), _record(r.get("Constructor"))
            rows.append(
                {
                    "driver_id": str(driver["driverId"]),
                    "driver_code": str(driver.get("code", "")),
                    "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}",
                    "car_number": int(str(r["number"])),
                    "constructor_id": str(constructor["constructorId"]),
                    "team": str(constructor.get("name", "")),
                    "grid": int(str(r.get("grid", 0))),
                    "laps_completed": int(str(r.get("laps", 0))),
                    "status": str(r.get("status", "")),
                }
            )
        return pd.DataFrame(rows, columns=RESULT_TABLE_COLUMNS)

    def quali_table(self, season: int, round_number: int) -> pd.DataFrame:
        """Best qualifying lap per driver (seconds)."""
        quali = self._paged(
            lambda offset: self._api.get_qualifying_results(season, round_number, offset=offset),
            "QualifyingResults",
        )
        rows = []
        for q in quali:
            times = [_seconds(q.get(k)) for k in ("Q1", "Q2", "Q3")]
            valid = [t for t in times if t == t]
            rows.append(
                {
                    "driver_id": str(_record(q.get("Driver"))["driverId"]),
                    "quali_best_s": min(valid) if valid else float("nan"),
                }
            )
        return pd.DataFrame(rows, columns=QUALI_TABLE_COLUMNS)

    def race_tables(self, season: int, race_name: str) -> RaceTables:
        """All historical tables for one race."""
        round_number = self.find_round(season, race_name)
        if round_number is None:
            raise LookupError(f"{race_name} not held in {season}")
        return RaceTables(
            season=season,
            round_number=round_number,
            race_name=race_name,
            laps=self.lap_table(season, round_number),
            pit_stops=self.pit_table(season, round_number),
            results=self.result_table(season, round_number),
            quali=self.quali_table(season, round_number),
        )
