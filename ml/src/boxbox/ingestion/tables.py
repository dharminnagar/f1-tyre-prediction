"""Normalised per-race historical tables shared by the Kaggle (jtrotman) and Ergast sources."""

from dataclasses import dataclass
from typing import Final

import pandas as pd

LAP_TABLE_COLUMNS: Final[list[str]] = ["driver_id", "lap", "position", "lap_time_s"]
PIT_TABLE_COLUMNS: Final[list[str]] = ["driver_id", "stop", "lap", "duration_s"]
RESULT_TABLE_COLUMNS: Final[list[str]] = [
    "driver_id",
    "driver_code",
    "driver_name",
    "car_number",
    "constructor_id",
    "team",
    "grid",
    "laps_completed",
    "status",
]
QUALI_TABLE_COLUMNS: Final[list[str]] = ["driver_id", "quali_best_s"]
TABLE_NAMES: Final[tuple[str, ...]] = ("laps", "pit_stops", "results", "quali")


@dataclass(frozen=True)
class RaceTables:
    """Timing backbone for one race: lap times, pit stops, results and qualifying."""

    season: int
    round_number: int
    race_name: str
    laps: pd.DataFrame
    pit_stops: pd.DataFrame
    results: pd.DataFrame
    quali: pd.DataFrame

    def table(self, name: str) -> pd.DataFrame:
        """Return a table by name (one of ``TABLE_NAMES``)."""
        tables = {
            "laps": self.laps,
            "pit_stops": self.pit_stops,
            "results": self.results,
            "quali": self.quali,
        }
        return tables[name]


def parse_lap_time_string(value: object) -> float:
    r"""Parse ``"1:26.572"``/``"86.572"`` into seconds; NaN for missing markers like ``\N``."""
    if not isinstance(value, str) or value.strip() in {"", "\\N"}:
        return float("nan")
    minutes, _, seconds = value.strip().rpartition(":")
    try:
        return (float(minutes) * 60 if minutes else 0.0) + float(seconds)
    except ValueError:
        return float("nan")


def best_quali_seconds(quali: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Fastest of the Q1/Q2/Q3 string columns per row, in seconds."""
    parsed = pd.DataFrame({c: quali[c].map(parse_lap_time_string) for c in columns})
    return parsed.min(axis=1, skipna=True)
