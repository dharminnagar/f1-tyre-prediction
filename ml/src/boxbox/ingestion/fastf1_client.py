"""FastF1 connector: the primary source of tyre-aware lap data (2018+)."""

from dataclasses import dataclass
from pathlib import Path

import fastf1
import pandas as pd


@dataclass(frozen=True)
class RawRaceSession:
    """Raw FastF1 tables for one race weekend (race + qualifying)."""

    season: int
    round_number: int
    event_name: str
    total_laps: int
    laps: pd.DataFrame
    results: pd.DataFrame
    weather: pd.DataFrame
    quali_results: pd.DataFrame


class FastF1Client:
    """Thin typed wrapper around the untyped ``fastf1`` package with an on-disk cache."""

    def __init__(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
        fastf1.set_log_level("WARNING")

    def event_round(self, season: int, event_name: str) -> int:
        """Championship round number of ``event_name`` in ``season``."""
        event = fastf1.get_event(season, event_name)
        return int(event["RoundNumber"])

    def load_race_weekend(self, season: int, event_name: str) -> RawRaceSession:
        """Load race laps, results, weather and qualifying results (no telemetry)."""
        race = fastf1.get_session(season, event_name, "R")
        race.load(laps=True, telemetry=False, weather=True, messages=False)
        quali = fastf1.get_session(season, event_name, "Q")
        quali.load(laps=False, telemetry=False, weather=False, messages=False)
        laps = pd.DataFrame(race.laps)
        return RawRaceSession(
            season=season,
            round_number=int(race.event["RoundNumber"]),
            event_name=str(race.event["EventName"]),
            total_laps=int(laps["LapNumber"].max()),
            laps=laps,
            results=pd.DataFrame(race.results),
            weather=pd.DataFrame(race.weather_data),
            quali_results=pd.DataFrame(quali.results),
        )
