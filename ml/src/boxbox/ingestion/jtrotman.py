"""Readers for the Kaggle ``jtrotman/formula-1-race-data`` dump (Ergast-format CSVs)."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from boxbox.ingestion.tables import (
    LAP_TABLE_COLUMNS,
    PIT_TABLE_COLUMNS,
    QUALI_TABLE_COLUMNS,
    RESULT_TABLE_COLUMNS,
    RaceTables,
    best_quali_seconds,
)


@dataclass(frozen=True)
class JtrotmanDump:
    """The subset of jtrotman CSV tables used by BoxBox."""

    races: pd.DataFrame
    lap_times: pd.DataFrame
    pit_stops: pd.DataFrame
    results: pd.DataFrame
    qualifying: pd.DataFrame
    drivers: pd.DataFrame
    constructors: pd.DataFrame
    status: pd.DataFrame


def load_jtrotman(directory: Path) -> JtrotmanDump:
    r"""Read the CSV dump (``\N`` is the dump's missing-value marker)."""

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(directory / f"{name}.csv", na_values=["\\N"], keep_default_na=True)

    return JtrotmanDump(
        races=read("races"),
        lap_times=read("lap_times"),
        pit_stops=read("pit_stops"),
        results=read("results"),
        qualifying=read("qualifying"),
        drivers=read("drivers"),
        constructors=read("constructors"),
        status=read("status"),
    )


def race_tables_from_jtrotman(dump: JtrotmanDump, season: int, race_name: str) -> RaceTables:
    """Normalised :class:`RaceTables` for one race from the dump."""
    races = dump.races[(dump.races["year"] == season) & (dump.races["name"] == race_name)]
    if races.empty:
        raise LookupError(f"{race_name} {season} not in jtrotman dump")
    race_id = int(races["raceId"].iloc[0])
    round_number = int(races["round"].iloc[0])
    driver_ref = dump.drivers.set_index("driverId")["driverRef"]

    laps = dump.lap_times[dump.lap_times["raceId"] == race_id].assign(
        driver_id=lambda d: d["driverId"].map(driver_ref),
        lap_time_s=lambda d: d["milliseconds"] / 1000.0,
    )[LAP_TABLE_COLUMNS]

    pit_stops = dump.pit_stops[dump.pit_stops["raceId"] == race_id].assign(
        driver_id=lambda d: d["driverId"].map(driver_ref),
        duration_s=lambda d: d["milliseconds"] / 1000.0,
    )[PIT_TABLE_COLUMNS]

    drivers = dump.drivers.set_index("driverId")
    constructors = dump.constructors.set_index("constructorId")
    status = dump.status.set_index("statusId")["status"]
    race_results = dump.results[dump.results["raceId"] == race_id]
    results = pd.DataFrame(
        {
            "driver_id": race_results["driverId"].map(driver_ref),
            "driver_code": race_results["driverId"].map(drivers["code"]).fillna(""),
            "driver_name": race_results["driverId"].map(
                drivers["forename"] + " " + drivers["surname"]
            ),
            "car_number": race_results["number"],
            "constructor_id": race_results["constructorId"].map(constructors["constructorRef"]),
            "team": race_results["constructorId"].map(constructors["name"]),
            "grid": race_results["grid"],
            "laps_completed": race_results["laps"],
            "status": race_results["statusId"].map(status),
        }
    )[RESULT_TABLE_COLUMNS]

    race_quali = dump.qualifying[dump.qualifying["raceId"] == race_id]
    quali = pd.DataFrame(
        {
            "driver_id": race_quali["driverId"].map(driver_ref),
            "quali_best_s": best_quali_seconds(race_quali, ["q1", "q2", "q3"]),
        }
    )[QUALI_TABLE_COLUMNS]

    return RaceTables(
        season=season,
        round_number=round_number,
        race_name=race_name,
        laps=laps.reset_index(drop=True),
        pit_stops=pit_stops.reset_index(drop=True),
        results=results.reset_index(drop=True),
        quali=quali.reset_index(drop=True),
    )
