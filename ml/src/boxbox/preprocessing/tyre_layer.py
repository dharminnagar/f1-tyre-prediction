"""Per-lap tyre/timing layer (2018+) from FastF1, OpenF1 or a Kaggle FastF1 export.

Every converter returns ``TYRE_LAYER_COLUMNS`` keyed by Ergast ``driver_id`` and ``lap``.
"""

from typing import Final

import numpy as np
import pandas as pd

from boxbox.ingestion.fastf1_client import RawRaceSession

TYRE_LAYER_COLUMNS: Final[list[str]] = [
    "driver_id",
    "lap",
    "lap_time_s",
    "sector_1_s",
    "sector_2_s",
    "sector_3_s",
    "compound_raw",
    "tyre_life",
    "pit_in_flag",
    "pit_out_flag",
    "track_status",
    "is_accurate",
    "air_temp",
    "track_temp",
    "humidity",
    "rainfall",
]

COMPOUND_ALIASES: Final[dict[str, str]] = {
    "SOFT": "SOFT",
    "MEDIUM": "MEDIUM",
    "HARD": "HARD",
    "SUPERSOFT": "SOFT",
    "ULTRASOFT": "SOFT",
    "HYPERSOFT": "SOFT",
    "SUPERHARD": "HARD",
    "INTERMEDIATE": "INTERMEDIATE",
    "WET": "WET",
}
NEUTRALISED_STATUS_CODES: Final[frozenset[str]] = frozenset({"4", "5", "6", "7"})


def normalise_compound(raw: pd.Series) -> pd.Series:
    """Map source compound names to SOFT/MEDIUM/HARD/INTERMEDIATE/WET (else NaN).

    Pre-2019 rainbow names are folded into the nearest relative label (super/ultra/hyper-soft
    -> SOFT, superhard -> HARD); this is a documented approximation for 2018 only.
    """
    return raw.astype("string").str.upper().map(COMPOUND_ALIASES)


def _td_seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series).dt.total_seconds()


def tyre_layer_from_fastf1(raw: RawRaceSession) -> pd.DataFrame:
    """Tyre layer from a FastF1 race session (DriverId comes from the session results)."""
    laps = raw.laps
    number_to_id = raw.results.set_index("DriverNumber")["DriverId"]
    end_time_s = _td_seconds(laps["Time"])
    weather = raw.weather.assign(time_s=_td_seconds(raw.weather["Time"])).sort_values("time_s")
    layer = pd.DataFrame(
        {
            "driver_id": laps["DriverNumber"].map(number_to_id),
            "lap": laps["LapNumber"].astype("Int64"),
            "lap_time_s": _td_seconds(laps["LapTime"]),
            "sector_1_s": _td_seconds(laps["Sector1Time"]),
            "sector_2_s": _td_seconds(laps["Sector2Time"]),
            "sector_3_s": _td_seconds(laps["Sector3Time"]),
            "compound_raw": laps["Compound"].astype("string"),
            "tyre_life": pd.to_numeric(laps["TyreLife"], errors="coerce"),
            "pit_in_flag": laps["PitInTime"].notna(),
            "pit_out_flag": laps["PitOutTime"].notna(),
            "track_status": laps["TrackStatus"].astype("string").fillna(""),
            "is_accurate": laps["IsAccurate"].fillna(value=False).astype(bool),
            "end_time_s": end_time_s,
        }
    ).sort_values("end_time_s")
    with_weather = pd.merge_asof(
        layer.dropna(subset=["end_time_s"]),
        weather[["time_s", "AirTemp", "TrackTemp", "Humidity", "Rainfall"]],
        left_on="end_time_s",
        right_on="time_s",
        direction="backward",
    ).rename(
        columns={
            "AirTemp": "air_temp",
            "TrackTemp": "track_temp",
            "Humidity": "humidity",
            "Rainfall": "rainfall",
        }
    )
    with_weather["rainfall"] = with_weather["rainfall"].fillna(value=False).astype(bool)
    return with_weather.dropna(subset=["driver_id", "lap"])[TYRE_LAYER_COLUMNS]


def track_status_by_lap(race_control: pd.DataFrame, total_laps: int) -> pd.Series:
    """FastF1-style track-status string per leader lap from OpenF1 race-control messages.

    Codes: 4 = safety car, 5 = red flag, 6 = VSC deployed, 7 = VSC ending; "1" = green.
    Sector-scoped yellow flags are not converted (they are local, not neutralisations).
    """
    statuses: dict[int, set[str]] = {lap: set() for lap in range(1, total_laps + 1)}
    active: dict[str, int] = {}
    messages = race_control.sort_values("date") if "date" in race_control else race_control
    for _, row in messages.iterrows():
        lap = int(row["lap_number"]) if pd.notna(row.get("lap_number")) else 1
        text = str(row.get("message", "")).upper()
        category = str(row.get("category", ""))
        flag = str(row.get("flag", "")).upper()
        scope = str(row.get("scope", "")).upper()
        if category == "SafetyCar" and "VIRTUAL" in text:
            if "DEPLOYED" in text:
                active["6"] = lap
            elif "ENDING" in text and "6" in active:
                _mark(statuses, "6", active.pop("6"), lap)
                _mark(statuses, "7", lap, lap)
        elif category == "SafetyCar":
            if "DEPLOYED" in text:
                active["4"] = lap
            elif "IN THIS LAP" in text and "4" in active:
                _mark(statuses, "4", active.pop("4"), lap)
        elif category == "Flag" and flag == "RED":
            active["5"] = lap
        elif (
            category == "Flag" and flag in {"GREEN", "CLEAR"} and scope == "TRACK" and "5" in active
        ):
            _mark(statuses, "5", active.pop("5"), lap)
    for code, start in active.items():
        _mark(statuses, code, start, total_laps)
    return pd.Series(
        {lap: "".join(sorted(codes)) if codes else "1" for lap, codes in statuses.items()},
        name="track_status",
    )


def _mark(statuses: dict[int, set[str]], code: str, start: int, end: int) -> None:
    for lap in range(start, end + 1):
        if lap in statuses:
            statuses[lap].add(code)


def tyre_layer_from_openf1(
    laps: pd.DataFrame,
    stints: pd.DataFrame,
    pits: pd.DataFrame,
    race_control: pd.DataFrame,
    weather: pd.DataFrame,
    results: pd.DataFrame,
    total_laps: int,
) -> pd.DataFrame:
    """Tyre layer from OpenF1 tables; driver numbers are mapped via the race ``results``."""
    number_to_id = results.set_index("car_number")["driver_id"]
    base = pd.DataFrame(
        {
            "driver_number": laps["driver_number"].astype(int),
            "lap": laps["lap_number"].astype(int),
            "lap_time_s": pd.to_numeric(laps["lap_duration"], errors="coerce"),
            "sector_1_s": pd.to_numeric(laps["duration_sector_1"], errors="coerce"),
            "sector_2_s": pd.to_numeric(laps["duration_sector_2"], errors="coerce"),
            "sector_3_s": pd.to_numeric(laps["duration_sector_3"], errors="coerce"),
            "pit_out_flag": laps["is_pit_out_lap"].fillna(value=False).astype(bool),
            "date_start": pd.to_datetime(laps["date_start"], utc=True, format="ISO8601"),
        }
    )
    stint_rows = [
        {
            "driver_number": int(s["driver_number"]),
            "lap": lap,
            "compound_raw": str(s["compound"]),
            "tyre_life": float(s["tyre_age_at_start"]) + (lap - int(s["lap_start"])) + 1,
        }
        for _, s in stints.dropna(subset=["lap_start", "lap_end"]).iterrows()
        for lap in range(int(s["lap_start"]), int(s["lap_end"]) + 1)
    ]
    stint_laps = pd.DataFrame(
        stint_rows, columns=["driver_number", "lap", "compound_raw", "tyre_life"]
    ).drop_duplicates(subset=["driver_number", "lap"], keep="last")
    pit_laps = pd.DataFrame(
        {
            "driver_number": pits["driver_number"].astype(int),
            "lap": pits["lap_number"].astype(int),
            "pit_in_flag": True,
        }
    ).drop_duplicates()
    merged = base.merge(stint_laps, on=["driver_number", "lap"], how="left").merge(
        pit_laps, on=["driver_number", "lap"], how="left"
    )
    merged["pit_in_flag"] = merged["pit_in_flag"].astype("boolean").fillna(value=False)
    merged["track_status"] = merged["lap"].map(track_status_by_lap(race_control, total_laps))
    merged["is_accurate"] = merged["lap_time_s"].notna() & ~merged["pit_out_flag"]
    merged["driver_id"] = merged["driver_number"].map(number_to_id)

    weather_sorted = weather.assign(
        date=pd.to_datetime(weather["date"], utc=True, format="ISO8601")
    ).sort_values("date")
    with_weather = pd.merge_asof(
        merged.dropna(subset=["date_start"]).sort_values("date_start"),
        weather_sorted[["date", "air_temperature", "track_temperature", "humidity", "rainfall"]],
        left_on="date_start",
        right_on="date",
        direction="backward",
    ).rename(columns={"air_temperature": "air_temp", "track_temperature": "track_temp"})
    with_weather["rainfall"] = with_weather["rainfall"].fillna(0).astype(float) > 0
    with_weather["pit_in_flag"] = with_weather["pit_in_flag"].astype(bool)
    return with_weather.dropna(subset=["driver_id"]).sort_values(["driver_id", "lap"])[
        TYRE_LAYER_COLUMNS
    ]


def tyre_layer_from_fastf1_export(
    export: pd.DataFrame, gp_name: str, season: int, results: pd.DataFrame
) -> pd.DataFrame:
    """Tyre layer from the Kaggle FastF1 lap export (columns follow FastF1 naming).

    The export carries no TrackStatus and unreliable pit flags; pit laps therefore come from
    Ergast pit stops downstream and neutralisations from the field-slow heuristic.
    """
    rows = export[(export["Year"] == season) & (export["GP"] == gp_name)]
    code_to_id = results.set_index("driver_code")["driver_id"]
    rain = rows["Rainfall"].astype("string").str.lower().isin(["true", "1", "1.0"])
    layer = pd.DataFrame(
        {
            "driver_id": rows["Driver"].map(code_to_id),
            "lap": rows["LapNumber"].astype("Int64"),
            "lap_time_s": pd.to_numeric(rows["LapTime"], errors="coerce"),
            "sector_1_s": pd.to_numeric(rows["Sector1Time"], errors="coerce"),
            "sector_2_s": pd.to_numeric(rows["Sector2Time"], errors="coerce"),
            "sector_3_s": pd.to_numeric(rows["Sector3Time"], errors="coerce"),
            "compound_raw": rows["Compound"].astype("string"),
            "tyre_life": pd.to_numeric(rows["TyreLife"], errors="coerce"),
            "pit_in_flag": False,
            "pit_out_flag": False,
            "track_status": "",
            "is_accurate": pd.to_numeric(rows["LapTime"], errors="coerce").notna(),
            "air_temp": pd.to_numeric(rows["AirTemp"], errors="coerce"),
            "track_temp": pd.to_numeric(rows["TrackTemp"], errors="coerce"),
            "humidity": pd.to_numeric(rows["Humidity"], errors="coerce"),
            "rainfall": rain.fillna(value=False).astype(bool),
        }
    )
    layer = layer.dropna(subset=["driver_id", "lap"])
    layer["lap"] = layer["lap"].astype(int)
    return layer.drop_duplicates(subset=["driver_id", "lap"]).sort_values(["driver_id", "lap"])[
        TYRE_LAYER_COLUMNS
    ]


def neutralised(track_status: pd.Series) -> pd.Series:
    """Whether a FastF1-style status string contains SC/VSC/red codes."""
    codes = track_status.fillna("").astype(str)
    return codes.map(lambda s: bool(set(s) & NEUTRALISED_STATUS_CODES)).astype(bool)


def empty_tyre_layer() -> pd.DataFrame:
    """A tyre layer with no rows (pre-2018 races)."""
    return pd.DataFrame({c: pd.Series(dtype=np.float64) for c in TYRE_LAYER_COLUMNS})
