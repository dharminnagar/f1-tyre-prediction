"""Canonical column names, categorical vocabularies and shared literal types."""

from typing import Final, Literal, get_args

Compound = Literal["SOFT", "MEDIUM", "HARD"]
ModelCompound = Literal["SOFT", "MEDIUM", "HARD", "UNKNOWN"]
ModelName = Literal["baseline_curve", "poly_ridge", "hgb", "lstm"]
Split = Literal["train", "val", "test"]
Era = Literal["refuel_2005_2009", "pre_tyre_data_2010_2017", "tyre_data_2018_plus"]
StintSource = Literal["observed_tyre", "inferred_pitstops", "inferred_laptimes"]
TablesSource = Literal["kaggle_jtrotman", "ergast"]
ExclusionReason = Literal[
    "missing_time",
    "lap_one",
    "pit_in",
    "pit_out",
    "track_status",
    "post_neutralisation",
    "field_slow",
    "inaccurate",
    "wet",
    "outlier",
]

DRY_COMPOUNDS: Final[tuple[Compound, ...]] = get_args(Compound)
MODEL_COMPOUNDS: Final[tuple[ModelCompound, ...]] = get_args(ModelCompound)
MODEL_NAMES: Final[tuple[ModelName, ...]] = get_args(ModelName)
ERAS: Final[tuple[Era, ...]] = get_args(Era)
EXCLUSION_REASONS: Final[tuple[ExclusionReason, ...]] = get_args(ExclusionReason)
COMPOUND_HARDNESS: Final[dict[Compound, int]] = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}

RACE_KEYS: Final[list[str]] = ["race_id"]
DRIVER_KEYS: Final[list[str]] = ["race_id", "driver_id"]
STINT_KEYS: Final[list[str]] = ["race_id", "driver_id", "stint_id"]
LAP_KEYS: Final[list[str]] = ["race_id", "driver_id", "lap"]

CANONICAL_LAP_COLUMNS: Final[list[str]] = [
    "race_id",
    "season",
    "round",
    "event_name",
    "driver_id",
    "driver_code",
    "driver_name",
    "constructor_id",
    "team",
    "lap",
    "lap_time_s",
    "sector_1_s",
    "sector_2_s",
    "sector_3_s",
    "lap_end_time_s",
    "position",
    "compound",
    "tyre_life",
    "fastf1_stint",
    "pit_in",
    "pit_out",
    "track_status",
    "is_accurate",
    "air_temp",
    "track_temp",
    "humidity",
    "rainfall",
    "total_laps",
    "quali_best_s",
    "source",
]


def era_for_season(season: int) -> Era:
    """Regulatory/data era of a season (refuelling until 2009, FastF1 tyre data from 2018)."""
    if season <= 2009:
        return "refuel_2005_2009"
    if season <= 2017:
        return "pre_tyre_data_2010_2017"
    return "tyre_data_2018_plus"


def race_id_for(season: int, round_number: int) -> str:
    """Canonical race identifier, e.g. ``2024_10``."""
    return f"{season}_{round_number:02d}"
