"""Assemble model features: deterministic lap-L schedule + state frozen at the end of L-1."""

from typing import Final

import numpy as np
import pandas as pd

from boxbox.config import Config
from boxbox.evaluation.splits import assign_splits
from boxbox.features.context import context_state
from boxbox.features.degradation import degradation_state
from boxbox.features.pace import (
    driver_offset,
    field_reference,
    fuel_corrected,
    quali_gap,
    relative_pace,
)
from boxbox.schema import DRIVER_KEYS, STINT_KEYS

DRIVER_STATE: Final[tuple[str, ...]] = (
    "driver_offset_s",
    "position",
    "gap_ahead_s",
    "in_traffic",
    "air_temp",
    "track_temp",
)
STINT_STATE: Final[tuple[str, ...]] = (
    "deg_slope_3",
    "deg_slope_5",
    "deg_accel",
    "rolling_mean_3",
    "rolling_mean_5",
    "lap_delta",
)
DETERMINISTIC_NUMERIC: Final[tuple[str, ...]] = (
    "tyre_age",
    "stint_lap",
    "lap_number",
    "race_progress",
    "quali_gap_s",
)
CATEGORICAL: Final[tuple[str, ...]] = ("compound", "era")
MAIN_NUMERIC: Final[tuple[str, ...]] = (*DETERMINISTIC_NUMERIC, *DRIVER_STATE)
DEGRADATION_FEATURES: Final[tuple[str, ...]] = ("deg_slope_5", "deg_accel")


def _shift_state(df: pd.DataFrame) -> pd.DataFrame:
    """Feature at lap L = state at the end of lap L-1 (stint state resets on a new stint)."""
    ordered = df.sort_values([*DRIVER_KEYS, "lap"])
    by_driver = ordered.groupby(DRIVER_KEYS)
    by_stint = ordered.groupby(STINT_KEYS)
    shifted = {name: by_driver[f"s_{name}"].shift(1) for name in DRIVER_STATE}
    shifted |= {name: by_stint[f"s_{name}"].shift(1) for name in STINT_STATE}
    previous_lap = by_driver["lap"].shift(1)
    return ordered.assign(**shifted, has_history=(previous_lap == ordered["lap"] - 1))


def build_features(laps: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Full feature table (one row per driver-lap) with target ``y_s`` and ``split``."""
    df = fuel_corrected(laps, cfg.fuel_s_per_lap)
    df = field_reference(df)
    df = relative_pace(df)
    df = driver_offset(df)
    df = quali_gap(df)
    df = degradation_state(df)
    df = context_state(df)
    df = df.assign(lap_number=df["lap"].astype(float))
    df = _shift_state(df)
    df = assign_splits(df, cfg)
    df["is_training_row"] = df["is_clean"] & df["has_history"] & df["y_s"].notna()
    return df.sort_values(["race_id", "driver_id", "lap"]).reset_index(drop=True)


def rollout_design(
    state: pd.Series,
    laps_ahead: np.ndarray,
    compound: str,
    first_tyre_age: float,
    first_stint_lap: float,
    new_stint: bool,
) -> pd.DataFrame:
    """Feature rows for future laps ``laps_ahead`` given the decision-lap ``state`` row.

    Deterministic features advance lap by lap from ``first_tyre_age``/``first_stint_lap``;
    driver state is frozen at the end of the decision lap; stint state is frozen too, or
    unknown (NaN) when the rows belong to a new stint after a pit stop.
    """
    steps = np.arange(len(laps_ahead), dtype=float)
    total_laps = float(state["total_laps"])
    frame: dict[str, object] = {
        "compound": compound,
        "era": state["era"],
        "tyre_age": first_tyre_age + steps,
        "stint_lap": first_stint_lap + steps,
        "lap_number": laps_ahead.astype(float),
        "race_progress": laps_ahead / total_laps,
        "quali_gap_s": state["quali_gap_s"],
    }
    frame |= {name: state[f"s_{name}"] for name in DRIVER_STATE}
    frame |= {name: np.nan if new_stint else state[f"s_{name}"] for name in STINT_STATE}
    return pd.DataFrame(frame, index=pd.RangeIndex(len(laps_ahead)))
