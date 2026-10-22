"""Multi-lap rollouts of a per-lap pace model from a decision-lap state."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from boxbox.features.build import rollout_design
from boxbox.models.types import FloatArray, PaceModel, predict_pace

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Rollout:
    """Predicted laps ``laps`` (absolute lap numbers) for one strategy branch."""

    laps: IntArray
    y_s: FloatArray
    lap_time_s: FloatArray
    tyre_age: FloatArray
    compound: tuple[str, ...]
    extrapolated: BoolArray


def fuel_terms(state: pd.Series, laps: IntArray, fuel_s_per_lap: float) -> FloatArray:
    """Remaining-fuel penalty per future lap (zero in the refuelling era)."""
    if state["era"] == "refuel_2005_2009":
        return np.zeros(len(laps), dtype=np.float64)
    remaining = np.clip(float(state["total_laps"]) - laps, 0, None)
    return np.asarray(fuel_s_per_lap * remaining, dtype=np.float64)


def lap_times(state: pd.Series, laps: IntArray, y: FloatArray, fuel_s_per_lap: float) -> FloatArray:
    """Lap time = relative pace + field reference (frozen at the decision lap) + fuel term."""
    field_ref = float(state["s_field_ref_next_s"])
    return np.asarray(y + field_ref + fuel_terms(state, laps, fuel_s_per_lap), dtype=np.float64)


def _branch_design(
    state: pd.Series, laps: IntArray, compound: str, age0: float, stint_lap0: float, new: bool
) -> pd.DataFrame:
    return rollout_design(state, laps, compound, age0, stint_lap0, new_stint=new)


def stay_out_design(state: pd.Series, end_lap: int) -> pd.DataFrame:
    """Rows for laps t+1..end_lap on the current tyre."""
    laps = np.arange(int(state["lap"]) + 1, end_lap + 1, dtype=np.int64)
    return _branch_design(
        state,
        laps,
        str(state["compound"]),
        float(state["tyre_age"]) + 1,
        float(state["stint_lap"]) + 1,
        new=False,
    )


def new_tyre_design(state: pd.Series, pit_lap: int, end_lap: int, compound: str) -> pd.DataFrame:
    """Rows for laps pit_lap+1..end_lap on a fresh set (out-lap has tyre age 1)."""
    laps = np.arange(pit_lap + 1, end_lap + 1, dtype=np.int64)
    return _branch_design(state, laps, compound, 1.0, 1.0, new=True)


def _to_rollout(
    model: PaceModel, state: pd.Series, design: pd.DataFrame, fuel_s_per_lap: float
) -> Rollout:
    laps = design["lap_number"].to_numpy(dtype=np.int64)
    y = predict_pace(model, design) if len(design) else np.zeros(0, dtype=np.float64)
    return Rollout(
        laps=laps,
        y_s=y,
        lap_time_s=lap_times(state, laps, y, fuel_s_per_lap),
        tyre_age=design["tyre_age"].to_numpy(dtype=np.float64),
        compound=tuple(design["compound"].astype(str)),
        extrapolated=model.tyre_clip.extrapolated(design).to_numpy(dtype=np.bool_),
    )


def stay_out_rollout(
    model: PaceModel, state: pd.Series, end_lap: int, fuel_s_per_lap: float
) -> Rollout:
    """Predicted laps t+1..end_lap without stopping."""
    return _to_rollout(model, state, stay_out_design(state, end_lap), fuel_s_per_lap)


def pit_rollout(
    model: PaceModel,
    state: pd.Series,
    pit_lap: int,
    end_lap: int,
    compound: str,
    fuel_s_per_lap: float,
) -> Rollout:
    """Predicted laps t+1..end_lap when boxing at the end of ``pit_lap`` (pit loss excluded)."""
    old = stay_out_design(state, pit_lap)
    new = new_tyre_design(state, pit_lap, end_lap, compound)
    return _to_rollout(model, state, pd.concat([old, new], ignore_index=True), fuel_s_per_lap)
