"""Remaining competitive laps: laps until the tyre-attributable loss exceeds a threshold."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from boxbox.models.types import FloatArray, PaceModel, predict_pace
from boxbox.strategy.rollout import lap_times, stay_out_design


@dataclass(frozen=True)
class RemainingLife:
    """``value`` competitive laps after the decision lap under ``threshold_s``."""

    value: int
    threshold_s: float
    reference_s: float
    capped_by_race_end: bool


def tyre_delta_curve(
    model: PaceModel, state: pd.Series, end_lap: int, reference_age: int
) -> FloatArray:
    """Predicted pace loss vs the same lap context on a ``reference_age`` tyre, laps t+1..end."""
    design = stay_out_design(state, end_lap)
    if design.empty:
        return np.zeros(0, dtype=np.float64)
    reference = design.assign(tyre_age=float(reference_age), stint_lap=float(reference_age))
    return np.asarray(
        predict_pace(model, design) - predict_pace(model, reference), dtype=np.float64
    )


def count_competitive(delta: FloatArray, threshold_s: float) -> tuple[int, bool]:
    """Leading laps with ``delta <= threshold``; second item is True if it never crossed."""
    over = np.flatnonzero(delta > threshold_s)
    if len(over) == 0:
        return len(delta), True
    return int(over[0]), False


def remaining_competitive_laps(
    model: PaceModel,
    state: pd.Series,
    threshold_s: float,
    reference_age: int,
    fuel_s_per_lap: float,
) -> RemainingLife:
    """Remaining competitive laps (capped at the chequered flag)."""
    end_lap = int(state["total_laps"])
    delta = tyre_delta_curve(model, state, end_lap, reference_age)
    value, capped = count_competitive(delta, threshold_s)
    design = stay_out_design(state, min(end_lap, int(state["lap"]) + 1))
    if design.empty:
        return RemainingLife(0, threshold_s, float("nan"), True)
    reference_row = design.assign(tyre_age=float(reference_age), stint_lap=float(reference_age))
    laps = design["lap_number"].to_numpy(dtype=np.int64)
    reference_s = float(
        lap_times(state, laps, predict_pace(model, reference_row), fuel_s_per_lap)[0]
    )
    return RemainingLife(value, threshold_s, reference_s, capped)
