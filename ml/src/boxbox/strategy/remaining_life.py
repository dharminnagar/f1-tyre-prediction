"""Remaining competitive laps: laps until the tyre-attributable loss exceeds a threshold."""

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from boxbox.models.forecaster import Forecaster
from boxbox.models.types import FloatArray
from boxbox.strategy.rollout import lap_times, predict_segments, stay_out_design

PERSISTENCE_LAPS: Final[int] = 2


@dataclass(frozen=True)
class RemainingLife:
    """``value`` competitive laps after the decision lap under ``threshold_s``."""

    value: int
    threshold_s: float
    reference_s: float
    capped_by_race_end: bool


def tyre_delta_curve(
    model: Forecaster, state: pd.Series, end_lap: int, reference_age: int
) -> FloatArray:
    """Predicted pace loss vs the same lap context on a ``reference_age`` tyre, laps t+1..end."""
    design = stay_out_design(state, end_lap)
    if design.empty:
        return np.zeros(0, dtype=np.float64)
    reference = design.assign(tyre_age=float(reference_age), stint_lap=float(reference_age))
    worn, fresh = predict_segments(model, state, [design, reference])
    return np.asarray(worn.y_s - fresh.y_s, dtype=np.float64)


def first_persistent_exceedance(
    values: FloatArray, threshold_s: float, persistence: int = PERSISTENCE_LAPS
) -> int | None:
    """First index starting ``persistence`` consecutive values above ``threshold_s``.

    A run cut short by the end of ``values`` does not count, so a one-lap spike that reverts
    never ends a tyre's competitive life.
    """
    over = np.asarray(values > threshold_s, dtype=np.int64)
    if len(over) < persistence:
        return None
    runs = np.convolve(over, np.ones(persistence, dtype=np.int64), mode="valid")
    hits = np.flatnonzero(runs == persistence)
    return int(hits[0]) if len(hits) else None


def count_competitive(
    delta: FloatArray, threshold_s: float, persistence: int = PERSISTENCE_LAPS
) -> tuple[int, bool]:
    """Leading laps before the loss persistently exceeds the threshold; True if it never did."""
    crossing = first_persistent_exceedance(delta, threshold_s, persistence)
    if crossing is None:
        return len(delta), True
    return crossing, False


def remaining_competitive_laps(
    model: Forecaster,
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
    reference_y = predict_segments(model, state, [reference_row])[0].y_s
    reference_s = float(lap_times(state, laps, reference_y, fuel_s_per_lap)[0])
    return RemainingLife(value, threshold_s, reference_s, capped)
