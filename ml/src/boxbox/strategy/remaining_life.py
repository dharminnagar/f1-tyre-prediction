"""Remaining competitive laps: laps until the tyre-attributable loss exceeds a threshold."""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import numpy as np
import pandas as pd

from boxbox.models.forecaster import Forecaster
from boxbox.models.types import FloatArray
from boxbox.strategy.rollout import lap_times, predict_segments, stay_out_design

PERSISTENCE_LAPS: Final[int] = 2
NOISE_DRAWS: Final[int] = 400
NOISE_SEED: Final[int] = 0
NOISE_OFFSET_HORIZON_LAPS: Final[int] = 10


@dataclass(frozen=True)
class RemainingLife:
    """``value`` competitive laps after the decision lap under ``threshold_s``."""

    value: int
    threshold_s: float
    reference_s: float
    capped_by_race_end: bool


@dataclass(frozen=True)
class DeltaNoise:
    """How a predicted tyre-delta curve relates to the noisy pace the truth is measured on.

    The truth ends a tyre's life on the first noisy run of laps over the threshold, which on
    average comes before a smooth mean curve crosses it. ``offset_s`` is the curve's mean
    level error; ``sigma_s`` and ``ar1`` are the standard deviation and lag-1 autocorrelation
    of the per-lap residual around it (fitted on validation stints by :func:`fit_delta_noise`).
    """

    offset_s: float
    sigma_s: float
    ar1: float


NO_NOISE: Final[DeltaNoise] = DeltaNoise(0.0, 0.0, 0.0)


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


@lru_cache(maxsize=256)
def _unit_ar1_paths(n_laps: int, ar1: float, draws: int, seed: int) -> FloatArray:
    """``draws`` stationary unit-variance AR(1) paths of ``n_laps`` (read-only, cached)."""
    shocks = np.random.default_rng(seed).standard_normal((draws, n_laps))
    paths = np.empty_like(shocks)
    paths[:, 0] = shocks[:, 0]
    innovation = np.sqrt(1.0 - ar1**2)
    for lap in range(1, n_laps):
        paths[:, lap] = ar1 * paths[:, lap - 1] + innovation * shocks[:, lap]
    paths.setflags(write=False)
    return paths


def _competitive_counts(samples: FloatArray, threshold_s: float, persistence: int) -> FloatArray:
    """:func:`count_competitive` of every row of ``samples`` at once."""
    n_laps = samples.shape[1]
    if n_laps < persistence:
        return np.full(len(samples), float(n_laps))
    over = np.cumsum(samples > threshold_s, axis=1, dtype=np.int64)
    padded = np.concatenate([np.zeros((len(samples), 1), dtype=np.int64), over], axis=1)
    runs = padded[:, persistence:] - padded[:, :-persistence]
    hit = runs == persistence
    counts = np.where(hit.any(axis=1), hit.argmax(axis=1), n_laps)
    return np.asarray(counts, dtype=np.float64)


def noisy_competitive_count(
    delta: FloatArray,
    threshold_s: float,
    noise: DeltaNoise,
    persistence: int = PERSISTENCE_LAPS,
) -> tuple[int, bool]:
    """:func:`count_competitive` as the truth would measure it on noisy laps.

    The median count over ``NOISE_DRAWS`` seeded AR(1) noise paths added to the
    offset-corrected curve; True if that median reaches the end of ``delta``. With zero
    ``sigma_s`` it is :func:`count_competitive` of the offset curve.
    """
    shifted = np.asarray(delta + noise.offset_s, dtype=np.float64)
    if noise.sigma_s == 0.0 or len(shifted) == 0:
        return count_competitive(shifted, threshold_s, persistence)
    paths = _unit_ar1_paths(len(shifted), noise.ar1, NOISE_DRAWS, NOISE_SEED)
    counts = _competitive_counts(shifted + noise.sigma_s * paths, threshold_s, persistence)
    value = int(np.quantile(counts, 0.5, method="lower"))
    return value, value >= len(shifted)


def fit_delta_noise(curves: Sequence[tuple[FloatArray, FloatArray]]) -> DeltaNoise:
    """Fit :class:`DeltaNoise` from ``(predicted delta, actual excess)`` decision curves.

    Both curves start at the lap after the decision; ``actual`` is NaN on unclean laps and
    ends with the stint. ``offset_s`` averages the residual over the first
    ``NOISE_OFFSET_HORIZON_LAPS`` laps, where crossings happen and before only the longest
    stints remain. ``sigma_s`` and ``ar1`` use each decision's demeaned residuals.
    """
    early: list[FloatArray] = []
    demeaned: list[FloatArray] = []
    lagged: list[tuple[FloatArray, FloatArray]] = []
    for delta, actual in curves:
        residual = actual - delta[: len(actual)]
        if not np.isfinite(residual).any():
            continue
        early.append(residual[:NOISE_OFFSET_HORIZON_LAPS])
        centred = residual - np.nanmean(residual)
        demeaned.append(centred)
        both = np.isfinite(centred[:-1]) & np.isfinite(centred[1:])
        lagged.append((centred[:-1][both], centred[1:][both]))
    if not demeaned:
        return NO_NOISE
    flat = np.concatenate(demeaned)
    previous = np.concatenate([p for p, _ in lagged])
    current = np.concatenate([c for _, c in lagged])
    ar1 = float(np.corrcoef(previous, current)[0, 1]) if len(previous) > 2 else 0.0
    return DeltaNoise(
        offset_s=float(np.nanmean(np.concatenate(early))),
        sigma_s=float(np.nanstd(flat)),
        ar1=float(np.clip(ar1, 0.0, 0.95)),
    )


def remaining_competitive_laps(
    model: Forecaster,
    state: pd.Series,
    threshold_s: float,
    reference_age: int,
    fuel_s_per_lap: float,
    noise: DeltaNoise = NO_NOISE,
) -> RemainingLife:
    """Remaining competitive laps (capped at the chequered flag), noise-aware via ``noise``."""
    end_lap = int(state["total_laps"])
    delta = tyre_delta_curve(model, state, end_lap, reference_age)
    value, capped = noisy_competitive_count(delta, threshold_s, noise)
    design = stay_out_design(state, min(end_lap, int(state["lap"]) + 1))
    if design.empty:
        return RemainingLife(0, threshold_s, float("nan"), True)
    reference_row = design.assign(tyre_age=float(reference_age), stint_lap=float(reference_age))
    laps = design["lap_number"].to_numpy(dtype=np.int64)
    reference_y = predict_segments(model, state, [reference_row])[0].y_s
    reference_s = float(lap_times(state, laps, reference_y, fuel_s_per_lap)[0])
    return RemainingLife(value, threshold_s, reference_s, capped)
