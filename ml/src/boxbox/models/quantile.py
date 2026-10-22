"""Quantile gradient boosting: per-lap prediction intervals on relative pace ``y_s``."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from boxbox.config import GbmParams
from boxbox.models.gbm import FEATURES
from boxbox.models.types import FloatArray, PaceModel, TyreAgeClip, predict_pace, with_categories


@dataclass(frozen=True)
class QuantileHgb:
    """Lower/upper quantile HistGradientBoosting models giving a central interval on y_s."""

    lower: HistGradientBoostingRegressor
    upper: HistGradientBoostingRegressor
    tyre_clip: TyreAgeClip
    features: tuple[str, ...]
    quantiles: tuple[float, float]


def _inputs(X: pd.DataFrame, tyre_clip: TyreAgeClip, features: tuple[str, ...]) -> pd.DataFrame:
    return with_categories(tyre_clip.clip(X))[list(features)]


def _fit_quantile(
    inputs: pd.DataFrame, target: FloatArray, quantile: float, params: GbmParams, seed: int
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=params.max_iter,
        learning_rate=params.learning_rate,
        max_leaf_nodes=params.max_leaf_nodes,
        min_samples_leaf=params.min_samples_leaf,
        categorical_features="from_dtype",
        monotonic_cst={"tyre_age": 1},
        random_state=seed,
    ).fit(inputs, target)


def fit_quantile_hgb(
    X: pd.DataFrame,
    y: pd.Series,
    tyre_clip: TyreAgeClip,
    params: GbmParams,
    seed: int,
    quantiles: tuple[float, float] = (0.1, 0.9),
    features: tuple[str, ...] = FEATURES,
) -> QuantileHgb:
    """Fit one quantile-loss HGB per bound, with the same inputs and constraints as ``fit_hgb``.

    Args:
        X: Feature rows (unclipped; tyre age is clipped with ``tyre_clip``).
        y: Target ``y_s`` aligned with ``X``.
        tyre_clip: Per-compound tyre-age caps learned on the training rows.
        params: Boosting hyper-parameters shared with the point HGB model.
        seed: Random state for both estimators.
        quantiles: ``(lower, upper)`` quantile levels; the default gives an 80% interval.
        features: Columns fed to the estimators; must include ``tyre_age``.

    Returns:
        The fitted lower/upper pair.

    Raises:
        ValueError: If the quantiles are not ``0 < lower < upper < 1``.
    """
    low_q, high_q = quantiles
    if not 0.0 < low_q < high_q < 1.0:
        raise ValueError(f"quantiles must satisfy 0 < lower < upper < 1, got {quantiles}")
    inputs = _inputs(X, tyre_clip, features)
    target = y.to_numpy(dtype=np.float64)
    return QuantileHgb(
        lower=_fit_quantile(inputs, target, low_q, params, seed),
        upper=_fit_quantile(inputs, target, high_q, params, seed),
        tyre_clip=tyre_clip,
        features=features,
        quantiles=quantiles,
    )


def predict_interval(model: QuantileHgb, X: pd.DataFrame) -> tuple[FloatArray, FloatArray]:
    """(lo, hi) on y_s per row; guarantees lo <= hi (sort if quantiles cross).

    The two bounds are fitted independently, so they can cross where data is sparse; taking
    the row-wise min/max keeps the interval valid without changing its coverage.
    """
    inputs = _inputs(X, model.tyre_clip, model.features)
    lower = np.asarray(model.lower.predict(inputs), dtype=np.float64)
    upper = np.asarray(model.upper.predict(inputs), dtype=np.float64)
    return np.minimum(lower, upper), np.maximum(lower, upper)


def interval_coverage(lo: FloatArray, hi: FloatArray, actual: FloatArray) -> float:
    """Share of finite actuals inside [lo, hi].

    Rows with a non-finite actual are ignored. With no finite actuals the coverage is
    undefined and ``nan`` is returned rather than a misleading 0.
    """
    finite = np.isfinite(actual)
    if not finite.any():
        return float("nan")
    inside = (actual[finite] >= lo[finite]) & (actual[finite] <= hi[finite])
    return float(inside.mean())


def interval_offsets(
    point: PaceModel, model: QuantileHgb, X: pd.DataFrame
) -> tuple[FloatArray, FloatArray]:
    """(lo, hi) offsets of the quantile bounds from the point model's own prediction.

    Both bounds and the point prediction are taken at the clipped tyre age, so adding the
    offsets to :func:`~boxbox.models.types.predict_pace` centres the interval on the
    beyond-cap continuation and keeps the cap-age width past the cap. Offsets are clamped so
    the interval always contains the point prediction (lo <= 0 <= hi).
    """
    lower, upper = predict_interval(model, X)
    base = point.predict(X)
    return np.minimum(lower - base, 0.0), np.maximum(upper - base, 0.0)


def centred_interval(
    point: PaceModel, model: QuantileHgb, X: pd.DataFrame
) -> tuple[FloatArray, FloatArray]:
    """80% (by default) interval on ``y_s`` around ``predict_pace(point, X)``."""
    centre = predict_pace(point, X)
    lo, hi = interval_offsets(point, model, X)
    return centre + lo, centre + hi
