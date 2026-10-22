"""Within-stint tyre degradation prior: a per-compound quadratic in tyre age.

Cross-sectional curves (median pace by tyre age) are selection-biased: drivers who stay out
long are the ones whose tyres last. Demeaning pace and tyre age within each stint (stint fixed
effects) removes driver, race and fuel-model level differences and keeps only how one set of
tyres slows down as it ages.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from boxbox.models.types import FloatArray, fit_tyre_age_clip
from boxbox.schema import DRY_COMPOUNDS, STINT_KEYS

POOLED_DRY: Final[str] = "POOLED_DRY"
REFUEL_ERA: Final[str] = "refuel_2005_2009"
MIN_STINT_LAPS: Final[int] = 3
MIN_COMPOUND_ROWS: Final[int] = 30


@dataclass(frozen=True)
class TyrePrior:
    """Within-stint tyre degradation curve per compound, fitted on clean non-refuel train laps.

    Every mapping also carries the ``POOLED_DRY`` key (all dry compounds fitted together),
    used for compounds without their own fit.
    """

    coefficients: dict[str, tuple[float, float]]  # compound -> (linear, quadratic) on tyre_age
    caps: dict[str, float]  # compound -> tyre-age cap (same P95 caps as TyreAgeClip)
    end_slope: dict[str, float]  # compound -> slope at the cap, floored at 0


def _demeaned(rows: pd.DataFrame) -> pd.DataFrame:
    age = rows["tyre_age"].astype(float)
    frame = pd.DataFrame({"y": rows["y_s"].astype(float), "a": age, "a2": age**2}, index=rows.index)
    means = frame.groupby([rows[k] for k in STINT_KEYS]).transform("mean")
    return frame - means


def _least_squares(demeaned: pd.DataFrame) -> tuple[float, float]:
    design = demeaned[["a", "a2"]].to_numpy(dtype=np.float64)
    solution = np.linalg.lstsq(design, demeaned["y"].to_numpy(dtype=np.float64), rcond=None)[0]
    return float(solution[0]), float(solution[1])


def _quadratic(coefficients: tuple[float, float], age: FloatArray) -> FloatArray:
    linear, quadratic = coefficients
    return np.asarray(linear * age + quadratic * age**2, dtype=np.float64)


def _running_max(coefficients: tuple[float, float], age: FloatArray) -> FloatArray:
    """``max(q(a') for 0 <= a' <= age)``: the quadratic made non-decreasing from age 0."""
    linear, quadratic = coefficients
    vertex = -linear / (2 * quadratic) if quadratic != 0 else 0.0
    at_vertex = _quadratic(coefficients, np.clip(np.full_like(age, vertex), 0.0, age))
    return np.maximum(np.maximum(_quadratic(coefficients, age), at_vertex), 0.0)


def _slope_at(coefficients: tuple[float, float], age: float) -> float:
    linear, quadratic = coefficients
    return max(linear + 2 * quadratic * age, 0.0)


def fit_tyre_prior(rows: pd.DataFrame, caps: dict[str, float]) -> TyrePrior:
    """Per-compound least squares of stint-demeaned ``y_s`` on stint-demeaned (age, age^2).

    The caller selects the training split; this keeps clean, non-refuelling-era laps (no fuel
    correction there, so within-stint slopes are negative) with a known tyre age up to the
    compound's cap, in stints with at least ``MIN_STINT_LAPS`` such laps. Compounds with fewer
    than ``MIN_COMPOUND_ROWS`` laps use the pooled dry fit.

    Args:
        rows: Feature rows with ``y_s``, ``tyre_age``, ``compound``, ``era``, ``is_clean``
            and the stint keys.
        caps: Per-compound tyre-age caps (the models' training P95 caps).

    Returns:
        The fitted prior with ``end_slope`` equal to the curve's last-lap slope at each cap.
    """
    compound = rows["compound"].astype(str)
    cap = compound.map(caps).fillna(max(caps.values()))
    usable = rows[
        rows["is_clean"].astype(bool)
        & (rows["era"].astype(str) != REFUEL_ERA)
        & rows["y_s"].notna()
        & rows["tyre_age"].notna()
        & (rows["tyre_age"] <= cap)
    ]
    usable = usable[usable.groupby(STINT_KEYS)["y_s"].transform("size") >= MIN_STINT_LAPS]
    demeaned = _demeaned(usable)
    usable_compound = usable["compound"].astype(str)

    dry = usable_compound.isin(DRY_COMPOUNDS)
    pooled = _least_squares(demeaned[dry]) if dry.sum() >= MIN_COMPOUND_ROWS else (0.0, 0.0)
    coefficients: dict[str, tuple[float, float]] = {POOLED_DRY: pooled}
    for name in DRY_COMPOUNDS:
        mask = usable_compound == name
        if mask.sum() >= MIN_COMPOUND_ROWS:
            coefficients[name] = _least_squares(demeaned[mask])

    dry_caps = [caps[c] for c in DRY_COMPOUNDS if c in caps]
    all_caps = {**caps, POOLED_DRY: max(dry_caps) if dry_caps else max(caps.values())}
    end_slope = {
        name: _slope_at(coefficients.get(name, pooled), value) for name, value in all_caps.items()
    }
    return TyrePrior(coefficients, all_caps, end_slope)


def tyre_prior_curve(prior: TyrePrior, compound: str, tyre_age: FloatArray) -> FloatArray:
    """Tyre-wear time (s) relative to a new tyre at ``tyre_age``; monotone non-decreasing.

    The fitted quadratic is followed up to the compound's cap (flattened wherever it would
    decrease), then continued linearly at ``end_slope``. A compound without its own fit (e.g.
    ``UNKNOWN`` before 2018) uses the pooled dry coefficients, and its own cap when it has one
    (otherwise the pooled cap).
    """
    coefficients = prior.coefficients.get(compound, prior.coefficients[POOLED_DRY])
    cap = prior.caps.get(compound, prior.caps[POOLED_DRY])
    slope = prior.end_slope.get(compound, prior.end_slope[POOLED_DRY])
    age = np.clip(np.asarray(tyre_age, dtype=np.float64), 0.0, None)
    inside = _running_max(coefficients, np.minimum(age, cap))
    return np.asarray(inside + slope * np.clip(age - cap, 0.0, None), dtype=np.float64)


def tyre_prior_values(prior: TyrePrior, rows: pd.DataFrame) -> FloatArray:
    """:func:`tyre_prior_curve` evaluated row by row on ``compound`` and ``tyre_age``."""
    out = np.zeros(len(rows), dtype=np.float64)
    compound = rows["compound"].astype(str).to_numpy()
    age = rows["tyre_age"].to_numpy(dtype=np.float64)
    for name in np.unique(compound):
        mask = compound == name
        out[mask] = tyre_prior_curve(prior, str(name), age[mask])
    return out


def prior_summary(prior: TyrePrior) -> dict[str, object]:
    """JSON-ready view of the prior (manifest / metrics ``tyre_prior`` block)."""
    return {
        "coefficients": {
            k: {"linear": v[0], "quadratic": v[1]} for k, v in prior.coefficients.items()
        },
        "caps": dict(prior.caps),
        "end_slope": dict(prior.end_slope),
    }


def fit_training_prior(rows: pd.DataFrame, cap_quantile: float) -> TyrePrior:
    """:func:`fit_tyre_prior` with caps from :func:`fit_tyre_age_clip` on the same rows.

    Called with the models' training rows this gives exactly the prior (and caps) the models
    are fitted with, so feature building and model fitting agree.
    """
    return fit_tyre_prior(rows, fit_tyre_age_clip(rows, cap_quantile).max_age)


def prior_from_summary(summary: Mapping[str, object]) -> TyrePrior:
    """Inverse of :func:`prior_summary` (e.g. to rebuild the prior from the manifest).

    Raises:
        ValueError: If a block is missing or not a mapping of numbers.
    """

    def block(name: str) -> dict[str, object]:
        value = summary.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"tyre prior summary has no '{name}' mapping")
        return {str(k): v for k, v in value.items()}

    def number(value: object) -> float:
        if not isinstance(value, int | float):
            raise ValueError(f"tyre prior value {value!r} is not a number")
        return float(value)

    coefficients: dict[str, tuple[float, float]] = {}
    for compound, pair in block("coefficients").items():
        if not isinstance(pair, dict):
            raise ValueError(f"tyre prior coefficients for {compound} are not a mapping")
        coefficients[compound] = (number(pair.get("linear")), number(pair.get("quadratic")))
    caps = {k: number(v) for k, v in block("caps").items()}
    end_slope = {k: number(v) for k, v in block("end_slope").items()}
    return TyrePrior(coefficients, caps, end_slope)
