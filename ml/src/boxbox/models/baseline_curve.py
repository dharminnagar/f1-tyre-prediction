"""Baseline 1: descriptive degradation curve (tyre prior + per-compound level + driver offset)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from boxbox.features.tyre_prior import TyrePrior, tyre_prior_curve, tyre_prior_values
from boxbox.models.types import FloatArray, TyreAgeClip
from boxbox.schema import ModelName

FEATURES = ("compound", "tyre_age", "stint_lap", "driver_offset_s")


@dataclass(frozen=True)
class BaselineCurveModel:
    """``y = prior_curve[compound](age) + compound_level[compound] + driver_offset``.

    The shape comes from the within-stint tyre prior (not cross-sectional medians, which are
    selection-biased towards drivers whose tyres last); compounds without a level use the
    global level.
    """

    prior: TyrePrior
    compound_level: dict[str, float]
    global_level: float
    tyre_clip: TyreAgeClip
    name: ModelName = "baseline_curve"
    features: tuple[str, ...] = FEATURES

    def predict(self, X: pd.DataFrame) -> FloatArray:
        clipped = self.tyre_clip.clip(X)
        level = (
            clipped["compound"]
            .astype(str)
            .map(self.compound_level)
            .fillna(self.global_level)
            .to_numpy(dtype=np.float64)
        )
        offset = clipped["driver_offset_s"].fillna(0.0).to_numpy(dtype=np.float64)
        return np.asarray(tyre_prior_values(self.prior, clipped) + level + offset, np.float64)

    def describe(self) -> dict[str, float]:
        """Fitted curve per ``compound@age`` for integer ages up to the compound's cap."""
        described: dict[str, float] = {}
        for compound, level in sorted(self.compound_level.items()):
            cap = int(self.tyre_clip.max_age.get(compound, max(self.tyre_clip.max_age.values())))
            ages = np.arange(1, cap + 1, dtype=np.float64)
            values = tyre_prior_curve(self.prior, compound, ages) + level
            described |= {
                f"{compound}@{int(a)}": float(v) for a, v in zip(ages, values, strict=True)
            }
        return described


def fit_baseline_curve(
    X: pd.DataFrame, y: pd.Series, tyre_clip: TyreAgeClip, prior: TyrePrior
) -> BaselineCurveModel:
    """Median of ``y - driver_offset - prior_curve`` per compound (the compound's level)."""
    clipped = tyre_clip.clip(X)
    residual = (
        y.to_numpy(dtype=np.float64)
        - clipped["driver_offset_s"].fillna(0.0).to_numpy(dtype=np.float64)
        - tyre_prior_values(prior, clipped)
    )
    keyed = pd.DataFrame({"compound": clipped["compound"].astype(str).to_numpy(), "r": residual})
    levels = {str(k): float(v) for k, v in keyed.groupby("compound")["r"].median().items()}
    return BaselineCurveModel(prior, levels, float(np.median(residual)), tyre_clip)
