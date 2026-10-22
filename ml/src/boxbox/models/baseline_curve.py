"""Baseline 1: descriptive degradation curve (median relative pace by compound x tyre age)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from boxbox.models.types import FloatArray, TyreAgeClip
from boxbox.schema import ModelName

FEATURES = ("compound", "tyre_age", "stint_lap", "driver_offset_s")


@dataclass(frozen=True)
class BaselineCurveModel:
    """``y = curve[compound, age] + driver_offset`` with compound/global fallbacks."""

    curve: dict[tuple[str, int], float]
    compound_median: dict[str, float]
    global_median: float
    tyre_clip: TyreAgeClip
    name: ModelName = "baseline_curve"
    features: tuple[str, ...] = FEATURES

    def predict(self, X: pd.DataFrame) -> FloatArray:
        clipped = self.tyre_clip.clip(X)
        ages = clipped["tyre_age"].round().astype(int)
        compounds = clipped["compound"].astype(str)
        base = [
            self.curve.get((c, a), self.compound_median.get(c, self.global_median))
            for c, a in zip(compounds, ages, strict=True)
        ]
        offset = clipped["driver_offset_s"].fillna(0.0).to_numpy(dtype=np.float64)
        return np.asarray(np.asarray(base, dtype=np.float64) + offset, dtype=np.float64)

    def describe(self) -> dict[str, float]:
        return {f"{c}@{a}": v for (c, a), v in sorted(self.curve.items())}


def fit_baseline_curve(
    X: pd.DataFrame, y: pd.Series, tyre_clip: TyreAgeClip, min_count: int = 5
) -> BaselineCurveModel:
    """Medians of ``y - driver_offset`` per (compound, clipped tyre age) with >= min_count laps."""
    clipped = tyre_clip.clip(X)
    residual = y - clipped["driver_offset_s"].fillna(0.0)
    keyed = pd.DataFrame(
        {
            "compound": clipped["compound"].astype(str),
            "age": clipped["tyre_age"].round().astype(int),
            "r": residual,
        }
    )
    grouped = keyed.groupby(["compound", "age"])["r"].agg(["median", "size"])
    kept = grouped[grouped["size"] >= min_count].reset_index()
    curve = {
        (str(c), int(a)): float(m)
        for c, a, m in zip(kept["compound"], kept["age"], kept["median"], strict=True)
    }
    compound_median = {str(k): float(v) for k, v in keyed.groupby("compound")["r"].median().items()}
    return BaselineCurveModel(curve, compound_median, float(keyed["r"].median()), tyre_clip)
