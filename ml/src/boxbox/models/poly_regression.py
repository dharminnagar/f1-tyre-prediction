"""Baseline 2: Ridge regression with per-compound quadratic tyre-age terms (interpretable)."""

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from boxbox.features.build import MAIN_NUMERIC
from boxbox.models.types import FloatArray, TyreAgeClip, with_categories
from boxbox.schema import ERAS, MODEL_COMPOUNDS, ModelName

CONTEXT_NUMERIC: Final[tuple[str, ...]] = tuple(
    c for c in MAIN_NUMERIC if c not in {"tyre_age", "stint_lap"}
)
FEATURES: Final[tuple[str, ...]] = ("compound", "era", *MAIN_NUMERIC)


def poly_design(X: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """Named design matrix: compound dummies, compound x (age, age^2), context, era dummies.

    Missing context values are imputed with training medians plus a missing indicator.
    """
    cat = with_categories(X)
    design: dict[str, pd.Series] = {}
    age = cat["tyre_age"].astype(float)
    for compound in MODEL_COMPOUNDS:
        is_c = (cat["compound"] == compound).astype(float)
        design[f"compound={compound}"] = is_c
        design[f"tyre_age:{compound}"] = age * is_c
        design[f"tyre_age^2:{compound}"] = age**2 * is_c
    for col in CONTEXT_NUMERIC:
        values = cat[col].astype(float)
        design[col] = values.fillna(medians.get(col, 0.0))
        design[f"{col}_missing"] = values.isna().astype(float)
    for era in ERAS[:-1]:
        design[f"era={era}"] = (cat["era"] == era).astype(float)
    return pd.DataFrame(design, index=X.index)


@dataclass(frozen=True)
class PolyRidgeModel:
    """Fitted Ridge on :func:`poly_design` features (coefficients are in seconds)."""

    estimator: Ridge
    columns: tuple[str, ...]
    medians: dict[str, float]
    tyre_clip: TyreAgeClip
    name: ModelName = "poly_ridge"
    features: tuple[str, ...] = FEATURES

    def predict(self, X: pd.DataFrame) -> FloatArray:
        design = poly_design(self.tyre_clip.clip(X), self.medians)[list(self.columns)]
        return np.asarray(self.estimator.predict(design.to_numpy()), dtype=np.float64)

    def describe(self) -> dict[str, float]:
        coefficients = np.asarray(self.estimator.coef_, dtype=np.float64)
        named = {c: float(v) for c, v in zip(self.columns, coefficients, strict=True)}
        return {"intercept": float(self.estimator.intercept_), **named}


def fit_poly_ridge(
    X: pd.DataFrame, y: pd.Series, tyre_clip: TyreAgeClip, alpha: float
) -> PolyRidgeModel:
    """Fit the interpretable polynomial model."""
    medians = {c: float(X[c].median()) for c in CONTEXT_NUMERIC}
    design = poly_design(tyre_clip.clip(X), medians)
    estimator = Ridge(alpha=alpha).fit(design.to_numpy(), y.to_numpy(dtype=np.float64))
    return PolyRidgeModel(estimator, tuple(design.columns), medians, tyre_clip)
