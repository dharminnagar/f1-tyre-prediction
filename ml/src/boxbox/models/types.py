"""The pace-model contract and shared input preparation."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from boxbox.schema import ERAS, MODEL_COMPOUNDS, ModelName

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TyreAgeClip:
    """Per-compound tyre-age cap (training P95); larger ages are clipped and flagged."""

    max_age: dict[str, float]

    def clip(self, X: pd.DataFrame) -> pd.DataFrame:
        caps = X["compound"].astype(str).map(self.max_age).fillna(max(self.max_age.values()))
        return X.assign(
            tyre_age=np.minimum(X["tyre_age"], caps), stint_lap=np.minimum(X["stint_lap"], caps)
        )

    def extrapolated(self, X: pd.DataFrame) -> pd.Series:
        caps = X["compound"].astype(str).map(self.max_age).fillna(max(self.max_age.values()))
        return pd.Series(X["tyre_age"] > caps, index=X.index, dtype=bool)


def fit_tyre_age_clip(X: pd.DataFrame, quantile: float) -> TyreAgeClip:
    """Learn per-compound tyre-age caps from training rows."""
    caps = X.groupby(X["compound"].astype(str))["tyre_age"].quantile(quantile)
    return TyreAgeClip({str(k): float(v) for k, v in caps.items()})


def with_categories(X: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns to fixed vocabularies (stable one-hot / native categories)."""
    return X.assign(
        compound=pd.Categorical(X["compound"].astype(str), categories=list(MODEL_COMPOUNDS)),
        era=pd.Categorical(X["era"].astype(str), categories=list(ERAS)),
    )


class PaceModel(Protocol):
    """A fitted per-lap relative-pace model."""

    @property
    def name(self) -> ModelName: ...

    @property
    def features(self) -> tuple[str, ...]: ...

    @property
    def tyre_clip(self) -> TyreAgeClip: ...

    def predict(self, X: pd.DataFrame) -> FloatArray:
        """Predicted ``y_s`` per row of ``X`` (tyre age clipped internally)."""
        ...

    def describe(self) -> dict[str, float]:
        """Interpretable parameters (coefficients or importances), may be empty."""
        ...


EXTRAPOLATION_SLOPE_SPAN = 4.0


def predict_pace(model: PaceModel, X: pd.DataFrame) -> FloatArray:
    """Model prediction with linear continuation beyond the training tyre-age cap.

    Inside the cap this equals ``model.predict``. Beyond it, a flat clip would make very long
    stints look free, so the model's own slope over the last ``EXTRAPOLATION_SLOPE_SPAN`` laps
    before the cap (floored at 0) is continued. Rows beyond the cap stay flagged extrapolated.
    """
    base = model.predict(X)
    if X.empty:
        return base
    caps = (
        X["compound"]
        .astype(str)
        .map(model.tyre_clip.max_age)
        .fillna(max(model.tyre_clip.max_age.values()))
        .to_numpy(dtype=np.float64)
    )
    excess = np.clip(X["tyre_age"].to_numpy(dtype=np.float64) - caps, 0.0, None)
    if not excess.any():
        return base
    at_cap = X.assign(tyre_age=caps, stint_lap=np.minimum(X["stint_lap"], caps))
    before = at_cap.assign(
        tyre_age=caps - EXTRAPOLATION_SLOPE_SPAN,
        stint_lap=np.minimum(X["stint_lap"], caps - EXTRAPOLATION_SLOPE_SPAN),
    )
    slope = np.clip(
        (model.predict(at_cap) - model.predict(before)) / EXTRAPOLATION_SLOPE_SPAN, 0.0, None
    )
    return np.asarray(base + slope * excess, dtype=np.float64)
