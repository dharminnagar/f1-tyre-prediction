"""Model 3: histogram gradient boosting, monotone non-decreasing in tyre age."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from boxbox.config import GbmParams
from boxbox.features.build import CATEGORICAL, MAIN_NUMERIC
from boxbox.models.types import FloatArray, TyreAgeClip, with_categories
from boxbox.schema import ModelName

FEATURES: tuple[str, ...] = (*CATEGORICAL, *MAIN_NUMERIC)


@dataclass(frozen=True)
class HgbModel:
    """Fitted HistGradientBoostingRegressor with native categorical compound/era."""

    estimator: HistGradientBoostingRegressor
    tyre_clip: TyreAgeClip
    features: tuple[str, ...] = FEATURES
    importances: dict[str, float] = field(default_factory=dict)
    name: ModelName = "hgb"

    def predict(self, X: pd.DataFrame) -> FloatArray:
        inputs = with_categories(self.tyre_clip.clip(X))[list(self.features)]
        return np.asarray(self.estimator.predict(inputs), dtype=np.float64)

    def describe(self) -> dict[str, float]:
        return dict(self.importances)


def fit_hgb(
    X: pd.DataFrame,
    y: pd.Series,
    tyre_clip: TyreAgeClip,
    params: GbmParams,
    seed: int,
    features: tuple[str, ...] = FEATURES,
) -> HgbModel:
    """Fit the boosted model (tyre age constrained to a non-negative effect)."""
    inputs = with_categories(tyre_clip.clip(X))[list(features)]
    estimator = HistGradientBoostingRegressor(
        max_iter=params.max_iter,
        learning_rate=params.learning_rate,
        max_leaf_nodes=params.max_leaf_nodes,
        min_samples_leaf=params.min_samples_leaf,
        categorical_features="from_dtype",
        monotonic_cst={"tyre_age": 1},
        random_state=seed,
    ).fit(inputs, y.to_numpy(dtype=np.float64))
    return HgbModel(estimator, tyre_clip, features)


def with_importances(model: HgbModel, importances: dict[str, float]) -> HgbModel:
    """Attach permutation importances computed on validation data."""
    return HgbModel(model.estimator, model.tyre_clip, model.features, importances, model.name)
