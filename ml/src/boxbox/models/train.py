"""Fit every model on the training split."""

from dataclasses import dataclass

import pandas as pd

from boxbox.config import Config
from boxbox.features.build import DEGRADATION_FEATURES
from boxbox.models.baseline_curve import fit_baseline_curve
from boxbox.models.gbm import FEATURES as HGB_FEATURES
from boxbox.models.gbm import HgbModel, fit_hgb
from boxbox.models.poly_regression import fit_poly_ridge
from boxbox.models.types import PaceModel, TyreAgeClip, fit_tyre_age_clip
from boxbox.schema import ModelName


@dataclass(frozen=True)
class FittedModels:
    """The three selectable models plus the degradation-feature ablation."""

    models: dict[ModelName, PaceModel]
    hgb_deg_ablation: HgbModel
    tyre_clip: TyreAgeClip


def training_rows(features: pd.DataFrame, splits: tuple[str, ...]) -> pd.DataFrame:
    """Clean rows with history in the given splits."""
    return features[features["is_training_row"] & features["split"].isin(splits)]


def fit_all(features: pd.DataFrame, cfg: Config) -> FittedModels:
    """Fit baseline curve, polynomial Ridge and monotone HGB on ``split == train``."""
    rows = training_rows(features, ("train",))
    return fit_models_on(rows, cfg)


def fit_models_on(rows: pd.DataFrame, cfg: Config) -> FittedModels:
    """Fit all models on the given training rows (used by LORO-CV too)."""
    y = rows["y_s"]
    clip = fit_tyre_age_clip(rows, cfg.tyre_age_clip_quantile)
    models: dict[ModelName, PaceModel] = {
        "baseline_curve": fit_baseline_curve(rows, y, clip),
        "poly_ridge": fit_poly_ridge(rows, y, clip, cfg.models.poly.alpha),
        "hgb": fit_hgb(rows, y, clip, cfg.models.gbm, cfg.random_seed),
    }
    ablation = fit_hgb(
        rows,
        y,
        clip,
        cfg.models.gbm,
        cfg.random_seed,
        features=(*HGB_FEATURES, *DEGRADATION_FEATURES),
    )
    return FittedModels(models, ablation, clip)
