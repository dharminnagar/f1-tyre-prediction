"""Fit every model on the training split."""

from dataclasses import dataclass, replace

import pandas as pd

from boxbox.config import Config, LstmParams
from boxbox.features.build import DEGRADATION_FEATURES
from boxbox.features.tyre_prior import REFUEL_ERA, TyrePrior, fit_training_prior
from boxbox.models.baseline_curve import fit_baseline_curve
from boxbox.models.forecaster import Forecaster, bind_sequence_model
from boxbox.models.gbm import FEATURES as HGB_FEATURES
from boxbox.models.gbm import HgbModel, fit_hgb
from boxbox.models.lstm import TrainedLstm, train_lstm
from boxbox.models.poly_regression import fit_poly_ridge
from boxbox.models.quantile import QuantileHgb, fit_quantile_hgb
from boxbox.models.types import PaceModel, TyreAgeClip, fit_tyre_age_clip
from boxbox.schema import ModelName


@dataclass(frozen=True)
class FittedModels:
    """The per-lap models, the sequence model(s), companions and the tyre prior.

    ``models`` holds the three per-lap selectable models; ``lstm`` is the fourth selectable
    model (``None`` when not trained, e.g. in LORO folds) and ``gru`` the same network with a
    GRU cell, trained for the evaluation-only cell comparison. ``hgb_quantile`` gives HGB its
    80% intervals. ``tyre_clip`` is fitted on all training rows; a model trained on a subset of
    them (see ``ModelParams.exclude_refuel_era_for`` / ``LstmParams.exclude_eras``) carries
    its own clip.
    """

    models: dict[ModelName, PaceModel]
    hgb_deg_ablation: HgbModel
    tyre_clip: TyreAgeClip
    prior: TyrePrior
    lstm: TrainedLstm | None = None
    gru: TrainedLstm | None = None
    hgb_quantile: QuantileHgb | None = None


def training_rows(features: pd.DataFrame, splits: tuple[str, ...]) -> pd.DataFrame:
    """Clean rows with history in the given splits."""
    return features[features["is_training_row"] & features["split"].isin(splits)]


def fit_all(features: pd.DataFrame, cfg: Config) -> FittedModels:
    """Fit every model on ``split == train``, plus the GRU comparison and HGB quantiles."""
    rows = training_rows(features, ("train",))
    fitted = fit_models_on(rows, cfg, sequences_from=features, quantiles=True)
    gru_params = cfg.models.lstm.model_copy(update={"cell": "gru"})
    gru = train_lstm(
        features, gru_params, sequence_clip(rows, cfg, fitted.prior, gru_params), fitted.prior
    )
    return replace(fitted, gru=gru)


def model_rows(rows: pd.DataFrame, name: ModelName, cfg: Config) -> pd.DataFrame:
    """The training rows model ``name`` is fitted on (refuel era dropped when configured)."""
    if name in cfg.models.exclude_refuel_era_for:
        return rows[rows["era"].astype(str) != REFUEL_ERA]
    return rows


def _clip(rows: pd.DataFrame, cfg: Config, prior: TyrePrior) -> TyreAgeClip:
    caps = fit_tyre_age_clip(rows, cfg.tyre_age_clip_quantile).max_age
    return TyreAgeClip(caps, {c: prior.end_slope.get(c, 0.0) for c in caps})


def sequence_clip(
    rows: pd.DataFrame, cfg: Config, prior: TyrePrior, params: LstmParams
) -> TyreAgeClip:
    """Tyre-age caps of the sequence model's training rows (its excluded eras dropped)."""
    return _clip(rows[~rows["era"].astype(str).isin(params.exclude_eras)], cfg, prior)


def fit_models_on(
    rows: pd.DataFrame,
    cfg: Config,
    prior: TyrePrior | None = None,
    sequences_from: pd.DataFrame | None = None,
    quantiles: bool = False,
) -> FittedModels:
    """Fit all models on the given training rows (used by LORO-CV and sensitivities too).

    ``prior`` defaults to the tyre prior fitted on ``rows``, which is the prior
    :func:`~boxbox.features.build.build_features` adjusted the driver offset with when
    ``rows`` are all training rows. Beyond-cap continuation is floored at its end slopes.

    Args:
        rows: Clean training rows of the per-lap models.
        cfg: Pipeline configuration (hyper-parameters).
        prior: Tyre prior to use instead of one fitted on ``rows``.
        sequences_from: Full feature table (every lap, so unclean laps can be history) the
            LSTM is trained on (its ``split == "train"`` rows); ``None`` skips the LSTM.
        quantiles: Also fit the HGB quantile companions (80% intervals).
    """
    prior = prior or fit_training_prior(rows, cfg.tyre_age_clip_quantile)
    names: tuple[ModelName, ...] = ("baseline_curve", "poly_ridge", "hgb")
    subset = {name: model_rows(rows, name, cfg) for name in names}
    clips = {name: _clip(subset[name], cfg, prior) for name in names}
    base, ridge, hgb = subset["baseline_curve"], subset["poly_ridge"], subset["hgb"]
    models: dict[ModelName, PaceModel] = {
        "baseline_curve": fit_baseline_curve(base, base["y_s"], clips["baseline_curve"], prior),
        "poly_ridge": fit_poly_ridge(
            ridge,
            ridge["y_s"],
            clips["poly_ridge"],
            cfg.models.poly.alpha,
            cfg.models.poly.age_degree,
        ),
        "hgb": fit_hgb(hgb, hgb["y_s"], clips["hgb"], cfg.models.gbm, cfg.random_seed),
    }
    ablation = fit_hgb(
        hgb,
        hgb["y_s"],
        clips["hgb"],
        cfg.models.gbm,
        cfg.random_seed,
        features=(*HGB_FEATURES, *DEGRADATION_FEATURES),
    )
    lstm_params = cfg.models.lstm
    lstm = (
        None
        if sequences_from is None
        else train_lstm(
            sequences_from, lstm_params, sequence_clip(rows, cfg, prior, lstm_params), prior
        )
    )
    quantile = (
        fit_quantile_hgb(hgb, hgb["y_s"], clips["hgb"], cfg.models.gbm, cfg.random_seed)
        if quantiles
        else None
    )
    return FittedModels(
        models, ablation, _clip(rows, cfg, prior), prior, lstm=lstm, hgb_quantile=quantile
    )


def forecasters(fitted: FittedModels, features: pd.DataFrame) -> dict[ModelName, Forecaster]:
    """Every selectable model ready for strategy/evaluation (the LSTM bound to ``features``)."""
    out: dict[ModelName, Forecaster] = dict(fitted.models)
    if fitted.lstm is not None:
        out["lstm"] = bind_sequence_model(fitted.lstm, features, fitted.prior)
    return out
