from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from boxbox.config import Config
from boxbox.evaluation.evaluate import lowest_val_mae, pit_timing
from boxbox.features.build import build_features
from boxbox.features.tyre_prior import tyre_prior_curve
from boxbox.models.baseline_curve import BaselineCurveModel
from boxbox.models.poly_regression import poly_design
from boxbox.models.train import fit_models_on, model_rows, training_rows
from boxbox.models.types import TyreAgeClip, predict_pace
from boxbox.strategy.uncertainty import SigmaCurve
from boxbox.synthetic import canonical_pipeline, synthetic_race


@pytest.fixture
def training(synthetic_laps: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    features = build_features(synthetic_laps, cfg)
    return features[features["is_training_row"]]


def _age_grid(training: pd.DataFrame, compound: str) -> pd.DataFrame:
    row = training.iloc[[0]]
    grid = pd.concat([row] * 20, ignore_index=True)
    return grid.assign(
        compound=compound, tyre_age=np.arange(1.0, 21.0), stint_lap=np.arange(1.0, 21.0)
    )


def test_all_models_fit_and_predict(training: pd.DataFrame, cfg: Config) -> None:
    gbm = cfg.models.gbm.model_copy(update={"min_samples_leaf": 5})
    small = cfg.model_copy(update={"models": cfg.models.model_copy(update={"gbm": gbm})})
    fitted = fit_models_on(training, small)
    for model in fitted.models.values():
        predictions = model.predict(training)
        assert predictions.shape == (len(training),)
        assert np.isfinite(predictions).all()


def test_poly_recovers_degradation_rate(training: pd.DataFrame, cfg: Config) -> None:
    poly = fit_models_on(training, cfg).models["poly_ridge"]
    cap = poly.tyre_clip.max_age["SOFT"]
    ages = np.array([2.0, np.floor(cap)])
    grid = _age_grid(training, "SOFT").iloc[:2].assign(tyre_age=ages, stint_lap=ages)
    pred = poly.predict(grid)
    slope = (pred[1] - pred[0]) / (ages[1] - ages[0])
    assert 0.06 < slope < 0.10
    assert "tyre_age:SOFT" in poly.describe()


def test_poly_age_degree_controls_quadratic_terms(training: pd.DataFrame) -> None:
    linear = poly_design(training, {}, age_degree=1)
    quadratic = poly_design(training, {}, age_degree=2)
    assert not any(c.startswith("tyre_age^2") for c in linear.columns)
    assert "tyre_age^2:SOFT" in quadratic.columns


def test_baseline_curve_slope_is_the_tyre_prior(training: pd.DataFrame, cfg: Config) -> None:
    fitted = fit_models_on(training, cfg)
    baseline = fitted.models["baseline_curve"]
    assert isinstance(baseline, BaselineCurveModel)
    grid = _age_grid(training, "MEDIUM").iloc[:10]
    ages = grid["tyre_age"].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(
        np.diff(baseline.predict(grid)), np.diff(tyre_prior_curve(fitted.prior, "MEDIUM", ages))
    )
    assert "MEDIUM@1" in baseline.describe()


def test_beyond_cap_slope_is_at_least_min_slope(training: pd.DataFrame, cfg: Config) -> None:
    poly = fit_models_on(training, cfg).models["poly_ridge"]
    floored = replace(poly, tyre_clip=TyreAgeClip(poly.tyre_clip.max_age, {"SOFT": 0.5}))
    cap = poly.tyre_clip.max_age["SOFT"]
    grid = _age_grid(training, "SOFT").assign(tyre_age=[cap + i for i in range(20)])
    assert np.diff(predict_pace(floored, grid)) == pytest.approx(np.full(19, 0.5))


def test_min_slope_comes_from_the_prior(training: pd.DataFrame, cfg: Config) -> None:
    fitted = fit_models_on(training, cfg)
    for model in fitted.models.values():
        for compound in model.tyre_clip.max_age:
            assert model.tyre_clip.min_slope[compound] == fitted.prior.end_slope[compound]


def test_refuel_era_excluded_per_model(training: pd.DataFrame, cfg: Config) -> None:
    rows = training.assign(
        era=np.where(training["driver_id"] == "alpha", "refuel_2005_2009", training["era"])
    )
    params = cfg.models.model_copy(update={"exclude_refuel_era_for": ("poly_ridge", "hgb")})
    excluding = cfg.model_copy(update={"models": params})
    assert len(model_rows(rows, "hgb", excluding)) < len(rows)
    assert len(model_rows(rows, "baseline_curve", excluding)) == len(rows)
    era_coef = "era=refuel_2005_2009"
    assert fit_models_on(rows, excluding).models["poly_ridge"].describe()[era_coef] == 0.0
    assert fit_models_on(rows, cfg).models["poly_ridge"].describe()[era_coef] != 0.0


def test_hgb_is_monotone_in_tyre_age(training: pd.DataFrame, cfg: Config) -> None:
    hgb = fit_models_on(training, cfg).models["hgb"]
    for compound in ("SOFT", "MEDIUM", "HARD"):
        assert (np.diff(hgb.predict(_age_grid(training, compound))) >= -1e-9).all()


def test_tyre_age_beyond_cap_is_extrapolated_not_flat(training: pd.DataFrame, cfg: Config) -> None:
    poly = fit_models_on(training, cfg).models["poly_ridge"]
    cap = poly.tyre_clip.max_age["SOFT"]
    grid = _age_grid(training, "SOFT").assign(tyre_age=[cap + i for i in range(20)])
    assert poly.tyre_clip.extrapolated(grid).iloc[1:].all()
    assert (np.diff(predict_pace(poly, grid)) >= 0).all()
    assert np.ptp(poly.predict(grid)) == pytest.approx(0.0)


def test_models_roundtrip_through_joblib(
    training: pd.DataFrame, cfg: Config, tmp_path: Path
) -> None:
    fitted = fit_models_on(training, cfg)
    for name, model in fitted.models.items():
        path = tmp_path / f"{name}.joblib"
        joblib.dump(model, path)
        restored = joblib.load(path)
        np.testing.assert_allclose(restored.predict(training), model.predict(training))


def test_strategy_selection_uses_its_own_target() -> None:
    rows: list[dict[str, object]] = [
        {"model": "hgb", "split": "val", "target": "next_lap", "mae": 0.40},
        {"model": "poly_ridge", "split": "val", "target": "next_lap", "mae": 0.45},
        {"model": "hgb", "split": "val", "target": "next_10_mean", "mae": 0.35},
        {"model": "poly_ridge", "split": "val", "target": "next_10_mean", "mae": 0.30},
        {"model": "poly_ridge", "split": "test", "target": "next_10_mean", "mae": 0.10},
    ]
    assert lowest_val_mae(rows, "next_lap") == "hgb"
    assert lowest_val_mae(rows, "next_10_mean") == "poly_ridge"
    assert lowest_val_mae(rows, "next_3_mean", fallback="hgb") == "hgb"


def test_pit_timing_keeps_safety_car_stops_with_a_reason(cfg: Config) -> None:
    train = canonical_pipeline(*synthetic_race(season=2023, seed=1), cfg)
    tables, tyres = synthetic_race(season=2024, seed=2)
    tyres.loc[tyres["lap"] == 13, "track_status"] = "4"
    features = build_features(
        pd.concat([train, canonical_pipeline(tables, tyres, cfg)], ignore_index=True), cfg
    )
    model = fit_models_on(training_rows(features, ("train",)), cfg).models["poly_ridge"]
    sigma = SigmaCurve((0.5,), (0.5, 0.8, 1.0), 0.5, 0.6)
    timing = pit_timing(features, model, sigma, 22.0, cfg).set_index(["driver_id", "stint_id"])
    assert timing.loc[("alpha", 1), "excluded_reason"] == "neutralised"
    assert timing.loc[("bravo", 1), "excluded_reason"] is None
    assert set(timing["split"]) == {"val"}
