from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from boxbox.config import Config
from boxbox.features.build import build_features
from boxbox.models.train import fit_models_on
from boxbox.models.types import predict_pace


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
    grid = _age_grid(training, "SOFT")
    pred = poly.predict(grid)
    slope = (pred[14] - pred[4]) / 10
    # Attenuated below the true 0.08: in this synthetic race all stints age together, so the
    # causal field reference and the expanding driver offset absorb part of the shared trend.
    assert 0.02 < slope < 0.12
    assert "tyre_age:SOFT" in poly.describe()


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
