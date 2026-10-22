import numpy as np
import pandas as pd
import pytest

from boxbox.config import Config, GbmParams
from boxbox.features.build import build_features
from boxbox.models.gbm import fit_hgb
from boxbox.models.quantile import (
    centred_interval,
    fit_quantile_hgb,
    interval_coverage,
    interval_offsets,
    predict_interval,
)
from boxbox.models.types import TyreAgeClip, fit_tyre_age_clip, predict_pace

PARAMS = GbmParams(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40)
SIMPLE_FEATURES = ("compound", "era", "tyre_age", "stint_lap")
CLIP = TyreAgeClip(max_age={"SOFT": 40.0, "MEDIUM": 40.0, "HARD": 40.0})


def _heteroscedastic(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Linear degradation whose noise grows with tyre age (sd 0.1 s new, 1.1 s at 40 laps)."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(1.0, 40.0, n)
    X = pd.DataFrame(
        {
            "compound": rng.choice(["SOFT", "MEDIUM", "HARD"], n),
            "era": "tyre_data_2018_plus",
            "tyre_age": age,
            "stint_lap": age,
        }
    )
    y = 0.05 * age + rng.normal(0.0, 0.1 + 0.025 * age)
    return X, pd.Series(y)


@pytest.fixture(scope="module")
def heteroscedastic_fit() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.Series]:
    X_train, y_train = _heteroscedastic(4000, seed=1)
    X_test, y_test = _heteroscedastic(2000, seed=2)
    model = fit_quantile_hgb(X_train, y_train, CLIP, PARAMS, seed=0, features=SIMPLE_FEATURES)
    lo, hi = predict_interval(model, X_test)
    return lo, hi, X_test, y_test


def test_interval_covers_about_80_percent_held_out(
    heteroscedastic_fit: tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.Series],
) -> None:
    lo, hi, _, y_test = heteroscedastic_fit
    assert 0.7 < interval_coverage(lo, hi, y_test.to_numpy(dtype=np.float64)) < 0.9


def test_interval_is_ordered(
    heteroscedastic_fit: tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.Series],
) -> None:
    lo, hi, _, _ = heteroscedastic_fit
    assert (lo <= hi).all()


def test_interval_widens_with_noise(
    heteroscedastic_fit: tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.Series],
) -> None:
    lo, hi, X_test, _ = heteroscedastic_fit
    width = hi - lo
    age = X_test["tyre_age"].to_numpy()
    assert width[age > 30].mean() > 2.0 * width[age < 10].mean()


def test_fit_is_deterministic_with_fixed_seed() -> None:
    X, y = _heteroscedastic(1000, seed=3)
    first = predict_interval(fit_quantile_hgb(X, y, CLIP, PARAMS, 7, features=SIMPLE_FEATURES), X)
    second = predict_interval(fit_quantile_hgb(X, y, CLIP, PARAMS, 7, features=SIMPLE_FEATURES), X)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_rejects_invalid_quantiles() -> None:
    X, y = _heteroscedastic(200, seed=4)
    with pytest.raises(ValueError, match="quantiles"):
        fit_quantile_hgb(X, y, CLIP, PARAMS, 0, quantiles=(0.9, 0.1), features=SIMPLE_FEATURES)


def test_fits_pipeline_features_with_default_columns(
    synthetic_laps: pd.DataFrame, cfg: Config
) -> None:
    features = build_features(synthetic_laps, cfg)
    training = features[features["is_training_row"]]
    clip = fit_tyre_age_clip(training, cfg.tyre_age_clip_quantile)
    params = PARAMS.model_copy(update={"min_samples_leaf": 5})
    model = fit_quantile_hgb(training, training["y_s"], clip, params, seed=cfg.random_seed)
    lo, hi = predict_interval(model, training)
    assert lo.shape == hi.shape == (len(training),)
    assert np.isfinite(lo).all() and (lo <= hi).all()


def test_coverage_ignores_non_finite_actuals() -> None:
    lo = np.array([0.0, 0.0, 0.0, 0.0])
    hi = np.array([1.0, 1.0, 1.0, 1.0])
    actual = np.array([0.5, 2.0, np.nan, np.inf])
    assert interval_coverage(lo, hi, actual) == 0.5


def test_coverage_includes_bounds() -> None:
    lo = np.array([0.0, 0.0])
    hi = np.array([1.0, 1.0])
    assert interval_coverage(lo, hi, np.array([0.0, 1.0])) == 1.0


def test_coverage_without_finite_actuals_is_nan() -> None:
    empty = np.array([], dtype=np.float64)
    assert np.isnan(interval_coverage(empty, empty, empty))
    assert np.isnan(interval_coverage(np.zeros(2), np.ones(2), np.array([np.nan, np.nan])))


def test_hgb_interval_is_centred_on_the_beyond_cap_continuation() -> None:
    X, y = _heteroscedastic(4000, seed=3)
    clip = TyreAgeClip({c: 30.0 for c in ("SOFT", "MEDIUM", "HARD")}, {"SOFT": 0.05})
    point = fit_hgb(X, y, clip, PARAMS, seed=0, features=SIMPLE_FEATURES)
    quantile = fit_quantile_hgb(X, y, clip, PARAMS, seed=0, features=SIMPLE_FEATURES)
    ages = np.array([30.0, 36.0, 45.0])
    rows = pd.DataFrame(
        {"compound": "SOFT", "era": "tyre_data_2018_plus", "tyre_age": ages, "stint_lap": ages}
    )
    lo, hi = centred_interval(point, quantile, rows)
    centre = predict_pace(point, rows)
    assert (lo <= centre).all() and (centre <= hi).all()
    assert centre[2] > centre[0]  # continued past the cap, not flat
    np.testing.assert_allclose(hi - lo, hi[0] - lo[0])  # cap-age width kept beyond the cap
    lo_off, hi_off = interval_offsets(point, quantile, rows)
    np.testing.assert_allclose(lo, centre + lo_off)
    np.testing.assert_allclose(hi, centre + hi_off)
    assert clip.extrapolated(rows).tolist() == [False, True, True]
