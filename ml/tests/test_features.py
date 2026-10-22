import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from boxbox.config import Config, load_config
from boxbox.evaluation.splits import assign_splits
from boxbox.features.build import (
    CATEGORICAL,
    DEGRADATION_FEATURES,
    MAIN_NUMERIC,
    STINT_STATE,
    build_features,
)
from boxbox.preprocessing.cleaning import flag_laps
from boxbox.synthetic import canonical_pipeline, synthetic_race

CAUSAL_COLUMNS = [*CATEGORICAL, *MAIN_NUMERIC, *STINT_STATE, "field_ref_s", "fuel_term_s"]
_CFG = load_config("subset")
_TABLES, _TYRES = synthetic_race()
_UNFLAGGED = canonical_pipeline(_TABLES, _TYRES, _CFG).drop(
    columns=["is_clean", "exclusion_reason", "track_neutralised"]
)


def _features(laps: pd.DataFrame) -> pd.DataFrame:
    return build_features(flag_laps(laps, _CFG), _CFG).set_index(["driver_id", "lap"])


@settings(max_examples=25, deadline=None)
@given(cut_lap=st.integers(min_value=3, max_value=29), seed=st.integers(0, 10_000))
def test_features_at_lap_l_ignore_laps_from_l_onwards(cut_lap: int, seed: int) -> None:
    """Perturbing every lap >= L (times, track status, weather) leaves features at L unchanged."""
    rng = np.random.default_rng(seed)
    perturbed = _UNFLAGGED.copy()
    future = perturbed["lap"] >= cut_lap
    perturbed.loc[future, "lap_time_s"] += rng.normal(0, 3, int(future.sum()))
    perturbed.loc[future, "track_status"] = rng.choice(["1", "4", "6"], int(future.sum()))
    perturbed.loc[future, "air_temp"] = rng.normal(30, 5, int(future.sum()))
    perturbed.loc[future, "rainfall"] = rng.random(int(future.sum())) < 0.2

    original = _features(_UNFLAGGED)
    changed = _features(perturbed)
    upto = original.index.get_level_values("lap") <= cut_lap
    pd.testing.assert_frame_equal(
        original.loc[upto, CAUSAL_COLUMNS], changed.loc[upto, CAUSAL_COLUMNS], check_dtype=False
    )


def test_target_is_field_relative_fuel_corrected(synthetic_laps: pd.DataFrame, cfg: Config) -> None:
    features = build_features(synthetic_laps, cfg)
    row = features[features["is_training_row"]].iloc[10]
    expected = row["lap_time_s"] - row["fuel_term_s"] - row["field_ref_s"]
    assert row["y_s"] == expected


def test_stint_state_resets_after_pit(synthetic_laps: pd.DataFrame, cfg: Config) -> None:
    features = build_features(synthetic_laps, cfg).set_index(["driver_id", "lap"])
    assert np.isnan(features.loc[("alpha", 13), "deg_slope_5"])
    assert not np.isnan(features.loc[("alpha", 13), "driver_offset_s"])


def test_degradation_slope_recovers_synthetic_rate(
    synthetic_laps: pd.DataFrame, cfg: Config
) -> None:
    features = build_features(synthetic_laps, cfg)
    late = features[(features["stint_lap"] >= 8) & features["deg_slope_5"].notna()]
    assert abs(late["deg_slope_5"].median() - 0.08) < 0.05


def test_no_race_in_more_than_one_split() -> None:
    for profile in ("subset", "full"):
        cfg = load_config(profile)  # type: ignore[arg-type]
        seasons = list(cfg.seasons)
        frame = pd.DataFrame({"season": seasons, "race_id": [f"{s}_05" for s in seasons]})
        splits = assign_splits(frame, cfg)
        assert not splits.duplicated(["race_id"]).any()
        assert set(splits["split"]) == {"train", "val", "test"}


def test_degradation_features_are_ablation_only() -> None:
    assert not set(DEGRADATION_FEATURES) & set(MAIN_NUMERIC)
