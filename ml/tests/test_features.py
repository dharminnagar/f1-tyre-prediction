import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from boxbox.config import Config, load_config
from boxbox.evaluation.evaluate import _stint_ids, rollout_slope_rows, within_group_slope
from boxbox.evaluation.splits import assign_splits
from boxbox.features.build import (
    CATEGORICAL,
    DEGRADATION_FEATURES,
    MAIN_NUMERIC,
    STINT_STATE,
    build_features,
)
from boxbox.features.pace import field_reference, fuel_corrected, teammate_reference
from boxbox.features.tyre_prior import (
    POOLED_DRY,
    TyrePrior,
    fit_tyre_prior,
    tyre_prior_curve,
)
from boxbox.models.train import fit_models_on, training_rows
from boxbox.preprocessing.cleaning import flag_laps
from boxbox.synthetic import canonical_pipeline, synthetic_race

CAUSAL_COLUMNS = [*CATEGORICAL, *MAIN_NUMERIC, *STINT_STATE, "field_ref_s", "fuel_term_s"]
_CFG = load_config("subset")
_TABLES, _TYRES = synthetic_race()
_UNFLAGGED = canonical_pipeline(_TABLES, _TYRES, _CFG).drop(
    columns=["is_clean", "exclusion_reason", "track_neutralised"]
)
# A prior fitted on the race itself would let later laps shape the offset at earlier laps.
_FIXED_PRIOR = TyrePrior(
    coefficients={POOLED_DRY: (0.08, 0.0), "SOFT": (0.1, -0.001)},
    caps={POOLED_DRY: 20.0, "SOFT": 15.0},
    end_slope={POOLED_DRY: 0.08, "SOFT": 0.07},
)


def _features(laps: pd.DataFrame) -> pd.DataFrame:
    return build_features(flag_laps(laps, _CFG), _CFG, prior=_FIXED_PRIOR).set_index(
        ["driver_id", "lap"]
    )


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


def _stint_rows(n_stints: int, rate: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for stint in range(n_stints):
        ages = np.arange(1.0 + stint % 3, 16.0 + stint % 3)
        frames.append(
            pd.DataFrame(
                {
                    "race_id": "r",
                    "driver_id": f"d{stint}",
                    "stint_id": 1,
                    "compound": "SOFT",
                    "era": "tyre_data_2018_plus",
                    "is_clean": True,
                    "tyre_age": ages,
                    "y_s": rng.normal(0, 1) + rate * ages + rng.normal(0, 0.05, len(ages)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_tyre_prior_recovers_within_stint_rate() -> None:
    prior = fit_tyre_prior(_stint_rows(20, 0.08), {"SOFT": 15.0})
    ages = np.array([2.0, 12.0])
    slope = float(np.diff(tyre_prior_curve(prior, "SOFT", ages))[0]) / 10
    assert abs(slope - 0.08) < 0.01
    assert prior.coefficients["SOFT"] == pytest.approx(prior.coefficients[POOLED_DRY])


def test_tyre_prior_ignores_refuel_era_and_driver_levels() -> None:
    rows = _stint_rows(20, 0.08)
    refuel = _stint_rows(20, -0.2, seed=1).assign(
        era="refuel_2005_2009", driver_id=lambda d: "old_" + d["driver_id"]
    )
    prior = fit_tyre_prior(pd.concat([rows, refuel], ignore_index=True), {"SOFT": 15.0})
    slope = float(np.diff(tyre_prior_curve(prior, "SOFT", np.array([2.0, 12.0])))[0]) / 10
    assert abs(slope - 0.08) < 0.01


def test_tyre_prior_curve_is_monotone_and_linear_beyond_cap() -> None:
    prior = TyrePrior(
        coefficients={POOLED_DRY: (0.05, 0.0), "SOFT": (0.12, -0.006)},
        caps={POOLED_DRY: 20.0, "SOFT": 15.0},
        end_slope={POOLED_DRY: 0.05, "SOFT": 0.0},
    )
    ages = np.arange(0.0, 40.0, 0.5)
    soft = tyre_prior_curve(prior, "SOFT", ages)
    assert (np.diff(soft) >= -1e-12).all()
    assert np.ptp(soft[ages >= 15.0]) == pytest.approx(0.0)
    unknown = tyre_prior_curve(prior, "UNKNOWN", np.array([10.0, 30.0]))
    assert unknown[1] - unknown[0] == pytest.approx(0.05 * 20)


def test_tyre_adjusted_offset_does_not_track_tyre_age(
    synthetic_laps: pd.DataFrame, cfg: Config
) -> None:
    def first_stint_offset_slope(mode: str) -> float:
        features = build_features(synthetic_laps, cfg.model_copy(update={"driver_offset": mode}))
        rows = features[features["is_training_row"] & (features["stint_id"] == 1)]
        return within_group_slope(rows["tyre_age"], rows["driver_offset_s"], _stint_ids(rows))

    raw, adjusted = first_stint_offset_slope("raw"), first_stint_offset_slope("tyre_adjusted")
    assert raw > 0.015
    assert abs(adjusted) < 0.015


def test_rollout_slope_rows_cover_every_model_and_split(
    synthetic_laps: pd.DataFrame, cfg: Config
) -> None:
    val = canonical_pipeline(*synthetic_race(season=2024, seed=3), cfg)
    features = build_features(pd.concat([synthetic_laps, val], ignore_index=True), cfg)
    fitted = fit_models_on(training_rows(features, ("train",)), cfg)
    rows = rollout_slope_rows(features, fitted.models)
    assert {(r["model"], r["split"]) for r in rows} == {
        (m, s) for m in fitted.models for s in ("train", "held_out")
    }
    for row in rows:
        assert row["n_decisions"] > 0
        assert np.isfinite(float(str(row["rollout_s_per_lap"])))
        assert 0.0 < float(str(row["empirical_s_per_lap"])) < 0.12


def _referenced(laps: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return field_reference(fuel_corrected(laps, cfg.fuel_s_per_lap))


def test_teammate_reference_excludes_own_laps(synthetic_laps: pd.DataFrame, cfg: Config) -> None:
    referenced = teammate_reference(_referenced(synthetic_laps, cfg))
    alpha = referenced[referenced["driver_id"] == "alpha"].set_index("lap")
    bravo = referenced[(referenced["driver_id"] == "bravo") & referenced["is_clean"]]
    expected = float(bravo.loc[bravo["lap"] <= 8, "lap_fc_s"].median())
    assert alpha.loc[8, "s_field_ref_next_s"] == pytest.approx(expected)
    assert alpha.loc[9, "field_ref_s"] == pytest.approx(expected)


def test_teammate_reference_falls_back_to_field(synthetic_laps: pd.DataFrame, cfg: Config) -> None:
    solo = _referenced(synthetic_laps[synthetic_laps["driver_id"] != "bravo"], cfg)
    referenced = teammate_reference(solo).set_index(["driver_id", "lap"])
    field = solo.set_index(["driver_id", "lap"])
    pd.testing.assert_series_equal(
        referenced.loc["alpha", "field_ref_s"], field.loc["alpha", "field_ref_s"]
    )
