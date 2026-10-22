from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from boxbox.features.build import DRIVER_STATE, STINT_STATE
from boxbox.features.tyre_prior import TyrePrior
from boxbox.models.forecaster import SequenceForecaster, bind_sequence_model, one_step_pace
from boxbox.models.lstm import (
    LstmParams,
    TrainedLstm,
    load_lstm,
    predict_batch,
    save_lstm,
    train_lstm,
    training_samples,
)
from boxbox.models.sequence_data import (
    FUTURE_FEATURES,
    HISTORY_FEATURES,
    BoolArray,
    build_sequences,
    driver_tables,
    future_from_design,
    histories_for_decisions,
    history_for_state,
)
from boxbox.models.types import FloatArray, TyreAgeClip
from boxbox.strategy.pit_window import StrategyInputs, pit_costs
from boxbox.strategy.rollout import (
    new_tyre_design,
    predict_segments,
    stay_out_design,
    stay_out_rollout,
)

DEG = 0.08
TOTAL_LAPS = 36
DESIGN_COLUMNS = ["compound", "tyre_age", "stint_lap", "lap_number", "race_progress"]
CLIP = TyreAgeClip({"SOFT": 40.0, "MEDIUM": 40.0, "HARD": 40.0, "UNKNOWN": 40.0})
FLAT_PRIOR = TyrePrior(
    coefficients={"POOLED_DRY": (0.0, 0.0)},
    caps={"POOLED_DRY": 40.0},
    end_slope={"POOLED_DRY": 0.0},
)


def _driver_rows(
    season: int, driver: str, offset: float, stops: tuple[int, ...], noise: float
) -> list[dict[str, object]]:
    rng = np.random.default_rng([season, ord(driver)])
    race_id = f"{season}_01"
    compounds = ("SOFT", "MEDIUM", "HARD")
    rows: list[dict[str, object]] = []
    stint, age = 1, 0
    for lap in range(1, TOTAL_LAPS + 1):
        if lap - 1 in stops:
            stint, age = stint + 1, 0
        age += 1
        pit_in, pit_out = lap in stops, lap - 1 in stops
        y = offset + DEG * age - 1.0 + rng.normal(0, noise)
        y += 4.0 * pit_in + 18.0 * pit_out
        rows.append(
            {
                "race_id": race_id,
                "season": season,
                "driver_id": driver,
                "lap": lap,
                "split": "train",
                "era": "tyre_data_2018_plus",
                "y_s": np.nan if lap == 1 else y,
                "is_clean": lap > 1 and not (pit_in or pit_out),
                "compound": compounds[(stint - 1) % 3],
                "stint_id": stint,
                "stint_lap": age,
                "tyre_age": float(age),
                "lap_number": float(lap),
                "race_progress": lap / TOTAL_LAPS,
                "total_laps": TOTAL_LAPS,
                "pit_in": pit_in,
                "pit_out": pit_out,
                "track_neutralised": False,
                "s_driver_offset_s": offset,
                "s_gap_ahead_s": 1.5,
                "s_in_traffic": 0.0,
            }
        )
    return rows


def synthetic_features(seasons: tuple[int, ...], noise: float = 0.0) -> pd.DataFrame:
    drivers = (
        ("a", 0.0, (14,)),
        ("b", 0.3, (10, 22)),
        ("c", 0.6, (18,)),
        ("d", 0.9, (12, 25)),
    )
    rows = [
        row
        for season in seasons
        for driver, offset, stops in drivers
        for row in _driver_rows(season, driver, offset, stops, noise)
    ]
    return pd.DataFrame(rows)


FAST = LstmParams(window=6, horizon=8, hidden=24, epochs=40, patience=10, dropout=0.0)


def test_sequence_shapes_and_keys() -> None:
    features = synthetic_features((2019,))
    batch = build_sequences(features, window=6, horizon=8)
    assert batch.history.shape == (len(batch), 6, len(HISTORY_FEATURES))
    assert batch.future.shape == (len(batch), 8, len(FUTURE_FEATURES))
    assert batch.target.shape == batch.target_mask.shape == (len(batch), 8)
    assert list(batch.keys.columns) == ["race_id", "driver_id", "lap", "split"]
    assert not np.isnan(batch.history).any()
    assert batch.keys["lap"].min() == 2  # lap 1 is never clean, so it has no clean history


@settings(max_examples=30, deadline=None)
@given(t=st.integers(min_value=2, max_value=TOTAL_LAPS - 1), seed=st.integers(0, 10_000))
def test_history_is_causal(t: int, seed: int) -> None:
    features = synthetic_features((2019,), noise=0.05)
    rng = np.random.default_rng(seed)
    later = features["lap"] > t
    n = int(later.sum())
    perturbed = features.copy()
    perturbed.loc[later, "y_s"] = rng.normal(0, 50, n)
    perturbed.loc[later, "is_clean"] = rng.random(n) > 0.5
    perturbed.loc[later, "compound"] = rng.choice(["SOFT", "HARD", "WET"], n)
    perturbed.loc[later, "stint_id"] = rng.integers(1, 9, n)
    perturbed.loc[later, "tyre_age"] = rng.normal(0, 99, n)
    perturbed.loc[later, "s_driver_offset_s"] = rng.normal(0, 9, n)

    before = build_sequences(features, 6, 8)
    after = build_sequences(perturbed, 6, 8)
    rows_before = (before.keys["lap"] == t).to_numpy()
    rows_after = (after.keys["lap"] == t).to_numpy()
    np.testing.assert_array_equal(before.history[rows_before], after.history[rows_after])
    np.testing.assert_array_equal(before.history_mask[rows_before], after.history_mask[rows_after])
    for driver in features["driver_id"].unique():
        h0, m0 = history_for_state(features, "2019_01", driver, t, 6)
        h1, m1 = history_for_state(perturbed, "2019_01", driver, t, 6)
        np.testing.assert_array_equal(h0, h1)
        np.testing.assert_array_equal(m0, m1)


def test_masks_for_pit_and_unclean_laps() -> None:
    features = synthetic_features((2019,))
    batch = build_sequences(features, 6, 8)
    # Driver "a" boxes at the end of lap 14: in-lap 14, out-lap 15 (both unclean).
    row = int(np.flatnonzero((batch.keys["driver_id"] == "a") & (batch.keys["lap"] == 12))[0])
    assert batch.target_mask[row].tolist() == [True, False, False, True, True, True, True, True]
    new_stint = batch.future[row, :, FUTURE_FEATURES.index("new_stint")]
    assert new_stint.tolist() == [0, 0, 1, 1, 1, 1, 1, 1]
    assert np.isfinite(batch.target[row, 1])  # the value is kept, only masked

    later = int(np.flatnonzero((batch.keys["driver_id"] == "a") & (batch.keys["lap"] == 17))[0])
    clean = batch.history[later, :, HISTORY_FEATURES.index("is_clean")]
    pit = batch.history[later, :, HISTORY_FEATURES.index("pit_lap")]
    current = batch.history[later, :, HISTORY_FEATURES.index("current_stint")]
    assert batch.history_mask[later].all()  # laps 12..17 exist; pit laps are kept
    assert clean.tolist() == [1, 1, 0, 0, 1, 1]
    assert pit.tolist() == [0, 0, 1, 1, 0, 0]
    assert current.tolist() == [0, 0, 0, 1, 1, 1]
    y = batch.history[later, :, HISTORY_FEATURES.index("y_s")]
    assert y.max() <= 5.0  # the 18 s out-lap is bounded in history

    last = int(np.flatnonzero((batch.keys["driver_id"] == "a") & (batch.keys["lap"] == 33))[0])
    assert batch.target_mask[last].tolist() == [True, True, True] + [False] * 5
    assert (batch.future[last, 3:] == 0).all()

    first = int(np.flatnonzero((batch.keys["driver_id"] == "a") & (batch.keys["lap"] == 3))[0])
    assert batch.history_mask[first].tolist() == [False, False, False, False, True, True]


def test_future_from_design_matches_build_sequences() -> None:
    features = synthetic_features((2019,))
    batch = build_sequences(features, 6, 8)
    row = int(np.flatnonzero((batch.keys["driver_id"] == "a") & (batch.keys["lap"] == 10))[0])
    rows = features[(features["driver_id"] == "a")].set_index("lap")
    stay = rows.loc[11:14, DESIGN_COLUMNS]
    fresh = rows.loc[15:18, DESIGN_COLUMNS]
    future = np.concatenate([future_from_design(stay, False), future_from_design(fresh, True)])
    np.testing.assert_array_equal(future, batch.future[row])
    h, m = history_for_state(features, "2019_01", "a", 10, 6)
    np.testing.assert_array_equal(h, batch.history[row])
    np.testing.assert_array_equal(m, batch.history_mask[row])


def test_overfits_tiny_synthetic_race() -> None:
    features = synthetic_features((2019,))
    params = FAST.model_copy(update={"epochs": 250})
    model = train_lstm(features, params, CLIP, FLAT_PRIOR)
    batch = build_sequences(features, params.window, params.horizon)
    error = np.abs(predict_batch(model, batch) - batch.target)[batch.target_mask]
    assert model.inner_val_seasons == ()
    assert error.mean() < 0.05


@pytest.fixture(scope="module")
def degradation_model() -> tuple[pd.DataFrame, TrainedLstm]:
    features = synthetic_features((2019, 2020, 2021, 2022), noise=0.05)
    params = LstmParams(window=8, horizon=12, hidden=32, epochs=120, patience=25, dropout=0.0)
    return features, train_lstm(features, params, CLIP, FLAT_PRIOR)


def test_learns_degradation_slope_on_rollout(
    degradation_model: tuple[pd.DataFrame, TrainedLstm],
) -> None:
    features, model = degradation_model
    assert model.inner_val_seasons == (2021, 2022)
    slopes = []
    for driver, t, end in (("a", 6, 14), ("c", 6, 18), ("b", 15, 22)):
        rows = features[(features["race_id"] == "2022_01") & (features["driver_id"] == driver)]
        stay = rows[(rows["lap"] > t) & (rows["lap"] <= end)]
        history, mask = history_for_state(features, "2022_01", driver, t, model.params.window)
        y = model.predict_rollout(history, mask, future_from_design(stay[DESIGN_COLUMNS], False))
        slopes.append(np.polyfit(stay["tyre_age"].to_numpy(dtype=float), y, 1)[0])
    assert np.median(slopes) == pytest.approx(DEG, abs=0.02)


def test_zero_offsets_and_extra_wear_follow_the_prior_from_the_history_level() -> None:
    features = synthetic_features((2019,), noise=0.05)
    prior = TyrePrior(
        coefficients={"POOLED_DRY": (0.05, 0.0)},
        caps={"POOLED_DRY": 40.0},
        end_slope={"POOLED_DRY": 0.05},
    )
    model = train_lstm(features, FAST.model_copy(update={"epochs": 1}), CLIP, prior)
    torch.nn.init.zeros_(model.network.head.weight)
    torch.nn.init.zeros_(model.network.head.bias)
    torch.nn.init.constant_(model.network.extra_wear, -30.0)
    rows = features[
        (features["driver_id"] == "a") & (features["lap"] > 6) & (features["lap"] <= 12)
    ]
    history, mask = history_for_state(features, "2019_01", "a", 6, FAST.window)
    y = model.predict_rollout(history, mask, future_from_design(rows[DESIGN_COLUMNS], False))
    clean = (mask & (history[:, HISTORY_FEATURES.index("is_clean")] > 0)).astype(bool)
    hist_age = history[:, HISTORY_FEATURES.index("tyre_age")]
    level = np.mean(history[clean, 0] - 0.05 * hist_age[clean])
    np.testing.assert_allclose(y, level + 0.05 * rows["tyre_age"].to_numpy(), atol=1e-5)


def test_inner_split_is_the_last_two_training_seasons_with_a_fallback_to_one() -> None:
    def seasons(years: tuple[int, ...]) -> tuple[int, ...]:
        return training_samples(synthetic_features(years), FAST)[2]

    assert seasons((2019, 2020, 2021)) == (2020, 2021)
    assert seasons((2019, 2020)) == (2020,)
    assert seasons((2019,)) == ()


def test_deterministic_with_fixed_seed() -> None:
    features = synthetic_features((2019, 2020), noise=0.05)
    params = FAST.model_copy(update={"epochs": 5})
    batch = build_sequences(features, params.window, params.horizon)
    first = predict_batch(train_lstm(features, params, CLIP, FLAT_PRIOR), batch)
    second = predict_batch(train_lstm(features, params, CLIP, FLAT_PRIOR), batch)
    np.testing.assert_array_equal(first, second)


def test_save_load_round_trip(tmp_path: Path) -> None:
    features = synthetic_features((2019, 2020), noise=0.05)
    params = FAST.model_copy(update={"epochs": 5, "cell": "gru"})
    model = train_lstm(features, params, CLIP, FLAT_PRIOR)
    save_lstm(model, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"lstm.pt", "lstm.meta.json"}
    loaded = load_lstm(tmp_path)
    batch = build_sequences(features, params.window, params.horizon)
    np.testing.assert_array_equal(predict_batch(model, batch), predict_batch(loaded, batch))
    assert loaded.curve == model.curve
    assert loaded.params == params
    assert loaded.tyre_clip == model.tyre_clip


def test_rollout_rejects_horizon_overflow(
    degradation_model: tuple[pd.DataFrame, TrainedLstm],
) -> None:
    features, model = degradation_model
    history, mask = history_for_state(features, "2021_01", "a", 3, model.params.window)
    rows = features[(features["race_id"] == "2021_01") & (features["driver_id"] == "a")]
    too_long = future_from_design(rows.iloc[3 : 3 + model.horizon + 1][DESIGN_COLUMNS], False)
    with pytest.raises(ValueError, match="horizon"):
        model.predict_rollout(history, mask, too_long)
    assert model.predict_rollout(history, mask, too_long[:0]).shape == (0,)


AGE = FUTURE_FEATURES.index("tyre_age")
NEW = FUTURE_FEATURES.index("new_stint")
PRIOR_SLOPE = 0.03
PRIOR = TyrePrior(
    coefficients={"POOLED_DRY": (PRIOR_SLOPE, 0.0)},
    caps={"POOLED_DRY": 100.0},
    end_slope={"POOLED_DRY": PRIOR_SLOPE},
)


@dataclass(frozen=True)
class RecordingSequenceModel:
    """y = 0.1 x clipped tyre age + 1 on a new set; records every decoder input."""

    horizon: int = 30
    window: int = 4
    tyre_clip: TyreAgeClip = CLIP
    calls: list[FloatArray] = field(default_factory=list)

    @property
    def name(self) -> Literal["lstm"]:
        return "lstm"

    def predict_rollout(
        self, history: FloatArray, history_mask: BoolArray, future: FloatArray
    ) -> FloatArray:
        return self.predict_rollouts(history[None], history_mask[None], future[None])[0]

    def predict_rollouts(
        self, history: FloatArray, history_mask: BoolArray, future: FloatArray
    ) -> FloatArray:
        self.calls.append(future.copy())
        cap = self.tyre_clip.max_age["SOFT"]
        age = np.minimum(future[..., AGE], cap)
        return np.asarray(0.1 * age + future[..., NEW], dtype=np.float64)


def _decision_state(features: pd.DataFrame, driver: str, lap: int) -> pd.Series:
    """A feature row with the extra state columns ``rollout_design`` reads (zeros)."""
    row = features[(features["driver_id"] == driver) & (features["lap"] == lap)].iloc[0]
    extra = {f"s_{c}": 0.0 for c in (*DRIVER_STATE, *STINT_STATE)} | {"quali_gap_s": 0.0}
    return pd.Series(extra | {"s_field_ref_next_s": 0.0} | row.to_dict())


def _forecaster(model: RecordingSequenceModel) -> tuple[pd.DataFrame, SequenceForecaster]:
    features = synthetic_features((2019,))
    return features, bind_sequence_model(model, features, PRIOR)


def test_pit_branch_decodes_stay_out_prefix_then_new_set_flags() -> None:
    model = RecordingSequenceModel()
    features, forecaster = _forecaster(model)
    state = _decision_state(features, "a", 6)  # SOFT, tyre age 6
    old, new = predict_segments(
        forecaster, state, [stay_out_design(state, 9), new_tyre_design(state, 9, 12, "HARD")]
    )
    (call,) = model.calls
    assert call.shape[:2] == (2, 6)
    assert call[1, :, NEW].tolist() == [0, 0, 0, 1, 1, 1]
    assert call[1, :, AGE].tolist() == [7, 8, 9, 1, 2, 3]
    assert call[0, :3, NEW].tolist() == [0, 0, 0] and not call[0, 3:].any()  # padding
    np.testing.assert_allclose(old.y_s, 0.1 * np.array([7, 8, 9]))
    np.testing.assert_allclose(new.y_s[1:], 0.1 * np.array([2, 3]) + 1.0)
    # The out-lap (masked in training) is an ordinary age-1 lap at the set's level.
    level = np.median((0.1 - PRIOR_SLOPE) * np.array([2.0, 3.0]) + 1.0)
    np.testing.assert_allclose(new.y_s[0], level + PRIOR_SLOPE)
    assert not old.past_cap.any() and not new.past_cap.any()


def test_rollout_beyond_horizon_continues_with_prior_and_counts_past_cap() -> None:
    model = RecordingSequenceModel(horizon=5)
    features, forecaster = _forecaster(model)
    state = _decision_state(features, "a", 6)
    rollout = stay_out_rollout(forecaster, state, 16, 0.0)
    ages = np.arange(7, 17, dtype=np.float64)
    level = np.median((0.1 - PRIOR_SLOPE) * ages[2:5])  # last three predicted laps
    expected = np.concatenate([0.1 * ages[:5], level + PRIOR_SLOPE * ages[5:]])
    np.testing.assert_allclose(rollout.y_s, expected)
    assert rollout.extrapolated.tolist() == [False] * 5 + [True] * 5

    inputs = StrategyInputs("HARD", True, 20.0, 0.0, 1.0, 0.0)
    costs = pit_costs(forecaster, state, inputs)
    # Boxing at lap 7: every plan lap after lap 11 (decision lap 6 + horizon 5) is continued.
    assert costs.pit_laps[0] == 7
    assert costs.laps_past_cap[0] == TOTAL_LAPS - 11


def test_beyond_horizon_level_ignores_the_out_lap_of_a_new_set() -> None:
    model = RecordingSequenceModel(horizon=5)
    features, forecaster = _forecaster(model)
    state = _decision_state(features, "a", 6)
    # Box at lap 10: the only predicted new-set lap (11) is the out-lap, so the level comes
    # from the old set's last laps (8-10) and the whole new set follows the prior from there.
    _, new = predict_segments(
        forecaster, state, [stay_out_design(state, 10), new_tyre_design(state, 10, 20, "HARD")]
    )
    level = np.median((0.1 - PRIOR_SLOPE) * np.array([8.0, 9.0, 10.0]))
    np.testing.assert_allclose(new.y_s, level + PRIOR_SLOPE * np.arange(1, 11))
    assert new.past_cap.tolist() == [False] + [True] * 9


@pytest.mark.parametrize(("min_slope", "slope"), [(0.2, 0.2), (0.05, 0.1)])
def test_beyond_cap_continues_with_the_larger_of_own_and_minimum_slope(
    min_slope: float, slope: float
) -> None:
    clip = TyreAgeClip({c: 8.0 for c in CLIP.max_age}, {"SOFT": min_slope})
    model = RecordingSequenceModel(tyre_clip=clip)
    features, forecaster = _forecaster(model)
    state = _decision_state(features, "a", 3)  # ages 4..13: cap 8 inside the window
    (prediction,) = predict_segments(forecaster, state, [stay_out_design(state, 13)])
    ages = np.arange(4, 14, dtype=np.float64)
    expected = 0.1 * np.minimum(ages, 8) + slope * np.clip(ages - 8, 0, None)
    np.testing.assert_allclose(prediction.y_s, expected)
    assert prediction.past_cap.tolist() == (ages > 8).tolist()


def test_one_step_pace_decodes_each_row_from_the_previous_lap() -> None:
    model = RecordingSequenceModel()
    features, forecaster = _forecaster(model)
    rows = features[features["lap"].between(10, 20)]
    previous = features.set_index(["driver_id", "lap"])["stint_id"]
    keys = list(zip(rows["driver_id"], rows["lap"] - 1, strict=True))
    new_set = rows["stint_id"].to_numpy() != previous.loc[keys].to_numpy()
    expected = 0.1 * rows["tyre_age"].to_numpy() + new_set
    np.testing.assert_allclose(one_step_pace(forecaster, rows), expected)


def test_batched_histories_equal_single_decision_histories() -> None:
    features = synthetic_features((2019, 2020), noise=0.05)
    keys = [("2019_01", "a", 2), ("2020_01", "b", 15), ("2019_01", "a", 30), ("2019_01", "d", 9)]
    history, mask = histories_for_decisions(
        driver_tables(features), [k[0] for k in keys], [k[1] for k in keys], [k[2] for k in keys], 6
    )
    for i, (race_id, driver_id, lap) in enumerate(keys):
        h, m = history_for_state(features, race_id, driver_id, lap, 6)
        np.testing.assert_array_equal(history[i], h)
        np.testing.assert_array_equal(mask[i], m)
    with pytest.raises(ValueError, match="no feature row"):
        histories_for_decisions(driver_tables(features), ["2019_01"], ["a"], [99], 6)


def test_lstm_respects_the_sequence_protocol(
    degradation_model: tuple[pd.DataFrame, TrainedLstm],
) -> None:
    features, model = degradation_model
    forecaster = bind_sequence_model(model, features, PRIOR)
    state = _decision_state(features[features["race_id"] == "2022_01"], "a", 6)
    long_run = stay_out_design(state, TOTAL_LAPS)
    (prediction,) = predict_segments(forecaster, state, [long_run])
    assert len(prediction.y_s) == TOTAL_LAPS - 6 and np.isfinite(prediction.y_s).all()
    assert prediction.past_cap.sum() == TOTAL_LAPS - 6 - model.horizon
