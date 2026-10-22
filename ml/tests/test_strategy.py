import itertools
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import pytest

from boxbox.config import Config
from boxbox.evaluation.evaluate import (
    TimedStop,
    actual_excess_curves,
    actual_remaining_life,
    calibrate_margin,
    strategy_gate,
)
from boxbox.features.build import build_features, rollout_design
from boxbox.models.train import fit_models_on, training_rows
from boxbox.models.types import FloatArray, TyreAgeClip, predict_pace
from boxbox.schema import DRY_COMPOUNDS, ModelName
from boxbox.strategy.explain import decompose_next_lap, explanation_lines
from boxbox.strategy.pit_loss import estimate_pit_loss
from boxbox.strategy.pit_window import (
    MAX_EXTRA_STOPS,
    PitCosts,
    PlannedStop,
    StrategyInputs,
    choose_replacement,
    counterfactual,
    evaluate_pit_window,
    pit_costs,
)
from boxbox.strategy.remaining_life import (
    NO_NOISE,
    DeltaNoise,
    count_competitive,
    fit_delta_noise,
    noisy_competitive_count,
    remaining_competitive_laps,
)
from boxbox.strategy.rollout import (
    new_tyre_design,
    predict_decisions,
    predict_segments,
    stay_out_design,
    stay_out_rollout,
)
from boxbox.strategy.uncertainty import SigmaCurve, fit_sigma_curve

DEG = 0.1


@dataclass(frozen=True)
class LinearTyreModel:
    """y = DEG * tyre_age + driver_offset, identical for every compound."""

    name: ModelName = "poly_ridge"
    features: tuple[str, ...] = ("tyre_age", "driver_offset_s")
    tyre_clip: TyreAgeClip = field(
        default_factory=lambda: TyreAgeClip({"SOFT": 500.0, "MEDIUM": 500.0, "HARD": 500.0})
    )

    def predict(self, X: pd.DataFrame) -> FloatArray:
        offset = X["driver_offset_s"].fillna(0.0).to_numpy(dtype=np.float64)
        return np.asarray(DEG * X["tyre_age"].to_numpy(dtype=np.float64) + offset, dtype=np.float64)

    def describe(self) -> dict[str, float]:
        return {}


def _state(lap: int = 10, tyre_age: float = 10.0, total_laps: int = 40) -> pd.Series:
    return pd.Series(
        {
            "race_id": "2023_07",
            "driver_id": "alpha",
            "lap": lap,
            "total_laps": total_laps,
            "compound": "SOFT",
            "era": "tyre_data_2018_plus",
            "tyre_age": tyre_age,
            "stint_lap": tyre_age,
            "quali_gap_s": 0.0,
            "s_field_ref_next_s": 80.0,
            "s_driver_offset_s": 0.0,
            "s_position": 1.0,
            "s_gap_ahead_s": 5.0,
            "s_in_traffic": 0.0,
            "s_air_temp": 25.0,
            "s_track_temp": 40.0,
            "s_deg_slope_3": DEG,
            "s_deg_slope_5": DEG,
            "s_deg_accel": 0.0,
            "s_rolling_mean_3": 1.0,
            "s_rolling_mean_5": 1.0,
            "s_lap_delta": 0.1,
        }
    )


def _constant_sigma(value: float) -> SigmaCurve:
    return SigmaCurve((value,), (value,), value, 0.0)


def _inputs(pit_loss: float = 20.0, feasible: bool = True) -> StrategyInputs:
    return StrategyInputs("MEDIUM", feasible, pit_loss, 0.0, 1.0, 0.0)


def test_rollout_advances_tyre_age_and_freezes_field_reference() -> None:
    rollout = stay_out_rollout(LinearTyreModel(), _state(), 15, 0.0)
    assert rollout.laps.tolist() == [11, 12, 13, 14, 15]
    np.testing.assert_allclose(rollout.lap_time_s, 80.0 + DEG * np.arange(11, 16))


def test_remaining_life_exact_on_linear_degradation() -> None:
    model = LinearTyreModel()
    # delta(k) = DEG * (10 + k - 3) <= X  =>  k <= X / DEG - 7
    assert remaining_competitive_laps(model, _state(), 1.0, 3, 0.0).value == 3
    assert remaining_competitive_laps(model, _state(), 2.0, 3, 0.0).value == 13
    capped = remaining_competitive_laps(model, _state(total_laps=14), 2.0, 3, 0.0)
    assert capped.value == 4 and capped.capped_by_race_end


def test_pit_window_matches_closed_form() -> None:
    """Linear degradation, equal compounds: C(p) - C(p*) = DEG * (p - p*)^2 with p* = 20."""
    result = evaluate_pit_window(LinearTyreModel(), _state(), _constant_sigma(1.0), _inputs())
    half_width = int(np.floor(np.sqrt(1.0 / DEG)))
    assert result.optimal_lap is not None and abs(result.optimal_lap - 20) <= 1
    assert result.window is not None
    assert abs(result.window.start_lap - (20 - half_width)) <= 1
    assert abs(result.window.end_lap - (20 + half_width)) <= 1
    best = next(c for c in result.candidates if c.pit_lap == result.optimal_lap)
    # stay out: ages 11..40; one stop at 20: ages 11..20 then 1..20, plus the pit loss
    assert best.cost_stay_out_s - best.cost_pit_s == pytest.approx(DEG * (765 - 155 - 210) - 20.0)


def test_no_window_when_pit_loss_dominates_but_forced_when_infeasible() -> None:
    expensive = evaluate_pit_window(
        LinearTyreModel(), _state(), _constant_sigma(1.0), _inputs(pit_loss=200.0)
    )
    assert expensive.window is None
    forced = evaluate_pit_window(
        LinearTyreModel(), _state(), _constant_sigma(1.0), _inputs(pit_loss=200.0, feasible=False)
    )
    assert forced.window is not None and not forced.stay_out_feasible


def test_counterfactual_charges_pit_loss_on_out_lap() -> None:
    cf = counterfactual(LinearTyreModel(), _state(), 20, _inputs())
    deltas = np.diff([0.0, *cf.cumulative_delta_s])
    assert deltas[cf.laps.index(21)] == pytest.approx(20.0 + DEG * (1 - 21))
    assert cf.cumulative_delta_s[-1] == pytest.approx(20.0 - DEG * (765 - 155 - 210))


def test_choose_replacement_prefers_actual_next_then_hardest_unused() -> None:
    laps = pd.DataFrame(
        {
            "lap": [1, 2, 3, 4],
            "stint_id": [1, 1, 2, 2],
            "compound": ["SOFT", "SOFT", "MEDIUM", "MEDIUM"],
        }
    )
    assert choose_replacement(laps, 1) == ("MEDIUM", False)
    assert choose_replacement(laps, 4) == ("HARD", True)


def test_decomposition_sums_to_prediction() -> None:
    model, state = LinearTyreModel(), _state().copy()
    state["s_driver_offset_s"] = 0.4
    parts = decompose_next_lap(model, state, 3, 0.05)
    expected = stay_out_rollout(model, state, 11, 0.05).lap_time_s[0]
    assert parts.total_s == pytest.approx(expected)
    assert parts.tyre_s == pytest.approx(DEG * (11 - 3))
    assert parts.driver_offset_s == pytest.approx(0.4)
    result = evaluate_pit_window(model, state, _constant_sigma(1.0), _inputs())
    life = remaining_competitive_laps(model, state, 1.0, 3, 0.05)
    lines = explanation_lines(state, parts, life, result, 3)
    assert any("pit loss" in line for line in lines)


def test_pit_loss_exact_on_synthetic_stops() -> None:
    rows = []
    for driver, in_lap in (("a", 10), ("b", 12), ("c", 14)):
        for lap in range(1, 25):
            time = 80.0 + (4.0 if lap == in_lap else 0.0) + (18.0 if lap == in_lap + 1 else 0.0)
            rows.append(
                {
                    "race_id": "r",
                    "driver_id": driver,
                    "lap": lap,
                    "lap_time_s": time,
                    "pit_in": lap == in_lap,
                    "is_clean": lap not in (1, in_lap, in_lap + 1),
                    "track_neutralised": False,
                }
            )
    estimate = estimate_pit_loss(pd.DataFrame(rows), ["r"])
    assert estimate.median_s == pytest.approx(22.0)
    assert estimate.n_stops == 3


def test_sigma_curve_grows_like_sqrt_k_for_iid_errors() -> None:
    rng = np.random.default_rng(0)
    errors = pd.DataFrame(
        [
            {"decision_id": d, "k": k, "error_s": rng.normal()}
            for d in range(400)
            for k in range(1, 9)
        ]
    )
    sigma = fit_sigma_curve(errors)
    assert sigma.cumulative_sigma(4) == pytest.approx(2.0, rel=0.15)
    assert sigma.power_b == pytest.approx(0.5, abs=0.1)
    assert sigma.cumulative_sigma(30) > sigma.cumulative_sigma(8)


@dataclass(frozen=True)
class CompoundModel:
    """y = level[c] + rate[c] * tyre_age + driver_offset."""

    level: dict[str, float]
    rate: dict[str, float]
    name: ModelName = "poly_ridge"
    features: tuple[str, ...] = ("tyre_age", "driver_offset_s")
    tyre_clip: TyreAgeClip = field(
        default_factory=lambda: TyreAgeClip({"SOFT": 500.0, "MEDIUM": 500.0, "HARD": 500.0})
    )

    def predict(self, X: pd.DataFrame) -> FloatArray:
        compound = X["compound"].astype(str)
        level = compound.map(self.level).to_numpy(dtype=np.float64)
        rate = compound.map(self.rate).to_numpy(dtype=np.float64)
        offset = X["driver_offset_s"].fillna(0.0).to_numpy(dtype=np.float64)
        return np.asarray(level + rate * X["tyre_age"].to_numpy(dtype=np.float64) + offset)

    def describe(self) -> dict[str, float]:
        return {}


def _uniform(rate: float, caps: float = 500.0) -> CompoundModel:
    return CompoundModel(
        {c: 0.0 for c in DRY_COMPOUNDS},
        {c: rate for c in DRY_COMPOUNDS},
        tyre_clip=TyreAgeClip({c: caps for c in DRY_COMPOUNDS}),
    )


def _brute_force_costs(
    model: CompoundModel, state: pd.Series, stop_cost: float, rule_met: bool
) -> dict[int, float]:
    """Cheapest plan per first in-lap by enumerating every stop sequence and compound."""
    t, total, current = int(state["lap"]), int(state["total_laps"]), str(state["compound"])

    def stint(compound: str, first_age: float, laps: int) -> float:
        ages = first_age + np.arange(laps)
        return float(np.sum(80.0 + model.level[compound] + model.rate[compound] * ages))

    best: dict[int, float] = {}
    for p in range(t + 1, total):
        stay = stint(current, float(state["tyre_age"]) + 1, p - t)
        for n_extra in range(MAX_EXTRA_STOPS + 1):
            for later in itertools.combinations(range(p + 1, total), n_extra):
                bounds = [p, *later, total]
                for compounds in itertools.product(DRY_COMPOUNDS, repeat=n_extra + 1):
                    if not rule_met and all(c == current for c in compounds):
                        continue
                    cost = stay + (n_extra + 1) * stop_cost
                    cost += sum(
                        stint(c, 1.0, bounds[i + 1] - bounds[i]) for i, c in enumerate(compounds)
                    )
                    best[p] = min(best.get(p, np.inf), cost)
    return best


@pytest.mark.parametrize("rule_met", [False, True])
def test_dp_equals_brute_force_on_short_race(rule_met: bool) -> None:
    model = CompoundModel(
        {"SOFT": 0.0, "MEDIUM": 0.4, "HARD": 0.9}, {"SOFT": 0.5, "MEDIUM": 0.25, "HARD": 0.1}
    )
    state = _state(lap=2, tyre_age=2.0, total_laps=13)
    costs = pit_costs(model, state, StrategyInputs("HARD", rule_met, 2.5, 0.5, 1.0, 0.0))
    expected = _brute_force_costs(model, state, 3.0, rule_met)
    assert costs.pit_laps == tuple(expected)
    np.testing.assert_allclose(costs.cost_pit_s, [expected[p] for p in costs.pit_laps])
    for plan, cost in zip(costs.plans, costs.cost_pit_s, strict=True):
        assert 1 <= len(plan) <= MAX_EXTRA_STOPS + 1
        assert rule_met or any(stop.compound != "SOFT" for stop in plan)
        assert np.isfinite(cost)


def test_dp_switches_to_the_cheaper_compound_unless_the_first_is_fixed() -> None:
    model = CompoundModel(
        {"SOFT": 0.0, "MEDIUM": -1.0, "HARD": 0.5}, {c: 0.1 for c in DRY_COMPOUNDS}
    )
    free = evaluate_pit_window(model, _state(), _constant_sigma(1.0), _inputs())
    assert {stop.compound for stop in free.plan} == {"MEDIUM"}
    assert free.replacement_compound == "MEDIUM"
    fixed = evaluate_pit_window(
        model,
        _state(),
        _constant_sigma(1.0),
        replace(_inputs(), replacement_compound="HARD", compound_policy="fixed_first"),
    )
    assert fixed.plan[0].compound == "HARD"


def test_third_stop_only_when_degradation_is_steep() -> None:
    gentle = evaluate_pit_window(_uniform(0.1), _state(), _constant_sigma(1.0), _inputs())
    steep = evaluate_pit_window(_uniform(2.0), _state(), _constant_sigma(1.0), _inputs(5.0))
    assert len(gentle.plan) == 1
    assert len(steep.plan) == MAX_EXTRA_STOPS + 1
    laps = [stop.lap for stop in steep.plan]
    assert laps == sorted(laps) and laps[-1] < 40


def test_laps_past_cap_counts_plan_laps_beyond_the_training_cap() -> None:
    within = evaluate_pit_window(_uniform(0.1), _state(), _constant_sigma(1.0), _inputs())
    assert all(c.laps_past_cap == 0 and not c.extrapolated for c in within.candidates)
    capped = evaluate_pit_window(
        _uniform(0.1, caps=12.0), _state(), _constant_sigma(1.0), _inputs()
    )
    # boxing at lap 39 runs the current set from age 11 to 39: 27 laps beyond age 12
    last = capped.candidates[-1]
    assert last.pit_lap == 39 and last.laps_past_cap >= 27 and last.extrapolated


def _quadratic_costs(optimum: int = 20) -> PitCosts:
    laps = tuple(range(11, 40))
    return PitCosts(
        40,
        laps,
        tuple(0.1 * (p - optimum) ** 2 for p in laps),
        1000.0,
        tuple(0 for _ in laps),
        tuple((PlannedStop(p, "MEDIUM"),) for p in laps),
        True,
    )


def _stop(actual: int, split: str = "val", excluded: bool = False) -> TimedStop:
    return TimedStop(
        "2024_10",
        f"d{actual}",
        split,
        1,
        5,
        actual,
        "neutralised" if excluded else None,
        _quadratic_costs(),
    )


@pytest.mark.parametrize(("target", "expected"), [(0.5, 0.25), (0.75, 0.5), (1.0, 1.0), (1.1, 1.0)])
def test_calibration_picks_the_smallest_margin_reaching_the_target(
    target: float, expected: float
) -> None:
    # half-widths at margin m (sigma 1, cost 0.1 d^2): 0.25 -> 1, 0.5 -> 2, 0.75 -> 2, 1.0 -> 3
    stops = [_stop(20 + d) for d in range(4)]
    ignored = [_stop(35, split="test"), _stop(35, excluded=True)]
    chosen, rows = calibrate_margin([*stops, *ignored], _constant_sigma(1.0), target)
    assert chosen == expected
    assert [r["in_window_rate"] for r in rows] == [0.5, 0.75, 0.75, 1.0]
    assert all(r["n"] == 4 for r in rows)


def test_sigma_curve_can_use_paired_errors() -> None:
    errors = pd.DataFrame(
        {
            "decision_id": [0, 0, 1, 1] * 10,
            "k": [1, 2, 1, 2] * 10,
            "error_s": [1.0, 1.0, -1.0, -1.0] * 10,
            "paired_error_s": [0.1, 0.2, -0.1, -0.2] * 10,
        }
    ).assign(decision_id=lambda d: d["decision_id"] + 2 * (d.index // 4))
    raw = fit_sigma_curve(errors, min_samples=5)
    paired = fit_sigma_curve(errors, min_samples=5, error_col="paired_error_s")
    assert paired.cumulative_sigma(2) < raw.cumulative_sigma(2)


def _stint_frame(y: list[float], stint_id: int = 1, first_lap: int = 1) -> pd.DataFrame:
    n = len(y)
    return pd.DataFrame(
        {
            "race_id": "2024_10",
            "driver_id": "alpha",
            "stint_id": stint_id,
            "lap": np.arange(first_lap, first_lap + n),
            "stint_lap": np.arange(1, n + 1),
            "y_s": y,
            "is_clean": [i > 0 for i in range(n)],
            "pit_in": [i == n - 1 for i in range(n)],
        }
    )


def test_remaining_life_truth_is_exact_on_linear_degradation() -> None:
    stint = _stint_frame([DEG * a for a in range(1, 31)])
    truth = actual_remaining_life(stint, 1.05, decision="stint_lap_5")
    # reference = median of ages 2-4 = 0.3; exceeds 1.05 from age 14 on -> laps 6..13 count
    assert truth["actual_remaining"].tolist() == [8]
    assert not truth["censored"].any()
    delta = DEG * (np.arange(6, 31) - 3.0)
    assert count_competitive(delta, 1.05) == (8, False)  # same rule on predictions (ref age 3)


def test_reverting_spike_is_ignored_with_persistence() -> None:
    y = [0.0] * 30
    y[11] = 3.0
    truth = actual_remaining_life(_stint_frame(y), 1.0)
    assert truth["censored"].all()
    single = actual_remaining_life(_stint_frame(y), 1.0, persistence=1)
    assert single["actual_remaining"].tolist() == [11 - 5]
    spike = np.zeros(20)
    spike[3] = 3.0
    assert count_competitive(spike, 1.0) == (20, True)


def test_noise_free_count_is_count_competitive_of_the_offset_curve() -> None:
    delta = DEG * np.arange(1.0, 30.0)
    assert noisy_competitive_count(delta, 1.05, NO_NOISE) == count_competitive(delta, 1.05)
    shifted = DeltaNoise(offset_s=0.5, sigma_s=0.0, ar1=0.0)
    assert noisy_competitive_count(delta, 1.05, shifted) == count_competitive(delta + 0.5, 1.05)


def test_lap_noise_brings_the_expected_crossing_forward() -> None:
    delta = 0.05 * np.arange(1.0, 40.0)
    smooth, _ = count_competitive(delta, 1.0)
    noisy, capped = noisy_competitive_count(delta, 1.0, DeltaNoise(0.0, 0.6, 0.3))
    assert noisy < smooth
    assert not capped
    assert noisy_competitive_count(delta, 1.0, DeltaNoise(0.0, 0.6, 0.3))[0] == noisy


def test_noise_fit_recovers_offset_and_spread() -> None:
    rng = np.random.default_rng(1)
    delta = 0.05 * np.arange(1.0, 26.0)
    curves = [(delta, delta + 0.2 + rng.normal(0.0, 0.5, len(delta))) for _ in range(400)]
    noise = fit_delta_noise(curves)
    assert noise.offset_s == pytest.approx(0.2, abs=0.03)
    assert noise.sigma_s == pytest.approx(0.5, abs=0.03)
    assert noise.ar1 == pytest.approx(0.0, abs=0.05)
    assert fit_delta_noise([]) == NO_NOISE


def test_excess_curve_is_the_series_the_truth_searches() -> None:
    stint = _stint_frame([DEG * a for a in range(1, 31)])
    truth = actual_remaining_life(stint, 1.05, decision="stint_lap_5")
    (curve,) = actual_excess_curves(stint, truth)
    assert len(curve) == int(stint["lap"].max()) - int(truth["lap"].iloc[0])
    assert count_competitive(curve, 1.05) == (8, False)


def test_stint_lap_5_mode_has_one_row_per_stint() -> None:
    frame = pd.concat(
        [_stint_frame([0.05 * a for a in range(20)]), _stint_frame([0.0] * 15, 2, 21)],
        ignore_index=True,
    )
    once = actual_remaining_life(frame, 1.0, decision="stint_lap_5")
    every = actual_remaining_life(frame, 1.0, decision="all")
    assert once.groupby("stint_id").size().tolist() == [1, 1]
    assert once["lap"].tolist() == [5, 25]
    assert len(every) > len(once)


def test_per_lap_segments_equal_one_predict_pace_call_on_the_designs(
    synthetic_laps: pd.DataFrame, cfg: Config
) -> None:
    """Per-lap models are untouched by the sequence dispatch: same rows, same single call."""
    features = build_features(synthetic_laps, cfg)
    fitted = fit_models_on(training_rows(features, ("train",)), cfg)
    states = [
        features[(features["lap"] == lap) & features["has_history"]].iloc[0] for lap in (5, 9)
    ]
    for model in fitted.models.values():
        decisions = []
        legacy = []
        for state in states:
            t = int(state["lap"])
            segments = [stay_out_design(state, t + 6), new_tyre_design(state, t + 6, 30, "HARD")]
            decisions.append((state, segments))
            stay_laps = np.arange(t + 1, t + 7, dtype=np.int64)
            new_laps = np.arange(t + 7, 31, dtype=np.int64)
            age, stint_lap = float(state["tyre_age"]) + 1, float(state["stint_lap"]) + 1
            compound = str(state["compound"])
            legacy += [
                rollout_design(state, stay_laps, compound, age, stint_lap, new_stint=False),
                rollout_design(state, new_laps, "HARD", 1.0, 1.0, new_stint=True),
            ]
        expected = predict_pace(model, pd.concat(legacy, ignore_index=True))
        batched = predict_decisions(model, decisions)
        got = np.concatenate([p.y_s for segments in batched for p in segments])
        np.testing.assert_array_equal(got, expected)
        # One decision is one call on its own rows (batch size can move Ridge by an ulp).
        single = predict_segments(model, *decisions[0])
        first = predict_pace(model, pd.concat(legacy[:2], ignore_index=True))
        np.testing.assert_array_equal(np.concatenate([p.y_s for p in single]), first)


def test_strategy_gate_needs_both_validation_slope_and_window_coverage() -> None:
    passed = strategy_gate("lstm", 0.065, 0.5)
    assert passed.eligible and passed.reason is None
    flat = strategy_gate("lstm", 0.044, 0.5)
    assert not flat.eligible
    assert flat.reason == "forecast only: validation rollout slope 0.044 s/lap < 0.06"
    missing = strategy_gate("lstm", 0.07, None)
    assert not missing.eligible
    assert missing.reason == "forecast only: validation in-window rate n/a < 0.4"
