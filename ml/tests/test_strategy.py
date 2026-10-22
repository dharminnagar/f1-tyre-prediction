from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from boxbox.models.types import FloatArray, TyreAgeClip
from boxbox.schema import ModelName
from boxbox.strategy.explain import decompose_next_lap, explanation_lines
from boxbox.strategy.pit_loss import estimate_pit_loss
from boxbox.strategy.pit_window import (
    StrategyInputs,
    choose_replacement,
    counterfactual,
    evaluate_pit_window,
)
from boxbox.strategy.remaining_life import remaining_competitive_laps
from boxbox.strategy.rollout import stay_out_rollout
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
