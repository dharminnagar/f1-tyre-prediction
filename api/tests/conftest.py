"""A tiny in-memory artifact store built from two synthetic races (2023 train, 2024 val)."""

from collections.abc import Iterator
from dataclasses import replace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from boxbox.config import Config, LstmParams, load_config
from boxbox.evaluation.evaluate import next_lap_predictions
from boxbox.export.artifacts import degradation_curves
from boxbox.features.build import build_features
from boxbox.features.tyre_prior import TyrePrior
from boxbox.models.forecaster import SequenceForecaster
from boxbox.models.lstm import TrainedLstm, train_lstm
from boxbox.models.train import fit_models_on, forecasters, sequence_clip
from boxbox.schema import MODEL_NAMES
from boxbox.strategy.pit_loss import PitLossEstimate
from boxbox.strategy.uncertainty import SigmaCurve
from boxbox.synthetic import canonical_pipeline, synthetic_race
from boxbox_api.main import create_app
from boxbox_api.store import ProfileStore, Store


def _metrics(selected: str) -> dict[str, object]:
    row = {"model": selected, "split": "val", "mae": 0.4, "n": 10}
    return {
        "profile": "subset",
        "selected_model": selected,
        "selection_rule": "lowest validation next-lap MAE",
        "metrics": [{**row, "target": "next_lap", "rmse": 0.5, "r2": 0.6, "mape": 0.5}],
        "horizon": [{**row, "k": 1}],
        "loro_cv": [{"model": selected, "race_id": "2023_05", "mae": 0.5, "n": 10}],
        "loro_cv_summary": [
            {"model": selected, "mean": 0.5, "std": None, "status": "run"},
            {"model": "lstm", "mean": None, "std": None, "status": "not_run"},
        ],
        "cumulative_stint": [
            {
                "model": selected,
                "split": "val",
                "mean_abs_cumulative_s": 2.0,
                "mean_laps": 9.0,
                "n": 3,
            }
        ],
        "ablation_deg_features": [
            {"model": "hgb_deg_ablation", "split": "val", "mae": 0.4, "n": 10}
        ],
        "fuel_sensitivity": [
            {"fuel_s_per_lap": 0.05, "model": selected, "split": "val", "mae": 0.4}
        ],
        "pit_loss": {
            "median_s": 22.0,
            "q25_s": 21.0,
            "q75_s": 23.0,
            "n_stops": 8,
            "races": ["2023_05"],
        },
        "stint_verification": {"match_rate": 1.0, "n_stops": 8, "by_race": {"2023_05": 1.0}},
        "pit_timing_in_window_rate": 0.5,
        "pit_timing_summary": {
            "val": {
                "n": 1,
                "in_window_rate": 0.0,
                "n_kept": 0,
                "in_window_rate_kept": None,
                "excluded": {"neutralised": 1},
                "in_window_rate_kept_by_stint": {},
                "median_width_kept": None,
                "no_window_share_kept": None,
                "optimum_bias_kept_by_stint": {},
                "near_optimum_past_cap_share_kept": None,
            }
        },
        "normalisation_sensitivity": [
            {
                "normalisation": "field",
                "model": selected,
                "val_mae": 0.4,
                "test_mae": None,
                "held_out_rollout_s_per_lap": 0.07,
            }
        ],
        "calibrated_margin_sigma": 0.5,
        "window_target_coverage": 0.5,
        "interval_coverage": [
            {
                "model": "hgb",
                "split": "val",
                "target": "next_lap",
                "method": "quantile",
                "coverage": 0.78,
                "mean_width_s": 1.3,
                "n": 10,
            }
        ],
        "sequence_cell_comparison": [
            {
                "cell": cell,
                "split": "val",
                "next_lap_mae": 0.45,
                "next_n_mean_mae": 0.5,
                "n": 5,
                "horizon_k": 20,
                "horizon_k_mae": 0.6,
                "rollout_s_per_lap_held_out": 0.05,
                "rollout_s_per_lap_train": 0.05,
                "train_seconds": 12.0,
                "best_epoch": 20,
            }
            for cell in ("lstm", "gru")
        ],
        "margin_calibration": [
            {"margin_sigma": 0.5, "in_window_rate": 0.5, "median_width_laps": 7.0, "n": 2}
        ],
    }


TINY_LSTM = LstmParams(window=4, horizon=6, hidden=8, epochs=3, patience=2, dropout=0.0)


def tiny_lstm(
    features: pd.DataFrame, rows: pd.DataFrame, cfg: Config, prior: TyrePrior
) -> TrainedLstm:
    """A few-epoch sequence model on the synthetic training race (fast, not accurate)."""
    return train_lstm(features, TINY_LSTM, sequence_clip(rows, cfg, prior, TINY_LSTM))


@pytest.fixture(scope="session")
def profile_store() -> ProfileStore:
    cfg = load_config("subset")
    gbm = cfg.models.gbm.model_copy(update={"min_samples_leaf": 5})
    cfg = cfg.model_copy(update={"models": cfg.models.model_copy(update={"gbm": gbm})})
    laps = pd.concat(
        [
            canonical_pipeline(*synthetic_race(season=2023, seed=1), cfg),
            canonical_pipeline(*synthetic_race(season=2024, seed=2), cfg),
        ],
        ignore_index=True,
    )
    features = build_features(laps, cfg)
    rows = features[features["is_training_row"] & (features["split"] == "train")]
    fitted = fit_models_on(rows, cfg, quantiles=True)
    fitted = replace(fitted, lstm=tiny_lstm(features, rows, cfg, fitted.prior))
    models = forecasters(fitted, features)
    predictions = pd.concat(
        [next_lap_predictions(features, m) for m in models.values()], ignore_index=True
    ).assign(lo_s=lambda d: d["pred_s"] - 0.5, hi_s=lambda d: d["pred_s"] + 0.5)
    selected = "poly_ridge"
    races = [
        {
            "race_id": f"{season}_05",
            "season": season,
            "round": 5,
            "name": cfg.event_name,
            "circuit": cfg.circuit,
            "total_laps": 30,
            "split": split,
            "tyre_source": "synthetic",
            "tables_source": "synthetic",
            "servable": True,
        }
        for season, split in ((2023, "train"), (2024, "val"))
    ]
    model_meta = {
        name: {
            "name": name,
            "features": [] if isinstance(m, SequenceForecaster) else list(m.features),
            "selected": name == selected,
            "selected_for_strategy": name == "hgb",
            "coefficients": None,
            "importances": None,
            "val_mae_s": 0.4,
            "test_mae_s": None,
            "tyre_age_cap": m.tyre_clip.max_age,
        }
        for name, m in models.items()
    }
    error_groups = pd.DataFrame(
        [
            {
                "model": n,
                "group_by": "compound",
                "group": "SOFT",
                "q10_s": 0.1,
                "q25_s": 0.2,
                "q50_s": 0.3,
                "q75_s": 0.4,
                "q90_s": 0.5,
                "n": 10,
            }
            for n in MODEL_NAMES
        ]
    )
    threshold = pd.DataFrame(
        [
            {
                "decision": decision,
                "threshold_s": 1.0,
                "model": n,
                "mae_laps": float("nan"),
                "bias_laps": float("nan"),
                "n_uncensored": 0,
                "n_censored": 4,
                "censored_consistency": 0.75,
                "n_total": 4,
            }
            for n in MODEL_NAMES
            for decision in ("stint_lap_5", "all")
        ]
    )
    pit_timing = pd.DataFrame(
        [
            {
                "race_id": "2024_05",
                "driver_id": "alpha",
                "stint_id": 1,
                "decision_lap": 5,
                "actual_pit_lap": 12,
                "window_start": None,
                "window_end": None,
                "optimal_lap": 14,
                "actual_in_window": False,
                "split": "val",
                "excluded_reason": "neutralised",
                "planned_stops": 1,
                "near_optimum_past_cap_share": 0.0,
            }
        ]
    )
    return ProfileStore(
        profile="subset",
        manifest={
            "schema_version": "1",
            "created_at": "now",
            "config_hash": cfg.config_hash(),
            "dataset_hash": "synthetic",
        },
        config=cfg,
        models=models,
        model_meta=model_meta,  # type: ignore[arg-type]
        selected_model=selected,
        strategy_selected_model="hgb",
        laps=features,
        curves=degradation_curves(features, fitted.models[selected]),
        metrics=_metrics(selected),
        predictions=predictions,
        error_groups=error_groups,
        threshold_sensitivity=threshold,
        pit_timing=pit_timing,
        sigma={n: SigmaCurve((0.5,), (0.5, 0.8, 1.0), 0.5, 0.6) for n in MODEL_NAMES},
        pit_loss=PitLossEstimate(22.0, 21.0, 23.0, 8),
        races=races,  # type: ignore[arg-type]
        strategy_sigma={n: SigmaCurve((0.4,), (0.4, 0.6, 0.7), 0.4, 0.5) for n in MODEL_NAMES},
        calibrated_margin_sigma=0.5,
        hgb_quantile=fitted.hgb_quantile,
    )


@pytest.fixture(scope="session")
def client(profile_store: ProfileStore) -> Iterator[TestClient]:
    store = Store({"subset": profile_store}, "subset")
    with TestClient(create_app(store=store)) as test_client:
        yield test_client
