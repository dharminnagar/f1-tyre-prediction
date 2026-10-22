"""Write models, data, evaluation tables, figures and the manifest for one profile."""

import datetime as dt
import json
import math
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd

from boxbox.config import PROFILES, Config
from boxbox.evaluation.evaluate import EvaluationResult, lowest_val_mae
from boxbox.ingestion.download import season_sources
from boxbox.models.lstm import META_FILE, WEIGHTS_FILE, TrainedLstm, save_lstm
from boxbox.models.sequence_data import FUTURE_FEATURES, HISTORY_FEATURES
from boxbox.models.types import PaceModel, predict_pace
from boxbox.paths import ArtifactPaths, artifact_paths
from boxbox.preprocessing.build import LapDataset, dataset_hash
from boxbox.reporting.eda import degradation_curve_table
from boxbox.reporting.figures import (
    degradation_curves_figure,
    error_distribution_figure,
    horizon_figure,
    pit_loss_figure,
    pred_vs_actual_figure,
)
from boxbox.schema import DRY_COMPOUNDS, ModelName
from boxbox.strategy.pit_loss import stop_losses

SCHEMA_VERSION: Final[str] = "1"
MODEL_FILE_SUFFIX: Final[str] = ".joblib"
HGB_QUANTILE_FILE: Final[str] = "hgb_quantile.joblib"


def _json_safe(value: object) -> object:
    """Recursively replace NaN/inf with None and numpy scalars with Python scalars."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: object) -> None:
    """Write strict JSON (no NaN tokens)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, default=str))


def degradation_curves(features: pd.DataFrame, model: PaceModel) -> pd.DataFrame:
    """Empirical curves (all train races and per served race) with the model's fitted curve."""
    train = features[features["split"] == "train"]
    tables = [degradation_curve_table(train).assign(race_id="all")]
    for race_id, race in features[features["tyre_source"] != "none"].groupby("race_id"):
        tables.append(degradation_curve_table(race).assign(race_id=race_id))
    curves = pd.concat(tables, ignore_index=True)
    curves = curves[curves["compound"].isin(DRY_COMPOUNDS)]
    medians = train[train["is_training_row"]].median(numeric_only=True)
    reference = pd.DataFrame(
        {
            "compound": curves["compound"],
            "era": "tyre_data_2018_plus",
            "tyre_age": curves["tyre_age"].astype(float),
            "stint_lap": curves["tyre_age"].astype(float),
        }
    )
    for column in model.features:
        if column not in reference:
            reference[column] = float(medians.get(column, 0.0))
    reference["driver_offset_s"] = 0.0
    return curves.assign(fitted_s=predict_pace(model, reference))


def curve_model_name(evaluation: EvaluationResult) -> ModelName:
    """Model whose fitted degradation curve is exported: the selected one if it is per-lap.

    A sequence model cannot score isolated (compound, tyre age) rows, so when it is selected
    the per-lap model with the lowest validation next-lap MAE draws the curve instead.
    """
    if evaluation.selected_model in evaluation.models.models:
        return evaluation.selected_model
    per_lap = [r for r in evaluation.metric_rows if r["model"] in evaluation.models.models]
    return lowest_val_mae(per_lap, "next_lap")


def lstm_entry(
    model: TrainedLstm, evaluation: EvaluationResult, mae: dict[tuple[str, str], object]
) -> dict[str, object]:
    """Manifest model entry of the sequence model (``ModelInfo`` shape)."""
    return {
        "name": "lstm",
        "features": [
            *(f"history:{c}" for c in HISTORY_FEATURES),
            *(f"future:{c}" for c in FUTURE_FEATURES),
        ],
        "selected": evaluation.selected_model == "lstm",
        "selected_for_strategy": evaluation.strategy_selected_model == "lstm",
        "val_mae_s": mae.get(("lstm", "val")),
        "test_mae_s": mae.get(("lstm", "test")),
        "coefficients": None,
        "importances": None,
        "tyre_age_cap": model.tyre_clip.max_age,
    }


def race_catalogue(laps: pd.DataFrame, cfg: Config) -> list[dict[str, object]]:
    """Races with split, length and the data sources actually used."""
    races: list[dict[str, object]] = []
    for race_id, race in laps.groupby("race_id"):
        season = int(race["season"].iloc[0])
        sources = season_sources(season)
        races.append(
            {
                "race_id": race_id,
                "season": season,
                "round": int(race["round"].iloc[0]),
                "name": cfg.event_name,
                "circuit": cfg.circuit,
                "total_laps": int(race["total_laps"].iloc[0]),
                "split": str(race["split"].iloc[0]),
                "tables_source": sources.tables_source,
                "tyre_source": sources.tyre_source,
                "source_fallbacks": sources.fallbacks,
                "servable": sources.tyre_source is not None,
            }
        )
    return races


def update_combined_metrics(root_metrics: Path) -> None:
    """Merge every profile's ``eval/metrics.json`` into ``artifacts/metrics.json``."""
    combined: dict[str, object] = {}
    for profile in PROFILES:
        path = artifact_paths(profile).eval / "metrics.json"
        if path.exists():
            combined[profile] = json.loads(path.read_text())
    write_json(
        root_metrics, {"profiles": combined, "updated_at": dt.datetime.now(dt.UTC).isoformat()}
    )


def export_all(
    cfg: Config, dataset: LapDataset, features: pd.DataFrame, evaluation: EvaluationResult
) -> ArtifactPaths:
    """Write the full artifact tree for ``cfg.profile``."""
    paths = artifact_paths(cfg.profile)
    for directory in (paths.models, paths.data, paths.eval, paths.figures):
        directory.mkdir(parents=True, exist_ok=True)

    model_mae = {
        (str(r["model"]), str(r["split"])): r["mae"]
        for r in evaluation.metric_rows
        if r.get("target") == "next_lap"
    }
    model_entries = []
    for name, model in evaluation.models.models.items():
        joblib.dump(model, paths.models / f"{name}{MODEL_FILE_SUFFIX}")
        description = model.describe()
        meta = {
            "name": name,
            "features": list(model.features),
            "selected": name == evaluation.selected_model,
            "selected_for_strategy": name == evaluation.strategy_selected_model,
            "val_mae_s": model_mae.get((name, "val")),
            "test_mae_s": model_mae.get((name, "test")),
            "coefficients": description if name == "poly_ridge" else None,
            "importances": description if name == "hgb" else None,
            "tyre_age_cap": model.tyre_clip.max_age,
        }
        write_json(paths.models / f"{name}.meta.json", meta)
        model_entries.append(meta)
    lstm = evaluation.models.lstm
    if lstm is not None:
        save_lstm(lstm, paths.models)
        model_entries.append(lstm_entry(lstm, evaluation, model_mae))
    quantile = evaluation.models.hgb_quantile
    if quantile is not None:
        joblib.dump(quantile, paths.models / HGB_QUANTILE_FILE)

    features.to_parquet(paths.data / "laps.parquet", index=False)
    dataset.pit_stops.to_parquet(paths.data / "pit_stops.parquet", index=False)
    dataset.stint_verification.to_parquet(paths.data / "stint_verification.parquet", index=False)
    curve_model = curve_model_name(evaluation)
    curves = degradation_curves(features, evaluation.models.models[curve_model])
    curves.to_parquet(paths.data / "degradation_curves.parquet", index=False)

    write_json(paths.eval / "metrics.json", evaluation.summary)
    evaluation.predictions.to_parquet(paths.eval / "predictions.parquet", index=False)
    evaluation.error_groups.to_parquet(paths.eval / "error_groups.parquet", index=False)
    evaluation.threshold_sensitivity.to_parquet(
        paths.eval / "threshold_sensitivity.parquet", index=False
    )
    evaluation.pit_timing.to_parquet(paths.eval / "pit_timing.parquet", index=False)

    horizon = pd.DataFrame(evaluation.horizon_rows)
    losses = stop_losses(dataset.laps[dataset.laps["race_id"].isin(evaluation.pit_loss_races)])
    figure_paths = (
        degradation_curves_figure(curves, paths.figures / "degradation_curves.png"),
        horizon_figure(horizon, paths.figures / "horizon_mae.png"),
        pred_vs_actual_figure(
            evaluation.predictions, evaluation.selected_model, paths.figures / "pred_vs_actual.png"
        ),
        pit_loss_figure(losses, evaluation.pit_loss.median_s, paths.figures / "pit_loss.png"),
        error_distribution_figure(
            evaluation.predictions,
            evaluation.selected_model,
            paths.figures / "error_by_compound.png",
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "profile": cfg.profile,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "config_hash": cfg.config_hash(),
        "config": cfg.model_dump(mode="json"),
        "dataset_hash": dataset_hash(dataset.laps),
        "selected_model": evaluation.selected_model,
        "selection_rule": "lowest validation next-lap MAE",
        "strategy_selected_model": evaluation.strategy_selected_model,
        "strategy_selection_rule": evaluation.summary["strategy_selection_rule"],
        "curve_model": curve_model,
        "sequence_model_files": [f"models/{WEIGHTS_FILE}", f"models/{META_FILE}"]
        if lstm is not None
        else [],
        "hgb_quantile_file": f"models/{HGB_QUANTILE_FILE}" if quantile is not None else None,
        "interval_coverage": evaluation.summary["interval_coverage"],
        "sequence_cell_comparison": evaluation.summary["sequence_cell_comparison"],
        "splits": cfg.splits.model_dump(),
        "races": race_catalogue(features, cfg),
        "models": model_entries,
        "pit_loss": evaluation.pit_loss.__dict__ | {"races": evaluation.pit_loss_races},
        "sigma": evaluation.summary["sigma"],
        "sigma_paired": evaluation.summary["sigma_paired"],
        "calibrated_margin_sigma": evaluation.calibrated_margin_sigma,
        "tyre_prior": evaluation.summary["tyre_prior"],
        "rollout_slope": evaluation.summary["rollout_slope"],
        "normalisation_sensitivity": evaluation.summary["normalisation_sensitivity"],
        "stint_verification": evaluation.summary["stint_verification"],
        "figures": [p.name for p in figure_paths],
        "files": sorted(
            str(p.relative_to(paths.profile_dir))
            for p in paths.profile_dir.rglob("*")
            if p.is_file() and p != paths.manifest
        ),
    }
    write_json(paths.manifest, manifest)
    update_combined_metrics(paths.combined_metrics)
    return paths
