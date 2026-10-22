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
from boxbox.evaluation.evaluate import EvaluationResult
from boxbox.ingestion.download import season_sources
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
from boxbox.schema import DRY_COMPOUNDS
from boxbox.strategy.pit_loss import stop_losses

SCHEMA_VERSION: Final[str] = "1"
MODEL_FILE_SUFFIX: Final[str] = ".joblib"


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
            "val_mae_s": model_mae.get((name, "val")),
            "test_mae_s": model_mae.get((name, "test")),
            "coefficients": description if name == "poly_ridge" else None,
            "importances": description if name == "hgb" else None,
            "tyre_age_cap": model.tyre_clip.max_age,
        }
        write_json(paths.models / f"{name}.meta.json", meta)
        model_entries.append(meta)

    features.to_parquet(paths.data / "laps.parquet", index=False)
    dataset.pit_stops.to_parquet(paths.data / "pit_stops.parquet", index=False)
    dataset.stint_verification.to_parquet(paths.data / "stint_verification.parquet", index=False)
    selected_model = evaluation.models.models[evaluation.selected_model]
    curves = degradation_curves(features, selected_model)
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
        "splits": cfg.splits.model_dump(),
        "races": race_catalogue(features, cfg),
        "models": model_entries,
        "pit_loss": evaluation.pit_loss.__dict__ | {"races": evaluation.pit_loss_races},
        "sigma": evaluation.summary["sigma"],
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
