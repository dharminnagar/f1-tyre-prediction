"""Held-out evaluation of every model; test races are scored once, after selection on val."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from boxbox.config import Config
from boxbox.evaluation.metrics import regression_metrics
from boxbox.features.build import build_features
from boxbox.models.gbm import HgbModel, with_importances
from boxbox.models.train import FittedModels, fit_models_on, training_rows
from boxbox.models.types import PaceModel, predict_pace, with_categories
from boxbox.schema import MODEL_NAMES, ModelName
from boxbox.strategy.pit_loss import PitLossEstimate, estimate_pit_loss
from boxbox.strategy.pit_window import StrategyInputs, choose_replacement, evaluate_pit_window
from boxbox.strategy.remaining_life import count_competitive, tyre_delta_curve
from boxbox.strategy.rollout import lap_times, stay_out_design
from boxbox.strategy.uncertainty import SigmaCurve, fit_sigma_curve

ROLLOUT_MAX_K = 40


@dataclass
class EvaluationResult:
    """All evaluation tables; ``summary`` is the JSON-ready metrics document."""

    selected_model: ModelName
    pit_loss: PitLossEstimate
    sigma: dict[ModelName, SigmaCurve]
    models: FittedModels
    predictions: pd.DataFrame
    rollout_errors: pd.DataFrame
    error_groups: pd.DataFrame
    threshold_sensitivity: pd.DataFrame
    pit_timing: pd.DataFrame
    pit_loss_races: list[str] = field(default_factory=list)
    metric_rows: list[dict[str, object]] = field(default_factory=list)
    horizon_rows: list[dict[str, object]] = field(default_factory=list)
    summary: dict[str, object] = field(default_factory=dict)


def next_lap_predictions(features: pd.DataFrame, model: PaceModel) -> pd.DataFrame:
    """One-step predictions on clean rows (decision lap = row lap - 1)."""
    rows = features[features["is_training_row"]]
    pred_y = predict_pace(model, rows)
    return pd.DataFrame(
        {
            "model": model.name,
            "split": rows["split"].to_numpy(),
            "race_id": rows["race_id"].to_numpy(),
            "driver_id": rows["driver_id"].to_numpy(),
            "stint_id": rows["stint_id"].to_numpy(),
            "lap": rows["lap"].to_numpy(),
            "compound": rows["compound"].to_numpy(),
            "stint_length": rows.groupby(["race_id", "driver_id", "stint_id"])["stint_lap"]
            .transform("max")
            .to_numpy(),
            "actual_s": rows["lap_time_s"].to_numpy(),
            "pred_s": pred_y + rows["field_ref_s"].to_numpy() + rows["fuel_term_s"].to_numpy(),
            "actual_rel": rows["y_s"].to_numpy(),
            "pred_rel": pred_y,
        }
    )


def decision_rows(features: pd.DataFrame, splits: tuple[str, ...]) -> pd.DataFrame:
    """Decision laps: not an in-lap, valid state, at least one more lap in the same stint."""
    stint_end = features.groupby(["race_id", "driver_id", "stint_id"])["lap"].transform("max")
    mask = (
        features["split"].isin(splits)
        & ~features["pit_in"]
        & (features["lap"] < stint_end)
        & features["s_field_ref_next_s"].notna()
        & (features["lap"] >= 2)
    )
    return features[mask].assign(stint_end_lap=stint_end[mask])


def rollout_errors(
    features: pd.DataFrame, model: PaceModel, splits: tuple[str, ...], fuel_s_per_lap: float
) -> pd.DataFrame:
    """Stay-out rollouts to stint end (<= ROLLOUT_MAX_K laps) vs actual clean laps."""
    decisions = decision_rows(features, splits)
    actual = features.set_index(["race_id", "driver_id", "lap"])
    designs, metas = [], []
    for decision_id, (_, state) in enumerate(decisions.iterrows()):
        end = int(min(state["stint_end_lap"], state["lap"] + ROLLOUT_MAX_K))
        design = stay_out_design(state, end)
        designs.append(design)
        metas.append(
            pd.DataFrame(
                {
                    "decision_id": decision_id,
                    "race_id": state["race_id"],
                    "driver_id": state["driver_id"],
                    "stint_id": state["stint_id"],
                    "split": state["split"],
                    "decision_lap": int(state["lap"]),
                    "lap": design["lap_number"].astype(int).to_numpy(),
                    "field_ref": float(state["s_field_ref_next_s"]),
                    "fuel": lap_times(
                        state,
                        design["lap_number"].to_numpy(dtype=np.int64),
                        np.zeros(len(design)),
                        fuel_s_per_lap,
                    )
                    - float(state["s_field_ref_next_s"]),
                }
            )
        )
    if not designs:
        return pd.DataFrame()
    stacked = pd.concat(designs, ignore_index=True)
    meta = pd.concat(metas, ignore_index=True)
    meta["pred_s"] = predict_pace(model, stacked) + meta["field_ref"] + meta["fuel"]
    meta["k"] = meta["lap"] - meta["decision_lap"]
    keys = pd.MultiIndex.from_frame(meta[["race_id", "driver_id", "lap"]])
    meta["actual_s"] = actual["lap_time_s"].reindex(keys).to_numpy()
    meta["is_clean"] = actual["is_clean"].reindex(keys).fillna(value=False).to_numpy(dtype=bool)
    meta["error_s"] = meta["pred_s"] - meta["actual_s"]
    meta["model"] = model.name
    return meta[meta["is_clean"]].drop(columns=["field_ref", "fuel", "is_clean"])


def _metric_rows(
    predictions: pd.DataFrame, errors: pd.DataFrame, horizons: tuple[int, ...]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (model, split), group in predictions.groupby(["model", "split"]):
        m = regression_metrics(
            group["actual_s"].to_numpy(),
            group["pred_s"].to_numpy(),
            group["actual_rel"].to_numpy(),
            group["pred_rel"].to_numpy(),
        )
        rows.append({"model": model, "split": split, "target": "next_lap", **m.__dict__})
    for horizon in horizons:
        window = errors[errors["k"] <= horizon]
        min_laps = max(2, (horizon + 1) // 2 + 1)
        counts = window.groupby(["model", "decision_id"])["k"].transform("size")
        means = (
            window[counts >= min_laps]
            .groupby(["model", "split", "decision_id"])[["pred_s", "actual_s"]]
            .mean()
        )
        for (model, split), group in means.groupby(level=["model", "split"]):
            actual, pred = group["actual_s"].to_numpy(), group["pred_s"].to_numpy()
            m = regression_metrics(actual, pred, actual - actual.mean(), pred - actual.mean())
            target = f"next_{horizon}_mean"
            rows.append({"model": model, "split": split, "target": target, **m.__dict__})
    return rows


def _horizon_rows(errors: pd.DataFrame, max_k: int) -> list[dict[str, object]]:
    subset = errors[errors["k"] <= max_k]
    table = (
        subset.assign(abs_error=subset["error_s"].abs())
        .groupby(["model", "split", "k"])["abs_error"]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": "mae", "size": "n"})
    )
    return [{str(k): v for k, v in row.items()} for row in table.to_dict(orient="records")]


def _cumulative_stint_rows(errors: pd.DataFrame) -> list[dict[str, object]]:
    first = errors.groupby(["model", "race_id", "driver_id", "stint_id"])["decision_lap"].transform(
        "min"
    )
    from_start = errors[errors["decision_lap"] == first]
    per_stint = from_start.groupby(["model", "split", "race_id", "driver_id", "stint_id"]).agg(
        cumulative_error_s=("error_s", "sum"), laps=("error_s", "size")
    )
    summary = (
        per_stint.assign(abs_cum=per_stint["cumulative_error_s"].abs())
        .groupby(level=["model", "split"])
        .agg(
            mean_abs_cumulative_s=("abs_cum", "mean"),
            mean_laps=("laps", "mean"),
            n=("laps", "size"),
        )
        .reset_index()
    )
    return [{str(k): v for k, v in row.items()} for row in summary.to_dict(orient="records")]


def loro_cv(features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Leave-one-race-out CV over training races (next-lap MAE per held-out race)."""
    train = training_rows(features, ("train",))
    rows = []
    for race_id in sorted(train["race_id"].unique()):
        fitted = fit_models_on(train[train["race_id"] != race_id], cfg)
        held = features[(features["race_id"] == race_id)]
        for name, model in fitted.models.items():
            p = next_lap_predictions(held, model)
            rows.append(
                {
                    "model": name,
                    "race_id": race_id,
                    "mae": float(np.mean(np.abs(p["pred_s"] - p["actual_s"]))),
                    "n": len(p),
                }
            )
    return pd.DataFrame(rows)


def _error_groups(predictions: pd.DataFrame) -> pd.DataFrame:
    held = predictions[predictions["split"].isin(["val", "test"])].copy()
    held["abs_error_s"] = (held["pred_s"] - held["actual_s"]).abs()
    held["stint_length"] = pd.cut(
        held["stint_length"], [0, 10, 20, 30, 1000], labels=["1-10", "11-20", "21-30", "31+"]
    ).astype(str)
    frames = []
    for group_by in ("compound", "race_id", "stint_length"):
        q = held.groupby(["model", group_by])["abs_error_s"].quantile(
            np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        )
        table = q.unstack().reset_index().rename(columns={group_by: "group"})
        table.columns = ["model", "group", "q10_s", "q25_s", "q50_s", "q75_s", "q90_s"]
        counts = held.groupby(["model", group_by]).size().to_numpy()
        frames.append(table.assign(group_by=group_by.replace("race_id", "race"), n=counts))
    return pd.concat(frames, ignore_index=True)


def actual_remaining_life(features: pd.DataFrame, threshold_s: float) -> pd.DataFrame:
    """Ground-truth remaining competitive laps for every decision lap of held-out stints.

    Counts laps after t until the rolling-3 clean relative pace exceeds the stint's early
    reference (median of stint laps 2-4) by ``threshold_s``; censored at the stint end.
    """
    rows = []
    for (race_id, driver_id, stint_id), stint in features.groupby(
        ["race_id", "driver_id", "stint_id"]
    ):
        stint = stint.sort_values("lap")
        clean = stint[stint["is_clean"]]
        early = clean[clean["stint_lap"].between(2, 4)]["y_s"]
        if len(early) < 2:
            continue
        reference = float(early.median())
        rolling = clean.set_index("lap")["y_s"].rolling(3, min_periods=3).mean() - reference
        crossing = rolling[rolling > threshold_s]
        for lap in stint.loc[(stint["stint_lap"] >= 4) & ~stint["pit_in"], "lap"]:
            later = crossing[crossing.index > lap]
            censored = later.empty
            value = int(stint["lap"].max() - lap) if censored else int(later.index[0] - lap - 1)
            rows.append(
                {
                    "race_id": race_id,
                    "driver_id": driver_id,
                    "stint_id": stint_id,
                    "lap": int(lap),
                    "actual_remaining": value,
                    "censored": censored,
                }
            )
    return pd.DataFrame(rows)


def threshold_sensitivity(
    features: pd.DataFrame, models: dict[ModelName, PaceModel], cfg: Config
) -> pd.DataFrame:
    """Remaining-life MAE (laps) per threshold and model on uncensored val/test decisions."""
    held = features[features["split"].isin(["val", "test"])]
    index = held.set_index(["race_id", "driver_id", "lap"])
    rows = []
    truths = {t: actual_remaining_life(held, t) for t in cfg.threshold_sensitivity_s}
    decisions = truths[cfg.threshold_sensitivity_s[0]][["race_id", "driver_id", "lap"]]
    for name, model in models.items():
        deltas = {}
        for key in map(tuple, decisions.to_numpy()):
            state = _state(index, key)
            deltas[key] = tyre_delta_curve(
                model, state, int(state["total_laps"]), cfg.reference_tyre_age
            )
        for threshold, truth in truths.items():
            uncensored = truth[~truth["censored"]]
            errors = [
                abs(count_competitive(deltas[(r, d, lap)], threshold)[0] - actual)
                for r, d, lap, actual in uncensored[
                    ["race_id", "driver_id", "lap", "actual_remaining"]
                ].itertuples(index=False)
                if (r, d, lap) in deltas
            ]
            rows.append(
                {
                    "threshold_s": threshold,
                    "model": name,
                    "mae_laps": float(np.mean(errors)) if errors else float("nan"),
                    "n_uncensored": len(errors),
                    "n_total": len(truth),
                }
            )
    return pd.DataFrame(rows)


def _state(index: pd.DataFrame, key: tuple[str, str, int]) -> pd.Series:
    row = index.loc[[key]].iloc[0]
    identity = pd.Series({"race_id": key[0], "driver_id": key[1], "lap": key[2]})
    state: pd.Series = pd.concat([row, identity])
    return state


def pit_timing(
    features: pd.DataFrame,
    model: PaceModel,
    sigma: SigmaCurve,
    pit_loss_s: float,
    cfg: Config,
    decision_stint_lap: int = 5,
) -> pd.DataFrame:
    """Descriptive check: is the actual in-lap inside the window computed at stint lap 5?"""
    held = features[features["split"].isin(["val", "test"])]
    rows = []
    for (race_id, driver_id), driver in held.groupby(["race_id", "driver_id"]):
        driver = driver.sort_values("lap")
        for stint_id, stint in driver.groupby("stint_id"):
            last = stint.iloc[-1]
            decision = stint[stint["stint_lap"] == decision_stint_lap]
            if not bool(last["pit_in"]) or decision.empty or decision.iloc[0]["lap"] >= last["lap"]:
                continue
            state = decision.iloc[0]
            compound, feasible = choose_replacement(driver, int(state["lap"]))
            result = evaluate_pit_window(
                model,
                state,
                sigma,
                StrategyInputs(
                    compound,
                    feasible,
                    pit_loss_s,
                    cfg.warmup_effect_s,
                    cfg.window_margin_sigma,
                    cfg.fuel_s_per_lap,
                ),
            )
            window = result.window
            actual_lap = int(last["lap"])
            rows.append(
                {
                    "race_id": race_id,
                    "driver_id": driver_id,
                    "stint_id": int(str(stint_id)),
                    "decision_lap": int(state["lap"]),
                    "actual_pit_lap": actual_lap,
                    "window_start": window.start_lap if window else None,
                    "window_end": window.end_lap if window else None,
                    "optimal_lap": result.optimal_lap,
                    "actual_in_window": bool(
                        window and window.start_lap <= actual_lap <= window.end_lap
                    ),
                }
            )
    return pd.DataFrame(rows)


def _hgb_importances(model: HgbModel, features: pd.DataFrame, seed: int) -> dict[str, float]:
    rows = features[features["is_training_row"] & (features["split"] == "val")]
    inputs = with_categories(model.tyre_clip.clip(rows))[list(model.features)]
    result = permutation_importance(
        model.estimator,
        inputs,
        rows["y_s"].to_numpy(),
        n_repeats=5,
        random_state=seed,
        scoring="neg_mean_absolute_error",
    )
    means = np.asarray(result.importances_mean, dtype=np.float64)
    return {f: float(v) for f, v in zip(model.features, means, strict=True)}


def fuel_sensitivity(
    laps: pd.DataFrame, cfg: Config, model_name: ModelName
) -> list[dict[str, object]]:
    """Refit with alternative fuel corrections and report val/test next-lap MAE."""
    rows: list[dict[str, object]] = []
    for fuel in (*cfg.fuel_sensitivity_s_per_lap, cfg.fuel_s_per_lap):
        variant = cfg.model_copy(update={"fuel_s_per_lap": fuel})
        features = build_features(laps, variant)
        model = fit_models_on(training_rows(features, ("train",)), variant).models[model_name]
        predictions = next_lap_predictions(features, model)
        for split in ("val", "test"):
            held = predictions[predictions["split"] == split]
            rows.append(
                {
                    "fuel_s_per_lap": fuel,
                    "model": model_name,
                    "split": split,
                    "mae": float(np.mean(np.abs(held["pred_s"] - held["actual_s"]))),
                }
            )
    return rows


def evaluate(
    laps: pd.DataFrame,
    features: pd.DataFrame,
    fitted: FittedModels,
    cfg: Config,
    stint_verification: pd.DataFrame,
) -> EvaluationResult:
    """Run every evaluation; model selection uses validation MAE only."""
    hgb = fitted.models["hgb"]
    if isinstance(hgb, HgbModel):
        fitted.models["hgb"] = with_importances(
            hgb, _hgb_importances(hgb, features, cfg.random_seed)
        )

    predictions = pd.concat(
        [next_lap_predictions(features, m) for m in fitted.models.values()], ignore_index=True
    )
    errors = pd.concat(
        [
            rollout_errors(features, m, ("val", "test"), cfg.fuel_s_per_lap)
            for m in fitted.models.values()
        ],
        ignore_index=True,
    )
    sigma = {
        name: fit_sigma_curve(errors[(errors["model"] == name) & (errors["split"] == "val")])
        for name in fitted.models
    }
    val_mae = (
        predictions[predictions["split"] == "val"]
        .assign(e=lambda d: (d["pred_s"] - d["actual_s"]).abs())
        .groupby("model")["e"]
        .mean()
    )
    selected: ModelName = next(n for n in MODEL_NAMES if n == val_mae.idxmin())
    for name in fitted.models:
        z = cfg.next_lap_interval_z * sigma[name].lap_sigma(1)
        mask = predictions["model"] == name
        predictions.loc[mask, "lo_s"] = predictions.loc[mask, "pred_s"] - z
        predictions.loc[mask, "hi_s"] = predictions.loc[mask, "pred_s"] + z

    train_races = sorted(
        features.loc[
            (features["split"] == "train") & (features["tyre_source"] != "none"), "race_id"
        ].unique()
    )
    pit_loss = estimate_pit_loss(laps, train_races)

    ablation = next_lap_predictions(features, fitted.hgb_deg_ablation).assign(
        model="hgb_deg_ablation"
    )
    loro = loro_cv(features, cfg)
    sensitivity = threshold_sensitivity(features, fitted.models, cfg)
    timing = pit_timing(features, fitted.models[selected], sigma[selected], pit_loss.median_s, cfg)

    metric_rows = _metric_rows(
        predictions[predictions["split"] != "unused"], errors, (cfg.horizon_n, 3)
    )
    horizon_rows = _horizon_rows(errors, cfg.horizon_max_k)
    summary: dict[str, object] = {
        "profile": cfg.profile,
        "selected_model": selected,
        "selection_rule": "lowest validation next-lap MAE",
        "metrics": metric_rows,
        "horizon": horizon_rows,
        "cumulative_stint": _cumulative_stint_rows(errors),
        "loro_cv": loro.to_dict(orient="records"),
        "loro_cv_summary": loro.groupby("model")["mae"]
        .agg(["mean", "std"])
        .reset_index()
        .to_dict(orient="records"),
        "ablation_deg_features": [
            {
                "model": "hgb_deg_ablation",
                "split": s,
                "mae": float(np.mean(np.abs(g["pred_s"] - g["actual_s"]))),
                "n": len(g),
            }
            for s, g in ablation.groupby("split")
        ],
        "fuel_sensitivity": fuel_sensitivity(laps, cfg, selected),
        "sigma": {
            n: {
                "per_lap": list(s.per_lap),
                "cumulative": list(s.cumulative),
                "power_a": s.power_a,
                "power_b": s.power_b,
            }
            for n, s in sigma.items()
        },
        "pit_loss": pit_loss.__dict__ | {"races": train_races},
        "stint_verification": {
            "match_rate": float(stint_verification["matched"].mean()),
            "n_stops": len(stint_verification),
            "by_race": stint_verification.groupby("race_id")["matched"].mean().round(4).to_dict(),
        },
        "pit_timing_in_window_rate": float(timing["actual_in_window"].mean())
        if len(timing)
        else None,
    }
    return EvaluationResult(
        selected_model=selected,
        pit_loss=pit_loss,
        sigma=sigma,
        models=fitted,
        predictions=predictions,
        rollout_errors=errors,
        error_groups=_error_groups(predictions),
        threshold_sensitivity=sensitivity,
        pit_timing=timing,
        pit_loss_races=train_races,
        metric_rows=metric_rows,
        horizon_rows=horizon_rows,
        summary=summary,
    )
