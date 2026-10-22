"""Held-out evaluation of every model; test races are scored once, after selection on val."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Final, Literal

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from boxbox.config import Config, Normalisation
from boxbox.evaluation.metrics import regression_metrics
from boxbox.features.build import build_features, with_driver_offset
from boxbox.features.tyre_prior import REFUEL_ERA, fit_training_prior, prior_summary
from boxbox.models.forecaster import Forecaster, bind_sequence_model, one_step_pace
from boxbox.models.gbm import HgbModel, with_importances
from boxbox.models.lstm import TrainedLstm
from boxbox.models.quantile import interval_coverage, interval_offsets
from boxbox.models.train import (
    FittedModels,
    fit_models_on,
    forecasters,
    training_rows,
)
from boxbox.models.types import FloatArray, with_categories
from boxbox.schema import LAP_KEYS, MODEL_NAMES, STINT_KEYS, ModelName
from boxbox.strategy.pit_loss import PitLossEstimate, estimate_pit_loss
from boxbox.strategy.pit_window import (
    PitCosts,
    StrategyInputs,
    choose_replacement,
    pit_costs,
    window_from_costs,
)
from boxbox.strategy.remaining_life import (
    NO_NOISE,
    PERSISTENCE_LAPS,
    DeltaNoise,
    first_persistent_exceedance,
    fit_delta_noise,
    noisy_competitive_count,
    tyre_delta_curve,
)
from boxbox.strategy.rollout import lap_times, predict_decisions, stay_out_design
from boxbox.strategy.uncertainty import SigmaCurve, fit_sigma_curve

ROLLOUT_MAX_K = 40
STRATEGY_HORIZON: Final[int] = 10
STRATEGY_SELECTION_TARGET: Final[str] = f"next_{STRATEGY_HORIZON}_mean"
SLOPE_DECISION_STINT_LAP: Final[int] = 5
PitTimingExclusion = Literal["retirement", "neutralised", "short_stint"]
DecisionMode = Literal["stint_lap_5", "all"]
DECISION_MODES: Final[tuple[DecisionMode, ...]] = ("stint_lap_5", "all")
MARGIN_CANDIDATES: Final[tuple[float, ...]] = (0.25, 0.5, 0.75, 1.0)
NEAR_OPTIMUM_LAPS: Final[int] = 5


@dataclass
class EvaluationResult:
    """All evaluation tables; ``summary`` is the JSON-ready metrics document."""

    selected_model: ModelName
    strategy_selected_model: ModelName
    pit_loss: PitLossEstimate
    sigma: dict[ModelName, SigmaCurve]
    strategy_sigma: dict[ModelName, SigmaCurve]
    calibrated_margin_sigma: float
    remaining_life_noise: dict[ModelName, DeltaNoise]
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


def next_lap_predictions(features: pd.DataFrame, model: Forecaster) -> pd.DataFrame:
    """One-step predictions on clean rows (decision lap = row lap - 1)."""
    rows = features[features["is_training_row"]]
    pred_y = one_step_pace(model, rows)
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
    features: pd.DataFrame, model: Forecaster, splits: tuple[str, ...], fuel_s_per_lap: float
) -> pd.DataFrame:
    """Stay-out rollouts to stint end (<= ROLLOUT_MAX_K laps) vs actual clean laps."""
    decisions = decision_rows(features, splits)
    actual = features.set_index(["race_id", "driver_id", "lap"])
    designs: list[tuple[pd.Series, list[pd.DataFrame]]] = []
    metas = []
    for decision_id, (_, state) in enumerate(decisions.iterrows()):
        end = int(min(state["stint_end_lap"], state["lap"] + ROLLOUT_MAX_K))
        design = stay_out_design(state, end)
        designs.append((state, [design]))
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
    meta = pd.concat(metas, ignore_index=True)
    meta["pred_s"] = _stacked_pace(model, designs) + meta["field_ref"] + meta["fuel"]
    meta["k"] = meta["lap"] - meta["decision_lap"]
    keys = pd.MultiIndex.from_frame(meta[["race_id", "driver_id", "lap"]])
    meta["actual_s"] = actual["lap_time_s"].reindex(keys).to_numpy()
    meta["is_clean"] = actual["is_clean"].reindex(keys).fillna(value=False).to_numpy(dtype=bool)
    meta["error_s"] = meta["pred_s"] - meta["actual_s"]
    meta["model"] = model.name
    clean = meta[meta["is_clean"]].drop(columns=["field_ref", "fuel", "is_clean"])
    return clean.assign(paired_error_s=clean["error_s"] - _stint_level_error(clean))


def _stacked_pace(
    model: Forecaster, decisions: list[tuple[pd.Series, list[pd.DataFrame]]]
) -> FloatArray:
    """Predicted ``y_s`` of every segment row of every decision, concatenated in order."""
    predicted = predict_decisions(model, decisions)
    parts = [p.y_s for segments in predicted for p in segments]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


def _stint_level_error(errors: pd.DataFrame) -> pd.Series:
    """Per-stint median one-step error (0 when the stint has none).

    Every strategy branch evaluated for one stint shares this level error (mostly the frozen
    driver offset), so it cancels in cost differences between branches.
    """
    one_step = errors["error_s"].where(errors["k"] == 1)
    level = one_step.groupby([errors["race_id"], errors["driver_id"], errors["stint_id"]])
    return level.transform("median").fillna(0.0)


def within_group_slope(x: pd.Series, y: pd.Series, groups: pd.Series) -> float:
    """Fixed-effects slope of ``y`` on ``x`` (both demeaned within ``groups``)."""
    frame = pd.DataFrame({"x": x.to_numpy(), "y": y.to_numpy(), "g": groups.to_numpy()})
    frame = frame.dropna()
    demeaned = frame[["x", "y"]] - frame.groupby("g")[["x", "y"]].transform("mean")
    denominator = float((demeaned["x"] ** 2).sum())
    if denominator == 0:
        return float("nan")
    return float((demeaned["x"] * demeaned["y"]).sum() / denominator)


def _stint_ids(rows: pd.DataFrame) -> pd.Series:
    return (
        rows["race_id"].astype(str)
        + "|"
        + rows["driver_id"].astype(str)
        + "|"
        + rows["stint_id"].astype(str)
    )


SLOPE_SPLITS: Final[dict[str, tuple[str, ...]]] = {
    "train": ("train",),
    "held_out": ("val", "test"),
}


def rollout_slope_rows(
    features: pd.DataFrame, models: Mapping[ModelName, Forecaster]
) -> list[dict[str, object]]:
    """Wear rate (s/lap of tyre age) implied by each model vs the data, per split group.

    All three are within-stint fixed-effects slopes: ``rollout`` on stay-out rollouts from
    stint lap ``SLOPE_DECISION_STINT_LAP`` to the stint end (driver state frozen, as in the
    strategy), ``one_step`` on next-lap predictions, ``empirical`` on actual clean ``y_s``.
    The refuelling era is left out (no fuel correction there, so its trend is not wear).
    """
    usable = features[features["era"].astype(str) != REFUEL_ERA]
    rows: list[dict[str, object]] = []
    for group, splits in SLOPE_SPLITS.items():
        decisions = decision_rows(usable, splits)
        decisions = decisions[decisions["stint_lap"] == SLOPE_DECISION_STINT_LAP]
        branches = [
            (
                state,
                [
                    stay_out_design(
                        state, int(min(state["stint_end_lap"], state["lap"] + ROLLOUT_MAX_K))
                    ).assign(decision_id=i)
                ],
            )
            for i, (_, state) in enumerate(decisions.iterrows())
        ]
        designs = [b[1][0] for b in branches]
        stacked = pd.concat(designs, ignore_index=True) if designs else pd.DataFrame()
        clean = training_rows(usable, splits)
        stints = _stint_ids(clean)
        empirical = within_group_slope(clean["tyre_age"], clean["y_s"], stints)
        for name, model in models.items():
            rollout = (
                within_group_slope(
                    stacked["tyre_age"],
                    pd.Series(_stacked_pace(model, branches)),
                    stacked["decision_id"],
                )
                if len(stacked)
                else float("nan")
            )
            one_step = pd.Series(one_step_pace(model, clean), index=clean.index)
            rows.append(
                {
                    "model": name,
                    "split": group,
                    "rollout_s_per_lap": rollout,
                    "one_step_s_per_lap": within_group_slope(clean["tyre_age"], one_step, stints),
                    "empirical_s_per_lap": empirical,
                    "n_decisions": len(designs),
                }
            )
    return rows


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
    """Leave-one-race-out CV over training races (next-lap MAE per held-out race).

    The tyre prior (and the tyre-adjusted driver offset built on it) is refitted per fold so
    the held-out race never informs its own features. The LSTM is only included when
    ``cfg.models.lstm_loro`` is set (its held-out race is moved out of the training split).
    """
    train = training_rows(features, ("train",))
    rows = []
    for race_id in sorted(train["race_id"].unique()):
        prior = fit_training_prior(train[train["race_id"] != race_id], cfg.tyre_age_clip_quantile)
        offset_prior = prior if cfg.driver_offset == "tyre_adjusted" else None
        fold = with_driver_offset(features, offset_prior)
        fold_train = training_rows(fold, ("train",))
        sequences = (
            fold.assign(split=fold["split"].where(fold["race_id"] != race_id, "unused"))
            if cfg.models.lstm_loro
            else None
        )
        fitted = fit_models_on(
            fold_train[fold_train["race_id"] != race_id], cfg, prior, sequences_from=sequences
        )
        held = fold[(fold["race_id"] == race_id)]
        for name, model in forecasters(fitted, fold).items():
            p = next_lap_predictions(held, model)
            rows.append(
                {
                    "model": name,
                    "race_id": race_id,
                    "mae": float(np.mean(np.abs(p["pred_s"] - p["actual_s"]))),
                    "n": len(p),
                }
            )
    return pd.DataFrame(rows, columns=["model", "race_id", "mae", "n"])


def loro_summary(loro: pd.DataFrame, cfg: Config) -> list[dict[str, object]]:
    """Mean/std of per-race LORO MAE per model; the LSTM is marked ``not_run`` when skipped."""
    summary = loro.groupby("model")["mae"].agg(["mean", "std"]).reset_index()
    rows: list[dict[str, object]] = [
        {**{str(k): v for k, v in r.items()}, "status": "run"}
        for r in summary.to_dict(orient="records")
    ]
    if not cfg.models.lstm_loro:
        rows.append({"model": "lstm", "mean": None, "std": None, "status": "not_run"})
    return rows


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


def actual_remaining_life(
    features: pd.DataFrame,
    threshold_s: float,
    persistence: int = PERSISTENCE_LAPS,
    decision: DecisionMode = "stint_lap_5",
) -> pd.DataFrame:
    """Ground-truth remaining competitive laps at the decision laps of every stint.

    The tyre's life ends on the first clean lap that starts ``persistence`` consecutive clean
    laps whose relative pace exceeds the stint's early reference (median of clean stint laps
    2-4) by ``threshold_s`` (the same rule :func:`count_competitive` applies to predictions);
    ``remaining`` counts the laps after the decision lap before it. Stints without such a run
    are censored at the stint end. ``decision="stint_lap_5"`` keeps one decision per stint
    (stint lap 5); ``"all"`` uses every lap from stint lap 4 on (autocorrelated).
    """
    rows = []
    for (race_id, driver_id, stint_id), stint in features.groupby(
        ["race_id", "driver_id", "stint_id"]
    ):
        stint = stint.sort_values("lap")
        clean = stint[stint["is_clean"]]
        reference = _early_reference(clean)
        if reference is None:
            continue
        excess = clean["y_s"].to_numpy(dtype=np.float64) - reference
        clean_laps = clean["lap"].to_numpy(dtype=np.int64)
        if decision == "stint_lap_5":
            laps = stint.loc[(stint["stint_lap"] == SLOPE_DECISION_STINT_LAP), "lap"]
        else:
            laps = stint.loc[stint["stint_lap"] >= 4, "lap"]
        laps = laps[~stint.loc[laps.index, "pit_in"].astype(bool)]
        for lap in laps:
            later = clean_laps > lap
            crossing = first_persistent_exceedance(excess[later], threshold_s, persistence)
            censored = crossing is None
            value = (
                int(stint["lap"].max() - lap)
                if crossing is None
                else int(clean_laps[later][crossing] - lap - 1)
            )
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
    return pd.DataFrame(
        rows, columns=["race_id", "driver_id", "stint_id", "lap", "actual_remaining", "censored"]
    )


def _early_reference(clean: pd.DataFrame) -> float | None:
    """Median relative pace of a stint's clean laps 2-4 (``None`` with fewer than two)."""
    early = clean[clean["stint_lap"].between(2, 4)]["y_s"]
    return float(early.median()) if len(early) >= 2 else None


def actual_excess_curves(features: pd.DataFrame, truth: pd.DataFrame) -> list[FloatArray]:
    """Per truth row, the excess over the stint's early reference on laps after the decision.

    The curve runs to the stint's last lap and is NaN on unclean laps; it is the series
    :func:`actual_remaining_life` searches for a crossing.
    """
    stints = {key: s.sort_values("lap") for key, s in features.groupby(STINT_KEYS)}
    curves = []
    for race_id, driver_id, stint_id, lap in truth[[*STINT_KEYS, "lap"]].itertuples(index=False):
        stint = stints[(race_id, driver_id, stint_id)]
        clean = stint["is_clean"].to_numpy(dtype=bool)
        reference = _early_reference(stint[clean])
        later = (stint["lap"] > lap).to_numpy()
        excess = stint["y_s"].to_numpy(dtype=np.float64) - (
            np.nan if reference is None else reference
        )
        curves.append(np.where(clean, excess, np.nan)[later])
    return curves


def remaining_life_scores(predicted: pd.Series, truth: pd.DataFrame) -> dict[str, object]:
    """MAE/bias on uncensored decisions; censored consistency = share predicted >= censor."""
    uncensored = ~truth["censored"].to_numpy(dtype=bool)
    actual = truth["actual_remaining"].to_numpy(dtype=np.float64)
    pred = predicted.to_numpy(dtype=np.float64)
    error = pred[uncensored] - actual[uncensored]
    censored_ok = pred[~uncensored] >= actual[~uncensored]
    return {
        "mae_laps": float(np.mean(np.abs(error))) if len(error) else float("nan"),
        "bias_laps": float(np.mean(error)) if len(error) else float("nan"),
        "n_uncensored": int(uncensored.sum()),
        "n_censored": int((~uncensored).sum()),
        "censored_consistency": float(censored_ok.mean()) if len(censored_ok) else float("nan"),
        "n_total": len(truth),
    }


def _decision_deltas(
    index: pd.DataFrame, truth: pd.DataFrame, model: Forecaster, cfg: Config
) -> list[FloatArray]:
    """Predicted tyre-delta curve to the flag at every decision lap of ``truth``."""
    deltas = []
    for key in map(tuple, truth[LAP_KEYS].to_numpy()):
        state = _state(index, key)
        deltas.append(
            tyre_delta_curve(model, state, int(state["total_laps"]), cfg.reference_tyre_age)
        )
    return deltas


def fit_remaining_life_noise(
    features: pd.DataFrame, models: Mapping[ModelName, Forecaster], cfg: Config
) -> dict[ModelName, DeltaNoise]:
    """Per-model :class:`DeltaNoise` fitted on the validation stints' stint-lap-5 decisions."""
    val = features[features["split"] == "val"]
    index = val.set_index(LAP_KEYS)
    truth = actual_remaining_life(val, cfg.competitive_threshold_s)
    actual = actual_excess_curves(val, truth)
    return {
        name: fit_delta_noise(
            list(zip(_decision_deltas(index, truth, model, cfg), actual, strict=True))
        )
        for name, model in models.items()
    }


def remaining_life_calibration_rows(
    features: pd.DataFrame,
    models: Mapping[ModelName, Forecaster],
    noise: Mapping[ModelName, DeltaNoise],
    cfg: Config,
) -> list[dict[str, object]]:
    """Stint-lap-5 remaining-life scores per model and split, raw curve vs noise-aware."""
    rows: list[dict[str, object]] = []
    for split in ("val", "test"):
        held = features[features["split"] == split]
        index = held.set_index(LAP_KEYS)
        truth = actual_remaining_life(held, cfg.competitive_threshold_s)
        for name, model in models.items():
            deltas = _decision_deltas(index, truth, model, cfg)
            for calibration, applied in (("raw", NO_NOISE), ("noise", noise[name])):
                predicted = pd.Series(
                    [
                        noisy_competitive_count(d, cfg.competitive_threshold_s, applied)[0]
                        for d in deltas
                    ],
                    dtype=np.float64,
                )
                rows.append(
                    {
                        "model": name,
                        "split": split,
                        "calibration": calibration,
                        **remaining_life_scores(predicted, truth),
                    }
                )
    return rows


def threshold_sensitivity(
    features: pd.DataFrame,
    models: Mapping[ModelName, Forecaster],
    noise: Mapping[ModelName, DeltaNoise],
    cfg: Config,
) -> pd.DataFrame:
    """Remaining-life accuracy (laps) per decision mode, threshold and model on val/test.

    Predictions keep the model's ``reference_tyre_age`` reference while the truth uses the
    stint's own early laps; each model's validation-fitted ``noise`` turns its curve into the
    crossing the truth would measure (val is pooled with test here, see
    :func:`remaining_life_calibration_rows` for test alone).
    """
    held = features[features["split"].isin(["val", "test"])]
    index = held.set_index(["race_id", "driver_id", "lap"])
    rows: list[dict[str, object]] = []
    for decision in DECISION_MODES:
        truths = {
            t: actual_remaining_life(held, t, decision=decision)
            for t in cfg.threshold_sensitivity_s
        }
        keys = sorted(
            {k for truth in truths.values() for k in map(tuple, truth[LAP_KEYS].to_numpy())}
        )
        for name, model in models.items():
            deltas = {}
            for key in keys:
                state = _state(index, key)
                deltas[key] = tyre_delta_curve(
                    model, state, int(state["total_laps"]), cfg.reference_tyre_age
                )
            for threshold, truth in truths.items():
                predicted = pd.Series(
                    [
                        noisy_competitive_count(deltas[k], threshold, noise[name])[0]
                        for k in map(tuple, truth[LAP_KEYS].to_numpy())
                    ],
                    dtype=np.float64,
                )
                rows.append(
                    {
                        "decision": decision,
                        "threshold_s": threshold,
                        "model": name,
                        **remaining_life_scores(predicted, truth),
                    }
                )
    return pd.DataFrame(rows)


def _state(index: pd.DataFrame, key: tuple[str, str, int]) -> pd.Series:
    row = index.loc[[key]].iloc[0]
    identity = pd.Series({"race_id": key[0], "driver_id": key[1], "lap": key[2]})
    state: pd.Series = pd.concat([row, identity])
    return state


def _timing_exclusion(
    stint: pd.DataFrame,
    driver: pd.DataFrame,
    neutral_laps: set[int],
    decision_stint_lap: int,
) -> PitTimingExclusion | None:
    """Why a stop is not a fair test of a tyre-wear window (first match), or ``None``."""
    in_lap = int(stint["lap"].max())
    if in_lap == int(driver["lap"].max()) and in_lap < int(stint["total_laps"].iloc[0]):
        return "retirement"
    if in_lap in neutral_laps or in_lap + 1 in neutral_laps:
        return "neutralised"
    if int(stint["stint_lap"].max()) < decision_stint_lap + 2:
        return "short_stint"
    return None


@dataclass(frozen=True)
class TimedStop:
    """A real stop and the plan costs computed at its decision lap (no margin applied yet)."""

    race_id: str
    driver_id: str
    split: str
    stint_id: int
    decision_lap: int
    actual_pit_lap: int
    excluded_reason: PitTimingExclusion | None
    costs: PitCosts


def timed_stops(
    features: pd.DataFrame,
    model: Forecaster,
    pit_loss_s: float,
    cfg: Config,
    decision_stint_lap: int = SLOPE_DECISION_STINT_LAP,
) -> list[TimedStop]:
    """Every held-out stop with a decision at ``decision_stint_lap`` and its pit-cost curve.

    Every stint's compound is optimised (the actual next compound is not used, so the window
    does not see the team's later choice).
    """
    held = features[features["split"].isin(["val", "test"])]
    neutral = held[held["track_neutralised"].astype(bool)].groupby("race_id")["lap"].unique()
    stops = []
    for (race_id, driver_id), driver in held.groupby(["race_id", "driver_id"]):
        driver = driver.sort_values("lap")
        neutral_laps = {int(v) for v in neutral.get(race_id, [])}
        for stint_id, stint in driver.groupby("stint_id"):
            last = stint.iloc[-1]
            decision = stint[stint["stint_lap"] == decision_stint_lap]
            if not bool(last["pit_in"]) or decision.empty or decision.iloc[0]["lap"] >= last["lap"]:
                continue
            state = decision.iloc[0]
            compound, feasible = choose_replacement(driver, int(state["lap"]))
            inputs = StrategyInputs(
                compound,
                feasible,
                pit_loss_s,
                cfg.warmup_effect_s,
                cfg.window_margin_sigma,
                cfg.fuel_s_per_lap,
            )
            stops.append(
                TimedStop(
                    race_id=str(race_id),
                    driver_id=str(driver_id),
                    split=str(state["split"]),
                    stint_id=int(str(stint_id)),
                    decision_lap=int(state["lap"]),
                    actual_pit_lap=int(last["lap"]),
                    excluded_reason=_timing_exclusion(
                        stint, driver, neutral_laps, decision_stint_lap
                    ),
                    costs=pit_costs(model, state, inputs),
                )
            )
    return stops


def pit_timing_table(
    stops: list[TimedStop], sigma: SigmaCurve, margin_sigma: float
) -> pd.DataFrame:
    """Window at ``margin_sigma`` per stop vs the actual in-lap (one row per stop)."""
    rows = []
    for stop in stops:
        decision = window_from_costs(stop.costs, sigma, margin_sigma)
        window = decision.window
        optimum = decision.optimal_lap
        near = [
            c
            for c in decision.candidates
            if optimum is not None and abs(c.pit_lap - optimum) <= NEAR_OPTIMUM_LAPS
        ]
        best = next((c for c in decision.candidates if c.pit_lap == optimum), None)
        rows.append(
            {
                "race_id": stop.race_id,
                "driver_id": stop.driver_id,
                "split": stop.split,
                "stint_id": stop.stint_id,
                "decision_lap": stop.decision_lap,
                "actual_pit_lap": stop.actual_pit_lap,
                "window_start": window.start_lap if window else None,
                "window_end": window.end_lap if window else None,
                "optimal_lap": optimum,
                "actual_in_window": bool(
                    window and window.start_lap <= stop.actual_pit_lap <= window.end_lap
                ),
                "excluded_reason": stop.excluded_reason,
                "planned_stops": len(best.plan) if best else None,
                "optimum_laps_past_cap": best.laps_past_cap if best else None,
                "near_optimum_past_cap_share": (
                    float(np.mean([c.laps_past_cap > 0 for c in near])) if near else None
                ),
            }
        )
    return pd.DataFrame(rows)


def pit_timing(
    features: pd.DataFrame,
    model: Forecaster,
    sigma: SigmaCurve,
    pit_loss_s: float,
    cfg: Config,
    decision_stint_lap: int = SLOPE_DECISION_STINT_LAP,
    margin_sigma: float | None = None,
) -> pd.DataFrame:
    """Descriptive check: is the actual in-lap inside the window computed at stint lap 5?

    Every stop is kept; ``excluded_reason`` marks stops that say little about tyre wear
    (retirement in the pits, SC/VSC/red on the in-lap or the lap after, stint ending within
    one lap of the decision). ``margin_sigma`` defaults to ``cfg.window_margin_sigma``.
    """
    stops = timed_stops(features, model, pit_loss_s, cfg, decision_stint_lap)
    margin = cfg.window_margin_sigma if margin_sigma is None else margin_sigma
    return pit_timing_table(stops, sigma, margin)


def calibrate_margin(
    stops: list[TimedStop],
    sigma: SigmaCurve,
    target_coverage: float,
    candidates: tuple[float, ...] = MARGIN_CANDIDATES,
) -> tuple[float, list[dict[str, object]]]:
    """Smallest margin multiplier whose validation in-window rate reaches ``target_coverage``.

    Only validation stops without an ``excluded_reason`` count. When no candidate reaches the
    target the largest is returned. The second item lists each candidate's rate and median
    window width.
    """
    kept = [s for s in stops if s.split == "val" and s.excluded_reason is None]
    rows: list[dict[str, object]] = []
    for multiplier in sorted(candidates):
        table = pit_timing_table(kept, sigma, multiplier)
        rows.append(
            {
                "margin_sigma": multiplier,
                "in_window_rate": _rate(table),
                "median_width_laps": _median_width(table),
                "n": len(table),
            }
        )
    reached = [
        float(str(r["margin_sigma"]))
        for r in rows
        if r["in_window_rate"] is not None and float(str(r["in_window_rate"])) >= target_coverage
    ]
    return (min(reached) if reached else max(candidates)), rows


def _rate(rows: pd.DataFrame) -> float | None:
    return float(rows["actual_in_window"].mean()) if len(rows) else None


def _median_width(rows: pd.DataFrame) -> float | None:
    if rows.empty:
        return None
    width = (rows["window_end"] - rows["window_start"] + 1).dropna()
    return float(width.median()) if len(width) else None


def _mean_or_none(values: pd.Series) -> float | None:
    clean = values.dropna()
    return float(clean.mean()) if len(clean) else None


def _median_or_none(values: pd.Series) -> float | None:
    clean = values.dropna()
    return float(clean.median()) if len(clean) else None


def pit_timing_summary(timing: pd.DataFrame) -> dict[str, object]:
    """In-window rates per split: all stops, stops not excluded, and by stint index.

    Also, for kept stops: median window width (laps), share without a window, median
    optimum-minus-actual in-lap per stint index, and the mean share of candidates within
    ``NEAR_OPTIMUM_LAPS`` of the optimum (and of optima) whose plan runs a tyre past its
    training age cap.
    """
    if timing.empty:
        return {}
    summary: dict[str, object] = {}
    for split, rows in timing.groupby("split"):
        kept = rows[rows["excluded_reason"].isna()]
        summary[str(split)] = {
            "n": len(rows),
            "in_window_rate": _rate(rows),
            "n_kept": len(kept),
            "in_window_rate_kept": _rate(kept),
            "excluded": rows["excluded_reason"].value_counts().to_dict(),
            "in_window_rate_kept_by_stint": {
                str(stint): _rate(group) for stint, group in kept.groupby("stint_id")
            },
            "median_width_kept": _median_width(kept),
            "no_window_share_kept": float(kept["window_start"].isna().mean())
            if len(kept)
            else None,
            "optimum_bias_kept_by_stint": {
                str(stint): _median_or_none(group["optimal_lap"] - group["actual_pit_lap"])
                for stint, group in kept.groupby("stint_id")
            },
            "near_optimum_past_cap_share_kept": _mean_or_none(kept["near_optimum_past_cap_share"])
            if "near_optimum_past_cap_share" in kept
            else None,
            "optimum_past_cap_share_kept": _mean_or_none(kept["optimum_laps_past_cap"] > 0)
            if "optimum_laps_past_cap" in kept
            else None,
        }
    return summary


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


def _held_out_mae(predictions: pd.DataFrame, model: str, split: str) -> float:
    held = predictions[(predictions["model"] == model) & (predictions["split"] == split)]
    return float(np.mean(np.abs(held["pred_s"] - held["actual_s"])))


def fuel_sensitivity(laps: pd.DataFrame, cfg: Config) -> list[dict[str, object]]:
    """Refit every model (LSTM included) with alternative fuel corrections; val/test MAE."""
    rows: list[dict[str, object]] = []
    for fuel in (*cfg.fuel_sensitivity_s_per_lap, cfg.fuel_s_per_lap):
        variant = cfg.model_copy(update={"fuel_s_per_lap": fuel})
        features = build_features(laps, variant)
        fitted = fit_models_on(
            training_rows(features, ("train",)), variant, sequences_from=features
        )
        for name, model in forecasters(fitted, features).items():
            predictions = next_lap_predictions(features, model)
            rows.extend(
                {
                    "fuel_s_per_lap": fuel,
                    "model": name,
                    "split": split,
                    "mae": _held_out_mae(predictions, name, split),
                }
                for split in ("val", "test")
            )
    return rows


NORMALISATIONS: Final[tuple[Normalisation, ...]] = ("field", "teammate")


def normalisation_sensitivity(
    laps: pd.DataFrame,
    cfg: Config,
    features: pd.DataFrame,
    models: Mapping[ModelName, Forecaster],
) -> list[dict[str, object]]:
    """Field vs teammate reference pace: val/test next-lap MAE and held-out rollout slope.

    The configured normalisation reuses the main ``features``/``models``; the other is rebuilt
    (every model refitted, LSTM included).
    """
    rows: list[dict[str, object]] = []
    for normalisation in NORMALISATIONS:
        variant_models = models
        if normalisation == cfg.normalisation:
            variant_features = features
        else:
            variant = cfg.model_copy(update={"normalisation": normalisation})
            variant_features = build_features(laps, variant)
            refitted = fit_models_on(
                training_rows(variant_features, ("train",)),
                variant,
                sequences_from=variant_features,
            )
            variant_models = forecasters(refitted, variant_features)
        slopes = {
            str(r["model"]): r["rollout_s_per_lap"]
            for r in rollout_slope_rows(variant_features, variant_models)
            if r["split"] == "held_out"
        }
        for name, model in variant_models.items():
            predictions = next_lap_predictions(variant_features, model)
            rows.append(
                {
                    "normalisation": normalisation,
                    "model": name,
                    "val_mae": _held_out_mae(predictions, name, "val"),
                    "test_mae": _held_out_mae(predictions, name, "test"),
                    "held_out_rollout_s_per_lap": slopes.get(name),
                }
            )
    return rows


def _sigma_summary(curves: dict[ModelName, SigmaCurve]) -> dict[str, object]:
    return {
        name: {
            "per_lap": list(curve.per_lap),
            "cumulative": list(curve.cumulative),
            "power_a": curve.power_a,
            "power_b": curve.power_b,
        }
        for name, curve in curves.items()
    }


def lowest_val_mae(
    rows: list[dict[str, object]], target: str, fallback: ModelName | None = None
) -> ModelName:
    """Model with the lowest validation MAE on ``target`` (ties: ``MODEL_NAMES`` order).

    ``fallback`` is returned when no model has a finite validation MAE for ``target``.
    """
    maes = {
        str(r["model"]): float(str(r["mae"]))
        for r in rows
        if r["split"] == "val" and r["target"] == target
    }
    candidates = [n for n in MODEL_NAMES if n in maes and np.isfinite(maes[n])]
    if not candidates:
        if fallback is None:
            raise ValueError(f"no validation MAE for {target}")
        return fallback
    return min(candidates, key=lambda n: maes[n])


IntervalMethod = Literal["quantile", "sigma"]


def interval_method(name: str, fitted: FittedModels) -> IntervalMethod:
    """Quantile-HGB intervals for ``hgb`` when its companions were fitted, else ±z·sigma."""
    return "quantile" if name == "hgb" and fitted.hgb_quantile is not None else "sigma"


def with_next_lap_intervals(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    fitted: FittedModels,
    sigma: Mapping[str, SigmaCurve],
    z: float,
) -> pd.DataFrame:
    """Add ``lo_s``/``hi_s``/``interval_method`` to next-lap predictions.

    ``hgb`` uses its quantile companions centred on its prediction (see
    :func:`~boxbox.models.quantile.interval_offsets`); every other model ±z x the one-step
    validation sigma. Rows are in :func:`next_lap_predictions` order for every model.
    """
    out = predictions.copy()
    rows = features[features["is_training_row"]]
    for name in out["model"].unique():
        mask = (out["model"] == name).to_numpy()
        pred = out.loc[mask, "pred_s"].to_numpy(dtype=np.float64)
        method = interval_method(str(name), fitted)
        if method == "quantile" and fitted.hgb_quantile is not None:
            lo, hi = interval_offsets(fitted.models["hgb"], fitted.hgb_quantile, rows)
        else:
            half = z * sigma[str(name)].lap_sigma(1)
            lo, hi = np.full(len(pred), -half), np.full(len(pred), half)
        out.loc[mask, "lo_s"] = pred + lo
        out.loc[mask, "hi_s"] = pred + hi
        out.loc[mask, "interval_method"] = method
    return out


def rollout_interval_offsets(
    features: pd.DataFrame, fitted: FittedModels, splits: tuple[str, ...], max_k: int
) -> pd.DataFrame:
    """HGB quantile offsets on the stay-out rollout rows k <= ``max_k`` (keyed by lap)."""
    if fitted.hgb_quantile is None:
        return pd.DataFrame(columns=[*LAP_KEYS, "decision_lap", "lo_off", "hi_off"])
    decisions = decision_rows(features, splits)
    frames = []
    for _, state in decisions.iterrows():
        end = int(min(state["stint_end_lap"], state["lap"] + max_k))
        frames.append(
            stay_out_design(state, end).assign(
                race_id=state["race_id"], driver_id=state["driver_id"], decision_lap=state["lap"]
            )
        )
    if not frames:
        return pd.DataFrame(columns=[*LAP_KEYS, "decision_lap", "lo_off", "hi_off"])
    stacked = pd.concat(frames, ignore_index=True)
    lo, hi = interval_offsets(fitted.models["hgb"], fitted.hgb_quantile, stacked)
    return pd.DataFrame(
        {
            "race_id": stacked["race_id"],
            "driver_id": stacked["driver_id"],
            "decision_lap": stacked["decision_lap"].astype(int),
            "lap": stacked["lap_number"].astype(int),
            "lo_off": lo,
            "hi_off": hi,
        }
    )


def _coverage_row(
    model: str,
    split: str,
    target: str,
    method: str,
    lo: FloatArray,
    hi: FloatArray,
    actual: FloatArray,
) -> dict[str, object]:
    return {
        "model": model,
        "split": split,
        "target": target,
        "method": method,
        "coverage": interval_coverage(lo, hi, actual),
        "mean_width_s": float(np.mean(hi - lo)) if len(lo) else None,
        "n": len(lo),
    }


def interval_coverage_rows(
    predictions: pd.DataFrame,
    errors: pd.DataFrame,
    offsets: pd.DataFrame,
    fitted: FittedModels,
    sigma: Mapping[str, SigmaCurve],
    cfg: Config,
) -> list[dict[str, object]]:
    """Empirical coverage and mean width of the served 80% intervals per model and split.

    Targets: ``next_lap`` (one-step), ``next_{n}_laps`` (each of the first n rollout laps, as
    ``/state`` serves them) and ``next_{n}_mean`` (mean of those laps, decisions with the same
    minimum lap count as the MAE table). Sigma intervals of a mean use cumulative_sigma(m)/m;
    quantile intervals of a mean average the per-lap offsets (exact only for a fully shared
    error, wider than needed for independent ones). Sigma is fitted on validation rollouts, so
    its validation coverage is in-sample; test is the fair comparison.
    """
    n = cfg.horizon_n
    z = cfg.next_lap_interval_z
    min_laps = max(2, (n + 1) // 2 + 1)
    rows: list[dict[str, object]] = []
    window = errors[errors["k"] <= n].merge(offsets, on=[*LAP_KEYS, "decision_lap"], how="left")
    for (name, split), group in predictions[predictions["split"].isin(["val", "test"])].groupby(
        ["model", "split"]
    ):
        model, part = str(name), str(split)
        method = interval_method(model, fitted)
        rows.append(
            _coverage_row(
                model,
                part,
                "next_lap",
                method,
                group["lo_s"].to_numpy(dtype=np.float64),
                group["hi_s"].to_numpy(dtype=np.float64),
                group["actual_s"].to_numpy(dtype=np.float64),
            )
        )
        laps = window[(window["model"] == model) & (window["split"] == part)]
        pred = laps["pred_s"].to_numpy(dtype=np.float64)
        if method == "quantile":
            lo_off = laps["lo_off"].to_numpy(dtype=np.float64)
            hi_off = laps["hi_off"].to_numpy(dtype=np.float64)
        else:
            curve = sigma[model]
            half = z * np.array([curve.lap_sigma(int(k)) for k in laps["k"]], dtype=np.float64)
            lo_off, hi_off = -half, half
        actual = laps["actual_s"].to_numpy(dtype=np.float64)
        rows.append(
            _coverage_row(
                model, part, f"next_{n}_laps", method, pred + lo_off, pred + hi_off, actual
            )
        )
        frame = pd.DataFrame(
            {
                "decision_id": laps["decision_id"].to_numpy(),
                "pred": pred,
                "actual": actual,
                "lo": lo_off,
                "hi": hi_off,
            }
        )
        means = frame.groupby("decision_id").agg(
            pred=("pred", "mean"),
            actual=("actual", "mean"),
            lo=("lo", "mean"),
            hi=("hi", "mean"),
            m=("pred", "size"),
        )
        means = means[means["m"] >= min_laps]
        centre = means["pred"].to_numpy(dtype=np.float64)
        if method == "quantile":
            lo_mean = means["lo"].to_numpy(dtype=np.float64)
            hi_mean = means["hi"].to_numpy(dtype=np.float64)
        else:
            half_mean = z * np.array(
                [sigma[model].cumulative_sigma(int(m)) / m for m in means["m"]], dtype=np.float64
            )
            lo_mean, hi_mean = -half_mean, half_mean
        rows.append(
            _coverage_row(
                model,
                part,
                f"next_{n}_mean",
                method,
                centre + lo_mean,
                centre + hi_mean,
                means["actual"].to_numpy(dtype=np.float64),
            )
        )
    return rows


def sequence_training_summary(model: TrainedLstm) -> dict[str, object]:
    """Training record of a sequence model (curve, early stopping, size, time)."""
    return {
        "cell": model.params.cell,
        "params": model.params.model_dump(mode="json"),
        "best_epoch": model.best_epoch,
        "epochs_run": len(model.curve),
        "train_seconds": model.train_seconds,
        "n_train_samples": model.n_train_samples,
        "inner_val_seasons": list(model.inner_val_seasons),
        "curve": [
            {"epoch": epoch, "train_loss": train, "inner_val_loss": inner}
            for epoch, train, inner in model.curve
        ],
    }


def sequence_cell_comparison(
    features: pd.DataFrame,
    fitted: FittedModels,
    cfg: Config,
    lstm_predictions: pd.DataFrame,
    lstm_errors: pd.DataFrame,
) -> list[dict[str, object]]:
    """LSTM vs GRU cell (same hyper-parameters): val/test accuracy, wear slope, train time.

    The GRU is evaluation-only (not selectable); the LSTM rows reuse its main tables.
    """
    rows: list[dict[str, object]] = []
    for cell, trained in (("lstm", fitted.lstm), ("gru", fitted.gru)):
        if trained is None:
            continue
        bound = bind_sequence_model(trained, features, fitted.prior)
        if cell == "lstm":
            predictions, errors = lstm_predictions, lstm_errors
        else:
            predictions = next_lap_predictions(features, bound)
            errors = rollout_errors(features, bound, ("val", "test"), cfg.fuel_s_per_lap)
        metrics = _metric_rows(
            predictions[predictions["split"].isin(["val", "test"])], errors, (cfg.horizon_n,)
        )
        slopes = {
            str(r["split"]): r["rollout_s_per_lap"]
            for r in rollout_slope_rows(features, {"lstm": bound})
        }
        for split in ("val", "test"):
            by_target = {str(r["target"]): r["mae"] for r in metrics if r["split"] == split}
            at_k = errors[(errors["split"] == split) & (errors["k"] == cfg.horizon_max_k)]
            rows.append(
                {
                    "cell": cell,
                    "split": split,
                    "next_lap_mae": by_target.get("next_lap"),
                    "next_n_mean_mae": by_target.get(f"next_{cfg.horizon_n}_mean"),
                    "n": cfg.horizon_n,
                    "horizon_k": cfg.horizon_max_k,
                    "horizon_k_mae": float(at_k["error_s"].abs().mean()) if len(at_k) else None,
                    "rollout_s_per_lap_held_out": slopes.get("held_out"),
                    "rollout_s_per_lap_train": slopes.get("train"),
                    "train_seconds": trained.train_seconds,
                    "best_epoch": trained.best_epoch,
                }
            )
    return rows


def pit_timing_by_model(
    features: pd.DataFrame,
    models: Mapping[ModelName, Forecaster],
    strategy_sigma: Mapping[ModelName, SigmaCurve],
    pit_loss_s: float,
    cfg: Config,
) -> list[dict[str, object]]:
    """Pit-timing check for every model, each with its own validation-calibrated margin."""
    rows: list[dict[str, object]] = []
    for name, model in models.items():
        stops = timed_stops(features, model, pit_loss_s, cfg)
        margin, _ = calibrate_margin(stops, strategy_sigma[name], cfg.window_target_coverage)
        table = pit_timing_table(stops, strategy_sigma[name], margin)
        rows.append(
            {
                "model": name,
                "calibrated_margin_sigma": margin,
                "summary": pit_timing_summary(table),
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
    models = forecasters(fitted, features)

    predictions = pd.concat(
        [next_lap_predictions(features, m) for m in models.values()], ignore_index=True
    )
    errors = pd.concat(
        [rollout_errors(features, m, ("val", "test"), cfg.fuel_s_per_lap) for m in models.values()],
        ignore_index=True,
    )
    val_errors = errors[errors["split"] == "val"]
    sigma = {name: fit_sigma_curve(val_errors[val_errors["model"] == name]) for name in models}
    strategy_sigma = {
        name: fit_sigma_curve(val_errors[val_errors["model"] == name], error_col="paired_error_s")
        for name in models
    }
    metric_rows = _metric_rows(
        predictions[predictions["split"] != "unused"],
        errors,
        (cfg.horizon_n, 3, STRATEGY_HORIZON),
    )
    selected = lowest_val_mae(metric_rows, "next_lap")
    strategy_model = lowest_val_mae(metric_rows, STRATEGY_SELECTION_TARGET, fallback=selected)
    sigma_by_name = {str(name): curve for name, curve in sigma.items()}
    predictions = with_next_lap_intervals(
        predictions, features, fitted, sigma_by_name, cfg.next_lap_interval_z
    )
    coverage = interval_coverage_rows(
        predictions,
        errors,
        rollout_interval_offsets(features, fitted, ("val", "test"), cfg.horizon_n),
        fitted,
        sigma_by_name,
        cfg,
    )

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
    life_noise = fit_remaining_life_noise(features, models, cfg)
    sensitivity = threshold_sensitivity(features, models, life_noise, cfg)
    stops = timed_stops(features, models[strategy_model], pit_loss.median_s, cfg)
    margin_sigma, margin_rows = calibrate_margin(
        stops, strategy_sigma[strategy_model], cfg.window_target_coverage
    )
    timing = pit_timing_table(stops, strategy_sigma[strategy_model], margin_sigma)
    horizon_rows = _horizon_rows(errors, cfg.horizon_max_k)
    summary: dict[str, object] = {
        "profile": cfg.profile,
        "selected_model": selected,
        "selection_rule": "lowest validation next-lap MAE",
        "strategy_selected_model": strategy_model,
        "strategy_selection_rule": f"lowest validation {STRATEGY_SELECTION_TARGET} MAE",
        "metrics": metric_rows,
        "horizon": horizon_rows,
        "cumulative_stint": _cumulative_stint_rows(errors),
        "loro_cv": loro.to_dict(orient="records"),
        "loro_cv_summary": loro_summary(loro, cfg),
        "ablation_deg_features": [
            {
                "model": "hgb_deg_ablation",
                "split": s,
                "mae": float(np.mean(np.abs(g["pred_s"] - g["actual_s"]))),
                "n": len(g),
            }
            for s, g in ablation.groupby("split")
        ],
        "fuel_sensitivity": fuel_sensitivity(laps, cfg),
        "normalisation_sensitivity": normalisation_sensitivity(laps, cfg, features, models),
        "tyre_prior": prior_summary(fitted.prior),
        "rollout_slope": rollout_slope_rows(features, models),
        "interval_coverage": coverage,
        "sequence_training": {
            cell: sequence_training_summary(trained)
            for cell, trained in (("lstm", fitted.lstm), ("gru", fitted.gru))
            if trained is not None
        },
        "sequence_cell_comparison": sequence_cell_comparison(
            features,
            fitted,
            cfg,
            predictions[predictions["model"] == "lstm"],
            errors[errors["model"] == "lstm"],
        ),
        "pit_timing_by_model": pit_timing_by_model(
            features, models, strategy_sigma, pit_loss.median_s, cfg
        ),
        "sigma": _sigma_summary(sigma),
        "sigma_paired": _sigma_summary(strategy_sigma),
        "calibrated_margin_sigma": margin_sigma,
        "margin_calibration": margin_rows,
        "remaining_life_noise": {name: asdict(n) for name, n in life_noise.items()},
        "remaining_life_calibration": remaining_life_calibration_rows(
            features, models, life_noise, cfg
        ),
        "window_target_coverage": cfg.window_target_coverage,
        "pit_loss": pit_loss.__dict__ | {"races": train_races},
        "stint_verification": {
            "match_rate": float(stint_verification["matched"].mean()),
            "n_stops": len(stint_verification),
            "by_race": stint_verification.groupby("race_id")["matched"].mean().round(4).to_dict(),
        },
        "pit_timing_in_window_rate": float(timing["actual_in_window"].mean())
        if len(timing)
        else None,
        "pit_timing_summary": pit_timing_summary(timing),
    }
    return EvaluationResult(
        selected_model=selected,
        strategy_selected_model=strategy_model,
        pit_loss=pit_loss,
        sigma=sigma,
        strategy_sigma=strategy_sigma,
        calibrated_margin_sigma=margin_sigma,
        remaining_life_noise=life_noise,
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
