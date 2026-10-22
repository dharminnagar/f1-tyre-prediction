"""Precomputed evaluation tables and degradation curves."""

from typing import Annotated

from fastapi import APIRouter, Query

from boxbox.schema import Compound
from boxbox_api.dependencies import ModelQuery, ProfileDep, StoreDep
from boxbox_api.schemas import (
    CurvePoint,
    ErrorGroupRow,
    GroupBy,
    MetricsResponse,
    PitTimingRow,
    PredictionRow,
    ProfileMetrics,
    ThresholdRow,
)
from boxbox_api.store import NotFoundError, ProfileStore, records, resolve_model

router = APIRouter(tags=["evaluation"])

METRIC_KEYS = (
    "profile",
    "selected_model",
    "selection_rule",
    "metrics",
    "horizon",
    "loro_cv",
    "loro_cv_summary",
    "cumulative_stint",
    "ablation_deg_features",
    "fuel_sensitivity",
    "pit_loss",
    "stint_verification",
    "pit_timing_in_window_rate",
)
# Added by later pipeline versions; older artifacts fall back to these defaults.
OPTIONAL_METRIC_DEFAULTS: dict[str, object] = {
    "pit_timing_summary": {},
    "normalisation_sensitivity": [],
    "calibrated_margin_sigma": None,
    "window_target_coverage": None,
    "margin_calibration": [],
    "interval_coverage": [],
    "sequence_cell_comparison": [],
    "pit_timing_by_model": [],
}


def _profile_metrics(store: ProfileStore) -> ProfileMetrics:
    payload = {key: store.metrics[key] for key in METRIC_KEYS}
    payload |= {k: store.metrics.get(k, v) for k, v in OPTIONAL_METRIC_DEFAULTS.items()}
    payload["train_races"] = len(store.config.splits.train)
    return ProfileMetrics.model_validate(payload)


@router.get("/evaluation/metrics")
def metrics(store: StoreDep) -> MetricsResponse:
    """Metrics of every loaded profile side by side (same val/test races)."""
    return MetricsResponse(
        profiles=[_profile_metrics(p) for _, p in sorted(store.profiles.items())]
    )


@router.get("/evaluation/predictions")
def predictions(
    store: ProfileDep, race_id: str, driver_id: str, stint_id: int, model: ModelQuery = None
) -> list[PredictionRow]:
    """One-step-ahead predictions vs actual lap times for one stint."""
    name, _ = resolve_model(store, model)
    frame = store.predictions
    rows = frame[
        (frame["model"] == name)
        & (frame["race_id"] == race_id)
        & (frame["driver_id"] == driver_id)
        & (frame["stint_id"] == stint_id)
    ].sort_values("lap")
    if rows.empty:
        raise NotFoundError(f"no predictions for {driver_id} stint {stint_id} in {race_id}")
    return [
        PredictionRow.model_validate(
            {k: r[k] for k in ("lap", "actual_s", "pred_s", "lo_s", "hi_s", "split")}
        )
        for r in rows.to_dict(orient="records")
    ]


@router.get("/evaluation/errors")
def errors(
    store: ProfileDep, group_by: GroupBy = "compound", model: ModelQuery = None
) -> list[ErrorGroupRow]:
    """Absolute next-lap error quantiles on held-out races by group."""
    name, _ = resolve_model(store, model)
    frame = store.error_groups
    rows = frame[(frame["model"] == name) & (frame["group_by"] == group_by)]
    columns = ("group", "q10_s", "q25_s", "q50_s", "q75_s", "q90_s", "n")
    return [
        ErrorGroupRow.model_validate({k: r[k] for k in columns})
        for r in rows.to_dict(orient="records")
    ]


@router.get("/evaluation/threshold-sensitivity")
def threshold_sensitivity(store: ProfileDep) -> list[ThresholdRow]:
    """Remaining-life MAE/bias (laps) per decision mode, competitive threshold and model."""
    return [ThresholdRow.model_validate(r) for r in records(store.threshold_sensitivity)]


@router.get("/evaluation/pit-timing")
def pit_timing(store: ProfileDep) -> list[PitTimingRow]:
    """Descriptive comparison of windows with the actual pit laps (not proof of optimality)."""
    return [PitTimingRow.model_validate(r) for r in records(store.pit_timing)]


@router.get("/degradation/curves")
def degradation_curves(
    store: ProfileDep,
    compound: Compound | None = None,
    race_id: Annotated[str, Query(description="'all' = training races")] = "all",
) -> list[CurvePoint]:
    """Median/IQR relative pace by tyre age with the selected model's fitted curve."""
    frame = store.curves[store.curves["race_id"] == race_id]
    if frame.empty:
        raise NotFoundError(f"no degradation curves for race '{race_id}'")
    if compound is not None:
        frame = frame[frame["compound"] == compound]
    columns = ("compound", "tyre_age", "median_s", "q25_s", "q75_s", "n", "fitted_s")
    return [
        CurvePoint.model_validate({k: r[k] for k in columns})
        for r in frame.sort_values(["compound", "tyre_age"]).to_dict(orient="records")
    ]
