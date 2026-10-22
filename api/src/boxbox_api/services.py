"""Compose ``boxbox.strategy`` calls into API responses (no strategy maths lives here)."""

import math

import numpy as np
import pandas as pd

from boxbox.schema import DRY_COMPOUNDS, Compound, ModelName
from boxbox.strategy.explain import decompose_next_lap, explanation_lines
from boxbox.strategy.pit_window import (
    PitWindowResult,
    StrategyInputs,
    choose_replacement,
    counterfactual,
    evaluate_pit_window,
)
from boxbox.strategy.remaining_life import remaining_competitive_laps, tyre_delta_curve
from boxbox.strategy.rollout import stay_out_rollout
from boxbox_api.schemas import (
    Assumptions,
    CounterfactualRequest,
    CounterfactualResponse,
    DecompositionOut,
    Degradation,
    Interval,
    LapWindow,
    PitCandidateOut,
    PitWindowRequest,
    PitWindowResponse,
    RemainingLifeOut,
    TyreState,
)
from boxbox_api.store import ProfileStore, driver_laps, lap_state, resolve_model

NEXT_LAPS = 5


class OutOfRangeError(ValueError):
    """A lap or option is outside what can be simulated (mapped to HTTP 422)."""


def optional_float(value: object) -> float | None:
    """Finite float or None (for NaN/NA/None)."""
    if isinstance(value, int | float | np.number):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def dry_compound(value: object) -> Compound:
    """Narrow a compound label to a dry ``Compound`` (422 otherwise)."""
    for compound in DRY_COMPOUNDS:
        if compound == value:
            return compound
    raise OutOfRangeError(f"compound {value!r} is not a dry compound; only dry stints are served")


def _decision(store: ProfileStore, race_id: str, driver_id: str, lap: int) -> pd.Series:
    state = lap_state(store, race_id, driver_id, lap)
    if lap >= int(state["total_laps"]):
        raise OutOfRangeError(f"lap {lap} is the final lap; nothing left to simulate")
    dry_compound(state["compound"])
    return state


def _inputs(
    store: ProfileStore,
    state: pd.Series,
    replacement: Compound | None,
    pit_loss_s: float | None,
    warmup_s: float | None,
) -> StrategyInputs:
    laps = driver_laps(store, str(state["race_id"]), str(state["driver_id"]))
    chosen, feasible = choose_replacement(laps, int(state["lap"]))
    cfg = store.config
    return StrategyInputs(
        replacement_compound=replacement or chosen,
        stay_out_feasible=feasible,
        pit_loss_s=pit_loss_s if pit_loss_s is not None else store.pit_loss.median_s,
        warmup_s=warmup_s if warmup_s is not None else cfg.warmup_effect_s,
        margin_sigma=cfg.window_margin_sigma,
        fuel_s_per_lap=cfg.fuel_s_per_lap,
    )


def tyre_state(
    store: ProfileStore, race_id: str, driver_id: str, lap: int, model_name: ModelName | None
) -> TyreState:
    """Current tyre state, short-term forecast, remaining life and explanation."""
    name, model = resolve_model(store, model_name)
    cfg = store.config
    state = _decision(store, race_id, driver_id, lap)
    total_laps = int(state["total_laps"])
    sigma = store.sigma[name]
    rollout = stay_out_rollout(model, state, min(total_laps, lap + NEXT_LAPS), cfg.fuel_s_per_lap)
    z = cfg.next_lap_interval_z
    intervals = [
        Interval(
            lap=int(lap_number),
            pred_s=float(pred),
            lo_s=float(pred - z * sigma.lap_sigma(k)),
            hi_s=float(pred + z * sigma.lap_sigma(k)),
            extrapolated=bool(extrapolated),
        )
        for k, (lap_number, pred, extrapolated) in enumerate(
            zip(rollout.laps, rollout.lap_time_s, rollout.extrapolated, strict=True), start=1
        )
    ]
    life = remaining_competitive_laps(
        model, state, cfg.competitive_threshold_s, cfg.reference_tyre_age, cfg.fuel_s_per_lap
    )
    delta = tyre_delta_curve(model, state, min(total_laps, lap + NEXT_LAPS), cfg.reference_tyre_age)
    model_slope = float(np.mean(np.diff(delta))) if len(delta) >= 2 else 0.0
    parts = decompose_next_lap(model, state, cfg.reference_tyre_age, cfg.fuel_s_per_lap)
    window = evaluate_pit_window(model, state, sigma, _inputs(store, state, None, None, None))
    return TyreState(
        race_id=race_id,
        driver_id=driver_id,
        lap=lap,
        model=name,
        compound=dry_compound(state["compound"]),
        tyre_age=float(state["tyre_age"]),
        stint_id=int(state["stint_id"]),
        stint_lap=int(state["stint_lap"]),
        degradation=Degradation(
            empirical_s_per_lap=optional_float(state.get("s_deg_slope_5")),
            model_s_per_lap=model_slope,
            acceleration_s_per_lap2=optional_float(state.get("s_deg_accel")),
        ),
        next_lap=intervals[0],
        next_laps=intervals,
        remaining_competitive_laps=RemainingLifeOut(
            value=life.value,
            threshold_s=life.threshold_s,
            reference_s=life.reference_s,
            capped_by_race_end=life.capped_by_race_end,
        ),
        decomposition=DecompositionOut(
            field_ref_s=parts.field_ref_s,
            driver_offset_s=parts.driver_offset_s,
            tyre_s=parts.tyre_s,
            fuel_s=parts.fuel_s,
            context_s=parts.context_s,
            total_s=parts.total_s,
        ),
        explanation=explanation_lines(state, parts, life, window, cfg.reference_tyre_age),
    )


def _window_out(result: PitWindowResult) -> LapWindow | None:
    if result.window is None:
        return None
    return LapWindow(start_lap=result.window.start_lap, end_lap=result.window.end_lap)


def pit_window(store: ProfileStore, request: PitWindowRequest) -> PitWindowResponse:
    """Candidate pit-lap cost curve and recommended window."""
    name, model = resolve_model(store, request.model)
    cfg = store.config
    state = _decision(store, request.race_id, request.driver_id, request.lap)
    inputs = _inputs(
        store, state, request.replacement_compound, request.pit_loss_s, request.warmup_s
    )
    result = evaluate_pit_window(model, state, store.sigma[name], inputs)
    threshold = request.threshold_s or cfg.competitive_threshold_s
    life = remaining_competitive_laps(
        model, state, threshold, cfg.reference_tyre_age, cfg.fuel_s_per_lap
    )
    parts = decompose_next_lap(model, state, cfg.reference_tyre_age, cfg.fuel_s_per_lap)
    laps = driver_laps(store, request.race_id, request.driver_id)
    return PitWindowResponse(
        model=name,
        candidates=[
            PitCandidateOut(
                pit_lap=c.pit_lap,
                cost_pit_s=c.cost_pit_s,
                cost_stay_out_s=c.cost_stay_out_s,
                advantage_s=c.advantage_s,
                margin_s=c.margin_s,
                in_window=c.in_window,
                extrapolated=c.extrapolated,
            )
            for c in result.candidates
        ],
        window=_window_out(result),
        optimal_lap=result.optimal_lap,
        stay_out_feasible=result.stay_out_feasible,
        replacement_compound=result.replacement_compound,
        pit_loss_s=result.pit_loss_s,
        actual_pit_laps=[int(v) for v in laps.loc[laps["pit_in"], "lap"]],
        assumptions=Assumptions(
            pit_loss_s=inputs.pit_loss_s,
            pit_loss_source="override" if request.pit_loss_s is not None else "estimated",
            warmup_s=inputs.warmup_s,
            fuel_s_per_lap=inputs.fuel_s_per_lap,
            margin_sigma=inputs.margin_sigma,
            threshold_s=threshold,
            reference_tyre_age=cfg.reference_tyre_age,
            replacement_rule="actual next compound, else hardest unused dry compound",
        ),
        explanation=explanation_lines(state, parts, life, result, cfg.reference_tyre_age),
    )


def counterfactual_response(
    store: ProfileStore, request: CounterfactualRequest
) -> CounterfactualResponse:
    """Stay-out vs pit-at-``pit_lap`` trajectories next to the actual lap times."""
    _, model = resolve_model(store, request.model)
    state = _decision(store, request.race_id, request.driver_id, request.lap)
    if not request.lap < request.pit_lap < int(state["total_laps"]):
        raise OutOfRangeError("pit_lap must be after the decision lap and before the final lap")
    inputs = _inputs(store, state, request.replacement_compound, None, None)
    result = counterfactual(model, state, request.pit_lap, inputs)
    actual = driver_laps(store, request.race_id, request.driver_id).set_index("lap")["lap_time_s"]
    return CounterfactualResponse(
        laps=list(result.laps),
        stay_out=list(result.stay_out_s),
        pit=list(result.pit_s),
        cumulative_delta_s=list(result.cumulative_delta_s),
        actual=[optional_float(actual.get(lap)) for lap in result.laps],
        replacement_compound=inputs.replacement_compound,
        pit_loss_s=inputs.pit_loss_s,
    )
