"""Forecasters: per-lap pace models and sequence models bound to their history source.

A per-lap :class:`~boxbox.models.types.PaceModel` predicts each design row on its own. A
:class:`~boxbox.models.sequence_data.SequencePaceModel` needs the driver's lap history up to
the decision lap and reads a whole branch (laps t+1..) as one decoder sequence, so it is bound
to the feature rows its histories come from (:class:`SequenceForecaster`).

The sequence model only predicts ``horizon`` steps and clips tyre age at the training cap
like the per-lap models. :func:`predict_branches` continues both the same way the per-lap path
does: beyond a cap with ``max(own end slope, TyreAgeClip.min_slope, 0)`` and beyond the horizon
with tyre-prior increments from the last predicted steps. Every continued row is flagged.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from boxbox.features.tyre_prior import TyrePrior, tyre_prior_values
from boxbox.models.sequence_data import (
    FUTURE_FEATURES,
    BoolArray,
    DriverKey,
    IntArray,
    SequencePaceModel,
    driver_tables,
    future_from_design,
    histories_for_decisions,
)
from boxbox.models.types import (
    EXTRAPOLATION_SLOPE_SPAN,
    FloatArray,
    PaceModel,
    TyreAgeClip,
    predict_pace,
)

NEW_STINT_INDEX: Final[int] = FUTURE_FEATURES.index("new_stint")
CONTINUATION_ANCHOR_LAPS: Final[int] = 3


@dataclass(frozen=True)
class SequenceForecaster:
    """A sequence model, the per-driver feature rows its histories are read from and the prior.

    ``prior`` continues rollouts beyond the model's ``horizon`` (tyre-wear increments).
    """

    model: SequencePaceModel
    tables: dict[DriverKey, pd.DataFrame]
    prior: TyrePrior

    @property
    def name(self) -> Literal["lstm"]:
        return self.model.name

    @property
    def tyre_clip(self) -> TyreAgeClip:
        return self.model.tyre_clip


Forecaster = PaceModel | SequenceForecaster


def bind_sequence_model(
    model: SequencePaceModel, features: pd.DataFrame, prior: TyrePrior
) -> SequenceForecaster:
    """Bind ``model`` to the feature table that holds every decision's lap history."""
    return SequenceForecaster(model, driver_tables(features), prior)


@dataclass(frozen=True)
class BranchDesign:
    """Stacked decoder rows of several branches, each branch's rows contiguous and in lap order.

    Branch ``b`` is a decision at the end of ``decision_lap[b]`` for one driver; its rows are
    laps decision_lap+1.. with ``new_stint`` True on rows run on a different set than the
    decision lap's.
    """

    race_id: Sequence[str]
    driver_id: Sequence[str]
    decision_lap: Sequence[int]
    design: pd.DataFrame
    branch: IntArray
    new_stint: BoolArray


def _row_caps(clip: TyreAgeClip, compound: pd.Series) -> FloatArray:
    default = max(clip.max_age.values())
    caps = compound.astype(str).map(clip.max_age).fillna(default)
    return np.asarray(caps.to_numpy(dtype=np.float64), dtype=np.float64)


def _set_ids(step: IntArray, new_stint: BoolArray, age: FloatArray) -> IntArray:
    """Global tyre-set id per row: a set starts each branch, a flag flip or a tyre-age reset."""
    changed = np.zeros(len(step), dtype=np.bool_)
    changed[1:] = (new_stint[1:] != new_stint[:-1]) | (age[1:] < age[:-1])
    starts = (step == 0) | changed
    return np.asarray(np.cumsum(starts) - 1, dtype=np.int64)


def _own_end_slopes(
    sets: IntArray, age: FloatArray, caps: FloatArray, y: FloatArray, predicted: BoolArray
) -> FloatArray:
    """Per set: the model's slope over its predicted laps in the last span before the cap.

    NaN where fewer than two such laps were predicted (the set started near or past the cap).
    """
    usable = predicted & (age <= caps) & (age >= caps - EXTRAPOLATION_SLOPE_SPAN)
    frame = pd.DataFrame({"set": sets[usable], "age": age[usable], "y": y[usable]})
    ends = frame.groupby("set").agg(
        a0=("age", "first"), a1=("age", "last"), y0=("y", "first"), y1=("y", "last")
    )
    span = (ends["a1"] - ends["a0"]).to_numpy(dtype=np.float64)
    rise = (ends["y1"] - ends["y0"]).to_numpy(dtype=np.float64)
    out = np.full(int(sets.max()) + 1 if len(sets) else 0, np.nan, dtype=np.float64)
    out[ends.index.to_numpy(dtype=np.int64)] = np.where(
        span > 0, rise / np.where(span > 0, span, 1.0), np.nan
    )
    return out


def predict_branches(
    forecaster: SequenceForecaster, branches: BranchDesign
) -> tuple[FloatArray, BoolArray]:
    """Relative pace and continued-row flags for every stacked row of ``branches``.

    The first ``horizon`` rows of each branch are one batched decoder call. Rows beyond the
    tyre-age cap add ``max(own end slope, min_slope, 0) x laps past the cap`` to the (clipped)
    prediction, the own slope being measured on the same set's predicted laps just before the
    cap. Rows beyond the horizon follow the tyre prior from the level of the last predicted
    laps (see :func:`_continue_with_prior`). Both kinds of row are flagged (they count as laps
    past the cap in strategy plans).
    """
    model = forecaster.model
    design = branches.design
    n_rows, n_branches = len(design), len(branches.decision_lap)
    if n_rows == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.bool_)
    first_row = np.searchsorted(branches.branch, np.arange(n_branches))
    step = np.arange(n_rows, dtype=np.int64) - first_row[branches.branch]
    in_horizon = step < model.horizon
    width = min(model.horizon, int(step.max()) + 1)

    encoded = future_from_design(design, new_stint=False)
    encoded[:, NEW_STINT_INDEX] = branches.new_stint.astype(np.float64)
    future = np.zeros((n_branches, width, encoded.shape[1]), dtype=np.float64)
    future[branches.branch[in_horizon], step[in_horizon]] = encoded[in_horizon]
    history, history_mask = histories_for_decisions(
        forecaster.tables,
        branches.race_id,
        branches.driver_id,
        branches.decision_lap,
        model.window,
    )
    raw = model.predict_rollouts(history, history_mask, future)

    y = np.full(n_rows, np.nan, dtype=np.float64)
    y[in_horizon] = raw[branches.branch[in_horizon], step[in_horizon]]
    age = design["tyre_age"].to_numpy(dtype=np.float64)
    caps = _row_caps(model.tyre_clip, design["compound"])
    excess = np.clip(age - caps, 0.0, None)
    sets = _set_ids(step, branches.new_stint, age)
    own = _own_end_slopes(sets, age, caps, y, in_horizon)[sets]
    floor = design["compound"].astype(str).map(model.tyre_clip.min_slope).fillna(0.0).to_numpy()
    slope = np.maximum(np.maximum(np.nan_to_num(own, nan=-np.inf), floor), 0.0)
    y[in_horizon] += (slope * excess)[in_horizon]

    beyond = ~in_horizon
    out_lap = branches.new_stint & np.r_[True, sets[1:] != sets[:-1]]
    if beyond.any() or out_lap.any():
        prior = tyre_prior_values(forecaster.prior, design)
        _continue_with_prior(y, prior, branches.branch, sets, in_horizon & ~out_lap)
    return y, np.asarray((excess > 0) | beyond, dtype=np.bool_)


def _continue_with_prior(
    y: FloatArray, prior: FloatArray, branch: IntArray, sets: IntArray, anchors: BoolArray
) -> None:
    """Fill every non-anchor row of ``y`` in place with ``level + prior curve``.

    Non-anchor rows are the laps beyond the horizon and each new set's out-lap: the out-lap's
    target is masked in training, so the network's value there is unconstrained (it gets the
    value of an ordinary lap at its tyre age, as a per-lap model would give it). The level is
    the median of ``y - prior`` over ``CONTINUATION_ANCHOR_LAPS`` anchor laps of the same set
    (the first ones for its out-lap, the last ones for laps beyond the horizon), or over the
    last anchor laps run before the set when it has none. Using a few laps keeps one noisy
    step from shifting the whole rest of the race.
    """
    residual = y - prior
    starts = np.flatnonzero(np.r_[True, sets[1:] != sets[:-1]])
    ends = np.r_[starts[1:], len(sets)]
    earlier: IntArray = np.zeros(0, dtype=np.int64)
    for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if i == 0 or branch[start] != branch[start - 1]:
            earlier = np.zeros(0, dtype=np.int64)
        own = start + np.flatnonzero(anchors[start:end])
        fill = start + np.flatnonzero(~anchors[start:end])
        if len(fill):
            fallback = earlier[-CONTINUATION_ANCHOR_LAPS:]
            for rows, reference in (
                (fill[fill < own[0]] if len(own) else fill, own[:CONTINUATION_ANCHOR_LAPS]),
                (fill[fill > own[-1]] if len(own) else fill[:0], own[-CONTINUATION_ANCHOR_LAPS:]),
            ):
                chosen = reference if len(reference) else fallback
                if not len(chosen):  # horizon of one lap on an out-lap: keep its own value
                    chosen = fill[np.isfinite(y[fill])][:1]
                if len(rows) and len(chosen):
                    y[rows] = float(np.median(residual[chosen])) + prior[rows]
        earlier = np.r_[earlier, own]


def _previous_stints(forecaster: SequenceForecaster, rows: pd.DataFrame) -> FloatArray:
    stints = pd.concat(forecaster.tables.values()).set_index(["race_id", "driver_id", "lap"])
    keys = pd.MultiIndex.from_arrays(
        [rows["race_id"].astype(str), rows["driver_id"].astype(str), rows["lap"] - 1]
    )
    return stints["stint_id"].reindex(keys).to_numpy(dtype=np.float64)


def one_step_pace(model: Forecaster, rows: pd.DataFrame) -> FloatArray:
    """Next-lap ``y_s`` for each feature row (decision lap = row lap - 1).

    Per-lap models predict the row itself (:func:`predict_pace`). A sequence model reads the
    history up to the previous lap and decodes the row's own schedule as a one-step branch;
    the previous lap must have a feature row.
    """
    if not isinstance(model, SequenceForecaster):
        return predict_pace(model, rows)
    if rows.empty:
        return np.zeros(0, dtype=np.float64)
    previous = _previous_stints(model, rows)
    new_stint = rows["stint_id"].to_numpy(dtype=np.float64) != previous
    branches = BranchDesign(
        race_id=list(rows["race_id"].astype(str)),
        driver_id=list(rows["driver_id"].astype(str)),
        decision_lap=[int(v) - 1 for v in rows["lap"]],
        design=rows.assign(lap_number=rows["lap"].astype(float)).reset_index(drop=True),
        branch=np.arange(len(rows), dtype=np.int64),
        new_stint=np.asarray(new_stint, dtype=np.bool_),
    )
    return predict_branches(model, branches)[0]
