"""Multi-lap rollouts of a pace model from a decision-lap state.

Every model prediction made by the strategy code goes through :func:`predict_segments` (or
its multi-decision form :func:`predict_decisions`), the single place that knows how a model
turns design segments into relative pace. A design segment is a block of future laps on one
tyre set, built by :func:`stay_out_design` (the current set, laps t+1..p) or
:func:`new_tyre_design` (a fresh set, laps p+1..end); its ``new_stint`` column says which.

A per-lap ``PaceModel`` predicts every row on its own. A ``SequenceForecaster`` decodes each
segment as its own branch from the decision lap: a segment starting after lap t+1 is preceded
by the stay-out laps t+1.. on the current set, whose predictions are discarded (the decoder is
causal, so they equal the stay-out segment's own).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from boxbox.features.build import rollout_design
from boxbox.models.forecaster import BranchDesign, Forecaster, SequenceForecaster, predict_branches
from boxbox.models.types import FloatArray, predict_pace

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Rollout:
    """Predicted laps ``laps`` (absolute lap numbers) for one strategy branch."""

    laps: IntArray
    y_s: FloatArray
    lap_time_s: FloatArray
    tyre_age: FloatArray
    compound: tuple[str, ...]
    extrapolated: BoolArray


def fuel_terms(state: pd.Series, laps: IntArray, fuel_s_per_lap: float) -> FloatArray:
    """Remaining-fuel penalty per future lap (zero in the refuelling era)."""
    if state["era"] == "refuel_2005_2009":
        return np.zeros(len(laps), dtype=np.float64)
    remaining = np.clip(float(state["total_laps"]) - laps, 0, None)
    return np.asarray(fuel_s_per_lap * remaining, dtype=np.float64)


def lap_times(state: pd.Series, laps: IntArray, y: FloatArray, fuel_s_per_lap: float) -> FloatArray:
    """Lap time = relative pace + field reference (frozen at the decision lap) + fuel term."""
    field_ref = float(state["s_field_ref_next_s"])
    return np.asarray(y + field_ref + fuel_terms(state, laps, fuel_s_per_lap), dtype=np.float64)


def _branch_design(
    state: pd.Series, laps: IntArray, compound: str, age0: float, stint_lap0: float, new: bool
) -> pd.DataFrame:
    return rollout_design(state, laps, compound, age0, stint_lap0, new_stint=new).assign(
        new_stint=new
    )


def stay_out_design(state: pd.Series, end_lap: int) -> pd.DataFrame:
    """Rows for laps t+1..end_lap on the current tyre."""
    laps = np.arange(int(state["lap"]) + 1, end_lap + 1, dtype=np.int64)
    return _branch_design(
        state,
        laps,
        str(state["compound"]),
        float(state["tyre_age"]) + 1,
        float(state["stint_lap"]) + 1,
        new=False,
    )


def new_tyre_design(state: pd.Series, pit_lap: int, end_lap: int, compound: str) -> pd.DataFrame:
    """Rows for laps pit_lap+1..end_lap on a fresh set (out-lap has tyre age 1)."""
    laps = np.arange(pit_lap + 1, end_lap + 1, dtype=np.int64)
    return _branch_design(state, laps, compound, 1.0, 1.0, new=True)


@dataclass(frozen=True)
class SegmentPrediction:
    """Relative pace and beyond-tyre-age-cap flags for the rows of one design segment."""

    y_s: FloatArray
    past_cap: BoolArray


def predict_segments(
    model: Forecaster, state: pd.Series, segments: Sequence[pd.DataFrame]
) -> list[SegmentPrediction]:
    """Predict every design segment from the decision ``state`` in one batched model call.

    Each segment must be a single tyre set (one :func:`stay_out_design` or
    :func:`new_tyre_design` block); a per-lap model ignores ``state``.
    """
    return predict_decisions(model, [(state, segments)])[0]


def predict_decisions(
    model: Forecaster, decisions: Sequence[tuple[pd.Series, Sequence[pd.DataFrame]]]
) -> list[list[SegmentPrediction]]:
    """:func:`predict_segments` for many decisions with one batched model call."""
    if isinstance(model, SequenceForecaster):
        stacked, positions = _sequence_branches(decisions)
        y, past_cap = predict_branches(model, stacked)
    else:
        frames = [s for _, segments in decisions for s in segments if len(s)]
        if not frames:
            return [[_empty() for _ in segments] for _, segments in decisions]
        rows = pd.concat(frames, ignore_index=True)
        y = predict_pace(model, rows)
        past_cap = model.tyre_clip.extrapolated(rows).to_numpy(dtype=np.bool_)
        offsets = np.cumsum([0, *[len(s) for _, segments in decisions for s in segments]])
        positions = [np.arange(offsets[i], offsets[i + 1]) for i in range(len(offsets) - 1)]
    out: list[list[SegmentPrediction]] = []
    index = 0
    for _, segments in decisions:
        out.append([])
        for _ in segments:
            rows_at = positions[index]
            out[-1].append(SegmentPrediction(y[rows_at], past_cap[rows_at]))
            index += 1
    return out


def _empty() -> SegmentPrediction:
    return SegmentPrediction(np.zeros(0), np.zeros(0, dtype=np.bool_))


def _sequence_branches(
    decisions: Sequence[tuple[pd.Series, Sequence[pd.DataFrame]]],
) -> tuple[BranchDesign, list[IntArray]]:
    """One decoder branch per segment (stay-out prefix + segment) and each segment's rows."""
    frames: list[pd.DataFrame] = []
    flags: list[BoolArray] = []
    branch_ids: list[IntArray] = []
    race_ids: list[str] = []
    driver_ids: list[str] = []
    decision_laps: list[int] = []
    positions: list[IntArray] = []
    n_rows = 0
    for state, segments in decisions:
        decision_lap = int(state["lap"])
        starts = [int(s["lap_number"].iloc[0]) for s in segments if len(s)]
        prefix_end = max(starts, default=decision_lap + 1) - 1
        stay = stay_out_design(state, prefix_end) if prefix_end > decision_lap else None
        for segment in segments:
            if not len(segment):
                positions.append(np.zeros(0, dtype=np.int64))
                continue
            n_prefix = int(segment["lap_number"].iloc[0]) - decision_lap - 1
            parts = [segment] if stay is None or n_prefix <= 0 else [stay.iloc[:n_prefix], segment]
            branch = pd.concat(parts, ignore_index=True)
            new = (
                segment["new_stint"].to_numpy(dtype=np.bool_)
                if "new_stint" in segment
                else np.zeros(len(segment), dtype=np.bool_)
            )
            flags.append(np.concatenate([np.zeros(len(branch) - len(segment), np.bool_), new]))
            frames.append(branch)
            branch_ids.append(np.full(len(branch), len(race_ids), dtype=np.int64))
            race_ids.append(str(state["race_id"]))
            driver_ids.append(str(state["driver_id"]))
            decision_laps.append(decision_lap)
            positions.append(np.arange(n_rows + len(branch) - len(segment), n_rows + len(branch)))
            n_rows += len(branch)
    columns = ["compound", "tyre_age", "stint_lap", "lap_number", "race_progress"]
    design = (
        pd.concat([f[columns] for f in frames], ignore_index=True)
        if frames
        else pd.DataFrame({c: [] for c in columns})
    )
    stacked = BranchDesign(
        race_id=race_ids,
        driver_id=driver_ids,
        decision_lap=decision_laps,
        design=design,
        branch=np.concatenate(branch_ids) if branch_ids else np.zeros(0, dtype=np.int64),
        new_stint=np.concatenate(flags) if flags else np.zeros(0, dtype=np.bool_),
    )
    return stacked, positions


def _to_rollout(
    model: Forecaster, state: pd.Series, segments: Sequence[pd.DataFrame], fuel_s_per_lap: float
) -> Rollout:
    predictions = predict_segments(model, state, segments)
    design = pd.concat(segments, ignore_index=True)
    laps = design["lap_number"].to_numpy(dtype=np.int64)
    y = np.concatenate([p.y_s for p in predictions]).astype(np.float64)
    return Rollout(
        laps=laps,
        y_s=y,
        lap_time_s=lap_times(state, laps, y, fuel_s_per_lap),
        tyre_age=design["tyre_age"].to_numpy(dtype=np.float64),
        compound=tuple(design["compound"].astype(str)),
        extrapolated=np.concatenate([p.past_cap for p in predictions]).astype(np.bool_),
    )


def stay_out_rollout(
    model: Forecaster, state: pd.Series, end_lap: int, fuel_s_per_lap: float
) -> Rollout:
    """Predicted laps t+1..end_lap without stopping."""
    return _to_rollout(model, state, [stay_out_design(state, end_lap)], fuel_s_per_lap)


def pit_rollout(
    model: Forecaster,
    state: pd.Series,
    pit_lap: int,
    end_lap: int,
    compound: str,
    fuel_s_per_lap: float,
) -> Rollout:
    """Predicted laps t+1..end_lap when boxing at the end of ``pit_lap`` (pit loss excluded)."""
    old = stay_out_design(state, pit_lap)
    new = new_tyre_design(state, pit_lap, end_lap, compound)
    return _to_rollout(model, state, [old, new], fuel_s_per_lap)


@dataclass(frozen=True)
class FreshSetTable:
    """Cumulative costs of a fresh ``compound`` set fitted for lap ``s`` and run to lap ``e``.

    ``cum_time_s[s, e]`` is the summed lap time of laps s..e and ``cum_past_cap[s, e]`` the
    number of those laps beyond the model's tyre-age cap (both indexed by absolute lap, NaN /
    0 where e < s or s was not tabulated).
    """

    compound: str
    first_start: int
    cum_time_s: FloatArray
    cum_past_cap: IntArray


def fresh_set_tables(
    model: Forecaster,
    state: pd.Series,
    compounds: Sequence[str],
    first_start: int,
    end_lap: int,
    fuel_s_per_lap: float,
) -> list[FreshSetTable]:
    """Tabulate fresh-set stints for every start lap ``first_start..end_lap`` and compound.

    Each stint is predicted as its own ``new_tyre_design(state, s - 1, end_lap, compound)``
    segment, so its laps are the same rows a pit branch boxing at ``s - 1`` would use.
    """
    starts = range(first_start, end_lap + 1)
    keys = [(c, s) for c in compounds for s in starts]
    segments = [new_tyre_design(state, s - 1, end_lap, c) for c, s in keys]
    predictions = predict_segments(model, state, segments)
    size = end_lap + 1
    times = {c: np.full((size, size), np.nan) for c in compounds}
    past = {c: np.zeros((size, size), dtype=np.int64) for c in compounds}
    for (compound, start), segment, prediction in zip(keys, segments, predictions, strict=True):
        laps = segment["lap_number"].to_numpy(dtype=np.int64)
        lap_time = lap_times(state, laps, prediction.y_s, fuel_s_per_lap)
        times[compound][start, start:] = np.cumsum(lap_time)
        past[compound][start, start:] = np.cumsum(prediction.past_cap)
    return [FreshSetTable(c, first_start, times[c], past[c]) for c in compounds]
