"""Encoder–decoder sequence samples: causal lap history plus the known future schedule.

One sample is a decision at the end of lap ``t`` for one driver. The encoder sees the last
``window`` laps ``t-window+1..t`` (left-padded, never anything after ``t``); the decoder sees
the deterministic schedule of laps ``t+1..t+horizon`` (compound, tyre age, stint lap, lap
number, race progress and whether the lap is on a different set than lap ``t``).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from boxbox.models.types import FloatArray, TyreAgeClip
from boxbox.schema import MODEL_COMPOUNDS

BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]

COMPOUND_COLUMNS: Final[tuple[str, ...]] = tuple(f"compound_{c}" for c in MODEL_COMPOUNDS)
SCHEDULE_NUMERIC: Final[tuple[str, ...]] = ("tyre_age", "stint_lap", "lap_number", "race_progress")
FUTURE_FEATURES: Final[tuple[str, ...]] = (
    "compound_SOFT",
    "compound_MEDIUM",
    "compound_HARD",
    "compound_UNKNOWN",
    "tyre_age",
    "stint_lap",
    "lap_number",
    "race_progress",
    "new_stint",
)
STATE_INPUTS: Final[tuple[str, ...]] = (
    "s_driver_offset_s",
    "s_gap_ahead_s",
    "s_in_traffic",
)
HISTORY_FEATURES: Final[tuple[str, ...]] = (
    "y_s",
    "is_clean",
    *COMPOUND_COLUMNS,
    *SCHEDULE_NUMERIC,
    "current_stint",
    "pit_lap",
    "track_neutralised",
    *STATE_INPUTS,
)
HISTORY_Y_CLIP_S: Final[float] = 5.0
"""Pit, Safety Car and lap-one laps can be 20-40 s off; history keeps them but bounded."""

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "race_id",
    "driver_id",
    "lap",
    "split",
    "y_s",
    "is_clean",
    "compound",
    "stint_id",
    "pit_in",
    "pit_out",
    "track_neutralised",
    *SCHEDULE_NUMERIC,
    *STATE_INPUTS,
)

HISTORY_Y_INDEX: Final[int] = HISTORY_FEATURES.index("y_s")
HISTORY_CLEAN_INDEX: Final[int] = HISTORY_FEATURES.index("is_clean")
_CURRENT_STINT = HISTORY_FEATURES.index("current_stint")


@dataclass(frozen=True)
class SequenceBatch:
    """Aligned encoder/decoder arrays, one row per (race, driver, decision lap)."""

    history: FloatArray
    history_mask: BoolArray
    future: FloatArray
    target: FloatArray
    target_mask: BoolArray
    keys: pd.DataFrame

    def __len__(self) -> int:
        return len(self.keys)

    def subset(self, rows: BoolArray) -> "SequenceBatch":
        """Samples selected by a boolean row mask."""
        return SequenceBatch(
            history=self.history[rows],
            history_mask=self.history_mask[rows],
            future=self.future[rows],
            target=self.target[rows],
            target_mask=self.target_mask[rows],
            keys=self.keys.loc[rows].reset_index(drop=True),
        )


class SequencePaceModel(Protocol):
    """A fitted encoder–decoder pace model that rolls out a whole future schedule at once."""

    @property
    def name(self) -> Literal["lstm"]: ...

    @property
    def horizon(self) -> int: ...

    @property
    def window(self) -> int: ...

    @property
    def tyre_clip(self) -> TyreAgeClip: ...

    def predict_rollout(
        self, history: FloatArray, history_mask: BoolArray, future: FloatArray
    ) -> FloatArray:
        """Predicted ``y_s`` for each future step of ONE decision (``future`` is [H', F_fut])."""
        ...

    def predict_rollouts(
        self, history: FloatArray, history_mask: BoolArray, future: FloatArray
    ) -> FloatArray:
        """[N, H'] predictions for N decisions (all-zero trailing future rows are padding)."""
        ...


def compound_one_hot(compound: pd.Series) -> FloatArray:
    """[n, 4] one-hot over ``MODEL_COMPOUNDS``; wet/unrecognised compounds are all zeros."""
    values = compound.astype(str).to_numpy()
    return np.stack([values == c for c in MODEL_COMPOUNDS], axis=1).astype(np.float64)


def _schedule(frame: pd.DataFrame) -> FloatArray:
    """[n, 8] compound one-hot plus the deterministic numeric schedule columns."""
    numeric = frame[list(SCHEDULE_NUMERIC)].to_numpy(dtype=np.float64)
    return np.concatenate([compound_one_hot(frame["compound"]), numeric], axis=1)


@dataclass(frozen=True)
class _DriverLaps:
    """One driver's race on a dense lap grid (index ``i`` is lap ``i + 1``)."""

    has_row: BoolArray
    usable: BoolArray
    """Row exists and ``y_s`` is known (usable as history)."""
    row_features: FloatArray
    """[L, F_hist] with ``current_stint`` left at 0 (it depends on the decision lap)."""
    schedule: FloatArray
    stint: IntArray
    y: FloatArray
    target_ok: BoolArray


def _driver_laps(rows: pd.DataFrame) -> _DriverLaps:
    """Dense per-lap arrays for one (race, driver), NaNs replaced by 0."""
    laps = rows["lap"].to_numpy(dtype=np.int64)
    n_laps = int(laps.max())
    index = laps - 1
    y = rows["y_s"].to_numpy(dtype=np.float64)
    clean = rows["is_clean"].to_numpy(dtype=np.bool_) & np.isfinite(y)
    pit = rows["pit_in"].to_numpy(dtype=np.bool_) | rows["pit_out"].to_numpy(dtype=np.bool_)
    schedule = _schedule(rows)
    per_row = np.column_stack(
        [
            np.clip(y, -HISTORY_Y_CLIP_S, HISTORY_Y_CLIP_S),
            clean,
            schedule,
            np.zeros(len(rows)),
            pit,
            rows["track_neutralised"].to_numpy(dtype=np.bool_),
            rows[list(STATE_INPUTS)].to_numpy(dtype=np.float64),
        ]
    )

    def dense(values: npt.NDArray[np.generic], fill: object) -> npt.NDArray[np.generic]:
        out = np.full((n_laps, *values.shape[1:]), fill, dtype=values.dtype)
        out[index] = values
        return out

    has_row = np.zeros(n_laps, dtype=np.bool_)
    has_row[index] = True
    usable = np.zeros(n_laps, dtype=np.bool_)
    usable[index] = np.isfinite(y)
    return _DriverLaps(
        has_row=has_row,
        usable=usable,
        row_features=np.nan_to_num(np.asarray(dense(per_row, 0.0), dtype=np.float64)),
        schedule=np.nan_to_num(np.asarray(dense(schedule, 0.0), dtype=np.float64)),
        stint=np.asarray(dense(rows["stint_id"].to_numpy(dtype=np.int64), -1), dtype=np.int64),
        y=np.asarray(dense(y, np.nan), dtype=np.float64),
        target_ok=np.asarray(dense(clean, False), dtype=np.bool_),
    )


def _history(
    driver: _DriverLaps, decision_index: IntArray, window: int
) -> tuple[FloatArray, BoolArray]:
    """[n, W, F_hist], [n, W] for decision laps at dense indices ``decision_index``."""
    offsets = np.arange(-window + 1, 1, dtype=np.int64)
    idx = decision_index[:, None] + offsets[None, :]
    valid = idx >= 0
    safe = np.where(valid, idx, 0)
    mask = valid & driver.usable[safe]
    history = np.where(mask[..., None], driver.row_features[safe], 0.0)
    same_stint = driver.stint[safe] == driver.stint[decision_index][:, None]
    history[..., _CURRENT_STINT] = (mask & same_stint).astype(np.float64)
    return history, mask


def _future(
    driver: _DriverLaps, decision_index: IntArray, horizon: int
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Future schedule [n, H, F_fut], target [n, H] and target mask [n, H]."""
    idx = decision_index[:, None] + np.arange(1, horizon + 1, dtype=np.int64)[None, :]
    valid = idx < len(driver.has_row)
    safe = np.where(valid, idx, 0)
    present = valid & driver.has_row[safe]
    new_stint = present & (driver.stint[safe] != driver.stint[decision_index][:, None])
    future = np.concatenate(
        [np.where(present[..., None], driver.schedule[safe], 0.0), new_stint[..., None]], axis=2
    )
    target = np.where(present, driver.y[safe], np.nan)
    target_mask = present & driver.target_ok[safe]
    return future.astype(np.float64), target.astype(np.float64), target_mask


def _check_columns(features: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(f"features are missing sequence columns: {missing}")


def _empty_batch(window: int, horizon: int) -> SequenceBatch:
    return SequenceBatch(
        history=np.zeros((0, window, len(HISTORY_FEATURES))),
        history_mask=np.zeros((0, window), dtype=np.bool_),
        future=np.zeros((0, horizon, len(FUTURE_FEATURES))),
        target=np.zeros((0, horizon)),
        target_mask=np.zeros((0, horizon), dtype=np.bool_),
        keys=pd.DataFrame({"race_id": [], "driver_id": [], "lap": [], "split": []}),
    )


def build_sequences(features: pd.DataFrame, window: int, horizon: int) -> SequenceBatch:
    """One sample per (race, driver, decision lap t) with >= 1 clean history lap.

    History covers laps ``t-window+1..t`` only (causal). Future covers the actual laps
    ``t+1..t+horizon``: ``new_stint`` is 1 on every lap whose stint differs from the stint of
    lap ``t`` (from the out-lap after a stop onward). Targets are ``y_s`` of those laps, masked
    to clean laps; missing laps past the flag or after a retirement are zero rows with
    ``target_mask`` False.
    """
    _check_columns(features)
    parts: list[tuple[FloatArray, BoolArray, FloatArray, FloatArray, BoolArray, pd.DataFrame]] = []
    ordered = features.sort_values(["race_id", "driver_id", "lap"])
    for (race_id, driver_id), rows in ordered.groupby(["race_id", "driver_id"], sort=True):
        driver = _driver_laps(rows)
        decision_index = np.flatnonzero(driver.has_row).astype(np.int64)
        history, mask = _history(driver, decision_index, window)
        keep = (mask & (history[..., HISTORY_CLEAN_INDEX] > 0)).any(axis=1)
        if not keep.any():
            continue
        decision_index = decision_index[keep]
        future, target, target_mask = _future(driver, decision_index, horizon)
        split = rows["split"].iloc[0]
        keys = pd.DataFrame(
            {
                "race_id": race_id,
                "driver_id": driver_id,
                "lap": decision_index + 1,
                "split": split,
            }
        )
        parts.append((history[keep], mask[keep], future, target, target_mask, keys))
    if not parts:
        return _empty_batch(window, horizon)
    return SequenceBatch(
        history=np.concatenate([p[0] for p in parts]),
        history_mask=np.concatenate([p[1] for p in parts]),
        future=np.concatenate([p[2] for p in parts]),
        target=np.concatenate([p[3] for p in parts]),
        target_mask=np.concatenate([p[4] for p in parts]),
        keys=pd.concat([p[5] for p in parts], ignore_index=True),
    )


def history_for_state(
    features: pd.DataFrame, race_id: str, driver_id: str, lap: int, window: int
) -> tuple[FloatArray, BoolArray]:
    """[W, F_hist], [W] history for one decision lap (same encoding as ``build_sequences``).

    Rows after ``lap`` are dropped before encoding, so the result cannot depend on them.
    """
    _check_columns(features)
    rows = features[
        (features["race_id"] == race_id)
        & (features["driver_id"] == driver_id)
        & (features["lap"] <= lap)
    ].sort_values("lap")
    if rows.empty or int(rows["lap"].max()) != lap:
        raise ValueError(f"no feature row for {race_id}/{driver_id} lap {lap}")
    history, mask = _history(_driver_laps(rows), np.array([lap - 1], dtype=np.int64), window)
    return history[0], mask[0]


DriverKey = tuple[str, str]


def driver_tables(features: pd.DataFrame) -> dict[DriverKey, pd.DataFrame]:
    """Feature rows per ``(race_id, driver_id)`` ordered by lap (index for history lookups)."""
    _check_columns(features)
    ordered = features.sort_values(["race_id", "driver_id", "lap"])
    return {
        (str(race_id), str(driver_id)): rows
        for (race_id, driver_id), rows in ordered.groupby(["race_id", "driver_id"], sort=False)
    }


def histories_for_decisions(
    tables: dict[DriverKey, pd.DataFrame],
    race_ids: Sequence[str],
    driver_ids: Sequence[str],
    laps: Sequence[int],
    window: int,
) -> tuple[FloatArray, BoolArray]:
    """[N, W, F_hist], [N, W] histories, identical to :func:`history_for_state` per decision.

    Each driver's lap grid is encoded once for all of their decisions.

    Raises:
        ValueError: If a decision lap has no feature row.
    """
    n = len(laps)
    history = np.zeros((n, window, len(HISTORY_FEATURES)), dtype=np.float64)
    mask = np.zeros((n, window), dtype=np.bool_)
    positions: dict[DriverKey, list[int]] = {}
    for i, key in enumerate(zip(race_ids, driver_ids, strict=True)):
        positions.setdefault((str(key[0]), str(key[1])), []).append(i)
    for key, rows_at in positions.items():
        rows = tables.get(key)
        decision_index = np.array([laps[i] for i in rows_at], dtype=np.int64) - 1
        if rows is None:
            raise ValueError(f"no feature rows for {key[0]}/{key[1]}")
        driver = _driver_laps(rows)
        in_grid = (decision_index >= 0) & (decision_index < len(driver.has_row))
        if not in_grid.all() or not driver.has_row[decision_index].all():
            raise ValueError(f"no feature row for {key[0]}/{key[1]} at a requested lap")
        h, m = _history(driver, decision_index, window)
        history[rows_at], mask[rows_at] = h, m
    return history, mask


def future_from_design(design: pd.DataFrame, new_stint: bool) -> FloatArray:
    """[H', F_fut] decoder input from a rollout design frame.

    ``design`` needs ``compound, tyre_age, stint_lap, lap_number, race_progress`` (as produced
    by ``features.build.rollout_design``). A pit branch is two calls concatenated on axis 0:
    the stay-out laps with ``new_stint=False``, then the new-tyre laps with ``new_stint=True``.
    """
    flag = np.full((len(design), 1), float(new_stint))
    return np.concatenate([np.nan_to_num(_schedule(design)), flag], axis=1).astype(np.float64)


def future_exists(future: FloatArray) -> BoolArray:
    """Mask of future steps that are real laps (padding rows are all-zero, lap_number = 0)."""
    return np.asarray(future[..., FUTURE_FEATURES.index("lap_number")] > 0, dtype=np.bool_)
