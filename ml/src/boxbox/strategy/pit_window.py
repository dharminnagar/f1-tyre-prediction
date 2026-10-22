"""Pit-window calculator: race-time cost of boxing at each candidate lap, with uncertainty.

For decision lap t, race length T and candidate in-lap p (t < p < T), all costs cover laps
t+1..T:
    StayOut      = laps t+1..T on the current tyre (no further stop)
    Pit(p)       = laps t+1..p on the current tyre + pit_loss + warm-up + the cheaper of
                   (a) laps p+1..T on a fresh replacement set, or
                   (b) a fresh set until a best second stop q, then another fresh set to T
                   (Barcelona is usually a two-stop race, so (b) avoids a late-biased optimum)
    Advantage(p) = StayOut - Pit(p)
p* = argmin Pit(p). p is in the window when it is within the model-uncertainty margin of the
optimum, ``Pit(p) - Pit(p*) <= k * sigma_cum(|p - p*|)``, and (when finishing on the current
set is allowed) also beats staying out, ``Advantage(p) > k * sigma_cum(T - p)``.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from boxbox.models.types import PaceModel, predict_pace
from boxbox.schema import COMPOUND_HARDNESS, DRY_COMPOUNDS, Compound
from boxbox.strategy.rollout import (
    lap_times,
    new_tyre_design,
    pit_rollout,
    stay_out_rollout,
)
from boxbox.strategy.uncertainty import SigmaCurve


@dataclass(frozen=True)
class PitCandidate:
    """Costs for boxing at the end of ``pit_lap``."""

    pit_lap: int
    cost_pit_s: float
    cost_stay_out_s: float
    advantage_s: float
    margin_s: float
    in_window: bool
    extrapolated: bool


@dataclass(frozen=True)
class LapRange:
    """Inclusive lap range."""

    start_lap: int
    end_lap: int


@dataclass(frozen=True)
class PitWindowResult:
    """Candidate curve, recommended window and the assumptions used."""

    candidates: tuple[PitCandidate, ...]
    window: LapRange | None
    optimal_lap: int | None
    stay_out_feasible: bool
    replacement_compound: Compound
    pit_loss_s: float
    warmup_s: float


@dataclass(frozen=True)
class StrategyInputs:
    """Strategy assumptions applied to one decision."""

    replacement_compound: Compound
    stay_out_feasible: bool
    pit_loss_s: float
    warmup_s: float
    margin_sigma: float
    fuel_s_per_lap: float


def choose_replacement(driver_laps: pd.DataFrame, lap: int) -> tuple[Compound, bool]:
    """Replacement compound and whether finishing on the current set satisfies the 2-compound rule.

    Uses the compound actually fitted next when known, otherwise the hardest dry compound not
    yet used (HARD when all were used).
    """
    current_stint = int(driver_laps.loc[driver_laps["lap"] == lap, "stint_id"].iloc[0])
    stints = driver_laps.groupby("stint_id")["compound"].first()
    used: set[Compound] = {
        c for c in DRY_COMPOUNDS if c in set(stints[stints.index <= current_stint])
    }
    feasible = len(used) >= 2
    later = stints[stints.index > current_stint]
    next_compound = str(later.iloc[0]) if len(later) else ""
    for candidate in DRY_COMPOUNDS:
        if candidate == next_compound:
            return candidate, feasible
    unused = sorted((c for c in DRY_COMPOUNDS if c not in used), key=COMPOUND_HARDNESS.__getitem__)
    return (unused[-1] if unused else "HARD"), feasible


def evaluate_pit_window(
    model: PaceModel, state: pd.Series, sigma: SigmaCurve, inputs: StrategyInputs
) -> PitWindowResult:
    """Cost curve over every candidate in-lap after the decision lap."""
    total_laps = int(state["total_laps"])
    decision_lap = int(state["lap"])
    stay = stay_out_rollout(model, state, total_laps, inputs.fuel_s_per_lap)
    pit_laps = list(range(decision_lap + 1, total_laps))
    if not pit_laps:
        return PitWindowResult(
            (),
            None,
            None,
            inputs.stay_out_feasible,
            inputs.replacement_compound,
            inputs.pit_loss_s,
            inputs.warmup_s,
        )
    designs = [new_tyre_design(state, p, total_laps, inputs.replacement_compound) for p in pit_laps]
    stacked = pd.concat(designs, ignore_index=True)
    stacked_laps = stacked["lap_number"].to_numpy(dtype=np.int64)
    new_times = lap_times(state, stacked_laps, predict_pace(model, stacked), inputs.fuel_s_per_lap)
    new_extrapolated = model.tyre_clip.extrapolated(stacked).to_numpy()
    offsets = np.cumsum([0, *[len(d) for d in designs]])
    branches = [new_times[int(offsets[i]) : int(offsets[i + 1])] for i in range(len(pit_laps))]
    branch_total = {p: float(b.sum()) for p, b in zip(pit_laps, branches, strict=True)}
    stop_cost = inputs.pit_loss_s + inputs.warmup_s

    def continuation(i: int, p: int) -> float:
        prefix = np.cumsum(branches[i])
        one_stop = float(prefix[-1])
        second_stops = [
            float(prefix[q - p - 1]) + stop_cost + branch_total[q] for q in range(p + 2, total_laps)
        ]
        return min([one_stop, *second_stops])

    cost_stay = float(stay.lap_time_s.sum())
    costs = []
    for i, p in enumerate(pit_laps):
        before = stay.laps <= p
        cost = float(stay.lap_time_s[before].sum()) + stop_cost + continuation(i, p)
        extrapolated = bool(
            stay.extrapolated[before].any() or new_extrapolated[offsets[i] : offsets[i + 1]].any()
        )
        costs.append((p, cost, extrapolated))

    optimal_lap, optimal_cost, _ = min(costs, key=lambda c: c[1])
    candidates = []
    for p, cost, extrapolated in costs:
        advantage = cost_stay - cost
        margin = inputs.margin_sigma * sigma.cumulative_sigma(max(1, abs(p - optimal_lap)))
        beats_stay_out = advantage > inputs.margin_sigma * sigma.cumulative_sigma(total_laps - p)
        near_optimum = cost - optimal_cost <= margin
        in_window = near_optimum and (beats_stay_out or not inputs.stay_out_feasible)
        candidates.append(
            PitCandidate(p, cost, cost_stay, advantage, margin, in_window, extrapolated)
        )

    best = next(c for c in candidates if c.pit_lap == optimal_lap)
    window = _window_around(candidates, best) if best.in_window else None
    return PitWindowResult(
        tuple(candidates),
        window,
        optimal_lap,
        inputs.stay_out_feasible,
        inputs.replacement_compound,
        inputs.pit_loss_s,
        inputs.warmup_s,
    )


def _window_around(candidates: list[PitCandidate], best: PitCandidate) -> LapRange:
    index = candidates.index(best)
    start = end = index
    while start > 0 and candidates[start - 1].in_window:
        start -= 1
    while end < len(candidates) - 1 and candidates[end + 1].in_window:
        end += 1
    return LapRange(candidates[start].pit_lap, candidates[end].pit_lap)


@dataclass(frozen=True)
class Counterfactual:
    """Stay-out vs pit-at-``pit_lap`` lap-time trajectories (pit loss charged on the out-lap)."""

    laps: tuple[int, ...]
    stay_out_s: tuple[float, ...]
    pit_s: tuple[float, ...]
    cumulative_delta_s: tuple[float, ...]


def counterfactual(
    model: PaceModel, state: pd.Series, pit_lap: int, inputs: StrategyInputs
) -> Counterfactual:
    """Compare staying out with boxing at ``pit_lap`` until the end of the race."""
    total_laps = int(state["total_laps"])
    stay = stay_out_rollout(model, state, total_laps, inputs.fuel_s_per_lap)
    pit = pit_rollout(
        model, state, pit_lap, total_laps, inputs.replacement_compound, inputs.fuel_s_per_lap
    )
    pit_times = pit.lap_time_s.copy()
    out_lap = np.flatnonzero(pit.laps == pit_lap + 1)
    if len(out_lap):
        pit_times[out_lap[0]] += inputs.pit_loss_s + inputs.warmup_s
    delta = np.cumsum(pit_times - stay.lap_time_s)
    return Counterfactual(
        tuple(int(v) for v in stay.laps),
        tuple(float(v) for v in stay.lap_time_s),
        tuple(float(v) for v in pit_times),
        tuple(float(v) for v in delta),
    )
