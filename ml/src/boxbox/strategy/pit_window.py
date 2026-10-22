"""Pit-window calculator: race-time cost of boxing at each candidate lap, with uncertainty.

For decision lap t, race length T and candidate in-lap p (t < p < T), all costs cover laps
t+1..T:
    StayOut      = laps t+1..T on the current tyre (no further stop)
    Pit(p)       = laps t+1..p on the current tyre + pit_loss + warm-up + the cheapest
                   continuation from a fresh set on lap p+1: up to ``MAX_EXTRA_STOPS`` more
                   stops, any dry compound per stint, and (unless already met) at least one
                   stint on a compound other than the current one (two-compound rule)
    Advantage(p) = StayOut - Pit(p)
p* = argmin Pit(p). p is in the window when it is within the model-uncertainty margin of the
optimum, ``Pit(p) - Pit(p*) <= k * sigma_cum(|p - p*|)``, and (when finishing on the current
set is allowed) also beats staying out, ``Advantage(p) > k * sigma_cum(T - p)``.
"""

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from boxbox.models.forecaster import Forecaster
from boxbox.models.types import FloatArray
from boxbox.schema import COMPOUND_HARDNESS, DRY_COMPOUNDS, Compound
from boxbox.strategy.rollout import (
    FreshSetTable,
    IntArray,
    fresh_set_tables,
    pit_rollout,
    stay_out_rollout,
)
from boxbox.strategy.uncertainty import SigmaCurve

MAX_EXTRA_STOPS: Final[int] = 2
CompoundPolicy = Literal["free", "fixed_first"]


@dataclass(frozen=True)
class PlannedStop:
    """Box at the end of ``lap`` and fit a fresh ``compound`` set."""

    lap: int
    compound: Compound


Plan = tuple[PlannedStop, ...]


@dataclass(frozen=True)
class PitCandidate:
    """Costs for boxing at the end of ``pit_lap`` and following ``plan`` to the flag.

    ``laps_past_cap`` counts the plan's laps (current set included) run beyond the model's
    training tyre-age cap, where the pace is extrapolated.
    """

    pit_lap: int
    cost_pit_s: float
    cost_stay_out_s: float
    advantage_s: float
    margin_s: float
    in_window: bool
    laps_past_cap: int
    plan: Plan

    @property
    def extrapolated(self) -> bool:
        return self.laps_past_cap > 0


@dataclass(frozen=True)
class LapRange:
    """Inclusive lap range."""

    start_lap: int
    end_lap: int


@dataclass(frozen=True)
class PitWindowResult:
    """Candidate curve, recommended window, the optimal plan and the assumptions used.

    ``replacement_compound`` is the set fitted at the optimal stop (the configured replacement
    when there is no candidate).
    """

    candidates: tuple[PitCandidate, ...]
    window: LapRange | None
    optimal_lap: int | None
    stay_out_feasible: bool
    replacement_compound: Compound
    pit_loss_s: float
    warmup_s: float
    plan: Plan = ()


@dataclass(frozen=True)
class StrategyInputs:
    """Strategy assumptions applied to one decision.

    ``replacement_compound`` is the set fitted in :func:`counterfactual`; with
    ``compound_policy="fixed_first"`` it is also forced at the first stop of every plan,
    otherwise every stint's compound is optimised.
    """

    replacement_compound: Compound
    stay_out_feasible: bool
    pit_loss_s: float
    warmup_s: float
    margin_sigma: float
    fuel_s_per_lap: float
    compound_policy: CompoundPolicy = "free"


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


@dataclass(frozen=True)
class ContinuationLayer:
    """Cheapest run to the flag from a fresh set fitted for lap ``s`` with ``k`` stops left.

    Arrays are indexed ``[rule_met, s]``: ``rule_met`` = 1 when the two-compound rule was
    already met before this stint. ``compound`` indexes the tables passed to
    :func:`continuation_layers`; ``next_stop`` is the in-lap of the next stop (-1 = none).
    """

    cost_s: FloatArray
    compound: IntArray
    next_stop: IntArray


def _stint_options(
    table: FreshSetTable,
    rule_met_after: bool,
    previous: ContinuationLayer | None,
    end_lap: int,
    stop_cost_s: float,
) -> tuple[FloatArray, IntArray]:
    """Per start lap: cheapest cost of this set then (finish | box at q and continue)."""
    cum = np.where(np.isnan(table.cum_time_s), np.inf, table.cum_time_s)
    finish = cum[:, end_lap] if rule_met_after else np.full(end_lap + 1, np.inf)
    if previous is None:
        return finish, np.full(end_lap + 1, -1, dtype=np.int64)
    follow = previous.cost_s[int(rule_met_after), 1 : end_lap + 1]
    stop_matrix = cum[:, :end_lap] + stop_cost_s + follow[None, :]
    best_q = np.argmin(stop_matrix, axis=1).astype(np.int64)
    best_stop = stop_matrix[np.arange(end_lap + 1), best_q]
    stop_wins = best_stop < finish
    return np.where(stop_wins, best_stop, finish), np.where(stop_wins, best_q, -1)


def continuation_layers(
    tables: list[FreshSetTable],
    current: Compound,
    end_lap: int,
    stop_cost_s: float,
    max_stops: int,
) -> list[ContinuationLayer]:
    """Dynamic programme over (stops left, rule met, start lap); layer k allows k more stops.

    Costs are prefix sums from the fresh-set tables, so each layer is a vectorised min over
    the next in-lap q (O(T^2) per layer and compound). Ties keep fewer stops and the first
    compound in ``tables`` order.
    """
    layers: list[ContinuationLayer] = []
    for _ in range(max_stops + 1):
        previous = layers[-1] if layers else None
        cost = np.full((2, end_lap + 1), np.inf)
        compound = np.zeros((2, end_lap + 1), dtype=np.int64)
        next_stop = np.full((2, end_lap + 1), -1, dtype=np.int64)
        for met in (0, 1):
            for index, table in enumerate(tables):
                after = bool(met) or table.compound != current
                option, q = _stint_options(table, after, previous, end_lap, stop_cost_s)
                better = option < cost[met]
                cost[met] = np.where(better, option, cost[met])
                compound[met] = np.where(better, index, compound[met])
                next_stop[met] = np.where(better, q, next_stop[met])
        layers.append(ContinuationLayer(cost, compound, next_stop))
    return layers


def _first_stint(
    layers: list[ContinuationLayer],
    tables: list[FreshSetTable],
    current: Compound,
    end_lap: int,
    stop_cost_s: float,
    fixed: Compound | None,
) -> ContinuationLayer:
    """The top layer, or the same layer with the first set's compound forced to ``fixed``."""
    if fixed is None:
        return layers[-1]
    index = next(i for i, t in enumerate(tables) if t.compound == fixed)
    previous = layers[-2] if len(layers) > 1 else None
    options = [
        _stint_options(tables[index], bool(met) or fixed != current, previous, end_lap, stop_cost_s)
        for met in (0, 1)
    ]
    return ContinuationLayer(
        np.vstack([o[0] for o in options]),
        np.full((2, end_lap + 1), index, dtype=np.int64),
        np.vstack([o[1] for o in options]),
    )


def _follow_plan(
    first: ContinuationLayer,
    layers: list[ContinuationLayer],
    tables: list[FreshSetTable],
    current: Compound,
    rule_met: bool,
    pit_lap: int,
    end_lap: int,
) -> tuple[Plan, int]:
    """Stops and beyond-cap laps of the cheapest continuation after boxing at ``pit_lap``."""
    stops: list[PlannedStop] = []
    past_cap, in_lap, start, met = 0, pit_lap, pit_lap + 1, int(rule_met)
    layer, depth = first, len(layers) - 1
    while True:
        table = tables[int(layer.compound[met, start])]
        next_stop = int(layer.next_stop[met, start])
        stops.append(PlannedStop(in_lap, _dry(table.compound)))
        past_cap += int(table.cum_past_cap[start, next_stop if next_stop >= 0 else end_lap])
        met = int(bool(met) or table.compound != current)
        if next_stop < 0:
            return tuple(stops), past_cap
        depth -= 1
        layer, in_lap, start = layers[depth], next_stop, next_stop + 1


def _dry(value: str) -> Compound:
    for compound in DRY_COMPOUNDS:
        if compound == value:
            return compound
    raise ValueError(f"{value!r} is not a dry compound")


@dataclass(frozen=True)
class PitCosts:
    """Plan cost of every candidate in-lap, before any uncertainty margin is applied."""

    total_laps: int
    pit_laps: tuple[int, ...]
    cost_pit_s: tuple[float, ...]
    cost_stay_out_s: float
    laps_past_cap: tuple[int, ...]
    plans: tuple[Plan, ...]
    stay_out_feasible: bool


def pit_costs(model: Forecaster, state: pd.Series, inputs: StrategyInputs) -> PitCosts:
    """Cheapest plan (and its cost) for boxing at each lap after the decision lap."""
    total_laps = int(state["total_laps"])
    decision_lap = int(state["lap"])
    stay = stay_out_rollout(model, state, total_laps, inputs.fuel_s_per_lap)
    pit_laps = tuple(range(decision_lap + 1, total_laps))
    if not pit_laps:
        return PitCosts(
            total_laps, (), (), float(stay.lap_time_s.sum()), (), (), inputs.stay_out_feasible
        )
    current = _dry(str(state["compound"]))
    stop_cost = inputs.pit_loss_s + inputs.warmup_s
    tables = fresh_set_tables(
        model, state, DRY_COMPOUNDS, decision_lap + 2, total_laps, inputs.fuel_s_per_lap
    )
    layers = continuation_layers(tables, current, total_laps, stop_cost, MAX_EXTRA_STOPS)
    fixed = inputs.replacement_compound if inputs.compound_policy == "fixed_first" else None
    met = inputs.stay_out_feasible
    first = _first_stint(layers, tables, current, total_laps, stop_cost, fixed)
    stay_cum = np.cumsum(stay.lap_time_s)
    stay_past = np.cumsum(stay.extrapolated)
    costs, past, plans = [], [], []
    for p in pit_laps:
        before = p - decision_lap - 1
        plan, plan_past = _follow_plan(first, layers, tables, current, met, p, total_laps)
        costs.append(float(stay_cum[before]) + stop_cost + float(first.cost_s[int(met), p + 1]))
        past.append(int(stay_past[before]) + plan_past)
        plans.append(plan)
    return PitCosts(
        total_laps,
        pit_laps,
        tuple(costs),
        float(stay_cum[-1]),
        tuple(past),
        tuple(plans),
        inputs.stay_out_feasible,
    )


@dataclass(frozen=True)
class WindowDecision:
    """Candidates with margins applied, the window around the optimum and the optimum."""

    candidates: tuple[PitCandidate, ...]
    window: LapRange | None
    optimal_lap: int | None


def window_from_costs(costs: PitCosts, sigma: SigmaCurve, margin_sigma: float) -> WindowDecision:
    """Apply the ``margin_sigma`` x cumulative-sigma margin to a cost curve."""
    if not costs.pit_laps:
        return WindowDecision((), None, None)
    best = int(np.argmin(costs.cost_pit_s))
    optimal_lap, optimal_cost = costs.pit_laps[best], costs.cost_pit_s[best]
    candidates = []
    for p, cost, past, plan in zip(
        costs.pit_laps, costs.cost_pit_s, costs.laps_past_cap, costs.plans, strict=True
    ):
        advantage = costs.cost_stay_out_s - cost
        margin = margin_sigma * sigma.cumulative_sigma(max(1, abs(p - optimal_lap)))
        beats_stay_out = advantage > margin_sigma * sigma.cumulative_sigma(costs.total_laps - p)
        near_optimum = cost - optimal_cost <= margin
        in_window = near_optimum and (beats_stay_out or not costs.stay_out_feasible)
        candidates.append(
            PitCandidate(p, cost, costs.cost_stay_out_s, advantage, margin, in_window, past, plan)
        )
    window = _window_around(candidates, best) if candidates[best].in_window else None
    return WindowDecision(tuple(candidates), window, optimal_lap)


def evaluate_pit_window(
    model: Forecaster, state: pd.Series, sigma: SigmaCurve, inputs: StrategyInputs
) -> PitWindowResult:
    """Cost curve over every candidate in-lap after the decision lap."""
    decision = window_from_costs(pit_costs(model, state, inputs), sigma, inputs.margin_sigma)
    best = next((c for c in decision.candidates if c.pit_lap == decision.optimal_lap), None)
    plan = best.plan if best is not None else ()
    return PitWindowResult(
        decision.candidates,
        decision.window,
        decision.optimal_lap,
        inputs.stay_out_feasible,
        plan[0].compound if plan else inputs.replacement_compound,
        inputs.pit_loss_s,
        inputs.warmup_s,
        plan,
    )


def _window_around(candidates: list[PitCandidate], index: int) -> LapRange:
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
    model: Forecaster, state: pd.Series, pit_lap: int, inputs: StrategyInputs
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
