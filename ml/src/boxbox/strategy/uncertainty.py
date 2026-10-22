"""Rollout uncertainty: per-horizon and cumulative error spread measured on validation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SigmaCurve:
    """Error spread of k-lap-ahead forecasts.

    ``per_lap[k-1]`` is the std of the k-step lap error and ``cumulative[k-1]`` the std of the
    summed error over laps 1..k. Beyond the measured horizon the cumulative curve follows the
    fitted power law ``a * k**b``.
    """

    per_lap: tuple[float, ...]
    cumulative: tuple[float, ...]
    power_a: float
    power_b: float

    def lap_sigma(self, k: int) -> float:
        """Std of a single-lap error k laps ahead."""
        index = min(max(k, 1), len(self.per_lap)) - 1
        return self.per_lap[index]

    def cumulative_sigma(self, k: int) -> float:
        """Std of the cumulative error over k laps."""
        if k <= 0:
            return 0.0
        if k <= len(self.cumulative):
            return self.cumulative[k - 1]
        return float(self.power_a * k**self.power_b)


def fit_sigma_curve(errors: pd.DataFrame, min_samples: int = 20) -> SigmaCurve:
    """Fit from rollout errors with columns ``decision_id``, ``k`` and ``error_s``.

    Cumulative errors only use decisions whose first k laps were all evaluable.
    """
    ordered = errors.sort_values(["decision_id", "k"])
    ordered = ordered.assign(cum=ordered.groupby("decision_id")["error_s"].cumsum())
    contiguous = ordered.groupby("decision_id")["k"].transform(
        lambda s: s.to_numpy() == np.arange(1, len(s) + 1)
    )
    contiguous_rows = ordered[contiguous.astype(bool)]
    per_lap_stats = ordered.groupby("k")["error_s"].agg(["std", "size"])
    cum_stats = contiguous_rows.groupby("k")["cum"].agg(["std", "size"])
    per_lap = tuple(float(v) for v in per_lap_stats[per_lap_stats["size"] >= min_samples]["std"])
    cum = cum_stats[cum_stats["size"] >= min_samples]["std"]
    cumulative = tuple(float(v) for v in cum)
    ks = np.asarray(cum.index, dtype=np.float64)
    if len(ks) >= 2:
        b, log_a = np.polyfit(np.log(ks), np.log(np.asarray(cumulative)), 1)
        power_a, power_b = float(np.exp(log_a)), float(b)
    else:
        first = cumulative[0] if cumulative else 1.0
        power_a, power_b = first, 0.5
    return SigmaCurve(per_lap or (1.0,), cumulative or (power_a,), power_a, power_b)
