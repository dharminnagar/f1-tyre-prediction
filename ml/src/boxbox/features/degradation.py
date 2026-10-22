"""Causal within-stint degradation state from clean laps up to and including each lap."""

import numpy as np
import pandas as pd

from boxbox.schema import STINT_KEYS


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(x) == 0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def _stint_state(stint: pd.DataFrame) -> pd.DataFrame:
    ages: list[float] = []
    ys: list[float] = []
    rows: list[dict[str, float]] = []
    previous_slope_5 = float("nan")
    for age, y, clean in zip(stint["tyre_age"], stint["y_s"], stint["is_clean"], strict=True):
        if clean and not np.isnan(y):
            ages.append(float(age))
            ys.append(float(y))
        x_arr, y_arr = np.asarray(ages), np.asarray(ys)
        slope_3 = _slope(x_arr[-3:], y_arr[-3:])
        slope_5 = _slope(x_arr[-5:], y_arr[-5:])
        rows.append(
            {
                "s_deg_slope_3": slope_3,
                "s_deg_slope_5": slope_5,
                "s_deg_accel": slope_5 - previous_slope_5,
                "s_rolling_mean_3": float(np.mean(y_arr[-3:])) if len(y_arr) else float("nan"),
                "s_rolling_mean_5": float(np.mean(y_arr[-5:])) if len(y_arr) else float("nan"),
                "s_lap_delta": float(y_arr[-1] - y_arr[-2]) if len(y_arr) >= 2 else float("nan"),
            }
        )
        if not np.isnan(slope_5):
            previous_slope_5 = slope_5
    return pd.DataFrame(rows, index=stint.index)


def degradation_state(laps: pd.DataFrame) -> pd.DataFrame:
    """Add ``s_deg_slope_3/5`` (s/lap of tyre age), ``s_deg_accel``, rolling means, lap delta.

    All are computed on clean laps of the current stint up to and including the row's lap.
    """
    ordered = laps.sort_values([*STINT_KEYS, "lap"])
    states = [_stint_state(stint) for _, stint in ordered.groupby(STINT_KEYS, sort=False)]
    return ordered.join(pd.concat(states))
