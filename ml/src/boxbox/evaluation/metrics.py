"""Regression metrics as plain functions."""

from dataclasses import dataclass

import numpy as np

from boxbox.models.types import FloatArray


@dataclass(frozen=True)
class RegressionMetrics:
    """MAE/RMSE in seconds, R^2 on relative pace, MAPE in percent of lap time."""

    mae: float
    rmse: float
    r2: float
    mape: float
    n: int


def regression_metrics(
    actual_s: FloatArray, pred_s: FloatArray, actual_rel: FloatArray, pred_rel: FloatArray
) -> RegressionMetrics:
    """Compute MAE/RMSE/MAPE on lap time and R^2 on relative pace.

    R^2 on absolute lap time would be trivially near 1 because it is dominated by the field
    reference, which the model does not predict.
    """
    if len(actual_s) == 0:
        return RegressionMetrics(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    error = pred_s - actual_s
    ss_res = float(np.sum((actual_rel - pred_rel) ** 2))
    ss_tot = float(np.sum((actual_rel - actual_rel.mean()) ** 2))
    return RegressionMetrics(
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        mape=float(np.mean(np.abs(error) / actual_s) * 100),
        n=len(actual_s),
    )
