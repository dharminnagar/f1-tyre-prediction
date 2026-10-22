"""Pit-lane time loss estimated from green-flag stops in training races."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PitLossEstimate:
    """Median loss across stops and its spread."""

    median_s: float
    q25_s: float
    q75_s: float
    n_stops: int


def stop_losses(laps: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Per green-flag stop: ``(in-lap + out-lap) - 2 * median(clean laps within +-window)``.

    A stop is green when neither the in-lap nor the out-lap was neutralised.
    """
    rows = []
    for (race_id, driver_id), driver in laps.groupby(["race_id", "driver_id"]):
        by_lap = driver.set_index("lap")
        for in_lap in driver.loc[driver["pit_in"], "lap"]:
            out_lap = in_lap + 1
            if out_lap not in by_lap.index:
                continue
            pair = by_lap.loc[[in_lap, out_lap]]
            if pair["track_neutralised"].any() or pair["lap_time_s"].isna().any():
                continue
            near = by_lap[
                (by_lap.index >= in_lap - window)
                & (by_lap.index <= out_lap + window)
                & by_lap["is_clean"]
            ]["lap_time_s"]
            if len(near) < 2:
                continue
            loss = float(pair["lap_time_s"].sum() - 2 * near.median())
            rows.append({"race_id": race_id, "driver_id": driver_id, "lap": in_lap, "loss_s": loss})
    return pd.DataFrame(rows, columns=["race_id", "driver_id", "lap", "loss_s"])


def estimate_pit_loss(laps: pd.DataFrame, race_ids: list[str]) -> PitLossEstimate:
    """Median green-flag pit loss over ``race_ids``."""
    losses = stop_losses(laps[laps["race_id"].isin(race_ids)])["loss_s"].to_numpy()
    if len(losses) == 0:
        raise ValueError("no green-flag stops to estimate pit loss")
    q25, median, q75 = np.quantile(losses, [0.25, 0.5, 0.75])
    return PitLossEstimate(float(median), float(q25), float(q75), len(losses))
