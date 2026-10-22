"""Pace normalisation: fuel correction, causal field reference and driver offset."""

import numpy as np
import pandas as pd


def fuel_corrected(laps: pd.DataFrame, fuel_s_per_lap: float) -> pd.DataFrame:
    """Add ``fuel_term_s`` and ``lap_fc_s = lap_time_s - fuel_term_s``.

    ``fuel_term_s = fuel_s_per_lap * (total_laps - lap)`` is the remaining-fuel time penalty.
    Refuelling-era laps (2005-2009) get no correction since fuel load was not monotone.
    """
    remaining = (laps["total_laps"] - laps["lap"]).clip(lower=0)
    term = np.where(laps["era"] == "refuel_2005_2009", 0.0, fuel_s_per_lap * remaining)
    return laps.assign(fuel_term_s=term, lap_fc_s=laps["lap_time_s"] - term)


def _expanding_median_by_lap(values: pd.DataFrame, value_col: str, max_lap: int) -> pd.Series:
    """Median of ``value_col`` over rows with lap <= L, for each L in 1..max_lap."""
    out: dict[int, float] = {}
    pool: list[float] = []
    by_lap = values.groupby("lap")[value_col].apply(list).to_dict()
    for lap in range(1, max_lap + 1):
        pool.extend(float(v) for v in by_lap.get(lap, []))
        out[lap] = float(np.median(pool)) if pool else float("nan")
    return pd.Series(out)


def field_reference(laps: pd.DataFrame) -> pd.DataFrame:
    """Add the causal field reference pace.

    ``s_field_ref_next_s`` is the median clean ``lap_fc_s`` up to and including this lap;
    ``field_ref_s`` is the same quantity as of the previous lap (known before lap L starts).
    """
    parts = []
    for race_id, race in laps.groupby("race_id", sort=False):
        clean = race[race["is_clean"]]
        max_lap = int(race["lap"].max())
        through = _expanding_median_by_lap(clean, "lap_fc_s", max_lap).ffill()
        frame = pd.DataFrame(
            {
                "race_id": race_id,
                "lap": through.index,
                "s_field_ref_next_s": through.to_numpy(),
                "field_ref_s": through.shift(1).to_numpy(),
            }
        )
        parts.append(frame)
    lookup = pd.concat(parts, ignore_index=True)
    merged = laps.merge(lookup, on=["race_id", "lap"], how="left")
    # Lap 2 has no clean history yet (lap 1 is never clean): fall back to the first known value.
    merged["field_ref_s"] = merged["field_ref_s"].fillna(merged["s_field_ref_next_s"])
    return merged


def relative_pace(laps: pd.DataFrame) -> pd.DataFrame:
    """Add the modelling target ``y_s = lap_fc_s - field_ref_s``."""
    return laps.assign(y_s=laps["lap_fc_s"] - laps["field_ref_s"])


def driver_offset(laps: pd.DataFrame) -> pd.DataFrame:
    """Add ``s_driver_offset_s``: expanding median of the driver's clean ``y_s`` to this lap."""
    ordered = laps.sort_values(["race_id", "driver_id", "lap"])
    clean_y = ordered["y_s"].where(ordered["is_clean"])
    offset = clean_y.groupby([ordered["race_id"], ordered["driver_id"]]).transform(
        lambda s: s.expanding().median().ffill()
    )
    return ordered.assign(s_driver_offset_s=offset)


def quali_gap(laps: pd.DataFrame, cap_s: float = 5.0) -> pd.DataFrame:
    """Add ``quali_gap_s``: best qualifying lap minus the race's fastest (capped)."""
    pole = laps.groupby("race_id")["quali_best_s"].transform("min")
    return laps.assign(quali_gap_s=(laps["quali_best_s"] - pole).clip(upper=cap_s))
