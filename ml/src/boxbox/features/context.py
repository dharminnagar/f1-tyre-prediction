"""Race-context state: position, gaps, traffic, weather and race progress."""

import pandas as pd

GAP_CAP_S = 10.0


def context_state(laps: pd.DataFrame) -> pd.DataFrame:
    """Add ``race_progress`` and end-of-lap context state columns (``s_*``)."""
    return laps.assign(
        race_progress=laps["lap"] / laps["total_laps"],
        s_position=laps["position"],
        s_gap_ahead_s=laps["gap_ahead_s"].clip(lower=0, upper=GAP_CAP_S),
        s_in_traffic=laps["in_traffic"],
        s_air_temp=laps["air_temp"],
        s_track_temp=laps["track_temp"],
    )
