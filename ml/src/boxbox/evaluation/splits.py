"""Race-level train/val/test assignment (never split laps of one race)."""

import pandas as pd

from boxbox.config import Config


def split_for_season(season: int, cfg: Config) -> str:
    """Split name of a season's race (``unused`` if not in any split)."""
    if season in cfg.splits.train:
        return "train"
    if season in cfg.splits.val:
        return "val"
    if season in cfg.splits.test:
        return "test"
    return "unused"


def assign_splits(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add a ``split`` column from the race season."""
    return df.assign(split=df["season"].map(lambda s: split_for_season(int(s), cfg)))
