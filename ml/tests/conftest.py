"""Pytest fixtures built on the synthetic race generator."""

import pandas as pd
import pytest

from boxbox.config import Config, load_config
from boxbox.synthetic import canonical_pipeline, synthetic_race


@pytest.fixture
def cfg() -> Config:
    return load_config("subset")


@pytest.fixture
def synthetic_laps(cfg: Config) -> pd.DataFrame:
    tables, tyre_layer = synthetic_race()
    return canonical_pipeline(tables, tyre_layer, cfg)
