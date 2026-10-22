"""Typed pipeline configuration loaded from ``configs/default.yaml`` plus a profile overlay."""

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

Profile = Literal["subset", "full"]
PROFILES: tuple[Profile, ...] = ("subset", "full")
CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


class Frozen(BaseModel):
    """Immutable pydantic base."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Splits(Frozen):
    """Season-level split assignment (one Spanish GP per season)."""

    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]


class PolyParams(Frozen):
    """Ridge polynomial model hyper-parameters."""

    alpha: float


class GbmParams(Frozen):
    """HistGradientBoosting hyper-parameters."""

    max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    min_samples_leaf: int


class ModelParams(Frozen):
    """Hyper-parameters for every trainable model."""

    poly: PolyParams
    gbm: GbmParams


TyreSource = Literal["fastf1", "openf1", "kaggle_fastf1_export"]


class SourceSettings(Frozen):
    """External dataset identifiers and the tyre-data fallback order."""

    jtrotman_dataset: str
    fastf1_export_dataset: str
    fastf1_export_gp_name: str
    openf1_circuit_short_name: str
    tyre_source_order: tuple[TyreSource, ...]


class Config(Frozen):
    """Full pipeline configuration for one profile."""

    profile: Profile
    event_name: str
    circuit: str
    fastf1_min_season: int
    seasons: tuple[int, ...]
    splits: Splits
    compounds: tuple[str, ...]
    fuel_s_per_lap: float
    fuel_sensitivity_s_per_lap: tuple[float, ...]
    competitive_threshold_s: float
    threshold_sensitivity_s: tuple[float, ...]
    reference_tyre_age: int
    horizon_n: int
    horizon_max_k: int
    warmup_effect_s: float
    warmup_sensitivity_s: tuple[float, ...]
    window_margin_sigma: float
    next_lap_interval_z: float
    traffic_gap_s: float
    outlier_pct: float
    outlier_window_laps: int
    neutralised_field_pct: float
    pit_detection_excess_s: float
    tyre_age_clip_quantile: float
    exclude_refuel_era: bool
    no_tyre_change_seasons: tuple[int, ...]
    historical_source: Literal["kaggle", "ergast"]
    sources: SourceSettings
    models: ModelParams
    random_seed: int

    @model_validator(mode="after")
    def _splits_are_disjoint_subsets(self) -> "Config":
        split_seasons = [*self.splits.train, *self.splits.val, *self.splits.test]
        if len(split_seasons) != len(set(split_seasons)):
            raise ValueError("a season appears in more than one split")
        if not set(split_seasons) <= set(self.seasons):
            raise ValueError("split seasons must be a subset of seasons")
        return self

    @property
    def fastf1_seasons(self) -> tuple[int, ...]:
        return tuple(s for s in self.seasons if s >= self.fastf1_min_season)

    @property
    def historical_seasons(self) -> tuple[int, ...]:
        return tuple(s for s in self.seasons if s < self.fastf1_min_season)

    def config_hash(self) -> str:
        """Stable short hash of the configuration (recorded in the manifest)."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


def _read_yaml(path: Path) -> dict[str, object]:
    loaded: object = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping")
    return {str(k): v for k, v in loaded.items()}


def load_config(profile: Profile = "subset", config_dir: Path = CONFIG_DIR) -> Config:
    """Load ``default.yaml`` and overlay ``<profile>.yaml`` (top-level keys replace defaults)."""
    merged = _read_yaml(config_dir / "default.yaml") | _read_yaml(config_dir / f"{profile}.yaml")
    merged["profile"] = profile
    return Config.model_validate(merged)
