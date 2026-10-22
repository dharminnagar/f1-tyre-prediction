"""Read-only artifact store loaded once at startup, plus request-time lookups."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from boxbox.config import PROFILES, Config, Profile
from boxbox.export.artifacts import MODEL_FILE_SUFFIX, SCHEMA_VERSION
from boxbox.models.types import PaceModel
from boxbox.schema import MODEL_NAMES, ModelName
from boxbox.strategy.pit_loss import PitLossEstimate
from boxbox.strategy.uncertainty import SigmaCurve

JsonObject = dict[str, object]


class ArtifactError(RuntimeError):
    """Artifacts missing, incomplete or from an incompatible schema version."""


class NotFoundError(LookupError):
    """A requested race/driver/lap/profile/model does not exist."""


@dataclass(frozen=True)
class ProfileStore:
    """Everything the API serves for one training profile."""

    profile: Profile
    manifest: JsonObject
    config: Config
    models: dict[ModelName, PaceModel]
    model_meta: dict[ModelName, JsonObject]
    selected_model: ModelName
    laps: pd.DataFrame
    curves: pd.DataFrame
    metrics: JsonObject
    predictions: pd.DataFrame
    error_groups: pd.DataFrame
    threshold_sensitivity: pd.DataFrame
    pit_timing: pd.DataFrame
    sigma: dict[ModelName, SigmaCurve]
    pit_loss: PitLossEstimate
    races: list[JsonObject]


@dataclass(frozen=True)
class Store:
    """All loaded profiles and the default one."""

    profiles: dict[Profile, ProfileStore]
    default_profile: Profile

    def get(self, profile: Profile | None) -> ProfileStore:
        """The requested profile, or the default when ``None``."""
        key = profile or self.default_profile
        if key not in self.profiles:
            raise NotFoundError(f"profile '{key}' has no artifacts")
        return self.profiles[key]


def _object(value: object, what: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ArtifactError(f"{what} must be a JSON object")
    return {str(k): v for k, v in value.items()}


def _list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _model_name(value: object) -> ModelName:
    for name in MODEL_NAMES:
        if name == value:
            return name
    raise ArtifactError(f"unknown model name {value!r}")


def _sigma(raw: JsonObject) -> SigmaCurve:
    per_lap = tuple(float(str(v)) for v in _list(raw.get("per_lap")))
    cumulative = tuple(float(str(v)) for v in _list(raw.get("cumulative")))
    return SigmaCurve(per_lap, cumulative, float(str(raw["power_a"])), float(str(raw["power_b"])))


def _load_profile(profile_dir: Path, profile: Profile) -> ProfileStore:
    manifest = _object(json.loads((profile_dir / "manifest.json").read_text()), "manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(
            f"{profile}: schema {manifest.get('schema_version')} != expected {SCHEMA_VERSION}"
        )
    listed = {str(f) for f in _list(manifest.get("files"))}

    def path(relative: str) -> Path:
        # Only files the pipeline itself listed in the manifest are ever opened (joblib = pickle).
        if relative not in listed:
            raise ArtifactError(f"{profile}: {relative} not listed in manifest")
        return profile_dir / relative

    models: dict[ModelName, PaceModel] = {}
    meta: dict[ModelName, JsonObject] = {}
    for entry in _list(manifest.get("models")):
        info = _object(entry, "model entry")
        name = _model_name(info["name"])
        models[name] = joblib.load(path(f"models/{name}{MODEL_FILE_SUFFIX}"))
        meta[name] = info
    sigma_raw = _object(manifest.get("sigma"), "sigma")
    pit_loss = _object(manifest.get("pit_loss"), "pit_loss")
    return ProfileStore(
        profile=profile,
        manifest=manifest,
        config=Config.model_validate(manifest["config"]),
        models=models,
        model_meta=meta,
        selected_model=_model_name(manifest["selected_model"]),
        laps=pd.read_parquet(path("data/laps.parquet")),
        curves=pd.read_parquet(path("data/degradation_curves.parquet")),
        metrics=_object(json.loads(path("eval/metrics.json").read_text()), "metrics"),
        predictions=pd.read_parquet(path("eval/predictions.parquet")),
        error_groups=pd.read_parquet(path("eval/error_groups.parquet")),
        threshold_sensitivity=pd.read_parquet(path("eval/threshold_sensitivity.parquet")),
        pit_timing=pd.read_parquet(path("eval/pit_timing.parquet")),
        sigma={_model_name(k): _sigma(_object(v, "sigma")) for k, v in sigma_raw.items()},
        pit_loss=PitLossEstimate(
            float(str(pit_loss["median_s"])),
            float(str(pit_loss["q25_s"])),
            float(str(pit_loss["q75_s"])),
            int(str(pit_loss["n_stops"])),
        ),
        races=[_object(r, "race") for r in _list(manifest.get("races"))],
    )


def load_store(artifacts_dir: Path, default_profile: Profile) -> Store:
    """Load every profile that has a manifest; the default falls back to any loaded profile."""
    profiles = {
        profile: _load_profile(artifacts_dir / profile, profile)
        for profile in PROFILES
        if (artifacts_dir / profile / "manifest.json").exists()
    }
    if not profiles:
        raise ArtifactError(f"no artifacts under {artifacts_dir}; run `make pipeline`")
    default = default_profile if default_profile in profiles else next(iter(profiles))
    return Store(profiles, default)


def records(frame: pd.DataFrame) -> list[JsonObject]:
    """Rows as dicts with NaN/NA mapped to None (JSON/pydantic friendly)."""
    return [
        {str(k): (None if pd.isna(v) else v) for k, v in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def servable_races(store: ProfileStore) -> list[JsonObject]:
    """Races with observed tyre data (strategy needs real compounds)."""
    return [r for r in store.races if r.get("servable") is True]


def race_laps(store: ProfileStore, race_id: str) -> pd.DataFrame:
    """Feature rows of a servable race."""
    if race_id not in {str(r["race_id"]) for r in servable_races(store)}:
        raise NotFoundError(f"race '{race_id}' not found")
    return store.laps[store.laps["race_id"] == race_id]


def driver_laps(store: ProfileStore, race_id: str, driver_id: str) -> pd.DataFrame:
    """Feature rows of one driver in a servable race, ordered by lap."""
    laps = race_laps(store, race_id)
    rows = laps[laps["driver_id"] == driver_id].sort_values("lap")
    if rows.empty:
        raise NotFoundError(f"driver '{driver_id}' not in race '{race_id}'")
    return rows


def lap_state(store: ProfileStore, race_id: str, driver_id: str, lap: int) -> pd.Series:
    """The decision-lap state row."""
    rows = driver_laps(store, race_id, driver_id)
    match = rows[rows["lap"] == lap]
    if match.empty:
        raise NotFoundError(f"lap {lap} not found for '{driver_id}' in '{race_id}'")
    state: pd.Series = match.iloc[0]
    return state


def resolve_model(store: ProfileStore, name: ModelName | None) -> tuple[ModelName, PaceModel]:
    """Requested model or the profile's selected model."""
    key = name or store.selected_model
    if key not in store.models:
        raise NotFoundError(f"model '{key}' not available")
    return key, store.models[key]
