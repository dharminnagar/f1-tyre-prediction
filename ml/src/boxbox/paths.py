"""Filesystem layout shared by the pipeline and the API."""

import os
from dataclasses import dataclass
from pathlib import Path

from boxbox.config import Profile

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DatasetPaths:
    """Locations of raw, interim and processed data."""

    root: Path

    @property
    def fastf1_cache(self) -> Path:
        return self.root / "raw" / "fastf1_cache"

    @property
    def ergast(self) -> Path:
        return self.root / "raw" / "ergast"

    @property
    def kaggle_race_data(self) -> Path:
        return self.root / "raw" / "kaggle" / "formula-1-race-data"

    @property
    def kaggle_fastf1_export(self) -> Path:
        return self.root / "raw" / "kaggle" / "fastf1-export-2018-2022"

    @property
    def openf1(self) -> Path:
        return self.root / "raw" / "openf1"

    @property
    def sources_record(self) -> Path:
        return self.root / "raw" / "sources.json"

    @property
    def f1db(self) -> Path:
        return self.root / "raw" / "f1db"

    @property
    def interim(self) -> Path:
        return self.root / "interim"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    def interim_laps(self, season: int) -> Path:
        return self.interim / f"laps_{season}.parquet"

    def interim_table(self, season: int, table: str) -> Path:
        return self.interim / f"{table}_{season}.parquet"

    def processed_file(self, profile: Profile, name: str) -> Path:
        return self.processed / profile / f"{name}.parquet"


@dataclass(frozen=True)
class ArtifactPaths:
    """Locations of one profile's exported artifacts (the ml -> api contract)."""

    root: Path
    profile: Profile

    @property
    def profile_dir(self) -> Path:
        return self.root / self.profile

    @property
    def manifest(self) -> Path:
        return self.profile_dir / "manifest.json"

    @property
    def models(self) -> Path:
        return self.profile_dir / "models"

    @property
    def data(self) -> Path:
        return self.profile_dir / "data"

    @property
    def eval(self) -> Path:
        return self.profile_dir / "eval"

    @property
    def figures(self) -> Path:
        return self.profile_dir / "figures"

    @property
    def combined_metrics(self) -> Path:
        return self.root / "metrics.json"


def dataset_paths() -> DatasetPaths:
    """Dataset paths, overridable with ``BOXBOX_DATASET_DIR``."""
    return DatasetPaths(Path(os.environ.get("BOXBOX_DATASET_DIR", PROJECT_ROOT / "dataset")))


def artifacts_root() -> Path:
    """Artifacts root, overridable with ``BOXBOX_ARTIFACTS_DIR``."""
    return Path(os.environ.get("BOXBOX_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))


def artifact_paths(profile: Profile) -> ArtifactPaths:
    """Artifact paths for one profile."""
    return ArtifactPaths(artifacts_root(), profile)
