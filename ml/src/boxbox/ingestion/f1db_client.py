"""Optional F1DB connector: downloads the latest CSV release (metadata layer only)."""

import logging
import zipfile
from pathlib import Path
from typing import Final

import requests

logger = logging.getLogger(__name__)

RELEASE_URL: Final[str] = "https://github.com/f1db/f1db/releases/latest/download/f1db-csv.zip"


class F1DBClient:
    """Downloads and extracts the F1DB CSV release into ``target_dir``."""

    def __init__(self, target_dir: Path) -> None:
        self._target_dir = target_dir

    def download(self) -> list[Path]:
        """Download once (no-op when already extracted); returns the CSV files."""
        self._target_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self._target_dir.glob("*.csv"))
        if existing:
            return existing
        archive = self._target_dir / "f1db-csv.zip"
        with requests.get(RELEASE_URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            with archive.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                # Guard against path traversal in archive member names.
                if Path(member).name == member and member.endswith(".csv"):
                    bundle.extract(member, self._target_dir)
        archive.unlink()
        files = sorted(self._target_dir.glob("*.csv"))
        logger.info("f1db: extracted %d CSV files", len(files))
        return files
