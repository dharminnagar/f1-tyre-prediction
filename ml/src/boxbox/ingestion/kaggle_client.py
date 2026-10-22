"""Kaggle connector: runs the kaggle CLI with credentials loaded from ``.env`` (never logged)."""

import logging
import os
import shutil
import subprocess
from pathlib import Path

from dotenv import dotenv_values

logger = logging.getLogger(__name__)

TOKEN_KEYS = ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")


class KaggleUnavailableError(RuntimeError):
    """Raised when the kaggle CLI or credentials are missing, or a download fails."""


class KaggleClient:
    """Downloads Kaggle datasets through the ``kaggle`` CLI in an isolated subprocess.

    Credentials are read from ``env_file`` and passed only to the subprocess environment;
    their values are never printed, logged or written anywhere.
    """

    def __init__(self, env_file: Path, cli_path: Path | None = None) -> None:
        self._env_file = env_file
        self._cli_path = cli_path

    def _cli(self) -> str:
        if self._cli_path is not None and self._cli_path.exists():
            return str(self._cli_path)
        found = shutil.which("kaggle")
        if found is None:
            raise KaggleUnavailableError("kaggle CLI not installed (pip install boxbox[kaggle])")
        return found

    def _subprocess_env(self) -> dict[str, str]:
        file_values = dotenv_values(self._env_file) if self._env_file.exists() else {}
        credentials = {k: v for k, v in file_values.items() if k in TOKEN_KEYS and v}
        if not credentials and not any(os.environ.get(k) for k in TOKEN_KEYS):
            raise KaggleUnavailableError("no Kaggle credentials found in .env or environment")
        return {**os.environ, **credentials}

    def has_credentials(self) -> bool:
        """Whether credentials are present (presence check only)."""
        try:
            self._subprocess_env()
        except KaggleUnavailableError:
            return False
        return True

    def download_dataset(self, dataset_ref: str, target_dir: Path) -> list[Path]:
        """Download and unzip ``dataset_ref`` into ``target_dir``; returns the extracted files."""
        target_dir.mkdir(parents=True, exist_ok=True)
        command = [self._cli(), "datasets", "download", dataset_ref, "-p", str(target_dir)]
        result = subprocess.run(
            [*command, "--unzip", "--force"],
            env=self._subprocess_env(),
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
        )
        if result.returncode != 0:
            # stderr from the CLI never contains the token; keep only the tail for diagnosis.
            raise KaggleUnavailableError(f"kaggle download failed: {result.stderr.strip()[-300:]}")
        files = sorted(p for p in target_dir.rglob("*") if p.is_file())
        logger.info("kaggle: downloaded %s (%d files)", dataset_ref, len(files))
        return files
