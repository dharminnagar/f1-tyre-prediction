"""OpenF1 connector (https://openf1.org): official timing feed incl. stints/compounds, 2023+."""

import json
import logging
import time
from pathlib import Path
from typing import Final

import requests

logger = logging.getLogger(__name__)

BASE_URL: Final[str] = "https://api.openf1.org/v1"
SESSION_ENDPOINTS: Final[tuple[str, ...]] = (
    "laps",
    "stints",
    "pit",
    "race_control",
    "weather",
    "drivers",
)
OPENF1_MIN_SEASON: Final[int] = 2023
JsonRows = list[dict[str, object]]


class OpenF1Client:
    """Fetches and caches raw OpenF1 JSON for a race session."""

    def __init__(self, raw_dir: Path, pause_s: float = 0.4, retries: int = 4) -> None:
        self._raw_dir = raw_dir
        self._pause_s = pause_s
        self._retries = retries
        self._session = requests.Session()

    def _get(self, endpoint: str, params: dict[str, str | int]) -> JsonRows:
        for attempt in range(self._retries):
            response = self._session.get(f"{BASE_URL}/{endpoint}", params=params, timeout=60)
            time.sleep(self._pause_s)
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(2.0 * (attempt + 1))
                continue
            response.raise_for_status()
            payload: object = response.json()
            if not isinstance(payload, list):
                return []
            return [{str(k): v for k, v in row.items()} for row in payload if isinstance(row, dict)]
        raise RuntimeError(f"OpenF1 {endpoint} failed after {self._retries} attempts")

    def race_session_key(self, season: int, circuit_short_name: str) -> int:
        """Session key of the race at ``circuit_short_name`` in ``season``."""
        sessions = self._get(
            "sessions",
            {"year": season, "circuit_short_name": circuit_short_name, "session_name": "Race"},
        )
        if not sessions:
            raise LookupError(f"no OpenF1 race at {circuit_short_name} in {season}")
        return int(str(sessions[0]["session_key"]))

    def season_dir(self, season: int) -> Path:
        return self._raw_dir / str(season)

    def download_race(self, season: int, circuit_short_name: str) -> Path:
        """Download every session endpoint once (cached on disk); returns the season directory."""
        target = self.season_dir(season)
        target.mkdir(parents=True, exist_ok=True)
        if all((target / f"{e}.json").exists() for e in SESSION_ENDPOINTS):
            return target
        session_key = self.race_session_key(season, circuit_short_name)
        for endpoint in SESSION_ENDPOINTS:
            rows = self._get(endpoint, {"session_key": session_key})
            (target / f"{endpoint}.json").write_text(json.dumps(rows))
            logger.info("openf1 %s %s: %d rows", season, endpoint, len(rows))
        return target


def read_openf1_endpoint(season_dir: Path, endpoint: str) -> JsonRows:
    """Read one cached OpenF1 endpoint file."""
    payload: object = json.loads((season_dir / f"{endpoint}.json").read_text())
    if not isinstance(payload, list):
        return []
    return [{str(k): v for k, v in row.items()} for row in payload if isinstance(row, dict)]
