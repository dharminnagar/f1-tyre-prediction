import math
from pathlib import Path

import pandas as pd
import pytest

from boxbox.ingestion import download as download_module
from boxbox.ingestion.kaggle_client import KaggleClient, KaggleUnavailableError
from boxbox.ingestion.tables import best_quali_seconds, parse_lap_time_string
from boxbox.preprocessing.tyre_layer import normalise_compound, track_status_by_lap
from boxbox.synthetic import synthetic_race


@pytest.mark.parametrize(
    ("raw", "expected"), [("1:26.572", 86.572), ("86.5", 86.5), ("\\N", math.nan), (None, math.nan)]
)
def test_parse_lap_time_string(raw: object, expected: float) -> None:
    parsed = parse_lap_time_string(raw)
    assert (math.isnan(parsed) and math.isnan(expected)) or parsed == pytest.approx(expected)


def test_best_quali_takes_fastest_session() -> None:
    quali = pd.DataFrame({"q1": ["1:20.0"], "q2": ["1:19.5"], "q3": ["\\N"]})
    assert best_quali_seconds(quali, ["q1", "q2", "q3"]).iloc[0] == pytest.approx(79.5)


def test_kaggle_without_credentials_reports_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(key, raising=False)
    client = KaggleClient(tmp_path / "missing.env")
    assert not client.has_credentials()
    with pytest.raises(KaggleUnavailableError):
        client.download_dataset("owner/dataset", tmp_path / "out")


def test_historical_tables_fall_back_to_ergast_when_kaggle_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from boxbox.config import load_config

    tables, _ = synthetic_race()

    def kaggle_fails(*_: object) -> None:
        raise KaggleUnavailableError("no credentials")

    class FakeErgast:
        def __init__(self, _: Path) -> None: ...

        def race_tables(self, season: int, race_name: str) -> object:
            return tables

    monkeypatch.setattr(download_module, "_jtrotman_dump", kaggle_fails)
    monkeypatch.setattr(download_module, "ErgastClient", FakeErgast)
    paths = download_module.DatasetPaths(tmp_path)
    result, source = download_module._historical_tables(load_config("subset"), paths, 2023)
    assert result is tables
    assert source == "ergast (kaggle fallback)"
    assert "falling back to Ergast" in caplog.text


def test_compound_aliases_fold_rainbow_names() -> None:
    raw = pd.Series(["SUPERSOFT", "medium", "HARD", "WET", "TEST_UNKNOWN"])
    assert normalise_compound(raw).tolist()[:4] == ["SOFT", "MEDIUM", "HARD", "WET"]
    assert pd.isna(normalise_compound(raw).iloc[4])


def test_track_status_from_race_control_messages() -> None:
    messages = pd.DataFrame(
        [
            {
                "date": "1",
                "lap_number": 3,
                "category": "SafetyCar",
                "message": "SAFETY CAR DEPLOYED",
            },
            {
                "date": "2",
                "lap_number": 5,
                "category": "SafetyCar",
                "message": "SAFETY CAR IN THIS LAP",
            },
            {
                "date": "3",
                "lap_number": 8,
                "category": "SafetyCar",
                "message": "VIRTUAL SAFETY CAR DEPLOYED",
            },
            {
                "date": "4",
                "lap_number": 9,
                "category": "SafetyCar",
                "message": "VIRTUAL SAFETY CAR ENDING",
            },
            {
                "date": "5",
                "lap_number": 6,
                "category": "Flag",
                "flag": "YELLOW",
                "scope": "Sector",
                "message": "YELLOW IN TRACK SECTOR 4",
            },
        ]
    )
    status = track_status_by_lap(messages, 10)
    assert status[2] == "1"
    assert status[3] == status[5] == "4"
    assert status[6] == "1"
    assert status[8] == "6"
    assert status[9] == "67"
