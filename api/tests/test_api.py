import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from boxbox_api.store import ArtifactError, load_store

STATE = "/api/state?race_id=2024_05&driver_id=alpha&lap=8"
WINDOW = {"race_id": "2024_05", "driver_id": "alpha", "lap": 8}


@pytest.mark.parametrize(
    "url",
    [
        "/api/health",
        "/api/profiles",
        "/api/races",
        "/api/races/2024_05/drivers",
        "/api/races/2024_05/drivers/alpha/laps",
        "/api/models",
        STATE,
        "/api/evaluation/metrics",
        "/api/evaluation/predictions?race_id=2024_05&driver_id=alpha&stint_id=1",
        "/api/evaluation/errors?group_by=compound",
        "/api/evaluation/threshold-sensitivity",
        "/api/evaluation/pit-timing",
        "/api/degradation/curves",
        "/api/degradation/curves?compound=SOFT&race_id=2024_05",
    ],
)
def test_get_endpoints_return_200(client: TestClient, url: str) -> None:
    response = client.get(url)
    assert response.status_code == 200, response.text
    assert response.json()


def test_state_contract(client: TestClient) -> None:
    body = client.get(STATE).json()
    assert body["compound"] == "SOFT"
    assert len(body["next_laps"]) == 5
    assert body["next_lap"]["lo_s"] < body["next_lap"]["pred_s"] < body["next_lap"]["hi_s"]
    parts = body["decomposition"]
    total = sum(
        parts[k] for k in ("field_ref_s", "driver_offset_s", "tyre_s", "fuel_s", "context_s")
    )
    assert total == pytest.approx(body["next_lap"]["pred_s"])
    assert body["explanation"]


def test_pit_window_and_overrides(client: TestClient) -> None:
    default = client.post("/api/strategy/pit-window", json=WINDOW).json()
    assert default["actual_pit_laps"] == [12]
    assert default["assumptions"]["pit_loss_source"] == "estimated"
    assert len(default["candidates"]) == 30 - 8 - 1
    cheap = client.post(
        "/api/strategy/pit-window",
        json={**WINDOW, "pit_loss_s": 5.0, "replacement_compound": "HARD"},
    ).json()
    assert cheap["replacement_compound"] == "HARD"
    assert cheap["assumptions"]["pit_loss_source"] == "override"
    assert cheap["candidates"][0]["cost_pit_s"] < default["candidates"][0]["cost_pit_s"]


def test_counterfactual(client: TestClient) -> None:
    body = client.post("/api/strategy/counterfactual", json={**WINDOW, "pit_lap": 12}).json()
    assert body["laps"][0] == 9 and body["laps"][-1] == 30
    assert len(body["stay_out"]) == len(body["pit"]) == len(body["actual"]) == 22


@pytest.mark.parametrize(
    "url",
    [
        "/api/races/1999_01/drivers",
        "/api/races/2024_05/drivers/nobody/laps",
        "/api/state?race_id=2024_05&driver_id=alpha&lap=99",
        "/api/state?race_id=2024_05&driver_id=nobody&lap=8",
        "/api/state?race_id=2024_05&driver_id=alpha&lap=8&profile=full",
        "/api/evaluation/predictions?race_id=2024_05&driver_id=alpha&stint_id=9",
        "/api/degradation/curves?race_id=2099_01",
    ],
)
def test_unknown_entities_return_404(client: TestClient, url: str) -> None:
    assert client.get(url).status_code == 404


@pytest.mark.parametrize(
    ("method", "url", "body"),
    [
        ("get", "/api/state?race_id=2024_05&driver_id=alpha&lap=0", None),
        ("get", "/api/state?race_id=2024_05&driver_id=alpha&lap=30", None),
        ("get", "/api/state?race_id=2024_05&driver_id=alpha&lap=8&model=lstm", None),
        ("get", "/api/evaluation/errors?group_by=driver", None),
        ("post", "/api/strategy/pit-window", {**WINDOW, "threshold_s": -1}),
        ("post", "/api/strategy/pit-window", {**WINDOW, "pit_loss_s": 500}),
        ("post", "/api/strategy/pit-window", {**WINDOW, "replacement_compound": "WET"}),
        ("post", "/api/strategy/counterfactual", {**WINDOW, "pit_lap": 5}),
        ("post", "/api/strategy/counterfactual", {**WINDOW, "pit_lap": 30}),
    ],
)
def test_out_of_range_returns_422(
    client: TestClient, method: str, url: str, body: dict[str, object] | None
) -> None:
    response = client.post(url, json=body) if method == "post" else client.get(url)
    assert response.status_code == 422, response.text


def test_metrics_lists_profiles_side_by_side(client: TestClient) -> None:
    body = client.get("/api/evaluation/metrics").json()
    assert [p["profile"] for p in body["profiles"]] == ["subset"]
    assert body["profiles"][0]["pit_loss"]["median_s"] == 22.0


def test_store_rejects_incompatible_schema(tmp_path: Path) -> None:
    (tmp_path / "subset").mkdir()
    (tmp_path / "subset" / "manifest.json").write_text(json.dumps({"schema_version": "0"}))
    with pytest.raises(ArtifactError):
        load_store(tmp_path, "subset")


def test_store_requires_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError):
        load_store(tmp_path, "full")
