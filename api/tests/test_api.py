import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from boxbox.export.artifacts import HGB_QUANTILE_FILE
from boxbox.features.tyre_prior import prior_summary
from boxbox.models.forecaster import SequenceForecaster
from boxbox.models.lstm import TrainedLstm, save_lstm
from boxbox.models.quantile import predict_interval
from boxbox.strategy.rollout import predict_segments, stay_out_design
from boxbox_api.main import create_app
from boxbox_api.store import ArtifactError, ProfileStore, Store, load_models, load_store

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


def test_pit_window_returns_plan_cap_counts_and_calibrated_margin(client: TestClient) -> None:
    body = client.post("/api/strategy/pit-window", json=WINDOW).json()
    assert body["assumptions"]["margin_sigma"] == 0.5
    assert body["assumptions"]["margin_source"] == "calibrated"
    plan = body["plan"]
    assert plan and plan[0]["lap"] == body["optimal_lap"]
    assert body["replacement_compound"] == plan[0]["compound"]
    laps = [stop["lap"] for stop in plan]
    assert laps == sorted(set(laps))
    for candidate in body["candidates"]:
        assert candidate["extrapolated"] == (candidate["laps_past_cap"] > 0)
    fixed = client.post("/api/strategy/pit-window", json={**WINDOW, "replacement_compound": "HARD"})
    assert fixed.json()["plan"][0]["compound"] == "HARD"


def test_evaluation_exposes_remaining_life_modes_and_timing_summary(client: TestClient) -> None:
    rows = client.get("/api/evaluation/threshold-sensitivity").json()
    assert {r["decision"] for r in rows} == {"stint_lap_5", "all"}
    assert {"bias_laps", "n_censored", "censored_consistency"} <= set(rows[0])
    profile = client.get("/api/evaluation/metrics").json()["profiles"][0]
    assert profile["pit_timing_summary"]["val"]["excluded"] == {"neutralised": 1}
    assert profile["normalisation_sensitivity"][0]["normalisation"] == "field"
    assert profile["calibrated_margin_sigma"] == 0.5
    timing = client.get("/api/evaluation/pit-timing").json()
    assert timing[0]["excluded_reason"] == "neutralised"


def test_strategy_endpoints_default_to_the_strategy_model(client: TestClient) -> None:
    health = client.get("/api/health").json()
    assert health["selected_model"] == "poly_ridge"
    assert health["strategy_selected_model"] == "hgb"
    flags = {m["name"]: m["selected_for_strategy"] for m in client.get("/api/models").json()}
    assert flags == {"baseline_curve": False, "poly_ridge": False, "hgb": True, "lstm": False}
    assert client.post("/api/strategy/pit-window", json=WINDOW).json()["model"] == "hgb"
    explicit = client.post("/api/strategy/pit-window", json={**WINDOW, "model": "poly_ridge"})
    assert explicit.json()["model"] == "poly_ridge"
    assert client.get(STATE).json()["model"] == "poly_ridge"


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
        ("get", "/api/state?race_id=2024_05&driver_id=alpha&lap=8&model=gru", None),
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


def test_lstm_serves_state_and_pit_window(client: TestClient) -> None:
    state = client.get(f"{STATE}&model=lstm")
    assert state.status_code == 200, state.text
    body = state.json()
    assert body["model"] == "lstm"
    assert len(body["next_laps"]) == 5
    assert all(i["method"] == "sigma" for i in body["next_laps"])
    window = client.post("/api/strategy/pit-window", json={**WINDOW, "model": "lstm"})
    assert window.status_code == 200, window.text
    candidates = window.json()["candidates"]
    assert len(candidates) == 30 - 8 - 1
    # The tiny fixture model has a 6-lap horizon: every plan runs laps beyond it.
    assert all(c["laps_past_cap"] > 0 for c in candidates)
    cf = client.post(
        "/api/strategy/counterfactual", json={**WINDOW, "pit_lap": 12, "model": "lstm"}
    )
    assert cf.status_code == 200, cf.text


def test_forecast_only_model_serves_state_but_no_pit_window(profile_store: ProfileStore) -> None:
    note = "forecast only: validation rollout slope 0.050 s/lap < 0.06"
    meta = {
        **profile_store.model_meta,
        "lstm": {
            **profile_store.model_meta["lstm"],
            "strategy_eligible": False,
            "strategy_note": note,
        },
    }
    store = Store({"subset": replace(profile_store, model_meta=meta)}, "subset")
    with TestClient(create_app(store=store)) as client:
        listed = {m["name"]: m for m in client.get("/api/models").json()}
        assert listed["lstm"]["strategy_eligible"] is False
        assert listed["lstm"]["strategy_note"] == note
        assert listed["hgb"]["strategy_eligible"] is True
        state = client.get(f"{STATE}&model=lstm")
        assert state.status_code == 200, state.text
        assert not any("pays back" in line for line in state.json()["explanation"])
        window = client.post("/api/strategy/pit-window", json={**WINDOW, "model": "lstm"})
        assert window.status_code == 422
        assert note in window.json()["detail"]
        default = client.post("/api/strategy/pit-window", json=WINDOW)
        assert default.status_code == 200, default.text


def test_hgb_intervals_come_from_quantile_models(client: TestClient) -> None:
    body = client.get(f"{STATE}&model=hgb").json()
    for interval in body["next_laps"]:
        assert interval["method"] == "quantile"
        assert interval["lo_s"] <= interval["pred_s"] <= interval["hi_s"]
    ridge = client.get(f"{STATE}&model=poly_ridge").json()
    assert {i["method"] for i in ridge["next_laps"]} == {"sigma"}


def test_store_round_trips_lstm_and_hgb_quantile(
    tmp_path: Path, profile_store: ProfileStore
) -> None:
    lstm = profile_store.models["lstm"]
    assert isinstance(lstm, SequenceForecaster)
    assert isinstance(lstm.model, TrainedLstm)
    assert profile_store.hgb_quantile is not None
    save_lstm(lstm.model, tmp_path / "models")
    joblib.dump(profile_store.hgb_quantile, tmp_path / "models" / HGB_QUANTILE_FILE)
    manifest: dict[str, object] = {
        "models": [{"name": "lstm"}],
        "hgb_quantile_file": f"models/{HGB_QUANTILE_FILE}",
        "tyre_prior": prior_summary(lstm.prior),
    }
    listed = {"models/lstm.pt", "models/lstm.meta.json", f"models/{HGB_QUANTILE_FILE}"}

    def path(relative: str) -> Path:
        assert relative in listed
        return tmp_path / relative

    models, _, quantile = load_models(manifest, path, profile_store.laps)
    loaded = models["lstm"]
    assert isinstance(loaded, SequenceForecaster)
    state = profile_store.laps.query("race_id == '2024_05' and driver_id == 'alpha' and lap == 8")
    design = stay_out_design(state.iloc[0], 30)
    original = predict_segments(lstm, state.iloc[0], [design])[0]
    reloaded = predict_segments(loaded, state.iloc[0], [design])[0]
    np.testing.assert_array_equal(original.y_s, reloaded.y_s)
    np.testing.assert_array_equal(original.past_cap, reloaded.past_cap)
    assert quantile is not None
    np.testing.assert_array_equal(
        predict_interval(quantile, design)[0],
        predict_interval(profile_store.hgb_quantile, design)[0],
    )


def test_metrics_accept_lstm_loro_not_run_and_coverage_rows(client: TestClient) -> None:
    profile = client.get("/api/evaluation/metrics").json()["profiles"][0]
    summary = {r["model"]: r for r in profile["loro_cv_summary"]}
    assert summary["lstm"]["status"] == "not_run" and summary["lstm"]["mean"] is None
    assert profile["interval_coverage"][0]["method"] == "quantile"
    assert {r["cell"] for r in profile["sequence_cell_comparison"]} == {"lstm", "gru"}
