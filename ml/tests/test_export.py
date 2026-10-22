import json
import math

import pandas as pd
import pytest

from boxbox.export.artifacts import SCHEMA_VERSION, _json_safe
from boxbox.paths import artifact_paths, dataset_paths


def test_json_safe_replaces_non_finite() -> None:
    assert _json_safe({"a": math.nan, "b": [1.0, math.inf], "c": "x"}) == {
        "a": None,
        "b": [1.0, None],
        "c": "x",
    }


@pytest.mark.parametrize("profile", ["subset", "full"])
def test_manifest_contract(profile: str) -> None:
    paths = artifact_paths(profile)  # type: ignore[arg-type]
    if not paths.manifest.exists():
        pytest.skip(f"no {profile} artifacts; run `make pipeline`")
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["selected_model"] in {m["name"] for m in manifest["models"]}
    assert {"train", "val", "test"} == set(manifest["splits"])
    for relative in manifest["files"]:
        assert (paths.profile_dir / relative).exists()
    assert manifest["pit_loss"]["median_s"] == pytest.approx(22.0, abs=4.0)


@pytest.mark.parametrize("profile", ["subset", "full"])
def test_real_stint_verification_at_least_95_percent(profile: str) -> None:
    path = dataset_paths().processed_file(profile, "stint_verification")  # type: ignore[arg-type]
    if not path.exists():
        pytest.skip("processed dataset not built")
    verification = pd.read_parquet(path)
    assert len(verification) > 100
    assert verification["matched"].mean() >= 0.95


@pytest.mark.parametrize("profile", ["subset", "full"])
def test_manifest_has_calibrated_window_inputs(profile: str) -> None:
    paths = artifact_paths(profile)  # type: ignore[arg-type]
    if not paths.manifest.exists():
        pytest.skip(f"no {profile} artifacts; run `make pipeline`")
    manifest = json.loads(paths.manifest.read_text())
    if "sigma_paired" not in manifest:
        pytest.skip(f"{profile} artifacts predate the paired sigma; rerun `make pipeline`")
    assert manifest["calibrated_margin_sigma"] in (0.25, 0.5, 0.75, 1.0)
    assert set(manifest["sigma_paired"]) == set(manifest["sigma"])
    timing = pd.read_parquet(paths.eval / "pit_timing.parquet")
    assert {"excluded_reason", "planned_stops", "near_optimum_past_cap_share"} <= set(timing)
    sensitivity = pd.read_parquet(paths.eval / "threshold_sensitivity.parquet")
    assert set(sensitivity["decision"]) == {"stint_lap_5", "all"}
    per_stint = sensitivity[sensitivity["decision"] == "stint_lap_5"]
    assert (per_stint["n_uncensored"] + per_stint["n_censored"] == per_stint["n_total"]).all()


@pytest.mark.parametrize("profile", ["subset", "full"])
def test_manifest_lists_sequence_and_quantile_models(profile: str) -> None:
    paths = artifact_paths(profile)  # type: ignore[arg-type]
    if not paths.manifest.exists():
        pytest.skip(f"no {profile} artifacts; run `make pipeline`")
    manifest = json.loads(paths.manifest.read_text())
    if "sequence_model_files" not in manifest:
        pytest.skip(f"{profile} artifacts predate the LSTM; rerun `make pipeline`")
    assert "lstm" in {m["name"] for m in manifest["models"]}
    assert set(manifest["sequence_model_files"]) <= set(manifest["files"])
    assert manifest["hgb_quantile_file"] in manifest["files"]
    methods = {(r["model"], r["method"]) for r in manifest["interval_coverage"]}
    assert ("hgb", "quantile") in methods and ("lstm", "sigma") in methods
    assert {r["cell"] for r in manifest["sequence_cell_comparison"]} == {"lstm", "gru"}
