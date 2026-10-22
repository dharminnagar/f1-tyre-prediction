import pytest
from pydantic import ValidationError

from boxbox.config import load_config


@pytest.mark.parametrize("profile", ["subset", "full"])
def test_profiles_load_with_identical_val_test(profile: str) -> None:
    cfg = load_config(profile)  # type: ignore[arg-type]
    assert cfg.profile == profile
    assert cfg.splits.val == (2024,)
    assert cfg.splits.test == (2025,)


def test_full_profile_adds_history_only_to_train() -> None:
    subset, full = load_config("subset"), load_config("full")
    assert set(subset.splits.train) < set(full.splits.train)
    assert min(full.splits.train) == 2005
    assert full.historical_seasons == tuple(range(2005, 2018))


def test_overlapping_splits_rejected() -> None:
    data = load_config("subset").model_dump()
    data["splits"]["val"] = [2023]
    with pytest.raises(ValidationError):
        type(load_config("subset")).model_validate(data)


def test_config_hash_is_stable() -> None:
    assert load_config("subset").config_hash() == load_config("subset").config_hash()
    assert load_config("subset").config_hash() != load_config("full").config_hash()


def test_full_profile_drops_refuel_era_for_ridge_and_hgb_only() -> None:
    subset, full = load_config("subset"), load_config("full")
    assert subset.models.exclude_refuel_era_for == ()
    assert full.models.exclude_refuel_era_for == ("poly_ridge", "hgb")
    assert full.models.gbm == subset.models.gbm
    assert full.models.poly == subset.models.poly


def test_tyre_adjusted_offset_and_field_normalisation_are_defaults() -> None:
    cfg = load_config("subset")
    assert cfg.driver_offset == "tyre_adjusted"
    assert cfg.normalisation == "field"
    assert cfg.models.poly.age_degree == 1


def test_lstm_hyper_parameters_live_in_config_without_torch() -> None:
    import subprocess
    import sys

    for profile in ("subset", "full"):
        lstm = load_config(profile).models.lstm
        assert (lstm.cell, lstm.lr, lstm.patience, lstm.horizon) == ("lstm", 1e-3, 30, 20)
        assert load_config(profile).models.lstm_loro is False
    code = "import sys, boxbox.config; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
