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
