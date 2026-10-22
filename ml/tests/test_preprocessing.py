import pandas as pd
import pytest

from boxbox.config import Config
from boxbox.preprocessing.build import dataset_hash
from boxbox.preprocessing.cleaning import flag_laps
from boxbox.preprocessing.stints import (
    apply_inferred_pit_laps,
    infer_pit_laps,
    reconstruct_stints,
    verify_stints,
)
from boxbox.schema import EXCLUSION_REASONS
from boxbox.synthetic import DriverPlan, canonical_pipeline, synthetic_race


def _driver(laps: pd.DataFrame, driver_id: str) -> pd.DataFrame:
    return laps[laps["driver_id"] == driver_id].set_index("lap")


def test_one_row_per_race_driver_lap(synthetic_laps: pd.DataFrame) -> None:
    assert not synthetic_laps.duplicated(["race_id", "driver_id", "lap"]).any()
    assert len(synthetic_laps) == 6 * 30


def test_one_stop_stints(synthetic_laps: pd.DataFrame) -> None:
    alpha = _driver(synthetic_laps, "alpha")
    assert alpha.loc[12, "stint_id"] == 1 and alpha.loc[13, "stint_id"] == 2
    assert alpha.loc[13, "stint_lap"] == 1
    assert alpha.loc[13, "tyre_age"] == 1
    assert alpha.loc[1, "tyre_age"] == 3  # started on a used set (life offset)
    assert alpha.loc[5, "compound"] == "SOFT" and alpha.loc[20, "compound"] == "MEDIUM"
    assert set(alpha["stint_source"]) == {"observed_tyre"}


def test_two_stop_stints(synthetic_laps: pd.DataFrame) -> None:
    bravo = _driver(synthetic_laps, "bravo")
    assert bravo["stint_id"].max() == 3
    assert bravo.loc[21, "stint_lap"] == 1
    assert bravo.loc[30, "stint_lap"] == 10


def test_penalty_stop_without_tyre_change_is_unmatched(cfg: Config) -> None:
    tables, tyre_layer = synthetic_race()
    penalty = pd.DataFrame([{"driver_id": "charlie", "stop": 2, "lap": 25, "duration_s": 25.0}])
    tables = type(tables)(
        **{**tables.__dict__, "pit_stops": pd.concat([tables.pit_stops, penalty])}
    )
    laps = canonical_pipeline(tables, tyre_layer, cfg)
    verification = verify_stints(laps, tables.pit_stops.assign(race_id=laps["race_id"].iloc[0]))
    charlie = verification[verification["driver_id"] == "charlie"].set_index("lap")
    assert bool(charlie.loc[15, "matched"]) is True
    assert bool(charlie.loc[25, "matched"]) is False
    assert verification["matched"].mean() == pytest.approx(8 / 9)


def test_verification_matches_all_synthetic_stops(synthetic_laps: pd.DataFrame) -> None:
    tables, _ = synthetic_race()
    stops = tables.pit_stops.assign(race_id=synthetic_laps["race_id"].iloc[0])
    assert verify_stints(synthetic_laps, stops)["matched"].all()


def test_no_tyre_change_season_is_single_stint(synthetic_laps: pd.DataFrame) -> None:
    restinted = reconstruct_stints(
        synthetic_laps.drop(columns=["stint_id", "stint_lap", "tyre_age", "stint_source"]),
        tyre_change_allowed=False,
        has_pit_table=True,
    )
    assert (restinted["stint_id"] == 1).all()


def test_pit_laps_inferred_from_lap_times(synthetic_laps: pd.DataFrame) -> None:
    stripped = synthetic_laps.assign(pit_in=False, pit_out=False)
    inferred = apply_inferred_pit_laps(stripped, infer_pit_laps(stripped, 12.0))
    assert (inferred["pit_in"] == synthetic_laps["pit_in"]).all()


def test_flags_keep_every_row(synthetic_laps: pd.DataFrame) -> None:
    assert synthetic_laps["is_clean"].sum() > 0.6 * len(synthetic_laps)
    assert synthetic_laps["is_clean"].eq(synthetic_laps["exclusion_reason"].isna()).all()


def _reason_at(laps: pd.DataFrame, driver_id: str, lap: int) -> object:
    return laps.loc[
        (laps["driver_id"] == driver_id) & (laps["lap"] == lap), "exclusion_reason"
    ].iloc[0]


def test_reason_lap_one_pit_in_pit_out(synthetic_laps: pd.DataFrame) -> None:
    assert _reason_at(synthetic_laps, "alpha", 1) == "lap_one"
    assert _reason_at(synthetic_laps, "alpha", 12) == "pit_in"
    assert _reason_at(synthetic_laps, "alpha", 13) == "pit_out"


@pytest.mark.parametrize("reason", EXCLUSION_REASONS)
def test_each_exclusion_reason(reason: str, cfg: Config) -> None:
    tables, tyre_layer = synthetic_race()
    base = canonical_pipeline(tables, tyre_layer, cfg).drop(
        columns=["is_clean", "exclusion_reason", "track_neutralised"]
    )
    target = (base["driver_id"] == "delta") & (base["lap"] == 7)
    modified = base.copy()
    if reason == "missing_time":
        modified.loc[target, "lap_time_s"] = float("nan")
        expected_lap = 7
    elif reason == "lap_one":
        expected_lap = 1
    elif reason == "pit_in":
        expected_lap = 11
    elif reason == "pit_out":
        expected_lap = 12
    elif reason == "track_status":
        modified.loc[target, "track_status"] = "4"
        expected_lap = 7
    elif reason == "field_slow":
        modified.loc[modified["lap"] == 7, "lap_time_s"] = 130.0
        expected_lap = 7
    elif reason == "post_neutralisation":
        modified.loc[modified["lap"] == 6, "track_status"] = "6"
        expected_lap = 7
    elif reason == "inaccurate":
        modified.loc[target, "is_accurate"] = False
        expected_lap = 7
    elif reason == "wet":
        modified.loc[target, "rainfall"] = True
        expected_lap = 7
    else:
        modified.loc[target, "lap_time_s"] = 95.0
        expected_lap = 7
    flagged = flag_laps(modified, cfg)
    assert len(flagged) == len(base)
    assert _reason_at(flagged, "delta", expected_lap) == reason


def test_dataset_hash_is_order_independent(synthetic_laps: pd.DataFrame) -> None:
    shuffled = synthetic_laps.sample(frac=1.0, random_state=1)
    assert dataset_hash(shuffled) == dataset_hash(synthetic_laps)
    changed = synthetic_laps.assign(lap_time_s=synthetic_laps["lap_time_s"] + 0.001)
    assert dataset_hash(changed) != dataset_hash(synthetic_laps)


def test_driver_plan_is_frozen() -> None:
    plan = DriverPlan("x", 0.0, (1,), ("SOFT",))
    with pytest.raises(AttributeError):
        plan.offset_s = 1.0  # type: ignore[misc]
