"""Health, profiles, races, drivers, laps and models."""

import pandas as pd
from fastapi import APIRouter

from boxbox_api.dependencies import ProfileDep, StoreDep
from boxbox_api.schemas import (
    DriverSummary,
    Health,
    LapRecord,
    ModelInfo,
    ProfileInfo,
    RaceSummary,
    StintSummary,
)
from boxbox_api.services import dry_compound, optional_float
from boxbox_api.store import driver_laps, race_laps, servable_races

router = APIRouter(tags=["meta"])


@router.get("/health")
def health(store: StoreDep) -> Health:
    """Liveness plus the loaded artifact versions."""
    default = store.get(None)
    return Health(
        status="ok",
        artifacts_version=str(default.manifest["schema_version"]),
        profiles=sorted(store.profiles),
        default_profile=store.default_profile,
        selected_model=default.selected_model,
        strategy_selected_model=default.strategy_selected_model,
    )


@router.get("/profiles")
def profiles(store: StoreDep) -> list[ProfileInfo]:
    """Loaded training profiles (subset = 2019-23 train, full = 2005-23 train)."""
    infos = []
    for name, profile in sorted(store.profiles.items()):
        splits = profile.config.splits
        infos.append(
            ProfileInfo(
                profile=name,
                created_at=str(profile.manifest["created_at"]),
                config_hash=str(profile.manifest["config_hash"]),
                dataset_hash=str(profile.manifest["dataset_hash"]),
                selected_model=profile.selected_model,
                strategy_selected_model=profile.strategy_selected_model,
                train_seasons=list(splits.train),
                val_seasons=list(splits.val),
                test_seasons=list(splits.test),
                is_default=name == store.default_profile,
            )
        )
    return infos


@router.get("/races")
def races(store: ProfileDep) -> list[RaceSummary]:
    """Races with observed tyre data, newest first."""
    summaries = [
        RaceSummary.model_validate({k: r[k] for k in RaceSummary.model_fields})
        for r in servable_races(store)
    ]
    return sorted(summaries, key=lambda r: r.season, reverse=True)


@router.get("/races/{race_id}/drivers")
def drivers(race_id: str, store: ProfileDep) -> list[DriverSummary]:
    """Drivers of a race with their stints."""
    laps = race_laps(store, race_id)
    summaries = []
    for driver_id, rows in laps.groupby("driver_id"):
        first = rows.iloc[0]
        stints = [
            StintSummary(
                stint_id=int(str(stint_id)),
                compound=dry_compound(stint["compound"].iloc[0]),
                start_lap=int(stint["lap"].min()),
                end_lap=int(stint["lap"].max()),
                stint_source=str(stint["stint_source"].iloc[0]),
            )
            for stint_id, stint in rows.groupby("stint_id")
            if stint["compound"].iloc[0] in {"SOFT", "MEDIUM", "HARD"}
        ]
        summaries.append(
            DriverSummary(
                driver_id=str(driver_id),
                code=str(first["driver_code"]),
                name=str(first["driver_name"]),
                constructor_id=str(first["constructor_id"]),
                team=str(first["team"]),
                stints=stints,
            )
        )
    return sorted(summaries, key=lambda d: d.code)


@router.get("/races/{race_id}/drivers/{driver_id}/laps")
def laps(race_id: str, driver_id: str, store: ProfileDep) -> list[LapRecord]:
    """Every lap of a driver with stint, cleaning and degradation state."""
    rows = driver_laps(store, race_id, driver_id)
    return [
        LapRecord(
            lap=int(r["lap"]),
            lap_time_s=optional_float(r["lap_time_s"]),
            lap_fc_s=optional_float(r["lap_fc_s"]),
            compound=str(r["compound"]),
            tyre_age=float(r["tyre_age"]),
            stint_id=int(r["stint_id"]),
            stint_lap=int(r["stint_lap"]),
            is_clean=bool(r["is_clean"]),
            exclusion_reason=None if pd.isna(r["exclusion_reason"]) else str(r["exclusion_reason"]),
            deg_slope_5=optional_float(r["s_deg_slope_5"]),
            deg_accel=optional_float(r["s_deg_accel"]),
            track_status=str(r["track_status"]),
            pit_in=bool(r["pit_in"]),
            pit_out=bool(r["pit_out"]),
            position=optional_float(r["position"]),
        )
        for _, r in rows.iterrows()
    ]


@router.get("/models")
def models(store: ProfileDep) -> list[ModelInfo]:
    """Models with features, held-out MAE, interpretable parameters and selection flags."""
    return [
        ModelInfo.model_validate(
            meta | {"selected_for_strategy": name == store.strategy_selected_model}
        )
        for name, meta in store.model_meta.items()
    ]
