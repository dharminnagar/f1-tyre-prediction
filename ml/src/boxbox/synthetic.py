"""Deterministic synthetic race used by tests and API fixtures (no network, no real data)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from boxbox.config import Config
from boxbox.ingestion.tables import RaceTables
from boxbox.preprocessing.cleaning import flag_laps
from boxbox.preprocessing.laps import add_gaps, to_canonical_laps
from boxbox.preprocessing.stints import reconstruct_stints
from boxbox.schema import era_for_season

DEG_S_PER_LAP = 0.08


@dataclass(frozen=True)
class DriverPlan:
    """One synthetic driver: pace offset, in-laps and the compound of each stint."""

    driver_id: str
    offset_s: float
    stops: tuple[int, ...]
    compounds: tuple[str, ...]


DEFAULT_PLANS = (
    DriverPlan("alpha", 0.0, (12,), ("SOFT", "MEDIUM")),
    DriverPlan("bravo", 0.3, (10, 20), ("SOFT", "MEDIUM", "SOFT")),
    DriverPlan("charlie", 0.6, (15,), ("MEDIUM", "HARD")),
    DriverPlan("delta", 0.9, (11,), ("SOFT", "HARD")),
    DriverPlan("echo", 1.2, (14, 22), ("MEDIUM", "SOFT", "SOFT")),
    DriverPlan("foxtrot", 1.5, (13,), ("SOFT", "MEDIUM")),
)


def synthetic_race(
    season: int = 2023,
    total_laps: int = 30,
    plans: tuple[DriverPlan, ...] = DEFAULT_PLANS,
    seed: int = 7,
    fuel_s_per_lap: float = 0.05,
) -> tuple[RaceTables, pd.DataFrame]:
    """Ergast-style tables plus a FastF1-style tyre layer for one synthetic race."""
    rng = np.random.default_rng(seed)
    lap_rows: list[dict[str, object]] = []
    tyre_rows: list[dict[str, object]] = []
    pit_rows: list[dict[str, object]] = []
    for plan in plans:
        stint, age = 0, 2.0
        for lap in range(1, total_laps + 1):
            if lap - 1 in plan.stops:
                stint, age = stint + 1, 0.0
            age += 1
            time = 80 + plan.offset_s + DEG_S_PER_LAP * age + fuel_s_per_lap * (total_laps - lap)
            time += rng.normal(0, 0.05)
            if lap == 1:
                time += 6
            if lap in plan.stops:
                time += 4
            if lap - 1 in plan.stops:
                time += 18
            lap_rows.append({"driver_id": plan.driver_id, "lap": lap, "lap_time_s": time})
            tyre_rows.append(
                {
                    "driver_id": plan.driver_id,
                    "lap": lap,
                    "lap_time_s": time,
                    "sector_1_s": time / 3,
                    "sector_2_s": time / 3,
                    "sector_3_s": time / 3,
                    "compound_raw": plan.compounds[stint],
                    "tyre_life": age,
                    "pit_in_flag": lap in plan.stops,
                    "pit_out_flag": lap - 1 in plan.stops,
                    "track_status": "1",
                    "is_accurate": True,
                    "air_temp": 25.0,
                    "track_temp": 40.0,
                    "humidity": 50.0,
                    "rainfall": False,
                }
            )
        pit_rows.extend(
            {"driver_id": plan.driver_id, "stop": i + 1, "lap": lap, "duration_s": 22.0}
            for i, lap in enumerate(plan.stops)
        )
    laps = pd.DataFrame(lap_rows)
    laps["position"] = (
        laps.assign(cum=laps.groupby("driver_id")["lap_time_s"].cumsum())
        .groupby("lap")["cum"]
        .rank()
        .astype(int)
    )
    results = pd.DataFrame(
        {
            "driver_id": [p.driver_id for p in plans],
            "driver_code": [p.driver_id[:3].upper() for p in plans],
            "driver_name": [p.driver_id.title() for p in plans],
            "car_number": list(range(1, len(plans) + 1)),
            "constructor_id": [f"team_{i // 2}" for i in range(len(plans))],
            "team": [f"Team {i // 2}" for i in range(len(plans))],
            "grid": list(range(1, len(plans) + 1)),
            "laps_completed": total_laps,
            "status": "Finished",
        }
    )
    quali = pd.DataFrame(
        {
            "driver_id": [p.driver_id for p in plans],
            "quali_best_s": [76 + p.offset_s for p in plans],
        }
    )
    tables = RaceTables(
        season=season,
        round_number=5,
        race_name="Spanish Grand Prix",
        laps=laps[["driver_id", "lap", "position", "lap_time_s"]],
        pit_stops=pd.DataFrame(pit_rows),
        results=results,
        quali=quali,
    )
    return tables, pd.DataFrame(tyre_rows)


def canonical_pipeline(
    tables: RaceTables, tyre_layer: pd.DataFrame | None, cfg: Config
) -> pd.DataFrame:
    """Run the same steps as ``build_season`` + ``flag_laps`` on in-memory tables."""
    laps = to_canonical_laps(tables, tyre_layer, cfg.event_name, "synthetic")
    laps = reconstruct_stints(laps, tyre_change_allowed=True, has_pit_table=True)
    laps = add_gaps(laps, cfg.traffic_gap_s)
    laps["era"] = era_for_season(tables.season)
    laps["tyre_source"] = "synthetic"
    return flag_laps(laps, cfg)
