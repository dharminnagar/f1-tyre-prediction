"""Request/response models: the HTTP contract consumed by the dashboard (via OpenAPI)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from boxbox.config import Profile
from boxbox.schema import Compound, ModelName, Split

GroupBy = Literal["compound", "race", "stint_length"]
Target = Literal["next_lap", "next_5_mean", "next_3_mean"]


class ApiModel(BaseModel):
    """Strict base: unknown fields rejected, instances immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Health(ApiModel):
    status: Literal["ok"]
    artifacts_version: str
    profiles: list[Profile]
    default_profile: Profile
    selected_model: ModelName


class ProfileInfo(ApiModel):
    profile: Profile
    created_at: str
    config_hash: str
    dataset_hash: str
    selected_model: ModelName
    train_seasons: list[int]
    val_seasons: list[int]
    test_seasons: list[int]
    is_default: bool


class RaceSummary(ApiModel):
    race_id: str
    season: int
    round: int
    name: str
    circuit: str
    total_laps: int
    split: Split
    tyre_source: str
    tables_source: str


class StintSummary(ApiModel):
    stint_id: int
    compound: Compound
    start_lap: int
    end_lap: int
    stint_source: str


class DriverSummary(ApiModel):
    driver_id: str
    code: str
    name: str
    constructor_id: str
    team: str
    stints: list[StintSummary]


class LapRecord(ApiModel):
    lap: int
    lap_time_s: float | None
    lap_fc_s: float | None
    compound: str
    tyre_age: float
    stint_id: int
    stint_lap: int
    is_clean: bool
    exclusion_reason: str | None
    deg_slope_5: float | None
    deg_accel: float | None
    track_status: str
    pit_in: bool
    pit_out: bool
    position: float | None


class ModelInfo(ApiModel):
    name: ModelName
    features: list[str]
    selected: bool
    coefficients: dict[str, float] | None
    importances: dict[str, float] | None
    val_mae_s: float | None
    test_mae_s: float | None
    tyre_age_cap: dict[str, float]


class Interval(ApiModel):
    lap: int
    pred_s: float
    lo_s: float
    hi_s: float
    extrapolated: bool


class Degradation(ApiModel):
    empirical_s_per_lap: float | None
    model_s_per_lap: float
    acceleration_s_per_lap2: float | None


class RemainingLifeOut(ApiModel):
    value: int
    threshold_s: float
    reference_s: float
    capped_by_race_end: bool


class DecompositionOut(ApiModel):
    field_ref_s: float
    driver_offset_s: float
    tyre_s: float
    fuel_s: float
    context_s: float
    total_s: float


class TyreState(ApiModel):
    race_id: str
    driver_id: str
    lap: int
    model: ModelName
    compound: Compound
    tyre_age: float
    stint_id: int
    stint_lap: int
    degradation: Degradation
    next_lap: Interval
    next_laps: list[Interval]
    remaining_competitive_laps: RemainingLifeOut
    decomposition: DecompositionOut
    explanation: list[str]


class PitWindowRequest(ApiModel):
    race_id: str
    driver_id: str
    lap: int = Field(ge=1)
    model: ModelName | None = None
    profile: Profile | None = None
    replacement_compound: Compound | None = None
    threshold_s: float | None = Field(default=None, gt=0, le=5)
    pit_loss_s: float | None = Field(default=None, gt=0, le=60)
    warmup_s: float | None = Field(default=None, ge=0, le=5)


class PitCandidateOut(ApiModel):
    pit_lap: int
    cost_pit_s: float
    cost_stay_out_s: float
    advantage_s: float
    margin_s: float
    in_window: bool
    extrapolated: bool


class LapWindow(ApiModel):
    start_lap: int
    end_lap: int


class Assumptions(ApiModel):
    pit_loss_s: float
    pit_loss_source: Literal["estimated", "override"]
    warmup_s: float
    fuel_s_per_lap: float
    margin_sigma: float
    threshold_s: float
    reference_tyre_age: int
    replacement_rule: str


class PitWindowResponse(ApiModel):
    model: ModelName
    candidates: list[PitCandidateOut]
    window: LapWindow | None
    optimal_lap: int | None
    stay_out_feasible: bool
    replacement_compound: Compound
    pit_loss_s: float
    actual_pit_laps: list[int]
    assumptions: Assumptions
    explanation: list[str]


class CounterfactualRequest(ApiModel):
    race_id: str
    driver_id: str
    lap: int = Field(ge=1)
    pit_lap: int = Field(ge=1)
    model: ModelName | None = None
    profile: Profile | None = None
    replacement_compound: Compound | None = None


class CounterfactualResponse(ApiModel):
    laps: list[int]
    stay_out: list[float]
    pit: list[float]
    cumulative_delta_s: list[float]
    actual: list[float | None]
    replacement_compound: Compound
    pit_loss_s: float


class MetricRow(ApiModel):
    model: str
    split: Split
    target: Target
    mae: float | None
    rmse: float | None
    r2: float | None
    mape: float | None
    n: int


class HorizonRow(ApiModel):
    model: ModelName
    split: Split
    k: int
    mae: float
    n: int


class LoroRow(ApiModel):
    model: ModelName
    race_id: str
    mae: float
    n: int


class LoroSummary(ApiModel):
    model: ModelName
    mean: float
    std: float | None


class CumulativeStintRow(ApiModel):
    model: ModelName
    split: Split
    mean_abs_cumulative_s: float
    mean_laps: float
    n: int


class AblationRow(ApiModel):
    model: str
    split: Split
    mae: float
    n: int


class FuelSensitivityRow(ApiModel):
    fuel_s_per_lap: float
    model: ModelName
    split: Split
    mae: float


class PitLossOut(ApiModel):
    median_s: float
    q25_s: float
    q75_s: float
    n_stops: int
    races: list[str]


class StintVerificationOut(ApiModel):
    match_rate: float
    n_stops: int
    by_race: dict[str, float]


class ProfileMetrics(ApiModel):
    profile: Profile
    selected_model: ModelName
    selection_rule: str
    metrics: list[MetricRow]
    horizon: list[HorizonRow]
    loro_cv: list[LoroRow]
    loro_cv_summary: list[LoroSummary]
    cumulative_stint: list[CumulativeStintRow]
    ablation_deg_features: list[AblationRow]
    fuel_sensitivity: list[FuelSensitivityRow]
    pit_loss: PitLossOut
    stint_verification: StintVerificationOut
    pit_timing_in_window_rate: float | None
    train_races: int


class MetricsResponse(ApiModel):
    profiles: list[ProfileMetrics]


class PredictionRow(ApiModel):
    lap: int
    actual_s: float
    pred_s: float
    lo_s: float
    hi_s: float
    split: Split


class ErrorGroupRow(ApiModel):
    group: str
    q10_s: float
    q25_s: float
    q50_s: float
    q75_s: float
    q90_s: float
    n: int


class ThresholdRow(ApiModel):
    threshold_s: float
    model: ModelName
    mae_laps: float | None
    n_uncensored: int
    n_total: int


class PitTimingRow(ApiModel):
    race_id: str
    driver_id: str
    stint_id: int
    decision_lap: int
    actual_pit_lap: int
    window_start: int | None
    window_end: int | None
    optimal_lap: int | None
    actual_in_window: bool


class CurvePoint(ApiModel):
    compound: Compound
    tyre_age: float
    median_s: float
    q25_s: float
    q75_s: float
    n: int
    fitted_s: float
