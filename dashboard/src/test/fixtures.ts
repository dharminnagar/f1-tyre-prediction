import type {
  CounterfactualResponse,
  DriverSummary,
  LapRecord,
  ModelInfo,
  PitWindowResponse,
  ProfileInfo,
  ProfileMetrics,
  RaceSummary,
  ThresholdRow,
  TyreState,
} from "../api/client";

export const profiles: ProfileInfo[] = [
  {
    profile: "subset",
    created_at: "2026-10-21",
    config_hash: "abc",
    dataset_hash: "def",
    selected_model: "poly_ridge",
    strategy_selected_model: "poly_ridge",
    train_seasons: [2019, 2023],
    val_seasons: [2024],
    test_seasons: [2025],
    is_default: true,
  },
];

export const races: RaceSummary[] = [
  {
    race_id: "2024_10",
    season: 2024,
    round: 10,
    name: "Spanish Grand Prix",
    circuit: "Circuit de Barcelona-Catalunya",
    total_laps: 30,
    split: "val",
    tyre_source: "openf1",
    tables_source: "kaggle_jtrotman",
  },
];

export const drivers: DriverSummary[] = [
  {
    driver_id: "alonso",
    code: "ALO",
    name: "Fernando Alonso",
    constructor_id: "aston_martin",
    team: "Aston Martin",
    stints: [
      { stint_id: 1, compound: "SOFT", start_lap: 1, end_lap: 12, stint_source: "observed_tyre" },
      { stint_id: 2, compound: "MEDIUM", start_lap: 13, end_lap: 30, stint_source: "observed_tyre" },
    ],
  },
];

export const laps: LapRecord[] = Array.from({ length: 30 }, (_, i) => {
  const lap = i + 1;
  const stint = lap <= 12 ? 1 : 2;
  return {
    lap,
    lap_time_s: 80 + (lap <= 12 ? lap : lap - 12) * 0.08,
    lap_fc_s: 79,
    compound: stint === 1 ? "SOFT" : "MEDIUM",
    tyre_age: stint === 1 ? lap : lap - 12,
    stint_id: stint,
    stint_lap: stint === 1 ? lap : lap - 12,
    is_clean: lap !== 1 && lap !== 12 && lap !== 13,
    exclusion_reason: lap === 1 ? "lap_one" : lap === 12 ? "pit_in" : lap === 13 ? "pit_out" : null,
    deg_slope_5: 0.08,
    deg_accel: 0,
    track_status: "1",
    pit_in: lap === 12,
    pit_out: lap === 13,
    position: 5,
  };
});

export const models: ModelInfo[] = [
  { name: "baseline_curve", features: [], selected: false, selected_for_strategy: false, coefficients: null, importances: null, val_mae_s: 0.5, test_mae_s: 0.55, tyre_age_cap: { SOFT: 25 } },
  { name: "poly_ridge", features: [], selected: true, selected_for_strategy: true, coefficients: null, importances: null, val_mae_s: 0.46, test_mae_s: 0.5, tyre_age_cap: { SOFT: 25 } },
  { name: "hgb", features: [], selected: false, selected_for_strategy: false, coefficients: null, importances: null, val_mae_s: 0.47, test_mae_s: 0.48, tyre_age_cap: { SOFT: 25 } },
];

export const state: TyreState = {
  race_id: "2024_10",
  driver_id: "alonso",
  lap: 8,
  model: "poly_ridge",
  compound: "SOFT",
  tyre_age: 8,
  stint_id: 1,
  stint_lap: 8,
  degradation: { empirical_s_per_lap: 0.081, model_s_per_lap: 0.075, acceleration_s_per_lap2: 0.001 },
  next_lap: { lap: 9, pred_s: 80.72, lo_s: 80.1, hi_s: 81.3, extrapolated: false, method: "sigma" },
  next_laps: [9, 10, 11, 12, 13].map((lap, i) => ({
    lap,
    pred_s: 80.72 + i * 0.08,
    lo_s: 80.1 + i * 0.08,
    hi_s: 81.3 + i * 0.08,
    extrapolated: false,
    method: "sigma" as const,
  })),
  remaining_competitive_laps: { value: 7, threshold_s: 1, reference_s: 80.2, capped_by_race_end: false },
  decomposition: { field_ref_s: 79.5, driver_offset_s: 0.3, tyre_s: 0.4, fuel_s: 1.05, context_s: -0.53, total_s: 80.72 },
  explanation: ["SOFT tyre, 8 laps old (stint lap 8).", "About 7 competitive laps left."],
};

export const pitWindow: PitWindowResponse = {
  model: "poly_ridge",
  candidates: Array.from({ length: 21 }, (_, i) => {
    const pitLap = 9 + i;
    const cost = 2400 + (pitLap - 14) ** 2 * 0.1;
    const lapsPastCap = Math.max(0, pitLap - 26);
    return {
      pit_lap: pitLap,
      cost_pit_s: cost,
      cost_stay_out_s: 2420,
      advantage_s: 2420 - cost,
      margin_s: 1,
      in_window: Math.abs(pitLap - 14) <= 3,
      laps_past_cap: lapsPastCap,
      extrapolated: lapsPastCap > 0,
    };
  }),
  window: { start_lap: 11, end_lap: 17 },
  optimal_lap: 14,
  plan: [
    { lap: 14, compound: "MEDIUM" },
    { lap: 22, compound: "HARD" },
  ],
  stay_out_feasible: false,
  replacement_compound: "MEDIUM",
  pit_loss_s: 23.2,
  actual_pit_laps: [12],
  assumptions: {
    pit_loss_s: 23.2,
    pit_loss_source: "estimated",
    warmup_s: 0,
    fuel_s_per_lap: 0.05,
    margin_sigma: 0.5,
    margin_source: "calibrated",
    threshold_s: 1,
    reference_tyre_age: 3,
    replacement_rule: "actual next",
  },
  explanation: [],
};

export const counterfactual: CounterfactualResponse = {
  laps: Array.from({ length: 22 }, (_, i) => 9 + i),
  stay_out: Array.from({ length: 22 }, () => 81),
  pit: Array.from({ length: 22 }, () => 80.5),
  cumulative_delta_s: Array.from({ length: 22 }, (_, i) => 23 - i * 0.5),
  actual: Array.from({ length: 22 }, () => null),
  replacement_compound: "MEDIUM",
  pit_loss_s: 23.2,
};

export function profileMetrics(profile: "subset" | "full", offset: number): ProfileMetrics {
  return {
    profile,
    selected_model: "poly_ridge",
    selection_rule: "lowest validation next-lap MAE",
    metrics: [
      { model: "poly_ridge", split: "val", target: "next_lap", mae: 0.463 + offset, rmse: 0.6, r2: 0.6, mape: 0.5, n: 100 },
      { model: "poly_ridge", split: "test", target: "next_lap", mae: 0.501 + offset, rmse: 0.6, r2: 0.6, mape: 0.5, n: 100 },
    ],
    horizon: [{ model: "poly_ridge", split: "val", k: 1, mae: 0.46, n: 100 }],
    loro_cv: [],
    loro_cv_summary: [
      { model: "poly_ridge", mean: 0.6, std: 0.05, status: "run" },
      { model: "lstm", mean: null, std: null, status: "not_run" },
    ],
    cumulative_stint: [],
    ablation_deg_features: [],
    fuel_sensitivity: [],
    pit_loss: { median_s: 23.16, q25_s: 22, q75_s: 24, n_stops: 190, races: [] },
    stint_verification: { match_rate: 0.99, n_stops: 301, by_race: {} },
    pit_timing_in_window_rate: 0.26,
    pit_timing_summary: {
      val: {
        n: 42,
        in_window_rate: 0.6,
        n_kept: 40,
        in_window_rate_kept: 0.62,
        excluded: { neutralised: 2 },
        in_window_rate_kept_by_stint: { "1": 0.6 },
        median_width_kept: 9,
        no_window_share_kept: 0.05,
        optimum_bias_kept_by_stint: { "1": 4 },
        near_optimum_past_cap_share_kept: 0.1,
      },
    },
    normalisation_sensitivity: [
      { normalisation: "field", model: "poly_ridge", val_mae: 0.452, test_mae: 0.49, held_out_rollout_s_per_lap: 0.073 },
      { normalisation: "teammate", model: "poly_ridge", val_mae: 0.47, test_mae: 0.5, held_out_rollout_s_per_lap: 0.07 },
    ],
    calibrated_margin_sigma: 0.5,
    window_target_coverage: 0.5,
    margin_calibration: [{ margin_sigma: 0.5, in_window_rate: 0.55, median_width_laps: 9, n: 40 }],
    interval_coverage: [
      { model: "hgb", split: "val", target: "next_lap", method: "quantile", coverage: 0.8, mean_width_s: 1.46, n: 1205 },
      { model: "hgb", split: "test", target: "next_lap", method: "quantile", coverage: 0.7, mean_width_s: 1.28, n: 972 },
      { model: "poly_ridge", split: "test", target: "next_lap", method: "sigma", coverage: 0.83, mean_width_s: 1.65, n: 972 },
    ],
    sequence_cell_comparison: (["lstm", "gru"] as const).map((cell) => ({
      cell,
      split: "val" as const,
      next_lap_mae: 0.459,
      next_n_mean_mae: 0.39,
      n: 5,
      horizon_k: 20,
      horizon_k_mae: 0.68,
      rollout_s_per_lap_held_out: 0.045,
      rollout_s_per_lap_train: 0.044,
      train_seconds: 12,
      best_epoch: 25,
    })),
    pit_timing_by_model: [],
    train_races: profile === "subset" ? 5 : 19,
  };
}

function thresholdRow(decision: ThresholdRow["decision"], threshold: number, mae: number): ThresholdRow {
  return {
    decision,
    threshold_s: threshold,
    model: "poly_ridge",
    mae_laps: mae,
    bias_laps: 0.5,
    n_uncensored: 20,
    n_censored: 60,
    censored_consistency: 0.8,
    n_total: 80,
  };
}

export const thresholdRows: ThresholdRow[] = [
  thresholdRow("stint_lap_5", 1, 3.2),
  thresholdRow("stint_lap_5", 0.5, 2.1),
  thresholdRow("all", 1, 4.4),
];
