# BoxBox — Implementation Plan (Tyre Degradation & Pit-Window Estimator)

> Status: APPROVED (2026-10-21). Defaults D1–D15 accepted; user answers recorded in §10.

Environment found: system Python 3.14.7, Homebrew `python3.12` at /opt/homebrew/bin, Node v24.10.0, pnpm, make. No kaggle CLI, no `~/.kaggle/`. Spec line 191: "train on selected historical Spanish races and hold out another race".

---

## 1. Folder tree

Rationale: `dataset/` = heavy, uncommitted data. `ml/` = all DS logic as installable package `boxbox`. `api/` = thin HTTP layer importing `boxbox` (strategy maths not duplicated). `artifacts/` = the only contract between ml→api (ml writes, api reads, api never trains). `dashboard/` = React, talks only to the API; TS types generated from the API's OpenAPI schema. One root `.venv` (Python 3.12) with `ml` + `api` installed editable.

```
BoxBox/
├── Makefile                      # single entry point for every workflow (§8)
├── pyproject.toml                # shared ruff/mypy/pytest config (no build, tool config only)
├── requirements.txt              # exact pinned versions (pip freeze), incl. dev + notebooks
├── .venv/                        # python3.12 -m venv .venv (gitignored)
├── .gitignore
├── .env.example                  # BOXBOX_DATASET_DIR, BOXBOX_ARTIFACTS_DIR, API_PORT (no secrets)
├── configs/default.yaml          # races, splits, thresholds, fuel/pit constants, model params
├── F1_Tyre_Specification_Doc.docx
├── dataset/                      # gitignored except .gitkeep
│   ├── raw/{fastf1_cache,ergast,kaggle/formula-1-race-data,f1db}/
│   ├── interim/laps_{season}_{round}.parquet
│   └── processed/{laps_clean,features,stint_verification}.parquet
├── ml/
│   ├── pyproject.toml            # package "boxbox", console script `boxbox`
│   ├── src/boxbox/
│   │   ├── cli.py config.py paths.py schema.py
│   │   ├── ingestion/            # fastf1_client, ergast_client, kaggle_client, f1db_client (connector classes), download.py
│   │   ├── preprocessing/        # laps.py, alignment.py, stints.py, cleaning.py
│   │   ├── features/             # pace.py, degradation.py, context.py, build.py
│   │   ├── models/               # types.py, baseline_curve.py, poly_regression.py, gbm.py, train.py
│   │   ├── evaluation/           # splits.py, metrics.py, evaluate.py
│   │   ├── strategy/             # rollout, uncertainty, pit_loss, remaining_life, pit_window, explain
│   │   ├── reporting/            # eda.py, figures.py
│   │   └── export/artifacts.py
│   ├── tests/                    # pytest, synthetic fixtures, no network by default
│   └── notebooks/                # 01_data_audit … 05_strategy_simulation (thin, call boxbox)
├── artifacts/                    # generated; manifest.json, models/, data/, eval/, figures/
├── reports/                      # executed notebooks → HTML
├── api/
│   ├── pyproject.toml            # package "boxbox_api", depends on "boxbox" (installed editable into the same venv)
│   ├── src/boxbox_api/{main,settings,schemas,store}.py, routers/{meta,state,strategy,evaluation}.py
│   └── tests/
└── dashboard/                    # Vite + React + TS
    └── src/{api/{schema.d.ts,client.ts,queries.ts}, pages/, components/{selectors,kpi,charts,explain}/, lib/}
```

## 2. Tooling

- **Python** (plain venv): `/opt/homebrew/bin/python3.12 -m venv .venv`, then `.venv/bin/pip install -r requirements.txt -e ./ml -e ./api` (not system 3.14 — FastF1 stack safest on 3.12). `requirements.txt` regenerated via `pip freeze --exclude-editable` whenever deps change. `fastf1>=3.6,<4`, `pandas>=2.2,<3`, `numpy>=2`, `scikit-learn>=1.6`, `pyarrow>=17`, `joblib`, `pydantic>=2.9`, `pyyaml`, `typer>=0.12`, `matplotlib>=3.9`. API: `fastapi>=0.115`, `uvicorn[standard]`, `pydantic-settings>=2.5`. Optional extras in `ml/pyproject.toml`: `kaggle`, `notebooks` (jupyterlab, nbconvert). Dev: pytest, pytest-cov, httpx, ruff, mypy (strict), pandas-stubs, types-PyYAML.
- ruff rules `E,F,I,UP,B,ANN,ANN401,D(google),SIM,PD,NPY,RUF`, line 100. mypy strict; `ignore_missing_imports` only for fastf1/sklearn/joblib — untyped libs wrapped behind connector classes / typed functions.
- **Frontend** (pnpm, Node 24): Vite 7, React 19, TS 5.9 strict (+`noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`), Recharts 3, TanStack Query 5, react-router 7, Tailwind 4, openapi-typescript 7 + openapi-fetch, Vitest 3 + Testing Library, ESLint 9 + typescript-eslint 8 strictTypeChecked (`no-explicit-any: error`). Vite proxy `/api` → `127.0.0.1:8000`.
- **Style**: pure `(df, cfg) -> df` functions, frozen dataclasses/pydantic, `Literal` types; classes only for external connectors.

## 3. Default decisions (confirm each)

| # | Gap | Decision |
|---|---|---|
| D1 | Seasons/circuit | Barcelona Spanish GP 2019–2025 (7 Race + Qualifying sessions, all dry). Exclude 2026 (new regs, Madrid). |
| D2 | Split | Train 2019–2023, val 2024, test 2025 (reported once). Secondary: leave-one-race-out GroupKFold. |
| D3 | Compounds | Dry only (S/M/H); drop INTER/WET and rainfall laps. |
| D4 | Fuel | `lap_fc = lap_time_s − 0.05·(total_laps − lap_number)` + `race_progress` feature; sensitivity 0.03/0.07. |
| D5 | Normalisation | `y = lap_fc − field_ref(L)` (causal median of all clean laps ≤ L−1); `driver_offset` (expanding causal median) + `quali_gap_s`. Teammate-delta as sensitivity variant. |
| D6 | Tyre age | `tyre_age = TyreLife`; `stint_lap` with out-lap = 1. |
| D7 | Competitive threshold | Reference = predicted `lap_fc` same compound at tyre_age 3; competitive while `pred − ref ≤ X`, X = 1.0 s; sensitivity 0.5/0.75/1.0/1.5/2.0. Ground truth same rule on actual rolling-3, `censored` if pitted first. |
| D8 | N | N = 5; horizon MAE for k = 1..10; N = 3 also reported. |
| D9 | Pit loss | Per circuit from green-flag stops in train races: `(in+out) − 2·median(clean laps ±3)`, median across stops; overridable. |
| D10 | Warm-up | 0 by default (already in in/out loss); sensitivity 0.5/1.0 s. |
| D11 | Replacement compound | Actual next compound historically, else hardest unused dry; override allowed; flag `stay_out_feasible=false` if two-compound rule unmet. |
| D12 | Uncertainty (MVP) | `sigma_cum(k)` = std of cumulative k-lap rollout error on val race; in window if `advantage > 1.0·sigma_cum`; next-lap ±1.28·σ(1) (~80%). |
| D13 | Extrapolation | Clip tyre_age at per-compound P95 of training; flag `extrapolated`. |
| D14 | Clean lap | TrackStatus only "1"; not pit-in/out; lap > 1; not within 1 lap after SC/VSC/red; IsAccurate; < 107% of causal rolling median; dry. Keep all rows with `is_clean` + `exclusion_reason`. |
| D15 | Selected model | Lowest val MAE (target A), recorded in manifest; all selectable in dashboard. |

```yaml
# configs/default.yaml
races: [{season: 2019, event: "Spanish Grand Prix"}, ..., {season: 2025, event: "Spanish Grand Prix"}]
splits: {train: [2019, 2020, 2021, 2022, 2023], val: [2024], test: [2025]}
compounds: [SOFT, MEDIUM, HARD]
fuel_s_per_lap: 0.05
competitive_threshold_s: 1.0
threshold_sensitivity_s: [0.5, 0.75, 1.0, 1.5, 2.0]
reference_tyre_age: 3
horizon_n: 5
warmup_effect_s: 0.0
window_margin_sigma: 1.0
traffic_gap_s: 1.0
outlier_pct: 1.07
models: {poly: {alpha: 1.0}, gbm: {max_iter: 300, learning_rate: 0.05, max_leaf_nodes: 15}}
random_seed: 42
```

**Rollout design**: one per-lap pace model `f(x_L) → y_L`. Deterministic features at L (compound, tyre_age, stint_lap, lap_number, race_progress) are advanced; causal state features (driver_offset, quali_gap_s, temps, position, gap_ahead, traffic, deg_slope_5 for GBM ablation) are frozen at t. Pit branch resets compound/tyre_age/stint_lap. Target A = k=1, B = mean k=1..5, D = full rollout.

## 4. Data acquisition

- **FastF1** (primary, no creds): cache at `dataset/raw/fastf1_cache`; load Race + Qualifying with `laps=True, telemetry=False, weather=True, messages=False`; write interim parquet.
- **Ergast/jolpica** via `fastf1.ergast.Ergast`: pit stops + results → `dataset/raw/ergast/`. Credential-free substitute for Kaggle tables.
- **Kaggle jtrotman** (optional): only if the user has set up their own token (presence check only — never read/print/write values); else log fallback and continue.
- **F1DB** (optional, `--with-f1db`): latest release CSV zip → `dataset/raw/f1db/`; pipeline must work without it.
- **IDs**: `race_id = f"{season}_{round:02d}"`; `driver_id` = FastF1 `DriverId` (= Ergast `driverRef`); `constructor_id` from Ergast.
- **gitignore**: `dataset/**`, `artifacts/**` (keep `.gitkeep`), `reports/*.html`, `.venv`, `node_modules`, `dashboard/dist`, `.env`, `*.joblib`.

## 5. Phased tasks (‖ = parallelisable; AC = acceptance criteria)

- **Phase 0 — Scaffold**: 0.1 root workspace, configs, Makefile, `.gitignore`, dataset/artifacts trees (AC: `make setup` creates `.venv` and installs; `make lint typecheck test` green). 0.2 `config.py`, `paths.py`, `schema.py` (AC: loads default.yaml). 0.3 ‖ API contract `schemas.py` + stub routes + `make gen-types`. 0.4 ‖ dashboard scaffold (AC: lint/typecheck/test pass).
- **Phase 1 — Acquisition**: 1.1 FastF1 client + `download_fastf1` (AC: 7+7 sessions cached, rerun is no-op). 1.2 ‖ Ergast pit stops. 1.3 ‖ Kaggle client (no-creds → exit 0 + warning, tested). 1.4 ‖ F1DB optional. Network tests marked `@pytest.mark.network`.
- **Phase 2 — Laps/stints/cleaning**: 2.1 `to_canonical_laps` (seconds, IDs, weather, gaps; AC: one row per race/driver/lap). 2.2 `reconstruct_stints` + `verify_stints` (AC: ≥95% stop match; synthetic 1-stop/2-stop/penalty tests). 2.3 `flag_laps` with `exclusion_reason` Literal (AC: test per reason, no rows dropped). 2.4 `build_lap_dataset` (AC: deterministic hash).
- **Phase 3 — EDA** (‖ Phase 4): `reporting/eda.py` + notebooks 01–03 (AC: `make report` exports HTML).
- **Phase 4 — Features & splits**: 4.1 pace (fuel, causal field_ref, driver_offset, quali_gap). 4.2 ‖ degradation (causal rolling slope 3/5, acceleration, rolling means, lap_delta). 4.3 ‖ context. 4.4 `build_features` + targets + `assign_splits` (AC: leakage property test — perturbing laps > L leaves features at L unchanged; no race in >1 split).
- **Phase 5 — Models & eval**: 5.1 `PaceModel` Protocol. 5.2 ‖ baseline curve, poly Ridge (coeffs exported), HistGBM monotone in tyre_age (permutation importances). 5.3 `fit_all` + `evaluate` (MAE/RMSE/R²/MAPE for A & B, horizon MAE, cumulative stint error, LORO-CV, error tables; AC: ≥2 models on val+test, test used once).
- **Phase 6 — Strategy**: 6.1 pit loss (AC: synthetic exact; Barcelona ~18–25 s). 6.2 ‖ rollouts (pure, clipping, flags). 6.3 sigma curve. 6.4 remaining life + sensitivity (AC: exact on synthetic 0.1 s/lap). 6.5 pit cost curve / window / counterfactual / compare-with-actual (AC: closed-form analytic test ±1 lap; infeasible case). 6.6 model-agnostic decomposition + explanation lines.
- **Phase 7 — Export**: `export_all` → models (joblib + meta), data parquet/json, eval files, figures, `manifest.json` (schema_version, config_hash, selected_model, splits). AC: `boxbox all` builds full tree; manifest test.
- **Phase 8 — API**: `load_store` (validates schema_version; loads only project-produced artifacts), routers call `boxbox.strategy.*`; tests with tiny fixture store (AC: 200s, 404 unknown race/driver/lap, 422 out-of-range).
- **Phase 9 — Dashboard** (can start after 0.3/0.4): query hooks + URL state, PitWall / Degradation / Evaluation pages, Vitest tests (AC: lint/typecheck/test green, no `any`).
- **Phase 10 — Integration**: `make all`, `make dev`, remove stubs, execute notebooks, `make check` green, DoD checklist.

## 6. API contract (FastAPI, all under `/api`)

Shared: `Compound = Literal["SOFT","MEDIUM","HARD"]`, `ModelName = Literal["baseline_curve","poly_ridge","hgb"]`, `Split = Literal["train","val","test"]`.

| Endpoint | Request | Response |
|---|---|---|
| `GET /health` | – | `{status, artifacts_version, selected_model}` |
| `GET /races` | – | `RaceSummary[]{race_id, season, round, name, circuit, total_laps, split}` |
| `GET /races/{race_id}/drivers` | – | `DriverSummary[]{driver_id, code, name, constructor_id, team, stints[]{stint_id, compound, start_lap, end_lap, stint_source}}` |
| `GET /races/{race_id}/drivers/{driver_id}/laps` | – | `LapRecord[]{lap, lap_time_s, lap_fc_s, compound, tyre_age, stint_id, stint_lap, is_clean, exclusion_reason, deg_slope_5, deg_accel, track_status}` |
| `GET /models` | – | `ModelInfo[]{name, features, selected, coefficients?, importances?, val_mae_s, test_mae_s}` |
| `GET /state` | `race_id, driver_id, lap, model?` | `TyreState{compound, tyre_age, stint_lap, degradation{empirical, model, acceleration}, next_lap{pred,lo,hi}, next_laps[5], remaining_competitive_laps{value, threshold_s, reference_s, capped_by_race_end}, decomposition{field_ref, driver_offset, tyre, fuel, context}, explanation[]}` |
| `POST /strategy/pit-window` | `{race_id, driver_id, lap, model?, replacement_compound?, threshold_s?, pit_loss_s?, warmup_s?}` | `{candidates[]{pit_lap, cost_pit_s, cost_stay_out_s, advantage_s, margin_s, in_window, extrapolated}, window{start_lap,end_lap}?, optimal_lap, stay_out_feasible, replacement_compound, pit_loss_s, actual_pit_laps, assumptions, explanation[]}` |
| `POST /strategy/counterfactual` | `{race_id, driver_id, lap, pit_lap, replacement_compound?, model?}` | `{stay_out[], pit[], cumulative_delta_s[], actual[]}` |
| `GET /evaluation/metrics` | – | metric rows + horizon + LORO-CV |
| `GET /evaluation/predictions` | `race_id, driver_id, stint_id, model?` | `{lap, actual_s, pred_s, lo_s, hi_s}[]` |
| `GET /evaluation/errors` | `group_by: compound\|race\|stint_length, model?` | quantile rows |
| `GET /evaluation/threshold-sensitivity` | – | `{threshold_s, model, mae_laps, n_uncensored}[]` |
| `GET /evaluation/pit-timing` | – | `{race_id, driver_id, stint_id, actual_pit_lap, window_start, window_end, optimal_lap, actual_in_window}[]` |
| `GET /degradation/curves` | `compound?, race_id?` | `{compound, tyre_age, median_s, q25_s, q75_s, n, fitted_s}[]` |

Flow: ml → `artifacts/` → API loads once at startup; evaluation endpoints serve precomputed files; `/state`, `/strategy/*`, `/degradation` compute on request via the same `boxbox.strategy` functions. Frontend types come only from `/openapi.json` (dumped offline).

## 7. Dashboard

- **PitWallPage**: Race/Driver/Lap/Model selectors; KPI tiles (compound+age, degradation s/lap, next lap ± interval, next-5 sparkline, remaining competitive laps, pit window); StintPaceChart; PitCostChart (window shaded, optimum + actual pit); CounterfactualChart; DegradationSlopeChart; WhyPanel (explanation + decomposition bars).
- **DegradationPage**: DegradationCurveChart (median + IQR per compound, poly fit overlay).
- **EvaluationPage**: metrics table + horizon chart, PredVsActualChart, ErrorDistributionChart, threshold sensitivity, pit-timing table ("descriptive only"), AssumptionsPanel.
- Compound colours: SOFT red, MEDIUM yellow, HARD white/grey.

## 8. Makefile

```
setup      /opt/homebrew/bin/python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt -e ./ml -e ./api && pnpm -C dashboard install
download   .venv/bin/boxbox download
dataset    .venv/bin/boxbox build
train      .venv/bin/boxbox train
evaluate   .venv/bin/boxbox evaluate
export     .venv/bin/boxbox export
pipeline   .venv/bin/boxbox all
report     .venv/bin/jupyter nbconvert --execute --to html ml/notebooks/*.ipynb --output-dir reports
api        .venv/bin/uvicorn boxbox_api.main:create_app --factory --port 8000
gen-types  (offline OpenAPI dump) → openapi-typescript → dashboard/src/api/schema.d.ts
dashboard  pnpm -C dashboard dev
dev        $(MAKE) -j2 api dashboard
lint / typecheck / test / check
all        setup download pipeline
```

## 9. Risks & scope

Risks: small sample (~6–7k clean laps → use LORO-CV); regime shifts (2022 cars, 2023 layout); tyre-age ⟂ fuel/traffic confounding; censored remaining-life ground truth; extrapolation in long rollouts; FastF1/jolpica rate limits (cache); pandas/sklearn typing friction.

MVP = Phases 0–10. **Phase 11 stretch**: quantile/conformal intervals; free-form "pit on lap X" simulator; undercut/overcut; cross-circuit (Bahrain, Silverstone); stint-shape clustering; telemetry features; GRU/LSTM; OpenF1; Kaggle pre-2018 inferred stints.

## 10. Confirmed decisions (user answers)

1. **Two training profiles** (config-selected, e.g. `boxbox all --profile subset|full`; `configs/subset.yaml` and `configs/full.yaml` overlay `default.yaml`):
   - `subset` (built first, used for all development): train Spanish GP 2019–2023, val 2024, test 2025.
   - `full` (final model): train Spanish GP 2005–2023, same val 2024 / test 2025 so both profiles are directly comparable.
   - FastF1 tyre data only exists from 2018. Pre-2018 laps come from Kaggle (or Ergast fallback) lap times + pit stops: `stint_source="inferred_pitstops"`, `tyre_age = laps since pit` (labelled inferred), `compound="UNKNOWN"`, no weather/track-status (clean-lap filter falls back to pit-in/out, lap 1, 107% outlier, and race-control-free heuristics such as field-wide slow-lap detection for SC).
   - Add an `era` feature (`refuel_2005_2009`, `pre_tyre_data_2010_2017`, `tyre_data_2018_plus`) and a config flag `exclude_refuel_era` (default false) — refuelling invalidates the fuel correction; 2005 had no in-race tyre changes. Fuel correction is skipped for refuel-era laps.
   - Models must handle `compound="UNKNOWN"` (one-hot category for poly; native categorical for GBM). Strategy/rollouts and the dashboard only serve real-compound races (2018+).
   - The Evaluation page and `metrics.json` report both profiles side by side on the same val/test races, so the effect of extra history is measured, not assumed.
2. **Kaggle**: the user put `KAGGLE_API_TOKEN` in the root `.env`. `KaggleClient` loads `.env` into the subprocess environment only (never prints/logs/copies the value) and runs the kaggle CLI (`kaggle` extra). If download fails for any reason, fall back to Ergast/jolpica and log it. Kaggle is the preferred source for pre-2018 lap times.
3. **Notebooks**: keep notebooks 01–05 plus HTML exports in `reports/`.
4. **Environment**: plain venv (`/opt/homebrew/bin/python3.12 -m venv .venv`), no uv. `git init` approved — `.gitignore` (including `.env`) must exist before anything is staged; no commits unless the user asks.
5. **Defaults D1–D15**: accepted as written (D1/D2 extended by item 1 for the `full` profile).
