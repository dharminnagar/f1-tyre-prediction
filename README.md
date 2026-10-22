# BoxBox

Tyre degradation, remaining competitive life and pit-window estimation for the Spanish Grand Prix
(Circuit de Barcelona-Catalunya). The ML package `boxbox` (`ml/`) builds artifacts. The FastAPI app
`boxbox_api` (`api/`) serves them. The React dashboard (`dashboard/`) shows them in three pages:
Pit wall, Degradation and Evaluation. The design is in `PLAN.md`.

## Setup

Requirements: Homebrew `python3.12`, Node 24 and pnpm. Optional: put a `KAGGLE_API_TOKEN` in `.env`.

```bash
make setup        # .venv (python3.12) + pip install -r requirements.txt -e ml -e api + pnpm install
```

## Run

```bash
make all          # setup, download, then the pipeline for both profiles, then notebooks -> reports/
make dev          # API on :8000 and dashboard on http://127.0.0.1:5173
```

Individual steps (`PROFILES="subset"` limits any step to one profile):

```bash
make download     # raw sources -> dataset/ (idempotent; records the source used per season)
make dataset      # clean lap table + features
make train        # baseline curve, polynomial ridge, monotone gradient boosting
make evaluate     # val 2024 (model selection), test 2025 (scored once), LORO-CV, sensitivity
make export       # artifacts/<profile>/ + artifacts/metrics.json (profiles side by side)
make report       # execute ml/notebooks/01-05 -> reports/*.html
make gen-types    # OpenAPI dump -> dashboard/src/api/schema.d.ts
make check        # ruff, mypy --strict, pytest; eslint, tsc, vitest
```

CLI: `.venv/bin/boxbox {download,build,train,evaluate,export,all} --profile subset|full`.

## Profiles

| Profile | Train | Val | Test |
|---|---|---|---|
| `subset` | Spanish GP 2019–2023 | 2024 | 2025 |
| `full` | Spanish GP 2005–2023 | 2024 | 2025 |

Before 2018 there is no tyre data, so stints are inferred from pit stops, the compound is `UNKNOWN`,
and an `era` feature separates the refuelling era and the eras before and after tyre data.

## Data sources

Each season's actual sources are recorded in `dataset/raw/sources.json` and in each manifest.

- Lap times, positions, pit stops, results and qualifying come from the Kaggle
  `jtrotman/formula-1-race-data` dump (Ergast format). The fallback is Ergast via jolpica.
- Tyre data for 2018 onwards is tried in this order: FastF1, then OpenF1 (2023+), then a Kaggle
  FastF1 export (`aryagowda121/summary-of-f1-telemetrydata-2018-2022`).
- F1DB is optional: `boxbox download --with-f1db`.

## Configuration

`configs/default.yaml` holds the shared settings, and `configs/{subset,full}.yaml` override them per
profile. The API reads `BOXBOX_ARTIFACTS_DIR` and `BOXBOX_DEFAULT_PROFILE` (see `.env.example`).
