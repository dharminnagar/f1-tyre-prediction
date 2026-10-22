# BoxBox — single entry point. PROFILES defaults to both training profiles.
PYTHON   := /opt/homebrew/bin/python3.12
VENV     := .venv
BIN      := $(VENV)/bin
PROFILES ?= subset full
API_PORT ?= 8000

.PHONY: setup download dataset train evaluate export pipeline report api gen-types dashboard dev \
        lint typecheck test check all clean-artifacts

setup:
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q -r requirements.txt -e ./ml -e ./api
	pnpm -C dashboard install --frozen-lockfile

download:
	@for p in $(PROFILES); do $(BIN)/boxbox download --profile $$p || exit 1; done

dataset:
	@for p in $(PROFILES); do $(BIN)/boxbox build --profile $$p || exit 1; done

train:
	@for p in $(PROFILES); do $(BIN)/boxbox train --profile $$p || exit 1; done

evaluate:
	@for p in $(PROFILES); do $(BIN)/boxbox evaluate --profile $$p || exit 1; done

export:
	@for p in $(PROFILES); do $(BIN)/boxbox export --profile $$p || exit 1; done

pipeline:
	@for p in $(PROFILES); do $(BIN)/boxbox all --profile $$p || exit 1; done

report:
	mkdir -p reports
	$(BIN)/jupyter nbconvert --execute --to html --output-dir reports ml/notebooks/*.ipynb

api:
	$(BIN)/uvicorn boxbox_api.main:create_app --factory --host 127.0.0.1 --port $(API_PORT)

gen-types:
	$(BIN)/python -m boxbox_api.openapi_dump api/openapi.json
	pnpm -C dashboard gen-types

dashboard:
	pnpm -C dashboard dev

dev:
	$(MAKE) -j2 api dashboard

lint:
	$(BIN)/ruff check ml api
	$(BIN)/ruff format --check ml api
	pnpm -C dashboard lint

typecheck:
	$(BIN)/mypy
	pnpm -C dashboard typecheck

test:
	$(BIN)/pytest
	pnpm -C dashboard test

check: lint typecheck test

all: setup download pipeline report

clean-artifacts:
	rm -rf artifacts/subset artifacts/full artifacts/metrics.json
