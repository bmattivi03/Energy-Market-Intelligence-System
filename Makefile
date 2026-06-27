# Energy Market Intelligence System — pipeline orchestration.
#
#   make all                 # splits -> A -> B -> C (assumes raw + imputed exist)
#   make pipeline            # smart full run with skip-if-exists (scripts/run_pipeline.py)
#   make module_b SEED=7     # one stage, custom seed
#   make test                # full test suite
#   make clean               # remove caches/artifacts
#
# Expensive prerequisites (run explicitly, not part of `all`):
#   make ingest              # ENTSO-E/weather/fuels download (~30-60 min)
#   make impute              # Glocal-IB SSHB imputation (retrains the imputer)

SEED ?= 42
export PYTHONPATH := $(CURDIR)/src
export PYTORCH_ENABLE_MPS_FALLBACK := 1
# Module C imports CatBoost (Module B) and torch (stable-baselines3) in one
# process; on macOS their bundled OpenMP runtimes collide. KMP_DUPLICATE_LIB_OK
# silences the abort; OMP_NUM_THREADS=1 prevents the SIGSEGV that otherwise hits
# torch during sustained RL training. Both are needed.
export KMP_DUPLICATE_LIB_OK := TRUE
export OMP_NUM_THREADS := 1
PY := python

.PHONY: help all pipeline ingest ingest-fresh impute impute-sshb splits \
        module_a module_b module_c ablation test clean

help:
	@echo "Targets:"
	@echo "  all         splits -> module_a -> module_b -> module_c (data must be prepped)"
	@echo "  pipeline    full run with skip-if-exists caching (scripts/run_pipeline.py)"
	@echo "  ingest      download raw data (~30-60 min)      ingest-fresh ignores checkpoint"
	@echo "  impute      Glocal-IB SSHB imputation            impute-sshb  SSHB inference only"
	@echo "  splits      build train/val/test from imputed parquet"
	@echo "  module_a    train -> calibrate -> export load quantiles"
	@echo "  module_b    train CatBoost + CQR price forecaster"
	@echo "  module_c    train PPO/SAC battery agents (consumes Module B)"
	@echo "  ablation    Module C raw vs B vs B+A observation ablation"
	@echo "  test        run pytest      clean   remove caches/artifacts"

# Model-building chain (assumes data/splits/* already exist).
all: splits module_a module_b module_c

# Smart full pipeline (ingest -> impute -> splits -> A -> B -> C), skips done stages.
pipeline:
	$(PY) scripts/run_pipeline.py --seed $(SEED)

ingest:
	$(PY) src/ingestion/run_ingestion.py

ingest-fresh:
	$(PY) src/ingestion/run_ingestion.py --fresh

impute:
	$(PY) -m preprocessing.impute --seed $(SEED) --apply-constraints

impute-sshb:
	$(PY) -m preprocessing.impute --seed $(SEED) --skip-train --sshb --apply-constraints

splits:
	$(PY) -m preprocessing.build_splits

module_a:
	$(PY) -m module_a.train --seed $(SEED)
	$(PY) -m module_a.calibrate
	$(PY) -m module_a.export_parquet

module_b:
	$(PY) -m module_b.train --seed $(SEED)

module_c:
	$(PY) -m module_c.run

ablation:
	$(PY) -m module_c.ablation

test:
	$(PY) -m pytest tests/ -q

clean:
	rm -rf catboost_info .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
