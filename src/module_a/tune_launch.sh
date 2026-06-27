#!/usr/bin/env bash
# Multi-GPU Optuna sweep for Module A.
# Uses 8 of the 9 available GPUs (leave GPU 8 free for anything else).
# Each worker runs 5 trials → 40 total, sharing one SQLite study DB.
#
# Usage (from repo root):
#   bash src/module_a/tune_launch.sh
#
# After all workers finish, run the collect step:
#   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m module_a.tune \
#     --n-trials 0 \
#     --storage sqlite:///optuna_module_a.db \
#     --study-name module_a_lstm

set -e

N_GPUS=8
TRIALS_PER_GPU=5
EPOCHS=100
PATIENCE=15
STORAGE="sqlite:///optuna_module_a.db"
STUDY="module_a_lstm"
PYTHONPATH_VAL="src"

echo "Launching ${N_GPUS} workers x ${TRIALS_PER_GPU} trials = $((N_GPUS * TRIALS_PER_GPU)) total"
echo "Storage: ${STORAGE}"
echo ""

# Pre-initialise the SQLite DB with a single process so all workers find
# the schema already created — avoids the race condition on create_all.
echo "Initialising Optuna study..."
PYTHONPATH=${PYTHONPATH_VAL} python - <<EOF
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
optuna.create_study(
    study_name="${STUDY}",
    storage="${STORAGE}",
    direction="minimize",
    load_if_exists=True,
)
print("  Study ready: ${STORAGE}")
EOF

PIDS=()
for i in $(seq 0 $((N_GPUS - 1))); do
    SEED=$((42 + i * 100))   # different seed per worker → diverse initial samples
    LOG="logs/tune_gpu${i}.log"
    mkdir -p logs

    CUDA_VISIBLE_DEVICES=${i} PYTHONPATH=${PYTHONPATH_VAL} \
        python -u -m module_a.tune \
            --n-trials   ${TRIALS_PER_GPU} \
            --epochs     ${EPOCHS} \
            --patience   ${PATIENCE} \
            --seed       ${SEED} \
            --storage    "${STORAGE}" \
            --study-name "${STUDY}" \
            --no-retrain \
        > "${LOG}" 2>&1 &

    PIDS+=($!)
    echo "  GPU ${i} → PID ${PIDS[-1]}  (seed=${SEED}, log=${LOG})"
done

echo ""
echo "Waiting for all workers..."
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  GPU ${i} done"
    else
        echo "  GPU ${i} FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

if [ "${FAILED}" -gt 0 ]; then
    echo "${FAILED} worker(s) failed. Check logs/tune_gpu*.log"
    exit 1
fi

echo ""
echo "All workers finished. Running collect + retrain..."
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH_VAL} \
    python -m module_a.tune \
        --n-trials   0 \
        --storage    "${STORAGE}" \
        --study-name "${STUDY}"

echo ""
echo "Done. Bring back:"
echo "  checkpoints/module_a/best.pt"
echo "  reports/module_a_optuna.json"
echo "  data/module_a/load_quantiles.parquet"
