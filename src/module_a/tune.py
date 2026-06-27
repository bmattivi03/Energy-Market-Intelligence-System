"""Optuna hyperparameter search for Module A (MultiScaleLSTM).

Supports single-GPU and multi-GPU parallel execution via shared SQLite storage.
Each worker process picks up independent trials from the same Optuna study.

Usage
-----
# Single GPU
PYTHONPATH=src python -m module_a.tune --n-trials 40 --epochs 100 --patience 15

# Multi-GPU: launch workers (each does 5 trials), then collect
bash src/module_a/tune_launch.sh   # see that script for details

# After all workers finish — retrain best config + export parquet
PYTHONPATH=src python -m module_a.tune --n-trials 0 --storage sqlite:///optuna_module_a.db
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

import numpy as np
import optuna
import torch

from data.loaders import load_train, load_val
from module_a.features import ALL_BUNDLES, build_features
from module_a.model import MultiScaleLSTMForecaster

PROJECT_ROOT = pathlib.Path(__file__).parents[2]
CKPT_PATH    = PROJECT_ROOT / "checkpoints" / "module_a" / "best.pt"
REPORT_PATH  = PROJECT_ROOT / "reports" / "module_a_optuna.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _val_pinball(forecaster: MultiScaleLSTMForecaster, val_feat) -> float:
    """Mean val pinball across q10/q50/q90 using the fitted scalers."""
    from module_a.model import LoadSequenceDataset, pinball_loss, QUANTILES
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forecaster._model.eval()
    forecaster._model.to(device)

    X_va, y_va = forecaster._to_arrays(val_feat)
    from module_a.model import LoadSequenceDataset
    ds = LoadSequenceDataset(X_va, y_va, stride=1)
    dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for x_s, x_l, y in dl:
            x_s, x_l, y = x_s.to(device), x_l.to(device), y.to(device)
            pred = forecaster._model(x_s, x_l)
            total_loss += pinball_loss(pred, y, QUANTILES).item() * len(y)
            n += len(y)
    return total_loss / n if n > 0 else float("inf")


def make_objective(train_feat, val_feat, max_epochs: int, patience: int, seed: int):
    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        hidden = trial.suggest_categorical("hidden", [64, 128, 192, 256])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        num_layers_short = trial.suggest_int("num_layers_short", 1, 3)
        num_layers_long = trial.suggest_int("num_layers_long", 1, 2)

        set_seed(seed + trial.number)

        forecaster = MultiScaleLSTMForecaster(
            hidden=hidden,
            dropout=dropout,
            num_layers_short=num_layers_short,
            num_layers_long=num_layers_long,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            random_state=seed + trial.number,
        )

        # Suppress per-epoch prints during search
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            forecaster.fit(train_feat, val_feat_df=val_feat)

        val_loss = _val_pinball(forecaster, val_feat)
        print(f"  trial {trial.number:3d}  val_pinball={val_loss:.5f}  "
              f"lr={lr:.2e}  hidden={hidden}  dropout={dropout:.2f}  "
              f"batch={batch_size}  layers=({num_layers_short},{num_layers_long})",
              flush=True)
        return val_loss

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for Module A")
    parser.add_argument("--n-trials",  type=int,   default=40)
    parser.add_argument("--epochs",    type=int,   default=100,
                        help="Max epochs per trial (shorter than full train)")
    parser.add_argument("--patience",  type=int,   default=15)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument(
        "--bundles", nargs="+", default=list(ALL_BUNDLES),
    )
    parser.add_argument(
        "--study-name", type=str, default="module_a_lstm",
    )
    parser.add_argument(
        "--storage", type=str, default=None,
        help="Optuna storage URL, e.g. sqlite:///optuna_module_a.db  "
             "Required for multi-GPU parallel workers sharing one study.",
    )
    parser.add_argument(
        "--no-retrain", action="store_true", default=False,
        help="Skip best-config retrain + parquet export. "
             "Use for worker processes; run without this flag once all workers finish.",
    )
    args = parser.parse_args()

    print("Loading and building features...")
    train_raw = load_train()
    val_raw   = load_val()
    train_feat = build_features(train_raw, bundles=args.bundles)
    val_feat   = build_features(val_raw,   bundles=args.bundles)
    print(f"  train: {train_feat.shape}  val: {val_feat.shape}")

    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,   # workers share existing study
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )

    if args.n_trials > 0:
        objective = make_objective(train_feat, val_feat, args.epochs, args.patience, args.seed)
        print(f"\nStarting {args.n_trials} trials (max {args.epochs} epochs each)...\n")
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    else:
        print(f"\nSkipping trials (--n-trials 0). Loading existing study results...")

    completed = [t for t in study.trials if t.value is not None]
    print(f"\n{len(completed)} completed trials in study.")
    best = study.best_trial
    print(f"Best trial #{best.number}  val_pinball={best.value:.5f}")
    print(f"  params: {best.params}")

    if args.no_retrain:
        print("\n--no-retrain set: skipping best-config retrain. Done.")
        return

    # Retrain with best params at full epochs and save checkpoint
    print("\nRetraining best config at full depth (200 epochs)...")
    set_seed(args.seed)
    best_forecaster = MultiScaleLSTMForecaster(
        hidden=best.params["hidden"],
        dropout=best.params["dropout"],
        num_layers_short=best.params["num_layers_short"],
        num_layers_long=best.params["num_layers_long"],
        lr=best.params["lr"],
        weight_decay=best.params["weight_decay"],
        batch_size=best.params["batch_size"],
        max_epochs=200,
        patience=20,
        random_state=args.seed,
        checkpoint_path=CKPT_PATH,
    )
    best_forecaster.fit(train_feat, val_feat_df=val_feat)

    # Export parquet (same as train.py)
    print("\nGenerating load_quantiles.parquet...")
    from data.loaders import load_test
    from module_a.train import OUTPUT_PARQUET, OUTPUT_DIR, _to_wide
    import pandas as pd

    test_raw  = load_test()
    test_feat = build_features(test_raw, bundles=args.bundles)
    train_long = best_forecaster.predict_quantiles(train_feat, stride=1)
    val_long   = best_forecaster.predict_quantiles(val_feat,   stride=1)
    test_long  = best_forecaster.predict_quantiles(test_feat,  stride=1)
    all_long   = pd.concat([train_long, val_long, test_long]).sort_index()
    wide       = _to_wide(all_long)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(OUTPUT_PARQUET)
    print(f"Saved: {OUTPUT_PARQUET}")

    # Save Optuna report
    all_trials = [
        {"number": t.number, "value": t.value, "params": t.params}
        for t in study.trials if t.value is not None
    ]
    report = {
        "best_trial": best.number,
        "best_val_pinball": best.value,
        "best_params": best.params,
        "n_trials": len(all_trials),
        "trials": all_trials,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Saved Optuna report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
