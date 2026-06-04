"""Quick CLI to run full_report on val and test splits.

PYTHONPATH=src python -m module_a.eval_report
"""

from __future__ import annotations

import pathlib

from data.loaders import load_test, load_val
from module_a.evaluation import full_report
from module_a.features import ALL_BUNDLES, TARGET_COL, build_features
from module_a.model import MultiScaleLSTMForecaster

CKPT = pathlib.Path(__file__).parents[2] / "checkpoints" / "module_a" / "best.pt"


def main() -> None:
    print(f"Loading checkpoint: {CKPT}")
    forecaster = MultiScaleLSTMForecaster.load(CKPT)

    for split_name, loader in [("val", load_val), ("test", load_test)]:
        raw = loader()
        feat = build_features(raw, bundles=list(ALL_BUNDLES))
        long_preds = forecaster.predict_quantiles(feat, stride=1)
        actual = raw[TARGET_COL]
        full_report(long_preds, actual, split_name=split_name)


if __name__ == "__main__":
    main()
