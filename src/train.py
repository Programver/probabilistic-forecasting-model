#!/usr/bin/env python3
"""
Train the forecasting model and write the committed artifact ``pickle/model.pkl``.

This runs **once, offline, at development time** — never during evaluation. The
eval pipeline only generates features and predicts against the pickle produced
here. Training does the actual learning: pooled seasonal shapes, spend
elasticities, and rolling-origin conformal calibration of the prediction
intervals.

Usage:
    python src/train.py --data-dir ./data --out ./pickle/model.pkl
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

# Ensure sibling modules import cleanly whether run as a script or a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from backtest import calibrate  # noqa: E402
from ingest import load_canonical  # noqa: E402
from logging_utils import get_logger  # noqa: E402
from model import fit_global_priors, save_model  # noqa: E402
from preprocess import prepare  # noqa: E402

log = get_logger("train")


def set_seeds(seed: int = config.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the AIgnition forecasting model.")
    parser.add_argument("--data-dir", default=str(config.DEFAULT_DATA_DIR))
    parser.add_argument("--out", default=str(config.DEFAULT_MODEL_PATH))
    parser.add_argument("--skip-backtest", action="store_true",
                        help="Skip conformal calibration (faster; intervals default to identity).")
    parser.add_argument("--backtest-sims", type=int, default=1500)
    args = parser.parse_args(argv)

    set_seeds()
    log.info("Loading data from %s", args.data_dir)
    panel = load_canonical(args.data_dir)
    prepared = prepare(panel)

    log.info("Fitting global priors (seasonality, elasticities) ...")
    model = fit_global_priors(prepared)

    if not args.skip_backtest:
        log.info("Calibrating prediction intervals via rolling-origin backtest ...")
        report = calibrate(prepared, model, n_sims=args.backtest_sims)
        model.conformal = report.conformal
        if not report.metrics.empty:
            model.trained_meta["backtest_metrics"] = report.metrics.to_dict("records")
        log.info("Conformal scales by horizon: %s", model.conformal.scale_by_horizon)

    save_model(model, args.out)
    log.info("Training complete. Model version %s trained at origin %s.",
             model.version, model.trained_origin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
