#!/usr/bin/env python3
"""
Step (a) of the scored pipeline: read whatever CSVs are in ``data/`` and generate
the features the model needs.

This deterministically ingests the three heterogeneous ad-platform schemas into a
canonical panel, attaches the campaign taxonomy, trims any incomplete trailing
extract, and builds the dense segment/campaign daily panels + recent shares. The
result (a :class:`preprocess.PreparedData`) is serialised for the predict step.

Everything here is closed-form (means, medians, group-bys) — no model fitting —
so it respects the "no retraining at eval" contract while still adapting to a
different advertiser / date range / size in the held-out ``data/``.

Usage:
    python src/generate_features.py --data-dir ./data --out ./features.pkl
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from ingest import load_canonical  # noqa: E402
from logging_utils import get_logger  # noqa: E402
from preprocess import prepare  # noqa: E402

log = get_logger("generate_features")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate features from data/ for prediction.")
    parser.add_argument("--data-dir", default=str(config.DEFAULT_DATA_DIR))
    parser.add_argument("--out", default=str(config.DEFAULT_FEATURES_PATH))
    args = parser.parse_args(argv)

    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    log.info("Ingesting data from %s", args.data_dir)
    panel = load_canonical(args.data_dir)
    prepared = prepare(panel)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(prepared, out, compress=3)
    log.info("Features written -> %s | origin=%s | segments=%d | campaigns=%d",
             out, prepared.origin.date(), len(prepared.segments),
             prepared.campaign_meta.shape[0] if prepared.campaign_meta is not None else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
