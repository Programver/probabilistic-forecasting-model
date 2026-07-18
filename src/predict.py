#!/usr/bin/env python3
"""
Step (b) of the scored pipeline: load the pickled model + generated features and
write ``predictions.csv``.

No fitting happens here — the frozen model's transferable priors are applied to
the current data's levels/seasonality, then a seeded Monte-Carlo simulation
produces probabilistic aggregate-period forecasts (30/60/90 days) for revenue,
spend and ROAS across the full hierarchy (aggregate → channel → campaign-type →
campaign). Output is written fresh in the contract's schema.

Usage:
    python src/predict.py --features ./features.pkl \
        --model ./pickle/model.pkl --output ./output/predictions.csv
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from forecast import run_forecast  # noqa: E402
from io_utils import write_predictions  # noqa: E402
from logging_utils import get_logger  # noqa: E402
from model import load_model  # noqa: E402

log = get_logger("predict")


def _write_summary_json(result, model, path: Path) -> None:
    """Bonus machine-readable summary for the app/insights (not required for scoring)."""
    try:
        agg = result.get_group("aggregate")
        summary = {
            "forecast_origin": result.origin.date().isoformat(),
            "model_version": model.version,
            "horizons": result.horizons,
            "headline": {},
        }
        pred = result.predictions
        for h in result.horizons:
            rev = pred[(pred.level == "aggregate") & (pred.metric == "revenue") &
                       (pred.horizon_days == h)]
            roas = pred[(pred.level == "aggregate") & (pred.metric == "roas") &
                        (pred.horizon_days == h)]
            if not rev.empty and not roas.empty:
                summary["headline"][f"{h}d"] = {
                    "revenue_p50": float(rev.iloc[0]["p50"]),
                    "revenue_p10": float(rev.iloc[0]["p10"]),
                    "revenue_p90": float(rev.iloc[0]["p90"]),
                    "roas_p50": float(roas.iloc[0]["p50"]),
                    "roas_p10": float(roas.iloc[0]["p10"]),
                    "roas_p90": float(roas.iloc[0]["p90"]),
                }
        path.write_text(json.dumps(summary, indent=2))
        log.info("wrote forecast summary -> %s", path)
    except Exception as exc:  # never let the bonus artifact break the run
        log.warning("could not write summary json: %s", exc)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Predict from features + pickled model.")
    parser.add_argument("--features", default=str(config.DEFAULT_FEATURES_PATH))
    parser.add_argument("--model", default=str(config.DEFAULT_MODEL_PATH))
    parser.add_argument("--output", default=str(config.DEFAULT_OUTPUT_PATH))
    args = parser.parse_args(argv)

    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    log.info("Loading features %s and model %s", args.features, args.model)
    prepared = joblib.load(args.features)
    model = load_model(args.model)

    log.info("Running probabilistic forecast (origin=%s, n_sims=%d) ...",
             prepared.origin.date(), model.cfg.n_simulations)
    result = run_forecast(prepared, model)

    out = Path(args.output)
    write_predictions(result.predictions, out)
    _write_summary_json(result, model, out.parent / "forecast_summary.json")

    # Console headline so a human watching the run sees a sane result.
    pred = result.predictions
    for h in result.horizons:
        r = pred[(pred.level == "aggregate") & (pred.metric == "revenue") &
                 (pred.horizon_days == h)]
        rs = pred[(pred.level == "aggregate") & (pred.metric == "roas") &
                  (pred.horizon_days == h)]
        if not r.empty and not rs.empty:
            msg = (f"  {h:2d}d revenue P50={r.iloc[0]['p50']:,.0f} "
                   f"[P10={r.iloc[0]['p10']:,.0f} .. P90={r.iloc[0]['p90']:,.0f}] "
                   f"| ROAS P50={rs.iloc[0]['p50']:.2f}")
            log.info(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
