"""
Turning simulated distributions into the submission's ``predictions.csv``.

The output is a **long / tidy** table (one row per horizon × hierarchy-node ×
metric) whose columns are defined once in :data:`config.PREDICTION_COLUMNS`. This
single structure encodes every deliverable the brief asks for — aggregate revenue
and blended ROAS, plus channel / campaign-type / campaign revenue and ROAS ranges
— each with a mean and a full quantile fan (P5…P95).

If the organizers publish a different exact schema, only :func:`write_predictions`
and ``config.PREDICTION_COLUMNS`` change; the forecasting engine is untouched.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from config import q_col
from hierarchy import GroupForecast
from logging_utils import get_logger
from uncertainty import apply_conformal

log = get_logger("io")


def summarize(x: np.ndarray, quantiles: List[float], scale: float = 1.0) -> Dict[str, float]:
    """Mean, std and quantiles of a simulated distribution (conformal-widened)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    out: Dict[str, float] = {"mean": np.nan, "std": np.nan}
    for q in quantiles:
        out[q_col(q)] = np.nan
    if x.size == 0:
        return out
    # Mean is taken from the RAW distribution so it stays additive across the
    # hierarchy (channel means sum to the aggregate mean). Conformal only
    # recalibrates the *spread*; it is median-preserving, so the P50 is unchanged.
    out["mean"] = float(np.mean(x))
    xs = apply_conformal(x, scale) if scale != 1.0 else x
    out["std"] = float(np.std(xs))
    qs = np.quantile(xs, quantiles)
    for q, v in zip(quantiles, qs):
        out[q_col(q)] = float(v)
    return out


def _currency_for(metric: str) -> str:
    return "ratio" if metric == "roas" else config.CURRENCY


def groups_to_frame(groups: List[GroupForecast], origin: pd.Timestamp,
                    model, cfg: Optional[config.ForecastConfig] = None) -> pd.DataFrame:
    """Build the tidy predictions frame from reconciled hierarchy groups."""
    cfg = cfg or config.ForecastConfig()
    quantiles = cfg.quantiles
    rows: List[dict] = []
    origin = pd.Timestamp(origin)

    for g in groups:
        for h in cfg.horizons:
            if h not in g.dist:
                continue
            period_start = (origin + pd.Timedelta(days=1)).date().isoformat()
            period_end = (origin + pd.Timedelta(days=h)).date().isoformat()
            scale = model.conformal.get(h, g.level) if model is not None else 1.0
            for metric in config.METRICS:
                stats = summarize(g.dist[h][metric], quantiles, scale)
                row = {
                    "forecast_origin": origin.date().isoformat(),
                    "horizon_days": int(h),
                    "period_start": period_start,
                    "period_end": period_end,
                    "level": g.level,
                    "channel": g.channel,
                    "campaign_type": g.campaign_type,
                    "campaign_id": g.campaign_id,
                    "campaign_name": g.campaign_name,
                    "metric": metric,
                    "currency": _currency_for(metric),
                    **stats,
                }
                rows.append(row)

    frame = pd.DataFrame(rows)
    # Guarantee exact column order/presence per the contract.
    for col in config.PREDICTION_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan
    frame = frame[config.PREDICTION_COLUMNS]
    # Deterministic ordering (aids scoring/diffing).
    level_rank = {lvl: i for i, lvl in enumerate(config.LEVELS)}
    frame = frame.assign(_lr=frame["level"].map(level_rank)).sort_values(
        ["_lr", "channel", "campaign_type", "campaign_id", "horizon_days", "metric"]
    ).drop(columns="_lr").reset_index(drop=True)
    return frame


def write_predictions(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write predictions fresh (never append). Rounds money to cents, ROAS to 4dp."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    num_cols = ["mean", "std"] + [q_col(q) for q in config.QUANTILES]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(4)
    out.to_csv(path, index=False)
    log.info("wrote %d prediction rows -> %s", len(out), path)
    return path
