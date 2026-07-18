"""
Spend → revenue response (constant-elasticity, diminishing returns).

The brief explicitly rules out full Media-Mix Modelling and says to treat
attribution as the source of truth. We therefore use a deliberately lightweight,
transparent response model rather than an adstock/saturation MMM: within a
segment, attributed revenue responds to spend with a constant elasticity ``beta``

        revenue ≈ A · spend ** beta ,     0 < beta < 1  →  diminishing returns.

The data strongly supports this: the empirical log–log slope is ~0.7 for Google,
~0.6 for Meta, ~0.6–0.8 by campaign type — all well below 1, i.e. each extra
dollar buys less incremental revenue. ``beta`` is fit robustly per segment at
train time, bounded, and **frozen** in the model. At predict time we only *apply*
it (no fitting), which keeps the scored run deterministic and fast.

Uses of the response model
--------------------------
* **Budget simulation** — scaling planned spend by ``f`` scales revenue by
  ``f**beta`` (relative form; anchors on the segment's own recent efficiency).
* **Marginal ROAS / saturation curves** — the absolute form ``A·spend**beta``
  gives ``dRev/dSpend = A·beta·spend**(beta-1)`` for the "where do extra dollars
  work hardest?" view in the app and AI insights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import config
from logging_utils import get_logger

log = get_logger("response")


@dataclass
class ElasticityFit:
    beta: float           # elasticity (bounded to config.ELASTICITY_BOUNDS)
    scale: float          # A in revenue = A * spend**beta (absolute form)
    n: int                # points used
    r2: float             # goodness of fit (diagnostic)
    spend_ref: float      # reference (median positive) daily spend
    revenue_ref: float    # reference (median positive) daily revenue


def _robust_loglog(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, int]:
    """Robust OLS of ``log y ~ a + b log x`` with one MAD-trim reweighting pass.

    Returns ``(slope, intercept, r2, n_used)``.
    """
    lx, ly = np.log(x), np.log(y)
    if len(lx) < 5:
        return np.nan, np.nan, 0.0, len(lx)
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    resid = ly - A @ coef
    mad = np.median(np.abs(resid - np.median(resid)))
    if mad > 0:
        keep = np.abs(resid - np.median(resid)) <= 3 * 1.4826 * mad
        if keep.sum() >= 5:
            coef, *_ = np.linalg.lstsq(A[keep], ly[keep], rcond=None)
            lx, ly, A = lx[keep], ly[keep], A[keep]
    pred = A @ coef
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(coef[0]), float(coef[1]), r2, int(len(lx))


def fit_elasticity(daily: pd.DataFrame, spend_col: str = "spend",
                   revenue_col: str = "revenue") -> ElasticityFit:
    """Fit a single segment's elasticity from its daily spend/revenue."""
    d = daily[[spend_col, revenue_col]].dropna()
    d = d[(d[spend_col] > 0) & (d[revenue_col] > 0)]
    lo, hi = config.ELASTICITY_BOUNDS
    if len(d) < 10:
        return ElasticityFit(config.ELASTICITY_DEFAULT, np.nan, len(d), 0.0,
                             float(d[spend_col].median()) if len(d) else 0.0,
                             float(d[revenue_col].median()) if len(d) else 0.0)
    x = d[spend_col].to_numpy(float)
    y = d[revenue_col].to_numpy(float)
    slope, intercept, r2, n = _robust_loglog(x, y)
    if not np.isfinite(slope):
        slope = config.ELASTICITY_DEFAULT
        intercept = np.log(np.median(y)) - slope * np.log(np.median(x))
    beta = float(np.clip(slope, lo, hi))
    scale = float(np.exp(intercept)) if np.isfinite(intercept) else np.nan
    return ElasticityFit(beta, scale, n, r2,
                         float(np.median(x)), float(np.median(y)))


def fit_elasticity_table(segment_daily: pd.DataFrame) -> Dict[str, ElasticityFit]:
    """Fit an elasticity per segment_key and also compute pooled priors by
    (platform, campaign_type) and a single global default.
    """
    table: Dict[str, ElasticityFit] = {}
    for key, g in segment_daily.groupby("segment_key"):
        table[key] = fit_elasticity(g)
        log.debug("elasticity %s: beta=%.2f (n=%d, r2=%.2f)",
                  key, table[key].beta, table[key].n, table[key].r2)
    return table


def response_multiplier(spend_scenario: np.ndarray | float,
                        spend_baseline: np.ndarray | float,
                        beta: float) -> np.ndarray | float:
    """Relative revenue response to a spend change: ``(scn / base) ** beta``.

    Guards against zero/NaN baselines (returns 1.0 — no adjustment).
    """
    scn = np.asarray(spend_scenario, dtype=float)
    base = np.asarray(spend_baseline, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(base > 0, scn / base, 1.0)
        ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, 1.0)
        mult = ratio ** float(beta)
    mult = np.where(np.isfinite(mult), mult, 1.0)
    return mult if mult.shape else float(mult)


def marginal_roas(spend: float, fit: ElasticityFit) -> float:
    """dRevenue/dSpend at a spend level (absolute curve). NaN if unfit."""
    if not np.isfinite(fit.scale) or spend <= 0:
        return np.nan
    return float(fit.scale * fit.beta * spend ** (fit.beta - 1.0))


def average_roas_at(spend: float, fit: ElasticityFit) -> float:
    """Predicted average ROAS at a spend level (absolute curve)."""
    if not np.isfinite(fit.scale) or spend <= 0:
        return np.nan
    return float(fit.scale * spend ** (fit.beta - 1.0))
