"""
Per-series structural decomposition and forward projection.

Given a dense daily series (revenue or spend) and a multiplicative seasonal
profile, we decompose it into

        value_t ≈ level · growth(t) · seasonal(t) · residual_t

and expose the pieces needed to project forward and to simulate:

* **level**   — robust, seasonally-adjusted "current" daily level (winsorised
  trailing mean), with an estimate of its own sampling uncertainty.
* **growth**  — a *damped* year-over-year trend (falls back to a shrunk linear
  slope when <1 year of history), so long-horizon extrapolation saturates
  instead of exploding — the single most common way naive trend models blow up.
* **residual**— the multiplicative noise, kept both as a block of recent samples
  (for a block bootstrap that preserves autocorrelation and fat tails) and as a
  log-scale sigma fallback.

All estimation here is closed-form / deterministic (means, medians, one robust
regression) — there is no iterative optimisation, so it is safe to run inside the
"feature generation" step of the scored pipeline without violating "no retrain".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from logging_utils import get_logger
from seasonality import SeasonalProfile, factor_for_dates

log = get_logger("decompose")


@dataclass
class Decomposition:
    value_col: str
    level: float                       # seasonally-adjusted current daily level
    level_log_sigma: float             # uncertainty of the level estimate (log scale)
    growth_annual: float               # YoY growth rate (e.g. 0.10 = +10%/yr)
    growth_sigma: float                # uncertainty on growth
    resid_log_sigma: float             # day-to-day multiplicative noise (log scale)
    resid_block: np.ndarray            # recent multiplicative residuals (for bootstrap)
    n_days: int
    n_positive: int
    mean_value: float                  # historical daily mean (diagnostic / fallbacks)
    sparse: bool = False

    def copy(self) -> "Decomposition":
        return Decomposition(self.value_col, self.level, self.level_log_sigma,
                             self.growth_annual, self.growth_sigma, self.resid_log_sigma,
                             self.resid_block.copy(), self.n_days, self.n_positive,
                             self.mean_value, self.sparse)


def _winsorized_mean(x: np.ndarray, p: float = 0.10) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    lo, hi = np.quantile(x, [p, 1 - p])
    return float(np.clip(x, lo, hi).mean())


def decompose_series(daily: pd.DataFrame, seasonal: SeasonalProfile,
                     value_col: str = "revenue",
                     cfg: config.ForecastConfig | None = None) -> Decomposition:
    cfg = cfg or config.ForecastConfig()
    d = daily.copy()
    if "date" in d.columns:
        d = d.set_index("date")
    d = d.sort_index()
    r = d[value_col].astype(float).clip(lower=0.0)
    n_days = int(len(r))
    n_pos = int((r > 0).sum())
    mean_value = float(r.mean()) if n_days else 0.0

    sparse = (n_days < config.MIN_DAYS_SEGMENT_MODEL) or (n_pos < 8) or (r.sum() <= 0)
    if sparse:
        lvl = _winsorized_mean(r.to_numpy()) if n_days else 0.0
        return Decomposition(value_col, lvl, 0.5, 0.0, 0.25, 0.7,
                             np.array([1.0]), n_days, n_pos, mean_value, sparse=True)

    seasonal_factor = factor_for_dates(seasonal, r.index)
    seasonal_factor = np.where(seasonal_factor > 0, seasonal_factor, 1.0)
    deseason = r.to_numpy() / seasonal_factor

    # --- Level: winsorised mean of deseasonalised value over recent window ---
    win = min(cfg.recent_level_window, n_days)
    recent_deseason = deseason[-win:]
    level = _winsorized_mean(recent_deseason)
    if level <= 0:
        level = max(_winsorized_mean(deseason), 1e-6)

    # --- Residual noise (log scale), from positive deseasonalised points ---
    pos = recent_deseason[recent_deseason > 0]
    if pos.size >= 5 and level > 0:
        log_resid = np.log(pos / level)
        resid_log_sigma = float(np.clip(np.std(log_resid), 0.05, 2.0))
    else:
        resid_log_sigma = 0.5
    # Level uncertainty: sigma of the mean, inflated for autocorrelation (eff N = N/5).
    eff_n = max(win / 5.0, 1.0)
    level_log_sigma = float(np.clip(resid_log_sigma / np.sqrt(eff_n), 0.02, 0.6))

    # --- Residual block for bootstrap (recent ratios incl. zeros) ---
    block_win = min(120, n_days)
    block = (r.to_numpy()[-block_win:] / (seasonal_factor[-block_win:] * level))
    block = block[np.isfinite(block)]
    if block.size < 10:
        block = np.array([1.0])
    resid_block = block

    # --- Growth: damped YoY, else shrunk linear slope ---
    growth_annual, growth_sigma = _estimate_growth(deseason, r.index, n_days)

    return Decomposition(value_col, float(level), level_log_sigma,
                         float(growth_annual), float(growth_sigma),
                         resid_log_sigma, resid_block, n_days, n_pos, mean_value, False)


def _estimate_growth(deseason: np.ndarray, index: pd.DatetimeIndex,
                     n_days: int) -> tuple[float, float]:
    """Year-over-year growth if we have >~13 months, else a shrunk linear trend."""
    if n_days >= 400:
        recent = _winsorized_mean(deseason[-56:])
        # window centred one year before the origin
        origin = index[-1]
        yr_ago_mask = np.asarray(
            (index >= origin - pd.Timedelta(days=365 + 28)) &
            (index <= origin - pd.Timedelta(days=365 - 28)))
        prior = _winsorized_mean(deseason[yr_ago_mask])
        if prior > 0 and recent > 0:
            g = recent / prior - 1.0
            return float(np.clip(g, -0.5, 0.8)), 0.15
    if n_days >= 120:
        # robust-ish linear slope of log(deseason_smoothed) vs day index
        y = pd.Series(deseason).clip(lower=1e-9)
        y = np.log(y.rolling(14, min_periods=5).mean().bfill().to_numpy())
        t = np.arange(len(y), dtype=float)
        A = np.vstack([t, np.ones_like(t)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        slope_per_day = float(coef[0])
        g = np.expm1(slope_per_day * 365.0)
        shrink = min(n_days / 400.0, 1.0)
        g = np.clip(g * shrink, -0.4, 0.6)
        return float(g), 0.25
    return 0.0, 0.2


def growth_multiplier(days_ahead: np.ndarray, growth_annual: float,
                      cfg: config.ForecastConfig | None = None) -> np.ndarray:
    """Damped-trend growth multiplier for each future day offset (1-indexed).

    The monthly growth increment is damped geometrically (factor ``phi`` per
    month) so cumulative growth saturates at ``g_month / (1 - phi)`` — bounded
    extrapolation. Result is additionally clipped to a sane [0.4, 2.5] band.
    """
    cfg = cfg or config.ForecastConfig()
    phi = cfg.trend_damping
    t = np.asarray(days_ahead, dtype=float)
    months = t / 30.0
    g_month = (1.0 + growth_annual) ** (1.0 / 12.0) - 1.0
    eff_months = (1.0 - phi ** months) / (1.0 - phi)
    log_growth = np.log1p(g_month) * eff_months
    mult = np.exp(log_growth)
    return np.clip(mult, 0.4, 2.5)


def expected_path(decomp: Decomposition, future_dates: pd.DatetimeIndex,
                  seasonal: SeasonalProfile,
                  cfg: config.ForecastConfig | None = None) -> np.ndarray:
    """Deterministic expected daily values over ``future_dates`` (point forecast)."""
    cfg = cfg or config.ForecastConfig()
    fdates = pd.DatetimeIndex(future_dates)
    days_ahead = np.arange(1, len(fdates) + 1, dtype=float)
    gmult = growth_multiplier(days_ahead, decomp.growth_annual, cfg)
    sfac = factor_for_dates(seasonal, fdates)
    sfac = np.where(sfac > 0, sfac, 1.0)
    return decomp.level * gmult * sfac
