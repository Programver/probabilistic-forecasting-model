"""
Monte-Carlo simulation of coupled revenue & spend daily paths per segment.

Why simulation (rather than a closed-form interval)? Because the target is an
*aggregate-period* quantity (30/60/90-day sums) and a *ratio* (ROAS = ΣRev/ΣSpend)
whose distribution is not analytically tractable once we layer seasonality,
damped trend, autocorrelated noise, level uncertainty and a non-linear spend
response. Simulating daily paths and summing them handles all of that exactly and
— crucially — lets us **sum the same simulated world across segments** to get a
hierarchy that is coherent by construction (segment paths add up to channel and
total paths) with correct cross-segment correlation.

Each simulated "world" (sim index) draws:
  * a level shock            (lognormal, from level-estimate uncertainty)
  * a growth shock           (normal, from trend uncertainty)
  * a residual noise path     (block bootstrap, autocorrelated & fat-tailed)
  * a shared macro shock      (optional, common across segments → correlation)
Revenue is then coupled to spend through the frozen elasticity, so a world that
spends more also earns more — sub-linearly (diminishing returns).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import config
from decompose import Decomposition, growth_multiplier
from response import response_multiplier
from seasonality import SeasonalProfile, factor_for_dates
from uncertainty import block_bootstrap


@dataclass
class SegmentSim:
    segment_key: str
    platform: str
    campaign_type: str
    future_dates: pd.DatetimeIndex
    revenue_daily: np.ndarray   # (n_sims, horizon)
    spend_daily: np.ndarray     # (n_sims, horizon)


def _draw_expected(decomp: Decomposition, seasonal: SeasonalProfile,
                   future_dates: pd.DatetimeIndex, n_sims: int,
                   rng: np.random.Generator, cfg: config.ForecastConfig) -> np.ndarray:
    """Expected daily values per sim with level & growth uncertainty ((n_sims, H))."""
    H = len(future_dates)
    days_ahead = np.arange(1, H + 1, dtype=float)
    sfac = factor_for_dates(seasonal, future_dates)
    sfac = np.where(sfac > 0, sfac, 1.0)[None, :]                       # (1, H)

    level_shock = rng.normal(0.0, decomp.level_log_sigma, size=(n_sims, 1))
    level_s = decomp.level * np.exp(level_shock)                         # (n_sims, 1)

    growth_s = rng.normal(decomp.growth_annual, decomp.growth_sigma, size=n_sims)
    growth_s = np.clip(growth_s, -0.6, 0.9)
    gmult = np.stack([growth_multiplier(days_ahead, g, cfg) for g in growth_s])  # (n_sims, H)

    return level_s * gmult * sfac


def simulate_segment(sm, future_dates: pd.DatetimeIndex, rng: np.random.Generator,
                     cfg: Optional[config.ForecastConfig] = None,
                     spend_override_daily: Optional[np.ndarray] = None,
                     common_log_shock: Optional[np.ndarray] = None) -> SegmentSim:
    """Simulate one segment. ``sm`` is a SegmentModel (duck-typed).

    Parameters
    ----------
    spend_override_daily : optional
        If given (shape ``(H,)`` or ``(n_sims, H)``), spend follows this scenario
        (budget simulation) instead of its own forecast; revenue responds via
        elasticity relative to the segment's baseline expected spend.
    common_log_shock : optional
        Shared macro log-shock ``(n_sims, H)`` applied to revenue across all
        segments to induce realistic cross-segment correlation.
    """
    cfg = cfg or config.ForecastConfig()
    n_sims = cfg.n_simulations
    H = len(future_dates)
    fdates = pd.DatetimeIndex(future_dates)

    # --- Spend ---
    baseline_spend = _draw_expected(sm.spend_decomp, sm.seasonal_spend, fdates, 1, rng, cfg)
    baseline_spend = np.maximum(baseline_spend, 1e-9)                    # (1, H) anchor
    if spend_override_daily is not None:
        ov = np.asarray(spend_override_daily, dtype=float)
        if ov.ndim == 1:
            ov = ov[None, :]
        spend_daily = np.broadcast_to(ov, (n_sims, H)).copy()
        # small execution noise so planned spend isn't perfectly deterministic
        spend_daily *= np.exp(rng.normal(0.0, 0.03, size=(n_sims, H)))
    else:
        exp_spend = _draw_expected(sm.spend_decomp, sm.seasonal_spend, fdates, n_sims, rng, cfg)
        # spend is agency-controlled → damp its residual noise vs revenue
        s_resid = block_bootstrap(sm.spend_decomp.resid_block, n_sims, H, rng)
        s_resid = 1.0 + 0.5 * (s_resid - 1.0)                            # halve dispersion
        spend_daily = np.maximum(exp_spend * s_resid, 0.0)

    # --- Revenue (own forecast) coupled to spend via elasticity ---
    exp_rev = _draw_expected(sm.rev_decomp, sm.seasonal_rev, fdates, n_sims, rng, cfg)
    r_resid = block_bootstrap(sm.rev_decomp.resid_block, n_sims, H, rng)
    resp = response_multiplier(spend_daily, baseline_spend, sm.beta)     # (n_sims, H)
    revenue_daily = exp_rev * r_resid * resp

    if common_log_shock is not None:
        revenue_daily = revenue_daily * np.exp(common_log_shock)

    revenue_daily = np.maximum(revenue_daily, 0.0)
    spend_daily = np.maximum(spend_daily, 0.0)
    return SegmentSim(sm.segment_key, sm.platform, sm.campaign_type,
                      fdates, revenue_daily, spend_daily)


def aggregate_to_periods(revenue_daily: np.ndarray, spend_daily: np.ndarray,
                         horizons: list[int]) -> dict:
    """Cumulative period sums + ROAS distribution for each horizon.

    Returns ``{horizon: {"revenue": (n_sims,), "spend": (n_sims,), "roas": (n_sims,)}}``.
    """
    out = {}
    H = revenue_daily.shape[1]
    for h in horizons:
        hh = min(h, H)
        rev = revenue_daily[:, :hh].sum(axis=1)
        spd = spend_daily[:, :hh].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            roas = np.where(spd > 0, rev / spd, np.nan)
        out[h] = {"revenue": rev, "spend": spd, "roas": roas}
    return out


def make_common_shock(n_sims: int, horizon: int, rng: np.random.Generator,
                      sigma: float = 0.08) -> np.ndarray:
    """A gently autocorrelated shared macro log-shock ((n_sims, horizon)).

    Modelled as a slow random walk (cumsum of small increments) so a "good month"
    persists rather than flickering day to day.
    """
    incr = rng.normal(0.0, sigma / np.sqrt(30.0), size=(n_sims, horizon))
    walk = np.cumsum(incr, axis=1)
    walk -= walk.mean(axis=1, keepdims=True)  # centre so it doesn't bias the median
    return walk
