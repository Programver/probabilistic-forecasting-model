"""
Seasonal profile estimation with partial pooling.

The dominant signal in this data is a very large annual cycle (a ~10× swing from
the summer trough to the December peak) observed over only ~2 holiday cycles.
Estimating a 52-week shape from 2 observations per week is noisy, so we:

* estimate each segment's own multiplicative **ISO-week** profile via
  ratio-to-annual-moving-average (isolates within-year shape from level drift),
* **circularly smooth** it (adjacent weeks are correlated), and
* **shrink** it toward a pooled global profile with a weight that grows with the
  amount of history the segment actually has (James–Stein-style partial pooling).

A separate day-of-week profile captures the within-week pattern (strong on Meta,
mild on Google). Profiles are multiplicative and normalised to geometric mean 1
so they never bias the level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from logging_utils import get_logger

log = get_logger("seasonality")

N_WEEKS = 53  # ISO weeks 1..53 -> index 0..52


@dataclass
class SeasonalProfile:
    week_factor: np.ndarray   # (53,) multiplicative, geom-mean 1, index = iso_week - 1
    dow_factor: np.ndarray    # (7,)  multiplicative, geom-mean 1, index = dayofweek (Mon=0)
    n_days: int = 0
    coverage_weeks: int = 0

    def copy(self) -> "SeasonalProfile":
        return SeasonalProfile(self.week_factor.copy(), self.dow_factor.copy(),
                               self.n_days, self.coverage_weeks)


def _geom_normalize(x: np.ndarray) -> np.ndarray:
    """Scale a positive vector to geometric mean 1; robust to zeros/NaNs."""
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x) & (x > 0), x, np.nan)
    if np.all(np.isnan(x)):
        return np.ones_like(x)
    gm = np.exp(np.nanmean(np.log(x)))
    if not np.isfinite(gm) or gm <= 0:
        return np.ones_like(x)
    out = x / gm
    return np.where(np.isnan(out), 1.0, out)


def _circular_smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Circular moving average (weeks wrap around the year)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if window <= 1 or n == 0:
        return x
    half = window // 2
    padded = np.concatenate([x[-half:], x, x[:half]])
    kernel = np.ones(window) / window
    sm = np.convolve(padded, kernel, mode="valid")
    return sm[:n]


def flat_profile() -> SeasonalProfile:
    return SeasonalProfile(np.ones(N_WEEKS), np.ones(7), 0, 0)


def fit_seasonal_profile(daily: pd.DataFrame, value_col: str = "revenue") -> SeasonalProfile:
    """Estimate a segment's multiplicative week + dow profile.

    ``daily`` must be a dense, date-indexed (or ``date`` column) daily frame.
    """
    if daily is None or len(daily) == 0:
        return flat_profile()
    d = daily.copy()
    if "date" in d.columns:
        d = d.set_index("date")
    d = d.sort_index()
    r = d[value_col].astype(float).clip(lower=0.0)
    n_days = int(len(r))
    if n_days < 14 or r.sum() <= 0:
        return SeasonalProfile(np.ones(N_WEEKS), np.ones(7), n_days, 0)

    # Ratio-to-annual-moving-average: isolates within-year shape from level drift.
    ma = r.rolling(365, center=True, min_periods=90).mean()
    global_mean = r[r > 0].mean()
    ma = ma.where(ma > 0, other=global_mean).replace(0, global_mean)
    ratio = (r / ma).replace([np.inf, -np.inf], np.nan)

    iso_week = r.index.isocalendar().week.astype(int).clip(1, N_WEEKS).values
    dow = r.index.dayofweek.values

    # --- Weekly (annual) profile ---
    wk = np.ones(N_WEEKS)
    tmp = pd.DataFrame({"w": iso_week, "ratio": ratio.values})
    grp = tmp[tmp["ratio"].notna() & (tmp["ratio"] > 0)].groupby("w")["ratio"].mean()
    for w, v in grp.items():
        wk[int(w) - 1] = v
    coverage_weeks = int(grp.shape[0])
    wk = _circular_smooth(wk, window=5)
    wk = _geom_normalize(wk)

    # --- Day-of-week profile (after removing annual shape) ---
    week_component = wk[iso_week - 1]
    deseason_annual = (r.values / (ma.values * np.where(week_component > 0, week_component, 1.0)))
    dw = np.ones(7)
    tmp2 = pd.DataFrame({"dow": dow, "v": deseason_annual})
    grp2 = tmp2[np.isfinite(tmp2["v"]) & (tmp2["v"] > 0)].groupby("dow")["v"].mean()
    for k, v in grp2.items():
        dw[int(k)] = v
    dw = _geom_normalize(dw)

    return SeasonalProfile(wk, dw, n_days, coverage_weeks)


def factor_for_dates(profile: SeasonalProfile, dates: pd.DatetimeIndex,
                     include_dow: bool = True) -> np.ndarray:
    """Multiplicative seasonal factor for each date."""
    idx = pd.DatetimeIndex(dates)
    iso_week = idx.isocalendar().week.astype(int).clip(1, N_WEEKS).values
    f = profile.week_factor[iso_week - 1]
    if include_dow:
        f = f * profile.dow_factor[idx.dayofweek.values]
    return f


def pooling_weight(n_days: int, k: int = 365) -> float:
    """Weight on the segment's *own* seasonal estimate vs the global prior.

    ``w = n / (n + k)`` — a segment needs ~one full extra year (k) of history to
    put half its trust in its own estimate. Long, stable segments trust
    themselves; short/sparse ones lean on the pooled global shape.
    """
    n = max(int(n_days), 0)
    return n / (n + k)


def blend(own: SeasonalProfile, prior: SeasonalProfile, weight: float) -> SeasonalProfile:
    """Geometric (log-space) blend of two multiplicative profiles."""
    w = float(np.clip(weight, 0.0, 1.0))

    def _blend(a, b):
        a = np.where((a > 0) & np.isfinite(a), a, 1.0)
        b = np.where((b > 0) & np.isfinite(b), b, 1.0)
        out = np.exp(w * np.log(a) + (1 - w) * np.log(b))
        return _geom_normalize(out)

    return SeasonalProfile(
        week_factor=_blend(own.week_factor, prior.week_factor),
        dow_factor=_blend(own.dow_factor, prior.dow_factor),
        n_days=own.n_days,
        coverage_weeks=own.coverage_weeks,
    )


def pool(profiles: list[SeasonalProfile], weights: list[float] | None = None) -> SeasonalProfile:
    """Build a global prior by pooling many segment profiles (revenue-weighted)."""
    profiles = [p for p in profiles if p is not None and p.n_days > 0]
    if not profiles:
        return flat_profile()
    if weights is None:
        weights = [max(p.n_days, 1) for p in profiles]
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    logw = np.zeros(N_WEEKS)
    logd = np.zeros(7)
    for p, wt in zip(profiles, weights):
        logw += wt * np.log(np.where(p.week_factor > 0, p.week_factor, 1.0))
        logd += wt * np.log(np.where(p.dow_factor > 0, p.dow_factor, 1.0))
    return SeasonalProfile(
        week_factor=_geom_normalize(np.exp(logw)),
        dow_factor=_geom_normalize(np.exp(logd)),
        n_days=int(sum(p.n_days for p in profiles)),
        coverage_weeks=max(p.coverage_weeks for p in profiles),
    )
