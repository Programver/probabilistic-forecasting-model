"""
Marketing event calendar (fully offline).

E-commerce revenue is dominated by a handful of calendar events — Black Friday,
Cyber Monday, the December gift-buying ramp, and the post-holiday lull. A smooth
annual seasonal curve captures the broad shape, but these sharp, business-known
spikes deserve explicit treatment so that (a) a 30/60/90-day window straddling
late November is not under-forecast, and (b) the AI layer can *name* the driver
("uplift concentrated in the Black Friday / Cyber Monday window").

Everything here is computed from the calendar with the offline ``holidays``
package — no network access, satisfying the eval constraint.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from logging_utils import get_logger

log = get_logger("calendar")

try:
    import holidays as _holidays_pkg
    _HAS_HOLIDAYS = True
except Exception:  # pragma: no cover - holidays is a pinned dependency
    _HAS_HOLIDAYS = False


@dataclass(frozen=True)
class MarketingEvent:
    name: str
    # function(year) -> (start_date, end_date) inclusive
    window: Tuple[int, int]  # placeholder; concrete windows built per-year below
    default_uplift: float    # prior multiplicative uplift over baseline seasonality


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """n-th `weekday` (Mon=0) of a month, e.g. 4th Thursday of November."""
    d = dt.date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + dt.timedelta(days=offset + 7 * (n - 1))


def thanksgiving(year: int) -> dt.date:
    return _nth_weekday(year, 11, 3, 4)  # 4th Thursday of November


def black_friday(year: int) -> dt.date:
    return thanksgiving(year) + dt.timedelta(days=1)


def cyber_monday(year: int) -> dt.date:
    return thanksgiving(year) + dt.timedelta(days=4)


# Prior uplift multipliers (multiplicative, *on top of* the smooth annual curve).
# These are conservative priors; the fitted model re-estimates observed uplift
# where data supports it, so the prior mainly stabilises sparse windows.
EVENT_PRIOR_UPLIFT: Dict[str, float] = {
    "black_friday_week": 1.8,   # Wed before BF .. Sat
    "cyber_week": 1.6,          # Cyber Mon .. following Fri
    "december_ramp": 1.35,      # Dec 1 .. Dec 23
    "christmas_lull": 0.6,      # Dec 24 .. Dec 31
    "january_hangover": 0.85,   # Jan 1 .. Jan 15
}


def event_flags(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Return a per-date frame of marketing-event indicator columns + a prior
    multiplicative ``event_uplift`` (1.0 when no event applies).

    Columns: one boolean per event plus ``event_uplift`` and ``event_name``.
    """
    idx = pd.DatetimeIndex(dates)
    df = pd.DataFrame(index=idx)
    years = sorted(set(idx.year.tolist()))

    bf = {y: black_friday(y) for y in years}
    cm = {y: cyber_monday(y) for y in years}

    def in_window(ts, start, end):
        return (ts.date() >= start) and (ts.date() <= end)

    names: List[str] = []
    uplift = np.ones(len(idx), dtype=float)
    flags = {k: np.zeros(len(idx), dtype=bool) for k in EVENT_PRIOR_UPLIFT}

    for i, ts in enumerate(idx):
        y = ts.year
        name = ""
        # Black Friday week: Wed before BF .. Saturday after
        bf_start = bf[y] - dt.timedelta(days=2)
        bf_end = bf[y] + dt.timedelta(days=1)
        cm_start = cm[y]
        cm_end = cm[y] + dt.timedelta(days=4)
        if in_window(ts, bf_start, bf_end):
            flags["black_friday_week"][i] = True
            uplift[i] = max(uplift[i], EVENT_PRIOR_UPLIFT["black_friday_week"])
            name = "black_friday_week"
        elif in_window(ts, cm_start, cm_end):
            flags["cyber_week"][i] = True
            uplift[i] = max(uplift[i], EVENT_PRIOR_UPLIFT["cyber_week"])
            name = "cyber_week"
        elif ts.month == 12 and ts.day <= 23:
            flags["december_ramp"][i] = True
            uplift[i] = EVENT_PRIOR_UPLIFT["december_ramp"]
            name = "december_ramp"
        elif ts.month == 12 and ts.day >= 24:
            flags["christmas_lull"][i] = True
            uplift[i] = EVENT_PRIOR_UPLIFT["christmas_lull"]
            name = "christmas_lull"
        elif ts.month == 1 and ts.day <= 15:
            flags["january_hangover"][i] = True
            uplift[i] = EVENT_PRIOR_UPLIFT["january_hangover"]
            name = "january_hangover"
        names.append(name)

    for k, v in flags.items():
        df[k] = v
    df["event_uplift_prior"] = uplift
    df["event_name"] = names
    return df


def us_holiday_flags(dates: pd.DatetimeIndex) -> pd.Series:
    """Boolean series: is `date` a US public holiday (offline lookup)."""
    idx = pd.DatetimeIndex(dates)
    if not _HAS_HOLIDAYS:
        return pd.Series(False, index=idx)
    years = range(idx.year.min(), idx.year.max() + 2)
    us = _holidays_pkg.US(years=list(years))
    return pd.Series([d.date() in us for d in idx], index=idx)


def calendar_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """One-stop per-date calendar features used by seasonality + insights."""
    idx = pd.DatetimeIndex(dates)
    df = event_flags(idx)
    df["is_us_holiday"] = us_holiday_flags(idx).values
    df["month"] = idx.month
    df["dow"] = idx.dayofweek
    df["doy"] = idx.dayofyear
    df["iso_week"] = idx.isocalendar().week.astype(int).values
    df["year"] = idx.year
    return df


def describe_window(start: pd.Timestamp, end: pd.Timestamp) -> List[str]:
    """Human-readable list of notable events inside a forecast window (for insights)."""
    rng = pd.date_range(start, end, freq="D")
    ev = event_flags(rng)
    present = []
    label = {
        "black_friday_week": "Black Friday",
        "cyber_week": "Cyber Monday week",
        "december_ramp": "December gift-buying ramp",
        "christmas_lull": "post-Christmas lull",
        "january_hangover": "January slowdown",
    }
    for col, lab in label.items():
        if col in ev.columns and bool(ev[col].any()):
            present.append(lab)
    return present
