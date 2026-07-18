"""
Hierarchical reconciliation and disaggregation.

The forecast hierarchy has four levels:

    aggregate  →  channel (platform)  →  campaign_type (channel × type)  →  campaign

We achieve coherence the clean way: every segment is simulated once, and we then
**sum the same simulated world (sim index) across segments** to build every
higher aggregate. Because a period sum is linear, summing segment period-totals
per world equals the period-total of the summed series — so channels sum to the
aggregate *exactly*, world by world, with the correct cross-segment correlation
baked in (no post-hoc MinT projection needed).

Campaigns are disaggregated **top-down** by recent revenue/spend share. Recent
share is the robust choice here because most individual campaigns are short-lived
and sparse (many have <180 days of history and near-zero revenue at the median),
so a bottom-up fit per campaign would be pure noise. Proportional shares keep the
campaign level exactly coherent with its segment; the extra idiosyncratic
volatility of a single campaign is restored through a per-level conformal widening
rather than ad-hoc noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

import config
from logging_utils import get_logger
from simulate import SegmentSim, aggregate_to_periods
from taxonomy import split_segment_key

log = get_logger("hierarchy")


@dataclass
class GroupForecast:
    level: str            # aggregate | channel | campaign_type | campaign
    channel: str          # blended | google | meta | bing
    campaign_type: str    # all | <type>
    campaign_id: str      # all | <id>
    campaign_name: str    # all | <name>
    dist: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    # per horizon -> {"revenue": (n_sims,), "spend": (n_sims,), "roas": (n_sims,)}


def _roas(rev: np.ndarray, spend: np.ndarray) -> np.ndarray:
    """Revenue / spend, with the two degenerate cases handled explicitly.

    A **dormant** node — no spend *and* no revenue, e.g. a campaign whose recent
    share is zero — has a conventional ROAS of **0**, not an undefined one. That
    is how the ad platforms themselves report it, and it matters here: ~45% of
    campaigns in the sample are dormant, so treating 0/0 as NaN put 189 null rows
    (13% of the file) into the scored output.

    Zero spend against *positive* revenue (lagged/view-through conversions on a
    paused campaign) is genuinely infinite and deliberately stays NaN — those are
    dropped by :func:`io_utils.summarize` rather than invented.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(spend > 0, rev / spend, np.nan)
    return np.where((spend <= 0) & (rev <= 0), 0.0, out)


def _combine(dists: List[Dict[int, Dict[str, np.ndarray]]],
             horizons: List[int]) -> Dict[int, Dict[str, np.ndarray]]:
    """Sum a list of per-horizon {revenue,spend} distributions element-wise."""
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for h in horizons:
        rev = np.sum([d[h]["revenue"] for d in dists], axis=0)
        spd = np.sum([d[h]["spend"] for d in dists], axis=0)
        out[h] = {"revenue": rev, "spend": spd, "roas": _roas(rev, spd)}
    return out


def reconcile(segment_sims: Dict[str, SegmentSim], campaign_meta: pd.DataFrame,
              cfg: config.ForecastConfig) -> List[GroupForecast]:
    """Build all four hierarchy levels from per-segment simulations."""
    horizons = cfg.horizons

    # 1. Segment period distributions (channel × type) — our modelling unit.
    seg_period: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    for key, sim in segment_sims.items():
        seg_period[key] = aggregate_to_periods(sim.revenue_daily, sim.spend_daily, horizons)

    groups: List[GroupForecast] = []

    # 2. Aggregate (blended total).
    agg = _combine(list(seg_period.values()), horizons)
    groups.append(GroupForecast("aggregate", "blended", "all", "all", "all", agg))

    # 3. Channel level (per platform).
    by_platform: Dict[str, List[str]] = {}
    for key in seg_period:
        plat, _ = split_segment_key(key)
        by_platform.setdefault(plat, []).append(key)
    for plat, keys in sorted(by_platform.items()):
        d = _combine([seg_period[k] for k in keys], horizons)
        groups.append(GroupForecast("channel", plat, "all", "all", "all", d))

    # 4a. Campaign-type per channel (the segment itself).
    for key, sim in segment_sims.items():
        plat, ctype = split_segment_key(key)
        groups.append(GroupForecast("campaign_type", plat, ctype, "all", "all",
                                    seg_period[key]))

    # 4b. Campaign-type blended across channels.
    by_type: Dict[str, List[str]] = {}
    for key in seg_period:
        _, ctype = split_segment_key(key)
        by_type.setdefault(ctype, []).append(key)
    for ctype, keys in sorted(by_type.items()):
        if len({split_segment_key(k)[0] for k in keys}) <= 1:
            continue  # only one channel has this type -> 4a already covers it
        d = _combine([seg_period[k] for k in keys], horizons)
        groups.append(GroupForecast("campaign_type", "blended", ctype, "all", "all", d))

    # 5. Campaign level (top-down by recent share, exactly coherent).
    if campaign_meta is not None and not campaign_meta.empty:
        groups.extend(_disaggregate_campaigns(seg_period, campaign_meta, horizons))

    log.info("reconciled %d forecast groups across %d segments",
             len(groups), len(segment_sims))
    return groups


def _disaggregate_campaigns(seg_period, campaign_meta, horizons) -> List[GroupForecast]:
    out: List[GroupForecast] = []
    meta = campaign_meta.sort_values(["segment_key", "recent_revenue"], ascending=[True, False])
    capped = meta.groupby("segment_key", sort=False).head(config.MAX_CAMPAIGNS_PER_SEGMENT)
    if len(capped) < len(meta):
        log.warning("capping campaign disaggregation to top %d/segment by recent revenue "
                    "(%d of %d campaigns dropped from per-campaign detail; still fully "
                    "represented at the campaign_type level)",
                    config.MAX_CAMPAIGNS_PER_SEGMENT, len(meta) - len(capped), len(meta))
    for _, row in capped.iterrows():
        key = row["segment_key"]
        if key not in seg_period:
            continue
        w_rev = float(row.get("rev_share", 0.0) or 0.0)
        w_spd = float(row.get("spend_share", np.nan))
        if not np.isfinite(w_spd):
            w_spd = w_rev
        dist = {}
        for h in horizons:
            rev = seg_period[key][h]["revenue"] * w_rev
            spd = seg_period[key][h]["spend"] * w_spd
            dist[h] = {"revenue": rev, "spend": spd, "roas": _roas(rev, spd)}
        out.append(GroupForecast(
            "campaign", row["platform"], row["campaign_type"],
            str(row["campaign_id"]), str(row["campaign_name"]), dist))
    return out
