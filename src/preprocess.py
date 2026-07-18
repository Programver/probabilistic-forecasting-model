"""
Preprocessing: canonical panel -> modelling-ready structures.

Responsibilities
----------------
1. Attach the campaign taxonomy (channel × campaign_type × brand_intent).
2. Trim the *ragged edge* — trailing days where the data extract is incomplete —
   using reporting-completeness (number of active series), which is robust to
   seasonality (a genuine low-season day still has the usual number of series
   reporting; an incomplete extract does not).
3. Build a **dense** segment-daily panel (channel × campaign_type), gap-filled
   with zeros from each segment's launch to the forecast origin. This is the
   level at which we fit robust seasonal/response models.
4. Build a campaign-daily panel and recent revenue **shares** for top-down
   disaggregation to individual campaigns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

import config
from logging_utils import get_logger
from taxonomy import add_taxonomy, segment_key

log = get_logger("preprocess")


@dataclass
class PreparedData:
    panel: pd.DataFrame                 # canonical + taxonomy (trimmed)
    origin: pd.Timestamp                # last reliable observation date
    segment_daily: pd.DataFrame         # dense per (segment_key, date)
    campaign_daily: pd.DataFrame        # per (campaign_uid, date)
    campaign_meta: pd.DataFrame         # one row per campaign_uid
    segments: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def segment_series(self, key: str) -> pd.DataFrame:
        """Dense daily frame (date-indexed) for one segment."""
        s = self.segment_daily[self.segment_daily["segment_key"] == key]
        return s.set_index("date").sort_index()


def _campaign_uid(row) -> str:
    return f"{row['platform']}::{row['campaign_id']}"


def trim_ragged_edge(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp, List[str]]:
    """Drop trailing dates whose reporting looks incomplete.

    Heuristic: for each date, count distinct active series (campaigns with any
    spend or revenue). Walking backwards from the last date, drop a trailing day
    only if its active-series count is < 50% of the trailing-28d median *and* the
    day is within the last 3 days (extracts are usually truncated by at most a
    day or two). Conservative by design — we never trim genuine history.
    """
    notes: List[str] = []
    if panel.empty:
        return panel, pd.NaT, notes

    active = panel[(panel["spend"] > 0) | (panel["revenue"] > 0)]
    per_day = active.groupby("date").apply(
        lambda d: d["platform"].str.cat(d["campaign_id"]).nunique(), include_groups=False
    ).sort_index()
    if per_day.empty:
        origin = panel["date"].max()
        return panel, origin, notes

    ref = per_day.rolling(28, min_periods=7).median()
    dates_desc = list(per_day.index[::-1])
    drop_dates = []
    for i, d in enumerate(dates_desc):
        if i >= 3:  # only inspect the final 3 days
            break
        r = ref.get(d, np.nan)
        if pd.notna(r) and r > 0 and per_day[d] < 0.5 * r:
            drop_dates.append(d)
        else:
            break  # once a complete day is seen, stop trimming

    if drop_dates:
        cutoff = min(drop_dates)
        panel = panel[panel["date"] < cutoff].copy()
        notes.append(
            f"Trimmed {len(drop_dates)} trailing day(s) with incomplete reporting "
            f"(from {cutoff.date()}); forecasting from last complete day."
        )
        log.info(notes[-1])

    origin = panel["date"].max()
    return panel, origin, notes


def _dense_daily(df: pd.DataFrame, key_cols: List[str], origin: pd.Timestamp) -> pd.DataFrame:
    """Expand a grouped daily series to a dense daily grid per key, launch→origin.

    Leading inactivity (before first spend/revenue) is excluded; internal gaps are
    zero-filled (a day with no spend genuinely produced ~no revenue).
    """
    frames = []
    metric_cols = ["spend", "revenue", "clicks", "impressions", "conversions"]
    for keys, g in df.groupby(key_cols, sort=False):
        g = g.sort_values("date")
        active = g[(g["spend"] > 0) | (g["revenue"] > 0)]
        if active.empty:
            start = g["date"].min()
        else:
            start = active["date"].min()
        if pd.isna(start) or start > origin:
            continue
        idx = pd.date_range(start, origin, freq="D")
        gg = g.set_index("date")[metric_cols].groupby(level=0).sum()
        gg = gg.reindex(idx, fill_value=0.0)
        gg.index.name = "date"
        gg = gg.reset_index()
        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, val in zip(key_cols, keys):
            gg[col] = val
        frames.append(gg)
    if not frames:
        return pd.DataFrame(columns=key_cols + ["date"] + metric_cols)
    return pd.concat(frames, ignore_index=True)


def prepare(panel: pd.DataFrame) -> PreparedData:
    """Full preprocessing entry point."""
    panel = add_taxonomy(panel)
    panel["segment_key"] = [segment_key(p, t) for p, t in
                            zip(panel["platform"], panel["campaign_type"])]
    panel["campaign_uid"] = panel.apply(_campaign_uid, axis=1)

    panel, origin, notes = trim_ragged_edge(panel)
    if pd.isna(origin):
        raise ValueError("No usable dated rows after preprocessing.")

    # --- Segment-daily (dense) ---
    seg_src = panel.groupby(["segment_key", "platform", "campaign_type", "date"],
                            as_index=False)[
        ["spend", "revenue", "clicks", "impressions", "conversions"]
    ].sum()
    segment_daily = _dense_daily(seg_src, ["segment_key", "platform", "campaign_type"], origin)

    # --- Campaign-daily (dense) ---
    camp_src = panel.groupby(
        ["campaign_uid", "segment_key", "platform", "campaign_type",
         "campaign_id", "campaign_name", "date"], as_index=False
    )[["spend", "revenue", "clicks", "impressions", "conversions"]].sum()
    campaign_daily = _dense_daily(
        camp_src,
        ["campaign_uid", "segment_key", "platform", "campaign_type",
         "campaign_id", "campaign_name"],
        origin,
    )

    # --- Campaign metadata + recent shares (for disaggregation) ---
    campaign_meta = _campaign_metadata(campaign_daily, origin)

    segments = sorted(segment_daily["segment_key"].unique().tolist())
    log.info("prepared %d segments, %d campaigns, origin=%s",
             len(segments), campaign_meta.shape[0], origin.date())

    return PreparedData(
        panel=panel,
        origin=origin,
        segment_daily=segment_daily,
        campaign_daily=campaign_daily,
        campaign_meta=campaign_meta,
        segments=segments,
        notes=notes,
    )


def _campaign_metadata(campaign_daily: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    if campaign_daily.empty:
        return pd.DataFrame()
    win_start = origin - pd.Timedelta(days=config.RECENT_SHARE_WINDOW - 1)
    recent = campaign_daily[campaign_daily["date"] >= win_start]

    agg = campaign_daily.groupby(
        ["campaign_uid", "segment_key", "platform", "campaign_type", "campaign_id", "campaign_name"],
        as_index=False
    ).agg(total_spend=("spend", "sum"), total_revenue=("revenue", "sum"),
          n_days=("date", "nunique"), first_day=("date", "min"), last_day=("date", "max"))

    rec = recent.groupby("campaign_uid", as_index=False).agg(
        recent_spend=("spend", "sum"), recent_revenue=("revenue", "sum"))
    meta = agg.merge(rec, on="campaign_uid", how="left").fillna(
        {"recent_spend": 0.0, "recent_revenue": 0.0})

    # Recent revenue & spend share within the segment (top-down weights).
    for col, share in [("recent_revenue", "rev_share"), ("recent_spend", "spend_share")]:
        seg_tot = meta.groupby("segment_key")[col].transform("sum")
        meta[share] = np.where(seg_tot > 0, meta[col] / seg_tot, np.nan)
    # Fallback: campaigns with no recent activity share equally by all-time revenue.
    meta["rev_share"] = meta["rev_share"].fillna(
        meta.groupby("segment_key")["total_revenue"].transform(
            lambda s: (s / s.sum()) if s.sum() > 0 else 1.0 / max(len(s), 1)
        )
    )
    meta["days_since_active"] = (origin - meta["last_day"]).dt.days
    return meta
