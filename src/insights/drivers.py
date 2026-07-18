"""
Deterministic driver attribution — the quantitative backbone of the AI layer.

Before any LLM is involved, we compute *from the model internals* a structured
explanation of the forecast: how the outlook compares to the recent trailing
period, and how much of the change is attributable to **seasonality** (the
marketing calendar), **trend** (year-over-year growth), and **spend & mix**. We
also surface channel movers, ROAS watch-list items, budget-efficiency
opportunities (marginal ROAS), forecast-window events, and data-quality caveats.

This structured :class:`DriverReport` is what the narrative/LLM layer turns into
prose. Because every number is derived (not generated), the "causal summaries"
are grounded and reproducible — the LLM explains real drivers rather than
inventing them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from calendar_features import describe_window
from logging_utils import get_logger
from seasonality import factor_for_dates
from taxonomy import split_segment_key

log = get_logger("drivers")


@dataclass
class DriverReport:
    origin: str
    currency: str
    horizons: List[int]
    horizon_summary: Dict[int, dict] = field(default_factory=dict)
    channel_summary: List[dict] = field(default_factory=list)
    movers: Dict[str, List[dict]] = field(default_factory=dict)     # growing / declining
    roas_watch: List[dict] = field(default_factory=list)
    opportunities: List[dict] = field(default_factory=list)
    data_quality: List[str] = field(default_factory=list)
    model_meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _trailing_actuals(panel: pd.DataFrame, origin: pd.Timestamp, days: int,
                      platform: Optional[str] = None) -> dict:
    lo = origin - pd.Timedelta(days=days - 1)
    sub = panel[(panel["date"] >= lo) & (panel["date"] <= origin)]
    if platform is not None:
        sub = sub[sub["platform"] == platform]
    rev = float(sub["revenue"].sum())
    spend = float(sub["spend"].sum())
    return {"revenue": rev, "spend": spend, "roas": (rev / spend) if spend > 0 else np.nan}


def _seasonal_index(segment_models, dates: pd.DatetimeIndex, weights: dict) -> float:
    """Revenue-weighted average aggregate seasonal factor over a set of dates."""
    total_w = sum(weights.values()) or 1.0
    idx = 0.0
    for key, sm in segment_models.items():
        w = weights.get(key, 0.0) / total_w
        if w <= 0:
            continue
        f = factor_for_dates(sm.seasonal_rev, dates)
        idx += w * float(np.mean(f))
    return idx


def _pct(a: float, b: float) -> float:
    return (a / b - 1.0) if (b and np.isfinite(b) and b > 0) else np.nan


def build_driver_report(result, prepared, model) -> DriverReport:
    origin = prepared.origin
    panel = prepared.panel
    seg_models = result.segment_models
    cfg = model.cfg

    # Recent-revenue weights per segment (for aggregate seasonal index).
    weights = {k: max(sm.rev_decomp.level, 0.0) for k, sm in seg_models.items()}

    report = DriverReport(
        origin=str(origin.date()), currency=cfg.currency, horizons=list(result.horizons),
        model_meta={
            "version": model.version,
            "trained_origin": model.trained_origin,
            "backtest_metrics": model.trained_meta.get("backtest_metrics", []),
        },
    )

    # ---- Per-horizon headline + attribution ----
    for h in result.horizons:
        g = result.get_group("aggregate")
        rev = g.dist[h]["revenue"]; roas = g.dist[h]["roas"]; spend = g.dist[h]["spend"]
        fc_rev = float(np.median(rev)); fc_roas = float(np.nanmedian(roas))
        fc_spend = float(np.median(spend))
        trailing = _trailing_actuals(panel, origin, h)

        fdates = pd.date_range(origin + pd.Timedelta(days=1), periods=h, freq="D")
        tdates = pd.date_range(origin - pd.Timedelta(days=h - 1), periods=h, freq="D")
        s_fore = _seasonal_index(seg_models, fdates, weights)
        s_trail = _seasonal_index(seg_models, tdates, weights)
        seasonal_effect = _pct(s_fore, s_trail)

        # Trend effect = avg growth multiplier over the forecast window.
        from decompose import growth_multiplier
        gm = []
        for key, sm in seg_models.items():
            w = weights.get(key, 0.0)
            if w <= 0:
                continue
            gmult = growth_multiplier(np.arange(1, h + 1), sm.rev_decomp.growth_annual, cfg)
            gm.append((w, float(np.mean(gmult))))
        trend_mult = (sum(w * v for w, v in gm) / sum(w for w, _ in gm)) if gm else 1.0
        trend_effect = trend_mult - 1.0

        total_change = _pct(fc_rev, trailing["revenue"])
        spend_mix_effect = np.nan
        if np.isfinite(total_change) and np.isfinite(seasonal_effect) and np.isfinite(trend_effect):
            # residual (spend/mix/nowcast) so the three add up to the total in log space
            spend_mix_effect = np.expm1(
                np.log1p(total_change) - np.log1p(seasonal_effect) - np.log1p(trend_effect))

        report.horizon_summary[h] = {
            "forecast_revenue_p50": fc_rev,
            "forecast_revenue_p10": float(np.quantile(rev, 0.10)),
            "forecast_revenue_p90": float(np.quantile(rev, 0.90)),
            "forecast_roas_p50": fc_roas,
            "forecast_spend_p50": fc_spend,
            "trailing_revenue": trailing["revenue"],
            "trailing_roas": trailing["roas"],
            "revenue_change_pct": total_change,
            "attribution": {
                "seasonality_pct": seasonal_effect,
                "trend_pct": trend_effect,
                "spend_and_mix_pct": spend_mix_effect,
            },
            "events": describe_window(fdates[0], fdates[-1]),
            "interval_width_pct": _pct(float(np.quantile(rev, 0.90)),
                                       float(np.quantile(rev, 0.10))),
        }

    # ---- Channel summary (use 30d as the reference window) ----
    ref_h = result.horizons[0]
    for g in result.groups:
        if g.level != "channel":
            continue
        rev = g.dist[ref_h]["revenue"]; roas = g.dist[ref_h]["roas"]
        trailing = _trailing_actuals(panel, origin, ref_h, platform=g.channel)
        report.channel_summary.append({
            "channel": g.channel,
            "forecast_revenue_p50": float(np.median(rev)),
            "forecast_roas_p50": float(np.nanmedian(roas)),
            "trailing_revenue": trailing["revenue"],
            "trailing_roas": trailing["roas"],
            "revenue_change_pct": _pct(float(np.median(rev)), trailing["revenue"]),
        })
    report.channel_summary.sort(key=lambda d: d["forecast_revenue_p50"], reverse=True)

    # ---- Segment movers (growth vs decline by trend) ----
    seg_moves = []
    for key, sm in seg_models.items():
        plat, ctype = split_segment_key(key)
        seg_moves.append({
            "segment": key, "channel": plat, "campaign_type": ctype,
            "growth_annual": float(sm.rev_decomp.growth_annual),
            "recent_daily_revenue": float(sm.rev_decomp.level),
            "beta": float(sm.beta),
        })
    seg_moves.sort(key=lambda d: d["growth_annual"], reverse=True)
    material = [m for m in seg_moves if m["recent_daily_revenue"] > 0]
    report.movers["growing"] = [m for m in material if m["growth_annual"] > 0.05][:4]
    report.movers["declining"] = [m for m in material if m["growth_annual"] < -0.05][-4:]

    # ---- ROAS watch (channels/segments with low or falling ROAS) ----
    for cs in report.channel_summary:
        if np.isfinite(cs["forecast_roas_p50"]) and cs["forecast_roas_p50"] < 2.0:
            report.roas_watch.append({
                "scope": f"{cs['channel']} (channel)",
                "forecast_roas": cs["forecast_roas_p50"],
                "trailing_roas": cs["trailing_roas"],
                "note": "forecast ROAS below 2.0x",
            })

    # ---- Opportunities: marginal-ROAS ranking ----
    try:
        from budget import suggest_reallocation
        realloc = suggest_reallocation(seg_models, model.elasticity_fits, origin, cfg, ref_h)
        realloc = realloc[np.isfinite(realloc["marginal_roas"])]
        for _, r in realloc.head(3).iterrows():
            report.opportunities.append({
                "type": "scale_up", "segment": r["segment"],
                "marginal_roas": float(r["marginal_roas"]), "avg_roas": float(r["avg_roas"]),
                "daily_spend": float(r["daily_spend"]),
                "rationale": "highest incremental return on the next dollar",
            })
        for _, r in realloc.tail(2).iterrows():
            if np.isfinite(r["marginal_roas"]):
                report.opportunities.append({
                    "type": "review", "segment": r["segment"],
                    "marginal_roas": float(r["marginal_roas"]), "avg_roas": float(r["avg_roas"]),
                    "daily_spend": float(r["daily_spend"]),
                    "rationale": "lowest incremental return — candidate to cap or optimise",
                })
    except Exception as exc:  # never let insights break
        log.warning("reallocation opportunities unavailable: %s", exc)

    # ---- Data quality ----
    report.data_quality = list(prepared.notes)
    bt = model.trained_meta.get("backtest_metrics", [])
    if bt:
        cov = ", ".join(f"{m['horizon']}d≈{m['coverage_P10_P90_%']:.0f}%" for m in bt)
        report.data_quality.append(f"Backtest interval coverage (P10–P90): {cov}.")
    return report
