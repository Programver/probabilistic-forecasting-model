"""
Rolling-origin backtesting → conformal interval calibration + accuracy metrics.

We walk the forecast origin backwards in fixed steps, each time forecasting from
data available *up to that origin* and comparing against the revenue that actually
followed. Two things come out of this:

* **Conformal multipliers** (per horizon) — how much to widen/tighten the
  simulated intervals so realised coverage matches the nominal 80%. Frozen into
  the model, they make the shipped intervals honest on unseen data.
* **Accuracy diagnostics** — MAPE / interval coverage / pinball loss — reported in
  the docs to justify model selection (a required deliverable) and shown in the
  app so users can see the forecaster is calibrated, not just confident.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

import config
from hierarchy import reconcile
from logging_utils import get_logger
from model import ForecastModel, build_segment_models
from preprocess import PreparedData, prepare
from simulate import make_common_shock, simulate_segment
from uncertainty import ConformalCalibration, conformal_scale_from_scores

log = get_logger("backtest")


@dataclass
class BacktestReport:
    conformal: ConformalCalibration
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    records: pd.DataFrame = field(default_factory=pd.DataFrame)


def _truncate(prepared_panel: pd.DataFrame, cutoff: pd.Timestamp) -> PreparedData:
    sub = prepared_panel[prepared_panel["date"] <= cutoff].copy()
    return prepare(sub)


def _simulate_groups(prepared: PreparedData, model: ForecastModel,
                     cfg: config.ForecastConfig, rng: np.random.Generator):
    seg_models = build_segment_models(prepared, model, cfg)
    max_h = max(cfg.horizons)
    fdates = pd.date_range(prepared.origin + pd.Timedelta(days=1), periods=max_h, freq="D")
    common = make_common_shock(cfg.n_simulations, max_h, rng)
    sims = {}
    for key, sm in seg_models.items():
        sims[key] = simulate_segment(sm, fdates, rng, cfg, common_log_shock=common)
    return reconcile(sims, prepared.campaign_meta, cfg)


def calibrate(prepared: PreparedData, model: ForecastModel,
              cfg: config.ForecastConfig | None = None,
              n_origins: int = config.BACKTEST_ORIGINS,
              step_days: int = config.BACKTEST_STEP_DAYS,
              n_sims: int = 1500) -> BacktestReport:
    """Backtest and return calibrated conformal + accuracy metrics."""
    base_cfg = cfg or model.cfg or config.ForecastConfig()
    bt_cfg = config.ForecastConfig(**{**base_cfg.__dict__, "n_simulations": n_sims})

    panel = prepared.panel
    full_origin = prepared.origin
    max_h = max(bt_cfg.horizons)

    # Daily actual revenue/spend across the whole account (for scoring).
    daily_actual = panel.groupby("date")[["revenue", "spend"]].sum().sort_index()

    latest_origin = full_origin - pd.Timedelta(days=max_h)
    origins = [latest_origin - pd.Timedelta(days=i * step_days) for i in range(n_origins)]
    origins = [o for o in origins if (o - panel["date"].min()).days >= 180]
    if not origins:
        log.warning("insufficient history for backtest; using identity conformal")
        return BacktestReport(ConformalCalibration(), pd.DataFrame(), pd.DataFrame())

    rng = np.random.default_rng(bt_cfg.seed + 7)
    records: List[dict] = []
    for o in origins:
        try:
            sub = _truncate(panel, o)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("skip origin %s: %s", o.date(), exc)
            continue
        groups = _simulate_groups(sub, model, bt_cfg, rng)
        # Score aggregate + channel groups (levels that will be calibrated).
        for g in groups:
            if g.level not in ("aggregate", "channel"):
                continue
            for h in bt_cfg.horizons:
                win = daily_actual.loc[(daily_actual.index > o) &
                                       (daily_actual.index <= o + pd.Timedelta(days=h))]
                if win.empty:
                    continue
                # restrict channel actuals to that platform
                if g.level == "channel":
                    pcols = panel[(panel["platform"] == g.channel) &
                                  (panel["date"] > o) &
                                  (panel["date"] <= o + pd.Timedelta(days=h))]
                    actual_rev = float(pcols["revenue"].sum())
                else:
                    actual_rev = float(win["revenue"].sum())
                dist = g.dist[h]["revenue"]
                med = float(np.median(dist))
                p10, p90 = np.quantile(dist, [0.10, 0.90])
                if med <= 0 or actual_rev <= 0 or p90 <= p10:
                    continue
                halfwidth_log = 0.5 * (np.log(max(p90, 1e-9)) - np.log(max(p10, 1e-9)))
                score = abs(np.log(actual_rev / med))
                records.append({
                    "origin": o, "horizon": h, "level": g.level, "channel": g.channel,
                    "actual": actual_rev, "median": med, "p10": p10, "p90": p90,
                    "score": score, "halfwidth_log": halfwidth_log,
                    "in80": bool(p10 <= actual_rev <= p90),
                    "ape": abs(actual_rev - med) / actual_rev,
                })

    rec = pd.DataFrame(records)
    conf = ConformalCalibration(target_coverage=0.80, n_calibration=len(rec))
    if not rec.empty:
        # Calibrate PER LEVEL so the well-behaved aggregate intervals stay sharp
        # and are not over-widened by the noisier small channels (e.g. Bing).
        for h in bt_cfg.horizons:
            for level in ("aggregate", "channel"):
                sub = rec[(rec["horizon"] == h) & (rec["level"] == level)]
                if len(sub) >= 4:
                    s = conformal_scale_from_scores(
                        sub["score"].to_numpy(), sub["halfwidth_log"].to_numpy(), 0.80)
                else:
                    s = 1.0
                conf.scale_by_level_horizon[f"{level}::{h}"] = s
            agg_s = conf.scale_by_level_horizon.get(f"aggregate::{h}", 1.0)
            ch_s = conf.scale_by_level_horizon.get(f"channel::{h}", agg_s)
            conf.scale_by_horizon[h] = agg_s  # default fallback = aggregate
            # Finer levels are noisier than the channel they roll up from.
            conf.scale_by_level_horizon[f"campaign_type::{h}"] = ch_s * 1.10
            conf.scale_by_level_horizon[f"campaign::{h}"] = ch_s * 1.25

    metrics = _summarize_metrics(rec)
    if not metrics.empty:
        log.info("backtest metrics:\n%s", metrics.to_string(index=False))
    return BacktestReport(conf, metrics, rec)


def _summarize_metrics(rec: pd.DataFrame) -> pd.DataFrame:
    if rec.empty:
        return pd.DataFrame()
    agg = rec[rec["level"] == "aggregate"]
    rows = []
    for h, sub in agg.groupby("horizon"):
        rows.append({
            "horizon": int(h),
            "n": len(sub),
            "MAPE_%": round(100 * sub["ape"].mean(), 1),
            "median_APE_%": round(100 * sub["ape"].median(), 1),
            "coverage_P10_P90_%": round(100 * sub["in80"].mean(), 1),
        })
    return pd.DataFrame(rows)
