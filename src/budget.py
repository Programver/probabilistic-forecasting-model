"""
Budget simulation ("what-if" media planning).

Agencies don't just want a passive forecast — they want to answer *"if I put
$X into Google and $Y into Meta next month, what revenue and ROAS should I
expect, with what confidence?"* This module turns a planned budget into a
per-segment daily spend path and feeds it through the same probabilistic engine,
so revenue responds through the frozen diminishing-returns elasticity.

A budget can be expressed as:
  * a **global multiplier** vs the baseline forecast (e.g. 1.2 = +20% everywhere),
  * **per-channel** window totals (the usual agency input),
  * **per-segment** window totals (power users), or
  * a single **total** budget split across segments by recent spend share.

The scenario is applied as a *sustained spend-rate change*: the per-segment daily
scale that hits the planned total over the chosen window is carried across the
whole forecast, so 30/60/90-day views stay internally consistent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

import config
from decompose import expected_path
from logging_utils import get_logger
from model import SegmentModel
from response import ElasticityFit, average_roas_at, marginal_roas
from taxonomy import split_segment_key

log = get_logger("budget")


@dataclass
class BudgetScenario:
    horizon: int = 30
    multiplier: Optional[float] = None
    channel_budgets: Optional[Dict[str, float]] = None   # platform -> total window spend
    segment_budgets: Optional[Dict[str, float]] = None   # segment_key -> total window spend
    total_budget: Optional[float] = None
    label: str = "scenario"


def baseline_daily_spend(segment_models: Dict[str, SegmentModel], origin: pd.Timestamp,
                         max_h: int, cfg: config.ForecastConfig) -> Dict[str, np.ndarray]:
    """Deterministic baseline expected daily spend per segment over max_h days."""
    future = pd.date_range(origin + pd.Timedelta(days=1), periods=max_h, freq="D")
    out = {}
    for key, sm in segment_models.items():
        out[key] = np.maximum(expected_path(sm.spend_decomp, future, sm.seasonal_spend, cfg), 1e-9)
    return out


def build_spend_overrides(scenario: BudgetScenario, segment_models: Dict[str, SegmentModel],
                          origin: pd.Timestamp, cfg: config.ForecastConfig
                          ) -> Dict[str, np.ndarray]:
    """Translate a BudgetScenario into per-segment daily spend arrays (length max_h)."""
    max_h = max(cfg.horizons)
    h = min(scenario.horizon, max_h)
    base = baseline_daily_spend(segment_models, origin, max_h, cfg)
    base_window_total = {k: float(v[:h].sum()) for k, v in base.items()}

    # Determine a per-segment multiplicative scale.
    scale: Dict[str, float] = {k: 1.0 for k in base}

    if scenario.multiplier is not None:
        scale = {k: float(scenario.multiplier) for k in base}

    elif scenario.segment_budgets:
        for k in base:
            if k in scenario.segment_budgets and base_window_total[k] > 0:
                scale[k] = scenario.segment_budgets[k] / base_window_total[k]

    elif scenario.channel_budgets:
        for plat, target in scenario.channel_budgets.items():
            seg_keys = [k for k in base if split_segment_key(k)[0] == plat]
            plat_base = sum(base_window_total[k] for k in seg_keys)
            if plat_base > 0:
                s = target / plat_base
                for k in seg_keys:
                    scale[k] = s

    elif scenario.total_budget is not None:
        tot_base = sum(base_window_total.values())
        if tot_base > 0:
            s = scenario.total_budget / tot_base
            scale = {k: s for k in base}

    # Guard against absurd scales (keep within 0..10x baseline).
    overrides = {}
    for k in base:
        s = float(np.clip(scale[k], 0.0, 10.0))
        overrides[k] = base[k] * s
    return overrides


def scenario_window_spend(overrides: Dict[str, np.ndarray], horizon: int) -> float:
    return float(sum(v[:horizon].sum() for v in overrides.values()))


def paired_forecasts(prepared, model, scenario: BudgetScenario,
                     cfg: Optional[config.ForecastConfig] = None):
    """Run baseline and scenario forecasts that are *directly comparable*.

    For a clean causal read of a budget change we hold the random draws fixed and
    vary only spend. Both runs therefore route spend through explicit overrides
    (baseline = each segment's own expected spend; scenario = the same with the
    requested budget applied) and share the model seed. Segments the user did not
    touch receive identical shocks and spend in both runs, so their contribution
    cancels in the delta — the difference reflects only the budget change and its
    diminishing-returns response.

    Returns ``(baseline_result, scenario_result)``.
    """
    from forecast import run_forecast  # local import avoids a module cycle
    from model import build_segment_models
    cfg = cfg or model.cfg or config.ForecastConfig()
    seg = build_segment_models(prepared, model, cfg)

    base_scn = BudgetScenario(horizon=scenario.horizon, multiplier=1.0, label="baseline")
    base_ov = build_spend_overrides(base_scn, seg, prepared.origin, cfg)
    scen_ov = build_spend_overrides(scenario, seg, prepared.origin, cfg)

    base = run_forecast(prepared, model, cfg, spend_overrides=base_ov, scenario_label="baseline")
    scen = run_forecast(prepared, model, cfg, spend_overrides=scen_ov,
                        scenario_label=scenario.label)
    return base, scen


# ---------------------------------------------------------------------------
# Response curves (diminishing returns) for the simulator UI + insights
# ---------------------------------------------------------------------------
def response_curve(fit: ElasticityFit, spend_grid: np.ndarray) -> pd.DataFrame:
    """Predicted daily revenue / average ROAS / marginal ROAS over a spend grid."""
    rows = []
    for s in spend_grid:
        rows.append({
            "spend": float(s),
            "revenue": float(fit.scale * s ** fit.beta) if np.isfinite(fit.scale) else np.nan,
            "avg_roas": average_roas_at(s, fit),
            "marginal_roas": marginal_roas(s, fit),
        })
    return pd.DataFrame(rows)


def suggest_reallocation(segment_models: Dict[str, SegmentModel],
                         elasticity_fits: Dict[str, ElasticityFit],
                         origin: pd.Timestamp, cfg: config.ForecastConfig,
                         horizon: int = 30) -> pd.DataFrame:
    """Rank segments by marginal ROAS at current spend — where the next dollar
    works hardest (grow) vs where it is saturated (trim). Pure decision support;
    the elasticity curves are the transparent basis for the recommendation.
    """
    base = baseline_daily_spend(segment_models, origin, max(cfg.horizons), cfg)
    rows = []
    for key, sm in segment_models.items():
        fit = elasticity_fits.get(key)
        daily_spend = float(base[key][:horizon].mean())
        m_roas = marginal_roas(daily_spend, fit) if fit else np.nan
        a_roas = average_roas_at(daily_spend, fit) if fit else np.nan
        plat, ctype = split_segment_key(key)
        rows.append({
            "segment": key, "channel": plat, "campaign_type": ctype,
            "daily_spend": daily_spend, "beta": sm.beta,
            "avg_roas": a_roas, "marginal_roas": m_roas,
        })
    df = pd.DataFrame(rows).sort_values("marginal_roas", ascending=False)
    return df.reset_index(drop=True)
