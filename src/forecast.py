"""
Forecast orchestration: PreparedData + frozen model -> probabilistic forecast.

This is the single call the predict CLI, the backtest, the budget simulator and
the Streamlit app all go through, so behaviour is identical everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from hierarchy import GroupForecast, reconcile
from io_utils import groups_to_frame
from logging_utils import get_logger
from model import ForecastModel, SegmentModel, build_segment_models
from preprocess import PreparedData
from simulate import SegmentSim, make_common_shock, simulate_segment

log = get_logger("forecast")


@dataclass
class ForecastResult:
    origin: pd.Timestamp
    horizons: List[int]
    groups: List[GroupForecast]
    predictions: pd.DataFrame
    segment_models: Dict[str, SegmentModel] = field(default_factory=dict)
    agg_daily: Optional[pd.DataFrame] = None   # cumulative revenue fan for plotting
    scenario_label: str = "baseline"

    def get_group(self, level: str, channel: str = "blended", campaign_type: str = "all",
                  campaign_id: str = "all") -> Optional[GroupForecast]:
        for g in self.groups:
            if (g.level == level and g.channel == channel and
                    g.campaign_type == campaign_type and g.campaign_id == campaign_id):
                return g
        return None


def run_forecast(prepared: PreparedData, model: ForecastModel,
                 cfg: Optional[config.ForecastConfig] = None,
                 spend_overrides: Optional[Dict[str, np.ndarray]] = None,
                 rng: Optional[np.random.Generator] = None,
                 scenario_label: str = "baseline") -> ForecastResult:
    """Run the full probabilistic forecast.

    ``spend_overrides`` maps segment_key -> daily spend array (length = max horizon)
    for budget simulation; segments absent from the map use their own spend
    forecast.
    """
    cfg = cfg or model.cfg or config.ForecastConfig()
    rng = rng or np.random.default_rng(cfg.seed)

    segment_models = build_segment_models(prepared, model, cfg)
    max_h = max(cfg.horizons)
    future_dates = pd.date_range(prepared.origin + pd.Timedelta(days=1),
                                 periods=max_h, freq="D")

    common = make_common_shock(cfg.n_simulations, max_h, rng)

    segment_sims: Dict[str, SegmentSim] = {}
    total_rev_daily = np.zeros((cfg.n_simulations, max_h))
    for key, sm in segment_models.items():
        ov = None if spend_overrides is None else spend_overrides.get(key)
        sim = simulate_segment(sm, future_dates, rng, cfg,
                               spend_override_daily=ov, common_log_shock=common)
        segment_sims[key] = sim
        total_rev_daily += sim.revenue_daily

    groups = reconcile(segment_sims, prepared.campaign_meta, cfg)
    predictions = groups_to_frame(groups, prepared.origin, model, cfg)
    agg_daily = _aggregate_daily_fan(total_rev_daily, future_dates, model, cfg)

    return ForecastResult(
        origin=prepared.origin, horizons=list(cfg.horizons), groups=groups,
        predictions=predictions, segment_models=segment_models,
        agg_daily=agg_daily, scenario_label=scenario_label,
    )


def _aggregate_daily_fan(total_rev_daily: np.ndarray, future_dates: pd.DatetimeIndex,
                         model: ForecastModel, cfg: config.ForecastConfig) -> pd.DataFrame:
    """Cumulative-revenue fan chart data (P10/P50/P90 of running total)."""
    cum = np.cumsum(total_rev_daily, axis=1)
    p10, p50, p90 = np.quantile(cum, [0.10, 0.50, 0.90], axis=0)
    return pd.DataFrame({
        "date": future_dates,
        "cum_rev_p10": p10, "cum_rev_p50": p50, "cum_rev_p90": p90,
    })


def forecast_from_panel(panel: pd.DataFrame, model: ForecastModel,
                        cfg: Optional[config.ForecastConfig] = None,
                        spend_overrides: Optional[Dict[str, np.ndarray]] = None,
                        scenario_label: str = "baseline") -> ForecastResult:
    """Convenience: raw canonical panel -> forecast (used by the app)."""
    from preprocess import prepare
    prepared = prepare(panel)
    return run_forecast(prepared, model, cfg, spend_overrides, scenario_label=scenario_label)
