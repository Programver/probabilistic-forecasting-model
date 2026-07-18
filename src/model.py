"""
The frozen model artifact (``pickle/model.pkl``) and how it is applied.

Design intent — satisfying *both* "no retraining at eval" *and* "generalises to
held-out data with the same schema":

* The pickle stores only **transferable, learned structure**, keyed by
  ``segment_key = platform::campaign_type`` (a key that recurs across
  advertisers): pooled seasonal *shapes*, spend elasticities, damping, a
  marketing calendar, and conformal interval multipliers.
* It deliberately does **not** hardcode any advertiser's levels or dates. At
  predict time :func:`build_segment_models` re-derives each segment's *current*
  level, trend and recent seasonal estimate from whatever is in ``data/``
  (deterministic, closed-form — this is "feature generation"), then **shrinks**
  that estimate toward the frozen prior. Long, stable segments trust the data;
  short/truncated ones lean on the remembered shape.

Consequences: the same pickle produces sensible probabilistic forecasts whether
the held-out data is the same advertiser truncated to a new origin, or a new
advertiser with the same schema. No optimisation runs at eval — only means,
medians, one closed-form blend, and seeded simulation.

The artifact is intentionally **dependency-light**: it contains only our own
dataclasses, numpy arrays, dicts and floats — never a third-party model object —
so it unpickles cleanly under any compatible numpy, side-stepping the single most
common cause of hackathon scoring failures (version-mismatched unpickling).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

import config
from decompose import Decomposition, decompose_series
from logging_utils import get_logger
from preprocess import PreparedData
from response import ElasticityFit, fit_elasticity
from seasonality import (SeasonalProfile, blend, factor_for_dates,
                         fit_seasonal_profile, flat_profile, pool, pooling_weight)
from taxonomy import split_segment_key
from uncertainty import ConformalCalibration

log = get_logger("model")


# ---------------------------------------------------------------------------
# Per-segment model built at predict time (data-derived + frozen priors)
# ---------------------------------------------------------------------------
@dataclass
class SegmentModel:
    segment_key: str
    platform: str
    campaign_type: str
    rev_decomp: Decomposition
    spend_decomp: Decomposition
    seasonal_rev: SeasonalProfile
    seasonal_spend: SeasonalProfile
    beta: float
    n_days: int = 0
    total_revenue: float = 0.0
    total_spend: float = 0.0
    recent_daily_spend: float = 0.0   # baseline daily spend anchor (for budget sim)


# ---------------------------------------------------------------------------
# The frozen artifact
# ---------------------------------------------------------------------------
@dataclass
class ForecastModel:
    version: str = config.MODEL_VERSION
    cfg: config.ForecastConfig = field(default_factory=config.ForecastConfig)

    # Seasonal priors (multiplicative shapes), keyed at three levels of specificity.
    global_seasonal_rev: SeasonalProfile = field(default_factory=flat_profile)
    global_seasonal_spend: SeasonalProfile = field(default_factory=flat_profile)
    seasonal_by_type_rev: Dict[str, SeasonalProfile] = field(default_factory=dict)
    seasonal_by_type_spend: Dict[str, SeasonalProfile] = field(default_factory=dict)
    seasonal_by_segment_rev: Dict[str, SeasonalProfile] = field(default_factory=dict)
    seasonal_by_segment_spend: Dict[str, SeasonalProfile] = field(default_factory=dict)

    # Spend elasticities.
    elasticity_by_segment: Dict[str, float] = field(default_factory=dict)
    elasticity_by_type: Dict[str, float] = field(default_factory=dict)
    elasticity_global: float = config.ELASTICITY_DEFAULT
    elasticity_fits: Dict[str, ElasticityFit] = field(default_factory=dict)  # for curves/insights

    # Calibrated interval widths.
    conformal: ConformalCalibration = field(default_factory=ConformalCalibration)

    # Metadata.
    trained_origin: Optional[str] = None
    trained_meta: Dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    # ---- prior lookups (specific -> general fallback) ----
    def seasonal_prior(self, segment_key: str, which: str) -> SeasonalProfile:
        _, ctype = split_segment_key(segment_key)
        seg = self.seasonal_by_segment_rev if which == "revenue" else self.seasonal_by_segment_spend
        typ = self.seasonal_by_type_rev if which == "revenue" else self.seasonal_by_type_spend
        glob = self.global_seasonal_rev if which == "revenue" else self.global_seasonal_spend
        if segment_key in seg:
            return seg[segment_key]
        if ctype in typ:
            return typ[ctype]
        return glob

    def elasticity(self, segment_key: str) -> float:
        _, ctype = split_segment_key(segment_key)
        if segment_key in self.elasticity_by_segment:
            return self.elasticity_by_segment[segment_key]
        if ctype in self.elasticity_by_type:
            return self.elasticity_by_type[ctype]
        return self.elasticity_global

    def elasticity_fit(self, segment_key: str) -> Optional[ElasticityFit]:
        return self.elasticity_fits.get(segment_key)


# ---------------------------------------------------------------------------
# Fitting global priors (train time)
# ---------------------------------------------------------------------------
def fit_global_priors(prepared: PreparedData,
                      cfg: Optional[config.ForecastConfig] = None) -> ForecastModel:
    """Learn transferable seasonal shapes and elasticities from training data."""
    cfg = cfg or config.ForecastConfig()
    model = ForecastModel(cfg=cfg)

    rev_profiles, spd_profiles, weights, types = [], [], [], []
    seg_keys = []
    for key in prepared.segments:
        daily = prepared.segment_series(key)
        _, ctype = split_segment_key(key)
        pr = fit_seasonal_profile(daily, "revenue")
        ps = fit_seasonal_profile(daily, "spend")
        model.seasonal_by_segment_rev[key] = pr
        model.seasonal_by_segment_spend[key] = ps
        rev_profiles.append(pr); spd_profiles.append(ps)
        weights.append(max(float(daily["revenue"].sum()), 1.0))
        types.append(ctype); seg_keys.append(key)

        efit = fit_elasticity(daily)
        model.elasticity_fits[key] = efit
        model.elasticity_by_segment[key] = efit.beta

    # Global pooled priors (revenue-weighted).
    model.global_seasonal_rev = pool(rev_profiles, weights)
    model.global_seasonal_spend = pool(spd_profiles, weights)

    # Per-type pooled priors + per-type elasticity (revenue-weighted median).
    df = pd.DataFrame({"key": seg_keys, "type": types, "w": weights,
                       "beta": [model.elasticity_by_segment[k] for k in seg_keys]})
    for ctype, grp in df.groupby("type"):
        idx = [seg_keys.index(k) for k in grp["key"]]
        model.seasonal_by_type_rev[ctype] = pool([rev_profiles[i] for i in idx],
                                                 [weights[i] for i in idx])
        model.seasonal_by_type_spend[ctype] = pool([spd_profiles[i] for i in idx],
                                                   [weights[i] for i in idx])
        model.elasticity_by_type[ctype] = float(
            np.average(grp["beta"], weights=grp["w"]))
    model.elasticity_global = float(np.median(df["beta"])) if len(df) else config.ELASTICITY_DEFAULT

    model.trained_origin = str(prepared.origin.date())
    model.trained_meta = {
        "n_segments": len(prepared.segments),
        "n_campaigns": int(prepared.campaign_meta.shape[0]) if prepared.campaign_meta is not None else 0,
        "origin": str(prepared.origin.date()),
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    model.notes.extend(prepared.notes)
    log.info("fit global priors: %d segments, %d types; elasticity global=%.2f",
             len(prepared.segments), len(model.elasticity_by_type), model.elasticity_global)
    return model


# ---------------------------------------------------------------------------
# Building per-segment models at PREDICT time (data + frozen priors)
# ---------------------------------------------------------------------------
def build_segment_models(prepared: PreparedData, model: ForecastModel,
                         cfg: Optional[config.ForecastConfig] = None
                         ) -> Dict[str, SegmentModel]:
    """Construct SegmentModels from current data, shrinking own seasonality toward
    the frozen prior. No optimisation — closed-form estimates + one blend."""
    cfg = cfg or model.cfg or config.ForecastConfig()
    out: Dict[str, SegmentModel] = {}
    for key in prepared.segments:
        daily = prepared.segment_series(key)
        plat, ctype = split_segment_key(key)

        own_rev = fit_seasonal_profile(daily, "revenue")
        own_spd = fit_seasonal_profile(daily, "spend")
        w = pooling_weight(own_rev.n_days, k=cfg.min_days_own_seasonality)
        seas_rev = blend(own_rev, model.seasonal_prior(key, "revenue"), w)
        seas_spd = blend(own_spd, model.seasonal_prior(key, "spend"), w)

        rev_dec = decompose_series(daily, seas_rev, "revenue", cfg)
        spd_dec = decompose_series(daily, seas_spd, "spend", cfg)

        recent = daily.tail(cfg.recent_level_window)
        recent_daily_spend = float(recent["spend"].mean()) if len(recent) else spd_dec.level

        out[key] = SegmentModel(
            segment_key=key, platform=plat, campaign_type=ctype,
            rev_decomp=rev_dec, spend_decomp=spd_dec,
            seasonal_rev=seas_rev, seasonal_spend=seas_spd,
            beta=model.elasticity(key),
            n_days=int(len(daily)),
            total_revenue=float(daily["revenue"].sum()),
            total_spend=float(daily["spend"].sum()),
            recent_daily_spend=recent_daily_spend,
        )
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_model(model: ForecastModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    log.info("saved model -> %s (%.1f KB)", path, path.stat().st_size / 1024)


def load_model(path: str | Path) -> ForecastModel:
    model = joblib.load(Path(path))
    if not isinstance(model, ForecastModel):
        raise TypeError(f"Loaded object is not a ForecastModel: {type(model)}")
    return model
