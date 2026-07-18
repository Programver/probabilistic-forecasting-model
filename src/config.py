"""
Central configuration for the AIgnition probabilistic forecasting utility.

Everything that a maintainer might want to change without touching business
logic lives here: canonical schema maps for each ad platform, the campaign
taxonomy vocabulary, forecast horizons, the quantile grid, simulation controls,
reproducibility seeds, and — importantly — the *output contract* used to write
``predictions.csv``.

The output schema is deliberately isolated (see ``OUTPUT_SCHEMA`` and
``PREDICTION_COLUMNS``) so that if the organizers publish an exact
``predictions.csv`` column spec, we can conform to it in one place without
rewriting the forecasting engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Paths (all RELATIVE — never absolute; the eval machine has different roots)
# ---------------------------------------------------------------------------
# Project root = parent of this file's directory (src/ -> project root).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DEFAULT_DATA_DIR: Path = PROJECT_ROOT / "data"
DEFAULT_MODEL_PATH: Path = PROJECT_ROOT / "pickle" / "model.pkl"
DEFAULT_OUTPUT_PATH: Path = PROJECT_ROOT / "output" / "predictions.csv"
DEFAULT_FEATURES_PATH: Path = PROJECT_ROOT / "features.parquet"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 20260712  # AIgnition submission deadline as a memorable seed

# ---------------------------------------------------------------------------
# Business / unit constants
# ---------------------------------------------------------------------------
MICROS_PER_UNIT: float = 1_000_000.0  # Google reports cost in micros
CURRENCY: str = "USD"

# ---------------------------------------------------------------------------
# Canonical (unified) schema — the internal column vocabulary every platform
# adapter maps onto. This is the contract between ingest.py and everything else.
# ---------------------------------------------------------------------------
CANON_COLUMNS: List[str] = [
    "date",            # datetime64[ns]  observation day
    "platform",        # str  google | meta | bing
    "campaign_id",     # str  stable id (kept as string to avoid int overflow / mixed types)
    "campaign_name",   # str
    "channel_type_raw",  # str | None  native type column if present
    "spend",           # float  media cost in CURRENCY
    "revenue",         # float  attributed conversion value in CURRENCY (source of truth)
    "clicks",          # float
    "impressions",     # float
    "conversions",     # float | NaN  attributed conversions (count) if available
    "budget_raw",      # float | NaN  native daily/campaign budget (used cautiously)
]

# Per-platform adapter specs. Each maps native column names -> canonical names,
# plus platform-specific transforms handled in ingest.py.
#
# We match platforms by the *set of columns present* (schema fingerprint), not by
# filename, because the eval harness may rename files. Filenames are only a hint.
@dataclass(frozen=True)
class PlatformSpec:
    name: str
    # Signature columns that uniquely identify this platform's schema.
    signature: Tuple[str, ...]
    # native -> canonical column rename map
    rename: Dict[str, str]
    date_col: str
    # transforms to apply (documented flags consumed by ingest.py)
    cost_in_micros: bool = False
    revenue_from_conversion_value: bool = False
    filename_hints: Tuple[str, ...] = ()


GOOGLE_SPEC = PlatformSpec(
    name="google",
    signature=("campaign_advertising_channel_type", "metrics_cost_micros",
               "metrics_conversions_value"),
    rename={
        "campaign_id": "campaign_id",
        "segments_date": "date",
        "metrics_clicks": "clicks",
        "metrics_impressions": "impressions",
        "metrics_conversions": "conversions",
        "metrics_conversions_value": "revenue",
        "campaign_advertising_channel_type": "channel_type_raw",
        "campaign_budget_amount": "budget_raw",
        "campaign_name": "campaign_name",
        # metrics_cost_micros handled specially (-> spend / 1e6)
    },
    date_col="segments_date",
    cost_in_micros=True,
    filename_hints=("google", "gads", "google_ads"),
)

META_SPEC = PlatformSpec(
    name="meta",
    # Meta is identified by cpc/cpm/ctr + a `conversion` (value) column and no
    # explicit revenue/type column.
    signature=("cpc", "cpm", "ctr", "conversion"),
    rename={
        "campaign_id": "campaign_id",
        "date_start": "date",
        "spend": "spend",
        "clicks": "clicks",
        "impressions": "impressions",
        "conversion": "revenue",     # PROVEN: `conversion` is conversion *value* (revenue)
        "daily_budget": "budget_raw",
        "campaign_name": "campaign_name",
    },
    date_col="date_start",
    revenue_from_conversion_value=True,
    filename_hints=("meta", "facebook", "fb", "meta_ads"),
)

BING_SPEC = PlatformSpec(
    name="bing",
    signature=("Revenue", "Spend", "CampaignType", "TimePeriod"),
    rename={
        "CampaignId": "campaign_id",
        "TimePeriod": "date",
        "Revenue": "revenue",
        "Spend": "spend",
        "Clicks": "clicks",
        "Impressions": "impressions",
        "Conversions": "conversions",
        "CampaignType": "channel_type_raw",
        "DailyBudget": "budget_raw",
        "CampaignName": "campaign_name",
    },
    date_col="TimePeriod",
    filename_hints=("bing", "microsoft", "msads", "ms_ads"),
)

PLATFORM_SPECS: List[PlatformSpec] = [GOOGLE_SPEC, META_SPEC, BING_SPEC]

# ---------------------------------------------------------------------------
# Campaign-type taxonomy
# ---------------------------------------------------------------------------
# Normalised, platform-agnostic campaign *format* vocabulary. Native type
# strings and campaign-name tokens are mapped onto these.
CAMPAIGN_TYPES: List[str] = [
    "Search",
    "Shopping",
    "Performance Max",
    "Display",
    "Video",
    "Demand Gen",
    "Audience",
    "Social - Prospecting",
    "Social - Remarketing",
    "Social - Generic",
    "Other",
]

# Native platform type-string -> canonical campaign type.
NATIVE_TYPE_MAP: Dict[str, str] = {
    # Google channel types
    "SEARCH": "Search",
    "PERFORMANCE_MAX": "Performance Max",
    "SHOPPING": "Shopping",
    "DISPLAY": "Display",
    "VIDEO": "Video",
    "DEMAND_GEN": "Demand Gen",
    # Bing campaign types
    "Search": "Search",
    "PerformanceMax": "Performance Max",
    "Shopping": "Shopping",
    "Audience": "Audience",
    "Display": "Display",
}

# Ordered (regex token, canonical type) rules applied to campaign_name when a
# native type is missing (Meta) or to refine it. First match wins.
#
# NOTE: campaign names delimit tokens with underscores (e.g. "Search_TM_Campaign"),
# and "_" is a regex word char, so `\bTM\b` would NOT match "_TM_". We use
# letter-boundary lookarounds `(?<![a-z])X(?![a-z])` (case-insensitive) so that
# underscores, digits, spaces and string ends all act as delimiters.
NAME_TYPE_RULES: List[Tuple[str, str]] = [
    (r"(?<![a-z])pmax(?![a-z])|performance[_ ]?max", "Performance Max"),
    (r"(?<![a-z])shopping(?![a-z])", "Shopping"),
    (r"demand[_ ]?gen", "Demand Gen"),
    (r"(?<![a-z])video(?![a-z])", "Video"),
    (r"(?<![a-z])display(?![a-z])", "Display"),
    (r"remarketing|retargeting", "Social - Remarketing"),
    (r"prospecting", "Social - Prospecting"),
    (r"(?<![a-z])dpa(?![a-z])|dynamic[_ ]?product", "Social - Prospecting"),
    (r"adv[_ ]?plus|advantage", "Social - Prospecting"),
    (r"generic", "Social - Generic"),
    (r"(?<![a-z])search(?![a-z])", "Search"),
]

# Brand-intent tokens. Digital agencies steer brand vs non-brand spend very
# differently, so we surface it as a first-class dimension. Same letter-boundary
# treatment as above (NTM checked before TM so "Pmax_NTM" is not read as brand).
BRAND_TOKENS: List[str] = [r"(?<![a-z])tm(?![a-z])", r"trademark", r"(?<![a-z])brand(?![a-z])"]
NONBRAND_TOKENS: List[str] = [r"(?<![a-z])ntm(?![a-z])", r"non[_ -]?brand",
                              r"generic", r"prospecting"]

BRAND_LABELS = ("Brand", "Non-Brand", "Unknown")

# ---------------------------------------------------------------------------
# Forecast controls
# ---------------------------------------------------------------------------
HORIZONS_DAYS: List[int] = [30, 60, 90]      # aggregate planning windows
QUANTILES: List[float] = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
N_SIMULATIONS: int = 4000                    # Monte-Carlo paths per segment
# Campaign-level disaggregation is per-campaign x per-horizon x n_simulations
# arrays; an upload with orders of magnitude more campaigns than the sample
# data (e.g. a full-history advertiser export) can otherwise blow up memory
# and get OOM-killed. Cap it to the top-N campaigns per segment by recent
# revenue; the tail is immaterial to spend decisions and stays reconciled at
# the campaign_type level.
MAX_CAMPAIGNS_PER_SEGMENT: int = 500
# Minimum history (days) for a segment to trust its *own* seasonal estimate;
# below this we lean harder on the pooled global profile.
MIN_DAYS_OWN_SEASONALITY: int = 365
MIN_DAYS_SEGMENT_MODEL: int = 28             # below this, a segment is treated as sparse
TREND_DAMPING: float = 0.90                  # damped-trend factor (phi) per 30d step
TREND_MAX_ABS_MONTHLY_GROWTH: float = 0.35   # clip runaway extrapolation (+/-35%/mo)
RECENT_LEVEL_WINDOW: int = 56                # days used to estimate current level
RECENT_SHARE_WINDOW: int = 90               # days used for campaign disaggregation shares

# ---------------------------------------------------------------------------
# Output contract  (predictions.csv)
# ---------------------------------------------------------------------------
# Long / tidy schema: one row per (horizon, level, segment, metric).
# This single structure encodes every deliverable the brief requires:
#   - aggregate ecommerce revenue + blended ROAS
#   - channel-level / campaign-type / campaign-level revenue & ROAS ranges
#   - probabilistic quantiles for each
# If the organizers announce a different exact schema, adapt PREDICTION_COLUMNS
# and the writer in io_utils.write_predictions — nothing else needs to change.
LEVELS: List[str] = ["aggregate", "channel", "campaign_type", "campaign"]
METRICS: List[str] = ["revenue", "spend", "roas"]

# Quantile column names, e.g. 0.05 -> "p05", 0.5 -> "p50".
def q_col(q: float) -> str:
    return f"p{int(round(q * 100)):02d}"

PREDICTION_COLUMNS: List[str] = [
    "forecast_origin",   # last observed date used as origin (ISO)
    "horizon_days",      # 30 | 60 | 90
    "period_start",      # ISO date (origin + 1)
    "period_end",        # ISO date (origin + horizon)
    "level",             # aggregate | channel | campaign_type | campaign
    "channel",           # blended | google | meta | bing
    "campaign_type",     # all | <canonical type>
    "campaign_id",       # all | <id>
    "campaign_name",     # all | <name>
    "metric",            # revenue | spend | roas
    "currency",          # USD (roas rows carry currency for uniformity)
    "mean",
    "std",
] + [q_col(q) for q in QUANTILES]

OUTPUT_SCHEMA = {
    "columns": PREDICTION_COLUMNS,
    "quantiles": QUANTILES,
    "levels": LEVELS,
    "metrics": METRICS,
    "format": "long",  # long | wide
}

# ---------------------------------------------------------------------------
# Model / training controls
# ---------------------------------------------------------------------------
MODEL_VERSION: str = "1.0.0"
BACKTEST_ORIGINS: int = 8          # rolling-origin folds for conformal calibration
BACKTEST_STEP_DAYS: int = 30       # spacing between backtest origins
# Elasticity guardrails (diminishing returns): revenue ~ spend**beta.
ELASTICITY_BOUNDS: Tuple[float, float] = (0.15, 1.05)
ELASTICITY_DEFAULT: float = 0.65   # global prior when a segment can't estimate its own


@dataclass
class ForecastConfig:
    """Runtime knobs bundled so they can be pickled inside the model and
    overridden by CLIs / the app without touching module globals."""
    horizons: List[int] = field(default_factory=lambda: list(HORIZONS_DAYS))
    quantiles: List[float] = field(default_factory=lambda: list(QUANTILES))
    n_simulations: int = N_SIMULATIONS
    seed: int = RANDOM_SEED
    trend_damping: float = TREND_DAMPING
    recent_level_window: int = RECENT_LEVEL_WINDOW
    recent_share_window: int = RECENT_SHARE_WINDOW
    min_days_own_seasonality: int = MIN_DAYS_OWN_SEASONALITY
    currency: str = CURRENCY

    def q_labels(self) -> List[str]:
        return [q_col(q) for q in self.quantiles]
