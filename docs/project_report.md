# AIgnition 3.0 — Full Project Report

Probabilistic Revenue Forecasting for E-commerce Marketing — everything we did,
step by step, with the numbers, algorithms, and libraries.

---

## 0. Objective

Build an AI-assisted forecasting utility that, from fragmented Google / Meta /
Microsoft (Bing) Ads data, produces **probabilistic** forecasts of **e-commerce
revenue** and **blended ROAS** over **30 / 60 / 90-day** windows, across a
coherent hierarchy (aggregate → channel → campaign-type → campaign), supports
**future-budget what-if** simulation with diminishing returns, and generates
**AI-assisted causal briefings** — while running fully **offline, deterministically,
from a single `run.sh`** against a **pre-trained, committed** model (the scoring
contract).

---

## 1. Datasets (raw)

| File | Rows | Platform | Notable schema quirks |
|---|---|---|---|
| `google_ads_campaign_stats.csv` | 19,272 | Google | cost in **micros** (`metrics_cost_micros`, ÷1e6); revenue = `metrics_conversions_value`; native type in `campaign_advertising_channel_type` |
| `meta_ads_campaign_stats.csv` | 3,417 | Meta | **no revenue or type column**; `conversion` is conversion *value*; `daily_budget` mostly null/zero |
| `bing_campaign_stats.csv` | 2,873 | Bing | clean `Revenue` / `Spend` / `CampaignType` |
| **Total** | **25,562** | — | date span **2024-01-01 → 2026-06-05** (~2.4 years) |

**Aggregate economics (full history):**

| Platform | Revenue | Spend | ROAS | Campaigns | Share of revenue |
|---|---|---|---|---|---|
| Google | $9,266,678 | $1,946,126 | 4.76x | 92 | 83% |
| Meta | $1,656,751 | $196,387 | 8.44x | 16 | 15% |
| Bing | $172,028 | $39,430 | 4.36x | 28 | ~2% |
| **Total** | **$11,095,456** | **$2,181,943** | **~5.09x** | **136** | 100% |

---

## 2. Data cleaning & normalization (step by step)

Module: `src/ingest.py`. Every platform's raw frame is mapped onto one canonical
schema `[date, platform, campaign_id, campaign_name, channel_type_raw, spend,
revenue, clicks, impressions, conversions, budget_raw]`.

1. **Platform detection by column fingerprint** — each file is matched to a
   platform by the *set of columns present*, not filename (the eval harness may
   rename files). Filename hints are only a tiebreak; unrecognized files are
   skipped, never fatal.
2. **Unit fix** — Google `metrics_cost_micros` ÷ 1,000,000 → spend in USD.
3. **Meta revenue fix** — `conversion` column mapped to `revenue` (proven to be
   conversion *value*, see §3).
4. **ID hygiene** — `campaign_id` coerced to clean strings (no float scientific
   notation / trailing `.0`), avoiding int-overflow on Meta's 18-digit ids.
5. **Numeric coercion** — `pd.to_numeric(errors="coerce")` on all metrics.
6. **Negative flooring** — negative spend/revenue (refund noise / errors) floored
   to 0 and logged.
7. **Date parsing** — `pd.to_datetime(errors="coerce")`; rows with unparseable
   dates dropped (can't be placed on a timeline).
8. **Deduplication / aggregation** — grouped to one row per
   `(platform, campaign, date)` by summation.
9. **Dense daily panel** (`src/preprocess.py`) — each segment/campaign series is
   gap-filled with zeros from its launch date to the forecast origin (a no-spend
   day genuinely earned ~$0), so time-series methods see a regular grid.
10. **Ragged-edge trimming** — the final day (2026-06-05) is a partial extract; a
    **reporting-completeness heuristic** (a trailing day is dropped only if its
    count of actively-reporting series < 50% of the trailing-28-day median, within
    the last 3 days) removes it. This uses *completeness*, not revenue *level*, so
    it is robust to genuine low-season days. Result: **forecast origin = 2026-06-04**.

---

## 3. Exploratory Data Analysis (findings that shaped the model)

EDA scripts profiled schema, date ranges, cardinality, missingness, seasonality,
elasticity, and series density. Key findings:

- **Meta `conversion` = revenue (not a count).** Proof: **1,892 of 3,417 rows
  (55%) have `conversion > clicks`**, impossible for a conversion count; implied
  ROAS (`conversion/spend`) median **4.08**. → treat as revenue.
- **Sparsity / zeros:** zero-revenue rows — Google **34.4%**, Meta **31.5%**,
  **Bing 84.8%** (Bing is extremely sparse; median Bing campaign earns ~$0/day).
- **Annual seasonality is enormous.** Google monthly revenue index (1.0 = avg):
  Jun **0.29**, Jul **0.33**, Aug **0.36** (summer trough) → Nov **2.25**, Dec
  **3.40** (holiday peak); Jan **0.69**. A **>10× peak-to-trough swing** over only
  **2 observed holiday cycles** → demands regularization/pooling.
- **Weekly seasonality:** strong on Meta (Tue **1.34**, Sat **0.71**, Sun 0.76),
  weak on Google (0.92–1.05), mild on Bing.
- **Diminishing returns (log–log spend→revenue slope, an elasticity proxy):**
  Google **0.71**, Meta **0.63**, Bing **0.18**; by type Search **0.80**, PMax
  **0.67**, Shopping **0.58**, Demand Gen 0.23, Video 1.10 — nearly all **< 1**,
  confirming concave response.
- **Unreliable budget columns:** spend/budget median — Google **0.84** (46%
  overspend, p90 6.7), Meta **0.03** (unusable), Bing 0.52 → drive off historical
  *spend*, not the raw budget field.
- **Series length / concentration:** 136 campaigns, but **83/136 have <180 days**,
  **48/136 have <60 days**; median days Google 142 / Meta 196 / Bing 72; **top 10
  campaigns = 58% of revenue** (Pareto). → model at the robust **channel × type**
  level (13 segments), disaggregate to campaigns by share.
- **Channel-type cardinality:** Google — PERFORMANCE_MAX (13,982 rows), SEARCH
  (4,096), VIDEO (476), DEMAND_GEN (368), SHOPPING (267), DISPLAY (83); Bing —
  PerformanceMax (1,385), Search (1,335), Shopping (87), Audience (66).

---

## 4. Feature engineering

The "features" are the components of a **multiplicative structural decomposition**,
computed deterministically from the data (no label-fitting at inference):

1. **Taxonomy features** (`src/taxonomy.py`) — every campaign classified along two
   orthogonal axes:
   - `campaign_type` (format): Search, Shopping, Performance Max, Display, Video,
     Demand Gen, Audience, Social-Prospecting/Remarketing/Generic — from the native
     type where present, else regex on the campaign name (letter-boundary
     lookarounds so `_TM_`/`_NTM_` tokens match correctly).
   - `brand_intent`: Brand / Non-Brand / Unknown (TM vs NTM, Brand vs Generic).
   - `segment_key = platform::campaign_type` → **13 modelling segments**.
2. **Seasonal profiles** (`src/seasonality.py`):
   - **ISO-week (annual) profile** (53-dim, multiplicative) via **ratio-to-annual-
     moving-average** (365-day centered MA isolates within-year shape from level
     drift), then **circular moving-average smoothing** (5-week window), normalized
     to geometric mean 1.
   - **Day-of-week profile** (7-dim) estimated on the annual-deseasonalized series.
3. **Level & trend** (`src/decompose.py`):
   - **Level** = winsorized (10%-trimmed) mean of the deseasonalized series over a
     56-day trailing window, plus a log-scale uncertainty on that estimate.
   - **Trend** = **year-over-year growth** (deseasonalized recent vs same window
     one year earlier) when ≥400 days of history; else a shrunk linear slope.
4. **Residual blocks** — the last 120 days of multiplicative residuals kept for a
   block bootstrap, plus a log-scale sigma.
5. **Elasticity** (`src/response.py`) — per-segment `β` (see §5).
6. **Recent campaign shares** (last 90 days) for top-down campaign disaggregation.

---

## 5. Modelling methodology (algorithms, in order)

Design choice: a **glass-box structural model + Monte-Carlo + conformal** rather
than Prophet / ARIMA / a black-box GBM — chosen for interpretability (the brief
prizes causal explanation), native probabilism, hierarchical coherence, and
offline unpickle-safety. Per segment:

```
value_t  ≈  level · growth(t) · seasonal_week(t) · seasonal_dow(t) · spend_response · residual_t
```

Algorithms used:
- **Ratio-to-moving-average seasonal decomposition** (multiplicative).
- **Circular smoothing** of the weekly profile.
- **Partial pooling / James–Stein-style shrinkage** — each segment's own seasonal
  estimate is blended (in log space) toward a pooled global/per-type prior with
  weight `w = n / (n + 365)`. Long segments trust themselves; short/truncated ones
  lean on the remembered population shape. The frozen model *stores* each
  segment's shape so a truncated held-out input still forecasts the holidays.
- **Damped trend** — monthly growth increments damped geometrically (φ = 0.90) so
  cumulative growth **saturates** instead of exploding at long horizons; total
  multiplier clipped to [0.4, 2.5].
- **Constant-elasticity spend response** — `revenue ≈ A · spend^β`, `β` fit by a
  **robust log–log regression with one MAD-trim reweighting pass**, bounded to
  [0.15, 1.05]. Fitted per-segment β (frozen in the model):

  | Segment type | β | Segment type | β |
  |---|---|---|---|
  | Search | 0.81 | Social-Remarketing | 0.88 |
  | Performance Max | 0.78 | Social-Prospecting | 0.79 |
  | Shopping | 0.56 | Social-Generic | 0.44 |
  | Display | 0.65 | Demand Gen | 0.30 |
  | Audience | 0.65 | Video | 1.05 |
  | **Global default** | **0.65** | | |

- **Winsorized mean** for robust level; **geometric (log-space) blending** for
  profile pooling.

---

## 6. Uncertainty quantification

Module: `src/uncertainty.py`, `src/simulate.py`. Because the target is an
aggregate-period **sum** and a **ratio** (ROAS), there's no closed form → we
simulate.

- **Monte-Carlo simulation** — **4,000 paths** per segment (configurable). Each
  "world" draws: a level shock (lognormal), a growth shock (normal), a
  **block-bootstrapped** residual path (block length 7, preserves autocorrelation &
  fat tails), and a **shared macro shock** (a slow random walk common to all
  segments → realistic cross-segment correlation).
- **Spend–revenue coupling** — revenue responds to each world's spend through the
  frozen elasticity (sub-linear), so higher-spend worlds earn more, with
  diminishing returns.
- **Aggregation** — sum daily paths to 30/60/90-day totals; ROAS = ΣRev/ΣSpend per
  world; take empirical quantiles P5…P95.
- **Split-conformal calibration** — a rolling-origin backtest measures how far
  actuals land from the simulated median relative to the simulated spread; a frozen
  **per-level, per-horizon multiplier** rescales interval width (median-preserving,
  log-space) so realized coverage matches nominal. Calibrated multipliers:

  | Horizon | Conformal width scale |
  |---|---|
  | 30d | 1.66 (widened — short-horizon intervals were too tight) |
  | 60d | 1.21 |
  | 90d | 0.97 (left ~as-is) |

- Means are taken from the **raw** distribution so they stay **additive** across
  the hierarchy (channel means sum to the aggregate mean).

---

## 7. Hierarchical reconciliation

Module: `src/hierarchy.py`. Coherence "for free": summing the **same simulated
world index across segments** makes channels and the aggregate add up **exactly**,
with correct correlation — no post-hoc MinT/OLS reconciliation needed. Campaigns
are **top-down disaggregated by recent revenue/spend share** (bottom-up per-campaign
fitting would be pure noise given the sparsity), exactly coherent with their
segment; the extra campaign-level volatility is restored via a per-level conformal
widening (campaign scale = channel × 1.25).

---

## 8. Budget simulation (what-if)

Module: `src/budget.py`. A planned budget (global multiplier / per-channel / per-
segment / total) becomes a per-segment daily spend path fed through the same
engine. **Paired** baseline-vs-scenario runs share the same random draws, so
untouched channels cancel and the delta reflects only the budget change.

Worked example (+30% Meta over 30 days):

| Channel | Baseline rev | Scenario rev | Δ | Spend Δ |
|---|---|---|---|---|
| Google | $238,193 | $238,193 | +0.0% | +0.0% (untouched → cancels) |
| Meta | $9,243 | $11,409 | **+23.4%** | +30.0% |
| Bing | $1,017 | $1,017 | +0.0% | +0.0% |
| **Aggregate** | $249,482 | $251,971 | +1.0% | blended ROAS 3.78 → 3.72 |

Note +30% spend → **+23.4%** revenue (not +30%) — diminishing returns at β≈0.85, as
designed. Also exposes **marginal ROAS** (`A·β·spend^(β-1)`) per segment for the
"where does the next dollar work hardest?" reallocation ranking.

---

## 9. AI insight layer

Module: `src/insights/`. **Deterministic driver attribution first** (grounded in
model internals, not an LLM): decomposes the forecast change vs the trailing period
into **seasonality / trend / spend & mix** contributions, builds a **ROAS
watch-list**, flags declining segments, and ranks **marginal-ROAS opportunities**.
A template engine renders a full offline briefing; an **optional LLM** (Gemini
default, OpenAI/Anthropic supported) then *polishes the prose*, strictly constrained
to the supplied figures. Any failure/no-key → offline fallback. The LLM is **never**
on the scored pipeline.

---

## 10. Training & the frozen artifact

Module: `src/train.py`, `src/model.py`. Training (dev-time only, `python
src/train.py`) learns transferable structure and freezes it in
`pickle/model.pkl` (~15 KB): pooled seasonal shapes (global / per-type /
per-segment), elasticities, damping, marketing calendar, and conformal multipliers.
**The pickle contains only numpy arrays + our own dataclasses + plain dicts — no
sklearn/xgboost objects** → unpickles under numpy alone (eliminates the #1 scoring
failure: version-mismatched unpickling). At predict time, per-segment *levels/trend/
recent seasonality* are re-derived from the current data (closed-form "feature
generation") and shrunk toward the frozen prior — so the same pickle generalizes to
a new advertiser or a new date range with **no retraining**.

---

## 11. Evaluation (rolling-origin backtest)

Module: `src/backtest.py`. **8 rolling origins, 30-day step**, forecasting from data
available up to each origin and scoring against what actually followed:

| Horizon | MAPE | Median APE | P10–P90 coverage (raw) |
|---|---|---|---|
| 30d | 34.2% | **21.8%** | 62.5% |
| 60d | 20.2% | 20.5% | 62.5% |
| 90d | 18.3% | **13.3%** | 75.0% |

- **Median APE 13–22%** across horizons (mean MAPE is inflated by the hard
  holiday-transition origins). Longer horizons are more accurate because averaging
  over more days smooths daily noise.
- Raw coverage was below nominal 80% → the **conformal step widens intervals** to
  restore honest ~80% coverage; that's exactly what the multipliers in §6 do.

---

## 12. Forecast results (origin 2026-06-04, into the summer trough)

Headline aggregate forecast:

| Horizon | Revenue P50 | 80% band (P10–P90) | Blended ROAS |
|---|---|---|---|
| 30d | $211,933 | $133,301 – $340,345 | 3.30x |
| 60d | $448,314 | $334,403 – $602,001 | 3.31x |
| 90d | $684,687 | $550,278 – $850,894 | 3.41x |

Channel split (30d): Google $198,531 (ROAS 3.61), Meta $10,815 (ROAS 1.94), Bing
$961 (ROAS 0.31). Output = **1,404 prediction rows** across **156 forecast groups**
(aggregate + 3 channels + campaign-types + 136 campaigns) × 3 horizons × 3 metrics
(revenue / spend / ROAS), each with mean, std, and a P5–P95 fan. The model
correctly projects the seasonal revenue *and ROAS* dip entering summer — surfaced by
the AI briefing as a risk.

---

## 13. Software architecture & libraries

- **Language / runtime:** Python **3.12.3**.
- **Scored pipeline (pinned, minimal):** `numpy==2.5.0`, `pandas==3.0.3`,
  `joblib==1.5.3`, `holidays==0.99`, `python-dateutil==2.9.0.post0`. No scipy /
  sklearn / statsmodels in the shipped path — the model is hand-rolled numpy.
- **App / AI (separate `requirements-app.txt`):** `streamlit==1.59.0`,
  `plotly==6.8.0`, `python-dotenv==1.1.0`, `google-genai==2.12.0` (OpenAI/Anthropic
  optional). The legacy `google-generativeai` SDK is EOL; `src/insights/llm.py`
  still accepts it as a fallback but pins and tests against `google-genai`.
- **Available but deliberately *not* shipped:** scipy 1.18, scikit-learn 1.9,
  lightgbm 4.6, matplotlib — considered as benchmarks/alternatives; rejected in
  favour of the transparent, dependency-light glass-box model.
- **Structure:** 23 source modules (~4,000 LOC) under `src/`, `app/streamlit_app.py`
  (4-tab product), `docs/` (methodology, architecture, demo, this report), `tests/`.
- **Entry points:** `run.sh` → `generate_features.py` (step a) + `predict.py`
  (step b); `train.py` (dev), `briefing.py` (demo CLI).

---

## 14. Testing & verification

- **Unit/contract tests** (`tests/test_pipeline.py`): **10/10 passing** — ingestion,
  Meta-revenue invariant, taxonomy rules, output-schema exactness, quantile
  monotonicity (P05≤…≤P95), non-negativity, hierarchy **mean coherence**
  (channels sum to aggregate within 2%), ROAS sanity, single-platform robustness,
  dependency-light model load.
- **Clean-clone verification:** on a **fresh venv** (`pip install -r
  requirements.txt`), the exact `./run.sh` produced a valid, **deterministic**
  `predictions.csv` (1,404 rows) — the definitive proof it scores on a machine that
  has never seen the project.
- **Determinism:** seeded RNG (`config.RANDOM_SEED = 20260712`), `PYTHONHASHSEED=0`.

---

## 15. Compliance with the submission contract

Single `run.sh` (accepts `DATA_DIR MODEL_PATH OUTPUT_PATH`, sensible defaults);
fully offline; no retraining; deterministic; reads `data/` dynamically by column
fingerprint; committed `pickle/model.pkl`; pinned `requirements.txt`; fresh output
each run. See `docs/submission_checklist.md` for the line-by-line mapping.
```
