# Technical Documentation — Forecasting Methodology

AIgnition 3.0 · Probabilistic Revenue Forecasting for E-commerce Marketing

This document explains the forecasting methodology, model selection, preprocessing
logic, assumptions, limitations, and AI-integration strategy.

---

## 1. The problem, restated precisely

Given historical daily campaign statistics from three heterogeneous ad platforms,
forecast, as **probabilistic ranges** over **30 / 60 / 90-day aggregate windows**:

- **Ecommerce revenue** (aggregate of Google + Meta + Bing) and **blended ROAS**,
- the same at **channel**, **campaign-type**, and **campaign** granularity,
- conditioned on **future media budgets** (what-if), and
- accompanied by **AI-assisted causal explanations**.

Attribution is taken as the source of truth; building an attribution engine or a
full Media-Mix Model is explicitly out of scope.

## 2. What the data told us (and how it shaped the design)

Profiling the sample (25,562 rows, 2024-01-01 → 2026-06-05) drove every modelling
decision:

| Observation | Design consequence |
|---|---|
| **Three different schemas.** Google reports cost in micros + a channel-type column; Meta reports **no revenue or type column** — its `conversion` field is actually conversion *value* (proven: 55% of rows have `conversion > clicks`, impossible for a count; implied ROAS median 4.1); Bing is clean. | Schema-fingerprint ingestion (§4) that normalises all three to one canonical panel and derives a shared taxonomy. |
| **Annual seasonality dominates.** Monthly revenue index swings from ~0.29 (summer) to ~3.40 (December) — a >10× peak-to-trough — over only **two** observed holiday cycles. | Seasonality is modelled explicitly with **partial pooling** across segments (§6) so two cycles don't overfit; an offline marketing calendar names the events. |
| **Diminishing returns are real.** Log–log spend→revenue slope ≈ 0.71 (Google), 0.63 (Meta); by type 0.80 (Search), 0.67 (PMax), 0.56 (Shopping) — all < 1. | A **constant-elasticity** spend-response (§7) is the transparent backbone of budget simulation. |
| **Campaign series are short & sparse.** 83 / 136 campaigns have < 180 days; Bing campaigns are ~$0 at the median day; the top 10 campaigns are 58% of revenue. | Model at the robust **channel × type** level, then **disaggregate to campaigns by recent share** (§8) rather than fitting noise per campaign. |
| **Budget columns are unreliable.** Meta `daily_budget` is mostly tiny/zero (spend/budget median 0.03); Google overspends its per-campaign budget 46% of the time. | Drive off historical **spend**, not the raw budget field; treat "future budget" as a planned-spend input to the simulator. |
| **The final period is ragged.** June 2026 is a partial extract. | A reporting-completeness heuristic trims incomplete trailing days (§5) so we never forecast a fake crash. |

## 3. Forecasting hierarchy

```
aggregate  →  channel (google / meta / bing)  →  campaign-type (channel × type)  →  campaign
```

The **segment** = *channel × campaign-type* (e.g. `google::Search`, `meta::Social -
Prospecting`) is the modelling unit: long enough to estimate seasonality and
elasticity robustly (~13 segments here), and the natural level at which agencies
actually plan. Everything above is an aggregation of segments; campaigns are a
disaggregation of them.

## 4. Preprocessing & ingestion

1. **Schema-fingerprint detection.** Each CSV is matched to a platform by the *set of
   columns present* (not filename — the harness may rename files), with filename
   hints only as a tiebreak. Unrecognised files are skipped, never fatal.
2. **Canonicalisation.** Google cost-micros → currency; Meta `conversion` → revenue;
   ids kept as strings (no float/scientific-notation corruption); negatives floored;
   unparseable dates dropped.
3. **Taxonomy.** Every campaign is classified along two orthogonal axes — a
   normalised **format** (Search, Shopping, Performance Max, Display, Video, Demand
   Gen, Audience, Social-Prospecting/Remarketing/Generic) from the native type or the
   campaign name, and a **brand intent** (Brand / Non-Brand / Unknown from TM/NTM,
   Brand/Generic tokens). Agencies steer brand vs. non-brand very differently, so this
   is first-class.
4. **Dense panels.** Segment- and campaign-daily series are gap-filled with zeros from
   each unit's launch to the forecast origin (a no-spend day genuinely earned ~$0).
5. **Recent shares** (last 90 days) drive campaign disaggregation.

## 5. Ragged-edge handling

We trim a trailing day only if its number of *actively reporting series* is < 50% of
the trailing-28-day median (and only within the last 3 days). Using reporting
**completeness** rather than revenue **level** makes this robust to seasonality — a
genuine low-season day still has the usual number of campaigns reporting; an
incomplete extract does not.

## 6. The structural model (per series)

For a segment's daily revenue (and, separately, spend):

```
value_t  ≈  level · growth(t) · seasonal_week(t) · seasonal_dow(t) · residual_t
```

- **Seasonality.** A multiplicative **ISO-week** profile (captures the Q4 ramp at the
  right calendar position) estimated by ratio-to-annual-moving-average (isolating
  within-year shape from level drift), circularly smoothed, plus a **day-of-week**
  profile. Each segment's own estimate is **shrunk toward a pooled global/per-type
  prior** with weight `n / (n + 365)` — James–Stein-style partial pooling that
  stabilises the two-cycle estimate. The frozen model *remembers* each segment's
  seasonal shape, so even a truncated held-out input forecasts the holidays correctly.
- **Level.** A winsorised trailing mean of the deseasonalised series (robust to
  outliers and the ragged edge), with an estimate of its own sampling uncertainty.
- **Trend.** A **damped** year-over-year growth rate (falls back to a shrunk linear
  slope when < 1 year is available). Damping makes cumulative growth *saturate*
  instead of exploding at long horizons — the single most common failure mode of naive
  trend extrapolation.
- **Residual.** Kept as a block of recent multiplicative residuals for a **block
  bootstrap** (preserves autocorrelation and fat tails) plus a log-scale sigma
  fallback.

## 7. Spend → revenue response (budget simulation)

Within a segment, `revenue ≈ A · spend^β` with `0 < β < 1`. `β` is fit robustly per
segment at train time (one MAD-trimmed log–log regression), bounded to `[0.15, 1.05]`,
and **frozen**. In a budget scenario, scaling spend by `f` scales revenue by `f^β`
(anchored on the segment's own recent efficiency), giving realistic **diminishing
returns**; the absolute form yields marginal ROAS `A·β·spend^(β-1)` for "where does the
next dollar work hardest?" analysis. This is a deliberately transparent alternative to
a full MMM, consistent with the brief's scope.

## 8. Uncertainty, simulation & coherence

Point-and-interval closed forms don't exist for an aggregate-period *sum* of a
seasonal, trended, autocorrelated, non-linearly spend-driven series — nor for the
*ratio* ROAS. So we simulate:

1. For each segment and each of *N* worlds, draw a level shock (lognormal), a growth
   shock (normal), a block-bootstrapped residual path, and a shared **macro shock**
   (a slow random walk common to all segments → realistic cross-segment correlation).
2. Couple revenue to spend through the frozen elasticity.
3. Sum daily → 30/60/90-day period totals; ROAS = ΣRev / ΣSpend per world.
4. **Coherence.** Summing the *same world index* across segments yields channel and
   aggregate paths that add up exactly, with correct correlation — no post-hoc
   reconciliation (MinT/OLS) needed. Campaigns are disaggregated top-down by recent
   share, exactly coherent with their segment.
5. **Split-conformal calibration.** A rolling-origin backtest measures how far actuals
   land from the simulated median relative to the simulated spread; a frozen per-level,
   per-horizon multiplier rescales interval width (median-preserving, log-space) so
   realised P10–P90 coverage matches nominal. Means are taken from the *raw*
   distribution so they stay additive across the hierarchy.

## 9. Model selection & why *not* the alternatives

| Candidate | Why not (here) |
|---|---|
| **Prophet / NeuralProphet** | Heavy `cmdstan`/compilation dependency that is fragile to ship and unpickle offline; per-series fitting is unstable with 2 seasonal cycles; harder to make coherent across a hierarchy. |
| **ARIMA / SARIMA / ETS** | Struggles with a >10× annual swing from 2 cycles and with exogenous spend; no native diminishing-returns response; per-campaign fitting on short/sparse series is noisy. |
| **Global gradient-boosted quantile model (LightGBM)** | Strong point accuracy but a black box (the brief prizes *interpretability* and *causal explanation*), harder to keep hierarchically coherent, and pickling a boosted model raises version-mismatch unpickling risk at eval. Used only as an offline benchmark. |
| **Full MMM (adstock + saturation)** | Explicitly out of scope; over-parameterised for the data and timeline. |

We chose a **glass-box structural + Monte-Carlo + conformal** design because it is
(a) transparent and directly explainable — which the AI layer exploits, (b) naturally
probabilistic and hierarchically coherent, (c) dependency-light and safe to unpickle
offline, and (d) validated: rolling-origin backtest median APE ≈ **13–22%** across
horizons with conformal-calibrated ~80% coverage (see the console output of
`python src/train.py` and `docs/architecture.md`).

## 10. Assumptions

- Reported attribution is the source of truth; Meta `conversion` = conversion value.
- Segment structure (`platform × campaign_type`) recurs across advertisers, so frozen
  per-segment priors transfer to held-out data of the same schema.
- Future spend either follows its own forecast or is supplied by the planner; the
  spend-response elasticity estimated historically holds locally around current spend.
- USD throughout; one advertiser per data drop.

## 11. Limitations

- Only ~2 holiday cycles: extreme Q4 magnitudes carry genuine irreducible uncertainty
  (reflected in wider calibrated intervals, not hidden).
- Constant elasticity is a local approximation, not a global saturation curve;
  extreme budget changes (>2–3× current) extrapolate beyond observed support.
- Campaign-level forecasts inherit their segment's shape (top-down); brand-new
  campaigns with no history are represented only through their segment.
- No cross-channel cannibalisation/halo modelling (out of scope: no MMM).
- Elasticity is associational, not a randomised causal estimate — the AI layer is
  careful to frame drivers as "primarily associated with", not proven causation.

## 12. AI integration strategy

See [`architecture.md`](architecture.md) §AI. In short: a **deterministic driver
engine** computes grounded attribution, risks and marginal-ROAS opportunities from
the model internals; an **optional LLM** turns that into an executive briefing under
strict no-fabrication instructions; and a **fully-offline template** guarantees quality
insights with no key and no network. The LLM never runs on the scored pipeline.
