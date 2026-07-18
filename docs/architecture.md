# Architecture Overview

AIgnition 3.0 · Probabilistic Revenue Forecasting

## Stacks at a glance

| Layer | Technology | Notes |
|---|---|---|
| **Scored pipeline** | Python 3.12, `numpy`, `pandas`, `joblib` | Minimal, pinned, **offline**, deterministic. The only thing the harness runs. |
| **Model artifact** | Custom dataclasses + numpy arrays (`joblib`) | Dependency-light → unpickles under numpy alone. |
| **Forecasting engine** | Pure numpy/pandas structural model + Monte-Carlo + conformal | Glass-box, no ML framework in the shipped path. |
| **Frontend (product)** | Streamlit + Plotly | Dashboard, budget simulator, AI insights. Optional (`requirements-app.txt`). |
| **AI layer** | Deterministic driver engine + optional Gemini/OpenAI/Anthropic | Offline template fallback; never on the scored path. |
| **Experimentation** | Rolling-origin backtest (`src/backtest.py`) | Calibration + accuracy metrics. |

## Two execution paths

The design cleanly separates the **scored, offline, deterministic** path from the
**interactive, optionally-online** product path. They share the same forecasting core,
so numbers are identical everywhere.

```
                          ┌───────────────────────────── data/ (Google / Meta / Bing CSVs) ─────────────────────────────┐
                          │                                                                                             │
  SCORED PATH (run.sh)    ▼                                        PRODUCT PATH (app / CLI)                              ▼
  ┌──────────────────────────────────┐                            ┌──────────────────────────────────────────────────────┐
  │ generate_features.py             │                            │ streamlit_app.py  /  briefing.py                       │
  │  ingest → taxonomy → preprocess  │                            │  (upload or bundled data)                              │
  │  → PreparedData (features.pkl)   │                            └───────────────────────┬────────────────────────────────┘
  └───────────────┬──────────────────┘                                                    │
                  │                                                                         │
                  ▼                                                                         ▼
  ┌──────────────────────────────────┐        ┌─────────────────────────────┐   ┌────────────────────────────────────────┐
  │ predict.py                       │        │ FORECAST ENGINE (shared)    │   │ budget.py  (what-if, paired scenarios) │
  │  load pickle/model.pkl (frozen)  │───────▶│ build_segment_models        │◀──│ response curves, marginal ROAS         │
  │  run_forecast → predictions.csv  │        │ simulate (MC) → reconcile   │   └────────────────────────────────────────┘
  └───────────────┬──────────────────┘        │ conformal → summarize        │
                  │                            └──────────────┬──────────────┘
                  ▼                                           ▼
        output/predictions.csv                    insights/  drivers → narrative → (optional) LLM
        output/forecast_summary.json                          │
                                                              ▼
                                                    executive briefing (markdown)
```

## Forecasting pipeline (module by module)

1. **`ingest.py`** — schema-fingerprint adapters → one canonical long panel.
2. **`taxonomy.py`** — campaign → (channel, campaign_type, brand_intent).
3. **`preprocess.py`** — ragged-edge trim, dense segment/campaign daily panels,
   recent shares → `PreparedData`.
4. **`seasonality.py` / `decompose.py`** — pooled seasonal profiles; level, damped
   trend, residual blocks.
5. **`response.py`** — constant-elasticity spend→revenue (frozen `β`).
6. **`simulate.py`** — Monte-Carlo coupled revenue/spend daily paths with a shared
   macro shock.
7. **`hierarchy.py`** — coherent reconciliation (sum worlds) + campaign disaggregation.
8. **`uncertainty.py`** — block bootstrap + split-conformal interval calibration.
9. **`io_utils.py`** — writes `predictions.csv` in the output contract.
10. **`model.py`** — the frozen `ForecastModel` (global priors, elasticities, conformal,
    calendar) + `build_segment_models` (applies priors to current data).
11. **`backtest.py`** — rolling-origin calibration + accuracy metrics (train-time).
12. **`forecast.py`** — orchestration used identically by predict, backtest, app.

### Why the model artifact generalises without retraining

`pickle/model.pkl` stores **only transferable structure** keyed by
`segment_key = platform::campaign_type` (a key that recurs across advertisers): pooled
seasonal *shapes*, spend elasticities, damping, the marketing calendar, and conformal
multipliers. At predict time, each segment's **current level, trend and recent seasonal
estimate are re-derived from whatever is in `data/`** (closed-form — this is "feature
generation") and shrunk toward the frozen prior. So the same pickle produces sensible
forecasts whether the held-out data is the same advertiser truncated to a new origin, or
a new advertiser with the same schema — and **no optimisation runs at eval**.

## LLM integration workflow (AI layer)

```
ForecastResult ─▶ drivers.build_driver_report ─▶ DriverReport (grounded facts:
   • per-horizon revenue/ROAS + P10–P90                    seasonality/trend/spend
   • change vs trailing, attributed to                     attribution, ROAS watch,
     seasonality / trend / spend & mix                     marginal-ROAS opportunities,
   • channel movers, calendar events, data quality)        calendar events)
        │
        ├─▶ narrative.to_markdown ──────────────▶ deterministic offline briefing (always available)
        │
        └─▶ llm.generate(report, draft) ─┬─ key+SDK+network present ─▶ Gemini/OpenAI/Anthropic
                                          │      (system prompt: use ONLY supplied figures,
                                          │       communicate P10–P90, no fabrication)
                                          └─ otherwise / on any error ─▶ None ─▶ fall back to offline draft
```

Key properties: the LLM only ever *rewrites* facts the deterministic engine already
computed (grounded, reproducible, no hallucinated numbers); it degrades gracefully to a
high-quality offline narrative; and it is **never** imported or called by
`generate_features.py` / `predict.py`, preserving the "no network at run time" guarantee.

Provider precedence and models are environment-configurable
(`AIGNITION_LLM_PROVIDER`, `GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`,
`AIGNITION_GEMINI_MODEL`, …).

## Reliability & engineering practices

- **Determinism:** seeded RNG (`config.RANDOM_SEED`), `PYTHONHASHSEED=0` in `run.sh`.
- **Fail loud:** `set -euo pipefail`; scripts raise non-zero on error.
- **Config-driven:** schema maps, taxonomy, horizons, quantiles, and the *output
  contract* are all isolated in `config.py`.
- **Logging:** consistent namespaced logging to stderr (stdout stays clean).
- **Defensive ingestion:** unknown files skipped, dirty rows coerced, never crashes on
  one bad record or a different data size.
- **Tests:** `tests/` covers ingestion, taxonomy, the end-to-end contract, output
  schema, and hierarchy coherence.
