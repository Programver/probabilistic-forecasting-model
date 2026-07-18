# AIgnition 3.0 — Probabilistic Revenue Forecasting for E-commerce Marketing

An AI-assisted forecasting utility for digital marketing agencies. It ingests
fragmented Google / Meta / Microsoft (Bing) Ads data, produces **probabilistic**
forecasts of **ecommerce revenue and blended ROAS** across a coherent hierarchy
(aggregate → channel → campaign-type → campaign) for **30 / 60 / 90-day** planning
windows, simulates the revenue/ROAS impact of **future media budgets** under
diminishing returns, and generates **AI-assisted causal briefings**.

> Built for NetElixir's AIgnition 3.0. The scored pipeline runs fully **offline**,
> **deterministically**, from a **single `run.sh`**, against a **pre-trained,
> committed model** — exactly as the submission contract requires.

**Two halves, both part of the product:** the forecasting **pipeline** (`run.sh` →
`predictions.csv`) and the **interactive dashboard** (`app/streamlit_app.py` — forecast,
budget simulator, AI insights, data). **[How to run it](#how-to-run-it)** covers the
complete setup for both, including the optional 30-second API-key step that enables
LLM-polished AI Insights. No key is *required* anywhere — every tab still works
without one — but the key shows the AI layer at full strength.

---

## Quickstart (the exact scored path)

> This section is the **pipeline only** — the compliance path. For the dashboard and
> the AI Insights tab, see **[How to run it](#how-to-run-it)** below.

```bash
# 1. Fresh environment (Python 3.12)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. One command: features -> model -> predictions
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
#   (defaults are identical, so `./run.sh` alone also works)

# 3. Read the forecast
open output/predictions.csv
```

`run.sh` (a) generates features from whatever CSVs are in `DATA_DIR`, then
(b) loads the committed `pickle/model.pkl` and writes `output/predictions.csv`.
No network, no prompts, no retraining, seeded for reproducibility.

**The trained model is already committed** (`pickle/model.pkl`). You do **not**
need to train to score. To rebuild it from data (optional, dev-time only):

```bash
python src/train.py --data-dir ./data --out ./pickle/model.pkl
```

> `run.sh` is **the** entry point required by the submission contract.
> `run_project.sh` is an optional convenience wrapper that simply calls `run.sh` or
> Streamlit for you (`./run_project.sh pipeline` | `./run_project.sh app`) — handy,
> but it adds nothing you cannot do with the two commands directly.

### Verify the whole thing

```bash
python tests/test_pipeline.py     # 13 tests, no pytest needed (pytest -q also works)
```

Covers ingestion, the Meta `conversion`=revenue invariant, taxonomy, preprocessing,
the end-to-end forecast, the **exact output schema**, quantile monotonicity, null-free
output, hierarchy coherence, single-platform robustness, and clean model loading.

---

## How to run it

The submission has two runnable halves, and **both are part of the product**:

- **The scored pipeline** — `run.sh` → `output/predictions.csv` (the forecasting engine).
- **The interactive product** — `app/streamlit_app.py`: a 4-tab dashboard
  (📊 Forecast · 🎛️ Budget Simulator · 🧠 AI Insights · 📄 Data & Predictions).

The recommended setup below runs **everything**, including LLM-polished AI Insights.
An API key is **optional** — every tab, the CLI and the pipeline are fully functional
without one — but adding a key is a 30-second step and shows the AI layer at full
strength, so it is worth doing.

### Recommended — the complete product (one venv, everything working)

```bash
# 1. Environment (Python 3.12) — one venv covers the pipeline AND the app
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-app.txt        # includes requirements.txt via -r

# 2. Optional but recommended: enable LLM-polished AI Insights (~30s)
cp .env.example .env
#    then edit .env and paste ONE key, e.g.:  GEMINI_API_KEY=your-key-here
#    Gemini (default, SDK already installed): https://aistudio.google.com/apikey
#    OpenAI:    https://platform.openai.com/api-keys     (uncomment `openai` in requirements-app.txt)
#    Anthropic: https://console.anthropic.com/settings/keys (uncomment `anthropic`)

# 3. The forecasting pipeline -> predictions.csv
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv

# 4. The dashboard (reads .env automatically — no `export` needed)
streamlit run app/streamlit_app.py
```

In the app: the sidebar states which LLM provider is live (or that it is running
offline). On **🧠 AI Insights**, switch on *"Use LLM to polish the briefing"* and
click **Generate briefing**.

### Without any API key — everything still runs

Skip step 2 entirely. Nothing breaks and nothing is hidden:

- All four tabs render, with the full forecast, budget simulator and response curves.
- **🧠 AI Insights** still produces a complete briefing — the driver attribution,
  ROAS watch-list and budget recommendations are computed **deterministically from
  the model internals, not from an LLM**, so the analysis is identical either way.
  Only the *prose* is templated instead of LLM-written, and the sidebar says so.
- `run.sh` is unaffected — it never imports the LLM layer at all, with or without a key.

This is a designed fallback, not a degraded mode: the LLM is deliberately confined to
rewriting figures it is handed, so the insight itself never depends on a network call.
If a key is missing, invalid, rate-limited, or offline, the app falls back silently
and never crashes.

### Scored pipeline on its own (minimal deps)

If you only want `predictions.csv`, the minimal path needs nothing but
`requirements.txt` — no key, no `.env`, no internet:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

### CLI briefing (no Streamlit needed)

```bash
python src/briefing.py            # LLM-polished if a key is present, offline if not
python src/briefing.py --no-llm   # force the offline narrative
```

> `.env` is git-ignored — never commit real keys. Only `.env.example` ships.

---

## What it produces (`output/predictions.csv`)

A tidy **long** table — one row per *horizon × hierarchy node × metric* — encoding
every required deliverable with a full probabilistic fan:

| column | meaning |
|---|---|
| `forecast_origin` | last observed date used as the origin |
| `horizon_days` | 30, 60 or 90 (aggregate planning window) |
| `period_start`, `period_end` | forecast window bounds |
| `level` | `aggregate` \| `channel` \| `campaign_type` \| `campaign` |
| `channel` | `blended` \| `google` \| `meta` \| `bing` |
| `campaign_type`, `campaign_id`, `campaign_name` | hierarchy node identifiers |
| `metric` | `revenue` \| `spend` \| `roas` |
| `currency` | `USD` (money) / `ratio` (ROAS) |
| `mean`, `std` | point + dispersion |
| `p05 … p95` | quantile fan (probabilistic interval) |

On the bundled sample this is **1,404 rows across all four levels, with no nulls**.
A *dormant* campaign — zero forecast spend **and** zero forecast revenue, which is
~45% of campaigns in this dataset — reports the conventional **ROAS of 0** (the same
way the ad platforms report it) rather than an undefined `0/0`. The one case left as
`NaN` is genuinely undefined: zero spend against *positive* revenue (infinite ROAS),
which we decline to invent a number for.

Per-campaign detail is capped at the top **500 campaigns per segment** by recent
revenue (`config.MAX_CAMPAIGNS_PER_SEGMENT`) so a very large advertiser cannot
exhaust memory during disaggregation. The largest segment in this dataset has 58
campaigns, so the cap never engages here; if it ever does it logs a warning, and the
dropped tail remains fully accounted for at the `campaign_type` level and above —
totals stay exact.

The exact column set lives in one place (`src/config.py::PREDICTION_COLUMNS`) so it
can be conformed to any officially-announced schema in minutes without touching the
forecasting engine.

---

## Repository structure

```
.
├── run.sh                     # single entry point (required, scored)
├── run_project.sh             # optional human convenience wrapper (pipeline | app)
├── requirements.txt           # scored-path deps, pinned (required)
├── requirements-app.txt       # optional app + LLM extras
├── .env.example               # template for optional LLM API keys (copy -> .env; .env is git-ignored)
├── .streamlit/config.toml     # dark theme for the product app
├── data/                      # sample CSVs — overwritten by the harness at test time
│   ├── google_ads_campaign_stats.csv
│   ├── meta_ads_campaign_stats.csv
│   └── bing_campaign_stats.csv
├── pickle/
│   └── model.pkl              # committed, pre-trained model (required)
├── output/                    # predictions.csv is written here
├── src/
│   ├── config.py              # schema maps, taxonomy, horizons, quantiles, OUTPUT contract
│   ├── ingest.py              # schema-fingerprint adapters -> canonical panel
│   ├── taxonomy.py            # campaign -> (channel, type, brand intent)
│   ├── preprocess.py          # ragged-edge trim, dense segment/campaign panels, shares
│   ├── calendar_features.py   # offline marketing-event calendar (BFCM, holidays)
│   ├── seasonality.py         # pooled ISO-week + day-of-week seasonal profiles
│   ├── decompose.py           # level, damped trend, residual bootstrap blocks
│   ├── response.py            # constant-elasticity spend->revenue (diminishing returns)
│   ├── uncertainty.py         # block bootstrap + split-conformal calibration
│   ├── simulate.py            # Monte-Carlo coupled revenue/spend daily paths
│   ├── hierarchy.py           # coherent reconciliation + campaign disaggregation
│   ├── model.py               # ForecastModel artifact (frozen priors) + build/apply
│   ├── backtest.py            # rolling-origin calibration + accuracy metrics
│   ├── forecast.py            # orchestration: prepared + model -> forecast
│   ├── budget.py              # what-if budget simulation + response curves
│   ├── io_utils.py            # predictions.csv writer (the output contract)
│   ├── logging_utils.py       # shared logger setup
│   ├── insights/              # AI layer: drivers -> narrative -> optional LLM
│   │                          #   drivers.py -> narrative.py -> llm.py, via engine.py
│   ├── generate_features.py   # run.sh step (a)
│   ├── predict.py             # run.sh step (b)
│   ├── train.py               # dev-time model builder (not run at eval)
│   └── briefing.py            # CLI AI briefing (demo)
├── app/streamlit_app.py       # interactive product (4 tabs)
├── docs/                      # methodology, architecture, demo, report, checklist
└── tests/test_pipeline.py     # 13 smoke + contract tests
```

---

## Methodology in one paragraph

We model each **channel × campaign-type segment** with a transparent, glass-box
structural decomposition — `revenue ≈ level · damped-trend · seasonality ·
spend-response · noise` — estimated with partial pooling so segments with only two
observed holiday cycles borrow a stabilised seasonal shape from the population.
Spend drives revenue through a **constant-elasticity** curve (empirically β ≈
0.6–0.8 < 1, i.e. diminishing returns), which powers budget simulation. We quantify
uncertainty by **Monte-Carlo simulating daily paths** (block-bootstrapped,
autocorrelated noise + level/trend/macro shocks), summing them to 30/60/90-day
period totals, and **calibrating interval widths with split conformal** so realised
coverage matches the nominal level. Summing the same simulated world across segments
makes the hierarchy **coherent by construction**. See
[`docs/methodology.md`](docs/methodology.md) and
[`docs/architecture.md`](docs/architecture.md) for the full treatment, and
[`docs/demo_workflow.md`](docs/demo_workflow.md) for the walkthrough.

## AI integration

The insight layer first computes **grounded, deterministic driver attribution**
(how much of the change is seasonality vs. trend vs. spend & mix), a ROAS watch-list,
and marginal-ROAS budget opportunities — all from the model internals, not from a
language model. An **optional LLM** then *polishes* this into an executive briefing,
strictly constrained to the supplied figures. With no API key it falls back to a
fully-offline template narrative, so the product always ships useful insights. The
LLM is **never** invoked on the scored offline pipeline — `run.sh` /
`generate_features.py` / `predict.py` make no network calls and do not even import
the LLM layer, with or without a key set.

Provider auto-detection order is **Gemini → OpenAI → Anthropic**: whichever key is
present (and whose SDK is installed) wins. Any failure — missing key, missing SDK,
no network, timeout, bad response — falls back to the offline narrative and never
raises. For Gemini either SDK works: the supported **`google-genai`** (pinned and
tested) or the EOL `google-generativeai` (accepted as a fallback).

## Reproducibility & compliance

- Single `run.sh`; accepts `DATA_DIR MODEL_PATH OUTPUT_PATH` with sensible defaults.
- **Offline** at run time — no network calls anywhere in `generate_features`/`predict`,
  verified by import-blocking every app/LLM dependency and re-running the pipeline.
- **Deterministic** — all randomness seeded (`config.RANDOM_SEED`, `PYTHONHASHSEED=0`);
  verified byte-identical across repeat runs.
- **No retraining** — the committed model is applied; only closed-form features are
  computed from `data/`.
- **Schema/size-robust** — platforms are detected by column fingerprint (not
  filename); the pipeline handles a different advertiser, date range, or row count.
  Exercised against an unseen campaign type, a 12-day history, all-zero-revenue and
  all-zero-spend data — all produce a valid, null-free output.
- **Null-free output** — see `test_no_nulls_in_scored_output`.
- **Pinned deps**; the model pickle is dependency-light (numpy-only state) to avoid
  version-mismatched unpickling.

Python **3.12** (developed and tested on 3.12.3).

## Team

- **Team:** natbarpradhannatbar008
- **Members:** Natabar Pradhan
- **College:** Indian Institute of Technology, Kharagpur
- **Command to run:** `./run.sh ./data ./pickle/model.pkl ./output/predictions.csv`
