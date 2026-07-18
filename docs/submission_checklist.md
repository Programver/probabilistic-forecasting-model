# Submission Checklist — compliance with the Hackathon Submission Guide

Every requirement from the guide, mapped to where and how it is satisfied. All
items verified on a **clean clone in a fresh virtual environment** (see
`docs/architecture.md` and the clean-run evidence below).

| # | Requirement | Status | Where / how |
|---|---|---|---|
| 1 | Public GitHub repo | ⬜ *(do at submission)* | `git init && git add -A && git commit && git push` to a **public** repo. `pickle/model.pkl` is committed (not LFS, not ignored). |
| 2 | `run.sh` at root, executable, one command end-to-end | ✅ | [`run.sh`](../run.sh); `chmod +x`; also runs via `bash run.sh`. Does feature-gen **and** predict in one invocation. |
| 3 | `run.sh` accepts `DATA_DIR MODEL_PATH OUTPUT_PATH` with defaults | ✅ | `./run.sh` == `./run.sh ./data ./pickle/model.pkl ./output/predictions.csv`. |
| 4 | `data/` folder, read dynamically | ✅ | Platforms detected by **column fingerprint**, not filename; reads every CSV in `data/`; handles a different advertiser / size / date range. |
| 5 | Trained model committed under `pickle/`, loads cleanly | ✅ | [`pickle/model.pkl`](../pickle/model.pkl) (~15 KB). Dependency-light (numpy arrays + our dataclasses) → unpickles under the pinned numpy alone. |
| 6 | `requirements.txt` with pinned versions | ✅ | [`requirements.txt`](../requirements.txt) — `numpy==2.5.0`, `pandas==3.0.3`, `joblib==1.5.3`, `python-dateutil==2.9.0.post0`, `holidays==0.99`. |
| 7 | Output to `OUTPUT_PATH`, announced format, written fresh | ✅ | [`src/io_utils.py`](../src/io_utils.py) writes fresh every run; schema in `config.PREDICTION_COLUMNS` (config-swappable). |
| 8 | No retraining at eval | ✅ | `predict.py` only *applies* the frozen model; feature-gen is closed-form (no optimisation). |
| 9 | Fully offline at run time | ✅ | No network imports on the scored path; the LLM layer is never imported by `generate_features.py`/`predict.py`. |
| 10 | Deterministic / seeds set | ✅ | `config.RANDOM_SEED`, `np.random`/`random` seeded in every entry script, `PYTHONHASHSEED=0` in `run.sh`. |
| 11 | No absolute paths | ✅ | All paths are CLI args or relative to the repo root (`Path(__file__)`). |
| 12 | No interactive input | ✅ | `argparse` only; no prompts. |
| 13 | Fail loudly | ✅ | `set -euo pipefail`; scripts raise → non-zero exit. |
| 14 | Python version stated | ✅ | 3.12 (developed on 3.12.3) — README + `requirements.txt`. |
| 15 | Tested on a clean clone in a fresh env | ✅ | Verified: fresh `venv` + `pip install -r requirements.txt` + `./run.sh` → valid `predictions.csv` (1404 rows, all 4 levels, deterministic). |
| 16 | Output contains **no nulls** | ✅ | `predictions.csv` is null-free. Dormant campaigns (zero spend *and* zero revenue — ~46% of the sample) report the conventional ROAS of **0** rather than NaN; previously 189/1404 rows were null. Locked by `test_no_nulls_in_scored_output`. |
| 17 | Robust to unusual held-out data | ✅ | Scored path exercised against: a new advertiser on future dates with an **unseen campaign type**, a **12-day** history, **all-zero-revenue** and **all-zero-spend** data. All exit 0 with a valid, null-free, monotone-quantile schema. |
| 18 | `run.sh` executable bit survives the commit | ✅ | Verified `git ls-files -s run.sh` → mode `100755` (a `100644` here would break `./run.sh` on a fresh clone). |
| 19 | No secrets committed | ✅ | `.env` is git-ignored and verified absent from a simulated `git add -A`; only `.env.example` ships. The LLM key is never required (see item 9). |

## Deliverables (Project Brief)

| Deliverable | Where |
|---|---|
| Working prototype (ingest, validate, budget input, probabilistic revenue+ROAS, channel/type/campaign outputs, AI summaries) | `run.sh` pipeline + `app/streamlit_app.py` |
| Technical documentation (methodology, model selection, preprocessing, assumptions, limitations, AI strategy) | [`docs/methodology.md`](methodology.md) |
| Architecture overview (frontend, backend, forecasting pipeline, LLM workflow) | [`docs/architecture.md`](architecture.md) |
| Demo workflow (ingestion, forecast, budget simulation, AI insights) | [`docs/demo_workflow.md`](demo_workflow.md) |

## What you must fill before sending

- Team name, members, college — in [`README.md`](../README.md) (Team section).
- Push to a **public** GitHub repo and submit the URL + the run command
  (`./run.sh ./data ./pickle/model.pkl ./output/predictions.csv`) to
  `sunitha.k@netelixir.us` by the deadline.
