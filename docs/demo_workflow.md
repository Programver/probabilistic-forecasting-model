# Demo Workflow

A short, reproducible walkthrough of the four things a marketing agency does with
AIgnition: **ingest → forecast → simulate a budget → get AI insights**.

Two halves, both part of the product: the **CLI pipeline** (the forecasting engine,
scored via `run.sh` → `predictions.csv`) and the **interactive app** (the 4-tab
dashboard). Both are covered below; run whichever you like, or both.

---

## A. Scored pipeline (60 seconds, offline)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run.sh
```

You'll see the pipeline (1) ingest and normalise the three platforms, (2) trim the
incomplete trailing day, (3) load the frozen model, and (4) write predictions. The
console prints the headline:

```
30d revenue P50=211,933 [P10=133,301 .. P90=340,345] | ROAS P50=3.30
60d revenue P50=448,314 [P10=334,403 .. P90=602,001] | ROAS P50=3.31
90d revenue P50=684,687 [P10=550,278 .. P90=850,894] | ROAS P50=3.41
```

Outputs:
- `output/predictions.csv` — full tidy forecast (every level × horizon × metric with a
  P5–P95 fan).
- `output/forecast_summary.json` — machine-readable headline.

Inspect any slice, e.g. channel-level 30-day ROAS:

```bash
python - <<'PY'
import pandas as pd
d = pd.read_csv("output/predictions.csv")
print(d[(d.level=="channel") & (d.horizon_days==30) & (d.metric=="roas")]
      [["channel","p10","p50","p90"]].to_string(index=False))
PY
```

## B. AI briefing (CLI)

```bash
python src/briefing.py                       # deterministic, fully offline
GEMINI_API_KEY=your_key python src/briefing.py   # LLM-polished (Gemini)
```

Produces an executive briefing: headline forecast with uncertainty, an **attribution**
of the change into seasonality / trend / spend & mix, a **channel outlook**, **risks**
(low/declining ROAS, wide bands), and **recommended actions** driven by *marginal ROAS*
("scale Meta Remarketing — marginal ROAS 6.3× — the next dollar works hardest here;
review Google Video — 0.4× — cap before adding budget"). With no key it still produces
the full briefing from templates; the numbers are identical.

## C. Interactive product (Streamlit)

```bash
pip install -r requirements-app.txt
streamlit run app/streamlit_app.py
```

The demo storyline across the four tabs:

1. **📊 Forecast** — Pick data (bundled sample or upload your own Google/Meta/Bing
   CSVs). See 30/60/90-day revenue KPI cards with 80% bands and blended ROAS, a
   **cumulative-revenue fan chart**, revenue by channel (with P10–P90 error bars),
   revenue by campaign type, and a top-campaigns table.

2. **🎛️ Budget Simulator** — Move each channel's spend slider (e.g. Meta +30%), hit
   **Run simulation**, and read the paired baseline-vs-scenario result: new spend,
   revenue (P50), blended ROAS, and — the number planners care about — **incremental
   ROAS** on the marginal budget. The *"where does the next dollar work hardest?"* table
   ranks segments by marginal ROAS, and the **diminishing-returns curve** shows the
   response for any segment with its current-spend marker.

3. **🧠 AI Insights** — Generate the briefing (toggle LLM polish on/off). The change is
   visualised as an **attribution waterfall** (seasonality vs trend vs spend & mix)
   beside the narrative.

4. **📄 Data & Predictions** — The full `predictions.csv` (downloadable), data-quality
   notes (e.g. the trimmed trailing day), and a preview of the normalised input.

## Suggested 3-minute demo script

1. *"Agencies plan spend before results exist."* Open **Forecast** on the bundled data:
   "Next 30 days: ~$212K revenue, but note the honest 80% band $133K–$340K, and ROAS
   3.3×. It's forecasting the summer trough — the model knows the calendar."
2. Switch to **Budget Simulator**: bump Meta +30%. "Revenue responds +23%, not +30% —
   diminishing returns (β≈0.85). Incremental ROAS on that budget is ~7×. The table says
   put the *next* dollar into Meta Remarketing and Google Search; cap Google Video."
3. Open **AI Insights**: "The briefing attributes the outlook to seasonality (−24%) more
   than spend, flags Meta/Bing ROAS to watch, and recommends the reallocation — grounded
   in the model, optionally written by an LLM."
4. Close on **Data & Predictions**: "Every number is in `predictions.csv` in the exact
   submission format, produced by one offline `run.sh`."
