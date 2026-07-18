"""
AIgnition — Probabilistic Revenue Forecasting for E-commerce Marketing.

An operational forecasting utility for digital marketing agencies:
  * ingest Google / Meta / Bing campaign data (bundled sample or your own upload),
  * probabilistic revenue & ROAS forecasts (30/60/90d) across the full hierarchy,
  * a what-if budget simulator with diminishing-returns response curves,
  * AI-assisted causal briefings (grounded drivers, optional LLM polish).

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")  # optional: GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "insights"))

import config  # noqa: E402
from budget import (BudgetScenario, baseline_daily_spend, paired_forecasts,  # noqa: E402
                    response_curve, suggest_reallocation)
from forecast import run_forecast  # noqa: E402
from ingest import canonical_from_frames, load_canonical  # noqa: E402
from model import build_segment_models, load_model  # noqa: E402
from preprocess import prepare  # noqa: E402
from taxonomy import split_segment_key  # noqa: E402
from engine import generate_briefing  # noqa: E402

# --------------------------------------------------------------------------- #
# Page setup + light theming
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="AIgnition · Revenue Forecasting", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

CHANNEL_COLORS = {"google": "#4285F4", "meta": "#7B61FF", "bing": "#22B8A6",
                  "blended": "#6C7A89"}
BAND = "rgba(66,133,244,0.18)"

st.markdown("""
<style>
.kpi {background:linear-gradient(135deg,#1b1f2a,#232838);border:1px solid #333a4d;
      border-radius:14px;padding:16px 18px;}
.kpi h3{margin:0;font-size:.8rem;color:#9aa4b2;font-weight:600;text-transform:uppercase;letter-spacing:.04em;}
.kpi .val{font-size:1.7rem;font-weight:700;color:#f2f5fa;margin:.2rem 0 0;}
.kpi .rng{font-size:.82rem;color:#8b96a5;}
.small{color:#8b96a5;font-size:.85rem;}
</style>
""", unsafe_allow_html=True)


def money(x, dp=0):
    return "n/a" if x is None or not np.isfinite(x) else f"${x:,.{dp}f}"


def pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x*100:+.0f}%"


def roas(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.2f}x"


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_model(model_path: str):
    return load_model(model_path)


@st.cache_data(show_spinner=False)
def panel_from_dir(data_dir: str) -> pd.DataFrame:
    return load_canonical(data_dir)


@st.cache_data(show_spinner=False)
def panel_from_uploads(payloads: tuple) -> pd.DataFrame:
    import io
    frames = [(name, pd.read_csv(io.BytesIO(data))) for name, data in payloads]
    return canonical_from_frames(frames)


@st.cache_data(show_spinner=True)
def forecast_cached(sig: str, _panel: pd.DataFrame, _model, n_sims: int):
    cfg = config.ForecastConfig(n_simulations=n_sims)
    _model.cfg = cfg
    prepared = prepare(_panel)
    result = run_forecast(prepared, _model, cfg)
    return prepared, result


def q(group, h, metric, level_q):
    x = group.dist[h][metric]
    x = x[np.isfinite(x)]
    return float(np.quantile(x, level_q)) if x.size else np.nan


# --------------------------------------------------------------------------- #
# Sidebar — data & controls
# --------------------------------------------------------------------------- #
st.sidebar.title("📈 AIgnition")
st.sidebar.caption("Probabilistic revenue & ROAS forecasting for e-commerce marketing.")

model_path = str(ROOT / "pickle" / "model.pkl")
if not Path(model_path).exists():
    st.sidebar.error("No trained model found at pickle/model.pkl. Run `python src/train.py` first.")
    st.stop()
model = get_model(model_path)

src_choice = st.sidebar.radio("Data source", ["Bundled sample (data/)", "Upload CSVs"], index=0)
panel = None
try:
    if src_choice == "Upload CSVs":
        ups = st.sidebar.file_uploader("Google / Meta / Bing campaign CSVs",
                                       type=["csv"], accept_multiple_files=True)
        if ups:
            payloads = tuple((u.name, u.getvalue()) for u in ups)
            panel = panel_from_uploads(payloads)
        else:
            st.sidebar.info("Upload one or more campaign CSVs, or switch to the bundled sample.")
    else:
        panel = panel_from_dir(str(ROOT / "data"))
except Exception as exc:
    st.sidebar.error(f"Could not load data: {exc}")

n_sims = st.sidebar.select_slider("Simulation paths", [1000, 2000, 4000, 6000], value=2000,
                                  help="More paths = smoother intervals, slightly slower.")
st.sidebar.markdown("---")
from llm import available_provider  # noqa: E402
_llm_provider = available_provider()
if _llm_provider:
    st.sidebar.caption(f"**AI insights**: LLM polish active via **{_llm_provider}**. "
                       "Fully-offline narrative is always the fallback.")
else:
    st.sidebar.caption("**AI insights**: no API key found — running fully offline. "
                       "Copy `.env.example` to `.env` and add a key "
                       "(Gemini/OpenAI/Anthropic) to enable LLM-polished briefings.")

if panel is None:
    st.title("AIgnition · Probabilistic Revenue Forecasting")
    st.info("Select a data source in the sidebar to begin.")
    st.stop()

sig = f"{src_choice}:{len(panel)}:{panel['date'].max()}:{n_sims}"
prepared, result = forecast_cached(sig, panel, model, n_sims)
origin = prepared.origin
horizons = result.horizons

st.title("Probabilistic Revenue Forecast")
st.caption(f"Forecast origin **{origin.date()}** · {len(prepared.segments)} segments · "
           f"{prepared.campaign_meta.shape[0]} campaigns · model v{model.version} · "
           f"{n_sims:,} Monte-Carlo paths")

tab_fc, tab_sim, tab_ai, tab_data = st.tabs(
    ["📊 Forecast", "🎛️ Budget Simulator", "🧠 AI Insights", "📄 Data & Predictions"])

# --------------------------------------------------------------------------- #
# TAB 1 — Forecast
# --------------------------------------------------------------------------- #
with tab_fc:
    agg = result.get_group("aggregate")
    cols = st.columns(len(horizons))
    for i, h in enumerate(horizons):
        p50 = q(agg, h, "revenue", 0.5)
        p10 = q(agg, h, "revenue", 0.10)
        p90 = q(agg, h, "revenue", 0.90)
        rs = q(agg, h, "roas", 0.5)
        with cols[i]:
            st.markdown(
                f"<div class='kpi'><h3>{h}-day revenue</h3>"
                f"<div class='val'>{money(p50)}</div>"
                f"<div class='rng'>80% band {money(p10)} – {money(p90)}</div>"
                f"<div class='rng'>blended ROAS {roas(rs)}</div></div>",
                unsafe_allow_html=True)

    st.markdown("### Cumulative revenue outlook")
    import plotly.graph_objects as go
    fan = result.agg_daily
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fan["date"], y=fan["cum_rev_p90"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fan["date"], y=fan["cum_rev_p10"], fill="tonexty",
                             fillcolor=BAND, line=dict(width=0), name="P10–P90"))
    fig.add_trace(go.Scatter(x=fan["date"], y=fan["cum_rev_p50"],
                             line=dict(color=CHANNEL_COLORS["google"], width=3),
                             name="Median (P50)"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="Cumulative revenue ($)", template="plotly_dark",
                      legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, width='stretch')

    c1, c2 = st.columns(2)
    ref_h = st.select_slider("Breakdown horizon", horizons, value=horizons[0], key="fc_h")

    with c1:
        st.markdown(f"#### Revenue by channel ({ref_h}d)")
        rows = []
        for g in result.groups:
            if g.level == "channel":
                rows.append(dict(channel=g.channel.title(),
                                 p50=q(g, ref_h, "revenue", 0.5),
                                 p10=q(g, ref_h, "revenue", 0.10),
                                 p90=q(g, ref_h, "revenue", 0.90),
                                 roas=q(g, ref_h, "roas", 0.5),
                                 color=CHANNEL_COLORS.get(g.channel, "#888")))
        dfc = pd.DataFrame(rows).sort_values("p50", ascending=True)
        figc = go.Figure(go.Bar(
            x=dfc["p50"], y=dfc["channel"], orientation="h",
            marker_color=dfc["color"],
            error_x=dict(type="data", symmetric=False,
                         array=dfc["p90"] - dfc["p50"], arrayminus=dfc["p50"] - dfc["p10"]),
            hovertext=[f"ROAS {roas(r)}" for r in dfc["roas"]]))
        figc.update_layout(height=260, template="plotly_dark",
                           margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Revenue ($)")
        st.plotly_chart(figc, width='stretch')

    with c2:
        st.markdown(f"#### Revenue by campaign type ({ref_h}d)")
        rows = []
        for g in result.groups:
            if g.level == "campaign_type" and g.channel != "blended":
                rows.append(dict(seg=f"{g.channel.title()} · {g.campaign_type}",
                                 p50=q(g, ref_h, "revenue", 0.5),
                                 color=CHANNEL_COLORS.get(g.channel, "#888")))
        dft = pd.DataFrame(rows).sort_values("p50", ascending=True).tail(10)
        figt = go.Figure(go.Bar(x=dft["p50"], y=dft["seg"], orientation="h",
                                marker_color=dft["color"]))
        figt.update_layout(height=260, template="plotly_dark",
                           margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Revenue ($)")
        st.plotly_chart(figt, width='stretch')

    st.markdown(f"#### Top campaigns ({ref_h}d)")
    crows = []
    for g in result.groups:
        if g.level == "campaign":
            crows.append(dict(Channel=g.channel, Type=g.campaign_type, Campaign=g.campaign_name,
                              P10=q(g, ref_h, "revenue", 0.10), P50=q(g, ref_h, "revenue", 0.5),
                              P90=q(g, ref_h, "revenue", 0.90), ROAS=q(g, ref_h, "roas", 0.5)))
    dfk = pd.DataFrame(crows).sort_values("P50", ascending=False).head(15)
    st.dataframe(dfk.style.format({"P10": "${:,.0f}", "P50": "${:,.0f}",
                                   "P90": "${:,.0f}", "ROAS": "{:.2f}x"}),
                 width='stretch', hide_index=True)

# --------------------------------------------------------------------------- #
# TAB 2 — Budget Simulator
# --------------------------------------------------------------------------- #
with tab_sim:
    st.markdown("### What-if budget simulation")
    st.caption("Set planned spend per channel over the window. Revenue responds through the "
               "frozen per-segment elasticity (diminishing returns); baseline and scenario share "
               "the same random draws for a clean, paired comparison.")

    cfg = config.ForecastConfig(n_simulations=n_sims)
    model.cfg = cfg
    seg_models = build_segment_models(prepared, model, cfg)
    base_daily = baseline_daily_spend(seg_models, origin, max(horizons), cfg)

    sim_h = st.select_slider("Planning window", horizons, value=horizons[0], key="sim_h")
    chan_base = {}
    for k, v in base_daily.items():
        p, _ = split_segment_key(k)
        chan_base[p] = chan_base.get(p, 0.0) + float(v[:sim_h].sum())

    st.markdown("#### Planned spend by channel")
    sc = st.columns(len(chan_base))
    channel_budgets = {}
    for i, (chan, base_spend) in enumerate(sorted(chan_base.items())):
        with sc[i]:
            mult = st.slider(f"{chan.title()}  (baseline {money(base_spend)})",
                             0.0, 2.5, 1.0, 0.05, key=f"budg_{chan}")
            channel_budgets[chan] = base_spend * mult

    if st.button("▶ Run simulation", type="primary"):
        scen = BudgetScenario(horizon=sim_h, channel_budgets=channel_budgets, label="Scenario")
        base_res, scen_res = paired_forecasts(prepared, model, scen, cfg)
        st.session_state["sim"] = (sim_h, base_res, scen_res)

    if "sim" in st.session_state:
        sim_h, base_res, scen_res = st.session_state["sim"]
        b = base_res.get_group("aggregate"); s = scen_res.get_group("aggregate")
        b_rev, s_rev = q(b, sim_h, "revenue", 0.5), q(s, sim_h, "revenue", 0.5)
        b_sp, s_sp = q(b, sim_h, "spend", 0.5), q(s, sim_h, "spend", 0.5)
        b_ro, s_ro = q(b, sim_h, "roas", 0.5), q(s, sim_h, "roas", 0.5)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Spend", money(s_sp), pct((s_sp / b_sp - 1) if b_sp else np.nan))
        m2.metric("Revenue (P50)", money(s_rev), pct((s_rev / b_rev - 1) if b_rev else np.nan))
        m3.metric("Blended ROAS", roas(s_ro), f"{s_ro - b_ro:+.2f}x")
        inc_roas = ((s_rev - b_rev) / (s_sp - b_sp)) if abs(s_sp - b_sp) > 1e-6 else np.nan
        m4.metric("Incremental ROAS", roas(inc_roas),
                  help="Δrevenue ÷ Δspend on the change — the return on the *marginal* budget.")

        st.markdown("##### Scenario vs baseline by channel")
        rows = []
        for ch in sorted(chan_base):
            gb = base_res.get_group("channel", ch); gs = scen_res.get_group("channel", ch)
            rows.append(dict(Channel=ch.title(),
                             Baseline=q(gb, sim_h, "revenue", 0.5),
                             Scenario=q(gs, sim_h, "revenue", 0.5),
                             ROAS_base=q(gb, sim_h, "roas", 0.5),
                             ROAS_scen=q(gs, sim_h, "roas", 0.5)))
        dfb = pd.DataFrame(rows)
        import plotly.graph_objects as go
        figb = go.Figure()
        figb.add_bar(x=dfb["Channel"], y=dfb["Baseline"], name="Baseline", marker_color="#4b5566")
        figb.add_bar(x=dfb["Channel"], y=dfb["Scenario"], name="Scenario", marker_color="#4285F4")
        figb.update_layout(barmode="group", height=300, template="plotly_dark",
                           margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Revenue ($)")
        st.plotly_chart(figb, width='stretch')

    st.markdown("---")
    st.markdown("### Where does the next dollar work hardest?")
    realloc = suggest_reallocation(seg_models, model.elasticity_fits, origin, cfg, sim_h)
    realloc_disp = realloc.rename(columns={
        "segment": "Segment", "daily_spend": "Daily spend", "beta": "Elasticity",
        "avg_roas": "Avg ROAS", "marginal_roas": "Marginal ROAS"})[
        ["Segment", "Daily spend", "Elasticity", "Avg ROAS", "Marginal ROAS"]]
    def _mroas_gradient(col: pd.Series) -> list[str]:
        vals = pd.to_numeric(col, errors="coerce")
        lo, hi = vals.min(), vals.max()
        span = (hi - lo) or 1.0
        out = []
        for v in vals:
            if pd.isna(v):
                out.append("")
                continue
            t = (v - lo) / span  # 0 (red) -> 1 (green)
            r = int(220 + (76 - 220) * t)
            g = int(76 + (175 - 76) * t)
            b = int(60 + (80 - 60) * t)
            out.append(f"background-color: rgba({r},{g},{b},0.65); color: #fff")
        return out

    st.dataframe(realloc_disp.style.format({
        "Daily spend": "${:,.0f}", "Elasticity": "{:.2f}",
        "Avg ROAS": "{:.2f}x", "Marginal ROAS": "{:.2f}x"}).apply(
        _mroas_gradient, subset=["Marginal ROAS"]), width='stretch', hide_index=True)

    st.markdown("##### Diminishing-returns curve")
    seg_pick = st.selectbox("Segment", list(seg_models.keys()),
                            format_func=lambda k: f"{split_segment_key(k)[0].title()} · {split_segment_key(k)[1]}")
    fit = model.elasticity_fits.get(seg_pick)
    if fit is not None and np.isfinite(fit.scale):
        cur = float(base_daily[seg_pick][:sim_h].mean())
        grid = np.linspace(max(cur * 0.2, 1.0), cur * 3.0 + 1.0, 40)
        curve = response_curve(fit, grid)
        import plotly.graph_objects as go
        figr = go.Figure()
        figr.add_trace(go.Scatter(x=curve["spend"], y=curve["revenue"], name="Expected daily revenue",
                                  line=dict(color="#4285F4", width=3)))
        figr.add_vline(x=cur, line_dash="dash", line_color="#aaa",
                       annotation_text="current spend")
        figr.update_layout(height=300, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title="Daily spend ($)", yaxis_title="Expected daily revenue ($)")
        st.plotly_chart(figr, width='stretch')
        st.caption(f"Elasticity β = {fit.beta:.2f}  →  a 10% spend increase yields "
                   f"≈ {((1.10**fit.beta)-1)*100:.1f}% more revenue (diminishing returns).")
    else:
        st.info("Not enough spend/revenue variation to fit a response curve for this segment.")

# --------------------------------------------------------------------------- #
# TAB 3 — AI Insights
# --------------------------------------------------------------------------- #
with tab_ai:
    st.markdown("### AI-assisted forecast briefing")
    use_llm = st.toggle("Use LLM to polish the briefing (needs an API key in the environment)",
                        value=False,
                        help=None if _llm_provider else
                        "No API key detected — the toggle will fall back to the offline "
                        "narrative. Add a key to .env to enable this.")

    # Regenerate whenever anything the briefing depends on changes — the toggle,
    # the data source, or the simulation count. Caching on "brief" alone meant
    # flipping the toggle silently re-displayed the previous briefing, which read
    # as "the LLM button does nothing".
    brief_sig = (sig, use_llm)
    if st.button("Generate briefing", type="primary") or \
            st.session_state.get("brief_sig") != brief_sig:
        with st.spinner("Analysing drivers and composing briefing..."):
            brief = generate_briefing(result, prepared, model, use_llm=use_llm)
            st.session_state["brief"] = brief
            st.session_state["brief_sig"] = brief_sig
    brief = st.session_state["brief"]

    badge = ("🟢 LLM-polished (" + str(brief.provider) + ")") if brief.provider != "offline" \
        else "⚪ Offline deterministic narrative"
    st.caption(f"Source: {badge}")

    # The LLM layer never raises — it just falls back. That is right for robustness
    # but leaves no clue *why*, so say it out loud when the user explicitly asked
    # for LLM polish and did not get it.
    if use_llm and brief.provider == "offline":
        if _llm_provider is None:
            st.warning(
                "**LLM polish requested, but no provider is available** — showing the "
                "offline narrative instead. No API key was found (or its SDK is not "
                "installed). Fix: `cp .env.example .env`, add e.g. `GEMINI_API_KEY=...`, "
                "then **restart Streamlit** (a running server does not pick up a new "
                "`.env` or newly installed packages).")
        else:
            st.warning(
                f"**LLM polish requested and `{_llm_provider}` was detected, but the call "
                "did not return text** — showing the offline narrative instead. Usual "
                "causes: an invalid/expired key, exhausted quota, a timeout, or no "
                "network. The terminal running Streamlit logs the exact reason.")

    left, right = st.columns([3, 2])
    with left:
        st.markdown(brief.markdown)
    with right:
        st.markdown("#### Change attribution")
        hs = brief.report.horizon_summary[horizons[0]]
        attr = hs["attribution"]
        import plotly.graph_objects as go
        names = ["Seasonality", "Trend", "Spend & mix"]
        vals = [attr.get("seasonality_pct", 0) or 0, attr.get("trend_pct", 0) or 0,
                attr.get("spend_and_mix_pct", 0) or 0]
        figw = go.Figure(go.Waterfall(
            orientation="v", measure=["relative", "relative", "relative"],
            x=names, y=[v * 100 for v in vals],
            connector={"line": {"color": "#555"}}))
        figw.update_layout(height=280, template="plotly_dark",
                           margin=dict(l=10, r=10, t=10, b=10),
                           yaxis_title="Contribution to change (%)")
        st.plotly_chart(figw, width='stretch')
        st.caption(f"Total {horizons[0]}d change vs trailing: "
                   f"{pct(hs['revenue_change_pct'])}")

# --------------------------------------------------------------------------- #
# TAB 4 — Data & Predictions
# --------------------------------------------------------------------------- #
with tab_data:
    st.markdown("### Predictions (submission format)")
    st.caption("The tidy long output written to output/predictions.csv — every horizon × "
               "hierarchy node × metric, with a full quantile fan.")
    st.dataframe(result.predictions, width='stretch', hide_index=True, height=380)
    st.download_button("⬇ Download predictions.csv",
                       result.predictions.to_csv(index=False).encode(),
                       "predictions.csv", "text/csv")

    if prepared.notes:
        st.markdown("#### Data quality notes")
        for n in prepared.notes:
            st.markdown(f"- {n}")

    st.markdown("#### Ingested data preview")
    st.dataframe(panel.head(200), width='stretch', hide_index=True, height=280)
