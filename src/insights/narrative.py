"""
Offline narrative generator.

Turns a :class:`drivers.DriverReport` into an executive-ready markdown briefing
using deterministic templates — no network, no API key. This guarantees the
product always ships high-quality, grounded insights; the optional LLM layer
(``insights.llm``) then *polishes* this same material into more fluent prose when
a key is available. Because both consume the identical structured facts, the
offline and LLM narratives never disagree on the numbers.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from drivers import DriverReport  # type: ignore  # (src on path at runtime)


def _money(x: float, cur: str = "USD") -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    sym = "$" if cur == "USD" else ""
    return f"{sym}{x:,.0f}"


def _pct(x: float, signed: bool = True) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return (f"{x*100:+.0f}%" if signed else f"{x*100:.0f}%")


def _roas(x: float) -> str:
    return "n/a" if (x is None or not np.isfinite(x)) else f"{x:.2f}x"


def _dominant_driver(attr: dict) -> str:
    labels = {"seasonality_pct": "seasonality (marketing calendar)",
              "trend_pct": "the underlying year-over-year trend",
              "spend_and_mix_pct": "planned spend and channel mix"}
    finite = {k: v for k, v in attr.items() if v is not None and np.isfinite(v)}
    if not finite:
        return "a mix of factors"
    k = max(finite, key=lambda k: abs(finite[k]))
    return labels.get(k, k)


def build_sections(report: DriverReport) -> Dict[str, str]:
    cur = report.currency
    sections: Dict[str, str] = {}

    # ---- Executive summary (uses the primary/shortest horizon as the headline) ----
    h0 = report.horizons[0]
    hs = report.horizon_summary[h0]
    change = hs["revenue_change_pct"]
    dom = _dominant_driver(hs["attribution"])
    ev = hs.get("events", [])
    ev_txt = f" The window includes {', '.join(ev)}." if ev else ""
    lines = [
        f"Over the next **{h0} days** (from {report.origin}), blended ecommerce "
        f"revenue is projected at **{_money(hs['forecast_revenue_p50'], cur)}** "
        f"(80% interval {_money(hs['forecast_revenue_p10'], cur)}–"
        f"{_money(hs['forecast_revenue_p90'], cur)}) at a blended ROAS of "
        f"**{_roas(hs['forecast_roas_p50'])}**.",
        f"That is **{_pct(change)}** versus the trailing {h0} days "
        f"({_money(hs['trailing_revenue'], cur)}), driven primarily by {dom}.{ev_txt}",
    ]
    # multi-horizon glance
    glance = []
    for h in report.horizons:
        s = report.horizon_summary[h]
        glance.append(f"- **{h}d:** {_money(s['forecast_revenue_p50'], cur)} revenue "
                      f"({_money(s['forecast_revenue_p10'], cur)}–{_money(s['forecast_revenue_p90'], cur)}), "
                      f"ROAS {_roas(s['forecast_roas_p50'])}")
    sections["executive_summary"] = "\n".join(lines) + "\n\n" + "\n".join(glance)

    # ---- Drivers ----
    d_lines = []
    attr = hs["attribution"]
    driver_labels = [
        ("seasonality_pct", "Seasonality", "the calendar position of the forecast window"),
        ("trend_pct", "Trend", "year-over-year momentum in the underlying business"),
        ("spend_and_mix_pct", "Spend & mix", "planned media spend and shifts between channels/types"),
    ]
    for key, name, desc in driver_labels:
        v = attr.get(key)
        if v is None or not np.isfinite(v):
            continue
        d_lines.append(f"- **{name}: {_pct(v)}** — {desc}.")
    if ev:
        d_lines.append(f"- **Events in window:** {', '.join(ev)} — expect concentrated uplift on those days.")
    sections["drivers"] = ("Approximate attribution of the projected change vs. the "
                           "trailing period:\n\n" + "\n".join(d_lines))

    # ---- Channel outlook ----
    c_lines = []
    for cs in report.channel_summary:
        c_lines.append(
            f"- **{cs['channel'].title()}**: {_money(cs['forecast_revenue_p50'], cur)} "
            f"({_pct(cs['revenue_change_pct'])} vs trailing) at ROAS "
            f"{_roas(cs['forecast_roas_p50'])}.")
    sections["channel_outlook"] = "\n".join(c_lines) if c_lines else "_No channel breakdown available._"

    # ---- Risks ----
    r_lines: List[str] = []
    for w in report.roas_watch:
        r_lines.append(f"- **{w['scope']}**: {w['note']} "
                       f"(forecast {_roas(w['forecast_roas'])}, trailing {_roas(w['trailing_roas'])}).")
    for m in report.movers.get("declining", []):
        r_lines.append(f"- **{m['channel'].title()} / {m['campaign_type']}** is trending down "
                       f"({_pct(m['growth_annual'])} YoY) — monitor for continued softness.")
    widest = max(report.horizon_summary.values(), key=lambda s: s.get("interval_width_pct", 0) or 0)
    if widest.get("interval_width_pct", 0) and widest["interval_width_pct"] > 1.2:
        r_lines.append("- **Wide uncertainty band**: the forecast interval is broad — treat the "
                       "point estimate as indicative and plan against the P10 (downside) case.")
    if not r_lines:
        r_lines.append("- No material risks flagged; ROAS and trend are stable across channels.")
    sections["risks"] = "\n".join(r_lines)

    # ---- Recommendations ----
    rec_lines: List[str] = []
    ups = [o for o in report.opportunities if o["type"] == "scale_up"]
    revs = [o for o in report.opportunities if o["type"] == "review"]
    for o in ups:
        plat, ctype = o["segment"].split("::", 1)
        rec_lines.append(
            f"- **Scale {plat.title()} / {ctype}**: highest incremental efficiency "
            f"(marginal ROAS {_roas(o['marginal_roas'])} vs average {_roas(o['avg_roas'])}) — "
            f"the next dollar works hardest here.")
    for o in revs:
        plat, ctype = o["segment"].split("::", 1)
        rec_lines.append(
            f"- **Review {plat.title()} / {ctype}**: low incremental return "
            f"(marginal ROAS {_roas(o['marginal_roas'])}) — cap or optimise before adding budget.")
    rec_lines.append("- Use the **budget simulator** to size the exact spend change and see the "
                     "revenue/ROAS response with confidence bands before committing.")
    sections["recommendations"] = "\n".join(rec_lines)

    # ---- Data quality ----
    if report.data_quality:
        sections["data_quality"] = "\n".join(f"- {n}" for n in report.data_quality)

    return sections


def to_markdown(report: DriverReport) -> str:
    s = build_sections(report)
    order = [
        ("Executive Summary", "executive_summary"),
        ("What's Driving the Forecast", "drivers"),
        ("Channel Outlook", "channel_outlook"),
        ("Risks to Watch", "risks"),
        ("Recommended Actions", "recommendations"),
        ("Data Quality & Confidence", "data_quality"),
    ]
    out = [f"# Forecast Briefing — origin {report.origin}\n"]
    for title, key in order:
        if key in s and s[key].strip():
            out.append(f"## {title}\n\n{s[key]}\n")
    out.append("\n_Generated by the AIgnition forecasting utility. Figures are model "
               "estimates with quantified uncertainty; not financial advice._")
    return "\n".join(out)
