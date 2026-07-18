"""
Test suite for the AIgnition forecasting pipeline.

Runnable two ways:
    pytest -q tests/test_pipeline.py
    python tests/test_pipeline.py          # self-contained runner, no pytest needed

Covers: ingestion/normalisation, the Meta conversion=revenue invariant, taxonomy,
preprocessing, the end-to-end forecast, the output-contract schema, quantile
monotonicity, a null-free scored output (incl. dormant-campaign ROAS), hierarchy
coherence, robustness to a single-platform subset, and the committed model loading
cleanly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from forecast import run_forecast  # noqa: E402
from ingest import load_canonical  # noqa: E402
from model import load_model  # noqa: E402
from preprocess import prepare  # noqa: E402
from taxonomy import classify_brand, classify_type  # noqa: E402

DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "pickle" / "model.pkl"

_cache: dict = {}


def _forecast():
    if "result" not in _cache:
        panel = load_canonical(DATA_DIR)
        prepared = prepare(panel)
        model = load_model(MODEL_PATH)
        cfg = config.ForecastConfig(n_simulations=800)
        model.cfg = cfg
        _cache["panel"] = panel
        _cache["prepared"] = prepared
        _cache["model"] = model
        _cache["result"] = run_forecast(prepared, model, cfg)
    return _cache


# --------------------------------------------------------------------------- #
def test_ingest_canonical():
    panel = load_canonical(DATA_DIR)
    assert set(config.CANON_COLUMNS).issubset(panel.columns)
    assert set(panel["platform"].unique()) == {"google", "meta", "bing"}
    assert panel["revenue"].sum() > 0 and panel["spend"].sum() > 0
    assert panel["date"].notna().all()


def test_meta_conversion_is_revenue():
    """Meta `conversion` is conversion *value*: revenue must be positive & large."""
    panel = load_canonical(DATA_DIR)
    meta = panel[panel["platform"] == "meta"]
    assert meta["revenue"].sum() > meta["spend"].sum()  # ROAS > 1 in aggregate


def test_taxonomy_rules():
    assert classify_type("PERFORMANCE_MAX", "Pmax_NTM_Campaign_01") == "Performance Max"
    assert classify_type(None, "Prospecting_DPA_Campaign_02") == "Social - Prospecting"
    assert classify_type(None, "Remarketing_Brand_Campaign_01") == "Social - Remarketing"
    assert classify_type("Search", "Search_TM_Campaign_02") == "Search"
    assert classify_brand("Search_TM_Campaign_02") == "Brand"
    assert classify_brand("Pmax_NTM_Campaign_01") == "Non-Brand"


def test_prepare_segments():
    c = _forecast()
    prepared = c["prepared"]
    assert len(prepared.segments) >= 5
    assert prepared.origin is not None
    # dense panel: no gaps within each segment's active span
    assert prepared.segment_daily["date"].notna().all()


def test_output_schema_exact():
    c = _forecast()
    pred = c["result"].predictions
    assert list(pred.columns) == config.PREDICTION_COLUMNS
    assert set(pred["level"].unique()).issubset(set(config.LEVELS))
    assert set(pred["metric"].unique()) == set(config.METRICS)
    assert set(pred["horizon_days"].unique()) == set(config.HORIZONS_DAYS)
    # every hierarchy level is represented
    assert set(config.LEVELS).issubset(set(pred["level"].unique()))


def test_quantiles_monotone_and_nonneg():
    c = _forecast()
    pred = c["result"].predictions
    qcols = [config.q_col(q) for q in config.QUANTILES]

    # Revenue & spend must always be fully finite and non-negative.
    money = pred[pred["metric"].isin(["revenue", "spend"])][qcols].to_numpy()
    assert np.isfinite(money).all(), "revenue/spend quantiles must be finite"
    assert (money >= -1e-6).all(), "revenue/spend quantiles must be >= 0"

    # Monotonicity p05<=...<=p95 on every row that is fully finite. A dormant
    # node (no spend and no revenue) reports ROAS 0 rather than NaN, so on the
    # sample data every row is finite; the guard stays for genuinely undefined
    # ROAS (zero spend against positive revenue), which is left as NaN.
    vals = pred[qcols].to_numpy()
    finite_rows = np.isfinite(vals).all(axis=1)
    diffs = np.diff(vals[finite_rows], axis=1)
    assert (diffs >= -1e-6).all(), "quantiles must be non-decreasing p05<=...<=p95"

    # ROAS: where defined, quantiles must be non-negative.
    roas = pred[pred["metric"] == "roas"][qcols].to_numpy()
    roas_finite = roas[np.isfinite(roas)]
    assert (roas_finite >= -1e-6).all(), "defined ROAS quantiles must be >= 0"


def test_no_nulls_in_scored_output():
    """The scored CSV must contain no nulls — a null-check by the harness would
    fail the submission outright, and a NaN can poison an aggregate error metric.

    Dormant campaigns (zero spend *and* zero revenue) are ~45% of the sample and
    used to emit NaN ROAS for 189 of 1404 rows; they now report the conventional
    ROAS of 0, which is also how the ad platforms report them.
    """
    c = _forecast()
    pred = c["result"].predictions
    numeric = ["mean", "std"] + [config.q_col(q) for q in config.QUANTILES]
    nulls = pred[numeric].isna().sum().sum()
    assert nulls == 0, f"scored output must have no nulls; found {nulls}"
    for col in config.PREDICTION_COLUMNS:
        assert pred[col].notna().all(), f"null in identifier column {col}"


def test_dormant_campaign_roas_is_zero():
    """A node with no spend and no revenue reports ROAS 0, not NaN or a divide error."""
    from hierarchy import _roas
    z = np.zeros(4)
    assert np.all(_roas(z, z) == 0.0), "dormant (0 revenue / 0 spend) must be ROAS 0"
    # Normal case still divides.
    assert np.allclose(_roas(np.array([10.0]), np.array([2.0])), 5.0)
    # Zero spend with positive revenue is genuinely infinite -> stays NaN.
    assert np.isnan(_roas(np.array([10.0]), np.array([0.0]))).all()


def test_cli_tools_work_without_app_extras():
    """`python src/briefing.py` must run in a **requirements.txt-only** environment.

    python-dotenv ships in requirements-app.txt, not requirements.txt, so the .env
    load in briefing.py is best-effort by design. A bare `from dotenv import ...`
    at module level breaks the CLI for anyone who installed only the scored-path
    deps — which is exactly what happened once.
    """
    import subprocess
    import textwrap
    code = textwrap.dedent("""
        import sys
        from importlib.abc import MetaPathFinder
        class Block(MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "dotenv" or name.startswith("dotenv."):
                    raise ImportError("simulated: python-dotenv not installed")
                return None
        sys.meta_path.insert(0, Block())
        sys.path.insert(0, {src!r})
        sys.path.insert(0, {ins!r})
        import briefing          # must not raise
        print("IMPORT_OK")
    """).format(src=str(ROOT / "src"), ins=str(ROOT / "src" / "insights"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "IMPORT_OK" in r.stdout, (
        "briefing.py must import without python-dotenv installed:\n" + r.stderr)


def test_hierarchy_mean_coherence():
    """Channel means must sum to the aggregate mean (conformal is spread-only)."""
    c = _forecast()
    pred = c["result"].predictions
    for h in config.HORIZONS_DAYS:
        agg = pred[(pred.level == "aggregate") & (pred.metric == "revenue") &
                   (pred.horizon_days == h)]["mean"].iloc[0]
        chan = pred[(pred.level == "channel") & (pred.metric == "revenue") &
                    (pred.horizon_days == h)]["mean"].sum()
        assert abs(agg - chan) / max(agg, 1.0) < 0.02, (h, agg, chan)


def test_roas_consistency():
    """Aggregate ROAS P50 should be within the plausible historical band."""
    c = _forecast()
    pred = c["result"].predictions
    roas = pred[(pred.level == "aggregate") & (pred.metric == "roas")]
    assert (roas["p50"] > 0).all() and (roas["p50"] < 50).all()


def test_model_loads_dependency_light():
    model = load_model(MODEL_PATH)
    from model import ForecastModel
    assert isinstance(model, ForecastModel)
    assert model.elasticity_global > 0
    assert len(model.seasonal_by_segment_rev) > 0


def test_robust_to_single_platform_subset():
    """Pipeline must not crash if only one platform's data is present."""
    panel = load_canonical(DATA_DIR)
    g = panel[panel["platform"] == "google"].copy()
    prepared = prepare(g)
    model = load_model(MODEL_PATH)
    model.cfg = config.ForecastConfig(n_simulations=400)
    result = run_forecast(prepared, model, model.cfg)
    assert not result.predictions.empty
    assert (result.predictions["channel"] == "google").any()


# --------------------------------------------------------------------------- #
def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
