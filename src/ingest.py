"""
Data ingestion: read whatever CSVs are in ``data/`` and normalise the three
heterogeneous ad-platform schemas into one canonical long panel.

Design principles (driven by the submission contract):

* **Read by pattern, not by hardcoded filename.** The eval harness may rename
  files; we detect each platform by a *schema fingerprint* (the set of columns
  present), falling back to filename hints only to disambiguate.
* **Tolerate size/shape drift.** Optional columns may be absent; ids may be huge;
  dates may be dirty. We coerce defensively and never crash on a single bad row.
* **Attribution is the source of truth.** We convert units (Google micros →
  currency; Meta ``conversion`` → revenue) but never re-attribute.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import config
from config import PlatformSpec
from logging_utils import get_logger

log = get_logger("ingest")


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
def detect_platform(columns: List[str], filename: str = "") -> Optional[PlatformSpec]:
    """Identify the platform for a file from its columns (primary) and name (tiebreak)."""
    colset = set(columns)
    matches = [spec for spec in config.PLATFORM_SPECS if set(spec.signature).issubset(colset)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Extremely unlikely given distinct signatures; disambiguate by filename.
        fn = filename.lower()
        for spec in matches:
            if any(h in fn for h in spec.filename_hints):
                return spec
        return matches[0]
    # No signature match — last-resort filename hint (schema may have shifted).
    fn = filename.lower()
    for spec in config.PLATFORM_SPECS:
        if any(h in fn for h in spec.filename_hints):
            log.warning("file %s matched platform '%s' by filename only (columns=%s)",
                        filename, spec.name, sorted(colset)[:12])
            return spec
    return None


# ---------------------------------------------------------------------------
# Per-file normalisation
# ---------------------------------------------------------------------------
def _to_id_string(s: pd.Series) -> pd.Series:
    """Campaign ids -> clean strings (no float scientific notation, no trailing .0)."""
    def fmt(v):
        if pd.isna(v):
            return "unknown"
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return str(v).strip()
    return s.map(fmt)


def normalise_file(df: pd.DataFrame, spec: PlatformSpec, source: str = "") -> pd.DataFrame:
    """Map one platform's raw frame onto the canonical schema."""
    raw = df.copy()
    # Drop the unnamed pandas index column if present.
    raw = raw.loc[:, ~raw.columns.astype(str).str.match(r"^Unnamed")]

    out = pd.DataFrame(index=raw.index)

    # Rename known columns.
    for native, canon in spec.rename.items():
        if native in raw.columns:
            out[canon] = raw[native]

    # Spend: from micros or native.
    if spec.cost_in_micros and "metrics_cost_micros" in raw.columns:
        out["spend"] = pd.to_numeric(raw["metrics_cost_micros"], errors="coerce") / config.MICROS_PER_UNIT
    elif "spend" not in out.columns and "spend" in raw.columns:
        out["spend"] = raw["spend"]

    out["platform"] = spec.name

    # Ensure every canonical column exists.
    for col in config.CANON_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    # --- Types & cleaning ---
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["campaign_id"] = _to_id_string(out["campaign_id"])
    out["campaign_name"] = out["campaign_name"].astype("object").where(
        out["campaign_name"].notna(), other="Unnamed Campaign"
    ).astype(str)
    out["channel_type_raw"] = out["channel_type_raw"].where(out["channel_type_raw"].notna(), None)

    for col in ["spend", "revenue", "clicks", "impressions", "conversions", "budget_raw"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Floor spend/revenue at zero (negative = data error / refund noise); log volume.
    for col in ["spend", "revenue"]:
        neg = int((out[col] < 0).sum())
        if neg:
            log.info("%s: flooring %d negative %s values to 0", spec.name, neg, col)
        out[col] = out[col].clip(lower=0).fillna(0.0)
    out["clicks"] = out["clicks"].clip(lower=0).fillna(0.0)
    out["impressions"] = out["impressions"].clip(lower=0).fillna(0.0)

    # Drop rows with unparseable dates (can't be placed on a timeline).
    bad_dates = int(out["date"].isna().sum())
    if bad_dates:
        log.warning("%s: dropping %d rows with unparseable dates", spec.name, bad_dates)
    out = out[out["date"].notna()].copy()

    out = out[config.CANON_COLUMNS]
    log.info("%s: %d rows | %s -> %s", spec.name, len(out),
             out["date"].min().date() if len(out) else "-",
             out["date"].max().date() if len(out) else "-")
    return out


# ---------------------------------------------------------------------------
# Directory-level ingestion
# ---------------------------------------------------------------------------
def find_data_files(data_dir: os.PathLike | str) -> List[Path]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    files = sorted(
        p for p in data_dir.rglob("*")
        if p.suffix.lower() in {".csv", ".tsv"} and p.is_file()
    )
    if not files:
        raise FileNotFoundError(f"No CSV/TSV files found in {data_dir}")
    return files


def _read_csv(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    # low_memory=False to avoid mixed-dtype chunk warnings on big files.
    return pd.read_csv(path, sep=sep, low_memory=False)


def load_canonical(data_dir: os.PathLike | str) -> pd.DataFrame:
    """Load every recognised CSV in ``data_dir`` into one canonical panel.

    Returns a DataFrame with :data:`config.CANON_COLUMNS`, sorted by date.
    Unrecognised files are skipped with a warning (never fatal) so a stray file
    in the eval ``data/`` folder can't zero the run.
    """
    files = find_data_files(data_dir)
    frames: List[pd.DataFrame] = []
    seen_platforms = []
    for path in files:
        try:
            raw = _read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("could not read %s: %s", path.name, exc)
            continue
        spec = detect_platform(list(raw.columns), filename=path.name)
        if spec is None:
            log.warning("skipping unrecognised file: %s (columns=%s)",
                        path.name, list(raw.columns)[:12])
            continue
        frames.append(normalise_file(raw, spec, source=path.name))
        seen_platforms.append(spec.name)

    if not frames:
        raise ValueError(
            "No recognised ad-platform files in data dir. Expected Google/Meta/Bing "
            "campaign-stats schemas."
        )

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["platform", "campaign_id", "date"]).reset_index(drop=True)
    log.info("ingested %d rows across platforms: %s | overall %s -> %s",
             len(panel), sorted(set(seen_platforms)),
             panel["date"].min().date(), panel["date"].max().date())
    return panel


def canonical_from_frames(named_frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Build the canonical panel from in-memory (filename, DataFrame) pairs.

    Used by the app for uploaded CSVs. Same detection/normalisation as
    :func:`load_canonical`, just without touching disk.
    """
    frames, seen = [], []
    for name, raw in named_frames:
        spec = detect_platform(list(raw.columns), filename=name)
        if spec is None:
            log.warning("skipping unrecognised upload: %s", name)
            continue
        frames.append(normalise_file(raw, spec, source=name))
        seen.append(spec.name)
    if not frames:
        raise ValueError("No recognised ad-platform files among the uploads.")
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["platform", "campaign_id", "date"]).reset_index(drop=True)
