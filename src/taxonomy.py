"""
Campaign taxonomy: turn heterogeneous, agency-tagged campaign structures into a
consistent, marketing-meaningful hierarchy.

Real agency data is fragmented: Google exposes a native
``campaign_advertising_channel_type``, Bing a ``CampaignType``, and Meta exposes
*nothing* — the format is encoded only in the campaign name. On top of the
format, agencies steer **brand vs non-brand** spend very differently (brand
trademark terms convert cheaply; non-brand prospecting is where growth and risk
live). We therefore classify every campaign along two orthogonal dimensions:

* ``campaign_type``  — normalised format (Search, Shopping, Performance Max, ...)
* ``brand_intent``   — Brand | Non-Brand | Unknown  (TM vs NTM, Brand vs Generic)

These power the "campaign-type" forecasting level and give the AI layer real
business vocabulary to reason with.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

import config
from logging_utils import get_logger

log = get_logger("taxonomy")

# Pre-compile name rules once.
_NAME_TYPE_RULES = [(re.compile(pat, re.IGNORECASE), lab) for pat, lab in config.NAME_TYPE_RULES]
_BRAND_RE = re.compile("|".join(config.BRAND_TOKENS), re.IGNORECASE)
_NONBRAND_RE = re.compile("|".join(config.NONBRAND_TOKENS), re.IGNORECASE)


def classify_type(channel_type_raw: Optional[str], campaign_name: Optional[str]) -> str:
    """Map a campaign to a canonical :data:`config.CAMPAIGN_TYPES` value.

    Precedence: native platform type (most reliable) → campaign-name tokens →
    ``"Other"``. Name tokens can still *refine* an ambiguous native type, but the
    native type wins when present and recognised.
    """
    # 1. Native type string, if recognised.
    if channel_type_raw is not None and not _is_missing(channel_type_raw):
        key = str(channel_type_raw).strip()
        if key in config.NATIVE_TYPE_MAP:
            return config.NATIVE_TYPE_MAP[key]
        # normalise case/spacing variants (e.g. "performance_max")
        norm = key.upper().replace(" ", "_")
        if norm in config.NATIVE_TYPE_MAP:
            return config.NATIVE_TYPE_MAP[norm]

    # 2. Campaign-name token rules.
    name = "" if _is_missing(campaign_name) else str(campaign_name)
    for rex, label in _NAME_TYPE_RULES:
        if rex.search(name):
            return label

    return "Other"


def classify_brand(campaign_name: Optional[str], channel_type_raw: Optional[str] = None) -> str:
    """Return ``"Brand"`` / ``"Non-Brand"`` / ``"Unknown"`` for a campaign.

    Non-brand tokens (NTM, prospecting, generic) are checked first because they
    are the more explicit signal in this dataset; brand tokens (TM, brand) next.
    """
    name = "" if _is_missing(campaign_name) else str(campaign_name)
    if _NONBRAND_RE.search(name):
        return "Non-Brand"
    if _BRAND_RE.search(name):
        return "Brand"
    return "Unknown"


def add_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``campaign_type`` and ``brand_intent`` columns to a canonical frame.

    Vectorised over unique (channel_type_raw, campaign_name) pairs so this stays
    fast even on large panels.
    """
    out = df.copy()
    # Normalise to plain python (None | str) so build and lookup keys hash
    # identically — pandas may otherwise store missing as NaN / pd.NA.
    ctr_norm = [None if _is_missing(x) else str(x) for x in out["channel_type_raw"]]
    name_norm = ["" if _is_missing(x) else str(x) for x in out["campaign_name"]]
    keys = list(zip(ctr_norm, name_norm))
    uniq = set(keys)
    type_map = {k: classify_type(k[0], k[1]) for k in uniq}
    brand_map = {k: classify_brand(k[1], k[0]) for k in uniq}

    out["campaign_type"] = [type_map[k] for k in keys]
    out["brand_intent"] = [brand_map[k] for k in keys]

    n_other = int((out["campaign_type"] == "Other").sum())
    if n_other:
        log.info("taxonomy: %d rows classified as 'Other' campaign type", n_other)
    return out


def segment_key(platform: str, campaign_type: str) -> str:
    """Stable key for the modelling segment (channel × campaign type)."""
    return f"{platform}::{campaign_type}"


def split_segment_key(key: str) -> tuple[str, str]:
    platform, ctype = key.split("::", 1)
    return platform, ctype


def _is_missing(x) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False
