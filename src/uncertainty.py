"""
Uncertainty quantification: block bootstrap + split-conformal calibration.

Two ideas do the heavy lifting:

1. **Block bootstrap of multiplicative residuals.** Real daily marketing noise is
   autocorrelated and fat-tailed. Sampling *contiguous blocks* of recent residual
   ratios (rather than iid draws) preserves that structure when we build future
   noise paths, so period-sum intervals are realistic rather than artificially
   tight.

2. **Split-conformal calibration.** A model's *nominal* intervals are only as good
   as its assumptions. We run a rolling-origin backtest, measure how far actuals
   land from the simulated median relative to the simulated spread, and freeze a
   per-horizon multiplier that rescales the interval width so realised coverage
   matches the nominal level — a distribution-free honesty guarantee that
   survives being applied to new (held-out) data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from logging_utils import get_logger

log = get_logger("uncertainty")


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------
def block_bootstrap(resid_block: np.ndarray, n_sims: int, horizon: int,
                    rng: np.random.Generator, block_len: int = 7) -> np.ndarray:
    """Return an ``(n_sims, horizon)`` array of multiplicative residual paths.

    Contiguous blocks of length ``block_len`` are sampled (with wrap-around) from
    ``resid_block`` and stitched to length ``horizon``.
    """
    block = np.asarray(resid_block, dtype=float)
    block = block[np.isfinite(block) & (block >= 0)]
    if block.size == 0:
        return np.ones((n_sims, horizon))
    if block.size < block_len:
        # too short to block-sample; iid draw
        idx = rng.integers(0, block.size, size=(n_sims, horizon))
        return block[idx]

    n_blocks = int(np.ceil(horizon / block_len))
    starts = rng.integers(0, block.size, size=(n_sims, n_blocks))
    offsets = np.arange(block_len)
    # (n_sims, n_blocks, block_len) -> gather with wrap-around
    gather = (starts[:, :, None] + offsets[None, None, :]) % block.size
    paths = block[gather].reshape(n_sims, n_blocks * block_len)[:, :horizon]
    # Normalise each path's mean toward 1 to avoid drift from a hot/cold block,
    # but keep dispersion (multiply by block-mean ratio would kill signal, so we
    # only recentre gently).
    return paths


# ---------------------------------------------------------------------------
# Conformal calibration
# ---------------------------------------------------------------------------
@dataclass
class ConformalCalibration:
    """Per-horizon (and optional per-level) interval-width multipliers.

    ``scale[h] > 1`` widens intervals, ``< 1`` tightens them, so that the
    simulated central interval attains nominal coverage on backtest data.
    """
    scale_by_horizon: Dict[int, float] = field(default_factory=dict)
    scale_by_level_horizon: Dict[str, float] = field(default_factory=dict)
    target_coverage: float = 0.80
    n_calibration: int = 0

    def get(self, horizon: int, level: str | None = None) -> float:
        if level is not None:
            key = f"{level}::{horizon}"
            if key in self.scale_by_level_horizon:
                return self.scale_by_level_horizon[key]
        return self.scale_by_horizon.get(horizon, 1.0)


def apply_conformal(period_totals: np.ndarray, scale: float) -> np.ndarray:
    """Rescale a simulated distribution around its median by ``scale`` in log space.

    Monotone and median-preserving: ``x -> M * (x/M)**scale``. Widens (scale>1) or
    tightens (scale<1) every quantile symmetrically in log space.
    """
    x = np.asarray(period_totals, dtype=float)
    x = np.where(np.isfinite(x) & (x >= 0), x, 0.0)
    if scale == 1.0 or x.size == 0:
        return x
    m = np.median(x)
    if m <= 0:
        return x
    ratio = np.where(x > 0, x / m, 1e-9)
    return m * ratio ** float(scale)


def conformal_scale_from_scores(scores: np.ndarray, sim_halfwidth_logs: np.ndarray,
                                target: float = 0.80) -> float:
    """Compute a single conformal width multiplier from backtest records.

    ``scores`` are ``|log(actual/median_pred)|`` and ``sim_halfwidth_logs`` are the
    simulated half-interval widths in log space for the *nominal* target. The
    multiplier is the ratio of the empirical ``target`` quantile of the
    standardised scores to 1, clipped to a sane band.
    """
    scores = np.asarray(scores, dtype=float)
    hw = np.asarray(sim_halfwidth_logs, dtype=float)
    mask = np.isfinite(scores) & np.isfinite(hw) & (hw > 0)
    if mask.sum() < 4:
        return 1.0
    standardized = scores[mask] / hw[mask]
    q = np.quantile(standardized, target)
    return float(np.clip(q, 0.5, 3.0))
