"""Confidence radii, certification decisions, safe selection, and theory budgets.

Formulas follow the paper:
  geometric radius (Prop. 2, Sec. 4):  b_j = 4 B Lam d_j sqrt(log(2K/delta) / n)
  empirical Bernstein (App. A.3):      b_j = sqrt(2 v_j log(4K/delta) / n) + 7 B^2 log(4K/delta) / (3 (n - 1))
Here B is the width of the declared output interval I, so rows lie in [-B^2/2, B^2/2].
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def geom_radius(B, Lam, d, n, K, delta):
    return 4.0 * B * Lam * np.asarray(d) * np.sqrt(np.log(2.0 * K / delta) / np.asarray(n))


def eb_radius(var, n, B, K, delta):
    n = np.asarray(n, dtype=float)
    L = np.log(4.0 * K / delta)
    return np.sqrt(2.0 * np.maximum(var, 0.0) * L / n) + 7.0 * B ** 2 * L / (3.0 * (n - 1.0))


def certify(mean, radius):
    """Certify a positive gain iff the lower confidence bound is positive."""
    return (np.asarray(mean) - np.asarray(radius)) > 0


def select(mean, radius):
    """Safe selection (Sec. 4).  Returns 0 for the baseline, else the 1-based candidate index.

    mean, radius: arrays (..., K).  Selects the largest positive L_j = mean_j - radius_j.
    """
    L = np.asarray(mean) - np.asarray(radius)
    best = np.argmax(L, axis=-1)
    return np.where(np.max(L, axis=-1) > 0, best + 1, 0)


# ---------------- theory budgets (rows) ----------------
def geom_sufficient_rows(B, Lam, d, gain, delta, beta):
    """Sufficient budget from App. A.4: 64 B^2 Lam^2 d^2 / gain^2 * log(2 / min(delta, beta))."""
    return 64.0 * B ** 2 * Lam ** 2 * d ** 2 / gain ** 2 * np.log(2.0 / min(delta, beta))


def predicted_rows(method, gain, var, B, d=None, Lam=None, delta=0.05, power=0.9, K=1):
    """Normal-approximation budget at which a method reaches the target power.

    Used only to size simulation grids; reported budgets come from simulation.
    """
    if gain <= 0:
        return np.inf
    z = norm.ppf(power)
    if method == "geometric":
        a = 4.0 * B * Lam * d * np.sqrt(np.log(2.0 * K / delta)) + z * np.sqrt(var)
        return float((a / gain) ** 2)

    def excess(n):
        return gain - eb_radius(var, n, B, K, delta) - z * np.sqrt(var / n)

    lo, hi = 3.0, 10.0
    while excess(hi) <= 0:
        hi *= 2.0
        if hi > 1e16:
            return np.inf
    return float(brentq(excess, lo, hi)) if excess(lo) < 0 else lo


def budget_grid(n_max: float, n_min: int = 64, per_octave: int = 4):
    """Quarter-octave grid of row budgets from n_min to at least n_max."""
    k_max = int(np.ceil(per_octave * np.log2(max(n_max, n_min) / n_min)))
    g = np.unique(np.round(n_min * 2.0 ** (np.arange(k_max + 1) / per_octave)).astype(np.int64))
    return g
