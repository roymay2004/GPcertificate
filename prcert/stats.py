"""Monte Carlo summaries: binomial intervals, budget selection, and slope fits."""
from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist


def clopper_pearson(k, n, alpha: float = 0.05):
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    lo = np.where(k > 0, beta_dist.ppf(alpha / 2, k, n - k + 1), 0.0)
    hi = np.where(k < n, beta_dist.ppf(1 - alpha / 2, k + 1, n - k), 1.0)
    return lo, hi


def isotonic_increasing(y, w=None):
    """Pool-adjacent-violators fit of a nondecreasing sequence."""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    vals, wts, sizes = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi)
        wts.append(wi)
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wts[-2] += wts[-1]
            sizes[-2] += sizes[-1]
            vals[-2] = v
            vals.pop(); wts.pop(); sizes.pop()
    return np.repeat(vals, sizes)


def select_budget(grid, power, target: float = 0.9):
    """Smallest budget whose isotonic power estimate reaches the target (batch A).

    Returns (index, budget) or (None, None) if the target is not reached on the grid.
    """
    iso = isotonic_increasing(power)
    hit = np.nonzero(iso >= target)[0]
    if hit.size == 0:
        return None, None
    i = int(hit[0])
    return i, int(np.asarray(grid)[i])


def loglog_slope(x, y):
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    A = np.column_stack([np.ones_like(lx), lx])
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    return float(coef[1])


def bootstrap_slope(h_values, decisions_by_h, grid_by_h, calls_per_row, target=0.9,
                    n_boot: int = 1000, seed: int = 0):
    """Percentile CI for the log-log slope of the power-attaining cost versus h.

    decisions_by_h[i] is a boolean array (reps, G) from batch A.  Repetitions are
    resampled within each h, the budget is reselected, and the slope is refitted.
    """
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_boot):
        hs, cs = [], []
        for h, dec, grid in zip(h_values, decisions_by_h, grid_by_h):
            idx = rng.integers(0, dec.shape[0], dec.shape[0])
            _, nb = select_budget(grid, dec[idx].mean(axis=0), target)
            if nb is not None:
                hs.append(h)
                cs.append(nb * calls_per_row)
        if len(hs) >= 3:
            slopes.append(loglog_slope(hs, cs))
    if not slopes:
        return np.nan, np.nan
    return tuple(np.percentile(slopes, [2.5, 97.5]))
