"""Coupled and independent audit rows, streamed into statistics at budget checkpoints.

Exact reduction used for speed.  Every quantity an audit observes depends on X and Xi
only through their projections onto S = span(rows of A, U0, U1, ..., UK).  With an
orthonormal basis Q of S, Q^T X and Q^T Xi are i.i.d. N(0, I_q), and

    Q^T X_P = Q^T Xi + (Q^T U)(Q^T U)^T (Q^T X - Q^T Xi),   f(X_P) = sum c sin((Q^T X_P)^T (A Q)^T).

Sampling q-dimensional normals therefore reproduces the D-dimensional audit exactly
in distribution, so added inactive coordinates cost nothing.

Simulator calls are counted as the paper does: a coupled row with K candidates costs
K + 2 calls (f(X), f(X_P0), f(X_Pj)); an independent-loss row costs 4 calls.
"""
from __future__ import annotations

import numpy as np

from .geometry import orth
from .simulators import SineSum


class _Reduced:
    def __init__(self, sim: SineSum, bases):
        Q = orth(np.hstack([sim.A.T] + list(bases)))
        self.q = Q.shape[1]
        self.Aq = sim.A @ Q
        self.c = sim.c
        self.bases_q = [Q.T @ U for U in bases]

    def f(self, Z: np.ndarray) -> np.ndarray:
        return np.sin(Z @ self.Aq.T) @ self.c

    def f_proj(self, x, xi, v, Uq) -> np.ndarray:
        return self.f(xi + (v @ Uq) @ Uq.T)


class CoupledSampler:
    """Rows W_j = Z_{P0} - Z_{Pj}, j = 1..K, sharing (X, Xi)."""

    def __init__(self, sim: SineSum, U0: np.ndarray, Us):
        self.red = _Reduced(sim, [U0] + list(Us))
        self.K = len(Us)
        self.calls_per_row = self.K + 2

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        red = self.red
        x = rng.standard_normal((n, red.q))
        xi = rng.standard_normal((n, red.q))
        v = x - xi
        fx = red.f(x)
        z0 = 0.5 * (fx - red.f_proj(x, xi, v, red.bases_q[0])) ** 2
        W = np.empty((n, self.K))
        for j in range(self.K):
            W[:, j] = z0 - 0.5 * (fx - red.f_proj(x, xi, v, red.bases_q[j + 1])) ** 2
        return W


class IndepSampler:
    """Rows Z_{P0}(pair 1) - Z_{P1}(pair 2) from two independent conditional pairs."""

    def __init__(self, sim: SineSum, U0: np.ndarray, U1: np.ndarray):
        self.red = _Reduced(sim, [U0, U1])
        self.K = 1
        self.calls_per_row = 4

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        red = self.red
        out = []
        for Uq in red.bases_q:
            x = rng.standard_normal((n, red.q))
            xi = rng.standard_normal((n, red.q))
            out.append(0.5 * (red.f(x) - red.f_proj(x, xi, x - xi, Uq)) ** 2)
        return (out[0] - out[1])[:, None]


def run_checkpoints(sampler, grid, reps: int, rng: np.random.Generator,
                    chunk_rows: int = 2 ** 20):
    """Stream `reps` independent audits and record statistics at each budget in `grid`.

    For every repetition, the statistics at grid[g] are computed from that repetition's
    first grid[g] rows.  Each budget is therefore a valid fixed-size audit; decisions at
    different budgets are never combined into a stopping rule.

    Returns mean and unbiased variance arrays of shape (reps, len(grid), K).
    """
    grid = np.asarray(sorted(set(int(n) for n in grid)), dtype=np.int64)
    if grid[0] < 2:
        raise ValueError("budgets must be at least 2 rows")
    K = sampler.K
    G = grid.size
    out_mean = np.empty((reps, G, K))
    out_var = np.empty((reps, G, K))
    cnt = np.zeros(reps)
    mean = np.zeros((reps, K))
    m2 = np.zeros((reps, K))
    prev = 0
    for g, ng in enumerate(grid):
        L = int(ng - prev)
        block = max(1, chunk_rows // L)
        piece = min(L, chunk_rows)
        for r0 in range(0, reps, block):
            r1 = min(reps, r0 + block)
            b = r1 - r0
            done = 0
            while done < L:
                p = min(piece, L - done)
                W = sampler.draw(rng, b * p).reshape(b, p, K)
                sm = W.mean(axis=1)
                sm2 = ((W - sm[:, None, :]) ** 2).sum(axis=1)
                na = cnt[r0:r1][:, None]
                delta = sm - mean[r0:r1]
                tot = na + p
                mean[r0:r1] += delta * (p / tot)
                m2[r0:r1] += sm2 + delta ** 2 * (na * p / tot)
                cnt[r0:r1] += p
                done += p
        out_mean[:, g, :] = mean
        out_var[:, g, :] = m2 / (cnt[:, None] - 1.0)
        prev = int(ng)
    return grid, out_mean, out_var
