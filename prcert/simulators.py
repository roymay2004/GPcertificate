"""Analytically tractable simulators with exact projection risks.

All simulators are sums of sines, f(x) = sum_j c_j sin(a_j^T x), x in R^D.
For X ~ N(0, I_D) and an orthogonal projector P = U U^T,

    R_f(P) = sum_{i,j} c_i c_j exp(-(|a_i|^2 + |a_j|^2)/2) [sinh(a_i^T a_j) - sinh(a_i^T P a_j)]

(Appendix B of the paper).  The sine ridge f(x) = sin(x_1) and the planar-arm
model are special cases.
"""
from __future__ import annotations

import numpy as np


class SineSum:
    """f(x) = sum_j c_j sin(a_j^T x); A holds the rows a_j (shape m x D)."""

    def __init__(self, c, A, name: str = "sinesum"):
        self.c = np.asarray(c, dtype=float).ravel()
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        if self.A.shape[0] != self.c.size:
            raise ValueError("c and A must have the same number of terms")
        self.name = name
        self.D = self.A.shape[1]
        self._n2 = (self.A ** 2).sum(axis=1)
        self._w = np.exp(-(self._n2[:, None] + self._n2[None, :]) / 2) * np.outer(self.c, self.c)
        self._G = self.A @ self.A.T

    # ----- evaluation -------------------------------------------------
    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.sin(X @ self.A.T) @ self.c

    def grad(self, X: np.ndarray) -> np.ndarray:
        return (np.cos(X @ self.A.T) * self.c) @ self.A

    # ----- certified constants (mathematical bounds, not pilot estimates) -----
    def lipschitz_bound(self) -> float:
        """Global Euclidean Lipschitz bound sum_j |c_j| |a_j|."""
        return float(np.abs(self.c) @ np.sqrt(self._n2))

    def sup_bound(self) -> float:
        """|f| <= sum_j |c_j|, so the range lies in [-sup, sup]."""
        return float(np.abs(self.c).sum())

    # ----- exact risks ---------------------------------------------------
    def second_moment(self) -> float:
        """E f(X)^2 (E f = 0 because every term is odd)."""
        return float((self._w * np.sinh(self._G)).sum())

    def risk(self, U: np.ndarray) -> float:
        """Exact R_f(P) for P = U U^T, U with orthonormal columns (D x r)."""
        AU = self.A @ U
        GP = AU @ AU.T
        return float((self._w * (np.sinh(self._G) - np.sinh(GP))).sum())

    def gain(self, U0: np.ndarray, U1: np.ndarray) -> float:
        """Exact Delta_f = R_f(P0) - R_f(P1)."""
        return self.risk(U0) - self.risk(U1)


def sine_ridge(D: int, a=None) -> SineSum:
    """f(x) = sin(a^T x); default a = e_1 (range [-1, 1], Lipschitz 1)."""
    if a is None:
        a = np.zeros(D)
        a[0] = 1.0
    return SineSum([1.0], np.asarray(a, float)[None, :], name=f"sine_ridge_D{D}")


def planar_arm(D: int, lengths=None, sigma=None) -> SineSum:
    """Normalized vertical displacement of a planar arm with D joints.

    f(x) = (1/sum l) sum_j l_j sin(sum_{k<=j} sigma_k x_k).
    Default sigma_k = D^{-1/2}, so |a_j| <= 1 and the Lipschitz bound is at most 1.
    """
    lengths = np.ones(D) if lengths is None else np.asarray(lengths, float)
    sigma = np.full(D, D ** -0.5) if sigma is None else np.asarray(sigma, float)
    if np.any(lengths <= 0):
        raise ValueError("link lengths must be positive")
    A = np.tril(np.ones((D, D))) * sigma[None, :]
    c = lengths / lengths.sum()
    return SineSum(c, A, name=f"planar_arm_D{D}")
