"""Projector geometry: bases, distances, and the repair paths used in the paper."""
from __future__ import annotations

import numpy as np


def orth(M: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Orthonormal basis of the column span of M."""
    M = np.atleast_2d(M)
    if M.size == 0:
        return np.zeros((M.shape[0], 0))
    Uu, s, _ = np.linalg.svd(M, full_matrices=False)
    keep = s > tol * max(s.max(), 1.0)
    return Uu[:, keep]


def proj_dist(U0: np.ndarray, U1: np.ndarray) -> float:
    """d = ||P1 - P0||_F for P = U U^T."""
    r0, r1 = U0.shape[1], U1.shape[1]
    c = np.linalg.norm(U0.T @ U1) ** 2
    return float(np.sqrt(max(r0 + r1 - 2.0 * c, 0.0)))


def s_op(U0: np.ndarray, U1: np.ndarray) -> float:
    """s = ||(I - P1) P0||_op: sine of the largest angle from range(P0) to range(P1)."""
    M = U0 - U1 @ (U1.T @ U0)
    return float(np.linalg.norm(M, 2)) if M.size else 0.0


def unit(D: int, i: int) -> np.ndarray:
    e = np.zeros(D)
    e[i] = 1.0
    return e


def u_theta(theta: float, D: int) -> np.ndarray:
    """Rank-one basis retaining (sin theta, cos theta) in the (x1, x2) plane."""
    u = np.zeros(D)
    u[0], u[1] = np.sin(theta), np.cos(theta)
    return u[:, None]


def equal_risk_pair(theta0: float, h: float, D: int):
    """Rank-one pair with identical exact risk for f = sin(x1) and distance of order h.

    The candidate rotates u(theta0) about e1 by angle h, so a^T u is unchanged.
    Requires D >= 3.
    """
    if D < 3:
        raise ValueError("equal-risk rotation needs D >= 3")
    u0 = np.zeros(D)
    u0[0], u0[1] = np.sin(theta0), np.cos(theta0)
    u1 = np.zeros(D)
    u1[0], u1[1], u1[2] = np.sin(theta0), np.cos(theta0) * np.cos(h), np.cos(theta0) * np.sin(h)
    return u0[:, None], u1[:, None]


def sine_path_pair(path: str, move: str, h: float, D: int):
    """Baseline/candidate bases for the paper's two sine paths.

    path:  'first'  -> theta0 = pi/4 (first-order gain),
           'second' -> theta0 = 0    (second-order gain).
    move:  'favorable' (theta0 -> theta0 + h), 'harmful' (reverse),
           'equal' (exact zero gain at the same distance scale).
    """
    theta0 = np.pi / 4 if path == "first" else 0.0
    if move == "favorable":
        return u_theta(theta0, D), u_theta(theta0 + h, D)
    if move == "harmful":
        return u_theta(theta0 + h, D), u_theta(theta0, D)
    if move == "equal":
        if path == "second":  # symmetric about the stationary point
            return u_theta(-h / 2, D), u_theta(h / 2, D)
        return equal_risk_pair(theta0, h, D)
    raise ValueError(move)


def inactive_rotation_pair(D: int, r: int, h: float, phi: float, theta0: float = np.pi / 4):
    """Rank-r pair for the distance study (f = sin(x1)).

    Baseline: u(theta0) plus inactive directions e_3..e_{r+1}.  Candidate: u(theta0 + h)
    plus the inactive directions rotated by phi toward e_{r+2}..e_{2r}.  The exact gain
    is that of the rank-one move, while d^2 = 2 sin^2 h + 2 (r - 1) sin^2 phi.
    """
    if D < 2 * r:
        raise ValueError("need D >= 2r")
    U0 = [u_theta(theta0, D)[:, 0]] + [unit(D, 1 + i) for i in range(1, r)]
    U1 = [u_theta(theta0 + h, D)[:, 0]] + [
        np.cos(phi) * unit(D, 1 + i) + np.sin(phi) * unit(D, r + i) for i in range(1, r)
    ]
    return np.column_stack(U0), np.column_stack(U1)


def geodesic(U0: np.ndarray, U1: np.ndarray, t: float) -> np.ndarray:
    """Point at fraction t on the Grassmann geodesic from span(U0) to span(U1) (equal ranks)."""
    Y, cos, Zt = np.linalg.svd(U0.T @ U1)
    cos = np.clip(cos, -1.0, 1.0)
    theta = np.arccos(cos)
    U0Y = U0 @ Y
    Q = U1 @ Zt.T - U0Y * cos
    sin = np.sin(theta)
    Q = np.where(sin > 1e-12, Q / np.where(sin > 1e-12, sin, 1.0), 0.0)
    return orth(U0Y * np.cos(t * theta) + Q * np.sin(t * theta))


def random_complement(U0: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k random orthonormal directions orthogonal to span(U0)."""
    D = U0.shape[0]
    G = rng.standard_normal((D, k))
    G -= U0 @ (U0.T @ G)
    return orth(G)[:, :k]


def random_rotation(U0: np.ndarray, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Rotate every baseline direction by angle phi into a random orthogonal direction."""
    W = random_complement(U0, U0.shape[1], rng)
    return U0 * np.cos(phi) + W * np.sin(phi)


def swap_direction(U0: np.ndarray, rng: np.random.Generator, which: int = 0) -> np.ndarray:
    """Replace one baseline direction by a random direction orthogonal to span(U0)."""
    U1 = U0.copy()
    U1[:, which] = random_complement(U0, 1, rng)[:, 0]
    return U1
