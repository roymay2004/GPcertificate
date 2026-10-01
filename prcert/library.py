"""Development stage for the kinematic-arm study: a learned baseline and a candidate library.

All projections are learned from independent development data (exact gradients of the
simulator at Gaussian inputs).  Each gradient evaluation is charged `grad_cost` simulator
calls (1 for an adjoint-type gradient; D + 1 for forward finite differences).
Candidates are classified afterwards by their exact gain; the certifier never sees it.
"""
from __future__ import annotations

import numpy as np

from .geometry import geodesic, proj_dist, random_rotation, s_op, swap_direction
from .simulators import SineSum

# Order in which candidate generators enter the library as K grows.
GENERATORS = [
    "as_refit",          # active subspace from n_dev1 gradients (typically useful)
    "random_rot_0.3",    # random rotation by 0.3 rad (typically harmful)
    "geodesic_half",     # halfway from baseline to the refit (partial repair)
    "perturb_0.05",      # small random rotation (typically near-neutral)
    "geodesic_0.1",      # small step toward the refit (local first-order repair)
    "as_refit_seed2",    # refit from an independent set of n_dev1 gradients
    "random_rot_0.8",    # large random rotation (harmful)
    "swap_direction",    # replace the leading baseline direction (harmful)
]


def active_subspace(sim: SineSum, n: int, r: int, rng: np.random.Generator) -> np.ndarray:
    X = rng.standard_normal((n, sim.D))
    G = sim.grad(X)
    C = G.T @ G / n
    w, V = np.linalg.eigh(C)
    return V[:, ::-1][:, :r]


def build_library(sim: SineSum, r: int, K: int, rng: np.random.Generator,
                  n_dev0: int, n_dev1: int, grad_cost: float = 1.0, rel_tol: float = 0.01):
    if K > len(GENERATORS):
        raise ValueError(f"K <= {len(GENERATORS)}")
    U0 = active_subspace(sim, n_dev0, r, rng)
    grad_evals = n_dev0
    cache = {}

    def refit():
        nonlocal grad_evals
        if "refit" not in cache:
            cache["refit"] = active_subspace(sim, n_dev1, r, rng)
            grad_evals += n_dev1
        return cache["refit"]

    cands, names = [], []
    for name in GENERATORS[:K]:
        if name == "as_refit":
            U = refit()
        elif name == "geodesic_half":
            U = geodesic(U0, refit(), 0.5)
        elif name.startswith("random_rot_"):
            U = random_rotation(U0, float(name.split("_")[-1]), rng)
        elif name == "perturb_0.05":
            U = random_rotation(U0, 0.05, rng)
        elif name == "geodesic_0.1":
            U = geodesic(U0, refit(), 0.1)
        elif name == "as_refit_seed2":
            U = active_subspace(sim, n_dev1, r, rng)
            grad_evals += n_dev1
        elif name == "swap_direction":
            U = swap_direction(U0, rng, which=0)
        else:
            raise ValueError(name)
        cands.append(U)
        names.append(name)

    R0 = sim.risk(U0)
    gains = np.array([sim.gain(U0, U) for U in cands])
    rel = gains / R0
    cls = np.where(rel > rel_tol, "useful", np.where(rel < -rel_tol, "harmful", "neutral"))
    return {
        "U0": U0,
        "cands": cands,
        "names": names,
        "R0": R0,
        "gains": gains,
        "classes": cls,
        "d": np.array([proj_dist(U0, U) for U in cands]),
        "s": np.array([s_op(U0, U) for U in cands]),
        "dev_calls": grad_evals * grad_cost,
    }
