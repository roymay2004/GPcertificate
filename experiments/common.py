"""Shared construction of simulators, projector pairs, and samplers from task records."""
from __future__ import annotations

import numpy as np

from prcert.geometry import inactive_rotation_pair, proj_dist, s_op, sine_path_pair
from prcert.library import build_library
from prcert.sampling import CoupledSampler, IndepSampler
from prcert.simulators import planar_arm, sine_ridge

STUDY_CODE = {"s1": 1, "s2dist": 2, "s3arm": 3, "rmse": 4}
BATCH_CODE = {"A": 1, "B": 2, "-": 0}

# Declared information shared by every audit of the sine benchmarks (Sec. 3 and 5).
SINE_DECLARED = {"B": 4.0, "Lam": 1.0}   # I = [-2, 2]; sin(x1) is 1-Lipschitz
ARM_DECLARED_B = 2.0                     # I = [-1, 1]: the arm's exact range; no margin is needed
                                         # because Study 3 uses only upper bounds and selection


def rng_for(seed: int, *key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed)] + [int(k) for k in key]))


def sine_pair(task):
    if task["study"] == "s2dist":
        return inactive_rotation_pair(task["D"], task["r"], task["h"], task["phi"])
    return sine_path_pair(task["path"], task["move"], task["h"], task["D"])


def sine_setup(task):
    """Simulator, bases, sampler, and exact quantities for sine-ridge tasks."""
    sim = sine_ridge(task["D"])
    U0, U1 = sine_pair(task)
    sampler = (CoupledSampler(sim, U0, [U1]) if task["kind"] == "coupled"
               else IndepSampler(sim, U0, U1))
    info = {"gain": sim.gain(U0, U1), "d": proj_dist(U0, U1), "s": s_op(U0, U1),
            "R0": sim.risk(U0), **SINE_DECLARED}
    return sampler, info


def arm_library(task, lib_index: int):
    D = task["D"]
    sim = planar_arm(D, sigma=np.full(D, task.get("arm_scale", 1.0) / np.sqrt(D)))
    rng = rng_for(task["seed"], STUDY_CODE["s3arm"], task["config"], 99, lib_index)
    lib = build_library(sim, task["r"], task["K"], rng, task["n_dev0"], task["n_dev1"],
                        task.get("grad_cost", 1.0))
    sampler = CoupledSampler(sim, lib["U0"], lib["cands"])
    info = {"B": task.get("arm_B", ARM_DECLARED_B), "Lam": sim.lipschitz_bound(), "R0": lib["R0"],
            "gains": lib["gains"], "d": lib["d"], "s": lib["s"], "classes": lib["classes"],
            "names": lib["names"], "dev_calls": lib["dev_calls"]}
    return sampler, info
