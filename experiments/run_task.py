"""Run tasks from a plan.  Results are written atomically and finished tasks are skipped.

    python -m experiments.run_task --plan plan.json --task 17
    python -m experiments.run_task --plan plan.json --array-offset 1000      # uses SLURM_ARRAY_TASK_ID
    python -m experiments.run_task --plan plan.json --all --workers 8        # local runs
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from prcert.sampling import run_checkpoints
from experiments.common import BATCH_CODE, STUDY_CODE, arm_library, rng_for, sine_setup


def out_path(root, task):
    return os.path.join(root, task["study"], f"c{task['config']:04d}_{task['batch']}_b{task['block']:03d}.npz")


def run_one(plan, ti, root, chunk_rows):
    task = plan["tasks"][ti]
    cfg = plan["configs"][task["config"]]
    path = out_path(root, task)
    if os.path.exists(path):
        return path, 0.0, "skip"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    rng = rng_for(cfg["seed"], STUDY_CODE[task["study"]], task["config"], BATCH_CODE[task["batch"]], task["block"])
    extra = {}
    if task["study"] == "s3arm":
        means, vars_, gains, ds, ss, cls, dev = [], [], [], [], [], [], []
        for lib in range(*task["libs"]):
            smp, info = arm_library(cfg, lib)
            grid, m, v = run_checkpoints(smp, cfg["grid"], cfg["reps"], rng, min(chunk_rows, 2 ** 18))
            means.append(m); vars_.append(v)
            gains.append(info["gains"]); ds.append(info["d"]); ss.append(info["s"])
            cls.append(info["classes"]); dev.append(info["dev_calls"])
        mean, var = np.stack(means), np.stack(vars_)
        extra = dict(gains=np.array(gains), d=np.array(ds), s=np.array(ss), classes=np.array(cls),
                     dev_calls=np.array(dev), names=np.array(info["names"]), R0=info["R0"],
                     Lam=info["Lam"], B=info["B"], libs=np.arange(*task["libs"]))
    else:
        smp, info = sine_setup(cfg)
        grid, mean, var = run_checkpoints(smp, cfg["grid"], task["reps"], rng, chunk_rows)
        extra = {k: info[k] for k in ("gain", "d", "s", "R0", "B", "Lam")}
    meta = json.dumps({"task": task, "config": cfg, "seconds": time.time() - t0,
                       "calls_per_row": smp.calls_per_row})
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, grid=np.asarray(grid), mean=mean.astype(np.float64),
                        var=var.astype(np.float64), meta=meta, **extra)
    os.replace(tmp, path)
    return path, time.time() - t0, "done"


def _worker(args):
    plan_file, ti, root, chunk = args
    with open(plan_file) as fh:
        plan = json.load(fh)
    return run_one(plan, ti, root, chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--task", type=int)
    ap.add_argument("--array-offset", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunk-rows", type=int, default=2 ** 20)
    a = ap.parse_args()
    with open(a.plan) as fh:
        plan = json.load(fh)
    if a.all:
        ids = list(range(len(plan["tasks"])))
    elif a.array_offset is not None:
        ids = [a.array_offset + int(os.environ["SLURM_ARRAY_TASK_ID"])]
    else:
        ids = [a.task]
    ids = [i for i in ids if i < len(plan["tasks"])]
    jobs = [(a.plan, i, a.results, a.chunk_rows) for i in ids]
    if a.workers > 1:
        with Pool(a.workers) as pool:
            for p, sec, st in pool.imap_unordered(_worker, jobs):
                print(f"{st:4s} {sec:8.1f}s {p}", flush=True)
    else:
        for j in jobs:
            p, sec, st = _worker(j)
            print(f"{st:4s} {sec:8.1f}s {p}", flush=True)


if __name__ == "__main__":
    main()
