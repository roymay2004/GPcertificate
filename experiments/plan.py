"""Build the task plan for all studies and estimate its compute cost.

    python -m experiments.plan --scale full --out plan_full.json
    python -m experiments.plan --scale smoke --out plan_smoke.json

Budgets are sized from exact gains and pilot variances; reported budgets come from the
simulations themselves.  Each task is sized to about `task_hours` of one core.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from prcert.audits import budget_grid, predicted_rows
from prcert.sampling import run_checkpoints
from experiments.common import BATCH_CODE, SINE_DECLARED, STUDY_CODE, arm_library, rng_for, sine_setup

SCALES = {
    "full": dict(
        D=3, delta=0.05, power=0.9,
        h_first=[0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
        h_second=[0.4, 0.2, 0.1, 0.05, 0.02, 0.01],
        RA=1000, RB=2000, RA_big=500, RB_big=1000, big_rows=1e8, cap_rows=2 ** 30,
        csweep_factor=16, nmax_factor=3,
        dist_D=[10, 100], dist_r=[1, 2, 4], dist_phi=[0.0, 0.05, 0.1, 0.2, 0.4], dist_h=0.02,
        arm_D=[10, 20, 50], arm_r=[1, 2, 4], arm_K=[2, 4, 8], arm_libs=40, arm_reps=25,
        arm_nmax=2 ** 24, arm_dev1_per_D=10, arm_scale=3.0, arm_B=2.0,
        rmse_h=[0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005], rmse_reps=32, rmse_rows=2 ** 22,
        pilot_rows=2 ** 18, task_hours=1.0,
    ),
    "smoke": dict(
        D=3, delta=0.05, power=0.9,
        h_first=[0.2, 0.1, 0.05], h_second=[0.4, 0.2],
        RA=100, RB=200, RA_big=50, RB_big=100, big_rows=1e6, cap_rows=2 ** 21,
        csweep_factor=16, nmax_factor=3,
        dist_D=[10], dist_r=[1, 2], dist_phi=[0.0, 0.2], dist_h=0.05,
        arm_D=[10], arm_r=[2], arm_K=[2, 4], arm_libs=4, arm_reps=25,
        arm_nmax=2 ** 15, arm_dev1_per_D=10, arm_scale=3.0, arm_B=2.0,
        rmse_h=[0.4, 0.2, 0.1], rmse_reps=8, rmse_rows=2 ** 16,
        pilot_rows=2 ** 15, task_hours=0.02,
    ),
}


def _throughput(sampler, rows=2 ** 15):
    rng = np.random.default_rng(0)
    run_checkpoints(sampler, [rows // 8], 1, rng)  # warm-up
    t = time.perf_counter()
    run_checkpoints(sampler, [rows], 1, rng)
    return rows / (time.perf_counter() - t)


def _blocks(total: int, per_block: int):
    per_block = max(1, int(per_block))
    return [(a, min(total, a + per_block)) for a in range(0, total, per_block)]


def build(scale: str, seed: int, speed: float):
    S = SCALES[scale]
    configs, tasks = [], []
    tput_cache = {}
    secs = S["task_hours"] * 3600.0

    def add_sine(study, base, grid, nmax, capped, reps_by_batch):
        cfg = dict(base, study=study, config=len(configs), grid=[int(g) for g in grid],
                   nmax=int(grid[-1]), capped=bool(capped), seed=seed)
        sampler, info = sine_setup(cfg)
        cfg.update({k: float(v) for k, v in info.items()})
        key = (cfg["kind"], cfg["study"], cfg.get("r", 1))
        if key not in tput_cache:
            tput_cache[key] = _throughput(sampler) * speed
        cfg["rows_per_sec"] = tput_cache[key]
        configs.append(cfg)
        per_block = max(1, int(secs * cfg["rows_per_sec"] / cfg["nmax"]))
        for batch, reps in reps_by_batch.items():
            for b, (r0, r1) in enumerate(_blocks(reps, per_block)):
                tasks.append(dict(config=cfg["config"], study=study, batch=batch, block=b,
                                  reps=r1 - r0, est_hours=(r1 - r0) * cfg["nmax"] / cfg["rows_per_sec"] / 3600))

    def nmax_and_reps(n_need):
        nmax = min(S["cap_rows"], max(256, S["nmax_factor"] * n_need))
        capped = S["nmax_factor"] * n_need > S["cap_rows"]
        big = nmax > S["big_rows"]
        return nmax, capped, (S["RA_big"] if big else S["RA"]), (S["RB_big"] if big else S["RB"])

    B, Lam, delta = SINE_DECLARED["B"], SINE_DECLARED["Lam"], S["delta"]
    # ---------------- Study 1 (and the c-sweep of Study 2, which reuses its rows) ----------
    for path, hs in (("first", S["h_first"]), ("second", S["h_second"])):
        for h in hs:
            pilot = {}
            for kind in ("coupled", "indep"):
                base = dict(study="s1", kind=kind, path=path, move="favorable", h=h, D=S["D"])
                smp, info = sine_setup(base)
                _, m, v = run_checkpoints(smp, [S["pilot_rows"]], 1, rng_for(seed, 0, int(h * 1e6), len(pilot)))
                pilot[kind] = (float(v[0, 0, 0]), info)
            var_c, info = pilot["coupled"]
            var_i, _ = pilot["indep"]
            n_geo = predicted_rows("geometric", info["gain"], var_c, B, info["d"], Lam, delta, S["power"])
            n_ebc = predicted_rows("eb", info["gain"], var_c, B, None, None, delta, S["power"])
            n_ebi = predicted_rows("eb", info["gain"], var_i, B, None, None, delta, S["power"])
            need = {"coupled": max(n_geo, n_ebc), "indep": n_ebi}
            if path == "first":  # cover declared-bound inflation up to ~4x the crossover
                need["coupled"] = max(need["coupled"], S["csweep_factor"] * n_ebc / S["nmax_factor"])
            for kind in ("coupled", "indep"):
                nmax, capped, RA, RB = nmax_and_reps(need[kind])
                grid = budget_grid(nmax)
                for move in ("favorable", "harmful", "equal"):
                    reps = {"A": RA, "B": RB} if move == "favorable" else {"B": RB}
                    add_sine("s1", dict(kind=kind, path=path, move=move, h=h, D=S["D"],
                                        n_pred_geo=float(n_geo), n_pred_eb=float(n_ebc if kind == "coupled" else n_ebi)),
                             grid, nmax, capped, reps)
    # ---------------- Study 2: projector distance and inactive coordinates ----------------
    for D in S["dist_D"]:
        for r in S["dist_r"]:
            for phi in (S["dist_phi"] if r > 1 else [0.0]):
                base = dict(study="s2dist", kind="coupled", D=D, r=r, h=S["dist_h"], phi=phi, path="first", move="favorable")
                smp, info = sine_setup(base)
                _, m, v = run_checkpoints(smp, [S["pilot_rows"]], 1, rng_for(seed, 2, D, r, int(phi * 1000)))
                var = float(v[0, 0, 0])
                n_geo = predicted_rows("geometric", info["gain"], var, B, info["d"], Lam, delta, S["power"])
                n_eb = predicted_rows("eb", info["gain"], var, B, None, None, delta, S["power"])
                nmax, capped, RA, RB = nmax_and_reps(max(n_geo, n_eb))
                add_sine("s2dist", dict(base, n_pred_geo=float(n_geo), n_pred_eb=float(n_eb)),
                         budget_grid(nmax), nmax, capped, {"A": RA, "B": RB})
    # ---------------- RMSE study (second-order path) ----------------
    for kind in ("coupled", "indep"):
        for h in S["rmse_h"]:
            add_sine("rmse", dict(kind=kind, path="second", move="favorable", h=h, D=S["D"]),
                     [S["rmse_rows"]], S["rmse_rows"], False, {"-": S["rmse_reps"]})
    # ---------------- Study 3: kinematic arm, safe selection ----------------
    grid = budget_grid(S["arm_nmax"], n_min=64, per_octave=2)
    for D in S["arm_D"]:
        for r in S["arm_r"]:
            for K in S["arm_K"]:
                cfg = dict(study="s3arm", config=len(configs), D=D, r=r, K=K, n_dev0=r + 1,
                           arm_scale=S["arm_scale"], arm_B=S["arm_B"],
                           n_dev1=S["arm_dev1_per_D"] * D, grad_cost=1.0, libs=S["arm_libs"],
                           reps=S["arm_reps"], grid=[int(g) for g in grid], nmax=int(grid[-1]), seed=seed)
                smp, _ = arm_library(cfg, 0)
                key = ("arm", D, r, K)
                if key not in tput_cache:
                    tput_cache[key] = _throughput(smp, 2 ** 13) * speed
                cfg["rows_per_sec"] = tput_cache[key]
                configs.append(cfg)
                per_lib = S["arm_reps"] * cfg["nmax"] / cfg["rows_per_sec"]
                per_block = max(1, int(secs / per_lib))
                for b, (l0, l1) in enumerate(_blocks(S["arm_libs"], per_block)):
                    tasks.append(dict(config=cfg["config"], study="s3arm", batch="-", block=b,
                                      libs=[l0, l1], est_hours=(l1 - l0) * per_lib / 3600))
    for i, t in enumerate(tasks):
        t["task"] = i
    return {"scale": scale, "seed": seed, "settings": S, "configs": configs, "tasks": tasks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="full", choices=sorted(SCALES))
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="multiply measured throughput (e.g. 1.3 if compute nodes are faster than this host)")
    ap.add_argument("--out", default="plan.json")
    a = ap.parse_args()
    plan = build(a.scale, a.seed, a.speed)
    with open(a.out, "w") as fh:
        json.dump(plan, fh, indent=1)
    by = {}
    for t in plan["tasks"]:
        by.setdefault(t["study"], [0, 0.0, 0.0])
        by[t["study"]][0] += 1
        by[t["study"]][1] += t["est_hours"]
        by[t["study"]][2] = max(by[t["study"]][2], t["est_hours"])
    tot = sum(v[1] for v in by.values())
    print(f"plan '{a.scale}': {len(plan['configs'])} configs, {len(plan['tasks'])} tasks -> {a.out}")
    for k, (n, hrs, mx) in sorted(by.items()):
        print(f"  {k:7s} {n:6d} tasks  {hrs:9.1f} core-hours  (longest task {mx:.2f} h)")
    print(f"  total   {tot:9.1f} core-hours = {tot / 24:.2f} core-days "
          f"= {tot / (200 * 24):.2f} days at 200 CPU-days/day")
    capped = [c for c in plan["configs"] if c.get("capped")]
    if capped:
        print(f"  {len(capped)} configs hit the row cap (cost beyond the cap is reported as not attained):")
        for c in capped:
            print(f"    {c['study']} {c['kind']} {c.get('path')} {c.get('move')} h={c.get('h')}")


if __name__ == "__main__":
    main()
