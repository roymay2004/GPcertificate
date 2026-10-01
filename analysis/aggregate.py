"""Aggregate task outputs into the tables behind Section 5.

    python -m analysis.aggregate --plan plan_full.json --results results --out tables

Budget protocol (Sec. 5.1): the power-attaining budget is the smallest grid budget whose
isotonic batch-A power reaches the target; power is then re-estimated on fresh batch-B
repetitions at that budget.  Every decision is a fixed-size audit at one grid budget.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from prcert.audits import eb_radius, geom_radius, select
from prcert.stats import bootstrap_slope, clopper_pearson, loglog_slope, select_budget

KAPPA_RMSE = 0.5          # relative-RMSE target for the estimator-specific comparison
C_GRID = 2.0 ** (np.arange(0, 25) / 2)   # declared-bound inflation factors 1 .. 4096
ARM_TABLE = {"D": 20, "r": 2, "K": 4, "n": 16384}   # prespecified table cell


def load(results, plan):
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(results, "*", "*.npz"))):
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        t = meta["task"]
        groups[(t["config"], t["batch"])].append((t["block"], z, meta))
    out = {}
    for key, items in groups.items():
        items.sort(key=lambda x: x[0])
        z0, meta = items[0][1], items[0][2]
        rec = {"cfg": plan["configs"][key[0]], "grid": z0["grid"], "calls": meta["calls_per_row"],
               "mean": np.concatenate([z["mean"] for _, z, _ in items]),
               "var": np.concatenate([z["var"] for _, z, _ in items]),
               "blocks": len(items)}
        for k in ("gains", "d", "s", "classes", "dev_calls", "libs"):
            if k in z0.files:
                rec[k] = (z0[k] if z0[k].ndim == 0
                          else np.concatenate([z[k] for _, z, _ in items]))
        for k in ("gain", "R0", "B", "Lam", "names"):
            if k in z0.files:
                rec[k] = z0[k]
        out[key] = rec
    return out


def decisions(rec, method, delta, c=1.0):
    cfg, n = rec["cfg"], rec["grid"]
    m, v = rec["mean"][..., 0], rec["var"][..., 0]
    if method == "geometric":
        return m - geom_radius(cfg["B"], c * cfg["Lam"], cfg["d"], n, 1, delta) > 0
    return m - eb_radius(v, n, cfg["B"], 1, delta) > 0


def method_list(kind):
    return [("geometric", "geometric"), ("eb_coupled", "eb")] if kind == "coupled" else [("eb_indep", "eb")]


def budget_row(recA, recB, method, delta, target, c=1.0):
    dA = decisions(recA, method, delta, c)
    i, n_sel = select_budget(recA["grid"], dA.mean(axis=0), target)
    row = {"n_selected": n_sel, "queries": None, "power_B": np.nan, "power_B_lo": np.nan, "power_B_hi": np.nan,
           "attained": n_sel is not None, "capped": recA["cfg"].get("capped", False)}
    if n_sel is not None and recB is not None:
        dB = decisions(recB, method, delta, c)[:, i]
        lo, hi = clopper_pearson(dB.sum(), dB.size)
        row.update(queries=n_sel * recA["calls"], power_B=dB.mean(), power_B_lo=float(lo), power_B_hi=float(hi))
    return row, dA


def study1(data, settings):
    delta, target = settings["delta"], settings["power"]
    power_rows, budget_rows, fc_rows, slope_rows = [], [], [], []
    by_key = {}
    for (ci, batch), rec in data.items():
        cfg = rec["cfg"]
        if cfg["study"] != "s1":
            continue
        by_key[(cfg["kind"], cfg["path"], cfg["move"], cfg["h"], batch)] = rec
        for label, meth in method_list(cfg["kind"]):
            d = decisions(rec, meth, delta)
            k = d.sum(axis=0)
            lo, hi = clopper_pearson(k, d.shape[0])
            for g, n in enumerate(rec["grid"]):
                power_rows.append(dict(path=cfg["path"], move=cfg["move"], h=cfg["h"], method=label, batch=batch,
                                       n=int(n), queries=int(n) * rec["calls"], rate=k[g] / d.shape[0],
                                       lo=lo[g], hi=hi[g], reps=d.shape[0]))
    slope_inputs = defaultdict(lambda: ([], [], [], []))
    for (kind, path, move, h, batch), recA in by_key.items():
        if move != "favorable" or batch != "A":
            continue
        recB = by_key.get((kind, path, move, h, "B"))
        cfg = recA["cfg"]
        for label, meth in method_list(kind):
            row, dA = budget_row(recA, recB, meth, delta, target)
            budget_rows.append(dict(path=path, h=h, method=label, gain=cfg["gain"], d=cfg["d"],
                                    n_pred=cfg["n_pred_geo"] if meth == "geometric" else cfg["n_pred_eb"], **row))
            s = slope_inputs[(path, label)]
            s[0].append(h); s[1].append(dA); s[2].append(recA["grid"]); s[3].append(recA["calls"])
            for bad in ("harmful", "equal"):
                recX = by_key.get((kind, path, bad, h, "B"))
                if recX is None:
                    continue
                dX = decisions(recX, meth, delta)
                rates = dX.mean(axis=0)
                gmax = int(np.argmax(rates))
                lo, hi = clopper_pearson(dX[:, gmax].sum(), dX.shape[0])
                at = {}
                if row["n_selected"] is not None:
                    j = int(np.nonzero(recX["grid"] == row["n_selected"])[0][0])
                    at = dict(rate_at_selected=rates[j])
                fc_rows.append(dict(path=path, h=h, method=label, move=bad, gain=recX["cfg"]["gain"],
                                    max_rate=rates[gmax], max_rate_n=int(recX["grid"][gmax]), max_rate_lo=lo,
                                    max_rate_hi=hi, reps=dX.shape[0], delta=delta, **at))
    budgets = pd.DataFrame(budget_rows).sort_values(["path", "method", "h"])
    for (path, label), (hs, decs, grids, calls) in slope_inputs.items():
        sub = budgets[(budgets.path == path) & (budgets.method == label) & budgets.attained]
        if len(sub) >= 3:
            sl = loglog_slope(sub.h, sub.n_selected * calls[0])
            lo, hi = bootstrap_slope(hs, decs, grids, calls[0], target, n_boot=500)
            slope_rows.append(dict(path=path, method=label, slope=sl, lo=lo, hi=hi, n_h=len(sub)))
    return (pd.DataFrame(power_rows), budgets, pd.DataFrame(fc_rows), pd.DataFrame(slope_rows))


def csweep(data, settings):
    delta, target = settings["delta"], settings["power"]
    rows = []
    recs = {(r["cfg"]["h"], b): r for (ci, b), r in data.items()
            if r["cfg"]["study"] == "s1" and r["cfg"]["kind"] == "coupled"
            and r["cfg"]["path"] == "first" and r["cfg"]["move"] == "favorable"}
    for h in sorted({h for h, _ in recs}):
        recA, recB = recs.get((h, "A")), recs.get((h, "B"))
        if recA is None:
            continue
        eb, _ = budget_row(recA, recB, "eb", delta, target)
        g1, _ = budget_row(recA, recB, "geometric", delta, target)
        cfg = recA["cfg"]
        for c in C_GRID:
            g, _ = budget_row(recA, recB, "geometric", delta, target, c)
            rows.append(dict(h=h, c=c, geo_queries=g["queries"], geo_attained=g["attained"],
                             geo_power_B=g["power_B"], eb_queries=eb["queries"],
                             c_cross_theory=np.sqrt(eb["queries"] / g1["queries"]) if eb["queries"] and g1["queries"] else np.nan,
                             c_threshold=(cfg["Lam"] + 32.0 * 1.0 / cfg["s"]) / cfg["Lam"], s=cfg["s"]))
    return pd.DataFrame(rows)


def distance(data, settings):
    delta, target = settings["delta"], settings["power"]
    rows = []
    recs = {(r["cfg"]["D"], r["cfg"]["r"], r["cfg"]["phi"], b): r for (ci, b), r in data.items()
            if r["cfg"]["study"] == "s2dist"}
    for (D, r, phi, b), recA in sorted(recs.items()):
        if b != "A":
            continue
        recB = recs.get((D, r, phi, "B"))
        for label, meth in method_list("coupled"):
            row, _ = budget_row(recA, recB, meth, delta, target)
            rows.append(dict(D=D, r=r, phi=phi, d=recA["cfg"]["d"], gain=recA["cfg"]["gain"], method=label, **row))
    return pd.DataFrame(rows)


def rmse(data):
    rows = []
    for (ci, b), rec in data.items():
        cfg = rec["cfg"]
        if cfg["study"] != "rmse":
            continue
        v = rec["var"][:, 0, 0]
        var, se = v.mean(), v.std(ddof=1) / np.sqrt(v.size)
        rows.append(dict(kind=cfg["kind"], h=cfg["h"], gain=cfg["gain"], d=cfg["d"], row_var=var, row_var_se=se,
                         var_over_d2=var / cfg["d"] ** 2, calls_per_row=rec["calls"],
                         rows_for_rel_rmse=var / (KAPPA_RMSE * cfg["gain"]) ** 2,
                         queries_for_rel_rmse=rec["calls"] * var / (KAPPA_RMSE * cfg["gain"]) ** 2))
    return pd.DataFrame(rows).sort_values(["kind", "h"])


def arm(data, settings, n_boot=1000, seed=0):
    delta = settings["delta"]
    rng = np.random.default_rng(seed)
    rows, cand_rows = [], []
    for (ci, b), rec in data.items():
        cfg = rec["cfg"]
        if cfg["study"] != "s3arm":
            continue
        m, v, n = rec["mean"], rec["var"], rec["grid"]            # (L, R, G, K)
        gains, d = rec["gains"], rec["d"]                         # (L, K)
        K, L = gains.shape[1], gains.shape[0]
        boot_idx = rng.integers(0, L, size=(n_boot, L))           # cluster bootstrap over libraries
        oracle = np.maximum(0.0, gains.max(axis=1))               # (L,)
        for cls in ("useful", "neutral", "harmful"):
            cand_rows.append(dict(D=cfg["D"], r=cfg["r"], K=K, cls=cls, frac=(rec["classes"] == cls).mean()))
        for label in ("geometric", "eb_coupled"):
            if label == "geometric":
                rad = geom_radius(float(rec["B"]), float(rec["Lam"]), d[:, None, None, :],
                                  n[None, None, :, None], K, delta)
                rad = np.broadcast_to(rad, m.shape)
            else:
                rad = eb_radius(v, n[None, None, :, None], float(rec["B"]), K, delta)
            sel = select(m, rad)                                  # (L, R, G)
            g_ext = np.concatenate([np.zeros((L, 1)), gains], axis=1)
            sel_gain = g_ext[np.arange(L)[:, None, None], sel]
            cover = (np.abs(m - gains[:, None, None, :]) <= rad).all(axis=-1)
            metrics = {"coverage": cover, "harmful_selection": sel_gain < 0, "retained_baseline": sel == 0,
                       "selected_gain": sel_gain, "gain_fraction": sel_gain / np.where(oracle > 0, oracle, np.nan)[:, None, None]}
            for g, ng in enumerate(n):
                row = dict(D=cfg["D"], r=cfg["r"], K=K, method=label, n=int(ng), audit_calls=int(ng) * (K + 2),
                           dev_calls=float(rec["dev_calls"].mean()), libs=L, reps=m.shape[1],
                           oracle_gain=float(oracle.mean()))
                row["total_calls"] = row["audit_calls"] + row["dev_calls"]
                for name, arr in metrics.items():
                    per_lib = np.nanmean(arr[:, :, g].astype(float), axis=1)
                    boots = np.nanmean(per_lib[boot_idx], axis=1)
                    row[name] = float(np.nanmean(per_lib))
                    row[name + "_lo"], row[name + "_hi"] = np.nanpercentile(boots, [2.5, 97.5])
                rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(cand_rows)


def arm_table(arm_df, out):
    t = ARM_TABLE
    sub = arm_df[(arm_df.D == t["D"]) & (arm_df.r == t["r"]) & (arm_df.K == t["K"])]
    if sub.empty:
        return
    n = sub.n.unique()[np.argmin(np.abs(np.log(sub.n.unique() / t["n"])))]
    sub = sub[sub.n == n].set_index("method")
    fmt = lambda x: f"{x:.3f}"
    lines = [r"\begin{tabular}{lll}", r"\toprule", r"Endpoint & Geometric & Range-only\\", r"\midrule"]
    for name, col, f in [("Audit calls", "audit_calls", lambda x: f"{int(x):,}"),
                         ("Total calls", "total_calls", lambda x: f"{int(round(x)):,}"),
                         ("Harmful selection", "harmful_selection", fmt),
                         ("Abstention", "retained_baseline", fmt),
                         ("Selected gain", "selected_gain", lambda x: f"{x:.4f}")]:
        lines.append(f"{name} & {f(sub.loc['geometric', col])} & {f(sub.loc['eb_coupled', col])}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              f"% D={t['D']}, r={t['r']}, K={t['K']}, n={int(n)} rows; means over libraries and audit repetitions"]
    with open(os.path.join(out, "table_arm.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    os.makedirs(a.out, exist_ok=True)
    data = load(a.results, plan)
    S = plan["settings"]
    power, budgets, fc, slopes = study1(data, S)
    power.to_csv(os.path.join(a.out, "s1_power.csv"), index=False)
    budgets.to_csv(os.path.join(a.out, "s1_budgets.csv"), index=False)
    fc.to_csv(os.path.join(a.out, "s1_false_certification.csv"), index=False)
    slopes.to_csv(os.path.join(a.out, "s1_slopes.csv"), index=False)
    csweep(data, S).to_csv(os.path.join(a.out, "s2_csweep.csv"), index=False)
    distance(data, S).to_csv(os.path.join(a.out, "s2_distance.csv"), index=False)
    rmse(data).to_csv(os.path.join(a.out, "s1_rmse.csv"), index=False)
    arm_df, cand = arm(data, S)
    arm_df.to_csv(os.path.join(a.out, "s3_arm.csv"), index=False)
    cand.to_csv(os.path.join(a.out, "s3_candidate_classes.csv"), index=False)
    arm_table(arm_df, a.out)
    n_expected = len(plan["tasks"])
    n_found = sum(r["blocks"] for r in data.values())
    print(f"aggregated {n_found}/{n_expected} task outputs into {a.out}/")
    if n_found < n_expected:
        print("WARNING: missing task outputs; rerun the array (finished tasks are skipped)")


if __name__ == "__main__":
    main()
