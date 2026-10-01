"""Figures 2-3 of Section 5 and two appendix figures, from the aggregated tables.

    python -m analysis.figures --tables tables --out figures

Static print figures: methods are distinguished by color AND marker AND direct label,
so identity survives grayscale printing and color-vision deficiency.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METHODS = {  # validated categorical slots 1-3 (light mode)
    "geometric": dict(color="#2a78d6", marker="o", label="Geometric (known Λ)"),
    "eb_coupled": dict(color="#eb6834", marker="s", label="Coupled EB (range only)"),
    "eb_indep": dict(color="#1baf7a", marker="^", label="Independent EB"),
}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.5, "lines.markersize": 4.5, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _clean(fig):
    """Major-tick labels only on log axes (minor labels collide on short ranges)."""
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_minor_formatter(NullFormatter())


def _line(ax, x, y, m, ls="-", label=True, **kw):
    s = METHODS[m]
    ax.plot(x, y, ls=ls, color=s["color"], marker=s["marker"], mec="white", mew=0.6,
            label=s["label"] if label else None, **kw)


def _ref_slope(ax, x, y_anchor, x_anchor, slope, text, offset=0.45):
    """Slope guide drawn parallel to, and below, the data it describes."""
    x = np.asarray(x, float)
    y = offset * y_anchor * (x / x_anchor) ** slope
    ax.plot(x, y, ls=(0, (3, 2)), color=INK2, lw=0.9, zorder=1)
    i = int(np.argmin(x))
    ax.annotate(text, (x[i], y[i]), xytext=(1, -7), textcoords="offset points", fontsize=6.5, color=INK2)


SHORT = {"geometric": "geometric", "eb_coupled": "coupled EB", "eb_indep": "indep. EB"}


def _end_label(ax, x, y, m):
    """Direct label at the large-h end of a series, in text ink (identity from the adjacent mark)."""
    ax.annotate(SHORT[m], (x, y), xytext=(4, 0), textcoords="offset points", va="center",
                fontsize=6.5, color=INK)


def fig2(T, out):
    b = T["s1_budgets"]
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.15), constrained_layout=True)
    for ax, path, refs in [(axes[0], "first", [(0, "slope 0", "geometric"), (-1, "slope −1", "eb_coupled")]),
                           (axes[1], "second", [(-2, "slope −2", "geometric"), (-4, "slope −4", "eb_indep")])]:
        sub = b[b.path == path]
        for m in METHODS:
            s = sub[(sub.method == m) & sub.attained].sort_values("h")
            if len(s):
                _line(ax, s.h, s.queries, m, label=(path == "first"))
                _end_label(ax, s.h.iloc[-1], s.queries.iloc[-1], m)
            cap = sub[(sub.method == m) & ~sub.attained]
            if len(cap):  # not attained within the row cap: open marker at the plotted ceiling
                ax.scatter(cap.h, np.full(len(cap), ax.get_ylim()[1] if len(s) else 1),
                           marker=METHODS[m]["marker"], facecolors="none", edgecolors=METHODS[m]["color"], s=18)
        for slope, text, m in refs:
            s = sub[(sub.method == m) & sub.attained].sort_values("h")
            if len(s) >= 2:
                _ref_slope(ax, s.h, s.queries.iloc[-1], s.h.iloc[-1], slope, text)
        ax.set(xscale="log", yscale="log", xlabel="step h",
               title=("(a) first-order path" if path == "first" else "(b) second-order path"))
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi * 2.2)  # room for direct labels
    axes[0].set_ylabel("simulator calls for 90% power")
    ax = axes[2]
    fc = T["s1_false_certification"]
    first = b[(b.path == "first") & b.attained]
    for m in METHODS:
        s = first[first.method == m].sort_values("h")
        if len(s):
            ax.errorbar(s.h, s.power_B, yerr=[s.power_B - s.power_B_lo, s.power_B_hi - s.power_B],
                        color=METHODS[m]["color"], marker=METHODS[m]["marker"], ls="-", capsize=1.5, lw=1.2)
        f = fc[(fc.path == "first") & (fc.method == m)].groupby("h").max_rate_hi.max()
        if len(f):
            ax.plot(f.index, f.values, ls=":", color=METHODS[m]["color"], marker=METHODS[m]["marker"], mfc="white")
    ax.axhline(0.9, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.axhline(0.05, color=INK2, lw=0.8, ls=(0, (1, 1.5)))
    ax.text(0.98, 0.80, "target power 0.9", transform=ax.transAxes, ha="right", fontsize=6.5, color=INK2)
    ax.text(0.98, 0.08, "δ = 0.05", transform=ax.transAxes, ha="right", fontsize=6.5, color=INK2)
    ax.set(xscale="log", ylim=(-0.02, 1.02), xlabel="step h", ylabel="rate",
           title="(c) power (solid) and worst false\ncertification upper CI (dotted)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
    _clean(fig)
    fig.savefig(os.path.join(out, "fig2_certification_cost.pdf"))
    fig.savefig(os.path.join(out, "fig2_certification_cost.png"), dpi=200)
    plt.close(fig)


def fig3(T, out, h_show=None, arm_cfg=(20, 2, 4), frac_target=0.5):
    cs, arm = T["s2_csweep"], T["s3_arm"]
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.15), constrained_layout=True)
    ax = axes[0]
    hs = sorted(cs.h.unique())
    h = h_show if h_show in hs else (hs[0] if hs else None)
    if h is not None:
        s = cs[(cs.h == h)].sort_values("c")
        g = s[s.geo_attained]
        _line(ax, g.c, g.geo_queries, "geometric")
        ax.axhline(s.eb_queries.iloc[0], color=METHODS["eb_coupled"]["color"], lw=1.5,
                   label=METHODS["eb_coupled"]["label"])
        for xv, txt in [(s.c_cross_theory.iloc[0], "c²-crossover"), (s.c_threshold.iloc[0], "1 + 32ρ/(sΛ)")]:
            if np.isfinite(xv):
                ax.axvline(xv, color=INK2, lw=0.8, ls=(0, (3, 2)))
                ax.annotate(txt, (xv, 0.97), xycoords=("data", "axes fraction"), rotation=90, ha="right",
                            va="top", fontsize=6.5, color=INK2)
        ax.set(xscale="log", yscale="log", xlabel=r"bound inflation $c=\Lambda_{\rm declared}/\Lambda$",
               ylabel="simulator calls for 90% power", title=f"(a) declared bound, h = {h:g}")
    ax = axes[1]
    near = lambda col, v: min(arm[col].unique(), key=lambda u: abs(np.log(u / v))) if len(arm) else v
    D, r0, K0 = near("D", arm_cfg[0]), near("r", arm_cfg[1]), near("K", arm_cfg[2])
    for (m, ls_r) in [(m, None) for m in ("geometric", "eb_coupled")]:
        for r, ls in zip(sorted(arm.r.unique()), ["-", "--", ":"]):
            pts = []
            for K in sorted(arm.K.unique()):
                s = arm[(arm.D == D) & (arm.r == r) & (arm.K == K) & (arm.method == m)].sort_values("n")
                ok = s[s.gain_fraction >= frac_target]
                pts.append((K, ok.audit_calls.iloc[0] if len(ok) else np.nan))
            pts = np.array(pts, float)
            if np.isfinite(pts[:, 1]).any():
                ax.plot(pts[:, 0], pts[:, 1], ls=ls, color=METHODS[m]["color"], marker=METHODS[m]["marker"],
                        mec="white", mew=0.6, label=f"r = {r}" if m == "geometric" else None)
    ax.set(xscale="log", yscale="log", xlabel="library size K", ylabel=f"audit calls to\n{int(frac_target*100)}% of oracle gain",
           title=f"(b) arm model, D = {D}")
    ax.set_xticks(sorted(arm.K.unique()), [str(k) for k in sorted(arm.K.unique())])
    if ax.get_legend_handles_labels()[0]:
        ax.legend(title="rank (line style)", frameon=False, fontsize=6.5, title_fontsize=6.5)
    ax = axes[2]
    for m in ("geometric", "eb_coupled"):
        s = arm[(arm.D == D) & (arm.r == r0) & (arm.K == K0) & (arm.method == m)].sort_values("n")
        if len(s):
            ax.plot(s.audit_calls, s.gain_fraction, color=METHODS[m]["color"], marker=METHODS[m]["marker"], mec="white", mew=0.6)
            ax.fill_between(s.audit_calls, s.gain_fraction_lo, s.gain_fraction_hi, color=METHODS[m]["color"], alpha=0.15, lw=0)
            ax.plot(s.audit_calls, s.harmful_selection_hi, color=METHODS[m]["color"], ls=":", lw=1.2)
    ax.set(xscale="log", ylim=(-0.02, 1.02), xlabel="audit calls",
           ylabel="fraction", title="(c) gain / oracle (solid),\nharmful selection, upper CI (dotted)")
    ax.text(0.02, 0.95, f"D = {D}, r = {r0}, K = {K0}", transform=ax.transAxes, fontsize=6.5, color=INK2, va="top")
    handles = [plt.Line2D([], [], color=METHODS[m]["color"], marker=METHODS[m]["marker"], label=METHODS[m]["label"])
               for m in ("geometric", "eb_coupled")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    _clean(fig)
    fig.savefig(os.path.join(out, "fig3_smoothness_selection.pdf"))
    fig.savefig(os.path.join(out, "fig3_smoothness_selection.png"), dpi=200)
    plt.close(fig)


def appendix_figs(T, out):
    d, rm = T["s2_distance"], T["s1_rmse"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.1), constrained_layout=True)
    ax = axes[0]
    for m in ("geometric", "eb_coupled"):
        s = d[(d.method == m) & d.attained].sort_values("d")
        for D, mk in zip(sorted(s.D.unique()), ["o", "x"]):
            t = s[s.D == D]
            ax.plot(t.d, t.queries, ls="none", marker=mk, color=METHODS[m]["color"], label=f"{METHODS[m]['label']}, D={D}")
    ax.set(xscale="log", yscale="log", xlabel="projector distance d (fixed gain)", ylabel="calls for 90% power",
           title="(a) distance without information")
    ax.legend(frameon=False, fontsize=6)
    ax = axes[1]
    for kind, m in (("coupled", "eb_coupled"), ("indep", "eb_indep")):
        s = rm[rm.kind == kind].sort_values("h")
        ax.plot(s.h, s.queries_for_rel_rmse, color=METHODS[m]["color"], marker=METHODS[m]["marker"],
                label="coupled mean" if kind == "coupled" else "independent means")
        if len(s) >= 2:
            _ref_slope(ax, s.h, s.queries_for_rel_rmse.iloc[-1], s.h.iloc[-1], -2 if kind == "coupled" else -4,
                       "slope −2" if kind == "coupled" else "slope −4")
    ax.set(xscale="log", yscale="log", xlabel="step h (second-order path)", ylabel="calls for relative RMSE 0.5",
           title="(b) estimator-specific RMSE cost")
    ax.legend(frameon=False, fontsize=6.5)
    _clean(fig)
    fig.savefig(os.path.join(out, "figS_distance_rmse.pdf"))
    fig.savefig(os.path.join(out, "figS_distance_rmse.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="tables")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--h-csweep", type=float, default=0.002)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    T = {n: pd.read_csv(os.path.join(a.tables, n + ".csv")) for n in
         ["s1_budgets", "s1_false_certification", "s1_rmse", "s2_csweep", "s2_distance", "s3_arm"]}
    fig2(T, a.out)
    fig3(T, a.out, h_show=a.h_csweep)
    appendix_figs(T, a.out)
    print(f"figures written to {a.out}/")


if __name__ == "__main__":
    main()
