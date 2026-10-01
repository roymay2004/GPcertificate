# Experiments for "Certifying Projection Repairs: The Value of Smoothness Information"

Code for Section 5 (Experimental design and protocol). Every simulator has exact projection
risks, so the sign and size of every gain is known without a reference estimate.

## Layout

| Path | Contents |
|---|---|
| `prcert/simulators.py` | Sum-of-sines simulators with exact `R_f(P)` (App. B): sine ridge, planar arm |
| `prcert/geometry.py` | Projector distance `d`, principal-angle quantity `s`, the sine paths, equal-risk and harmful moves, inactive-coordinate rotations, Grassmann geodesics |
| `prcert/sampling.py` | Coupled rows (`K + 2` calls) and independent-loss rows (4 calls), streamed into statistics at every budget on a grid |
| `prcert/audits.py` | Geometric and empirical Bernstein radii (Prop. 2, App. A.3), certification, safe selection (Sec. 4) |
| `prcert/library.py` | Arm study: learned baseline and candidate library from development gradients |
| `experiments/plan.py` | Builds every task, sizes them to about one core-hour, prints the compute estimate |
| `experiments/run_task.py` | Runs one task (SLURM array element) or all tasks locally |
| `analysis/aggregate.py` | Budget selection on batch A, evaluation on batch B, slopes, false certification, all CSV tables, `table_arm.tex` |
| `analysis/figures.py` | Figures 2 and 3 and an appendix figure |
| `slurm/` | Narval environment setup, job-array submission, analysis job |
| `tests/test_core.py` | Checks against exact formulas (risk, gain, variance limit, reduction, radii) |

## Mapping to the paper

| Paper | Code | Output |
|---|---|---|
| Sec. 5.1 cost at fixed confidence and power, both sine paths | `s1` tasks, kinds `coupled`/`indep`, moves `favorable`/`harmful`/`equal` | `s1_budgets.csv`, `s1_slopes.csv`, `s1_false_certification.csv`, Fig. 2 |
| Sec. 5.1 relative-RMSE comparison (App. B) | `rmse` tasks | `s1_rmse.csv`, `figS_distance_rmse` (b) |
| Sec. 5.2 declared-bound sweep, Prop. 3 reference scales | reuses the first-order `s1` coupled rows | `s2_csweep.csv`, Fig. 3 (a) |
| Sec. 5.2 inactive coordinates and controlled-rank rotations | `s2dist` tasks | `s2_distance.csv`, `figS_distance_rmse` (a) |
| Sec. 5.3 kinematic arm, safe selection | `s3arm` tasks | `s3_arm.csv`, `s3_candidate_classes.csv`, `table_arm.tex`, Fig. 3 (b, c) |

## Protocol details implemented

- **Fixed-size audits only.** Each repetition streams rows once; statistics are recorded at
  every grid budget from that repetition's first `n` rows. Each budget is a separate
  fixed-size audit; no decision combines budgets, so there is no optional stopping.
- **Budget selection.** On batch A, the power-attaining budget is the smallest grid budget
  whose isotonic power estimate reaches 0.90. Power is then re-estimated on fresh batch-B
  repetitions at that budget, with Clopper–Pearson intervals. Slope intervals come from a
  bootstrap over batch-A repetitions.
- **False certification.** Harmful (reversed) and equal-risk moves at every grid budget,
  batch B. The first-order equal-risk move rotates the candidate about the active direction,
  so its gain is exactly zero at distance of order `h`.
- **Declared information.** Sine studies declare `I = [-2, 2]` (`B = 4`) and `Λ = 1`; the
  declared-bound sweep uses `Λ_declared = c Λ` with `c` from 1 to 4096. The arm study
  declares `I = [-1, 1]` (`B = 2`) and the exact Lipschitz bound `Σ c_j |a_j|`.
- **Exact dimension reduction.** All audit quantities depend on the Gaussian inputs only
  through their projection onto `span(A, U0, U1, ..., UK)`. Sampling that projection is
  exact in distribution, so added inactive coordinates leave the cost unchanged.
- **Call accounting.** A coupled row costs `K + 2` simulator calls, an independent-loss row 4.
  Development gradients in the arm study cost one call each (`grad_cost`; set it to `D + 1`
  for finite differences). Development and audit calls are reported separately and summed.
- **Reproducibility.** Every task's random stream is `SeedSequence([seed, study, config,
  batch, block])`. Each result file stores its task, configuration, and run time, and
  finished tasks are never recomputed.

## Compute

The full plan (`python -m experiments.plan --scale full`) has 147 configurations and about
1,700 tasks. Its estimate on a single core of the development machine is about **1,500
core-hours, roughly 0.3 of one day of a 200 CPU-day/day allocation**:

| Study | Tasks | Core-hours |
|---|---|---|
| s1 (both sine paths, three moves, coupled and independent) | ~880 | ~760 |
| s2dist | ~70 | ~40 |
| s3arm (27 configurations, 40 libraries × 25 audits, up to 2^24 rows) | ~730 | ~700 |
| rmse | 14 | <1 |

The planner measures throughput where it runs. Pass `--speed 1.3` if compute nodes are
faster than the planning host; this only changes how repetitions are split into tasks.
Two independent-loss configurations on the second-order path (`h = 0.02, 0.01`) need about
`h^-4` rows. They hit the row cap of `2^30` and are reported as "not attained within cap".

## Running

Local end-to-end check (about 3 minutes):

```bash
bash run_smoke.sh
```

On Narval:

```bash
bash slurm/setup_env.sh                       # once
sed -i 's/def-CHANGEME/def-YOURPI/' slurm/array.sbatch slurm/analyze.sbatch
module load StdEnv/2023 python/3.11 && source ~/envs/prcert/bin/activate
python -m tests.test_core
python -m experiments.plan --scale full --out plan_full.json
bash slurm/submit.sh plan_full.json $SCRATCH/prcert-results
# after the arrays finish (rerun submit.sh to fill any gaps; finished tasks are skipped):
sbatch --export=ALL,PLAN=plan_full.json,RESULTS=$SCRATCH/prcert-results slurm/analyze.sbatch
```

Keep `plan_full.json` with the results: the plan, together with the seed, determines every
random stream.

## Expected qualitative outcomes (from the smoke run and the planner's predictions)

- **First-order path.** The geometric audit needs about 1.2e4 rows at every `h`. Coupled EB
  grows like `h^-1`, and independent EB like `h^-2`. The geometric advantage therefore
  appears only for `h ≲ 0.02`.
- **Second-order path.** Geometric and coupled EB both scale like `h^-2`, but coupled EB is
  about 20× cheaper. Its variance term uses the realized variance, about `0.37 d^2`, whereas
  the geometric radius uses the worst-case `B Λ d`.
- **Arm model.** Range-only selection certifies useful repairs with roughly 1e4–1e5 rows. The
  geometric radius needs millions, because these repairs are not local and `64 B²Λ²` is
  large.

These are statements about constants at accessible step sizes. They are consistent with the
theory, which is asymptotic in `h`.
