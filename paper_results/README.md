# Paper Results

This directory maps paper sections to the code, data, and canonical outputs used
to produce the reported results. Each section keeps scripts under `code/`, source
or generated artifacts under `data/`, and publication figures/videos under
`outputs/`.

The latest end-to-end validation results and unresolved reproducibility gaps are
recorded in [`REPRODUCTION_REPORT.md`](REPRODUCTION_REPORT.md).

| Paper section | Directory | Reproduced artifacts |
| --- | --- | --- |
| IVa, sequential CPU benchmarks | `secIVa_benchmarking_sequential_cpu/` | rollout comparison figures |
| IVb, parallel GPU rollouts | `secIVb_parallel_rollouts_gpu/` | batch-scaling figures |
| Va, system identification | `secVa_system_identification/` | shape, error, and RMSE figures |
| Vc, model-based control | [`secVc_model_based_control/`](secVc_model_based_control/README.md) | configuration- and operational-space comparisons |
| Vd, control-gain optimization | `secVd_control_gain_optimization/` | optimized-gain comparison |
| Ve, safety-constrained control | `secVe_safety_constrained_control/` | rollout data, plots, and videos |
| Vf, parallel reinforcement learning | `secVf_parallel_rl/` | reward curve and rollout media |

`final_outputs/` contains composed figures assembled from one or more section
outputs. Section Vc binary data and canonical outputs are stored with Git LFS.
Hydrate them before plotting with `git lfs pull`.

## Shared conventions

The standardized Matplotlib plotters use the base publication settings in
[`paper.mplstyle`](paper.mplstyle). Plotters apply that file first, then keep
figure-specific font sizes, DPI, padding, and layout settings next to the figure
code. This centralizes typography, grids, legends, spines, and line defaults
without forcing differently sized paper panels to use one font scale.

Artifact paths remain case-local. Entry points derive their case directory from
the script location and default to that case's `data/` and `outputs/`
directories, so commands work from any current working directory. Shared schemas
and filename rules within a case live in its common module.
