# Paper Results

This directory maps paper sections to the code, data, and canonical outputs used
to produce the reported results. Each section keeps scripts under `code/`, source
or generated artifacts under `data/`, and publication figures/videos under
`outputs/`.

| Paper section | Directory | Reproduced artifacts |
| --- | --- | --- |
| IVa, sequential CPU benchmarks | [`secIVa_benchmarking_sequential_cpu/`](secIVa_benchmarking_sequential_cpu/README.md) | SoRoMoX, SoRoSim, and PyElastica rollout data and comparisons |
| IVb, parallel GPU rollouts | [`secIVb_parallel_rollouts_gpu/`](secIVb_parallel_rollouts_gpu/README.md) | GPU timing tables and batch-scaling figures |
| Va, system identification | [`secVa_system_identification/`](secVa_system_identification/README.md) | parameter/residual data and shape, error, and RMSE figures |
| Vc, model-based control | [`secVc_model_based_control/`](secVc_model_based_control/README.md) | standardized configuration- and operational-space figures |
| Vd, control-gain optimization | [`secVd_control_gain_optimization/`](secVd_control_gain_optimization/README.md) | collocated/synergistic optimization data and comparison |
| Ve, safety-constrained control | `secVe_safety_constrained_control/` | rollout data, plots, and videos |
| Vf, parallel reinforcement learning | [`secVf_parallel_rl/`](secVf_parallel_rl/README.md) | training, checkpoints, reward curve, rollout data, and media |

`final_outputs/` contains composed figures assembled from one or more section
outputs. Trained model checkpoints are stored with Git LFS; section data and
paper figures remain regular Git objects unless their size requires otherwise.

The model-based-control application composite is regenerated directly from the
canonical Section Vc data and operational snapshots. Control-gain optimization
remains a separate Section Vd figure:

```bash
uv run --extra paper_results python paper_results/secVc_model_based_control/code/plot_model_based_control.py
```

The command writes `model_based_control.pdf` and a matching SVG under the
Section Vc `outputs/` directory, and requires `--force` before replacing either
existing output.

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
