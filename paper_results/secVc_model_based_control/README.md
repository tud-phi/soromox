# Section Vc: Model-Based Control

Section Vc is reproduced by two independent cases:

- [`configuration_space_comparison/`](configuration_space_comparison/README.md)
  compares setpoint regulation, trajectory tracking, and regulation followed by
  tracking.
- [`operational_space_impedance_control/`](operational_space_impedance_control/README.md)
  evaluates operational-space impedance tracking with partial feedback
  linearization. A full-versus-partial comparison remains available as a
  standalone diagnostic.

Run commands from the repository root after installing paper dependencies:

```bash
uv sync --extra paper_results
```

Each generator refuses to overwrite data unless `--force` is passed. Plotters
and renderers consume saved NPZ data and write only to their case `outputs/`
directory by default. The case READMEs list complete reproduction commands and
artifact inventories.

Compose the full Section Vc application figure from the canonical comparison
data and operational snapshots with:

```bash
uv run --extra paper_results python paper_results/secVc_model_based_control/code/plot_model_based_control.py
```

The command writes `outputs/model_based_control.pdf` and a matching SVG; pass
`--force` to replace existing outputs.

## Evaluation metrics

Generate the complete human-readable report and its machine-readable JSON from
the committed trajectory artifacts without rerunning either simulation:

```bash
uv run --extra paper_results python paper_results/secVc_model_based_control/code/report_metrics.py --json-output
```

The configuration-space report contains component-wise RMSE for all six strain
coordinates in the regulation and tracking phases. For regulation it also
reports the mean absolute error over the final 10% of each post-change setpoint
interval, first per setpoint and then as an equal-weight mean over the four
setpoints. The initial zero-reference interval is excluded. The operational
report contains Euclidean position and geodesic orientation RMSE, the maximum
pairwise reference span for each task quantity, and RMSE normalized by that
span in percent. The command writes `data/metrics.json` by default; an explicit
path may be supplied after `--json-output`.

The committed Section Vc paper figures are:

- `outputs/model_based_control.pdf`
- `configuration_space_comparison/outputs/configuration_space_comparison.pdf`
- `configuration_space_comparison/outputs/regulation_tracking_rmse.pdf`
- `operational_space_impedance_control/outputs/operational_space_impedance_control.pdf`

All use the shared `paper_results/paper.mplstyle` publication settings and the
regenerated Section Vc trajectory data.
