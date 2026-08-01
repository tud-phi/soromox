# Section Vc: Model-Based Control

Section Vc is reproduced by two independent cases:

- [`configuration_space_comparison/`](configuration_space_comparison/README.md)
  compares setpoint regulation, trajectory tracking, and regulation followed by
  tracking.
- [`operational_space_impedance_control/`](operational_space_impedance_control/README.md)
  evaluates operational-space impedance tracking and full versus partial
  feedback linearization.

Run commands from the repository root after installing paper dependencies:

```bash
uv sync --extra paper_results
git lfs pull
```

Each generator refuses to overwrite data unless `--force` is passed. Plotters
and renderers consume saved NPZ data and write only to their case `outputs/`
directory by default. The case READMEs list complete reproduction commands and
artifact inventories.

The two canonical paper figures are:

- `configuration_space_comparison/outputs/configuration_space_comparison.pdf`
- `operational_space_impedance_control/outputs/operational_space_impedance_control.pdf`

Both use the shared `paper_results/paper.mplstyle` publication settings and the
regenerated Section Vc trajectory data.
