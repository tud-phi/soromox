# Section Vd: Control-Gain Optimization

This case compares gain optimization for collocated actuation-space control and
synergistic operational-space control. Each generator writes a MAT result under
its controller-specific directory in `data/`; the plotter combines both files.

Run the two optimizations without GUI rendering:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/control_gain_optimization_with_collocated.py \
  --result-dir /tmp/soromox-secVd/collocated --no-show --no-render
uv run python paper_results/secVd_control_gain_optimization/code/control_gain_optimization_with_synergistic.py \
  --result-dir /tmp/soromox-secVd/synergistic --no-show --no-render
```

To replace the canonical case-local data, omit `--result-dir` and pass
`--force`. Diagnostic plots are generated only with `--save-figures`; Open3D
visualization is skipped with `--no-render`.

Recreate the canonical comparison from the committed MAT files:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/plot_control_gain_optimization.py
```
