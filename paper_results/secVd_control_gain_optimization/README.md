# Section Vd: Control-Gain Optimization

This case compares gain optimization for collocated actuation-space control and
synergistic operational-space control. Each generator writes a MAT result under
its controller-specific directory in `data/`; the plotter combines both files.

> [!WARNING]
> The committed Section Vd MAT files and derived plots are stale. In particular,
> the collocated results were generated with the former, non-unit-preserving
> integral-error saturation and the former `gamma=10` setting. Do not treat the
> current files as canonical results. Regenerate the complete Section Vd results
> only after the saturation-scale change has been merged and the other open
> control-gain-optimization issues (#128 and #129) have been fixed.

## Collocated integral-error saturation

The collocated controller integrates tendon-length errors, all expressed in
meters. It uses the unit-preserving saturation

```text
sat(e) = tanh(gamma * e) / gamma,
gamma = 1 / e_sat.
```

The committed legacy trajectories were inspected to select a physical error
scale. The undeformed tendon lengths are approximately `-100 mm`, and the
setpoints are `[-85.49, -96.41, -100.05] mm`. This produces initial errors of
`[14.51, 3.59, -0.05] mm`. Across the saved initial and optimized trajectories,
all other initial/transient absolute errors remain below `3.85 mm`.

The default is therefore `e_sat = 10 mm`, or `gamma = 100 1/m`. At the exceptional
`14.51 mm` reference step, saturation reduces the integral-error derivative by
about 38%. At `3.85 mm`, the reduction is only about 4.7%, keeping normal
closed-loop errors nearly linear while limiting integral accumulation during the
large step. Override the physical scale, if needed, with
`--integral-error-saturation-scale` in meters.

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
