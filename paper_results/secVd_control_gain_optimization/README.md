# Section Vd: Control-Gain Optimization

This case compares single-start gain optimization for collocated
actuation-space control and synergistic operational-space control.

> [!WARNING]
> The committed archives and comparison figures are single-start results from
> the current workflow. They do not reproduce the published six-start Section
> Vd study and must not be treated as recovered paper results.

## Canonical result format

Each optimizer writes only `optimization_results.npz`. The archive uses schema
version 1 and an explicit batch dimension of one, even though the optimizer is
currently single-start. Pose vectors use

```text
[rotation-vector x, y, z, Cartesian position x, y, z].
```

`q_ts_best` is the authoritative trajectory for reconstructing the complete
robot geometry. `x_ts_best` provides a directly validated full end-effector
pose trajectory. The archive also retains iteration-aligned loss, gain,
velocity, control-input, and timing histories. Legacy archives,
`animation_data.pkl`, and `intermediate_metrics.json` are not supported.

`completed_iterations` records the actual run length and is validated against
the stored histories. `is_placeholder` identifies development-only archives;
omit `--placeholder` when producing full results.

## Optimization

The two optimization programs only run optimization and generate their NPZ:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/control_gain_optimization_with_collocated.py \
  --num-iters 100 --force
uv run python paper_results/secVd_control_gain_optimization/code/control_gain_optimization_with_synergistic.py \
  --num-iters 100 --force
```

`--num-iters` defaults to 100. Pass another positive value when intentionally
running a shorter or longer optimization.

Use `--result-dir` to stage results elsewhere. A run is saved only if every
requested iteration completed with finite data, so an interrupted or non-finite
run cannot replace an archive.

Device selection defaults to `--device auto`. The current optimizers have one
optimization start (`B=1`), so auto mode selects the CPU; the `vmap` used by the
synergistic controller over rollout timesteps does not make it a batched
optimization. The shared selector chooses the GPU when an optimizer supplies
multiple independent starts (`B>1`) in a future batched implementation. Use
`--device cpu` or `--device gpu` to override either automatic choice. The GPU
label maps to JAX's `cuda` platform name. An automatically selected GPU may fall
back to CPU with a warning, while an explicit `--device gpu` request fails if
CUDA is unavailable.

The collocated controller integrates tendon-length errors in metres and uses
the unit-preserving saturation

```text
sat(e) = tanh(gamma * e) / gamma,
gamma = 1 / e_sat.
```

The default is `e_sat = 10 mm` (`gamma = 100 1/m`) and can be changed with
`--integral-error-saturation-scale` in metres.

## Plotting

The standalone plotter is the only comparison-figure entrypoint. It requires
schema version 1, plots the single run without synthetic min/max bands, and
writes both the canonical PDF and PNG:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/plot_control_gain_optimization.py \
  --force
```

## Robot rendering

The renderer reconstructs the optimized robot from `q_ts_best`, validates it
against the stored full pose, resamples the dense rollout to the requested FPS,
and follows the paper rendering style in Viser. The solid coral body is the
current robot. Collocated control additionally shows the desired configuration
as a pale blue-gray wireframe, without task-space markers. Synergistic control
uses a translucent coral body so that the current task position in magenta and
the larger desired task position in green remain visible. A dotted trail shows
the desired path only when the reference is time-varying; the animation does
not reveal the robot's future actual trajectory. It writes MP4 files and, when
requested, GIF previews:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/render_control_gain_optimization_animations.py \
  --method both --fps 24 --gif --force
```

Use `--method collocated` or `--method synergistic` to render one controller.
The CLI also exposes the Viser port/browser and recording timeout settings.

## Version-control policy

Canonical comparison figures (`plot_control_gain_opt.pdf` and `.png`) are kept
in normal Git. Canonical optimization archives and robot-rendering MP4/GIF
files are also versioned, but routed through Git LFS because full optimization
and rendering runs can make them substantially larger. Directories matching
`outputs/*_diagnostics/` are local scratch artifacts from the removed embedded
diagnostic workflow and are intentionally ignored.
