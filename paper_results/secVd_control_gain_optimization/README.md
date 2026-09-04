# Section Vd: Control-Gain Optimization

This case optimizes PID gains by differentiating through a closed-loop rollout,
and compares two controllers on the same one-segment PCS robot: **collocated**
actuation-space control, and **synergistic** operational-space control. Each
optimization runs six independent gain initializations at once and the committed
archives hold the full histories.

## Method

**Plant.** One PCS segment with three tendons.

**Objective.** Both controllers import one definition from
[`code/secvd_objective.py`](code/secvd_objective.py), so "Loss" means the same
thing in both panels of the figure:

```text
L = (1 / T) * integral_0^T ||e_hat(t)||^2 dt

collocated   e_hat = [L_seg * e_kappa ; e_sigma]     (strains)
synergistic  e_hat = [e_theta ; e_x / L_seg]         (pose)
```

Both sides are dimensionless.

**Optimizer.** One `scale_by_yogi` whose update is rescaled per gain family,
built in `secvd_case.build_optimizer`. No gradient clipping: divergence here is
a step-size effect, since Yogi's update magnitude is set by the learning rate
almost independently of the gradient.

**Initialization.** Each gain's nominal vector is scaled by one log-uniform
factor per start over `[1/spread, spread]`.

Tuned values, all defined once in
[`code/secvd_case.py`](code/secvd_case.py) and used as CLI defaults:

| constant | collocated | synergistic |
| --- | --- | --- |
| `TUNED_ALPHA` | `1500` | `1547` |
| `TUNED_INIT_SEED` | `2` | `0` |
| `COMMITTED_LR_RATIO` (P/I/D) | `1.0 / 0.5 / 0.1` | same |
| `TUNED_SOLVER_DT` | `1e-3` | same |
| `COMMITTED_E_SAT` | `10 mm` | not applicable |

`TUNED_ALPHA` is held at or below a quarter of the rate that diverges.
`COMMITTED_E_SAT` bounds the integral state through `tanh(e/e_sat) * e_sat`.

## Running

```bash
uv sync --extra paper_results
```

One program runs both optimizations, selected by `--method`, so the two cannot
drift apart:

```bash
G=paper_results/secVd_control_gain_optimization/code/optimize_control_gains.py
uv run python $G --method collocated --num-iters 100 --force
uv run python $G --method synergistic --num-iters 100 --force
```

Defaults reproduce what ships. A run is saved only if every requested iteration
completed and every start produced a finite iterate, so an interrupted run
cannot replace an archive; `--result-dir` stages results elsewhere. `--device`
takes `auto`, `cpu` or `gpu`, and pinning one makes the run fail rather than
silently start on another backend.

Redraw the four-panel comparison figure from the committed archives:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/plot_control_gain_optimization.py
```

The plotter overwrites its outputs by default (`--no-force` refuses), since
redrawing costs seconds while a generator overwrite destroys an optimization.
Panels A and C share one loss axis; `--ylim-a` through `--ylim-d` take no values
to autoscale one panel to its own data, or `LOW HIGH` to set it by hand.

Render the optimized robot from `q_ts_best`, validating every stored pose
against forward kinematics first:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/render_control_gain_optimization_animations.py \
  --method both --fps 24 --gif --force
```

[`code/secvd_tuning_study.py`](code/secvd_tuning_study.py) is a study, not a
results generator: it writes small summaries to the git-ignored `data/tuning/`
and builds its problem and optimizer through the same two calls the generator
makes, so a setting it measures is measured on what actually runs. Flag names
match the generator's, so a recommendation is pasted across unchanged. Stages are
`solver-dt`, `learning-rate` and `saturation`.

```bash
uv run python paper_results/secVd_control_gain_optimization/code/secvd_tuning_study.py \
  --method collocated --stage learning-rate
```

## Archive format

Each optimizer writes only `optimization_results.npz`, schema version 3. The
batch axis of width `B` appears **only** on quantities that vary per start, so a
batch axis in this archive always means something:

| field | shape |
| --- | --- |
| `history_loss`, `history_finite_mask` | `(iterations, B)` |
| `history_Kp`, `history_Ki`, `history_Kd` | `(iterations, B, m)` |
| `init_Kp`, `init_Ki`, `init_Kd` | `(B, m)` |
| `q_ts_init/best`, `qd_ts_init/best` | `(B, timesteps, dofs)` |
| `u_ts_init/best` | `(B, timesteps, actuators)` |
| `x_ts_init/best` | `(B, timesteps, 6)` |
| `t_ts`, `q_des_ts`, `x_des_ts` | shared by every start, no batch axis |

Pose vectors are `[rotation-vector x, y, z, position x, y, z]`. Per-iteration
trajectories are absent by design -- they would reach several gigabytes -- so
each start keeps its initial and best rollout, with complete loss and gain
histories.

The archive also records what produced it (`init_seed`, `init_scheme`,
`init_spread`, `objective_name`, `objective_scales`, `saturation_config`,
`optimizer_metadata`), enough for `recompute_archive_losses` to re-derive the
stored losses from the archived trajectories alone. A start that becomes
non-finite is frozen rather than aborting the run, and `history_finite_mask`
marks which entries are real. Selection is re-checked on load, always from that
archive's own losses. Older formats are rejected rather than partially read.

## Version control

Comparison figures are kept in normal Git. Optimization archives and rendered
MP4/GIF files are versioned through Git LFS, since full runs make them large.
`outputs/*_diagnostics/` is local scratch and is ignored.
