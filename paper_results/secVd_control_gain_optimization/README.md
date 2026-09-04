# Section Vd: Control-Gain Optimization

This case optimizes PID gains through closed-loop rollouts for collocated
actuation-space and synergistic operational-space control of a one-segment PCS
robot. Each method evaluates six gain initializations in parallel.

## Method

Both controllers minimize the dimensionless objective

```text
L = (1 / T) * integral_0^T ||e_hat(t)||^2 dt

collocated   e_hat = [L_seg * e_kappa ; e_sigma]
synergistic  e_hat = [e_theta ; e_x / L_seg]
```

The optimizer uses Yogi with per-gain-family learning-rate multipliers. Initial
gains are sampled from a log-uniform distribution around the nominal values.

| Setting | Collocated | Synergistic |
| --- | ---: | ---: |
| Learning rate | `1500` | `1547` |
| Initialization seed | `2` | `0` |
| P/I/D rate multipliers | `1.0 / 0.5 / 0.1` | `1.0 / 0.5 / 0.1` |
| Solver step | `1e-3` | `1e-3` |
| Integral-error saturation | `10 mm` | n/a |

The collocated controller applies `tanh(e / e_sat) * e_sat` to the error before
integration.

## Run

Install the paper-results dependencies:

```bash
uv sync --extra paper_results
```

Run either optimization:

```bash
G=paper_results/secVd_control_gain_optimization/code/optimize_control_gains.py
uv run python $G --method collocated --num-iters 100 --force
uv run python $G --method synergistic --num-iters 100 --force
```

`--device auto` defers to JAX; `--device cpu` and `--device gpu` are strict.
Use `--result-dir` to stage results outside the committed data directory.

Regenerate the comparison figure:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/plot_control_gain_optimization.py
```

Panels A and C share a loss axis. `--ylim-a` through `--ylim-d` accept no values
for per-panel autoscaling or `LOW HIGH` for explicit limits.

Render the optimized rollouts:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/render_control_gain_optimization_animations.py \
  --method both --fps 24 --gif --force
```

Use `--trajectory initial-median` for the finite initial rollout closest to the
median initial loss; the default is `optimized-best`. Initial-median files use
the `_initial_median_tracking` suffix; best files use `_optimized_best_tracking`.
All renderings share one gravity-down camera.

Run a solver-step, learning-rate, or saturation study:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/secvd_tuning_study.py \
  --method collocated --stage learning-rate
```

Tuning summaries are written to the ignored `data/tuning/` directory.

## Result archive

Each method writes `optimization_results.npz` using schema version 3.

| Field | Shape |
| --- | --- |
| `history_loss`, `history_finite_mask` | `(iterations, B)` |
| `history_Kp`, `history_Ki`, `history_Kd` | `(iterations, B, m)` |
| `init_Kp`, `init_Ki`, `init_Kd` | `(B, m)` |
| `q_ts_init/best`, `qd_ts_init/best` | `(B, timesteps, dofs)` |
| `u_ts_init/best` | `(B, timesteps, actuators)` |
| `x_ts_init/best` | `(B, timesteps, 6)` |
| `t_ts`, `q_des_ts`, `x_des_ts` | shared time-series arrays |

Pose vectors contain rotation-vector xyz followed by Cartesian xyz. Archives
also record the initialization, objective scales, saturation, solver, material,
and optimizer metadata needed to validate and rescore the stored rollouts.

Optimization archives and rendered MP4/GIF files use Git LFS. Comparison
figures remain in normal Git.
