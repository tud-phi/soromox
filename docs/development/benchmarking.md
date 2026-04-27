# Soromox Benchmarking Toolkit

Soromox ships two complementary benchmarking CLIs under `tools/benchmarks`:

- `benchmark_system_methods.py` profiles individual model routines (forward kinematics,
  dynamics, etc.) to track JIT compile and steady-state execution costs.
- `benchmark_simulation_batch_scaling.py` measures how simulation throughput scales as
  you increase the number of batched environments (e.g., via `jax.vmap`) and reports
  the simulated-time / wall-time ratio.

Both scripts share the same system registry and integration defaults, so adding a new
robot once makes it accessible throughout the benchmarking suite.

## Prerequisites

- Activate the `soromox` environment (or otherwise ensure the package is on `PYTHONPATH`).
- Install JAX (CPU or GPU build), Matplotlib, Seaborn (optional, for nicer plots),
  and any accelerator-specific drivers. GPU users should follow the
  [official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## Benchmarking individual system methods

The `benchmark_system_methods.py` CLI times a range of core routines for a sweep of
system sizes. Each measurement captures a cold call (compile + first execution) and
averages a number of warm calls to estimate steady-state latency.

```bash
python tools/benchmarks/benchmark_system_methods.py \
  --systems articulated_soft_robot pendulum planar_pcs pcs \
  --segment-counts 1 3 5 7 \
  --duration 2.0 \
  --solver-dt 5e-4 \
  --execution-repeats 5 \
  --csv benchmarks/methods.csv \
  --plot benchmarks/methods.png
```

### Key options

- `--systems`: subset of available robots (defaults to all registered systems).
- `--segment-counts`: link/segment sweep; a fresh system instance is created per value.
- `--duration`, `--solver-dt` (`--dt` alias), `--save-dt`: integration controls when benchmarking
  `rollout_to`.
- `--execution-repeats`: number of warm calls to average after the cold run.
- `--json` / `--csv`: export raw results for regression tracking.
- `--plot` / `--show-plot`: render Matplotlib summaries (compile vs. exec time).

### Interpreting the results

For each system/function pair the script prints:

1. Cold-call latency (compile + execution), synchronised via `block_until_ready()`.
2. Mean warm-call latency, reflecting the steady-state cost once XLA caches the
   executable.
3. Derived compile time = cold − warm. Pure-Python methods show near-zero compile
   time but still track runtime cost.

The plots group results by system, highlighting how complexity evolves with segment
count for each tracked method.

For `articulated_soft_robot`, the method benchmark includes both the default
articulated-body forward dynamics path and a dense Jacobian-energy forward
dynamics solve (`forward_dynamics_dense`). Comparing these two cases is useful
for tracking ABA performance against the controller-facing dense dynamics API.

## Benchmarking simulation batch scaling

`benchmark_simulation_batch_scaling.py` runs full simulations in parallel batches
using `jax.vmap`. It records both the per-environment speed
(`simulated_time / wall_time`) and the aggregate throughput
(`num_envs * simulated_time / wall_time`), so you can see whether each environment is
running faster than real-time *and* how many simulated seconds are produced per wall
second. Each configuration can export tables and plots spanning multiple segment counts.

```bash
python tools/benchmarks/benchmark_simulation_batch_scaling.py \
  --systems articulated_soft_robot pcs planar_pcs \
  --segment-counts 1 3 5 \
  --batch-sizes 1 2 4 8 16 32 64 \
  --duration 2.0 \
  --solver-dt 5e-4 \
  --csv benchmarks/batch-scaling.csv \
  --plot benchmarks/batch-scaling.png \
  --log-x --log-y
```

### Key options

- `--batch-sizes`: number of environments to launch per measurement.
- Shared `--systems`, `--segment-counts`, `--duration`, `--solver-dt` (`--dt` alias), `--save-dt`.
- `--noise-scale`: per-environment perturbation applied to `q`/`qd` to avoid feeding
  identical states to all replicas (helps stress vectorisation paths).
- `--repeats`, `--warmup-runs`: control timing stability.
- `--csv`, `--npz`, `--plot`, `--show-plot`, `--log-x`, `--log-y`: artifact and
  visualisation controls.

### Output and interpretation

For each combination of system, segment count, and number of environments the script reports:

- Wall-clock time averaged over the requested repeats.
- Mean simulated time per environment (i.e., the final timestamp returned by
  `rollout_to`).
- Per-environment speed ratio `simulated_time / wall_time`, total throughput
  `number_of_environments * simulated_time / wall_time`, and per-environment wall time. Ratios > 1
  indicate faster-than-real-time performance.

Plots now show two stacked panels per system: the top tracks per-environment
speed-up, while the bottom tracks aggregate throughput (total simulated time per
wall second). Each line corresponds to a segment count so scaling trends remain
easy to compare, with the horizontal axis representing the number of environments.

## Visualising existing runs

To revisit stored measurements (JSON or CSV) without re-running the benchmarks, use:

```bash
python tools/benchmarks/visualize_benchmarks.py benchmarks/methods.json \
  --systems planar_pcs pcs \
  --functions rollout_to forward_dynamics \
  --output benchmarks/methods-focus.png
```

The helper mirrors the plotting style of `benchmark_system_methods.py` and accepts
optional filters for systems/functions. Pass `--show` to open a window interactively.

## Extending the registry

Both CLIs share `tools/benchmarks/_benchmark_common.py`. To add a new system:

1. Implement a factory that builds the system for a requested segment/link count.
2. Provide a context builder returning representative `q`, `qd`, `u`, `tau_ext`, and
   any auxiliary data required by the benchmarking cases.
3. Register the new entry in the shared registry so it automatically becomes available
   to every benchmarking tool.

Following this pattern keeps benchmarks for future robot models (e.g., new
articulated variants or GVS variants) consistent and easy to maintain.
