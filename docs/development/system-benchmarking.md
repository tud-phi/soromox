# System Benchmarking Toolkit

The `tools/benchmarks/run_system_benchmarks.py` script provides a reproducible way to
measure how long the core system routines take to compile with JAX and to execute once
they are compiled. It currently targets the `Pendulum`, `PlanarPCS`, and `PCS` systems
and is designed so that additional backends (for example `GVS`) can be plugged in with
minimal boilerplate.

## Prerequisites

- Python environment with Soromox available on the module path (running from the
  repository root is sufficient).
- JAX and Matplotlib installed. If you are using CPU only you can install the default
  wheels via:

  ```bash
  pip install jax matplotlib
  ```

  GPU users should follow the [official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## Running the benchmark suite

The CLI can be executed directly from the repository root:

```bash
python tools/benchmarks/run_system_benchmarks.py \
  --systems pendulum planar_pcs pcs \
  --segment-counts 1 3 5 7 \
  --duration 2.0 \
  --dt 5e-4 \
  --execution-repeats 5 \
  --csv benchmarks/latest.csv \
  --plot benchmarks/latest.png
```

Key options:

- `--systems`: subset of system models to benchmark (defaults to all configured
  systems).
- `--segment-counts`: sweep of link or segment counts. The script recreates the
  system for each entry so each configuration starts from a cold JIT cache.
- `--duration`, `--dt`, `--save-every`: integration settings for `resolve_upon_time`.
- `--execution-repeats`: how many post-compilation executions to average for the
  steady-state timing.
- `--json` / `--csv`: optional export paths for the raw timing table.
- `--plot` / `--show-plot`: create a Matplotlib overview of compile and execution
  trends.

All timings are printed to stdout and stored alongside metadata about the run so
regressions can be compared across commits or machines.

## Understanding the measurements

For each system/function pair the tool performs:

1. A cold-call measurement (`compile + execute`), synchronising via
   `block_until_ready()` to avoid under-reporting asynchronous JAX kernels.
2. A configurable number of warm calls whose mean reports the steady-state execution
   time once XLA has cached the compiled executable.

The reported compilation time is the difference between the cold-call latency and the
mean warm-call latency. Pure Python functions (such as `resolve_upon_time`) therefore
show a compilation time close to zero while still tracking their execution cost.

## Visualising results

When `--plot` or `--show-plot` is supplied the script will produce a
system-by-system grid of line plots, contrasting compilation and execution times across
segment counts for every tracked method. The saved figure (PNG by default) is suitable
for attaching to pull requests or slide decks.

### Visualising existing data

Previously recorded runs can be explored without re-executing the benchmarks via:

```bash
python tools/benchmarks/visualize_benchmarks.py benchmarks/latest.json \
  --systems planar_pcs pcs \
  --functions resolve_upon_time forward_dynamics \
  --output benchmarks/latest-focus.png
```

The helper accepts either the JSON or CSV export, optional filters for systems and
functions, and produces a figure using the same layout as the main benchmarking CLI.
Pass `--show` to open the plot interactively.

## Extending to new systems

New systems can be added by extending `_build_system_registry()` inside the CLI:

1. Define a factory that builds the system for a given segment count.
2. Provide a context builder that prepares representative `q`, `qd`, actuation, and
   spatial evaluation points.
3. Add the benchmark cases that should be tracked. The reusable helpers already
   measure compile and execution latency.

Following the existing patterns keeps all DevOps automation aligned and makes it easy
for future work (for example benchmarking `GVS`).
