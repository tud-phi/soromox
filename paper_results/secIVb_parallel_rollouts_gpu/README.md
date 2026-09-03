# Section IVb: Parallel GPU Rollouts

This case measures per-environment simulation speed and aggregate throughput as
the model size and number of parallel environments increase. Timings depend on
the JAX backend and accelerator, so the committed CSV files record the paper
machine's results and should not be expected to match on different hardware.

The parallel-rollout generator reuses the model registry and timing helpers from
`tools/benchmarks/_benchmark_common.py`; the publication plotter remains
case-local so it can apply the common paper style.

GPU execution is the default. If JAX cannot initialize a compatible GPU, the
generator warns and exits without writing CPU timings under a GPU configuration.
Pass `--device cpu` explicitly to benchmark the same workload on the CPU.

Every generator run writes `data/benchmark_results.csv` by default. Pass
`--csv PATH` to override the destination. The CSV records the source revision
and dirty state, software versions, FP64 mode, UTC timestamp, and
accelerator/runtime identity needed to interpret results across machines. Use
`--backend jax` or
`--backend warp` to select the dynamics implementation for GVS, PCS, and
PlanarPCS. Run each backend in a separate process for a controlled comparison;
JAX-only systems continue to use JAX. The CSV records the requested backend,
the backend resolved for each system, and whether backend selection applies.

Each `(system, size, Gauss-point setting, batch size)` case runs in a fresh
process. Successful rows are checkpointed to the results CSV immediately, so
an out-of-memory failure cannot discard earlier measurements. Failed cases are
skipped and summarized after the remaining cases finish; the machine-readable
report defaults to `data/benchmark_results_failures.csv`. Pass
`--failures-csv PATH` to choose another location. Expected OOM-only failures do
not make the generator exit unsuccessfully, while unexpected worker errors do.

Use `--solver` to select `euler`, `semi-implicit-euler`, `heun`, `bosh3`,
`tsit5`, or `dopri5`; the default remains `tsit5`. The selected solver applies
to every requested system and model size. The CSV records the solver together
with `rollout_finite` and `max_abs_state`, so throughput measurements can be
screened for numerical divergence rather than interpreted on timing alone.

Install dependencies and confirm that JAX sees the intended GPU:

```bash
uv sync --extra paper_results
uv run python -c "import jax; print(jax.devices())"
```

Generate the segment-scaling CSV with the paper integration settings:

```bash
uv run python paper_results/secIVb_parallel_rollouts_gpu/code/generate_parallel_rollout_benchmark.py \
  --systems articulated_soft_robot planar_pcs pcs gvs \
  --gvs-scaling segments --segment-counts 1 2 4 8 16 32 \
  --batch-sizes 1 2 4 8 16 32 64 128 256 \
  --duration 1 --solver tsit5 --solver-dt 1e-4 --save-dt 1e-2
```

Generate the GVS strain-basis-order sweep:

```bash
uv run python paper_results/secIVb_parallel_rollouts_gpu/code/generate_parallel_rollout_benchmark.py \
  --systems gvs --gvs-scaling basis-order --gvs-basis-orders 0 1 2 3 4 5 \
  --batch-sizes 1 2 4 8 16 32 64 128 256 \
  --duration 1 --solver-dt 1e-4 --save-dt 1e-2 \
  --csv paper_results/secIVb_parallel_rollouts_gpu/data/gvs_basis_order_1s.csv
```

Regenerate the canonical segment-scaling figures from the committed CSVs:

```bash
uv run python paper_results/secIVb_parallel_rollouts_gpu/code/plot_parallel_rollout_benchmark.py \
  --systems articulated_soft_robot planar_pcs pcs gvs \
  --segment-counts 1 2 4 8 16 32
```

Use an explicit temporary `--csv` path for validation runs to avoid replacing
the committed hardware measurements.
