# Section IVb: Parallel GPU Rollouts

This case measures per-environment simulation speed and aggregate throughput as
the model size and number of parallel environments increase. Timings depend on
the JAX backend and accelerator, so the committed CSV files record the paper
machine's results and should not be expected to match on different hardware.

The data generator reuses the model registry and timing helpers from
`tools/benchmarks/_benchmark_common.py`; the publication plotter remains
case-local so it can apply the common paper style.

Install dependencies and confirm that JAX sees the intended GPU:

```bash
uv sync --extra paper_results
uv run python -c "import jax; print(jax.devices())"
```

Generate the segment-scaling CSVs with the paper integration settings:

```bash
for system in articulated_soft_robot planar_pcs pcs; do
  uv run python paper_results/secIVb_parallel_rollouts_gpu/code/generate_benchmark_gpu.py \
    --systems "$system" \
    --segment-counts 1 2 4 8 16 32 \
    --batch-sizes 1 2 4 8 16 32 64 128 256 \
    --duration 1 --solver-dt 1e-4 --save-dt 1e-2 \
    --csv "paper_results/secIVb_parallel_rollouts_gpu/data/${system}_1s.csv"
done
```

The GVS segment sweep in the committed paper data stops at 16 segments:

```bash
uv run python paper_results/secIVb_parallel_rollouts_gpu/code/generate_benchmark_gpu.py \
  --systems gvs --gvs-scaling segments --segment-counts 1 2 4 8 16 \
  --batch-sizes 1 2 4 8 16 32 64 128 256 \
  --duration 1 --solver-dt 1e-4 --save-dt 1e-2 \
  --csv paper_results/secIVb_parallel_rollouts_gpu/data/gvs_segments_1s.csv
```

Generate the GVS strain-basis-order sweep:

```bash
uv run python paper_results/secIVb_parallel_rollouts_gpu/code/generate_benchmark_gpu.py \
  --systems gvs --gvs-scaling basis-order --gvs-basis-orders 0 1 2 3 4 5 \
  --batch-sizes 1 2 4 8 16 32 64 128 256 \
  --duration 1 --solver-dt 1e-4 --save-dt 1e-2 \
  --csv paper_results/secIVb_parallel_rollouts_gpu/data/gvs_basis_order_1s.csv
```

Regenerate the canonical segment-scaling figures from the committed CSVs:

```bash
uv run python paper_results/secIVb_parallel_rollouts_gpu/code/plot_benchmark_gpu.py \
  --systems articulated_soft_robot planar_pcs pcs gvs \
  --segment-counts 1 2 4 8 16 32
```

Use explicit temporary `--csv` paths for validation runs to avoid replacing the
committed hardware measurements.
