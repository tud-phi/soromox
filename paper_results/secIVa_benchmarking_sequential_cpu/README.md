# Section IVa: Sequential CPU Benchmarks

This case compares end-effector rollouts produced by SoRoMoX, PyElastica, and
SoRoSim. Generator code, the committed paper data, and the canonical plots are
kept in `code/`, `data/`, and `outputs/`, respectively.

Install the paper dependencies from the repository root:

```bash
uv sync --extra paper_results
```

Generate the SoRoMoX data:

```bash
for script in paper_results/secIVa_benchmarking_sequential_cpu/code/soromox/*.py; do
  uv run python "$script"
done
```

Generate the PyElastica data:

```bash
for script in paper_results/secIVa_benchmarking_sequential_cpu/code/pyelastica/*.py; do
  uv run python "$script"
done
```

SoRoSim requires MATLAB and a configured SoRoSim installation. Run all four
cases by setting `Case` to 1 through 4:

```bash
matlab -batch "Case=1; run('paper_results/secIVa_benchmarking_sequential_cpu/code/sorosim/simulate_sorosim.m')"
```

Repeat that command for `Case=2`, `Case=3`, and `Case=4`. The script resolves
the system MAT files relative to itself and writes rollout MAT files to `data/`.

Recreate the four canonical figures without rerunning simulations:

```bash
uv run python paper_results/secIVa_benchmarking_sequential_cpu/code/plot_benchmark_cpu.py
```

The generators replace same-named files in `data/`. Use a clean worktree or
copy the committed canonical data before intentionally regenerating it.
