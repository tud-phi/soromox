# Section IVa: Sequential CPU Benchmarks

This case compares end-effector rollouts produced by SoRoMoX, PyElastica, and
SoRoSim. Generator code, the committed paper data, and the canonical plots are
kept in `code/`, `data/`, and `outputs/`, respectively.

The paper reports three-second rollouts on an Intel Core Ultra 7 165H CPU with
16 GB RAM. SoRoSim uses adaptive `ode45`; SoRoMoX uses `Tsit5` and PyElastica
uses `PositionVerlet`, both with a fixed 0.1 ms step. See
[Paper & Results](https://tud-phi.github.io/soromox/research/#sequential-cpu-rollouts)
for the reported runtimes and trajectory agreement.

Install the paper dependencies from the repository root:

```bash
uv sync --extra paper_results
```

Generate the SoRoMoX data:

```bash
for script in paper_results/secIVa_benchmarking_sequential_cpu/code/soromox/*.py; do
  JAX_PLATFORMS=cpu uv run python "$script"
done
```

Generate the PyElastica data:

```bash
for script in paper_results/secIVa_benchmarking_sequential_cpu/code/pyelastica/simulate_*_pyelastica.py; do
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

## PyElastica compatibility

All three generators are compatible with **PyElastica 1.0.0** and have also
been verified with 0.3.3.post2. They use the public `PositionVerlet.step()` API.
The tendon model uses COOMM while preserving force inputs in newtons and
constant stored tendon area.

Run the short rollout tests with:

```bash
uv sync --extra test --extra paper_results
MPLBACKEND=Agg uv run python -m pytest -q tests/paper_results/secIVa/test_simulate_pyelastica.py
```

COOMM installs automatically from the Git revision pinned in
[`pyproject.toml`](../../pyproject.toml), which contains the compatibility fixes
from [upstream PR #11](https://github.com/hanson-hschang/COOMM/pull/11).

## Runtime comparison

Recreate the publication-style grouped bar chart from the four-case wall-clock
time table in [`docs/research.md`](../../docs/research.md#sequential-cpu-rollouts),
updated in commit `3b3815c` (paper Table IV):

```bash
uv run --extra paper_results python paper_results/secIVa_benchmarking_sequential_cpu/code/plot_benchmark_cpu_runtime.py
```

This writes `outputs/benchmark_cpu_runtime.pdf`, `.svg`, and a 300 dpi `.png`.
All three formats have transparent figure, axes, and legend backgrounds.
Pass `--force` to replace existing figures. The transcribed values and source
metadata are stored in `data/benchmark_cpu_runtime.json`; simulations are not rerun.
The figure uses `paper.mplstyle`, the shared blue/orange palette, Computer Modern
serif fonts rendered with LaTeX, and a linear axis starting at zero. Runtimes,
units, and speedup ratios use LaTeX math formatting. Regeneration requires LaTeX
and `dvipng`, as used by Matplotlib's TeX renderer. SVG text is saved as paths
to preserve the typography when imported into other applications.
A bar chart is used because the table contains scalar timings without repeated
measurements or uncertainty estimates needed for box plots or error bars.

Suggested caption: Sequential CPU computation time for a 3 s simulation with
adaptive Dormand-Prince `ode45` for SoRoSim and fixed-step Tsitouras `Tsit5`
(0.1 ms) for SoRoMoX on an Intel Core Ultra 7 165H CPU. Labels above bars give
reported runtimes in seconds; annotations give the SoRoMoX speedup, calculated
as the SoRoSim runtime divided by the SoRoMoX runtime. Lower runtime is better.
Values are reproduced from the updated table in `docs/research.md`; the exact
source commit is recorded in the data JSON. The spatial GVS case
corresponds to the repository's `complex-gvs` rollout.
