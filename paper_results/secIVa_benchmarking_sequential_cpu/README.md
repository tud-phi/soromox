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

## PyElastica tests and compatibility

The generators use the public `PositionVerlet.step()` API and have been
verified with PyElastica 0.3.3.post2 and 1.0.0. The compatibility CI job runs
four short rollout tests against both versions. These tests import all three
generators without constructing rods, opening plots, or writing paper data;
then exercise planar/spatial gravity rollouts and tendon forcing with
processing callbacks both enabled and disabled.

Run the tests from the repository root:

```bash
uv sync --extra test
MPLBACKEND=Agg uv run python -m pytest -q tests/paper_results/secIVa/test_simulate_pyelastica.py
```

Manual full-rollout validation against the committed data found maximum
absolute numeric differences of `1.42e-13` with PyElastica 0.3.3.post2 and
`1.03e-11` with PyElastica 1.0.0 across all three generators.

## Shared implementation and possible COOMM migration

The local rod, actuation, and muscle helpers share implementations with
[COOMM](https://github.com/hanson-hschang/COOMM). Its original copyright and
MIT notice are retained in
[`code/pyelastica/LICENSE-COOMM.txt`](code/pyelastica/LICENSE-COOMM.txt).
The original adaptation's upstream revision remains unconfirmed. The
comparison below uses COOMM commit
[`d20a38ad87895051163bc2a7c186cd07e01b53b8`](https://github.com/hanson-hschang/COOMM/tree/d20a38ad87895051163bc2a7c186cd07e01b53b8)
and the four helpers initially added in SoRoMoX PR #220.

| Local helper | COOMM counterpart | Behavior to preserve |
| --- | --- | --- |
| `_rod_tool.py` | `coomm/_rod_tool.py` | All 14 definitions match after excluding formatting and docstrings. |
| `actuation.py` | `coomm/actuations/actuation.py` | Load-conversion kernels match; translate the explicit callback flag to COOMM's optional callback list. |
| `muscles.py` | `coomm/actuations/muscles/muscle.py` | Keep benchmark inputs as forces in newtons and retain constant stored muscle area. |
| `tendon.py` | Custom benchmark routing | Retain the benchmark's tendon offsets and wrappers. |

A temporary prototype replaced the first three helpers (942 of the original
1,047 added lines) with COOMM plus a 39-line adapter, leaving the 105-line
tendon routing module. That would remove approximately 86% of the locally
maintained lines. The adapter overrides COOMM's area update and force law,
and preserves the benchmark's callback enable/disable behavior.

The prototype reproduced the default three-second, 30,000-step tendon
rollout and three short rollouts with inputs `[1, 0.5]`, `[0, 0]`, and
`[0.2, 1.3]`. Comparisons covered tip trajectories, final rod position,
velocity, directors and angular velocity, actuator forces/couples, muscle
state, and enabled processing callbacks. The tested differences were zero.
This is a comparison for finite benchmark states: COOMM clears actuation
arrays by assignment, while the local multiplication by zero preserves NaNs.

An actual dependency migration needs a COOMM release or maintained fork with
updated package constraints. Published COOMM 0.1.1 declares Python
`>=3.10,<3.11`, NumPy `<2`, Numba `<0.56`, and PyElastica `<0.4`; these differ
from the supported SoRoMoX environment. COOMM's default muscle law also scales
activation by stress and changing area, whereas this benchmark applies the
input directly as force. Replacing that law without an adapter changes the
physical model.
