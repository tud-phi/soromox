# Section Vd: Control-Gain Optimization

This case compares single-start gain optimization for collocated
actuation-space control and synergistic operational-space control.

> [!WARNING]
> The committed archives and comparison figures are single-start results from
> the current workflow. They do not reproduce the published six-start Section
> Vd study and must not be treated as recovered paper results. What the
> published study actually did is documented under
> [Provenance of the published six-start results](#provenance-of-the-published-six-start-results).

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


## Provenance of the published six-start results

The Section Vd results in the preprint came from a **six-start** optimization
that no committed script reproduced. Issue #154 asked what that run was. The
original generator was recovered in September 2026 and is archived, with the
scripts that verify it, under [`legacy/`](legacy/). This section answers #154
directly; `legacy/README.md` carries the forensics.

### Initialization

`PRNGKey(15)`, nominal `Kp`/`Ki`/`Kd` of `40`/`20`/`2` across three tendons.
Start 0 is the nominal set; starts 1 to 5 are drawn **i.i.d. per component**
from the absolute box `Kp ∈ [20, 60]`, `Ki ∈ [10, 30]`, `Kd ∈ [2, 3]`.

| start | Kp | Ki | Kd |
| --- | --- | --- | --- |
| 0 * | 40.0000, 40.0000, 40.0000 | 20.0000, 20.0000, 20.0000 | 2.0000, 2.0000, 2.0000 |
| 1 | 31.2469, 54.3426, 47.1750 | 25.1556, 18.8813, 21.6813 | 2.9701, 2.2024, 2.6164 |
| 2 | 32.4273, 54.8669, 57.0937 | 29.2178, 20.6345, 12.3033 | 2.4348, 2.0921, 2.2355 |
| 3 | 49.2084, 33.9355, 45.0627 | 16.4979, 22.2535, 22.8547 | 2.3996, 2.0074, 2.4094 |
| 4 | 36.8086, 40.0154, 52.9908 | 12.5373, 23.5919, 15.6348 | 2.0766, 2.4019, 2.0736 |
| 5 | 26.0071, 43.6550, 21.3733 | 29.8076, 27.3036, 10.3260 | 2.9530, 2.1560, 2.0395 |

`*` nominal. Reproduce with `sample_initial_gains(..., scheme="legacy_mixed_v1",
seed=15)` from [`code/secvd_init.py`](code/secvd_init.py). Two details are
load-bearing: the three PRNG subkeys were consumed in **sorted** key order
(`Kd`, `Ki`, `Kp`), not `Kp, Ki, Kd`; and the run was in double precision, which
changes the stream. Both are enforced in code.

These are the **synergistic** gains. The collocated run came from the same
script family with a different controller block, and its own generator has not
been recovered, so its initialization is **presumed shared but unverified**.
Because `generate_mixed_batch` draws depend only on the key and the box, an
unchanged box would make the collocated starts numerically identical to the
table above; a rescaled box would not.

### Objectives

Recovered exactly, and **different for the two controllers**:

```text
collocated   1e2·∫₀^2.5 eᵀWe dt + 2e2·∫_2.5^5 eᵀWe dt,  W = diag([1,1,1,100,100,100])
synergistic  1e3·∫₀^2.5 ‖e‖² dt + 5e3·∫_2.5^5 ‖e‖² dt + 2.5·∫₀^5 P_t² dt
```

`W` is `1/L²` for the shear/axial strains with `L = 0.1 m`, so the collocated
objective is exactly the strain error non-dimensionalized by segment length.
`P_t = τ·(Aᵀq̇)` is tendon power. The collocated early/late ratio is **1:2**, the
synergistic **1:5**; today's scripts use 1:5 for both. Nothing was rescaled at
serialization, and every trajectory is exactly paired with its own loss.

This resolves the order-of-magnitude discrepancy reported on #154: it is a
coincidental composite of a 10× global factor, a 2.5× change in the late weight,
and the strain weighting — not a hidden normalization. The plotter's historical
division of the collocated loss by 100 was cosmetic axis alignment.

### Execution, selection and aggregation

Execution was **vectorized, not independent runs**: a `vmap` over both the
optimization variables and the optimizer state, giving six independent optimizers
stepping in lockstep with gradients never averaged. It is *multi-start*, not
multiple shooting — one forward integration, no segment boundaries, no continuity
constraints.

Selection was `best_batch = argmin_b min_i history_loss[i, b]`, which is `3` for
both archives. That coincidence is what hid the defect in #128.

Aggregation, and the 62% / 57% in `docs/research.md`, is
`1 - min(history_loss[-1]) / median(history_loss[0])`, giving **62.62%** and
**57.82%**.

### Why the published numbers are not regenerable here

- The two archives are not even from the same robot. The synergistic generator
  drives a **two-segment** robot (tip at `q = 0` is `[0, 0, 0.20]`, matching the
  archive bit-for-bit); the collocated archive stores 6-DOF configurations, so
  one segment. This case builds a one-segment robot with different material
  parameters.
- The legacy objectives above differ from the current ones in global factor,
  early/late ratio, strain weighting and the power penalty.
- The legacy run carried a bug: its Optax state was assigned to an unused name,
  so the Yogi moments stayed frozen at their iteration-1 value for the remaining
  99 iterations. It was not a standard Yogi run.

The figures in `docs/research.md` therefore describe the legacy models and are
left as published.

### What the archives could not answer

The archives hold no gain history at all, so the initialization could not be read
back out of them. The generator does write `hist_opt_ctr_params_*` via
`scipy.io.savemat`, but the committed files carry a `PCWIN64` header, a single
shared timestamp and only five variables each: they are a subset re-saved from
MATLAB, and the gains were discarded in that step. Recovering the generator was
the only route.
