# Section Vd: Control-Gain Optimization

This case compares six-start gain optimization for collocated actuation-space
control and synergistic operational-space control.

> [!WARNING]
> The committed archives and comparison figures are single-start results from
> the current workflow. They do not reproduce the published six-start Section
> Vd study and must not be treated as recovered paper results. What the
> published study actually did is documented under
> [Provenance of the published six-start results](#provenance-of-the-published-six-start-results).

## Canonical result format

Each optimizer writes only `optimization_results.npz`, using schema version 2.
The archive carries a real multi-start batch axis of width `B`, and that axis
appears **only** on quantities that genuinely vary per start:

| field | shape |
| --- | --- |
| `history_loss`, `history_finite_mask` | `(iterations, B)` |
| `history_time` | `(iterations,)` |
| `history_Kp`, `history_Ki`, `history_Kd` | `(iterations, B, m)` |
| `init_Kp`, `init_Ki`, `init_Kd` | `(B, m)` |
| `q_ts_init/best`, `qd_ts_init/best` | `(B, timesteps, dofs)` |
| `u_ts_init/best` | `(B, timesteps, actuators)` |
| `x_ts_init/best` | `(B, timesteps, 6)` |
| `best_iteration` | `(B,)` |
| `best_batch`, `batch_size`, `completed_iterations` | scalars |
| `t_ts`, `x_des_ts`, `q_des_ts` | `(timesteps,)`, `(timesteps, 6)`, `(timesteps, dofs)` |

The time grid and the references are shared by every start and so carry no batch
axis at all: a batch axis in this archive always means something. Pose vectors
use

```text
[rotation-vector x, y, z, Cartesian position x, y, z].
```

`q_ts_best` is the authoritative trajectory for reconstructing the complete robot
geometry. `x_ts_best` provides a directly validated full end-effector pose
trajectory. Schema-v1 archives, legacy MAT archives, `animation_data.pkl` and
`intermediate_metrics.json` are not supported.

**Per-iteration trajectory histories are deliberately absent.** Storing every
start's rollout at every iteration would reach several gigabytes at 100
iterations and six starts, so the archive keeps each start's initial and best
rollouts only. The loss and gain histories remain complete.

`init_Kp` / `init_Ki` / `init_Kd`, together with `init_seed`, `init_scheme` and
`init_spread`, record the initialization directly rather than leaving it to be
inferred from the history. The legacy results were unrecoverable from their
archives precisely because this was never written; see the provenance section
below.

A start whose evaluation becomes non-finite is frozen rather than aborting the
whole run, and `history_finite_mask` marks exactly which entries are real. Every
start must still reach at least one finite iterate: one that was non-finite from
its first evaluation has no trajectory worth archiving and indicates an
initialization that should be resampled with a different `--init-seed`.

Selection is recorded and re-checked on load. `best_iteration` is each start's
own lowest-loss iteration and `best_batch` is `argmin_b min_i loss[i, b]`,
derived from **that archive's** losses alone -- the defect behind issue #128 was
a single index taken from one method and reused for the other.

`completed_iterations` records the actual run length and is validated against
the stored histories. `is_placeholder` identifies development-only archives;
omit `--placeholder` when producing full results.

### Initialization

Start 0 is always the nominal gain set, so a single-start run stays a strict
subset of a batched one. Starts 1 to `B-1` scale the nominal by one factor per
gain, drawn log-uniformly from `[1 / spread, spread]` -- gains are scale
parameters, so the neighbourhood is multiplicative rather than additive. Every
run prints its table and records the initialization in the archive.

| flag | default | meaning |
| --- | --- | --- |
| `--batch-size` | `6` | independent starts `B` |
| `--init-seed` | `0` | PRNG seed behind the sampled starts |
| `--init-scheme` | `log_uniform_v1` | sampling scheme, recorded in the archive |
| `--init-spread` | `3.0` | multiplicative half-width for `log_uniform_v1` |

The alternative `legacy_mixed_v1` scheme reproduces the recovered generator
exactly, for regenerating or checking against the legacy results. Its box is
absolute and belongs to the legacy two-segment robot, so it is not appropriate
for new work.

### Device

`--device auto` selects the **CPU at any batch size**. A rollout here is 50000
strictly sequential solver steps, so the GPU is bound by kernel-launch latency
rather than throughput, and batching adds parallel width without reducing that
sequential depth. Measured on this case (Threadripper PRO 5975WX, RTX 4070 Ti
SUPER, three iterations timed at the 100-iteration settings):

| device | compile + iteration 0 | steady iteration |
| --- | --- | --- |
| CPU, `B=1` | 51.5 s | 40.5 s |
| CPU, `B=6` | 132.8 s | 120.7 s |
| GPU, `B=1` | did not finish one iteration in 950 s | -- |

So a six-start run costs about three times a single-start one, not six, and a
full 100-iteration run takes roughly **3 h 20 m per method** on CPU. The two
methods are independent and each uses under two cores, so running them
concurrently costs no more wall time than one.

`--device gpu` remains available for settings with far fewer solver steps, where
the trade-off reverses. Note that JAX names the CUDA backend `cuda`, not `gpu`;
`JAX_PLATFORMS=gpu` is rejected outright, so an explicit `--device gpu` requests
`cuda` and fails loudly rather than running somewhere the user did not ask for.
Device selection must happen before JAX is imported, since JAX ignores
`JAX_PLATFORMS` afterwards, which is why the CLI and initialization modules
import JAX lazily.

## Objective

Both optimizers import one definition, from
[`code/secvd_objective.py`](code/secvd_objective.py), so "Loss" means the same
thing in both panels of the figure:

```text
L = (1 / T) * integral_0^T ||e_hat(t)||^2 dt

collocated   e_hat = [L_seg * e_kappa ; e_sigma]     (strains)
synergistic  e_hat = [e_theta ; e_x / L_seg]         (pose)
```

Angular strains are `1/m` and shear/axial strains already dimensionless;
task-space orientation error is in radians and position in metres. Both sides
come out **dimensionless**, so there is no global factor and no per-controller
divisor in the plotter.

The characteristic scale is not a free choice. The legacy collocated weighting
recovered from the archives, `W = diag([1, 1, 1, 100, 100, 100])`, equals
`(1/L^2) * diag(L^2, L^2, L^2, 1, 1, 1)` for `L = 0.1 m` -- it was already
non-dimensionalization by segment length, with only the leading constant
arbitrary. `test_strain_scales_reproduce_the_recovered_legacy_weighting` pins
that correspondence.

**There is no early/late weighting.** The legacy runs disagreed on it -- 1:2
collocated against 1:5 synergistic -- and the previous scripts had harmonized on
1:5, which was never the original on either side. The transient/steady-state
split is *reported* instead, via `steady_state_report`, reusing Section Vc's
`evaluation_metrics` rather than being folded into the loss.

Every archive records `objective_name` and `objective_scales`, so a stored loss
can always be traced to the definition that produced it and recomputed from the
archive alone. `recompute_archive_losses` does exactly that, and a regression
test asserts the stored initial and best losses match their own trajectories --
the check that would have caught the order-of-magnitude discrepancy reported on
issue #154.

### Optimizer diagnostics

`history_grad_norm` and `history_update_norm` record the per-start global L2
norms of the gradient and of the applied update, and `optimizer_metadata`
records the transforms, learning rates and clipping thresholds. These are stored
so learning rates and clipping can be tuned against something observable,
separately from the loss definition. Note the legacy run is not a guide here: it
used a single `yogi(4e-3)` on `[0, 1]`-normalized, box-clipped gains, and its
optimizer state was frozen by a bug after the first iteration.

### Integral-error saturation

The collocated controller integrates tendon-length errors in metres and uses the
unit-preserving saturation

```text
sat(e) = tanh(gamma * e) / gamma,   gamma = 1 / e_sat.
```

The default is `e_sat = 10 mm` (`gamma = 100 1/m`), settable with
`--integral-error-saturation-scale` in metres and recorded in the archive as
`saturation_config`. The recovered legacy generator gives no guidance on this
value: the synergistic program saturated *torque* with `tanh` at
`tau_max = 30`, not the integral error, and the collocated generator was never
recovered. Retuning it therefore has to be done against the current model --
sweep the flag and compare tracking, windup, the gradient and update norms above,
and the optimized gains.

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
requested iteration completed and every start produced at least one finite
iterate, so an interrupted run cannot replace an archive.

Both optimizers optimize six independent gain initializations at once, matching
the width of the published Section Vd results. Use `--batch-size` to change that;
see [Initialization](#initialization) and [Device](#device) above.

## Plotting

The standalone plotter is the only comparison-figure entrypoint. It requires
schema version 2 and writes both the canonical PDF and PNG:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/plot_control_gain_optimization.py \
  --force
```

The loss panels show the min-max range across starts with the best loss per
iteration drawn on top; the tracking panels show the initial spread as a median
with a band, and the best start as the solid line. Starts frozen by
`history_finite_mask` are excluded from the bands rather than plotted as gaps. A
single-start archive is drawn without bands, since there is no spread to shade.

Each panel takes its best start from **its own** archive's `best_batch`. The
defect in issue #128 was one index derived from the collocated losses and reused
for the synergistic panel; it went unnoticed only because both methods happened
to peak at start 3 in the legacy data.

The plotter also prints the loss reduction for each method, computed by the same
formula the paper reports, so the published percentages are regenerated rather
than retyped.

## Robot rendering

The renderer reconstructs the optimized robot from `q_ts_best` for that
archive's own `best_batch`, validates every start's stored pose against forward
kinematics,, validates it
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
