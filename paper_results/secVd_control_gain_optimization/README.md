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

`--device auto` is the default and **leaves the choice to JAX**. Which backend
runs this case faster is a property of the machine -- the relative FP64
throughput of its CPU and GPU, and how its GPU handles a long chain of small
sequential kernels -- not of Section Vd, so nothing here prescribes one. Pass
`--device cpu` or `--device gpu` to pin one; the run then fails rather than
starting on another backend, so a timing claim is never attached to hardware
that did not produce it.

The cost structure is worth knowing before choosing, because it is unusual for
a batched optimization. A rollout is 50000 strictly sequential solver steps on
a state of a few dozen degrees of freedom, so per-step work is tiny and cannot
be parallelized along time; `--batch-size` adds parallel width without reducing
that sequential depth. A GPU therefore tends to be bound by kernel-launch
latency rather than throughput here, and double precision -- which
[`legacy/`](legacy/) reproduction requires -- runs at a small fraction of
single-precision rate on consumer GPUs.

Note that JAX names the CUDA backend `cuda`, not `gpu`; `JAX_PLATFORMS=gpu` is
rejected outright, so an explicit `--device gpu` requests `cuda`. Device
selection must happen before JAX is imported, since JAX ignores `JAX_PLATFORMS`
afterwards, which is why the CLI and initialization modules import JAX lazily.

## Plant

`secvd_case.MATERIAL_DAMPING` is `7.2e1`, down from the published `3.6e2`. This
changes the *model*, not the optimizer, and it is here because the published
plant sat on the wrong side of the objective's own optimum.

For a second-order step response the integral squared error has a closed form:

```text
J(zeta, w_n) = (4 zeta^2 + 1) / (4 zeta w_n)
dJ/dzeta = 1 - 1/(4 zeta^2) = 0   =>   zeta* = 1/2,  independent of w_n
```

ISE is therefore minimized at `zeta = 0.5` -- **not** at critical damping. That
optimum carries 16.3 % overshoot; `zeta = 1` has none but costs `1.25x` the ISE.
And `J` scales as `1/w_n`, so bandwidth always pays: there is no interior
optimum in `Kp`, and `Kd` must rise with `Kp` merely to hold `zeta` fixed.

Reading `zeta` back from the measured overshoot of the nominal closed loop,
`zeta = -ln(PO) / sqrt(pi^2 + ln^2 PO)`:

| `c` | nominal `Kd` | overshoot | `zeta` | direction the loss moves `Kd` |
| --- | --- | --- | --- | --- |
| `3.6e2` | `x 1.0` | 1.18 % | 0.816 | down |
| `3.6e2` | `x 0.2` | 2.20 % | 0.772 | down |
| `1.8e2` | `x 1.0` | 9.72 % | 0.596 | up |
| `1.8e2` | `x 0.2` | 16.56 % | 0.497 | up |
| `7.2e1` | `x 1.0` | 15.92 % | 0.505 | up |
| `7.2e1` | `x 0.2` | 22.05 % | 0.434 | up (strongly) |

The last column is the sign of the loss difference between `Kd x 0.2` and
`Kd x 1.0` at fixed `c` and `Kp`, i.e. which way the objective actually pulls the
derivative gain -- not a gradient, a finite difference across the two
initializations the study runs.

At `c = 3.6e2` the nominal loop sits at `zeta = 0.816`, above the optimum, so the
optimizer reaches `zeta = 0.5` by *removing* damping. That is why the published
configuration drives `Kd` negative: correct control action for this objective,
not an optimizer defect. It is also why optimization could only ever *increase*
the overshoot the figure plots, starting from 1.18 % -- there was nothing to
reduce.

At `c = 7.2e1` the nominal loop lands at `zeta = 0.505`, essentially on the ISE
optimum, and the committed initialization then starts *below* it
(`INIT_GAIN_SCALE`, `Kd x 0.2`, `zeta = 0.434`) so the optimizer has damping to
add and a visible overshoot to remove.

`7.2e1` is also close to a lower bound. Below it the setpoint rollout in
`build_sec_vd_case` stops settling inside its 10 s window -- 11.0 s at `3.6e1`,
21.1 s at `1.8e1` -- so the reference the objective tracks would itself be
unconverged. `1.8e2` was measured and is admissible (`zeta = 0.596`), but leaves
less room between the nominal loop and the optimum.

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
separately from the loss definition; see [Tuning study](#tuning-study) for what
they measured. Note the legacy run is not a guide here: it used a single
`yogi(4e-3)` on `[0, 1]`-normalized, box-clipped gains, and its optimizer state
was frozen by a bug after the first iteration.

The optimizer is built once, in `secvd_case.build_optimizer`: one
`scale_by_yogi` whose update is rescaled per gain family, rather than three
`optax.yogi` instances under a `multi_transform`. Those are the same optimizer --
Yogi's moment recursions never see the learning rate, so a per-family rate
factors out of the state -- verified to `1e-9` relative over 30 steps spanning
gradient magnitudes `1e-6` to `1`. Gradient clipping is **off by default**: the
committed `clip_by_global_norm(10.0)` never bound, the largest family gradient
norm measured on the real objective being 1.55 against a threshold of 10. Pass
`--clip 10.0` to reproduce the committed configuration.

### Gain identifiability

Neither objective determines all three gains, and each is blind to a *different*
one. Measured as the logarithmic sensitivity at the nominal gains -- the
fractional change in loss per fractional change in gain, which is scale-free and
so comparable across families and across the two controllers:

```text
S_f = |K_f * dL/dK_f| / L
```

| | `Kp` | `Ki` | `Kd` |
| --- | --- | --- | --- |
| collocated | 4.50e-01 | **4.74e-04** | 7.75e-02 |
| synergistic | 5.59e-01 | 2.74e-01 | **4.63e-04** |

The bold entries sit three orders of magnitude below `Kp`. The collocated
regulator acts in actuation space, where the integral term contributes almost
nothing to a 5 s tracking error on this setpoint; the synergistic controller
acts in the task space, where the objective is dominated by the proportional
response and barely registers damping at all.

The consequence is visible in the archives: the optimized `Ki` is negative in
5/18 collocated components, and the optimized `Kd` in 5/18 synergistic ones.
**These signs carry no physical meaning and should not be read as a tuned
result.** Along a direction the loss cannot resolve, the optimizer's motion is
set by whatever gradient noise it sees, so the value it lands on is arbitrary
within a wide band; that it lands negative reflects the absence of a constraint,
not a preference for negative feedback. Nothing in this code constrains a gain's
sign -- there are no box constraints and no projection anywhere in the pipeline.

This is a property of the objectives, not a tuning failure, and it is not
fixable by changing the learning rate, the plant or the initialization: at the
committed rate the same gains simply do not move far enough from their positive
starts to reveal it. Reporting it is the honest option, since a reader comparing
gain tables across the two methods would otherwise assume both are determined.

### Integral-error saturation

The collocated controller integrates tendon-length errors in metres and uses the
unit-preserving saturation

```text
sat(e) = tanh(gamma * e) / gamma,   gamma = 1 / e_sat.
```

The default is `e_sat = 10 mm` (`gamma = 100 1/m`), defined once as
`secvd_case.COMMITTED_E_SAT`, settable with
`--integral-error-saturation-scale` in metres and recorded in the archive as
`saturation_config`.

**This is a controller design parameter, not an optimizer setting, and it is not
tuned against the loss.** Its job is to bound the integral state. The objective
has no stability term, no actuator model and a finite horizon, so a loss ranking
over `e_sat` measures how much integral action a 5 s window happens to want --
which moves with the horizon -- rather than anything about windup. The measured
ordering is monotone toward *tighter* saturation, which is that effect and not a
recommendation.

The recovered legacy generator gives no guidance either: the synergistic program
saturated *torque* with `tanh` at `tau_max = 30`, not the integral error, and the
collocated generator was never recovered. So the value stays where PR #135 put it
on physical grounds, and the tuning study's `saturation` stage answers what the
issue actually asks -- its effect on tracking, windup, gradients and the
optimized gains. Measured across the whole range over which the scale can act,
from the steady-state error (5.28e-05 m, below which the integrator is clamped
beneath the offset it exists to remove) to the peak error (1.451e-02 m, above
which `tanh` never leaves its linear region):

| `e_sat` [m] | best loss | rollout saturated | peak integral norm | final integral norm |
| --- | --- | --- | --- | --- |
| 5.28e-05 | 2.184549e-03 | 9.41 % | 3.760e-05 | 2.302e-05 |
| 2.15e-04 | 2.184721e-03 | 5.38 % | 9.454e-05 | 4.664e-05 |
| 8.75e-04 | 2.185288e-03 | 2.60 % | 2.045e-04 | 1.303e-04 |
| 3.56e-03 | 2.186172e-03 | 0.62 % | 4.576e-04 | 2.692e-04 |
| 1.45e-02 | 2.186859e-03 | 0.00 % | 8.331e-04 | 4.680e-04 |

Across 275x in `e_sat` the loss moves 0.106 %, every point completes its full
budget with gradients in one regime, and the integrator's peak moves 22x. The
scale reaches the integrator and not the optimization, so `e_sat = 10 mm` stands.

Two properties of this plant explain that, and both hold independently of the
sweep. The regulator feeds the potential forward, so the offset left for the
integrator is 52.8 um against a 14.5 mm initial step. And there is no actuator
limit -- tendon force bounds default to `[0, inf)` and are metadata, never
enforced in the dynamics; the archived command reaches 2.34 N and dips to
-0.045 N, below its own nominal lower bound, unclipped. Windup is accumulation
that occurs while an actuator is saturated, so the mechanism is latent here.
Give the actuators a finite bound and this becomes a live parameter again.

Note the sweep must threshold the error the PID *integrates* -- the actuated
coordinates, three components in metres, exposed as `problem.control_error_fn` --
not the configuration-space strain error the loss is built from, which has six
components in mixed units. The regulator is underactuated, so the two do not even
share a shape.

## Tuning study

`code/secvd_tuning_study.py` is a study, not a results generator: it writes small
summaries to `data/tuning/` and never a schema-v3 archive. Every stage builds its
problem with `secvd_case.build_evaluator` and its optimizer with
`secvd_case.build_optimizer` -- the same two calls the generators make -- so a
setting measured here is measured on what actually runs.

Three learning rates cannot be searched directly at full fidelity, so the search
is decomposed into a magnitude and a shape, measured by different stages:

```text
(lr_P, lr_I, lr_D) = alpha * (r_P, r_I, r_D)
committed:  (0.5, 0.25, 0.05) = 0.5 * (1, 0.5, 0.1)
```

Run one stage at a time:

```bash
S=paper_results/secVd_control_gain_optimization/code/secvd_tuning_study.py
uv run python $S --method collocated --stage solver-dt
uv run python $S --method collocated --stage lr-range
uv run python $S --method collocated --stage lr-confirm --alphas 300 1500 3000
uv run python $S --method collocated --stage ratio --alpha 3000
uv run python $S --method collocated --stage saturation --alpha 3000
```

### What it measured

**`solver-dt` -- the integration step.** Gated on the *gradient*, not the loss: a
loss can stay accurate while its derivative drifts, and here it does by two
orders of magnitude. Accept the coarsest `h` with
`|grad(h) - grad(1e-4)| / |grad(1e-4)| <= 1e-3`.

| solver dt | collocated | synergistic | speed-up |
| --- | --- | --- | --- |
| 5e-4 | 7.5e-08 | 5.0e-10 | 5x |
| 1e-3 | 2.0e-06 | 5.2e-09 | 10x |
| 2e-3 | 3.2e-05 | 1.8e-07 | 20x |
| 5e-3 | NaN | NaN | -- |

Both methods answer identically, as a limit set by the plant and a fixed-step
solver rather than by a controller should. The previously published `1e-4` is
~20x finer than needed. The stages and the archives both run at `1e-3`, five
times below failure -- the margin matters because this is measured at the
*initial* gains and an optimizer that works raises them, which stiffens the
closed loop. Re-measured on the `c = 7.2e1` plant at the initializations the
generators actually start from, the relative gradient error at `1e-3` is 7.0e-08
collocated and 3.2e-09 synergistic. Schema v3 records `solver_dt`, so a coarse
run is no longer indistinguishable from a full-fidelity one.

**`lr-range` -- screening.** A geometric ramp, one rate per iteration, so N rates
cost N iterations. It **locates, it does not measure**: the loss at each point
depends on where the earlier points left the parameters, so the divergence point
is a property of the ramp. Three ramps over one identical problem recommended
6.4e4, 3.7e4 and 3.3e3 -- a 19x spread -- tracking exactly how much loss each had
banked on the way up (67 %, 76 %, -8 %). Consequences: start well *below* the
rate being looked for, and read the output as a bracket rather than a value. The
stage does both, and gates on `log10(located / committed)` against one decade.

**`lr-confirm` -- measurement.** Independent constant-rate runs over the bracket.

| method | alpha | final loss, 20 iterations | outcome |
| --- | --- | --- | --- |
| collocated | 300 | 1.806e-03 | stable |
| collocated | 1500 | 1.441e-03 | stable |
| collocated | **3000** | 1.200e-03 | stable; confirmed to 100 iterations |
| collocated | 6000 | -- | non-finite at iteration 6 |
| collocated | 12000 | -- | non-finite at iteration 3 |
| synergistic | **1547** | 7.490e-04 | stable; confirmed to 100 iterations |
| synergistic | 4893 | 8.833e-04 | stable but worse |
| synergistic | 15472 | -- | non-finite at iteration 3 |

The committed `alpha = 0.5` is 3.8 decades short for collocated and 3.5 for
synergistic. `secvd_case.TUNED_ALPHA` holds the resulting defaults.

**A 20-iteration ranking is not a 100-iteration ranking**, and the two methods
illustrate opposite failure modes. Collocated's table is monotone up to the
divergence point: every rate that survives beats every slower one, so the
"best" is simply the last survivor and the *only* real question is where it
dies. That makes the confirmation at the full budget mandatory rather than
decorative -- a rate that looks best over 20 iterations may not reach 100, and
all of the collocated candidates were still descending at iteration 20.
Synergistic's is not monotone: 1547 beats 4893 outright at equal budget, so it
is an interior optimum and the ranking is informative on its own.

Note that the screening rows above ran with the study's former default of
`clip_by_global_norm(10.0)` while the 100-iteration confirmations ran with
clipping off, matching the generators. The clip never binds at these gradient
magnitudes, so this does not affect the ranking -- but the study's default has
since been changed to match the generators, because "measured no difference" is
not the same as "measured the same thing".

**Horizon.** The rollout length is a property of the case, not of the optimizer,
but it interacts with the objective directly: `L = (1/T) * integral ||e_hat||^2`
is time-*averaged*, so lengthening the window dilutes the transient against the
settled tail and gives the optimizer less reason to shape the part of the
response the figure shows. Measured on collocated at `alpha = 3000`, 100
iterations, identical initializations:

| | `T = 5 s` | `T = 10 s` |
| --- | --- | --- |
| median peak-overshoot change, plotted strains | **-20.97 pts** | -17.75 pts |
| median optimized overshoot | **12.9 %** | 14.2 % |
| median optimized `zeta` | 0.542 | 0.525 |
| median per-start loss improvement | **67.72 %** | 62.53 % |
| `Kd` negative | **0/18** | 1/18 |

The longer horizon lands marginally closer to the theoretical `zeta = 0.5` but
buys that with a smaller visible overshoot reduction and a worse loss
improvement, so the case keeps `T = 5 s`. Losses are **not** comparable across
the two columns -- the `1/T` factor alone changes them -- which is why the
comparison is made on `zeta` and overshoot.

**`ratio` -- the per-family split.** At the confirmed magnitude, the committed
`(1, 0.5, 0.1)` is compared against equal weighting. Equal weighting goes
non-finite on its *first* update on both problems: the derivative gain tolerates
a far smaller step than the proportional one. The committed split is therefore
kept by measurement rather than by inheritance.

**`saturation` -- item 2.** See
[Integral-error saturation](#integral-error-saturation). This stage does not
select `e_sat`; it measures that the optimization is insensitive to it.

### What the retune is worth

At equal budget and on one plant -- 100 iterations, six starts, the same
objective, the same initializations, the same `solver_dt`, `c = 7.2e1`. Both
columns are measured; the rate is the only thing that differs, so the comparison
does not conflate the retune with the plant change:

| collocated | committed `alpha = 0.5` | tuned `alpha = 3000` |
| --- | --- | --- |
| median per-start improvement | 3.14 % | **67.72 %** |
| median final loss | 2.188530e-03 | 7.273457e-04 |
| best loss over six starts | 1.338258e-03 | 6.017164e-04 |
| spread of per-start final | 2.348x | **1.212x** |
| median relative step per iteration | 2.287e-05 | 6.986e-03 |

The last two rows carry the finding. A relative step of `2e-05` is the stall
stated directly: at the committed rate the gains move by two parts in a hundred
thousand per iteration, so 100 iterations cannot travel anywhere, and the run's
outcome is decided by which start was sampled lowest -- hence the 2.35x spread.
At the tuned rate the optimizer dominates the initialization and the six starts
land within 21 % of each other.

This is also why `improvement_over_initial_median` has to be read alongside the
per-start median. The two agree here (73.33 % against 67.72 %, with the winning
start itself improving 84.11 %), but they need not: on the previously published
configuration -- `alpha = 0.5` on the `c = 3.6e2` plant -- the statistic
reported 28.39 % for a run whose median start improved 1.11 %,
because the winning start began lowest and improved 0.20 %. The legacy 62.62 % /
57.82 % figures use the same formula and carry the same caveat.

**Reproducible in outcome, not bit-reproducible in trajectory.** The archive and
an independent tuning-study run at identical settings agree exactly on the
problem -- their iteration-0 losses match to 1.3e-15, which is what pins the
shared builder -- but their optimizer trajectories separate: 2.5e-07 after one
step, peaking near 5e-02 around iteration 57, back to 1.6e-03 by iteration 99,
with final median gains within 0.3 %. The tuned rate is aggressive enough that
the first step *raises* the loss 2.7x, and in that regime the floating-point
difference between two scripts' compiled gradients amplifies geometrically. It
concentrates in the lowest-`Kp` start, which takes the largest relative step;
two of the six starts stay at 1e-09 throughout. Expect the same basin and the
same statistics, not the same digits.

### Ruled out

- **Yogi's `eps`.** optax defaults to `1e-3`, four orders above this problem's
  gradients, which should degenerate Yogi into SGD at `lr / eps`. Rerunning at
  `1e-8` gave nearly the same curve, so this was not the cause. `YOGI_EPS = 1e-8`
  is kept regardless: a numerical floor above the signal is indefensible whether
  or not it binds.
- **Gradient clipping.** Never bound; see
  [Optimizer diagnostics](#optimizer-diagnostics).
- **Loss scaling.** Scaling the objective by `1e4` *does* move Yogi -- 539x
  larger steps at small `alpha`, decaying to 2.7x at large `alpha` -- so
  Adam-family scale invariance is not exact here. But it is interchangeable with
  the learning rate and reaches a worse optimum, so it is a confounder to remove
  rather than a knob to tune.

Reading the step size directly is what made this legible. Rather than inspecting
optimizer state, whose layout varies across optax versions and nests under
`multi_transform`, the study records `|update| / lr` per family. A well-scaled
adaptive optimizer sits near 1; a small *constant* value means the step is still
linear in the rate and the rate is short by that factor.

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
schema version 3 and writes both the canonical PDF and PNG:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/plot_control_gain_optimization.py
```

It overwrites its outputs by default, so it runs with no arguments; pass
`--no-force` to refuse instead. That is the opposite of the generators, and
deliberately: an overwrite here re-renders committed archives in seconds, while
an overwrite there destroys a completed optimization.

The loss panels show the min-max range across starts with the best loss per
iteration drawn on top; the tracking panels show the initial spread as a median
with a band, and the best start as the solid line. Starts frozen by
`history_finite_mask` are excluded from the bands rather than plotted as gaps. A
single-start archive is drawn without bands, since there is no spread to shade.

The band is the range over starts of the **initial** rollouts, and it has its own
legend entry for that reason: it sits behind a solid line that is one *optimized*
start, so without a label it reads as an interval around the optimized response,
which it is not.

Two presentation choices follow from what the tuned optimizer does, and neither
changes a number:

- **The loss panels are log-scaled.** At the tuned rates the loss falls by more
  than a decade within a few iterations, so on a linear axis everything after
  that is sub-pixel. Cropping the axis instead would misreport the collocated
  run, which still completes 21 % of its total descent after iteration 50
  (synergistic completes 99.75 % by then -- the two panels are not alike).
- **The tracking panels display the first `TRAJECTORY_WINDOW_S = 2.0` s** of the
  5 s rollout. The objective still integrates the whole horizon; this is the
  display window only. Measured 2 % settling times are 0.81 s initial and 0.57 s
  optimized for the collocated strains and 2.02 s initial for the synergistic
  position, so a 5 s axis spent its last 70 % on flat lines and compressed the
  across-start band into an invisible sliver at the left edge.

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
