"""Tuning study for the Section Vd optimizer and controller saturation.

This is a study, not a results generator: it writes small summaries to
``data/tuning/`` and never a schema-v2 paper archive. Follow-up items 2 and 6 on
[issue #154](https://github.com/tud-phi/soromox/issues/154) both ask for values
to be *tuned* against something observable, and the first regenerated collocated
run showed why that has to come before any further results: at 100 iterations
and six starts its median per-start improvement was 1.11 %, with the loss still
descending at the last iteration. The optimizer was not converged, it was barely
moving.

Every stage builds its problem with :func:`secvd_case.build_evaluator` and its
optimizer with :func:`secvd_case.build_optimizer`, the same two calls the
generators make, so a setting measured here is measured on what actually runs.

## Stages

Run one at a time with ``--stage``; each writes its own summary.

``solver-dt``
    How coarse the integration step can be before the *gradient* moves. Cost is
    linear in the step count, so this is the study's largest speed lever and
    every later stage inherits it. Gated on the gradient rather than the loss,
    because a loss can stay accurate while its derivative drifts -- and here it
    does, by two orders of magnitude.

    Both methods answer the same way, which is what one would expect of a limit
    set by the plant and a fixed-step solver rather than by a controller: every
    step up to ``2e-3`` holds the gradient to ``3e-5`` relative or better,
    ``5e-3`` produces ``NaN``, and there is nothing in between. The published
    ``1e-4`` is therefore about 20x finer than it needs to be. The later stages
    run at ``1e-3``, keeping a factor of five below the failure, because this is
    measured at the *initial* gains and an optimizer that works raises them.

``lr-range``
    A geometric ramp over the learning-rate magnitude, locating the usable
    region in one run.

``lr-confirm``
    Independent constant-rate runs over a bracketing set, measuring the value
    the ramp located.

``ratio``
    The per-family split, compared at the confirmed magnitude.

``saturation``
    The collocated integral-error saturation scale ``e_sat`` -- follow-up item 2.
    Skipped, with a reason, on a controller that does not saturate.

``verify``
    One more ramp at the chosen ``e_sat``, checking that treating the two
    sequentially held.

## Why the magnitude and the ratio are searched separately

Three learning rates cannot be searched directly at full fidelity -- even a
coarse ``4x4x4`` grid is tens of hours -- so the search is decomposed into

```text
(lr_P, lr_I, lr_D) = alpha * (1, 0.5, 0.1)
```

where the ratio is the committed one (``0.5, 0.25, 0.05``, so the committed
``alpha`` is 0.5). ``lr-range`` and ``lr-confirm`` move ``alpha``; ``ratio``
moves the split.

## A ramp locates, it does not measure

A range test is a single trajectory through parameter space, not independent
runs at each rate: the loss at high ``alpha`` reflects where the earlier steps
left the parameters. That is what makes it cheap, and it is also why the
divergence point is a property of the *ramp* rather than of the problem. Three
ramps over the identical collocated problem:

```text
starts at   step    diverged at   recommends   loss cut on the way up
    100     1.27x        1.91e5       6.4e4        67 %
    100     1.22x        1.12e5       3.7e4        76 %
   0.01     2.15x        1.00e4       3.3e3        -8 %
```

A 19x spread, tracking the last column exactly. The first two started *above*
the usable rate and banked two thirds of the available loss reduction climbing
to their divergence point, arriving in a far better-conditioned region that
genuinely tolerates rates a cold start cannot. The third arrives at ``1e4``
having made no net progress and dies there, which is what the constant-rate scan
independently finds: ``1e4`` diverges on its second iteration.

Two consequences. Start the ramp well *below* the rate being looked for --
:data:`DEFAULT_ALPHA_LOW` does -- or the overshoot is guaranteed. And read the
result as a screening test with an order-of-magnitude dependence on the ramp
itself, never as a located value: ``lr-confirm`` measures, this stage only says
which direction to measure in.

## What was actually wrong, and what was not

The stall is a learning-rate magnitude, nothing subtler. ``alpha = 1000``
against a committed ``0.5`` -- roughly 2000x -- where 20 iterations already beat
the committed run's 100, and 100 iterations reach ``1.487e-3`` against
``1.890e-3``. Three other explanations were tested first and are recorded here
because each is plausible enough to be worth not re-deriving:

* **Yogi's ``eps``.** ``optax.yogi`` defaults to ``eps=1e-3``, four orders above
  this problem's gradients, which should degenerate it into SGD at ``lr / eps``.
  Rerunning at ``1e-8`` gave nearly the same curve, so this was not the cause.
  The lower value is kept anyway -- a numerical floor above the signal is
  indefensible whether or not it binds.
* **Gradient clipping.** ``clip_by_global_norm(10.0)`` never bound: the largest
  family gradient norm measured on the real objective anywhere in this study is
  1.55, against a threshold of 10. (Deliberately loss-scaled probes reach 2.65e3,
  which is the scale factor and not the problem.) Divergence here is a step-size
  effect, so a gradient clip could not have prevented it either. Dropped.
* **Loss scaling.** Scaling the objective by ``1e4`` does move Yogi -- 539x
  larger steps at low ``alpha``, decaying to 2.7x at high ``alpha`` -- so it is
  not the no-op that Adam-family scale invariance suggests. But it is
  interchangeable with the learning rate and reaches a worse optimum, so it is
  not a substitute for tuning one.

## Reading the step size

Rather than reaching into the optimizer's internal state, whose layout differs
across optax versions and nests under ``multi_transform``,
:func:`report_step_efficiency` records the observed ``|update| / lr`` per gain
family. A well-scaled adaptive optimizer sits near 1; a small constant value
means the step is linear in the rate and the rate is short by that factor.
"""

from __future__ import annotations

import csv
import sys
import time
from collections.abc import Callable
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from secvd_optimization import (  # noqa: E402
    DEVICE_CHOICES,
    configure_optimization_device,
)

# Must run before JAX is imported; JAX ignores JAX_PLATFORMS afterwards.
configure_optimization_device()

import argparse  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from jax import vmap  # noqa: E402
from secvd_case import (  # noqa: E402
    CASE_HORIZON,
    COMMITTED_E_SAT,
    COMMITTED_LR_RATIO,
    INIT_GAIN_SCALE,
    METHODS,
    build_evaluator,
    build_optimizer,
    build_sec_vd_case,
)
from secvd_init import (  # noqa: E402
    GAIN_ORDER,
    SECVD_BATCH_SIZE,
    SECVD_INIT_SCHEME,
    SECVD_INIT_SEED,
    SECVD_INIT_SPREAD,
    sample_initial_gains,
)

jax.config.update("jax_enable_x64", True)

CASE_DIR = CODE_DIR.parent
TUNING_DIR = CASE_DIR / "data" / "tuning"

# The learning-rate ratio currently in both generators, normalized so that the
# committed setting (0.5, 0.25, 0.05) is exactly alpha = 0.5.
GAIN_FAMILIES = GAIN_ORDER
# The committed scaling comes from secvd_case, so the study cannot drift from
# what the generators actually run.
LR_RATIO = COMMITTED_LR_RATIO

# The committed clip threshold, kept only as the value --clip documents. The
# study's default matches the generators' -- off -- so a rate measured here is
# measured under the transform the archives are produced with. It defaulted on
# for a while, which is exactly the study/generator drift the shared builder
# exists to prevent; it made no numerical difference, because the largest family
# gradient norm ever measured on the real objective is 1.55 against a threshold
# of 10, but "measured no difference" is not the same as "measured the same
# thing".
CLIP_THRESHOLD = 10.0

STAGES = ("solver-dt", "lr-range", "lr-confirm", "ratio", "saturation", "verify")

# A committed learning rate within this many decades of the located optimum is
# treated as adequately scaled, and the later stages are reported as unnecessary
# rather than run. The band is generous, and has to be: the ramp's divergence
# point moved by 19x across three ramps of the same problem, depending only on
# where each started and how coarsely it stepped. One decade is wide enough to
# survive that, which is exactly why this is a screening test and lr-confirm is
# the measurement.
HEALTHY_ALPHA_DECADES = 1.0

# Ramps must start well below the rate being looked for. One that starts above
# it banks real loss reduction on the way up and then tolerates rates a cold
# start cannot, overshooting by more than an order of magnitude.
DEFAULT_ALPHA_LOW = 1e-2

# How much later a ramp diverges than a constant rate does, so the bracket it
# hands lr-confirm can be shifted back down. Measured on the collocated problem:
# the ramp survived alpha = 1e4 where a constant 3000 dies on iteration 3.
RAMP_DIVERGENCE_BIAS = 3.0

# Relative gradient error, against the finest step compared, that a coarser step
# may introduce before it is rejected. The study only has to locate a learning
# rate to within a factor of a few, so this is far tighter than it needs to be.
SOLVER_DT_GRADIENT_TOLERANCE = 1e-3


def _family_norms(tree: dict) -> dict:
    """Per-gain-family L2 norm over components, keeping the batch axis."""
    return {name: jnp.linalg.norm(tree[name], axis=-1) for name in GAIN_FAMILIES}


def windup_metrics(aux: dict) -> dict:
    """Per-start integrator diagnostics from a batched rollout.

    What the anti-windup scale controls is the integrator, so that is what
    follow-up item 2 has to be judged on. Two numbers separate the cases a loss
    comparison cannot: a large ``integral_peak`` with a small
    ``integral_terminal`` is a transient the integrator recovered from, while
    the two being equal and large is accumulated windup that never unwound.
    """
    integral = jnp.abs(aux["integral_error_ts"])
    return {
        "integral_peak": jnp.max(integral, axis=(1, 2)),
        "integral_terminal": jnp.linalg.norm(integral[:, -1, :], axis=-1),
    }


def run_alpha_schedule(
    *,
    gradient_fn,
    init_gains: dict,
    alpha_low: float,
    alpha_high: float,
    num_points: int,
    eps: float,
    clip: float | None,
    ratio: dict | None = None,
    aux_metrics: Callable[[dict], dict] | None = None,
) -> dict:
    """Ramp the learning-rate magnitude geometrically, one point per iteration.

    Args:
        gradient_fn: ``value_and_grad``-shaped callable over one start.
        init_gains: Batched initial gains, each of shape ``(batch, actuators)``.
        alpha_low: First learning-rate magnitude.
        alpha_high: Last learning-rate magnitude.
        num_points: Number of ramp points, one per iteration.
        eps: Yogi epsilon, passed through to :func:`build_optimizer`.
        clip: Global gradient-norm clip, or ``None`` for no clipping, which is
            what :func:`build_optimizer` and the generators default to.
        ratio: Per-family learning-rate ratio; defaults to :data:`LR_RATIO`.
            Setting ``alpha_low == alpha_high`` turns the ramp into a constant
            rate, which is how the confirmation runs are expressed.
        aux_metrics: Optional callable mapping a batched rollout ``aux`` to
            named per-start scalars, recorded alongside the standard
            diagnostics. Defaults to :func:`windup_metrics`, so every stage
            reports what the integrator is doing, not only the saturation sweep.

    Returns:
        Arrays of shape ``(num_points,)`` or ``(num_points, batch)`` recording
        the ramp and every diagnostic taken along it, plus ``final_gains`` of
        shape ``(batch, families, actuators)`` in :data:`GAIN_FAMILIES` order.
    """
    batch_size = int(init_gains["Kp"].shape[0])
    alphas = jnp.geomspace(alpha_low, alpha_high, num_points)
    ratio = dict(LR_RATIO if ratio is None else ratio)
    aux_metrics = windup_metrics if aux_metrics is None else aux_metrics

    # One ramp point per iteration, held at the last value if the loop outlives
    # the schedule. The optimizer comes from secvd_case, so a rate measured here
    # is applied by exactly the transform the generators run.
    optimizer = build_optimizer(
        lambda count: jnp.take(alphas, count, mode="clip"),
        ratio=ratio,
        eps=eps,
        clip=clip,
    )
    opt_vars = {"opt_ctr_params": init_gains, "opt_atr_params": {}}
    opt_state = vmap(optimizer.init)(opt_vars)

    record: dict[str, list] = {
        key: []
        for key in ("loss", "grad_norm", "update_norm", "gain_norm", "relative_step")
    }
    record["time_iter"] = []

    for index in range(num_points):
        started = time.time()
        (loss, aux), grad = vmap(gradient_fn)(opt_vars)
        gains = opt_vars["opt_ctr_params"]
        grad_families = _family_norms(grad["opt_ctr_params"])
        gain_families = _family_norms(gains)

        updates, opt_state = vmap(optimizer.update)(grad, opt_state, opt_vars)
        update_families = _family_norms(updates["opt_ctr_params"])
        opt_vars = optax.apply_updates(opt_vars, updates)
        jax.block_until_ready((loss, opt_vars))

        record["loss"].append(np.asarray(loss))
        for key, value in aux_metrics(aux).items():
            record.setdefault(key, []).append(np.asarray(value))
        for key, families in (
            ("grad_norm", grad_families),
            ("update_norm", update_families),
            ("gain_norm", gain_families),
        ):
            record[key].append(
                np.stack([np.asarray(families[n]) for n in GAIN_FAMILIES], axis=-1)
            )
        # Relative movement is the quantity that matters when Kp ~ 50 and
        # Kd ~ 1: the same absolute step means very different things to them.
        record["relative_step"].append(
            record["update_norm"][-1] / np.maximum(record["gain_norm"][-1], 1e-30)
        )
        record["time_iter"].append(time.time() - started)

        alpha = float(alphas[index])
        finite = bool(np.isfinite(record["loss"][-1]).all())
        print(
            f"  {index + 1:>3d}/{num_points}  alpha = {alpha:10.4g}  "
            f"loss = {np.median(record['loss'][-1]):.6e}  "
            f"|g| = {np.median(record['grad_norm'][-1]):.3e}  "
            f"rel step = {np.median(record['relative_step'][-1]):.3e}"
            + ("" if finite else "   [NON-FINITE]"),
            flush=True,
        )
        if not finite:
            print(f"  ramp diverged at alpha = {alpha:.4g}; stopping.", flush=True)
            break

    stacked = {key: np.asarray(values) for key, values in record.items()}
    stacked["final_gains"] = np.stack(
        [np.asarray(opt_vars["opt_ctr_params"][n]) for n in GAIN_FAMILIES], axis=1
    )
    stacked["alpha"] = np.asarray(alphas)[: len(stacked["loss"])]
    stacked["batch_size"] = np.asarray(batch_size)
    stacked["eps"] = np.asarray(eps)
    stacked["clip"] = np.asarray(np.inf if clip is None else clip)
    stacked["ratio"] = np.asarray([ratio[n] for n in GAIN_FAMILIES])
    return stacked


def report_step_efficiency(result: dict) -> None:
    """Report how much of the requested learning rate reaches the parameters.

    The first ramp made it clear that guessing at Yogi's internal regime does
    not work: both the scale-invariant prediction (``|update| ~ lr``) and the
    eps-dominated one (``|update| ~ lr * |grad| / eps``) missed the observation
    by one to two orders of magnitude, so picking the closer of two wrong models
    told us nothing.

    What is directly measurable, and turned out to be the informative quantity,
    is the ratio ``|update| / lr``. A well-behaved adaptive optimizer holds this
    near one. If it is small and *constant* across the ramp, the step is still
    linear in the learning rate -- nothing is saturating -- and the learning
    rate is simply too small by that factor.
    """
    alpha = result["alpha"]
    ratio = result.get("ratio", np.array([LR_RATIO[n] for n in GAIN_FAMILIES]))
    print("  family   median |update|/lr   spread over the ramp")
    for position, name in enumerate(GAIN_FAMILIES):
        lr = alpha * float(ratio[position])
        observed = np.median(result["update_norm"][..., position], axis=-1)
        efficiency = observed / np.maximum(lr, 1e-30)
        # A ramp that diverges ends in NaN, which is the interesting outcome
        # rather than a failure: summarize what it managed before that.
        finite = efficiency[np.isfinite(efficiency)]
        if finite.size == 0:
            print(f"  {name:>5}   no finite iterations")
            continue
        print(
            f"  {name:>5}   {np.median(finite):16.3e}   "
            f"{finite.min():.2e} .. {finite.max():.2e}"
        )
    print(
        "  (a well-scaled adaptive optimizer sits near 1; a small constant "
        "value means\n   the step is linear in lr and lr is short by that factor)"
    )


def save_schedule(result: dict, out_dir: Path, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{tag}.npz"
    np.savez_compressed(npz_path, **result)
    csv_path = out_dir / f"{tag}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["alpha", "loss_median"]
            + [
                f"{key}_{name}"
                for key in ("grad", "update", "rel")
                for name in GAIN_FAMILIES
            ]
        )
        for index, alpha in enumerate(result["alpha"]):
            row = [f"{alpha:.6g}", f"{np.median(result['loss'][index]):.6e}"]
            for key in ("grad_norm", "update_norm", "relative_step"):
                row += [f"{np.median(result[key][index, :, p]):.6e}" for p in range(3)]
            writer.writerow(row)
    return npz_path


def _sample_starts(problem, batch_size: int, seed: int) -> dict:
    """Draw the same starts the generators use, so losses stay comparable.

    The centre is the nominal gains scaled by the method's
    :data:`secvd_case.INIT_GAIN_SCALE`, which is what the generators sample --
    not the nominal gains themselves. Tuning a learning rate at a different
    initialization than the run uses would measure the wrong landscape, which is
    the whole failure the shared builder exists to prevent.
    """
    scale = INIT_GAIN_SCALE[problem.method]
    centre = {
        name: value * scale[name] for name, value in problem.nominal_gains.items()
    }
    return sample_initial_gains(
        centre,
        batch_size=batch_size,
        seed=seed,
        scheme=SECVD_INIT_SCHEME,
        spread=SECVD_INIT_SPREAD,
    )


def stiffest_start(init_gains: dict) -> int:
    """Return the start whose closed loop is hardest to integrate.

    Every start shares the plant, the initial state and the setpoint -- only the
    gains differ -- so the most demanding system is the one with the largest
    proportional gain, not the one that tracks worst. Those are different starts
    on the committed collocated batch: ``Kp`` peaks at start 2, which also has
    the *best* tracking, while the worst tracking belongs to start 1, which has
    the lowest ``Kp`` and is the easiest thing the solver ever sees.
    """
    return int(np.argmax(np.linalg.norm(np.asarray(init_gains["Kp"]), axis=-1)))


def _gradient_vector(grad: dict) -> np.ndarray:
    return np.concatenate(
        [np.asarray(grad["opt_ctr_params"][n]).reshape(-1) for n in GAIN_FAMILIES]
    )


def stage_solver_dt(args, case) -> dict:
    """Measure how far the integration step can be raised.

    Cost is linear in the step count, so this is by far the largest speed lever
    in the study, and every later stage inherits what it settles on. It runs on
    one start -- the stiffest -- because this asks how hard the dynamics are to
    integrate, not how the optimizer behaves, and one representative system
    answers that.

    The gate is the **gradient**, not the loss: the study exists to compute
    gradients, and a loss can stay accurate while its derivative drifts.
    """
    reference = build_evaluator(
        args.method, case=case, horizon=args.horizon, e_sat=args.e_sat
    )
    init_gains = _sample_starts(reference, args.batch_size, args.seed)
    index = stiffest_start(init_gains)
    gains = {name: value[index] for name, value in init_gains.items()}
    print(
        f"Stiffest of {args.batch_size} starts is index {index} "
        f"(|Kp| = {float(np.linalg.norm(np.asarray(gains['Kp']))):.1f})",
        flush=True,
    )
    header = (
        f"{'solver_dt':>10} {'steps':>8} {'eval':>8} {'loss':>14} "
        f"{'d_loss':>9} {'d_grad':>9}  verdict"
    )
    print(header, flush=True)
    baseline = None
    rows = []
    for solver_dt in sorted(args.solver_dts):
        problem = build_evaluator(
            args.method,
            case=case,
            horizon=args.horizon,
            solver_dt=solver_dt,
            e_sat=args.e_sat,
        )
        opt_vars = {"opt_ctr_params": gains, "opt_atr_params": {}}
        # Time the second call: the first pays compilation, which is a fixed
        # cost per configuration and would swamp the ratio this stage reports.
        (loss, _aux), grad = problem.gradient_fn(opt_vars)
        float(loss)
        started = time.time()
        (loss, _aux), grad = problem.gradient_fn(opt_vars)
        vector = _gradient_vector(grad)
        elapsed = time.time() - started
        loss = float(loss)
        if baseline is None:
            baseline = (loss, vector)
        d_loss = abs(loss - baseline[0]) / abs(baseline[0])
        d_grad = float(
            np.linalg.norm(vector - baseline[1]) / np.linalg.norm(baseline[1])
        )
        finite = bool(np.isfinite(loss) and np.all(np.isfinite(vector)))
        accepted = bool(finite and d_grad <= SOLVER_DT_GRADIENT_TOLERANCE)
        rows.append((solver_dt, elapsed, loss, d_loss, d_grad, accepted))
        print(
            f"{solver_dt:>10.1e} {int(args.horizon / solver_dt):>8d} "
            f"{elapsed:>7.1f}s {loss:>14.6e} {d_loss:>9.2e} {d_grad:>9.2e}  "
            f"{'ok' if accepted else 'REJECTED'}",
            flush=True,
        )
    accepted = [row for row in rows if row[5]]
    largest = max(row[0] for row in accepted)
    fastest = next(row[1] for row in rows if row[0] == largest)
    print("")
    print(
        f"Largest step within {SOLVER_DT_GRADIENT_TOLERANCE:g} relative gradient "
        f"error: solver_dt = {largest:g}, {rows[0][1] / fastest:.1f}x faster than "
        f"{rows[0][0]:g}"
    )
    print(
        "The controller is queried at every integration step, so this is also "
        "the effective control rate."
    )
    return {
        "stage": np.asarray("solver-dt"),
        "solver_dt": np.asarray([row[0] for row in rows]),
        "eval_seconds": np.asarray([row[1] for row in rows]),
        "loss": np.asarray([row[2] for row in rows]),
        "rel_loss_error": np.asarray([row[3] for row in rows]),
        "rel_gradient_error": np.asarray([row[4] for row in rows]),
        "accepted": np.asarray([row[5] for row in rows]),
        "recommended_solver_dt": np.asarray(largest),
        "start_index": np.asarray(index),
    }


def locate_alpha(result: dict) -> tuple[float, float]:
    """Return the ``(low, high)`` bracket a ramp puts the usable rate in.

    A ramp yields one trustworthy number -- where it diverged -- and one that is
    an artefact of how it was configured. Standard range-test practice reports a
    third of the divergence point as *the* learning rate; on this problem that
    was 3.3x high, so this returns the bracket instead and leaves the point
    estimate to :func:`stage_constant_rates`, which measures it from cold.

    The bracket is a decade wide and sits below the divergence point, because a
    ramp diverges *later* than a cold run at the same rate, for the same reason
    its point estimate is high. Measured once, on the collocated problem: the
    ramp survived to ``1e4`` where a constant ``3000`` dies on its third
    iteration, a factor of :data:`RAMP_DIVERGENCE_BIAS`. That gives a bracket of
    ``[333, 3333]``, which contains both the confirmed rate of 1000 and the cold
    divergence point of 3000.

    One method is one calibration point, so the shift is deliberately modest and
    the bracket deliberately wide; the confirm scan straddles the boundary
    either way, and a diverging point costs two iterations to discover.

    Why not the divergence point alone: it moved by 19x across three ramps of
    the same problem, tracking only how much loss reduction each had banked on
    the way up. A ramp starting above the usable rate arrives at high rates
    already optimized and survives rates a cold start cannot -- which is why
    :data:`DEFAULT_ALPHA_LOW` starts two decades below the committed rate.

    A ramp that never diverged brackets its own top, and the caller should widen
    the span rather than trust it.
    """
    alpha = np.asarray(result["alpha"])
    loss = np.median(np.asarray(result["loss"]), axis=-1)
    finite = np.isfinite(loss)
    divergence = float(alpha[~finite][0]) if (~finite).any() else float(alpha[-1])
    high = divergence / RAMP_DIVERGENCE_BIAS
    return high / 10.0, high


def bracket_alphas(low: float, high: float, count: int = 3) -> list[float]:
    """Constant rates spanning a bracket, for :func:`stage_constant_rates`."""
    return [float(value) for value in np.geomspace(low, high, count)]


def report_alpha_gate(result: dict, committed_alpha: float) -> bool:
    """Say whether the committed rate is short, and by how much.

    A method that passes is reported as adequately scaled *with the measured
    number*, which is a result rather than a skipped stage.
    """
    low, high = locate_alpha(result)
    divergence = high
    decades = float(np.log10(low / committed_alpha))
    short = decades > HEALTHY_ALPHA_DECADES
    print("")
    print(
        f"Ramp diverged at alpha = {divergence * RAMP_DIVERGENCE_BIAS:.3g}, "
        f"bracketing the usable rate in [{low:.3g}, {divergence:.3g}]; "
        f"committed is {committed_alpha:g}, {decades:+.2f} decades below it."
    )
    if short:
        rates = " ".join(f"{value:g}" for value in bracket_alphas(low, divergence))
        print("  SHORT. Measure it from cold:")
        print(f"    --stage lr-confirm --alphas {rates}")
    else:
        print(
            f"  Adequately scaled: within the {HEALTHY_ALPHA_DECADES:g}-decade "
            "band of the bracket, so no retuning is needed."
        )
    return short


def stage_constant_rates(args, case, alphas, ratio, label) -> dict:
    """Run one independent constant-rate optimization per configuration.

    A ramp locates a region but cannot measure it: it carries warmed-up
    optimizer moments and already-moved gains into every new rate. These runs
    each start cold from the same initializations, which is what makes their
    final losses comparable, and is why the collocated ramp's 6.4e4
    recommendation collapsed to 1000 when measured this way.
    """
    problem = build_evaluator(
        args.method,
        case=case,
        horizon=args.horizon,
        solver_dt=args.solver_dt,
        e_sat=args.e_sat,
        loss_scale=args.loss_scale,
    )
    init_gains = _sample_starts(problem, args.batch_size, args.seed)
    results = {}
    for alpha in alphas:
        print("")
        print(f"[{label}] alpha = {alpha:g}, ratio = {ratio}", flush=True)
        results[alpha] = run_alpha_schedule(
            gradient_fn=problem.gradient_fn,
            init_gains=init_gains,
            alpha_low=alpha,
            alpha_high=alpha,
            num_points=args.num_iters,
            eps=args.eps,
            clip=args.clip,
            ratio=ratio,
        )
    print("")
    print(f"{label} summary:")
    print(f"  {'alpha':>12}  {'final loss':>13}  {'improvement':>11}")
    best = None
    for alpha, result in results.items():
        loss = np.median(np.asarray(result["loss"]), axis=-1)
        finite = loss[np.isfinite(loss)]
        if finite.size == 0:
            print(f"  {alpha:>12.4g}  {'diverged':>13}")
            continue
        improvement = 100 * (1 - finite.min() / finite[0])
        print(f"  {alpha:>12.4g}  {finite.min():>13.6e}  {improvement:>10.2f} %")
        if best is None or finite.min() < best[1]:
            best = (alpha, float(finite.min()))
    if best is not None:
        print("")
        print(f"Best: alpha = {best[0]:g} at loss {best[1]:.6e}")
    return results


def saturation_window(problem) -> tuple[float, float]:
    """The physically meaningful range for ``e_sat``, measured from the model.

    ``e_sat`` is a *metre* scale on the error the PID integrates, so it can be
    bounded from both sides by errors this plant actually produces, without any
    sweep:

    * **Lower bound, the steady-state error.** Below it the integrator is
      clamped at a level it can never integrate away, so the offset the I-term
      exists to remove becomes permanent.
    * **Upper bound, the initial step.** Above it ``tanh`` never leaves its
      linear region, so the saturation is not merely mild -- it is absent, and
      the controller is a plain PID.

    Anything outside that window is a knob that provably does nothing or a
    controller that provably cannot converge, so a sweep belongs inside it.
    Measured at the nominal gains, which is what makes it a model-based bound
    rather than a result of the tuning.
    """
    (_loss, aux), _grad = problem.gradient_fn(
        {"opt_ctr_params": problem.nominal_gains, "opt_atr_params": {}}
    )
    error = np.abs(np.asarray(problem.control_error_fn(aux)))
    return float(error[-1].max()), float(error[0].max())


def stage_saturation(args, case, alpha, ratio) -> dict:
    """Sensitivity of the optimization to the saturation scale -- item 2.

    **This stage does not select ``e_sat``.** The committed 1e-2 m is a
    controller design choice, made on physical grounds in PR #135 and kept: it
    is the error magnitude above which the ``tanh`` starts compressing what
    reaches the integrator, and its purpose is to bound the integral state, not
    to trade tracking against anything.

    Selecting it by loss would be wrong on its own terms. The objective has no
    stability term, no actuator model and a finite horizon, so a ranking over
    ``e_sat`` says how much integral action a 5 s window happens to want -- a
    quantity that moves with the horizon -- rather than anything about windup.
    The measured ordering is monotone toward *tighter* saturation, which is that
    effect and not a recommendation.

    What follow-up item 2 asks is whether the committed choice affects tracking,
    windup, gradients and the optimized gains. That is a sensitivity check, and
    it is what this sweeps for: points spanning the range over which the scale
    can do anything at all, each reporting the loss, how much of the rollout was
    actually in saturation, how far the integrator wound up, and -- for learning
    stability -- the gradient magnitude and whether every iteration stayed
    finite. The optimizer runs at the confirmed learning rate so the check is
    made on an optimizer that actually moves.
    """
    probe = build_evaluator(
        args.method, case=case, horizon=args.horizon, solver_dt=args.solver_dt
    )
    steady, step = saturation_window(probe)
    print("")
    print(
        f"Range over which e_sat can act at all: [{steady:.3g}, {step:.3g}] m "
        f"-- the steady-state error up to the peak, at the nominal gains. "
        f"Below it the integrator is clamped beneath the offset it exists to "
        f"remove; above it tanh never leaves its linear region. Points are "
        f"placed across it to bracket the committed 1e-2 m, not to search it."
    )
    e_sats = args.e_sats
    if e_sats is None:
        e_sats = [float(v) for v in np.geomspace(steady, step, 5)]
        print(f"  Sweeping it: {' '.join(f'{v:.3g}' for v in e_sats)}")
    else:
        outside = [v for v in e_sats if not steady <= v <= step]
        if outside:
            print(
                "  Requested "
                + ", ".join(f"{v:g}" for v in outside)
                + " m falls outside it, so those points measure a controller "
                "that is either permanently offset or unsaturated."
            )
    results = {}
    summary = []
    for e_sat in e_sats:
        problem = build_evaluator(
            args.method,
            case=case,
            horizon=args.horizon,
            solver_dt=args.solver_dt,
            e_sat=e_sat,
            loss_scale=args.loss_scale,
        )

        def metrics(aux, e_sat=e_sat, problem=problem):
            # Against the error the PID *integrates*, which is the controller's
            # own -- actuated coordinates in metres for the collocated
            # regulator, task-space pose for the synergistic one. Not the
            # configuration-space error the loss is built from: comparing that
            # to e_sat compares different spaces in different units.
            error = jnp.abs(vmap(problem.control_error_fn)(aux))
            return windup_metrics(aux) | {
                "saturated_fraction": jnp.mean(error > e_sat, axis=(1, 2))
            }

        print("")
        print(f"[saturation] e_sat = {e_sat:g} m, alpha = {alpha:g}", flush=True)
        result = run_alpha_schedule(
            gradient_fn=problem.gradient_fn,
            init_gains=_sample_starts(problem, args.batch_size, args.seed),
            alpha_low=alpha,
            alpha_high=alpha,
            num_points=args.num_iters,
            eps=args.eps,
            clip=args.clip,
            ratio=ratio,
            aux_metrics=metrics,
        )
        result["e_sat"] = np.asarray(e_sat)
        results[e_sat] = result
        # Diagnostics are read off the iterations that actually evaluated. A
        # start that goes non-finite poisons the later rows, and reporting an
        # integral norm of NaN says nothing about the saturation scale.
        loss = np.median(np.asarray(result["loss"]), axis=-1)
        live = np.isfinite(loss)
        last = int(np.flatnonzero(live)[-1]) if live.any() else 0
        summary.append(
            (
                e_sat,
                float(loss[live].min()) if live.any() else float("nan"),
                float(np.median(result["saturated_fraction"][last])),
                float(np.median(result["integral_peak"][live].max(axis=0)))
                if live.any()
                else float("nan"),
                float(np.median(result["integral_terminal"][last])),
                # Over every family and start, so one number stands for
                # "did the gradients stay in the same regime".
                float(np.median(result["grad_norm"][live]))
                if live.any()
                else float("nan"),
                int(live.sum()),
            )
        )

    print("")
    print(f"Saturation sensitivity, medians over starts, {args.num_iters} iterations.")
    print("'saturated' is the fraction of the rollout with |e| > e_sat, on the")
    print("error the PID integrates -- actuation space here, not the loss's.")
    print(
        f"  {'e_sat [m]':>10}  {'best loss':>13}  {'saturated':>10}  "
        f"{'peak |I|':>10}  {'final |I|':>10}  {'|grad|':>10}  {'iters':>6}"
    )
    for e_sat, loss, fraction, peak, terminal, grad, live in summary:
        print(
            f"  {e_sat:>10.4g}  {loss:>13.6e}  {100 * fraction:>9.1f}%  "
            f"{peak:>10.3e}  {terminal:>10.3e}  {grad:>10.3e}  {live:>6d}"
        )
    report_saturation_sensitivity(summary, args.num_iters)
    return results


def report_saturation_sensitivity(summary: list, num_iters: int) -> None:
    """State whether the saturation scale reached the optimization at all.

    The verdict follow-up item 2 asks for is a comparison, not a ranking: how
    far the loss, the gradient magnitude and the run's survival move across the
    whole admissible range of ``e_sat``, relative to how far ``e_sat`` itself
    moved. If they barely move, the committed choice is not steering the
    optimization and there is nothing to revisit.
    """
    usable = [row for row in summary if np.isfinite(row[1])]
    print("")
    if len(usable) < 2:
        print("Too few finite points to judge sensitivity.")
        return

    scales = [row[0] for row in usable]
    losses = [row[1] for row in usable]
    peaks = [row[3] for row in usable]
    grads = [row[5] for row in usable]
    span = max(scales) / min(scales)
    loss_spread = max(losses) / min(losses) - 1.0
    grad_spread = max(grads) / min(grads)
    peak_spread = max(peaks) / min(peaks)
    stalled = [row[0] for row in summary if row[6] < num_iters]

    print(
        f"Across {span:.0f}x in e_sat: loss moves {100 * loss_spread:.3f} %, "
        f"gradient magnitude {grad_spread:.2f}x, integrator peak "
        f"{peak_spread:.1f}x."
    )
    if stalled:
        print(
            "  Runs that did not complete, at e_sat = "
            + ", ".join(f"{value:g}" for value in stalled)
            + " m."
        )
    else:
        print(f"  Every point completed all {num_iters} iterations.")
    settled = loss_spread < 1e-2 and grad_spread < 2.0 and not stalled
    print(
        "  The scale reaches the integrator and not the optimization: keeping "
        f"the committed e_sat = {COMMITTED_E_SAT:g} m."
        if settled
        else "  The scale does reach the optimization; report it with the run."
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument("--stage", choices=STAGES, default="lr-range")
    parser.add_argument("--method", choices=METHODS, default="collocated")
    parser.add_argument("--horizon", type=float, default=CASE_HORIZON)
    parser.add_argument(
        "--solver-dt",
        type=float,
        default=None,
        help="Integration step for every stage but solver-dt; defaults to the case value.",
    )
    parser.add_argument(
        "--solver-dts",
        type=float,
        nargs="+",
        default=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
        help="Steps the solver-dt stage compares against the finest.",
    )
    parser.add_argument("--e-sat", type=float, default=COMMITTED_E_SAT)
    parser.add_argument(
        "--e-sats",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Saturation scales in metres. Defaults to five points across the "
            "model-based window the saturation stage measures."
        ),
    )
    parser.add_argument("--loss-scale", type=float, default=1.0)
    parser.add_argument(
        "--ratio",
        type=float,
        nargs=3,
        metavar=("KP", "KI", "KD"),
        default=[LR_RATIO[n] for n in GAIN_FAMILIES],
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0],
        help="Constant rates the lr-confirm stage compares.",
    )
    parser.add_argument("--alpha-low", type=float, default=DEFAULT_ALPHA_LOW)
    parser.add_argument("--alpha-high", type=float, default=1e6)
    # Three points per decade. The ramp only has to say whether the committed
    # rate is short and which way; lr-confirm measures the value actually used.
    parser.add_argument("--num-points", type=int, default=25)
    parser.add_argument("--num-iters", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument(
        "--clip",
        type=float,
        default=None,
        help=(
            "Global gradient-norm clip. Off by default, matching the "
            f"generators; pass --clip {CLIP_THRESHOLD:g} to reproduce the "
            "committed configuration."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=SECVD_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SECVD_INIT_SEED)
    parser.add_argument("--out-dir", type=Path, default=TUNING_DIR)
    # Consumed before JAX is imported; declared so argparse accepts it here too.
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tag = args.tag or f"{args.stage}_{args.method}_T{args.horizon:g}"
    case = build_sec_vd_case()
    probe = build_evaluator(args.method, case=case, horizon=args.horizon)
    print(
        f"[{args.stage}] {args.method}, horizon {args.horizon} s, "
        f"batch {args.batch_size}, eps {args.eps:g}, "
        f"clip {'off' if args.clip is None else format(args.clip, 'g')}",
        flush=True,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "solver-dt":
        result = stage_solver_dt(args, case)
        np.savez_compressed(args.out_dir / f"{tag}.npz", **result)
        print(f"Saved {args.out_dir / (tag + '.npz')}")
        return

    if args.stage == "saturation" and not probe.supports_saturation:
        # Gated on the problem, never on the method name: a controller that does
        # not saturate its integral error has nothing for item 2 to sweep.
        print("")
        print(
            f"Skipped: the {args.method} controller does not saturate its "
            "integral error, so there is no saturation scale to tune."
        )
        return

    ratio = dict(zip(GAIN_FAMILIES, args.ratio, strict=True))
    if args.stage in ("lr-range", "verify"):
        problem = build_evaluator(
            args.method,
            case=case,
            horizon=args.horizon,
            solver_dt=args.solver_dt,
            e_sat=args.e_sat,
            loss_scale=args.loss_scale,
        )
        result = run_alpha_schedule(
            gradient_fn=problem.gradient_fn,
            init_gains=_sample_starts(problem, args.batch_size, args.seed),
            alpha_low=args.alpha_low,
            alpha_high=args.alpha_high,
            num_points=args.num_points,
            eps=args.eps,
            clip=args.clip,
            ratio=ratio,
        )
        print("")
        print("Step efficiency, per gain family:")
        report_step_efficiency(result)
        report_alpha_gate(result, probe.committed_alpha)
        print(f"Saved {save_schedule(result, args.out_dir, tag)}")
        return

    if args.stage == "lr-confirm":
        results = stage_constant_rates(args, case, args.alphas, ratio, "lr-confirm")
    elif args.stage == "ratio":
        alpha = args.alpha or args.alphas[len(args.alphas) // 2]
        candidates = [
            ("committed", dict(LR_RATIO)),
            ("equal", dict.fromkeys(GAIN_FAMILIES, 1.0)),
        ]
        # --ratio defaults to the committed split, so without this the stage
        # spends a third of its budget recomputing a number it already has.
        if any(ratio == candidate for _, candidate in candidates):
            print("")
            print("--ratio matches a candidate already scanned; not repeating it.")
        else:
            candidates.append(("custom", ratio))
        results = {
            name: stage_constant_rates(args, case, [alpha], candidate, f"ratio:{name}")[
                alpha
            ]
            for name, candidate in candidates
        }
    else:
        results = stage_saturation(args, case, args.alpha or 1000.0, ratio)

    for key, result in results.items():
        suffix = f"{key:g}" if isinstance(key, float) else str(key)
        print(f"Saved {save_schedule(result, args.out_dir, tag + '_' + suffix)}")


if __name__ == "__main__":
    main()
