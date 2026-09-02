"""Tuning study for the Section Vd optimizer and controller saturation.

This is a study, not a results generator: it writes small summaries to
``data/tuning/`` and never a schema-v2 paper archive. Follow-up items 2 and 6 on
[issue #154](https://github.com/tud-phi/soromox/issues/154) both ask for values
to be *tuned* against something observable, and the first regenerated collocated
run showed why that has to come before any further results.

## What the first full run showed

At 100 iterations and six starts the median per-start improvement was 1.11 %.
The loss was still descending at the last iteration -- the optimizer was not
converged, it was barely moving. Two measurements from that archive explain it:

* gradient norms peaked at ``4.8e-4``, five orders of magnitude below the
  ``clip_by_global_norm(10.0)`` threshold, so the clip never bound once;
* ``optax.yogi`` defaults to ``eps=1e-3``, not Adam's ``1e-8``.

Yogi is scale-invariant only while ``sqrt(v) >> eps``. Below that floor it
degenerates into SGD with an effective rate of ``lr / eps``. With gradients at
``1.8e-4`` the case sits firmly below the floor, which is exactly the failure
the follow-up predicted when it warned that removing the ``1e3`` global factor
"would not be equivalent for Yogi and gradient clipping". Making the objective
dimensionless was right; leaving the learning rates calibrated for the old
scale was not.

## Stage 1: the coupled range test

Three learning rates cannot be searched directly at full fidelity -- even a
coarse ``4x4x4`` grid is tens of hours. So the search is decomposed into a
magnitude and a ratio,

```text
(lr_P, lr_I, lr_D) = alpha * (1, 0.5, 0.1)
```

where the ratio is the one currently in the generators (``0.5, 0.25, 0.05``, so
the current ``alpha`` is 0.5). Only ``alpha`` is ramped here, geometrically over
one run: 60 points across six decades. Six decades because the two hypotheses
sit far apart -- an ``eps``-dominated optimizer needs ``alpha`` above 100 to
move at all, a scale-invariant one wants ``alpha`` near 1 -- and one ramp should
discriminate between them rather than assume.

A range test is a single trajectory through parameter space, not independent
runs at each rate: the loss at high ``alpha`` reflects where the earlier steps
left the parameters. That is what makes it cheap, and why it locates a region
rather than measuring a value. The chosen rate is confirmed on independent runs
in a later stage.

## Reading the regime

Rather than reaching into the optimizer's internal state, whose layout differs
across optax versions and nests under ``multi_transform``, each iteration
records the observed update norm against the two regimes' predictions:

```text
scale-invariant   |update| ~ lr * sqrt(n)        (independent of |grad|)
eps-dominated     |update| ~ lr * |grad| / eps
```

Whichever prediction tracks the observation identifies the regime, and the
comparison holds whatever optax does internally.
"""

from __future__ import annotations

import csv
import sys
import time
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
    COMMITTED_LR_RATIO,
    METHODS,
    build_evaluator,
    build_sec_vd_case,
)
from secvd_init import (  # noqa: E402
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
GAIN_FAMILIES = ("Kp", "Ki", "Kd")
FAMILY_LABEL = {"Kp": "P", "Ki": "I", "Kd": "D"}
# The committed scaling comes from secvd_case, so the study cannot drift from
# what the generators actually run.
LR_RATIO = COMMITTED_LR_RATIO

# The committed clip threshold. Kept so the ramp reports when it starts to bind
# rather than silently changing the configuration under test.
CLIP_THRESHOLD = 10.0

STAGES = ("solver-dt", "lr-range", "lr-confirm", "ratio", "saturation", "verify")

# A committed learning rate within this many decades of the located optimum is
# treated as adequately scaled, and the later stages are reported as unnecessary
# rather than run. The band is generous because a ramp locates rather than
# measures: on the collocated problem it recommended 6.4e4 where the
# constant-rate scan found 1000, a 64x overshoot.
HEALTHY_ALPHA_DECADES = 1.0

# Relative gradient error, against the finest step compared, that a coarser step
# may introduce before it is rejected. The study only has to locate a learning
# rate to within a factor of a few, so this is far tighter than it needs to be.
SOLVER_DT_GRADIENT_TOLERANCE = 1e-3


def _family_norms(tree: dict) -> dict:
    """Per-gain-family L2 norm over components, keeping the batch axis."""
    return {name: jnp.linalg.norm(tree[name], axis=-1) for name in GAIN_FAMILIES}


def run_alpha_schedule(
    *,
    gradient_fn,
    init_gains: dict,
    alpha_low: float,
    alpha_high: float,
    num_points: int,
    eps: float,
    use_clip: bool,
    ratio: dict | None = None,
) -> dict:
    """Ramp the learning-rate magnitude geometrically, one point per iteration.

    Args:
        gradient_fn: ``value_and_grad``-shaped callable over one start.
        init_gains: Batched initial gains, each of shape ``(batch, actuators)``.
        alpha_low: First learning-rate magnitude.
        alpha_high: Last learning-rate magnitude.
        num_points: Number of ramp points, one per iteration.
        eps: ``optax.yogi`` epsilon. The default of ``1e-3`` is what puts this
            case below Yogi's adaptivity floor.
        use_clip: Whether to keep the committed ``clip_by_global_norm``.
        ratio: Per-family learning-rate ratio; defaults to :data:`LR_RATIO`.
            Setting ``alpha_low == alpha_high`` turns the ramp into a constant
            rate, which is how the confirmation runs are expressed.

    Returns:
        Arrays of shape ``(num_points,)`` or ``(num_points, batch)`` recording
        the ramp and every diagnostic taken along it.
    """
    batch_size = int(init_gains["Kp"].shape[0])
    alphas = jnp.geomspace(alpha_low, alpha_high, num_points)
    ratio = dict(LR_RATIO if ratio is None else ratio)

    def schedule_for(family: str):
        scale = ratio[family]
        return lambda count: scale * jnp.take(alphas, count, mode="clip")

    def transform_for(family: str):
        yogi = optax.yogi(schedule_for(family), eps=eps)
        if not use_clip:
            return yogi
        return optax.chain(optax.clip_by_global_norm(CLIP_THRESHOLD), yogi)

    optimizer = optax.multi_transform(
        {FAMILY_LABEL[name]: transform_for(name) for name in GAIN_FAMILIES},
        {
            "opt_ctr_params": {name: FAMILY_LABEL[name] for name in GAIN_FAMILIES},
            "opt_atr_params": {},
        },
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
        (loss, _aux), grad = vmap(gradient_fn)(opt_vars)
        gains = opt_vars["opt_ctr_params"]
        grad_families = _family_norms(grad["opt_ctr_params"])
        gain_families = _family_norms(gains)

        updates, opt_state = vmap(optimizer.update)(grad, opt_state, opt_vars)
        update_families = _family_norms(updates["opt_ctr_params"])
        opt_vars = optax.apply_updates(opt_vars, updates)
        jax.block_until_ready((loss, opt_vars))

        record["loss"].append(np.asarray(loss))
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
    stacked["alpha"] = np.asarray(alphas)[: len(stacked["loss"])]
    stacked["batch_size"] = np.asarray(batch_size)
    stacked["eps"] = np.asarray(eps)
    stacked["clip"] = np.asarray(CLIP_THRESHOLD if use_clip else np.inf)
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
    """Draw the same starts the generators use, so losses stay comparable."""
    return sample_initial_gains(
        problem.nominal_gains,
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
    """Return ``(recommended, divergence)`` alpha from a ramp.

    Standard range-test practice takes a third of the divergence point rather
    than the minimum, which sits adjacent to it and is usually already
    oscillating. A ramp that never diverged reports its own top, and the caller
    should widen the span rather than trust it.
    """
    alpha = np.asarray(result["alpha"])
    loss = np.median(np.asarray(result["loss"]), axis=-1)
    finite = np.isfinite(loss)
    divergence = float(alpha[~finite][0]) if (~finite).any() else float(alpha[-1])
    return divergence / 3.0, divergence


def report_alpha_gate(result: dict, committed_alpha: float) -> bool:
    """Say whether the committed rate is short, and by how much.

    A method that passes is reported as adequately scaled *with the measured
    number*, which is a result rather than a skipped stage.
    """
    recommended, divergence = locate_alpha(result)
    decades = float(np.log10(recommended / committed_alpha))
    short = decades > HEALTHY_ALPHA_DECADES
    print("")
    print(
        f"Located alpha ~ {recommended:.3g} (diverged at {divergence:.3g}); "
        f"committed is {committed_alpha:g}, {decades:+.2f} decades away."
    )
    if short:
        print("  SHORT: run lr-confirm to measure a usable constant rate.")
    else:
        print(
            f"  Adequately scaled: inside the {HEALTHY_ALPHA_DECADES:g}-decade "
            "band, so no retuning is needed."
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
            use_clip=not args.no_clip,
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
    parser.add_argument("--e-sat", type=float, default=1e-2)
    parser.add_argument(
        "--e-sats",
        type=float,
        nargs="+",
        default=[1e-3, 3.2e-3, 1e-2, 3.2e-2, 1e-1],
        help="Saturation scales in metres, swept by the saturation stage.",
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
    parser.add_argument("--alpha-low", type=float, default=1e-2)
    parser.add_argument("--alpha-high", type=float, default=1e6)
    # Three points per decade. The ramp only has to say whether the committed
    # rate is short and which way; lr-confirm measures the value actually used.
    parser.add_argument("--num-points", type=int, default=25)
    parser.add_argument("--num-iters", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--no-clip", action="store_true")
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
        f"clip {'off' if args.no_clip else CLIP_THRESHOLD}",
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
            use_clip=not args.no_clip,
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
        results = {}
        for name, candidate in (
            ("committed", dict(LR_RATIO)),
            ("equal", dict.fromkeys(GAIN_FAMILIES, 1.0)),
            ("custom", ratio),
        ):
            results[name] = stage_constant_rates(
                args, case, [alpha], candidate, f"ratio:{name}"
            )[alpha]
    else:
        alpha = args.alpha or 1000.0
        results = {}
        for e_sat in args.e_sats:
            args.e_sat = e_sat
            results[e_sat] = stage_constant_rates(
                args, case, [alpha], ratio, f"e_sat={e_sat:g}"
            )[alpha]

    for key, result in results.items():
        suffix = f"{key:g}" if isinstance(key, float) else str(key)
        print(f"Saved {save_schedule(result, args.out_dir, tag + '_' + suffix)}")


if __name__ == "__main__":
    main()
