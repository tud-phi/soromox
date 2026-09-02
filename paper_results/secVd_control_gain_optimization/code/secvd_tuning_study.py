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
from jax import value_and_grad, vmap  # noqa: E402
from secvd_case import build_sec_vd_case  # noqa: E402
from secvd_init import (  # noqa: E402
    SECVD_BATCH_SIZE,
    SECVD_INIT_SCHEME,
    SECVD_INIT_SEED,
    SECVD_INIT_SPREAD,
    sample_initial_gains,
)
from secvd_objective import (  # noqa: E402
    configuration_space_scales,
    time_averaged_squared_error,
)

from soromox.control import PIDControl, PIDControllerState, ReferenceTrajectory
from soromox.control.actuation_space import PotentialCompensationRegulator
from soromox.coordinate_transformations import ActuationSpaceDynamics
from soromox.systems import SystemState

jax.config.update("jax_enable_x64", True)

CASE_DIR = CODE_DIR.parent
TUNING_DIR = CASE_DIR / "data" / "tuning"

# The learning-rate ratio currently in both generators, normalized so that the
# committed setting (0.5, 0.25, 0.05) is exactly alpha = 0.5.
GAIN_FAMILIES = ("Kp", "Ki", "Kd")
FAMILY_LABEL = {"Kp": "P", "Ki": "I", "Kd": "D"}
LR_RATIO = {"Kp": 1.0, "Ki": 0.5, "Kd": 0.1}
COMMITTED_ALPHA = 0.5

# The committed clip threshold. Kept so the ramp reports when it starts to bind
# rather than silently changing the configuration under test.
CLIP_THRESHOLD = 10.0


def build_collocated_evaluator(
    *, horizon: float, e_sat: float, loss_scale: float = 1.0
):
    """Return ``(gradient_fn, nominal_gains)`` for the collocated problem.

    Mirrors ``control_gain_optimization_with_collocated.py`` exactly, except that
    the rollout horizon is settable. The solver step stays at the case value:
    gradient magnitude is what this study tunes against, so changing the
    discretization would move the target.

    Args:
        horizon: Rollout length in seconds.
        e_sat: Integral-error saturation scale in metres.
        loss_scale: Constant multiplying the objective. Adam-family optimizers
            are scale-invariant in the gradient, so this should do nothing
            except through ``eps`` and through ``clip_by_global_norm``, which is
            not scale-invariant. Exposed to test that rather than assume it.

    Returns:
        A ``value_and_grad``-shaped callable over one start's variables, and the
        nominal gains the initializations are drawn around.
    """
    case = build_sec_vd_case()
    robot = case.robot
    asd = ActuationSpaceDynamics(robot)
    save_ts = robot._compute_save_times(
        0.0, horizon, solver_dt=case.solver_dt, save_dt=case.solver_dt
    )
    reference = ReferenceTrajectory(ts=save_ts, x_des_fn=lambda _t: case.q_des)
    q_des_ts = reference.x_des_ts
    nominal = {
        "Kp": 5e1 * jnp.ones(robot.num_actuators),
        "Ki": 5e0 * jnp.ones(robot.num_actuators),
        "Kd": 1e0 * jnp.ones(robot.num_actuators),
    }
    controller = PotentialCompensationRegulator(
        actuation_space_dynamics=asd,
        reference_trajectory=reference,
        pid_control=PIDControl(**nominal, saturation_fn="tanh", gamma=1.0 / e_sat),
    )
    initial_state = SystemState(
        t=jnp.array(0.0),
        y=jnp.concatenate([case.q0, case.qd0]),
        u=jnp.zeros(robot.num_actuators),
        control_state=PIDControllerState.zero(robot.num_actuators),
    )
    num_ts = len(save_ts)
    strain_scales = configuration_space_scales(robot.params.link.length)

    def evaluate(opt_vars):
        active = controller.update_gains(opt_vars["opt_ctr_params"])
        trajectory = robot.rollout_closed_loop_to(
            initial_state=initial_state,
            controller=active,
            t1=horizon,
            solver_dt=case.solver_dt,
            save_ts=save_ts,
            max_steps=num_ts + 1,
        )
        q_ts, _qd_ts = jnp.split(trajectory.y, 2, axis=1)
        loss = time_averaged_squared_error(q_des_ts - q_ts, strain_scales, trajectory.t)
        return loss_scale * loss, {}

    return value_and_grad(evaluate, has_aux=True), nominal


def _family_norms(tree: dict) -> dict:
    """Per-gain-family L2 norm over components, keeping the batch axis."""
    return {name: jnp.linalg.norm(tree[name], axis=-1) for name in GAIN_FAMILIES}


def run_range_test(
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


def save_range_test(result: dict, out_dir: Path, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"range_test_{tag}.npz"
    np.savez_compressed(npz_path, **result)
    csv_path = out_dir / f"range_test_{tag}.csv"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--horizon", type=float, default=1.0)
    parser.add_argument("--e-sat", type=float, default=1e-2)
    parser.add_argument("--loss-scale", type=float, default=1.0)
    parser.add_argument(
        "--ratio",
        type=float,
        nargs=3,
        metavar=("KP", "KI", "KD"),
        default=[LR_RATIO[n] for n in GAIN_FAMILIES],
        help="Per-family learning-rate ratio multiplying alpha.",
    )
    parser.add_argument("--alpha-low", type=float, default=1e-3)
    parser.add_argument("--alpha-high", type=float, default=1e3)
    parser.add_argument("--num-points", type=int, default=60)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--batch-size", type=int, default=SECVD_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SECVD_INIT_SEED)
    parser.add_argument("--out-dir", type=Path, default=TUNING_DIR)
    # Consumed before JAX is imported; declared so argparse accepts it here too.
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    tag = args.tag or f"eps{args.eps:g}_T{args.horizon:g}"
    print(
        f"Range test [{tag}]: alpha {args.alpha_low:g} -> {args.alpha_high:g} "
        f"over {args.num_points} points, horizon {args.horizon} s, "
        f"eps {args.eps:g}, clip {'off' if args.no_clip else CLIP_THRESHOLD}",
        flush=True,
    )
    print(f"Committed setting is alpha = {COMMITTED_ALPHA}", flush=True)

    gradient_fn, nominal = build_collocated_evaluator(
        horizon=args.horizon, e_sat=args.e_sat, loss_scale=args.loss_scale
    )
    init_gains = sample_initial_gains(
        nominal,
        batch_size=args.batch_size,
        seed=args.seed,
        scheme=SECVD_INIT_SCHEME,
        spread=SECVD_INIT_SPREAD,
    )
    result = run_range_test(
        gradient_fn=gradient_fn,
        init_gains=init_gains,
        alpha_low=args.alpha_low,
        alpha_high=args.alpha_high,
        num_points=args.num_points,
        eps=args.eps,
        use_clip=not args.no_clip,
        ratio=dict(zip(GAIN_FAMILIES, args.ratio)),
    )
    print("\nStep efficiency, per gain family:")
    report_step_efficiency(result)

    loss = np.median(result["loss"], axis=-1)
    best = int(np.nanargmin(loss))
    print(
        f"\nLowest median loss {loss[best]:.6e} at alpha = "
        f"{result['alpha'][best]:.4g} (committed {COMMITTED_ALPHA})"
    )
    print(f"Saved {save_range_test(result, args.out_dir, tag)}")


if __name__ == "__main__":
    main()
