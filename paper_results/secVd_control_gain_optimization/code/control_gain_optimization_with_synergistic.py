"""Optimize Section Vd synergistic controller gains and write the canonical NPZ."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from secvd_init import SECVD_BATCH_SIZE  # noqa: E402
from secvd_optimization import configure_optimization_device  # noqa: E402

# Number of independently optimized gain initializations; timestep vmaps do not count.
OPTIMIZATION_BATCH_SIZE = SECVD_BATCH_SIZE
configure_optimization_device()

import jax  # noqa: E402
import optax  # noqa: E402
from gain_optimization_loop import run_gain_optimization  # noqa: E402
from jax import vmap  # noqa: E402
from secvd_case import build_evaluator, end_effector_pose_trajectory  # noqa: E402
from secvd_init import (  # noqa: E402
    describe_initial_gains,
    sample_initial_gains,
)
from secvd_objective import OBJECTIVE_NAME  # noqa: E402
from secvd_optimization import (  # noqa: E402
    parse_optimization_args,
    prepare_result_dir,
)
from secvd_results import (  # noqa: E402
    improvement_over_initial_median,
    save_optimization_outputs,
)

jax.config.update("jax_enable_x64", True)

CASE_DIR = CODE_DIR.parent


def main() -> None:
    args = parse_optimization_args(
        description="Optimize synergistic control gains for Section Vd.",
        default_result_dir=CASE_DIR / "data" / "synergistic",
        optimization_batch_size=OPTIMIZATION_BATCH_SIZE,
    )
    result_dir = prepare_result_dir(args)
    # The problem is defined once, in secvd_case, so the tuning study provably
    # tunes what this script runs (issue #154 follow-up, item 6).
    problem = build_evaluator("synergistic")
    case = problem.case
    robot = case.robot
    x_des_ts = problem.reference_ts
    init_gains = sample_initial_gains(
        problem.nominal_gains,
        batch_size=args.batch_size,
        seed=args.init_seed,
        scheme=args.init_scheme,
        spread=args.init_spread,
    )
    print(f"Gain initializations ({args.init_scheme}, seed {args.init_seed}):")
    print(describe_initial_gains(init_gains))

    opt_vars = {"opt_ctr_params": init_gains, "opt_atr_params": {}}
    labels = {
        "opt_ctr_params": {"Kp": "P", "Ki": "I", "Kd": "D"},
        "opt_atr_params": {},
    }
    optimizer = optax.multi_transform(
        {
            "P": optax.chain(optax.clip_by_global_norm(10.0), optax.yogi(0.5)),
            "I": optax.chain(optax.clip_by_global_norm(10.0), optax.yogi(0.25)),
            "D": optax.chain(optax.clip_by_global_norm(10.0), optax.yogi(0.05)),
        },
        labels,
    )
    # Recorded in the archive so optimizer tuning can be reported and compared
    # separately from the loss definition (issue #154 follow-up, item 6).
    optimizer_metadata = (
        "multi_transform{P: clip_by_global_norm(10.0)+yogi(0.5), "
        "I: clip_by_global_norm(10.0)+yogi(0.25), "
        "D: clip_by_global_norm(10.0)+yogi(0.05)}"
    )
    history = run_gain_optimization(
        gradient_fn=problem.gradient_fn,
        optimizer=optimizer,
        opt_vars=opt_vars,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        progress_label="Synergistic",
    )
    if len(history) != args.num_iters:
        raise RuntimeError(
            f"Refusing to save partial run: completed {len(history)}/{args.num_iters}; "
            f"{history.stop_reason}"
        )

    dead = history.dead_starts()
    if dead:
        raise RuntimeError(
            f"Starts {dead} never produced a finite iterate; rerun with a "
            "different --init-seed or a narrower --init-spread"
        )

    init_aux, best_aux = history.init_aux, history.best_aux
    loss_history = history.loss_history()
    mask_history = history.mask_history()
    # The time grid is shared by every start, so archive one copy.
    t_ts = init_aux["t_ts"][0]
    pose_of = vmap(lambda q: end_effector_pose_trajectory(robot, q, case.total_length))
    save_optimization_outputs(
        result_dir,
        method="synergistic",
        batch_size=args.batch_size,
        history_loss=loss_history,
        history_time=history.time_iter,
        history_finite_mask=mask_history,
        history_grad_norm=history.grad_norm_history(),
        history_update_norm=history.update_norm_history(),
        opt_vars_history=history.opt_vars,
        init_gains=init_gains,
        init_seed=args.init_seed,
        init_scheme=args.init_scheme,
        init_spread=args.init_spread,
        objective_name=OBJECTIVE_NAME,
        objective_scales=problem.objective_scales,
        saturation_config=problem.saturation_config,
        optimizer_metadata=optimizer_metadata,
        t_ts=t_ts,
        q_ts_init=init_aux["q_ts"],
        q_ts_best=best_aux["q_ts"],
        qd_ts_init=init_aux["qd_ts"],
        qd_ts_best=best_aux["qd_ts"],
        u_ts_init=init_aux["u_ts"],
        u_ts_best=best_aux["u_ts"],
        x_ts_init=pose_of(init_aux["q_ts"]),
        x_ts_best=pose_of(best_aux["q_ts"]),
        x_des_ts=x_des_ts,
        # Supplied by the loop and re-derived by the validator, so the two
        # cannot silently disagree.
        best_iteration=history.best_iteration,
        best_batch=history.best_batch(),
        is_placeholder=args.placeholder,
    )
    print(
        "Improvement over the initial median: "
        f"{100 * improvement_over_initial_median(loss_history, mask_history):.2f} %"
    )


if __name__ == "__main__":
    main()
