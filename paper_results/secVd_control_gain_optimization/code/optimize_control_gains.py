"""Optimize Section Vd control gains and write the canonical NPZ."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from secvd_cli import configure_optimization_device  # noqa: E402

configure_optimization_device()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import vmap  # noqa: E402
from secvd_case import (  # noqa: E402
    INIT_GAIN_SCALE,
    MATERIAL_DAMPING,
    build_evaluator,
    build_optimizer,
    describe_optimizer,
    end_effector_pose_trajectory,
)
from secvd_cli import parse_optimization_args, prepare_result_dir  # noqa: E402
from secvd_init import (  # noqa: E402
    SECVD_INIT_SCHEME,
    describe_initial_gains,
    sample_initial_gains,
)
from secvd_loop import run_gain_optimization  # noqa: E402
from secvd_objective import OBJECTIVE_NAME  # noqa: E402
from secvd_results import (  # noqa: E402
    improvement_over_initial_median,
    save_optimization_outputs,
)

jax.config.update("jax_enable_x64", True)

CASE_DIR = CODE_DIR.parent


def main() -> None:
    args = parse_optimization_args(
        description="Optimize control gains for Section Vd.",
        default_result_dir_for=lambda method: CASE_DIR / "data" / method,
    )
    method = args.method
    result_dir = prepare_result_dir(args)

    saturation = {"e_sat": args.e_sat} if method == "collocated" else {}
    problem = build_evaluator(method, solver_dt=args.solver_dt, **saturation)
    case = problem.case
    robot = case.robot

    init_center = {
        name: value * INIT_GAIN_SCALE[method][name]
        for name, value in problem.nominal_gains.items()
    }
    init_gains = sample_initial_gains(
        init_center,
        batch_size=args.batch_size,
        seed=args.init_seed,
        spread=args.init_spread,
    )
    print(f"Gain initializations ({SECVD_INIT_SCHEME}, seed {args.init_seed}):")
    print(describe_initial_gains(init_gains))

    history = run_gain_optimization(
        gradient_fn=problem.gradient_fn,
        optimizer=build_optimizer(args.alpha, ratio=args.ratio),
        opt_vars={"opt_ctr_params": init_gains, "opt_atr_params": {}},
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        progress_label=method.capitalize(),
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

    t_ts = init_aux["t_ts"][0]
    pose_of = vmap(lambda q: end_effector_pose_trajectory(robot, q, case.total_length))
    reference = (
        {"x_des_ts": problem.reference_ts}
        if method == "synergistic"
        else {
            "x_des_ts": jnp.broadcast_to(case.x_des, (len(t_ts), 6)),
            "q_des_ts": problem.reference_ts,
        }
    )
    save_optimization_outputs(
        result_dir,
        method=method,
        batch_size=args.batch_size,
        history_loss=loss_history,
        history_time=history.time_iter,
        history_finite_mask=mask_history,
        history_grad_norm=history.grad_norm_history(),
        history_update_norm=history.update_norm_history(),
        opt_vars_history=history.opt_vars,
        init_gains=init_gains,
        init_seed=args.init_seed,
        init_scheme=SECVD_INIT_SCHEME,
        init_spread=args.init_spread,
        objective_name=OBJECTIVE_NAME,
        objective_scales=problem.objective_scales,
        saturation_config=problem.saturation_config,
        optimizer_metadata=describe_optimizer(args.alpha, ratio=args.ratio),
        solver_dt=problem.solver_dt,
        material_damping_coefficient=MATERIAL_DAMPING,
        t_ts=t_ts,
        q_ts_init=init_aux["q_ts"],
        q_ts_best=best_aux["q_ts"],
        qd_ts_init=init_aux["qd_ts"],
        qd_ts_best=best_aux["qd_ts"],
        u_ts_init=init_aux["u_ts"],
        u_ts_best=best_aux["u_ts"],
        x_ts_init=pose_of(init_aux["q_ts"]),
        x_ts_best=pose_of(best_aux["q_ts"]),
        best_iteration=history.best_iteration,
        best_batch=history.best_batch(),
        is_placeholder=args.placeholder,
        **reference,
    )
    print(
        "Improvement over the initial median: "
        f"{100 * improvement_over_initial_median(loss_history, mask_history):.2f} %"
    )


if __name__ == "__main__":
    main()
