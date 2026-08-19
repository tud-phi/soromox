"""Optimize Section Vd synergistic controller gains and write the canonical NPZ."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from secvd_optimization import configure_optimization_device  # noqa: E402

# Number of independently optimized gain initializations; timestep vmaps do not count.
OPTIMIZATION_BATCH_SIZE = 1
configure_optimization_device(batch_size=OPTIMIZATION_BATCH_SIZE)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from gain_optimization_loop import run_gain_optimization  # noqa: E402
from jax import jit, value_and_grad, vmap  # noqa: E402
from secvd_case import build_sec_vd_case, end_effector_pose_trajectory  # noqa: E402
from secvd_optimization import (  # noqa: E402
    parse_optimization_args,
    prepare_result_dir,
)
from secvd_results import save_optimization_outputs  # noqa: E402

from soromox.control import (
    OperationalSpaceSynergisticController,
    PIDControl,
    PIDControllerState,
    ReferenceTrajectory,
)
from soromox.coordinate_transformations import OperationalSpaceDynamics
from soromox.systems import SystemState

jax.config.update("jax_enable_x64", True)

CASE_DIR = CODE_DIR.parent


def main() -> None:
    args = parse_optimization_args(
        description="Optimize synergistic control gains for Section Vd.",
        default_result_dir=CASE_DIR / "data" / "synergistic",
        optimization_batch_size=OPTIMIZATION_BATCH_SIZE,
    )
    result_dir = prepare_result_dir(args)
    case = build_sec_vd_case()
    robot = case.robot
    osd = OperationalSpaceDynamics(
        robot=robot,
        s_ps=jnp.array([case.total_length]),
        task_selector=jnp.array([False, False, False, True, True, True]),
    )
    reference = ReferenceTrajectory(
        ts=case.save_ts,
        x_des_fn=lambda _t: case.x_des,
        rotation_representation=osd.rotation_representation,
        n_points=osd.n_points,
        is_planar=osd.is_planar,
    )
    x_des_ts = reference.x_des_ts
    gains = {
        "Kp": 10.0 * jnp.ones(osd.n_operational_space),
        "Ki": 2.0 * jnp.ones(osd.n_operational_space),
        "Kd": 0.25 * jnp.ones(osd.n_operational_space),
    }
    controller = OperationalSpaceSynergisticController(
        operational_space_dynamics=osd,
        reference_trajectory=reference,
        pid_control=PIDControl(**gains),
    )
    initial_state = SystemState(
        t=jnp.array(0.0),
        y=jnp.concatenate([case.q0, case.qd0]),
        u=jnp.zeros(robot.num_actuators),
        control_state=PIDControllerState.zero(osd.n_operational_space),
    )
    num_ts = len(case.save_ts)

    def evaluate(opt_vars):
        active_controller = controller.update_gains(opt_vars["opt_ctr_params"])
        trajectory = robot.rollout_closed_loop_to(
            initial_state=initial_state,
            controller=active_controller,
            t1=5.0,
            solver_dt=case.solver_dt,
            save_ts=case.save_ts,
            max_steps=num_ts + 1,
        )
        q_ts, qd_ts = jnp.split(trajectory.y, 2, axis=1)
        x_ts = vmap(osd.operational_space_poses)(q_ts)
        error = vmap(osd.compute_task_pose_error)(x_ts, x_des_ts)
        error_sq = jnp.sum(error**2, axis=1)
        split = len(trajectory.t) // 2
        loss = 1e3 * jnp.trapezoid(error_sq[:split], trajectory.t[:split])
        loss += 5e3 * jnp.trapezoid(error_sq[split:], trajectory.t[split:])
        return loss, {
            "t_ts": trajectory.t,
            "q_ts": q_ts,
            "qd_ts": qd_ts,
            "u_ts": trajectory.u,
        }

    opt_vars = {"opt_ctr_params": gains, "opt_atr_params": {}}
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
    gradient_fn = jit(value_and_grad(evaluate, has_aux=True))
    history = run_gain_optimization(
        gradient_fn=gradient_fn,
        optimizer=optimizer,
        opt_vars=opt_vars,
        num_iters=args.num_iters,
        progress_label="Synergistic",
    )
    if len(history) != args.num_iters:
        raise RuntimeError(
            f"Refusing to save partial run: completed {len(history)}/{args.num_iters}; "
            f"{history.stop_reason}"
        )

    q_history = history.stacked_aux("q_ts")
    qd_history = history.stacked_aux("qd_ts")
    u_history = history.stacked_aux("u_ts")
    best = history.best_index()
    t_ts = history.stacked_aux("t_ts")[0]
    x_init = end_effector_pose_trajectory(robot, q_history[0], case.total_length)
    x_best = end_effector_pose_trajectory(robot, q_history[best], case.total_length)
    save_optimization_outputs(
        result_dir,
        method="synergistic",
        history_loss=history.loss,
        history_time=history.time_iter,
        opt_vars_history=history.opt_vars,
        history_qd_ts=qd_history,
        history_u_ts=u_history,
        t_ts=t_ts,
        q_ts_init=q_history[0],
        q_ts_best=q_history[best],
        qd_ts_init=qd_history[0],
        qd_ts_best=qd_history[best],
        u_ts_init=u_history[0],
        u_ts_best=u_history[best],
        x_ts_init=x_init,
        x_ts_best=x_best,
        x_des_ts=x_des_ts,
        best_iteration=best,
        is_placeholder=args.placeholder,
    )


if __name__ == "__main__":
    main()
