"""Shared robot, setpoint, and optimization problem for Section Vd.

Both gain-optimization generators and the tuning study build their problem here,
so there is one definition of what "the collocated problem" and "the synergistic
problem" are. Each generator previously constructed its own controller,
reference and objective inline, and the tuning study held a hand-copy of the
collocated setup. Nothing kept those in sync, so the study could silently tune a
different problem than the generator ran.

:func:`build_evaluator` returns a :class:`SecVdProblem` rather than a bare
callable. A consumer that needs to know what a problem supports -- the tuning
study deciding whether a saturation sweep applies, for instance -- reads a field
instead of branching on the method name, so adding a controller does not mean
editing every consumer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array, value_and_grad, vmap
from secvd_objective import (
    configuration_space_scales,
    operational_space_scales,
    time_averaged_squared_error,
)

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.control import (
    OperationalSpaceSynergisticController,
    PIDControl,
    PIDControllerState,
    ReferenceTrajectory,
)
from soromox.control.actuation_space import PotentialCompensationRegulator
from soromox.coordinate_transformations import (
    ActuationSpaceDynamics,
    OperationalSpaceDynamics,
)
from soromox.systems import PCS, LinkSpec, SystemState
from soromox.utils.geometry.rotations import rotation_matrix_to_rotation_vector

jax.config.update("jax_enable_x64", True)

METHODS = ("collocated", "synergistic")

# The rollout horizon the published runs use. Only the tuning study varies it,
# to buy sweep throughput; solver_dt is never varied, because gradient magnitude
# is what the study tunes against.
CASE_HORIZON = 5.0

# The committed learning-rate structure, split into a magnitude and a per-family
# ratio so the two can be tuned separately. The committed rates
# (0.5, 0.25, 0.05) are exactly COMMITTED_ALPHA * COMMITTED_LR_RATIO.
COMMITTED_ALPHA = 0.5
COMMITTED_LR_RATIO = {"Kp": 1.0, "Ki": 0.5, "Kd": 0.1}


@dataclass(frozen=True)
class SecVdCase:
    robot: PCS
    routing: ThreadlikeRouting
    total_length: float
    q0: Array
    qd0: Array
    q_des: Array
    x_des: Array
    solver_dt: float
    save_ts: Array


def end_effector_pose(robot: PCS, q: Array, total_length: float) -> Array:
    """Return ``[rotation-vector xyz, Cartesian position xyz]`` at the tip."""
    transform = robot.forward_kinematics(q, total_length)
    return jnp.concatenate(
        [rotation_matrix_to_rotation_vector(transform[:3, :3]), transform[:3, 3]]
    )


def end_effector_pose_trajectory(robot: PCS, q_ts: Array, total_length: float) -> Array:
    return vmap(lambda q: end_effector_pose(robot, q, total_length))(q_ts)


def build_sec_vd_robot() -> tuple[PCS, ThreadlikeRouting, float]:
    """Construct the robot without running the steady-state setpoint rollout."""
    num_segments = 1
    radius = 2e-2
    segment_lengths = 1e-1 * jnp.ones((num_segments,))
    links = [
        LinkSpec.circular(
            length=float(segment_lengths[index]),
            radius=radius,
            density=1070.0,
            young_modulus=2e3,
            shear_modulus=1e3,
            material_damping_coefficient=3.6e2,
            reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )
        for index in range(num_segments)
    ]
    angles = jnp.deg2rad(jnp.array([0.0, 120.0, 240.0]))
    routing = ThreadlikeRouting.linear(
        intercept=0.8
        * radius
        * jnp.stack(
            [jnp.zeros_like(angles), jnp.sin(angles), jnp.cos(angles)], axis=-1
        ),
        start_segment_index=0,
        end_segment_index=(0, 0, 0),
    )
    robot = PCS.from_links(
        links,
        # Scalar-last quaternion pose: [qx, qy, qz, qw, x, y, z].
        base_pose=jnp.array([0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 0.0]),
        gravity=jnp.array([0.0, 0.0, 9.81]),
        actuators=ThreadlikeActuator.tendons(routing),
    )
    return robot, routing, float(jnp.sum(segment_lengths))


def build_sec_vd_case() -> SecVdCase:
    robot, routing, total_length = build_sec_vd_robot()
    q0 = jnp.zeros(robot.num_active_strains)
    qd0 = jnp.zeros_like(q0)
    steady = robot.rollout_to(
        initial_state=SystemState(t=jnp.array(0.0), y=jnp.concatenate([q0, qd0])),
        u=jnp.array([0.5, 0.2, 0.1]),
        t1=10.0,
        solver_dt=1e-4,
        save_dt=0.1,
    )
    q_des, qd_des = jnp.split(steady.y[-1], 2)
    speed = float(jnp.linalg.norm(qd_des))
    if speed > 1e-3:
        raise RuntimeError(
            f"Section Vd setpoint did not settle (velocity norm {speed:.3e})"
        )
    save_ts = robot._compute_save_times(0.0, 5.0, solver_dt=1e-4, save_dt=1e-4)
    return SecVdCase(
        robot=robot,
        routing=routing,
        total_length=total_length,
        q0=q0,
        qd0=qd0,
        q_des=q_des,
        x_des=end_effector_pose(robot, q_des, total_length),
        solver_dt=1e-4,
        save_ts=save_ts,
    )


@dataclass(frozen=True)
class SecVdProblem:
    """One Section Vd optimization problem, plus what a caller may ask of it.

    Attributes:
        method: ``"collocated"`` or ``"synergistic"``.
        case: The robot, setpoint and time grid the problem is built on.
        horizon: Rollout length in seconds.
        gradient_fn: ``value_and_grad``-shaped callable over **one** start's
            ``opt_vars``. The optimization loop vmaps it over the batch.
        nominal_gains: Gains the initializations are drawn around.
        objective_scales: Per-component factors making the error dimensionless.
            Archived, so a stored loss can be recomputed from the archive alone.
        reference_ts: The tracked reference over the time grid -- strains for
            collocated, poses for synergistic.
        committed_alpha: Learning-rate magnitude the generators currently use.
        committed_lr_ratio: Per-family ratio the generators currently use.
        supports_saturation: Whether the controller saturates its integral
            error. Consumers gate on this rather than on ``method``.
        saturation_config: Human-readable saturation setting, archived as-is.
    """

    method: str
    case: SecVdCase
    horizon: float
    gradient_fn: Callable
    nominal_gains: dict[str, Array]
    objective_scales: Array
    reference_ts: Array
    committed_alpha: float
    committed_lr_ratio: dict[str, float]
    supports_saturation: bool
    saturation_config: str


def _save_times(case: SecVdCase, horizon: float) -> Array:
    """Return the time grid for a horizon, reusing the case grid when it fits."""
    if horizon == CASE_HORIZON:
        return case.save_ts
    return case.robot._compute_save_times(
        0.0, horizon, solver_dt=case.solver_dt, save_dt=case.solver_dt
    )


def _rollout_aux(trajectory) -> tuple[Array, Array, dict]:
    """Split a closed-loop rollout into strains, rates, and the archived aux."""
    q_ts, qd_ts = jnp.split(trajectory.y, 2, axis=1)
    return (
        q_ts,
        qd_ts,
        {
            "t_ts": trajectory.t,
            "q_ts": q_ts,
            "qd_ts": qd_ts,
            "u_ts": trajectory.u,
        },
    )


def _build_collocated(
    case: SecVdCase, horizon: float, e_sat: float, loss_scale: float
) -> SecVdProblem:
    robot = case.robot
    asd = ActuationSpaceDynamics(robot)
    y_des = asd.actuated_unactuated_coordinates(case.q_des)
    minimum_length = float(jnp.min(jnp.abs(y_des[: robot.num_actuators])))
    if minimum_length <= 1e-2 * case.total_length:
        raise RuntimeError("Generated setpoint is too close to tendon path collapse")

    save_ts = _save_times(case, horizon)
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
    # Angular strains are 1/m and shear/axial strains already dimensionless;
    # scaling by segment length reproduces the recovered legacy weighting.
    scales = configuration_space_scales(robot.params.link.length)

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
        q_ts, _qd_ts, aux = _rollout_aux(trajectory)
        loss = time_averaged_squared_error(q_des_ts - q_ts, scales, trajectory.t)
        return loss_scale * loss, aux

    return SecVdProblem(
        method="collocated",
        case=case,
        horizon=horizon,
        gradient_fn=value_and_grad(evaluate, has_aux=True),
        nominal_gains=nominal,
        objective_scales=scales,
        reference_ts=q_des_ts,
        committed_alpha=COMMITTED_ALPHA,
        committed_lr_ratio=dict(COMMITTED_LR_RATIO),
        supports_saturation=True,
        saturation_config=f"tanh, e_sat={e_sat} m",
    )


def _build_synergistic(
    case: SecVdCase, horizon: float, loss_scale: float
) -> SecVdProblem:
    robot = case.robot
    osd = OperationalSpaceDynamics(
        robot=robot,
        s_ps=jnp.array([case.total_length]),
        task_selector=jnp.array([False, False, False, True, True, True]),
    )
    save_ts = _save_times(case, horizon)
    reference = ReferenceTrajectory(
        ts=save_ts,
        x_des_fn=lambda _t: case.x_des,
        rotation_representation=osd.rotation_representation,
        n_points=osd.n_points,
        is_planar=osd.is_planar,
    )
    x_des_ts = reference.x_des_ts
    nominal = {
        "Kp": 10.0 * jnp.ones(osd.n_operational_space),
        "Ki": 2.0 * jnp.ones(osd.n_operational_space),
        "Kd": 0.25 * jnp.ones(osd.n_operational_space),
    }
    controller = OperationalSpaceSynergisticController(
        operational_space_dynamics=osd,
        reference_trajectory=reference,
        pid_control=PIDControl(**nominal),
    )
    initial_state = SystemState(
        t=jnp.array(0.0),
        y=jnp.concatenate([case.q0, case.qd0]),
        u=jnp.zeros(robot.num_actuators),
        control_state=PIDControllerState.zero(osd.n_operational_space),
    )
    num_ts = len(save_ts)
    # Position error in metres becomes dimensionless against the backbone
    # length; orientation error is already in radians.
    scales = operational_space_scales(osd.task_selector, case.total_length)

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
        q_ts, _qd_ts, aux = _rollout_aux(trajectory)
        x_ts = vmap(osd.operational_space_poses)(q_ts)
        error = vmap(osd.compute_task_pose_error)(x_ts, x_des_ts)
        loss = time_averaged_squared_error(error, scales, trajectory.t)
        return loss_scale * loss, aux

    return SecVdProblem(
        method="synergistic",
        case=case,
        horizon=horizon,
        gradient_fn=value_and_grad(evaluate, has_aux=True),
        nominal_gains=nominal,
        objective_scales=scales,
        reference_ts=x_des_ts,
        committed_alpha=COMMITTED_ALPHA,
        committed_lr_ratio=dict(COMMITTED_LR_RATIO),
        # This controller does not saturate its integral error at all, so
        # follow-up item 2 has nothing to sweep here.
        supports_saturation=False,
        saturation_config="none",
    )


def build_evaluator(
    method: str,
    *,
    case: SecVdCase | None = None,
    horizon: float = CASE_HORIZON,
    e_sat: float = 1e-2,
    loss_scale: float = 1.0,
) -> SecVdProblem:
    """Build the differentiable optimization problem for one Section Vd method.

    Args:
        method: One of :data:`METHODS`.
        case: A prebuilt case, to avoid repeating the setpoint rollout. Built on
            demand when omitted.
        horizon: Rollout length in seconds. Only the tuning study varies this.
        e_sat: Integral-error saturation scale in metres. Ignored by methods
            whose controller does not saturate.
        loss_scale: Constant multiplying the objective. Adam-family optimizers
            are largely insensitive to it, and a measured sweep found it reaches
            a worse optimum than tuning the learning rate; it exists so that can
            be re-tested rather than assumed.

    Returns:
        The problem, its nominal gains, and the metadata consumers gate on.

    Raises:
        ValueError: If ``method`` is not a known Section Vd method.
        RuntimeError: If the generated setpoint is degenerate.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown Section Vd method {method!r}; expected {METHODS}")
    if not horizon > 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if not e_sat > 0:
        raise ValueError(f"e_sat must be positive, got {e_sat}")
    case = build_sec_vd_case() if case is None else case
    if method == "collocated":
        return _build_collocated(case, horizon, e_sat, loss_scale)
    return _build_synergistic(case, horizon, loss_scale)
