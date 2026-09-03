"""Shared robot, setpoint, and optimization problem for Section Vd."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array, value_and_grad, vmap
from secvd_init import GAIN_ORDER
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
CASE_HORIZON = 5.0
COMMITTED_ALPHA = 0.5
COMMITTED_LR_RATIO = {"Kp": 1.0, "Ki": 0.5, "Kd": 0.1}
TUNED_ALPHA = {"collocated": 3000.0, "synergistic": 1547.0}
MATERIAL_DAMPING = 7.2e1
INIT_GAIN_SCALE = {
    "collocated": {"Kp": 1.0, "Ki": 1.0, "Kd": 0.2},
    "synergistic": {"Kp": 3.0, "Ki": 1.0, "Kd": 0.2},
}
COMMITTED_E_SAT = 1e-2
FAMILY_LABEL = {"Kp": "P", "Ki": "I", "Kd": "D"}
YOGI_EPS = 1e-8


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
            material_damping_coefficient=MATERIAL_DAMPING,
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
        solver_dt: Integration step the rollout uses.
        gradient_fn: ``value_and_grad``-shaped callable over **one** start's
            ``opt_vars``. The optimization loop vmaps it over the batch.
        nominal_gains: Gains the initializations are drawn around.
        objective_scales: Per-component factors making the error dimensionless.
        reference_ts: The tracked reference over the time grid.
        committed_alpha: Learning-rate magnitude.
        committed_lr_ratio: Per-family ratio.
        supports_saturation: Whether the controller saturates its integral
            error.
        saturation_config: Human-readable saturation setting, archived as-is.
    """

    method: str
    case: SecVdCase
    horizon: float
    solver_dt: float
    gradient_fn: Callable
    nominal_gains: dict[str, Array]
    objective_scales: Array
    reference_ts: Array
    committed_alpha: float
    committed_lr_ratio: dict[str, float]
    control_error_fn: Callable[[dict], Array]
    supports_saturation: bool
    saturation_config: str


def _save_times(case: SecVdCase, horizon: float, solver_dt: float) -> Array:
    """Return the time grid for a horizon, reusing the case grid when it fits."""
    if horizon == CASE_HORIZON and solver_dt == case.solver_dt:
        return case.save_ts
    return case.robot._compute_save_times(
        0.0, horizon, solver_dt=solver_dt, save_dt=solver_dt
    )


def _max_solver_steps(horizon: float, solver_dt: float) -> int:
    """Bound the solver step count for a rollout."""
    return int(round(horizon / solver_dt)) + 16


def _rollout_aux(trajectory) -> tuple[Array, Array, dict]:
    """Split a closed-loop rollout into strains, rates, and the aux signals."""
    q_ts, qd_ts = jnp.split(trajectory.y, 2, axis=1)
    return (
        q_ts,
        qd_ts,
        {
            "t_ts": trajectory.t,
            "q_ts": q_ts,
            "qd_ts": qd_ts,
            "u_ts": trajectory.u,
            "integral_error_ts": trajectory.control_state.integral_error,
        },
    )


def _build_collocated(
    case: SecVdCase, horizon: float, solver_dt: float, e_sat: float
) -> SecVdProblem:
    robot = case.robot
    asd = ActuationSpaceDynamics(robot)
    y_des = asd.actuated_unactuated_coordinates(case.q_des)
    minimum_length = float(jnp.min(jnp.abs(y_des[: robot.num_actuators])))
    if minimum_length <= 1e-2 * case.total_length:
        raise RuntimeError("Generated setpoint is too close to tendon path collapse")

    save_ts = _save_times(case, horizon, solver_dt)
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
    max_steps = _max_solver_steps(horizon, solver_dt)
    # Angular strains are 1/m and shear/axial strains already dimensionless;
    # scaling by segment length reproduces the recovered legacy weighting.
    scales = configuration_space_scales(robot.params.link.length)

    def control_error(aux: dict) -> Array:
        """The error the PID integrates: actuated coordinates, in metres."""
        n_a = robot.num_actuators
        y_a = vmap(lambda q: asd.actuated_unactuated_coordinates(q)[:n_a])(aux["q_ts"])
        return y_des[:n_a] - y_a

    def evaluate(opt_vars):
        active = controller.update_gains(opt_vars["opt_ctr_params"])
        trajectory = robot.rollout_closed_loop_to(
            initial_state=initial_state,
            controller=active,
            t1=horizon,
            solver_dt=solver_dt,
            save_ts=save_ts,
            max_steps=max_steps,
        )
        q_ts, _qd_ts, aux = _rollout_aux(trajectory)
        loss = time_averaged_squared_error(q_des_ts - q_ts, scales, trajectory.t)
        return loss, aux

    return SecVdProblem(
        method="collocated",
        case=case,
        horizon=horizon,
        solver_dt=solver_dt,
        gradient_fn=value_and_grad(evaluate, has_aux=True),
        nominal_gains=nominal,
        objective_scales=scales,
        reference_ts=q_des_ts,
        committed_alpha=COMMITTED_ALPHA,
        committed_lr_ratio=dict(COMMITTED_LR_RATIO),
        control_error_fn=control_error,
        supports_saturation=True,
        saturation_config=f"tanh, e_sat={e_sat} m",
    )


def _build_synergistic(
    case: SecVdCase, horizon: float, solver_dt: float
) -> SecVdProblem:
    robot = case.robot
    osd = OperationalSpaceDynamics(
        robot=robot,
        s_ps=jnp.array([case.total_length]),
        task_selector=jnp.array([False, False, False, True, True, True]),
    )
    save_ts = _save_times(case, horizon, solver_dt)
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
    max_steps = _max_solver_steps(horizon, solver_dt)
    scales = operational_space_scales(osd.task_selector, case.total_length)

    def control_error(aux: dict) -> Array:
        """The task-space pose error this controller's PID integrates."""
        x_ts = vmap(osd.operational_space_poses)(aux["q_ts"])
        return vmap(osd.compute_task_pose_error)(x_ts, x_des_ts)

    def evaluate(opt_vars):
        active = controller.update_gains(opt_vars["opt_ctr_params"])
        trajectory = robot.rollout_closed_loop_to(
            initial_state=initial_state,
            controller=active,
            t1=horizon,
            solver_dt=solver_dt,
            save_ts=save_ts,
            max_steps=max_steps,
        )
        q_ts, _qd_ts, aux = _rollout_aux(trajectory)
        x_ts = vmap(osd.operational_space_poses)(q_ts)
        error = vmap(osd.compute_task_pose_error)(x_ts, x_des_ts)
        loss = time_averaged_squared_error(error, scales, trajectory.t)
        return loss, aux

    return SecVdProblem(
        method="synergistic",
        case=case,
        horizon=horizon,
        solver_dt=solver_dt,
        gradient_fn=value_and_grad(evaluate, has_aux=True),
        nominal_gains=nominal,
        objective_scales=scales,
        reference_ts=x_des_ts,
        control_error_fn=control_error,
        committed_alpha=COMMITTED_ALPHA,
        committed_lr_ratio=dict(COMMITTED_LR_RATIO),
        supports_saturation=False,
        saturation_config="none",
    )


def build_evaluator(
    method: str,
    *,
    case: SecVdCase | None = None,
    horizon: float = CASE_HORIZON,
    solver_dt: float | None = None,
    e_sat: float = COMMITTED_E_SAT,
) -> SecVdProblem:
    """Build the differentiable optimization problem for one Section Vd method.

    Args:
        method: One of :data:`METHODS`.
        case: A prebuilt case, to avoid repeating the setpoint rollout.
        horizon: Rollout length in seconds.
        solver_dt: Integration step. Defaults to the case value.
        e_sat: Integral-error saturation scale in metres.

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
    solver_dt = case.solver_dt if solver_dt is None else solver_dt
    if not solver_dt > 0:
        raise ValueError(f"solver_dt must be positive, got {solver_dt}")
    if solver_dt > horizon:
        raise ValueError(f"solver_dt {solver_dt} exceeds the {horizon} s horizon")
    if method == "collocated":
        return _build_collocated(case, horizon, solver_dt, e_sat)
    return _build_synergistic(case, horizon, solver_dt)


def build_optimizer(alpha: float, *, ratio: dict[str, float] | None = None):
    """Build the Section Vd gain optimizer: one Yogi, per-family learning rates.

    Args:
        alpha: Learning-rate magnitude.
        ratio: Per-family multipliers on ``alpha``. Defaults to
            :data:`COMMITTED_LR_RATIO`.

    Returns:
        An ``optax`` gradient transformation over the
        ``{"opt_ctr_params": ..., "opt_atr_params": ...}`` pytree.

    Raises:
        ValueError: If ``ratio`` does not cover exactly the gain families.
    """
    import optax  # Imported lazily

    ratio = dict(COMMITTED_LR_RATIO if ratio is None else ratio)
    if set(ratio) != set(GAIN_ORDER):
        raise ValueError(f"ratio must cover exactly {GAIN_ORDER}, got {sorted(ratio)}")
    return optax.chain(
        optax.scale_by_yogi(eps=YOGI_EPS),
        optax.multi_transform(
            {
                FAMILY_LABEL[name]: optax.scale(-alpha * ratio[name])
                for name in GAIN_ORDER
            },
            {
                "opt_ctr_params": {name: FAMILY_LABEL[name] for name in GAIN_ORDER},
                "opt_atr_params": {},
            },
        ),
    )


def describe_optimizer(alpha: float, *, ratio: dict[str, float] | None = None) -> str:
    """Render :func:`build_optimizer`'s configuration for the archive."""
    ratio = dict(COMMITTED_LR_RATIO if ratio is None else ratio)
    rates = ", ".join(f"{FAMILY_LABEL[n]}: {alpha * ratio[n]:g}" for n in GAIN_ORDER)
    return f"scale_by_yogi(eps={YOGI_EPS:g}) + per-family rate{{{rates}}}"
