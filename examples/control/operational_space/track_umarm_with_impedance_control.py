"""Track a circular UMArm end-effector path with impedance control.

The physical UMArm has 24 McKibben pressure channels for 12 joint coordinates.
This example reuses the balanced antagonistic pressure map from the joint-space
tracking example to expose 12 virtual differential pressures.  The resulting
12-by-12 actuation matrix satisfies the full-actuation requirement of
``OperationalSpaceImpedanceControlTracker``.  A uniform nominal pressure
provides co-contraction, while the controller commands equal and opposite
pressure changes within every antagonist pair.

Only the end-effector position is selected as the operational-space task.  Its
reference traces a circle parallel to the mounting base with a period of two
seconds and a default diameter close to the robot length; orientation is
unconstrained.
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array

from soromox.control import (
    OperationalSpaceImpedanceControlTracker,
    ReferenceTrajectory,
)
from soromox.coordinate_transformations import OperationalSpaceDynamics
from soromox.rendering import UMArmViserRenderer
from soromox.systems import McKibbenActuatedUMArm, SystemState
from soromox.utils.geometry import poses

DEFAULT_DURATION = 4.0
DEFAULT_SOLVER_DT = 5.0e-4
DEFAULT_SAVE_DT = 1.0e-2
CIRCLE_PERIOD = 2.0
DEFAULT_NOMINAL_PRESSURE = 100.0e3
DEFAULT_MAX_DELTA_PRESSURE = 100.0e3
DEFAULT_NATURAL_FREQUENCY = 16.0
DEFAULT_DAMPING_RATIO = 1.0
DEFAULT_RECORDING_FPS = 30.0
DEFAULT_FIGURES_DIR = Path(__file__).resolve().parent / "figures"
DEFAULT_VIDEO_PATH = (
    Path(__file__).resolve().parent
    / "videos"
    / "track_umarm_with_impedance_control.mp4"
)

# A bent initial posture keeps the three-dimensional position Jacobian away
# from the axial singularity of the perfectly straight arm.
DEFAULT_INITIAL_CONFIGURATION = jnp.tile(
    jnp.array([0.12, -0.08], dtype=jnp.float64),
    6,
)
POSITION_TASK_SELECTOR = jnp.array(
    [False, False, False, True, True, True],
    dtype=bool,
)
UMARM_Z_DOWN_BASE_POSE = poses.spatial_mounting_pose(
    "hanging", jnp.array([0.0, 0.0, 1.2], dtype=jnp.float64)
)


@dataclass(frozen=True)
class CircleReference:
    """Position-only circle and its geometry in the world frame."""

    trajectory: ReferenceTrajectory
    position_fn: Callable[[Array], Array]
    center: Array
    normal: Array
    radius: Array


@dataclass(frozen=True)
class TrackingResult:
    """Saved closed-loop states and end-effector position reference."""

    t: Array
    q: Array
    qd: Array
    position: Array
    position_des: Array
    pressures: Array
    circle_center: Array
    circle_normal: Array
    circle_radius: Array


class DifferentialPressureUMArmModel(McKibbenActuatedUMArm):
    """UMArm dynamics reparameterized by 12 differential pressures.

    All body kinematics and dynamics use the canonical UMArm implementation.
    Only the actuator-coordinate contract changes: virtual pressure efforts are
    expanded through ``pressure_map``, and the known nominal-pressure load is
    moved to the left-hand side of the equations of motion.
    """

    pressure_map: Array
    nominal_pressures: Array

    def __init__(
        self,
        robot: McKibbenActuatedUMArm,
        pressure_map: Array,
        nominal_pressures: Array,
    ) -> None:
        if robot.floating_base:
            raise ValueError(
                "DifferentialPressureUMArmModel requires a fixed-base UMArm."
            )
        pressure_map = jnp.asarray(pressure_map)
        nominal_pressures = jnp.asarray(nominal_pressures)
        if pressure_map.ndim != 2 or pressure_map.shape[0] != robot.num_actuators:
            raise ValueError(
                "pressure_map must have shape "
                f"({robot.num_actuators}, num_virtual_pressures), got "
                f"{pressure_map.shape}."
            )
        if nominal_pressures.shape != (robot.num_actuators,):
            raise ValueError(
                "nominal_pressures must have shape "
                f"({robot.num_actuators},), got {nominal_pressures.shape}."
            )

        assert robot.fixed_base_pose is not None
        super().__init__(
            robot.params,
            actuator=robot.actuators[0],
            passive_elements=robot.passive_elements,
            base_pose=robot.fixed_base_pose,
            eps=robot.global_eps,
        )
        self.pressure_map = pressure_map
        self.nominal_pressures = nominal_pressures
        self.num_actuators = pressure_map.shape[1]

    def _physical_actuation_matrix(self, q: Array) -> Array:
        """Return the retained McKibben transmission's 24-channel matrix."""
        return self.actuators[0].transmission.moment_matrix(self, q)

    def _actuation_matrix(self, q: Array) -> Array:
        """Map virtual differential pressures to generalized joint forces."""
        return self._physical_actuation_matrix(q) @ self.pressure_map

    def actuator_coordinates(self, q: Array) -> Array:
        """Return coordinates work-conjugate to differential pressures."""
        physical_coordinates = self.actuators[0].coordinates(self, q)
        return self.pressure_map.T @ physical_coordinates

    def actuator_velocities(self, q: Array, qd: Array) -> Array:
        """Return differential-pressure coordinate velocities."""
        return self._actuation_matrix(q).T @ qd

    def actuator_efforts(
        self,
        q: Array,
        u: Array,
        qd: Array | None = None,
        *,
        actuation_matrix: Array | None = None,
    ) -> Array:
        """Treat each virtual control directly as a pressure effort."""
        del q, qd
        u = jnp.asarray(u)
        if u.shape != (self.num_actuators,):
            raise ValueError(
                f"u must have shape ({self.num_actuators},), got {u.shape}."
            )
        if actuation_matrix is not None and actuation_matrix.shape != (
            self.num_velocities,
            self.num_actuators,
        ):
            raise ValueError(
                "actuation_matrix must have shape "
                f"({self.num_velocities}, {self.num_actuators}), got "
                f"{actuation_matrix.shape}."
            )
        return u

    def elastic_force(self, q: Array) -> Array:
        """Move the known nominal-pressure force to the dynamics left side."""
        nominal_force = self._physical_actuation_matrix(q) @ self.nominal_pressures
        return super().elastic_force(q) - nominal_force


class BalancedAntagonisticPressureController(eqx.Module):
    """Map virtual impedance inputs to bounded physical pressure changes."""

    tracker: OperationalSpaceImpedanceControlTracker
    pressure_map: Array
    max_delta_pressure: float = eqx.field(static=True)

    def __init__(
        self,
        tracker: OperationalSpaceImpedanceControlTracker,
        pressure_map: Array,
        max_delta_pressure: float,
    ) -> None:
        self.tracker = tracker
        self.pressure_map = jnp.asarray(pressure_map)
        self.max_delta_pressure = float(max_delta_pressure)

    def __call__(self, system_state: SystemState) -> tuple[Array, object | None]:
        delta_pressure, control_state_dot = self.tracker(system_state)
        delta_pressure = jnp.clip(
            delta_pressure,
            -self.max_delta_pressure,
            self.max_delta_pressure,
        )
        return self.pressure_map @ delta_pressure, control_state_dot


def _repo_params_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "assets"
        / "robot_parameters"
        / "mckibben_umarm"
        / "reference_parameters.npz"
    )


def build_robot(params_path: Path | None = None) -> McKibbenActuatedUMArm:
    """Build a vertically hanging UMArm from cached parameters."""
    source_path = _repo_params_path() if params_path is None else params_path
    robot = McKibbenActuatedUMArm.from_cached_parameters(source_path)
    return robot.with_fixed_base_pose(UMARM_Z_DOWN_BASE_POSE)


def antagonistic_pressure_map(robot: McKibbenActuatedUMArm) -> Array:
    """Return the fixed 24-by-12 UMArm differential-pressure expansion."""
    transmission = robot.actuators[0].params.transmission
    expected_joint_pairs = jnp.arange(robot.num_internal_dofs).reshape((-1, 2))
    if transmission.group_shape != (6, 4) or not bool(
        jnp.array_equal(transmission.joint_pair_indices, expected_joint_pairs)
    ):
        raise ValueError(
            "The antagonistic map requires the cached six-group UMArm topology "
            "with four muscles and two consecutive joints per group."
        )

    # Virtual inputs within each group are [delta_p_x, delta_p_y].  Channels
    # 3/1 oppose each other about x, while channels 0/2 oppose about y.
    group_map = jnp.array(
        [[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]],
        dtype=jnp.float64,
    )
    return jnp.kron(jnp.eye(transmission.num_groups), group_map)


def create_operational_space(
    control_model: DifferentialPressureUMArmModel,
) -> OperationalSpaceDynamics:
    """Create a position-only operational space at the UMArm tip."""
    return OperationalSpaceDynamics(
        robot=control_model,
        s_ps=jnp.array([control_model.length]),
        task_selector=POSITION_TASK_SELECTOR,
    )


def create_circle_position_reference(
    initial_position: Array,
    center: Array,
    normal: Array,
    *,
    period: float = CIRCLE_PERIOD,
) -> Callable[[Array], Array]:
    """Create a base-parallel circular reference starting at the robot tip."""
    if period <= 0.0:
        raise ValueError("period must be positive.")
    initial_position = jnp.asarray(initial_position)
    center = jnp.asarray(center)
    normal = jnp.asarray(normal)
    if initial_position.shape != (3,):
        raise ValueError("initial_position must have shape (3,).")
    if center.shape != (3,):
        raise ValueError("center must have shape (3,).")
    if normal.shape != (3,):
        raise ValueError("normal must have shape (3,).")

    normal_norm = jnp.linalg.norm(normal)
    if float(normal_norm) <= 0.0:
        raise ValueError("normal must be nonzero.")
    normal = normal / normal_norm
    radial = initial_position - center
    if not bool(jnp.isclose(radial @ normal, 0.0, atol=1.0e-10)):
        raise ValueError("initial_position and center must lie in the circle plane.")
    radius = jnp.linalg.norm(radial)
    if float(radius) <= 0.0:
        raise ValueError("initial_position must differ from center.")
    first_axis = radial / radius
    second_axis = jnp.cross(normal, first_axis)

    angular_frequency = 2.0 * jnp.pi / period

    def position_des_fn(t: Array) -> Array:
        phase = angular_frequency * t
        return center + radius * (
            jnp.cos(phase) * first_axis + jnp.sin(phase) * second_axis
        )

    return position_des_fn


def create_reference_trajectory(
    operational_space: OperationalSpaceDynamics,
    q0: Array,
    *,
    duration: float = DEFAULT_DURATION,
    radius: float | None = None,
    num_samples: int = 401,
) -> CircleReference:
    """Create the position-only two-second circular tip trajectory."""
    if duration <= 0.0:
        raise ValueError("duration must be positive.")
    if num_samples < 2:
        raise ValueError("num_samples must be at least two.")

    initial_pose = operational_space.operational_space_poses(q0)
    initial_position = initial_pose[3:]
    base_transform = operational_space.robot.base_transform
    base_position = base_transform[:3, 3]
    circle_normal = base_transform[:3, 0]
    circle_normal = circle_normal / jnp.linalg.norm(circle_normal)
    axis_center = base_position + circle_normal * (
        circle_normal @ (initial_position - base_position)
    )
    radial = initial_position - axis_center
    radial_norm = jnp.linalg.norm(radial)
    if float(radial_norm) <= 0.0:
        raise ValueError(
            "q0 must place the tip away from the base axis to define the circle."
        )
    if radius is None:
        circle_center = axis_center
    else:
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
        circle_center = initial_position - radius * radial / radial_norm

    position_des_fn = create_circle_position_reference(
        initial_position,
        circle_center,
        circle_normal,
        period=CIRCLE_PERIOD,
    )

    # The impedance tracker accepts full-pose reference containers.  The
    # orientation entries remain at their initial value and are discarded by
    # POSITION_TASK_SELECTOR; only position defines the controlled trajectory.
    def pose_container_fn(t: Array) -> Array:
        return initial_pose.at[3:].set(position_des_fn(t))

    return CircleReference(
        trajectory=ReferenceTrajectory(
            ts=jnp.linspace(0.0, duration, num_samples),
            x_des_fn=pose_container_fn,
            rotation_representation=operational_space.rotation_representation,
            n_points=operational_space.n_points,
            is_planar=operational_space.is_planar,
        ),
        position_fn=position_des_fn,
        center=circle_center,
        normal=circle_normal,
        radius=jnp.linalg.norm(initial_position - circle_center),
    )


def create_controller(
    robot: McKibbenActuatedUMArm,
    *,
    q0: Array = DEFAULT_INITIAL_CONFIGURATION,
    duration: float = DEFAULT_DURATION,
    circle_radius: float | None = None,
    nominal_pressure: float = DEFAULT_NOMINAL_PRESSURE,
    max_delta_pressure: float = DEFAULT_MAX_DELTA_PRESSURE,
    natural_frequency: float = DEFAULT_NATURAL_FREQUENCY,
    damping_ratio: float = DEFAULT_DAMPING_RATIO,
) -> tuple[
    BalancedAntagonisticPressureController,
    DifferentialPressureUMArmModel,
    OperationalSpaceDynamics,
    CircleReference,
    Array,
]:
    """Build the position impedance tracker and physical pressure wrapper."""
    if nominal_pressure <= 0.0:
        raise ValueError("nominal_pressure must be positive.")
    if max_delta_pressure <= 0.0:
        raise ValueError("max_delta_pressure must be positive.")
    if max_delta_pressure > nominal_pressure:
        raise ValueError(
            "max_delta_pressure cannot exceed nominal_pressure because that "
            "would permit negative physical pressures."
        )
    if natural_frequency <= 0.0:
        raise ValueError("natural_frequency must be positive.")
    if damping_ratio <= 0.0:
        raise ValueError("damping_ratio must be positive.")

    q0 = jnp.asarray(q0)
    if q0.shape != (robot.num_internal_dofs,):
        raise ValueError(f"q0 must have shape ({robot.num_internal_dofs},).")

    pressure_map = antagonistic_pressure_map(robot)
    nominal_pressures = jnp.full((robot.num_actuators,), nominal_pressure)
    control_model = DifferentialPressureUMArmModel(
        robot,
        pressure_map,
        nominal_pressures,
    )
    operational_space = create_operational_space(control_model)
    num_reference_samples = max(2, int(round(duration / DEFAULT_SAVE_DT)) + 1)
    circle_reference = create_reference_trajectory(
        operational_space,
        q0,
        duration=duration,
        radius=circle_radius,
        num_samples=num_reference_samples,
    )

    task_inertia = operational_space.inertia_matrix(q0)
    stiffness = natural_frequency**2 * task_inertia
    damping = 2.0 * damping_ratio * natural_frequency * task_inertia
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=("ImpedanceControlTracker: For full theoretical guarantees"),
            category=UserWarning,
        )
        tracker = OperationalSpaceImpedanceControlTracker(
            operational_space_dynamics=operational_space,
            reference_trajectory=circle_reference.trajectory,
            K_x=stiffness,
            D_x=damping,
            feedback_linearization="full",
        )

    controller = BalancedAntagonisticPressureController(
        tracker,
        pressure_map,
        max_delta_pressure,
    )
    return (
        controller,
        control_model,
        operational_space,
        circle_reference,
        nominal_pressures,
    )


def simulate(
    robot: McKibbenActuatedUMArm,
    *,
    duration: float = DEFAULT_DURATION,
    solver_dt: float = DEFAULT_SOLVER_DT,
    save_dt: float = DEFAULT_SAVE_DT,
    circle_radius: float | None = None,
    nominal_pressure: float = DEFAULT_NOMINAL_PRESSURE,
    max_delta_pressure: float = DEFAULT_MAX_DELTA_PRESSURE,
    natural_frequency: float = DEFAULT_NATURAL_FREQUENCY,
    damping_ratio: float = DEFAULT_DAMPING_RATIO,
    q0: Array = DEFAULT_INITIAL_CONFIGURATION,
) -> TrackingResult:
    """Run the position-only circular end-effector tracking maneuver."""
    if solver_dt <= 0.0 or save_dt <= 0.0:
        raise ValueError("solver_dt and save_dt must be positive.")

    (
        controller,
        _,
        operational_space,
        circle_reference,
        nominal_pressures,
    ) = create_controller(
        robot,
        q0=q0,
        duration=duration,
        circle_radius=circle_radius,
        nominal_pressure=nominal_pressure,
        max_delta_pressure=max_delta_pressure,
        natural_frequency=natural_frequency,
        damping_ratio=damping_ratio,
    )
    q0 = jnp.asarray(q0)
    assert circle_reference.trajectory.xd_des_fn is not None
    desired_task_velocity = (
        operational_space.B_task.T
        @ circle_reference.trajectory.xd_des_fn(jnp.array(0.0, dtype=q0.dtype))
    )
    qd0 = (
        operational_space.dynamically_consistent_pseudoinverse(q0)
        @ desired_task_velocity
    )
    initial_state = SystemState(
        t=jnp.array(0.0),
        y=jnp.concatenate([q0, qd0]),
        u=nominal_pressures,
    )
    trajectory = robot.rollout_closed_loop_to(
        initial_state=initial_state,
        controller=controller,
        t1=duration,
        solver_dt=solver_dt,
        save_dt=save_dt,
        max_steps=None,
    )
    q, qd = jnp.split(trajectory.y, 2, axis=1)
    assert trajectory.u is not None
    return TrackingResult(
        t=trajectory.t,
        q=q,
        qd=qd,
        position=jax.vmap(operational_space.operational_space_coordinates)(q),
        position_des=jax.vmap(circle_reference.position_fn)(trajectory.t),
        pressures=trajectory.u,
        circle_center=circle_reference.center,
        circle_normal=circle_reference.normal,
        circle_radius=circle_reference.radius,
    )


def tracking_metrics(result: TrackingResult) -> dict[str, Array]:
    """Return Cartesian position tracking errors."""
    error = result.position_des - result.position
    error_norm = jnp.linalg.norm(error, axis=1)
    return {
        "axis_rmse": jnp.sqrt(jnp.mean(error**2, axis=0)),
        "position_rmse": jnp.sqrt(jnp.mean(error_norm**2)),
        "maximum_position_error": jnp.max(error_norm),
    }


def plot_results(
    result: TrackingResult,
    *,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    show: bool = True,
) -> Path:
    """Plot Cartesian tracking, the circular path, and physical pressures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10.colors
    figure, axes = plt.subplots(3, 1, figsize=(10, 10))

    labels = ("x", "y", "z")
    for index, label in enumerate(labels):
        axes[0].plot(
            result.t,
            result.position[:, index],
            color=colors[index],
            linewidth=1.8,
            label=rf"$p_{label}$",
        )
        axes[0].plot(
            result.t,
            result.position_des[:, index],
            "--",
            color=colors[index],
            linewidth=2.4,
            label=rf"$p_{{{label},d}}$",
        )
    axes[0].set_ylabel("Position [m]")
    axes[0].set_title("End-effector position tracking")
    axes[0].legend(ncol=3)
    axes[0].grid(True, alpha=0.3)

    first_circle_axis = (
        result.position_des[0] - result.circle_center
    ) / result.circle_radius
    second_circle_axis = jnp.cross(result.circle_normal, first_circle_axis)
    desired_relative = result.position_des - result.circle_center
    actual_relative = result.position - result.circle_center
    desired_in_plane = jnp.stack(
        [desired_relative @ first_circle_axis, desired_relative @ second_circle_axis],
        axis=1,
    )
    actual_in_plane = jnp.stack(
        [actual_relative @ first_circle_axis, actual_relative @ second_circle_axis],
        axis=1,
    )
    axes[1].plot(
        desired_in_plane[:, 0],
        desired_in_plane[:, 1],
        "--",
        color="#E24A33",
        linewidth=2.4,
        label="reference",
    )
    axes[1].plot(
        actual_in_plane[:, 0],
        actual_in_plane[:, 1],
        color="#348ABD",
        linewidth=1.8,
        label="actual",
    )
    axes[1].set_xlabel("Base-plane axis 1 [m]")
    axes[1].set_ylabel("Base-plane axis 2 [m]")
    axes[1].set_title(
        f"Base-parallel circle (diameter {2.0 * float(result.circle_radius):.2f} m, "
        f"period {CIRCLE_PERIOD:g} s)"
    )
    axes[1].axis("equal")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    pressure_kpa = result.pressures / 1.0e3
    axes[2].fill_between(
        result.t,
        jnp.min(pressure_kpa, axis=1),
        jnp.max(pressure_kpa, axis=1),
        color="#988ED5",
        alpha=0.35,
        label="range across 24 muscles",
    )
    axes[2].plot(
        result.t,
        jnp.mean(pressure_kpa, axis=1),
        color="#555555",
        label="mean pressure",
    )
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Pressure [kPa]")
    axes[2].set_title("Antagonistic pressure activity")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    figure.tight_layout()
    output_path = figures_dir / "track_umarm_with_impedance_control.pdf"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    return output_path


def render_motion(
    robot: McKibbenActuatedUMArm,
    result: TrackingResult,
    *,
    record_path: Path | None = None,
) -> None:
    """Render the circular motion with pressure-colored McKibben actuators."""
    if UMArmViserRenderer is None:
        print("UMArmViserRenderer is unavailable. Install the Viser extras.")
        return
    if record_path is not None:
        record_path.parent.mkdir(parents=True, exist_ok=True)

    render_t = result.t
    render_q = result.q
    render_pressures = result.pressures
    render_position_des = result.position_des
    sample_period = (
        float(jnp.mean(jnp.diff(result.t))) if result.t.shape[0] > 1 else CIRCLE_PERIOD
    )
    samples_per_circle = max(1, int(round(CIRCLE_PERIOD / sample_period)))
    circle_path_stride = max(1, samples_per_circle // 48)
    circle_path = result.position_des[
        : min(result.t.shape[0], samples_per_circle + 1) : circle_path_stride
    ]
    if record_path is not None and result.t.shape[0] > 1:
        frame_stride = max(
            1,
            int(round(1.0 / (DEFAULT_RECORDING_FPS * sample_period))),
        )
        frame_indices = list(range(0, result.t.shape[0], frame_stride))
        if frame_indices[-1] != result.t.shape[0] - 1:
            frame_indices.append(result.t.shape[0] - 1)
        indices = jnp.asarray(frame_indices)
        render_t = result.t[indices]
        render_q = result.q[indices]
        render_pressures = result.pressures[indices]
        render_position_des = result.position_des[indices]

    renderer = UMArmViserRenderer(
        robot,
        width=1280,
        height=720,
        num_points=80,
        backbone_style="discrete",
        actuator_color_mode="pressure",
    )
    renderer.render_sequence(
        ts=render_t,
        q_ts=render_q,
        pressures=render_pressures,
        actuator_color_mode="pressure",
        playback_speed=1.0,
        autoplay=True,
        loop=record_path is None,
        record_path=None if record_path is None else str(record_path),
        stop_when_recording_done=record_path is not None,
        record_client_timeout=120.0 if record_path is not None else 10.0,
        static_spheres_positions=circle_path,
        static_spheres_radii=jnp.full((circle_path.shape[0],), 0.008),
        static_spheres_colors=jnp.tile(
            jnp.array([[0.85, 0.12, 0.08, 0.45]]),
            (circle_path.shape[0], 1),
        ),
        dynamic_spheres_positions=render_position_des[None, :, :],
        dynamic_spheres_radii=jnp.array([0.022]),
        dynamic_spheres_colors=jnp.array([[0.9, 0.05, 0.02]]),
        plot_configurations=True,
        robot_name="UMArm operational-space impedance tracking",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--solver-dt", type=float, default=DEFAULT_SOLVER_DT)
    parser.add_argument("--save-dt", type=float, default=DEFAULT_SAVE_DT)
    parser.add_argument(
        "--circle-radius",
        type=float,
        default=None,
        help=(
            "End-effector circle radius [m]. By default, use the initial tip's "
            "distance from the base axis."
        ),
    )
    parser.add_argument(
        "--nominal-pressure-kpa",
        type=float,
        default=DEFAULT_NOMINAL_PRESSURE / 1.0e3,
    )
    parser.add_argument(
        "--max-delta-pressure-kpa",
        type=float,
        default=DEFAULT_MAX_DELTA_PRESSURE / 1.0e3,
    )
    parser.add_argument(
        "--natural-frequency",
        type=float,
        default=DEFAULT_NATURAL_FREQUENCY,
    )
    parser.add_argument(
        "--damping-ratio",
        type=float,
        default=DEFAULT_DAMPING_RATIO,
    )
    parser.add_argument(
        "--render",
        choices=("none", "viser"),
        default="viser",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
    )
    parser.add_argument(
        "--record",
        type=Path,
        nargs="?",
        const=DEFAULT_VIDEO_PATH,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    """Run the example from the command line."""
    args = parse_args()
    robot = build_robot(args.params)
    result = simulate(
        robot,
        duration=args.duration,
        solver_dt=args.solver_dt,
        save_dt=args.save_dt,
        circle_radius=args.circle_radius,
        nominal_pressure=args.nominal_pressure_kpa * 1.0e3,
        max_delta_pressure=args.max_delta_pressure_kpa * 1.0e3,
        natural_frequency=args.natural_frequency,
        damping_ratio=args.damping_ratio,
    )
    metrics = tracking_metrics(result)
    pressure_map = antagonistic_pressure_map(robot)
    virtual_actuation = (
        robot.actuation_matrix(DEFAULT_INITIAL_CONFIGURATION) @ pressure_map
    )
    print(f"Saved samples: {result.t.shape[0]}")
    print(f"Circle period: {CIRCLE_PERIOD:.2f} s")
    print(
        "Circle diameter: "
        f"{2.0 * float(result.circle_radius):.3f} m "
        f"({2.0 * float(result.circle_radius) / float(robot.length):.1%} "
        "of robot length)"
    )
    print(
        "Virtual actuation matrix: "
        f"{virtual_actuation.shape[0]} x {virtual_actuation.shape[1]} "
        f"(rank {int(jnp.linalg.matrix_rank(virtual_actuation))})"
    )
    print(f"Position RMSE: {1.0e3 * float(metrics['position_rmse']):.3f} mm")
    print(
        "Maximum position error: "
        f"{1.0e3 * float(metrics['maximum_position_error']):.3f} mm"
    )
    print(
        "Physical pressure range: "
        f"{float(jnp.min(result.pressures)) / 1.0e3:.2f} to "
        f"{float(jnp.max(result.pressures)) / 1.0e3:.2f} kPa"
    )

    if not args.no_plots:
        plot_path = plot_results(
            result,
            figures_dir=args.figures_dir,
            show=not args.no_show,
        )
        print(f"Tracking plot: {plot_path}")
    if args.render == "viser":
        render_motion(robot, result, record_path=args.record)


if __name__ == "__main__":
    main()
