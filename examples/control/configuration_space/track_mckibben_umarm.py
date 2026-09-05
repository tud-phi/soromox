"""Track a dynamic UMArm joint trajectory with antagonistic pressure control.

The physical UMArm has four McKibben muscles around every two-axis universal
joint.  This example exposes one differential-pressure input per revolute joint
to :class:`FeedforwardCompensationTracker` and maps it to the physical muscles
as

``[delta_p_y, -delta_p_x, -delta_p_y, delta_p_x]``.

A uniform nominal pressure supplies antagonistic co-contraction.  Its
configuration-dependent generalized force is included in the controller model,
so the feedforward term solves for pressure *changes* around that nominal value
instead of treating the preload as an unmodelled disturbance.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array

from soromox.control import PIDControl, ReferenceTrajectory
from soromox.control.configuration_space import FeedforwardCompensationTracker
from soromox.rendering import UMArmViserRenderer
from soromox.systems import McKibbenActuatedUMArm, SystemState
from soromox.utils.geometry import poses

DEFAULT_DURATION = 3.0
DEFAULT_SOLVER_DT = 5.0e-4
DEFAULT_SAVE_DT = 1.0e-2
DEFAULT_NOMINAL_PRESSURE = 55.0e3
DEFAULT_MAX_DELTA_PRESSURE = 55.0e3
DEFAULT_NATURAL_FREQUENCY = 20.0
DEFAULT_DAMPING_RATIO = 1.0
DEFAULT_RECORDING_FPS = 30.0
DEFAULT_FIGURES_DIR = Path(__file__).resolve().parent / "figures"
DEFAULT_VIDEO_PATH = (
    Path(__file__).resolve().parent / "videos" / "track_mckibben_umarm.mp4"
)

UMARM_Z_DOWN_BASE_POSE = poses.spatial_mounting_pose(
    "hanging", jnp.array([0.0, 0.0, 1.2], dtype=jnp.float64)
)

# The amplitudes approach the reported +/-0.3 rad joint range while leaving
# margin for tracking error.  Multiple carrier frequencies excite proximal and
# distal joint dynamics during the same maneuver.
REFERENCE_AMPLITUDES = jnp.array(
    [0.26, 0.24, 0.24, 0.22, 0.22, 0.20, 0.20, 0.18, 0.18, 0.16, 0.16, 0.14],
    dtype=jnp.float64,
)
REFERENCE_FREQUENCIES = jnp.array(
    [1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 1.5, 1.5, 2.0, 2.0, 2.5],
    dtype=jnp.float64,
)
REFERENCE_PHASES = jnp.array(
    [
        0.0,
        0.5 * jnp.pi,
        jnp.pi,
        0.5 * jnp.pi,
        0.0,
        jnp.pi,
        0.5 * jnp.pi,
        0.0,
        jnp.pi,
        0.5 * jnp.pi,
        0.0,
        jnp.pi,
    ],
    dtype=jnp.float64,
)


@dataclass(frozen=True)
class TrackingResult:
    """Saved closed-loop states and their joint-space reference."""

    t: Array
    q: Array
    qd: Array
    q_des: Array
    qd_des: Array
    qdd_des: Array
    pressures: Array


class DifferentialPressureUMArmModel(McKibbenActuatedUMArm):
    """UMArm dynamics reparameterized by 12 differential pressures.

    This is a true UMArm subclass so all body kinematics and dynamics stay on
    the canonical implementation.  Only the actuator-coordinate contract is
    changed: virtual pressure efforts are expanded through ``pressure_map``,
    and the known nominal-pressure load is moved to the left-hand side of the
    equations of motion.
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
    """Expand bounded virtual pressure differences into physical channels."""

    tracker: FeedforwardCompensationTracker
    pressure_map: Array
    max_delta_pressure: float = eqx.field(static=True)

    def __init__(
        self,
        tracker: FeedforwardCompensationTracker,
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


def create_reference_trajectory(
    duration: float = DEFAULT_DURATION,
    *,
    num_samples: int = 301,
) -> ReferenceTrajectory:
    """Create a windowed, multi-frequency 12-joint dynamic trajectory."""
    if duration <= 0.0:
        raise ValueError("duration must be positive.")
    if num_samples < 2:
        raise ValueError("num_samples must be at least two.")

    def q_des_fn(t: Array) -> Array:
        normalized_time = t / duration
        window = jnp.sin(jnp.pi * normalized_time) ** 2
        carrier = jnp.sin(
            2.0 * jnp.pi * REFERENCE_FREQUENCIES * normalized_time + REFERENCE_PHASES
        )
        return REFERENCE_AMPLITUDES * window * carrier

    return ReferenceTrajectory(
        ts=jnp.linspace(0.0, duration, num_samples),
        x_des_fn=q_des_fn,
    )


def create_controller(
    robot: McKibbenActuatedUMArm,
    reference_trajectory: ReferenceTrajectory,
    *,
    nominal_pressure: float = DEFAULT_NOMINAL_PRESSURE,
    max_delta_pressure: float = DEFAULT_MAX_DELTA_PRESSURE,
    natural_frequency: float = DEFAULT_NATURAL_FREQUENCY,
    damping_ratio: float = DEFAULT_DAMPING_RATIO,
) -> tuple[
    BalancedAntagonisticPressureController,
    DifferentialPressureUMArmModel,
    Array,
]:
    """Build the feedforward tracker and its physical pressure wrapper."""
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

    pressure_map = antagonistic_pressure_map(robot)
    nominal_pressures = jnp.full((robot.num_actuators,), nominal_pressure)
    control_model = DifferentialPressureUMArmModel(
        robot,
        pressure_map,
        nominal_pressures,
    )

    q_straight = jnp.zeros((robot.num_internal_dofs,))
    actuation_straight = control_model.actuation_matrix(q_straight)
    actuation_gramian = actuation_straight @ actuation_straight.T
    inertia_straight = robot.inertia_matrix(q_straight)

    # PIDController maps its generalized feedback through A.T.  Premultiplying
    # by inv(A A.T) makes the straight-arm closed-loop acceleration approach the
    # familiar qdd = wn^2 e + 2 zeta wn ed form.
    proportional_gain = jnp.linalg.solve(
        actuation_gramian,
        natural_frequency**2 * inertia_straight,
    )
    derivative_gain = jnp.linalg.solve(
        actuation_gramian,
        2.0 * damping_ratio * natural_frequency * inertia_straight,
    )
    pid_control = PIDControl(
        Kp=proportional_gain,
        Ki=jnp.zeros_like(proportional_gain),
        Kd=derivative_gain,
    )

    # A_delta(q) is intentionally configuration-dependent; the example accepts
    # the corresponding theoretical-guarantee warning and demonstrates the
    # feedforward-compensation controller on the nonlinear plant directly.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=("FeedforwardCompensationTracker: For full theoretical guarantees"),
            category=UserWarning,
        )
        tracker = FeedforwardCompensationTracker(
            control_model,
            reference_trajectory,
            pid_control,
        )

    controller = BalancedAntagonisticPressureController(
        tracker,
        pressure_map,
        max_delta_pressure,
    )
    return controller, control_model, nominal_pressures


def simulate(
    robot: McKibbenActuatedUMArm,
    *,
    duration: float = DEFAULT_DURATION,
    solver_dt: float = DEFAULT_SOLVER_DT,
    save_dt: float = DEFAULT_SAVE_DT,
    nominal_pressure: float = DEFAULT_NOMINAL_PRESSURE,
    max_delta_pressure: float = DEFAULT_MAX_DELTA_PRESSURE,
    natural_frequency: float = DEFAULT_NATURAL_FREQUENCY,
    damping_ratio: float = DEFAULT_DAMPING_RATIO,
) -> TrackingResult:
    """Run the dynamic joint-space tracking maneuver."""
    if solver_dt <= 0.0 or save_dt <= 0.0:
        raise ValueError("solver_dt and save_dt must be positive.")

    num_reference_samples = max(2, int(round(duration / save_dt)) + 1)
    reference = create_reference_trajectory(
        duration,
        num_samples=num_reference_samples,
    )
    controller, _, nominal_pressures = create_controller(
        robot,
        reference,
        nominal_pressure=nominal_pressure,
        max_delta_pressure=max_delta_pressure,
        natural_frequency=natural_frequency,
        damping_ratio=damping_ratio,
    )

    q0 = jnp.zeros((robot.num_internal_dofs,))
    initial_state = SystemState(
        t=jnp.array(0.0),
        y=jnp.concatenate([q0, jnp.zeros_like(q0)]),
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
    assert reference.x_des_fn is not None
    assert reference.xd_des_fn is not None
    assert reference.xdd_des_fn is not None
    return TrackingResult(
        t=trajectory.t,
        q=q,
        qd=qd,
        q_des=jax.vmap(reference.x_des_fn)(trajectory.t),
        qd_des=jax.vmap(reference.xd_des_fn)(trajectory.t),
        qdd_des=jax.vmap(reference.xdd_des_fn)(trajectory.t),
        pressures=trajectory.u,
    )


def tracking_metrics(result: TrackingResult) -> dict[str, Array]:
    """Return per-joint and aggregate tracking errors."""
    error = result.q_des - result.q
    return {
        "per_joint_rmse": jnp.sqrt(jnp.mean(error**2, axis=0)),
        "overall_rmse": jnp.sqrt(jnp.mean(error**2)),
        "maximum_absolute_error": jnp.max(jnp.abs(error)),
    }


def plot_results(
    result: TrackingResult,
    *,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    show: bool = True,
) -> tuple[Path, Path]:
    """Plot joint tracking, reference dynamics, and antagonistic pressures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10.colors

    tracking_figure, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    tracking_figure.suptitle(
        "UMArm joint-space tracking with FeedforwardCompensationTracker\n"
        "Balanced antagonistic differential-pressure control",
        fontsize=14,
    )
    for segment_index, axis in enumerate(axes[:3]):
        start = 4 * segment_index
        for local_index, joint_index in enumerate(range(start, start + 4)):
            color = colors[local_index]
            axis.plot(
                result.t,
                result.q[:, joint_index],
                color=color,
                linewidth=1.5,
                alpha=0.9,
                zorder=2,
                label=f"q{joint_index}",
            )
            axis.plot(
                result.t,
                result.q_des[:, joint_index],
                linestyle=(0, (4, 2)),
                color=color,
                linewidth=2.6,
                zorder=3,
            )
        axis.set_ylabel("Angle [rad]")
        axis.set_title(
            f"Segment {segment_index + 1} (solid: actual, thick dashed: reference)"
        )
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", ncol=4)

    speed_norm = jnp.linalg.norm(result.qd_des, axis=1)
    acceleration_norm = jnp.linalg.norm(result.qdd_des, axis=1)
    axes[3].plot(result.t, speed_norm, color="#348ABD", label=r"$\|\dot q_d\|$")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel("Speed [rad/s]", color="#348ABD")
    axes[3].tick_params(axis="y", labelcolor="#348ABD")
    axes[3].grid(True, alpha=0.3)
    acceleration_axis = axes[3].twinx()
    acceleration_axis.plot(
        result.t,
        acceleration_norm,
        color="#E24A33",
        label=r"$\|\ddot q_d\|$",
    )
    acceleration_axis.set_ylabel(r"Acceleration [rad/s$^2$]", color="#E24A33")
    acceleration_axis.tick_params(axis="y", labelcolor="#E24A33")
    axes[3].set_title("Reference motion intensity")

    tracking_figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    tracking_path = figures_dir / "track_mckibben_umarm_joint_tracking.pdf"
    tracking_figure.savefig(tracking_path, dpi=200, bbox_inches="tight")

    pressure_figure, pressure_axes = plt.subplots(
        3,
        1,
        figsize=(11, 9),
        sharex=True,
    )
    pressure_kpa = result.pressures / 1.0e3
    pressure_axes[0].plot(result.t, pressure_kpa[:, 3], label="channel 3 (+x)")
    pressure_axes[0].plot(result.t, pressure_kpa[:, 1], label="channel 1 (-x)")
    pressure_axes[0].set_title("First universal joint: x-axis antagonist pair")
    pressure_axes[1].plot(result.t, pressure_kpa[:, 0], label="channel 0 (+y)")
    pressure_axes[1].plot(result.t, pressure_kpa[:, 2], label="channel 2 (-y)")
    pressure_axes[1].set_title("First universal joint: y-axis antagonist pair")
    pressure_axes[2].fill_between(
        result.t,
        jnp.min(pressure_kpa, axis=1),
        jnp.max(pressure_kpa, axis=1),
        color="#988ED5",
        alpha=0.35,
        label="range across all 24 muscles",
    )
    pressure_axes[2].plot(
        result.t,
        jnp.mean(pressure_kpa, axis=1),
        color="#555555",
        label="mean pressure",
    )
    pressure_axes[2].set_title("Pressure activity across the arm")
    pressure_axes[2].set_xlabel("Time [s]")
    for axis in pressure_axes:
        axis.set_ylabel("Pressure [kPa]")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")

    pressure_figure.tight_layout()
    pressure_path = figures_dir / "track_mckibben_umarm_pressures.pdf"
    pressure_figure.savefig(pressure_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(tracking_figure)
        plt.close(pressure_figure)
    return tracking_path, pressure_path


def render_motion(
    robot: McKibbenActuatedUMArm,
    result: TrackingResult,
    *,
    record_path: Path | None = None,
) -> None:
    """Render the tracked motion with actuator colors driven by pressure."""
    if UMArmViserRenderer is None:
        print("UMArmViserRenderer is unavailable. Install the Viser extras.")
        return
    if record_path is not None:
        record_path.parent.mkdir(parents=True, exist_ok=True)

    render_t = result.t
    render_q = result.q
    render_pressures = result.pressures
    if record_path is not None and result.t.shape[0] > 1:
        sample_period = float(jnp.mean(jnp.diff(result.t)))
        frame_stride = max(
            1,
            int(round(1.0 / (DEFAULT_RECORDING_FPS * sample_period))),
        )
        frame_indices = list(range(0, result.t.shape[0], frame_stride))
        if frame_indices[-1] != result.t.shape[0] - 1:
            frame_indices.append(result.t.shape[0] - 1)
        frame_indices_array = jnp.asarray(frame_indices)
        render_t = result.t[frame_indices_array]
        render_q = result.q[frame_indices_array]
        render_pressures = result.pressures[frame_indices_array]

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
        plot_configurations=True,
        robot_name="UMArm joint-space tracking",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="Optional path to a UMArm .npz parameter file.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="Maneuver duration [s]. Shorter values increase dynamic excitation.",
    )
    parser.add_argument(
        "--solver-dt",
        type=float,
        default=DEFAULT_SOLVER_DT,
        help="Internal integration step [s].",
    )
    parser.add_argument(
        "--save-dt",
        type=float,
        default=DEFAULT_SAVE_DT,
        help="Saved trajectory sample period [s].",
    )
    parser.add_argument(
        "--nominal-pressure-kpa",
        type=float,
        default=DEFAULT_NOMINAL_PRESSURE / 1.0e3,
        help="Uniform antagonistic nominal pressure [kPa].",
    )
    parser.add_argument(
        "--max-delta-pressure-kpa",
        type=float,
        default=DEFAULT_MAX_DELTA_PRESSURE / 1.0e3,
        help="Symmetric differential-pressure limit [kPa].",
    )
    parser.add_argument(
        "--natural-frequency",
        type=float,
        default=DEFAULT_NATURAL_FREQUENCY,
        help="Straight-arm feedback natural frequency [rad/s].",
    )
    parser.add_argument(
        "--damping-ratio",
        type=float,
        default=DEFAULT_DAMPING_RATIO,
        help="Straight-arm feedback damping ratio.",
    )
    parser.add_argument(
        "--render",
        choices=("none", "viser"),
        default="viser",
        help="Motion renderer to launch after plotting.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plots.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save plots without opening Matplotlib windows.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory for generated PDF plots.",
    )
    parser.add_argument(
        "--record",
        type=Path,
        nargs="?",
        const=DEFAULT_VIDEO_PATH,
        default=None,
        help=f"Record the Viser motion; default path is {DEFAULT_VIDEO_PATH}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot = build_robot(args.params)
    result = simulate(
        robot,
        duration=args.duration,
        solver_dt=args.solver_dt,
        save_dt=args.save_dt,
        nominal_pressure=args.nominal_pressure_kpa * 1.0e3,
        max_delta_pressure=args.max_delta_pressure_kpa * 1.0e3,
        natural_frequency=args.natural_frequency,
        damping_ratio=args.damping_ratio,
    )
    metrics = tracking_metrics(result)

    print(f"Saved samples: {result.t.shape[0]}")
    print(f"Overall joint RMSE: {float(metrics['overall_rmse']):.5f} rad")
    print(
        "Maximum absolute joint error: "
        f"{float(metrics['maximum_absolute_error']):.5f} rad"
    )
    print(
        "Peak desired speed/acceleration: "
        f"{float(jnp.max(jnp.abs(result.qd_des))):.3f} rad/s, "
        f"{float(jnp.max(jnp.abs(result.qdd_des))):.3f} rad/s^2"
    )
    print(
        "Physical pressure range: "
        f"{float(jnp.min(result.pressures)) / 1.0e3:.2f} to "
        f"{float(jnp.max(result.pressures)) / 1.0e3:.2f} kPa"
    )

    if not args.no_plots:
        tracking_path, pressure_path = plot_results(
            result,
            figures_dir=args.figures_dir,
            show=not args.no_show,
        )
        print(f"Tracking plot: {tracking_path}")
        print(f"Pressure plot: {pressure_path}")
    if args.render == "viser":
        render_motion(robot, result, record_path=args.record)


if __name__ == "__main__":
    main()
