"""Simulation and saved-data helpers for configuration-space control examples."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from soromox.control import PIDControl, PIDControllerState, ReferenceTrajectory
from soromox.control.configuration_space import (
    ComputedTorqueTracker,
    FeedforwardCompensationTracker,
    GravityCancellationRegulator,
    MixedStateFeedbackTracker,
    PIDController,
    PotentialCancellationRegulator,
    PotentialCompensationRegulator,
)
from soromox.systems import PCS, PCSParams, SystemState

DEFAULT_SOLVER_DT = 5e-5
DEFAULT_SAVE_DT = 0.01
DEFAULT_SETPOINT_OUTPUT = "setpoint_regulation_comparison.npz"
DEFAULT_TRAJECTORY_OUTPUT = "trajectory_tracking_comparison.npz"
DEFAULT_REGULATION_TRACKING_OUTPUT = "regulation_tracking_comparison.npz"
DEFAULT_REGULATION_TRACKING_FREQUENCY_SCALE = 10.0
DEFAULT_FEEDBACK_NATURAL_FREQUENCY = 30.0
DEFAULT_FEEDBACK_DAMPING_RATIO = 0.9
DEFAULT_FEEDBACK_INTEGRAL_FREQUENCY = 0.75
DEFAULT_BASE_POSE = (0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0)
HORIZONTAL_BASE_POSE = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

STRAIN_INDICES = (1, 2, 3)
STRAIN_NAMES = (r"$\kappa_y$", r"$\kappa_z$", r"$\sigma_x$")

SETPOINT_COLORS = {
    "PID (model-free)": "#E24A33",
    "Potential Compensation": "#348ABD",
    "Potential Cancellation": "#988ED5",
    "Gravity Cancellation": "#8EBA42",
}

TRACKER_COLORS = {
    "Computed Torque": "#E24A33",
    "Feedforward Compensation": "#348ABD",
    "Mixed State Feedback": "#988ED5",
}

REGULATOR_COLORS = SETPOINT_COLORS

REGULATION_TRACKING_COLORS = {
    "PD (model-free)": "#B2B2B2",
    "PID (model-free)": "#7F7F7F",
    "Potential Compensation": "#348ABD",
    "Feedforward Compensation": "#988ED5",
    "Computed Torque": "#E24A33",
}


@dataclass(frozen=True)
class ScenarioInfo:
    """Metadata for one simulated scenario."""

    key: str
    title: str


@dataclass(frozen=True)
class PlotGroup:
    """A controller group to plot from one scenario."""

    key: str
    scenario_key: str
    title: str
    filename_prefix: str
    controller_names: tuple[str, ...]
    colors: dict[str, str]
    event_times: tuple[float, ...] = ()
    phase_boundary_time: float | None = None
    phase_boundary_label: str | None = None


@dataclass(frozen=True)
class RmsePanel:
    """One panel in an RMSE bar-chart summary."""

    title: str
    scenario_key: str
    controller_names: tuple[str, ...]


@dataclass(frozen=True)
class RmseSummary:
    """A saved RMSE summary figure specification."""

    key: str
    filename: str
    panels: tuple[RmsePanel, ...]


@dataclass(frozen=True)
class RenderTarget:
    """Default Viser render target for a saved comparison."""

    key: str
    scenario_key: str
    title: str
    default_controller: str


@dataclass
class ComparisonRun:
    """Saved rollout data for plotting and rendering."""

    example_key: str
    title: str
    strain_indices: tuple[int, ...]
    strain_names: tuple[str, ...]
    scenarios: tuple[ScenarioInfo, ...]
    results: dict[str, dict[str, dict[str, Any]]]
    plot_groups: tuple[PlotGroup, ...]
    rmse_summaries: tuple[RmseSummary, ...]
    render_targets: tuple[RenderTarget, ...]

    def scenario_title(self, scenario_key: str) -> str:
        for scenario in self.scenarios:
            if scenario.key == scenario_key:
                return scenario.title
        return scenario_key


def create_robot(
    *,
    base_pose: tuple[float, ...] = DEFAULT_BASE_POSE,
) -> tuple[PCS, int, int]:
    """Create the PCS robot used by the configuration-space examples."""
    num_segments = 1
    rho = 1070 * jnp.ones((num_segments,))

    segment_lengths = 1e-1 * jnp.ones((num_segments,))
    damping_matrix = 3e-4 * jnp.diag(
        (
            jnp.repeat(
                jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]),
                num_segments,
                axis=0,
            )
            * segment_lengths[:, None]
        ).flatten()
    )
    params = PCSParams(
        base_pose=jnp.asarray(base_pose),
        length=segment_lengths,
        radius=2e-2 * jnp.ones((num_segments,)),
        density=rho,
        gravity=jnp.array([0.0, 0.0, -9.81]),
        young_modulus=2e3 * jnp.ones((num_segments,)),
        shear_modulus=1e3 * jnp.ones((num_segments,)),
        damping_matrix=damping_matrix,
        reference_strain=jnp.tile(
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            num_segments,
        ),
    )

    robot = PCS(params=params)
    return robot, int(robot.num_active_strains), num_segments


def create_regulation_tracking_robot() -> tuple[PCS, int, int]:
    """Create the horizontally mounted robot used by the combined task."""
    return create_robot(base_pose=HORIZONTAL_BASE_POSE)


def create_pid_control(
    robot: PCS,
    *,
    natural_frequency: float = DEFAULT_FEEDBACK_NATURAL_FREQUENCY,
    damping_ratio: float = DEFAULT_FEEDBACK_DAMPING_RATIO,
    integral_frequency: float = DEFAULT_FEEDBACK_INTEGRAL_FREQUENCY,
    q_normalization: jnp.ndarray | None = None,
) -> PIDControl:
    """Create inertia-scaled generalized-force PID gains.

    A common acceleration-domain feedback law is selected first,

    ``qdd_fb = kp_acc * e + ki_acc * integral(e) + kd_acc * ed``,

    and then mapped to generalized force using the inertia matrix at the
    normalization configuration. This gives rotational and linear strains the
    same local acceleration correction per unit error despite their very
    different inertias and units.
    """
    if natural_frequency <= 0:
        raise ValueError("natural_frequency must be positive.")
    if damping_ratio <= 0:
        raise ValueError("damping_ratio must be positive.")
    if integral_frequency < 0:
        raise ValueError("integral_frequency must be nonnegative.")
    if q_normalization is None:
        q_normalization = jnp.zeros((robot.num_dofs,))
    if q_normalization.shape != (robot.num_dofs,):
        raise ValueError(
            "q_normalization must have shape "
            f"({robot.num_dofs},), got {q_normalization.shape}."
        )

    inertia = robot.inertia_matrix(q_normalization)
    kp_acc = natural_frequency**2
    kd_acc = 2.0 * damping_ratio * natural_frequency
    ki_acc = integral_frequency * kp_acc

    print("Inertia-scaled generalized-force PID gains:")
    print(
        "  Acceleration-domain gains: "
        f"Kp={kp_acc:.6g}, Ki={ki_acc:.6g}, Kd={kd_acc:.6g}"
    )
    print(f"  Normalization configuration: {q_normalization}")

    return PIDControl(
        Kp=kp_acc * inertia,
        Ki=ki_acc * inertia,
        Kd=kd_acc * inertia,
        saturation_fn="tanh",
        gamma=1.0,
    )


def remove_integral_action(pid_control: PIDControl) -> PIDControl:
    """Return a PD controller with the same proportional and derivative gains."""
    return PIDControl(
        Kp=pid_control.Kp,
        Ki=jnp.zeros_like(pid_control.Ki),
        Kd=pid_control.Kd,
        saturation_fn="identity",
        gamma=pid_control.gamma,
    )


def normalize_pid_control_for_computed_torque(
    robot: PCS,
    pid_control: PIDControl,
    q_normalization: jnp.ndarray | None = None,
) -> PIDControl:
    """Convert generalized-force PID gains to computed-torque acceleration gains.

    The normalized gains satisfy ``M(q_normalization) @ K_ct = K_tau``. Thus,
    at the normalization configuration, the computed-torque feedback produces
    the same generalized force as the PID feedback used by the other controllers.
    """
    if q_normalization is None:
        q_normalization = jnp.zeros((robot.num_dofs,))
    if q_normalization.shape != (robot.num_dofs,):
        raise ValueError(
            "q_normalization must have shape "
            f"({robot.num_dofs},), got {q_normalization.shape}."
        )

    def gain_matrix(gain: jnp.ndarray) -> jnp.ndarray:
        if gain.ndim == 1:
            if gain.shape == (1,):
                return gain[0] * jnp.eye(robot.num_dofs)
            if gain.shape == (robot.num_dofs,):
                return jnp.diag(gain)
        elif gain.shape == (robot.num_dofs, robot.num_dofs):
            return gain
        raise ValueError(
            "PID gains must be scalar, length-num_dofs vectors, or "
            f"({robot.num_dofs}, {robot.num_dofs}) matrices."
        )

    inertia = robot.inertia_matrix(q_normalization)
    saturation_fn: str | Any
    if pid_control._saturation_fn_name == "custom":
        saturation_fn = pid_control._custom_saturation_fn
    else:
        saturation_fn = pid_control._saturation_fn_name

    return PIDControl(
        Kp=jnp.linalg.solve(inertia, gain_matrix(pid_control.Kp)),
        Ki=jnp.linalg.solve(inertia, gain_matrix(pid_control.Ki)),
        Kd=jnp.linalg.solve(inertia, gain_matrix(pid_control.Kd)),
        saturation_fn=saturation_fn,
        gamma=pid_control.gamma,
    )


def create_setpoint_trajectory(
    num_dofs: int,
    t0: float,
    t1: float,
) -> tuple[ReferenceTrajectory, jnp.ndarray]:
    """Create step setpoints in kappa_y, kappa_z, and sigma_x."""
    ts = jnp.linspace(t0, t1, 1000)
    step_times = jnp.array([0.0, 3.0, 6.0, 9.0, 12.0])

    kappa_y_values = jnp.array([0.0, 20.0, -10.0, 15.0, 0.0])
    kappa_z_values = jnp.array([0.0, 5.0, -8.0, 3.0, 0.0])
    sigma_x_values = jnp.array([0.0, 0.06, -0.03, 0.03, 0.0])

    def x_des_fn(t):
        q_des = jnp.zeros((num_dofs,))
        step_idx = jnp.searchsorted(step_times, t, side="right") - 1
        step_idx = jnp.clip(step_idx, 0, len(kappa_y_values) - 1)

        q_des = q_des.at[1].set(kappa_y_values[step_idx])
        q_des = q_des.at[2].set(kappa_z_values[step_idx])
        q_des = q_des.at[3].set(sigma_x_values[step_idx])
        return q_des

    return ReferenceTrajectory(ts=ts, x_des_fn=x_des_fn), step_times


def create_smooth_trajectory(
    num_dofs: int,
    t0: float,
    t1: float,
    frequency_scale: float = 1.0,
) -> ReferenceTrajectory:
    """Create a smooth sinusoidal trajectory in kappa_y, kappa_z, and sigma_x."""
    ts = jnp.linspace(t0, t1, 1000)

    omega_y = 0.3 * frequency_scale
    omega_z = 0.2 * frequency_scale
    omega_x = 0.25 * frequency_scale

    amp_kappa_y = 2.0
    amp_kappa_z = 1.0
    amp_sigma_x = 0.02

    def x_des_fn(t):
        q_des = jnp.zeros((num_dofs,))
        duration = t1 - t0
        ramp_duration = 0.1 * duration

        ramp_up = 0.5 * (1 - jnp.cos(jnp.pi * jnp.clip((t - t0) / ramp_duration, 0, 1)))
        ramp_down = 0.5 * (
            1
            + jnp.cos(
                jnp.pi * jnp.clip((t - (t1 - ramp_duration)) / ramp_duration, 0, 1)
            )
        )
        ramp = ramp_up * ramp_down

        q_des = q_des.at[1].set(ramp * amp_kappa_y * jnp.sin(omega_y * t))
        q_des = q_des.at[2].set(ramp * amp_kappa_z * jnp.sin(omega_z * t + jnp.pi / 4))
        q_des = q_des.at[3].set(ramp * amp_sigma_x * jnp.sin(omega_x * t + jnp.pi / 2))
        return q_des

    return ReferenceTrajectory(ts=ts, x_des_fn=x_des_fn)


def create_regulation_tracking_trajectory(
    num_dofs: int,
    t0: float,
    tracking_start: float,
    t1: float,
    frequency_scale: float = DEFAULT_REGULATION_TRACKING_FREQUENCY_SCALE,
) -> tuple[ReferenceTrajectory, jnp.ndarray]:
    """Create setpoint regulation followed by smooth trajectory tracking.

    The setpoint phase bends with and against gravity, exercises the orthogonal
    bending plane, and returns to the neutral configuration before tracking
    begins. The tracking phase is multiplied by a quintic ramp, making desired
    position, velocity, and acceleration continuous at the phase boundary.
    """
    if not t0 < tracking_start < t1:
        raise ValueError("Expected t0 < tracking_start < t1.")
    if num_dofs < 4:
        raise ValueError("The combined task requires at least four DOFs.")
    if frequency_scale <= 0:
        raise ValueError("frequency_scale must be positive.")

    ts = jnp.linspace(t0, t1, 1000)
    setpoint_times = jnp.linspace(t0, tracking_start, 6)[:-1]
    # With the horizontal mounting, positive/negative kappa_y bends with/against
    # gravity, while kappa_z exercises the orthogonal bending plane.
    kappa_y_values = jnp.array([0.0, 14.0, -14.0, 7.0, 0.0])
    kappa_z_values = jnp.array([0.0, 7.0, -7.0, -10.0, 0.0])
    sigma_x_values = jnp.array([0.0, 0.10, -0.05, 0.06, 0.0])

    tracking_duration = t1 - tracking_start
    ramp_duration = 0.1 * tracking_duration
    omega_y = 0.3 * frequency_scale
    omega_z = 0.2 * frequency_scale
    omega_x = 0.25 * frequency_scale

    amp_kappa_y = 12.0
    amp_kappa_z = 6.0
    amp_sigma_x = 0.06

    def x_des_fn(t):
        setpoint_idx = jnp.searchsorted(setpoint_times, t, side="right") - 1
        setpoint_idx = jnp.clip(setpoint_idx, 0, len(kappa_y_values) - 1)
        q_setpoint = jnp.zeros((num_dofs,))
        q_setpoint = q_setpoint.at[1].set(kappa_y_values[setpoint_idx])
        q_setpoint = q_setpoint.at[2].set(kappa_z_values[setpoint_idx])
        q_setpoint = q_setpoint.at[3].set(sigma_x_values[setpoint_idx])

        tracking_time = t - tracking_start
        ramp_phase = jnp.clip(tracking_time / ramp_duration, 0.0, 1.0)
        ramp = ramp_phase**3 * (10.0 - 15.0 * ramp_phase + 6.0 * ramp_phase**2)
        q_tracking = jnp.zeros((num_dofs,))
        q_tracking = q_tracking.at[1].set(
            ramp * amp_kappa_y * jnp.sin(omega_y * tracking_time)
        )
        q_tracking = q_tracking.at[2].set(
            ramp * amp_kappa_z * jnp.sin(omega_z * tracking_time + jnp.pi / 4)
        )
        q_tracking = q_tracking.at[3].set(
            ramp * amp_sigma_x * jnp.sin(omega_x * tracking_time + jnp.pi / 2)
        )

        return jnp.where(t < tracking_start, q_setpoint, q_tracking)

    return ReferenceTrajectory(ts=ts, x_des_fn=x_des_fn), setpoint_times


def run_simulation(
    robot: PCS,
    controller: Any,
    num_dofs: int,
    t0: float,
    t1: float,
    *,
    solver_dt: float = DEFAULT_SOLVER_DT,
    save_dt: float = DEFAULT_SAVE_DT,
) -> dict[str, Any]:
    """Run one closed-loop simulation."""
    q0 = jnp.zeros((num_dofs,))
    qd0 = jnp.zeros((num_dofs,))
    y0 = jnp.concatenate([q0, qd0])
    control_state_0 = PIDControllerState.zero(num_dofs)

    initial_state = SystemState(
        t=jnp.array(t0),
        y=y0,
        u=jnp.zeros((robot.num_actuators,)),
        control_state=control_state_0,
    )

    trajectory = robot.rollout_closed_loop_to(
        initial_state=initial_state,
        controller=controller,
        t1=t1,
        solver_dt=solver_dt,
        save_dt=save_dt,
    )
    q_traj, qd_traj = jnp.split(trajectory.y, 2, axis=1)

    return {
        "t": trajectory.t,
        "q": q_traj,
        "qd": qd_traj,
        "u": trajectory.u,
    }


def compute_metrics(
    results: dict[str, Any],
    q_des_fn: Any,
    strain_indices: tuple[int, ...],
) -> dict[str, Any]:
    """Compute tracking metrics for one simulation result."""
    t = results["t"]
    q = results["q"]
    q_des = jax.vmap(q_des_fn)(t)
    tracking_error = q_des - q
    tracked_error = tracking_error[:, list(strain_indices)]
    dt = t[1] - t[0]

    return {
        "rmse": jnp.sqrt(jnp.mean(tracked_error**2, axis=0)),
        "max_error": jnp.max(jnp.abs(tracked_error), axis=0),
        "ise": jnp.sum(tracked_error**2, axis=0) * dt,
        "tracking_error": tracking_error,
        "q_des": q_des,
    }


def run_controller_set(
    robot: PCS,
    controller_classes: dict[str, type],
    reference_trajectory: ReferenceTrajectory,
    pid_control: PIDControl,
    num_dofs: int,
    t0: float,
    t1: float,
    *,
    solver_dt: float,
    save_dt: float,
    controller_pid_controls: dict[str, PIDControl] | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run controllers, optionally overriding the PID gains for selected names."""
    all_results: dict[str, dict[str, Any]] = {}
    for name, controller_class in controller_classes.items():
        if verbose:
            print(f"\nRunning simulation with {name}...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            controller_pid_control = (
                controller_pid_controls.get(name, pid_control)
                if controller_pid_controls is not None
                else pid_control
            )
            controller = controller_class(
                robot=robot,
                reference_trajectory=reference_trajectory,
                pid_control=controller_pid_control,
            )

        results = run_simulation(
            robot,
            controller,
            num_dofs,
            t0,
            t1,
            solver_dt=solver_dt,
            save_dt=save_dt,
        )
        results["metrics"] = compute_metrics(
            results,
            reference_trajectory.x_des_fn,
            STRAIN_INDICES,
        )
        all_results[name] = results

        if verbose:
            print(f"  Completed {len(results['t'])} time steps.")

    return all_results


def slice_results(
    all_results: dict[str, dict[str, Any]],
    reference_trajectory: ReferenceTrajectory,
    *,
    start: float | None = None,
    stop: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Slice controller results and recompute metrics over the selected interval."""
    sliced_results: dict[str, dict[str, Any]] = {}
    for name, results in all_results.items():
        t = results["t"]
        mask = jnp.ones_like(t, dtype=bool)
        if start is not None:
            mask = jnp.logical_and(mask, t >= start)
        if stop is not None:
            mask = jnp.logical_and(mask, t < stop)

        sliced = {key: results[key][mask] for key in ("t", "q", "qd", "u")}
        if len(sliced["t"]) < 2:
            raise ValueError("A metrics interval must contain at least two samples.")
        sliced["metrics"] = compute_metrics(
            sliced,
            reference_trajectory.x_des_fn,
            STRAIN_INDICES,
        )
        sliced_results[name] = sliced

    return sliced_results


def run_setpoint_comparison(
    *,
    t0: float = 0.0,
    t1: float = 15.0,
    solver_dt: float = DEFAULT_SOLVER_DT,
    save_dt: float = DEFAULT_SAVE_DT,
    verbose: bool = True,
) -> ComparisonRun:
    """Run the setpoint regulation comparison."""
    robot, num_dofs, _ = create_robot()
    if verbose:
        print(f"Number of DOFs: {num_dofs}")
        print(f"Number of actuators: {robot.num_actuators}")

    pid_control = create_pid_control(robot)
    reference_trajectory, step_times = create_setpoint_trajectory(num_dofs, t0, t1)
    controllers = {
        "PID (model-free)": PIDController,
        "Potential Compensation": PotentialCompensationRegulator,
        "Potential Cancellation": PotentialCancellationRegulator,
        "Gravity Cancellation": GravityCancellationRegulator,
    }

    results = run_controller_set(
        robot,
        controllers,
        reference_trajectory,
        pid_control,
        num_dofs,
        t0,
        t1,
        solver_dt=solver_dt,
        save_dt=save_dt,
        verbose=verbose,
    )

    run = ComparisonRun(
        example_key="setpoint_regulation",
        title="Setpoint Regulation",
        strain_indices=STRAIN_INDICES,
        strain_names=STRAIN_NAMES,
        scenarios=(ScenarioInfo("setpoint", "Setpoint Regulation"),),
        results={"setpoint": results},
        plot_groups=(
            PlotGroup(
                key="controllers",
                scenario_key="setpoint",
                title="Setpoint Regulation: Strain Tracking Comparison",
                filename_prefix="setpoint_regulation",
                controller_names=tuple(controllers.keys()),
                colors=SETPOINT_COLORS,
                event_times=tuple(float(t) for t in step_times),
            ),
        ),
        rmse_summaries=(
            RmseSummary(
                key="rmse",
                filename="setpoint_regulation_rmse.pdf",
                panels=(
                    RmsePanel(
                        title="Setpoint Regulation: RMSE Comparison",
                        scenario_key="setpoint",
                        controller_names=tuple(controllers.keys()),
                    ),
                ),
            ),
        ),
        render_targets=(
            RenderTarget(
                key="setpoint",
                scenario_key="setpoint",
                title="Setpoint Regulation",
                default_controller="PID (model-free)",
            ),
        ),
    )
    if verbose:
        print_metrics_table(results, STRAIN_NAMES, "Setpoint Regulation: RMSE")
    return run


def run_regulation_tracking_comparison(
    *,
    t0: float = 0.0,
    tracking_start: float = 15.0,
    t1: float = 30.0,
    tracking_frequency_scale: float = DEFAULT_REGULATION_TRACKING_FREQUENCY_SCALE,
    feedback_natural_frequency: float = DEFAULT_FEEDBACK_NATURAL_FREQUENCY,
    feedback_damping_ratio: float = DEFAULT_FEEDBACK_DAMPING_RATIO,
    feedback_integral_frequency: float = DEFAULT_FEEDBACK_INTEGRAL_FREQUENCY,
    solver_dt: float = DEFAULT_SOLVER_DT,
    save_dt: float = DEFAULT_SAVE_DT,
    verbose: bool = True,
) -> ComparisonRun:
    """Run one continuous regulation-then-tracking controller comparison."""
    robot, num_dofs, _ = create_regulation_tracking_robot()
    if verbose:
        print(f"Number of DOFs: {num_dofs}")
        print(f"Number of actuators: {robot.num_actuators}")

    pid_control = create_pid_control(
        robot,
        natural_frequency=feedback_natural_frequency,
        damping_ratio=feedback_damping_ratio,
        integral_frequency=feedback_integral_frequency,
    )
    pd_control = remove_integral_action(pid_control)
    computed_torque_pid_control = normalize_pid_control_for_computed_torque(
        robot,
        pid_control,
    )
    if verbose:
        print("Computed-torque gains normalized with M(q=0).")
    reference_trajectory, setpoint_times = create_regulation_tracking_trajectory(
        num_dofs,
        t0,
        tracking_start,
        t1,
        frequency_scale=tracking_frequency_scale,
    )
    controllers = {
        "PD (model-free)": PIDController,
        "PID (model-free)": PIDController,
        "Potential Compensation": PotentialCompensationRegulator,
        "Feedforward Compensation": FeedforwardCompensationTracker,
        "Computed Torque": ComputedTorqueTracker,
    }

    if verbose:
        print("\n" + "=" * 80)
        print("SETPOINT REGULATION FOLLOWED BY TRAJECTORY TRACKING")
        print("=" * 80)
    combined_results = run_controller_set(
        robot,
        controllers,
        reference_trajectory,
        pid_control,
        num_dofs,
        t0,
        t1,
        solver_dt=solver_dt,
        save_dt=save_dt,
        controller_pid_controls={
            "PD (model-free)": pd_control,
            "Computed Torque": computed_torque_pid_control,
        },
        verbose=verbose,
    )
    regulation_results = slice_results(
        combined_results,
        reference_trajectory,
        stop=tracking_start,
    )
    tracking_results = slice_results(
        combined_results,
        reference_trajectory,
        start=tracking_start,
    )

    run = ComparisonRun(
        example_key="regulation_tracking",
        title="Setpoint Regulation Followed by Trajectory Tracking",
        strain_indices=STRAIN_INDICES,
        strain_names=STRAIN_NAMES,
        scenarios=(
            ScenarioInfo("combined", "Complete Task"),
            ScenarioInfo("regulation", "Setpoint Regulation Phase"),
            ScenarioInfo("tracking", "Trajectory Tracking Phase"),
        ),
        results={
            "combined": combined_results,
            "regulation": regulation_results,
            "tracking": tracking_results,
        },
        plot_groups=(
            PlotGroup(
                key="combined",
                scenario_key="combined",
                title="Regulation and Tracking: Configuration-Space Comparison",
                filename_prefix="regulation_tracking",
                controller_names=tuple(controllers.keys()),
                colors=REGULATION_TRACKING_COLORS,
                event_times=tuple(float(t) for t in setpoint_times),
                phase_boundary_time=tracking_start,
                phase_boundary_label="Trajectory tracking starts",
            ),
            PlotGroup(
                key="tracking-phase",
                scenario_key="tracking",
                title="Trajectory Tracking Phase: Configuration-Space Comparison",
                filename_prefix="trajectory_tracking_phase",
                controller_names=tuple(controllers.keys()),
                colors=REGULATION_TRACKING_COLORS,
            ),
        ),
        rmse_summaries=(
            RmseSummary(
                key="phase-rmse",
                filename="regulation_tracking_rmse.pdf",
                panels=(
                    RmsePanel(
                        title="Setpoint Regulation: RMSE",
                        scenario_key="regulation",
                        controller_names=tuple(controllers.keys()),
                    ),
                    RmsePanel(
                        title="Trajectory Tracking: RMSE",
                        scenario_key="tracking",
                        controller_names=tuple(controllers.keys()),
                    ),
                ),
            ),
        ),
        render_targets=(
            RenderTarget(
                key="combined",
                scenario_key="combined",
                title="Regulation and Tracking",
                default_controller="Feedforward Compensation",
            ),
        ),
    )

    if verbose:
        for phase_title, phase_results in (
            ("Setpoint Regulation Phase", regulation_results),
            ("Trajectory Tracking Phase", tracking_results),
        ):
            for metric_name, metric_title in (
                ("rmse", "RMSE"),
                ("max_error", "Maximum Absolute Error"),
                ("ise", "ISE"),
            ):
                print_metrics_table(
                    phase_results,
                    STRAIN_NAMES,
                    f"{phase_title}: {metric_title}",
                    metric_name=metric_name,
                )
    return run


def run_trajectory_tracking_comparison(
    *,
    t0: float = 0.0,
    t1_slow: float = 30.0,
    t1_fast: float = 15.0,
    slow_frequency_scale: float = 1.0,
    fast_frequency_scale: float = 3.0,
    solver_dt: float = DEFAULT_SOLVER_DT,
    save_dt: float = DEFAULT_SAVE_DT,
    verbose: bool = True,
) -> ComparisonRun:
    """Run the trajectory tracking comparison."""
    robot, num_dofs, _ = create_robot()
    if verbose:
        print(f"Number of DOFs: {num_dofs}")
        print(f"Number of actuators: {robot.num_actuators}")

    pid_control = create_pid_control(robot)
    computed_torque_pid_control = normalize_pid_control_for_computed_torque(
        robot,
        pid_control,
    )
    if verbose:
        print("Computed-torque gains normalized with M(q=0).")

    trackers = {
        "Computed Torque": ComputedTorqueTracker,
        "Feedforward Compensation": FeedforwardCompensationTracker,
        "Mixed State Feedback": MixedStateFeedbackTracker,
    }
    regulators = {
        "PID (model-free)": PIDController,
        "Potential Compensation": PotentialCompensationRegulator,
        "Potential Cancellation": PotentialCancellationRegulator,
        "Gravity Cancellation": GravityCancellationRegulator,
    }

    if verbose:
        print("\n" + "=" * 80)
        print("PART 1: SLOW TRAJECTORY (Quasi-static)")
        print("=" * 80)
    reference_slow = create_smooth_trajectory(
        num_dofs,
        t0,
        t1_slow,
        frequency_scale=slow_frequency_scale,
    )
    slow_results = run_controller_set(
        robot,
        {**trackers, **regulators},
        reference_slow,
        pid_control,
        num_dofs,
        t0,
        t1_slow,
        solver_dt=solver_dt,
        save_dt=save_dt,
        controller_pid_controls={
            "Computed Torque": computed_torque_pid_control,
        },
        verbose=verbose,
    )
    if verbose:
        print_metrics_table(slow_results, STRAIN_NAMES, "Slow Trajectory: RMSE")

    if verbose:
        print("\n" + "=" * 80)
        print("PART 2: FAST TRAJECTORY (Dynamic)")
        print("=" * 80)
    reference_fast = create_smooth_trajectory(
        num_dofs,
        t0,
        t1_fast,
        frequency_scale=fast_frequency_scale,
    )
    fast_results = run_controller_set(
        robot,
        trackers,
        reference_fast,
        pid_control,
        num_dofs,
        t0,
        t1_fast,
        solver_dt=solver_dt,
        save_dt=save_dt,
        controller_pid_controls={
            "Computed Torque": computed_torque_pid_control,
        },
        verbose=verbose,
    )
    if verbose:
        print_metrics_table(
            fast_results,
            STRAIN_NAMES,
            "Fast Trajectory: RMSE Comparison (Trackers)",
        )

    return ComparisonRun(
        example_key="trajectory_tracking",
        title="Trajectory Tracking",
        strain_indices=STRAIN_INDICES,
        strain_names=STRAIN_NAMES,
        scenarios=(
            ScenarioInfo("slow", "Slow Trajectory"),
            ScenarioInfo("fast", "Fast Trajectory"),
        ),
        results={"slow": slow_results, "fast": fast_results},
        plot_groups=(
            PlotGroup(
                key="slow-trackers",
                scenario_key="slow",
                title="Slow Trajectory: Tracking Controllers",
                filename_prefix="trajectory_slow_trackers",
                controller_names=tuple(trackers.keys()),
                colors=TRACKER_COLORS,
            ),
            PlotGroup(
                key="slow-regulators",
                scenario_key="slow",
                title="Slow Trajectory: Regulators (quasi-static)",
                filename_prefix="trajectory_slow_regulators",
                controller_names=tuple(regulators.keys()),
                colors=REGULATOR_COLORS,
            ),
            PlotGroup(
                key="fast-trackers",
                scenario_key="fast",
                title="Fast Trajectory: Tracking Controllers",
                filename_prefix="trajectory_fast_trackers",
                controller_names=tuple(trackers.keys()),
                colors=TRACKER_COLORS,
            ),
        ),
        rmse_summaries=(
            RmseSummary(
                key="rmse-summary",
                filename="trajectory_tracking_rmse_summary.pdf",
                panels=(
                    RmsePanel(
                        title="Slow Trajectory: All Controllers",
                        scenario_key="slow",
                        controller_names=(*trackers.keys(), *regulators.keys()),
                    ),
                    RmsePanel(
                        title="Fast Trajectory: Tracking Controllers",
                        scenario_key="fast",
                        controller_names=tuple(trackers.keys()),
                    ),
                ),
            ),
        ),
        render_targets=(
            RenderTarget(
                key="slow",
                scenario_key="slow",
                title="Slow Trajectory",
                default_controller="Computed Torque",
            ),
            RenderTarget(
                key="fast",
                scenario_key="fast",
                title="Fast Trajectory",
                default_controller="Computed Torque",
            ),
        ),
    )


def print_metrics_table(
    all_results: dict[str, dict[str, Any]],
    strain_names: tuple[str, ...],
    title: str,
    *,
    metric_name: str = "rmse",
) -> None:
    """Print a formatted table for one per-strain metric."""
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)
    print(
        f"{'Controller':<30} "
        f"{strain_names[0]:>15} {strain_names[1]:>15} {strain_names[2]:>15}"
    )
    print("-" * 80)

    for name, results in all_results.items():
        metric = results["metrics"][metric_name]
        print(f"{name:<30} {metric[0]:>15.4f} {metric[1]:>15.4f} {metric[2]:>15.4f}")


def save_comparison_npz(path: str | Path, run: ComparisonRun) -> Path:
    """Save a comparison run as compressed NPZ plus JSON metadata."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "format_version": 1,
        "example_key": run.example_key,
        "title": run.title,
        "strain_indices": list(run.strain_indices),
        "strain_names": list(run.strain_names),
        "scenarios": [
            {
                "key": scenario.key,
                "title": scenario.title,
                "controller_names": list(run.results[scenario.key].keys()),
            }
            for scenario in run.scenarios
        ],
        "plot_groups": [_plot_group_to_dict(group) for group in run.plot_groups],
        "rmse_summaries": [
            _rmse_summary_to_dict(summary) for summary in run.rmse_summaries
        ],
        "render_targets": [
            _render_target_to_dict(target) for target in run.render_targets
        ],
    }

    arrays: dict[str, np.ndarray] = {"metadata": np.array(json.dumps(metadata))}
    for scenario_idx, scenario in enumerate(run.scenarios):
        scenario_results = run.results[scenario.key]
        for controller_idx, controller_name in enumerate(scenario_results):
            result = scenario_results[controller_name]
            prefix = f"s{scenario_idx}_c{controller_idx}"
            arrays[f"{prefix}_t"] = np.asarray(result["t"])
            arrays[f"{prefix}_q"] = np.asarray(result["q"])
            arrays[f"{prefix}_qd"] = np.asarray(result["qd"])
            arrays[f"{prefix}_u"] = np.asarray(result["u"])

            metrics = result["metrics"]
            for metric_name in ("rmse", "max_error", "ise", "tracking_error", "q_des"):
                arrays[f"{prefix}_metrics_{metric_name}"] = np.asarray(
                    metrics[metric_name]
                )

    np.savez_compressed(output, **arrays)
    return output


def load_comparison_npz(path: str | Path) -> ComparisonRun:
    """Load a comparison run saved by :func:`save_comparison_npz`."""
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        scenarios = tuple(
            ScenarioInfo(scenario["key"], scenario["title"])
            for scenario in metadata["scenarios"]
        )

        results: dict[str, dict[str, dict[str, Any]]] = {}
        for scenario_idx, scenario in enumerate(metadata["scenarios"]):
            scenario_key = scenario["key"]
            scenario_results: dict[str, dict[str, Any]] = {}
            for controller_idx, controller_name in enumerate(
                scenario["controller_names"]
            ):
                prefix = f"s{scenario_idx}_c{controller_idx}"
                scenario_results[controller_name] = {
                    "t": data[f"{prefix}_t"],
                    "q": data[f"{prefix}_q"],
                    "qd": data[f"{prefix}_qd"],
                    "u": data[f"{prefix}_u"],
                    "metrics": {
                        "rmse": data[f"{prefix}_metrics_rmse"],
                        "max_error": data[f"{prefix}_metrics_max_error"],
                        "ise": data[f"{prefix}_metrics_ise"],
                        "tracking_error": data[f"{prefix}_metrics_tracking_error"],
                        "q_des": data[f"{prefix}_metrics_q_des"],
                    },
                }
            results[scenario_key] = scenario_results

    return ComparisonRun(
        example_key=metadata["example_key"],
        title=metadata["title"],
        strain_indices=tuple(int(i) for i in metadata["strain_indices"]),
        strain_names=tuple(metadata["strain_names"]),
        scenarios=scenarios,
        results=results,
        plot_groups=tuple(
            _plot_group_from_dict(group) for group in metadata["plot_groups"]
        ),
        rmse_summaries=tuple(
            _rmse_summary_from_dict(summary) for summary in metadata["rmse_summaries"]
        ),
        render_targets=tuple(
            _render_target_from_dict(target) for target in metadata["render_targets"]
        ),
    )


def _plot_group_to_dict(group: PlotGroup) -> dict[str, Any]:
    return {
        "key": group.key,
        "scenario_key": group.scenario_key,
        "title": group.title,
        "filename_prefix": group.filename_prefix,
        "controller_names": list(group.controller_names),
        "colors": group.colors,
        "event_times": list(group.event_times),
        "phase_boundary_time": group.phase_boundary_time,
        "phase_boundary_label": group.phase_boundary_label,
    }


def _plot_group_from_dict(data: dict[str, Any]) -> PlotGroup:
    return PlotGroup(
        key=data["key"],
        scenario_key=data["scenario_key"],
        title=data["title"],
        filename_prefix=data["filename_prefix"],
        controller_names=tuple(data["controller_names"]),
        colors=dict(data["colors"]),
        event_times=tuple(float(t) for t in data.get("event_times", ())),
        phase_boundary_time=data.get("phase_boundary_time"),
        phase_boundary_label=data.get("phase_boundary_label"),
    )


def _rmse_summary_to_dict(summary: RmseSummary) -> dict[str, Any]:
    return {
        "key": summary.key,
        "filename": summary.filename,
        "panels": [
            {
                "title": panel.title,
                "scenario_key": panel.scenario_key,
                "controller_names": list(panel.controller_names),
            }
            for panel in summary.panels
        ],
    }


def _rmse_summary_from_dict(data: dict[str, Any]) -> RmseSummary:
    return RmseSummary(
        key=data["key"],
        filename=data["filename"],
        panels=tuple(
            RmsePanel(
                title=panel["title"],
                scenario_key=panel["scenario_key"],
                controller_names=tuple(panel["controller_names"]),
            )
            for panel in data["panels"]
        ),
    )


def _render_target_to_dict(target: RenderTarget) -> dict[str, str]:
    return {
        "key": target.key,
        "scenario_key": target.scenario_key,
        "title": target.title,
        "default_controller": target.default_controller,
    }


def _render_target_from_dict(data: dict[str, str]) -> RenderTarget:
    return RenderTarget(
        key=data["key"],
        scenario_key=data["scenario_key"],
        title=data["title"],
        default_controller=data["default_controller"],
    )
