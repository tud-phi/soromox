"""Shared simulation, plotting, and rendering helpers for the impedance case."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from matplotlib.lines import Line2D

SECVC_CODE_DIR = Path(__file__).parents[2] / "code"
if str(SECVC_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(SECVC_CODE_DIR))

from evaluation_metrics import operational_space_metrics  # noqa: E402
from trajectory_primitives import (
    TaskSpaceTrajectoryConfig,
    make_end_effector_task_selector,
    make_operational_space_reference_trajectory,
)
from viser_operational_space_renderer import render_operational_space_tracking

from soromox.control import OperationalSpaceImpedanceControlTracker
from soromox.coordinate_transformations import OperationalSpaceDynamics
from soromox.rendering import ViserRenderer
from soromox.rendering.camera_config import CameraConfig
from soromox.systems import PCS, PCSParams, SystemState

jax.config.update("jax_enable_x64", True)

CASE_DIR = Path(__file__).parent.parent
DATA_DIR = CASE_DIR / "data"
OUTPUTS_DIR = CASE_DIR / "outputs"
PAPER_STYLE = Path(__file__).parents[3] / "paper.mplstyle"
CM_TO_INCH = 1.0 / 2.54

COORDINATE_COLORS = ("#2a9d8f", "#e9c46a", "#D81159")

DEFAULT_SOLVER_DT = 2e-5
DEFAULT_SAVE_DT = 0.01
DEFAULT_TRAJECTORY_OUTPUT = DATA_DIR / "control_pcs_with_impedance_trajectory.npz"
POSITION_STIFFNESS = 1.0
DAMPING_RATIO = 1.0


def configure_matplotlib() -> None:
    """Apply the shared publication style with dense multi-panel sizing."""
    plt.style.use(PAPER_STYLE)
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "legend.fontsize": 6,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": None,
        }
    )
    if shutil.which("latex") is not None:
        plt.rcParams.update({"text.usetex": True})


def cm_to_inches(size_cm: tuple[float, float]) -> tuple[float, float]:
    """Convert a figure size in centimeters to inches."""
    return tuple(value * CM_TO_INCH for value in size_cm)


@dataclass(frozen=True)
class PCSImpedanceProblem:
    """Fixed robot/control problem setup for this example."""

    robot: PCS
    params: PCSParams
    osd: OperationalSpaceDynamics
    q0: Array
    num_dofs: int
    total_length: float
    trajectory_config: TaskSpaceTrajectoryConfig


@dataclass(frozen=True)
class PCSImpedanceRun:
    """Saved rollout data for plotting and rendering."""

    trajectory_config: TaskSpaceTrajectoryConfig
    t1: float
    solver_dt: float
    save_dt: float
    t_traj: Array
    q_traj: Array
    qd_traj: Array
    u_traj: Array
    x_traj_full: Array
    x_des_traj_full: Array
    metrics: dict[str, float] = field(default_factory=dict)


def build_problem(trajectory_config: TaskSpaceTrajectoryConfig) -> PCSImpedanceProblem:
    """Build the fixed two-segment PCS robot and operational-space task."""
    num_segments = 2
    rho = 1070 * jnp.ones((num_segments,))
    p0 = jnp.array([0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0])
    segment_lengths = 1e-1 * jnp.ones((num_segments,))
    damping_matrix = 1e-3 * jnp.diag(
        (
            jnp.repeat(
                jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), num_segments, axis=0
            )
            * segment_lengths[:, None]
        ).flatten()
    )
    params = PCSParams(
        base_pose=p0,
        length=segment_lengths,
        radius=2e-2 * jnp.ones((num_segments,)),
        density=rho,
        gravity=jnp.array([0.0, 0.0, -9.81]),
        young_modulus=2e3 * jnp.ones((num_segments,)),
        shear_modulus=1e3 * jnp.ones((num_segments,)),
        damping_matrix=damping_matrix,
        reference_strain=jnp.tile(
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), num_segments
        ),
    )

    robot = PCS(params=params)
    num_dofs = robot.num_active_strains
    total_length = float(jnp.sum(segment_lengths))
    q0 = jnp.zeros((num_dofs,))
    osd = OperationalSpaceDynamics(
        robot=robot,
        s_ps=jnp.array([total_length]),
        task_selector=make_end_effector_task_selector(
            trajectory_config.tracking,
            n_velocity_dim=6,
        ),
    )
    return PCSImpedanceProblem(
        robot=robot,
        params=params,
        osd=osd,
        q0=q0,
        num_dofs=num_dofs,
        total_length=total_length,
        trajectory_config=trajectory_config,
    )


def print_problem_summary(problem: PCSImpedanceProblem) -> None:
    """Print a concise setup summary."""
    config = problem.trajectory_config
    osd = problem.osd
    print(f"Number of DOFs: {problem.num_dofs}")
    print(f"Number of actuators: {problem.robot.num_actuators}")
    print(f"Total robot length: {problem.total_length:.3f} m")
    print(f"Tracking mode: {config.tracking}")
    print(f"Position primitive: {config.position_primitive}")
    print(f"Orientation primitive: {config.orientation_primitive}")
    print(f"Surface: {config.surface}")
    print(f"Operational space dimension (velocity): {osd.n_operational_space}")
    print(f"Operational space dimension (pose): {osd.n_pose_operational_space}")
    print(f"Full pose dimension: {osd.n_points * osd.n_pose_dim}")


def _compute_impedance_gains(problem: PCSImpedanceProblem) -> tuple[Array, Array]:
    """Scale full-pose stiffness to the position-task modal bandwidth."""
    osd = problem.osd
    Lambda_diag = jnp.diag(osd.inertia_matrix(problem.q0))
    K_x = POSITION_STIFFNESS * jnp.ones((osd.n_operational_space,))

    if problem.trajectory_config.tracking == "pose":
        n_angular = osd.n_angular_velocity_dim
        Lambda_rot = Lambda_diag[:n_angular]
        Lambda_pos = Lambda_diag[n_angular:]

        # Preserve the origin/main positional stiffness and use its mean
        # operational inertia to define a common target natural frequency.
        # Rotational stiffness then produces the same modal bandwidth instead
        # of the much faster response caused by assigning the same numerical
        # stiffness to angular errors in radians and position errors in metres.
        omega_n = jnp.sqrt(POSITION_STIFFNESS / jnp.mean(Lambda_pos))
        K_x = K_x.at[:n_angular].set(Lambda_rot * omega_n**2)

    D_x = 2.0 * DAMPING_RATIO * jnp.sqrt(K_x * Lambda_diag)
    return K_x, D_x


def run_impedance_simulation(
    problem: PCSImpedanceProblem,
    *,
    t1: float,
    solver_dt: float = DEFAULT_SOLVER_DT,
    save_dt: float = DEFAULT_SAVE_DT,
    verbose: bool = True,
) -> PCSImpedanceRun:
    """Run the closed-loop simulation and return saved rollout data."""
    osd = problem.osd
    t0 = 0.0
    ts = jnp.linspace(t0, t1, 1000)
    x0_full = osd.operational_space_poses(problem.q0)
    if verbose:
        print(f"Initial end-effector pose (full): {x0_full}")

    reference_trajectory = make_operational_space_reference_trajectory(
        osd=osd,
        q0=problem.q0,
        ts=ts,
        config=problem.trajectory_config,
    )
    assert reference_trajectory.x_des_fn is not None

    K_x, D_x = _compute_impedance_gains(problem)
    feedback_linearization = "partial"
    controller = OperationalSpaceImpedanceControlTracker(
        operational_space_dynamics=osd,
        reference_trajectory=reference_trajectory,
        K_x=K_x,
        D_x=D_x,
        feedback_linearization=feedback_linearization,
    )

    if verbose:
        print(f"Feedback linearization: {feedback_linearization}")
        print(f"Operational space stiffness K_x: {K_x}")
        print(f"Operational space damping D_x: {D_x}")

    qd0 = jnp.zeros((problem.num_dofs,))
    initial_state = SystemState(
        t=jnp.array(t0),
        y=jnp.concatenate([problem.q0, qd0]),
        u=jnp.zeros((problem.robot.num_actuators,)),
        control_state=None,
    )

    if verbose:
        print("\nRunning closed-loop simulation with impedance control...")

    trajectory = problem.robot.rollout_closed_loop_to(
        initial_state=initial_state,
        controller=controller,
        t1=t1,
        solver_dt=solver_dt,
        save_dt=save_dt,
    )

    t_traj = trajectory.t
    q_traj, qd_traj = jnp.split(trajectory.y, 2, axis=1)
    assert trajectory.u is not None
    u_traj = trajectory.u
    x_traj_full = jax.vmap(osd.operational_space_poses)(q_traj)
    x_des_traj_full = jax.vmap(reference_trajectory.x_des_fn)(t_traj)

    if verbose:
        print(f"Simulation completed. {len(t_traj)} time steps saved.")

    metrics = compute_operational_metrics(x_traj_full, x_des_traj_full)
    return PCSImpedanceRun(
        trajectory_config=problem.trajectory_config,
        t1=float(t1),
        solver_dt=float(solver_dt),
        save_dt=float(save_dt),
        t_traj=t_traj,
        q_traj=q_traj,
        qd_traj=qd_traj,
        u_traj=u_traj,
        x_traj_full=x_traj_full,
        x_des_traj_full=x_des_traj_full,
        metrics=_scalar_operational_metrics(metrics),
    )


def compute_operational_metrics(
    x_traj_full: Array | np.ndarray,
    x_des_traj_full: Array | np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Compute Section V.C geometric pose errors, RMSE, spans, and NRMSE."""
    x = np.asarray(x_traj_full, dtype=float)
    x_des = np.asarray(x_des_traj_full, dtype=float)
    if x.shape != x_des.shape or x.ndim != 2 or x.shape[1] != 6:
        raise ValueError(
            "Section V.C pose trajectories must have matching shape (samples, 6)."
        )
    return operational_space_metrics(
        position=x[:, 3:6],
        position_reference=x_des[:, 3:6],
        orientation=x[:, :3],
        orientation_reference=x_des[:, :3],
    )


def _scalar_operational_metrics(
    metrics: dict[str, np.ndarray | float],
) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in metrics.items()
        if name not in {"position_error", "orientation_error"}
    }


def save_run_npz(
    path: str | Path, run: PCSImpedanceRun, *, force: bool = False
) -> Path:
    """Save rollout data to an ``.npz`` file."""
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {path}. Pass --force to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    config = run.trajectory_config
    metrics = run.metrics or _scalar_operational_metrics(
        compute_operational_metrics(run.x_traj_full, run.x_des_traj_full)
    )
    np.savez(
        path,
        tracking=np.array(config.tracking),
        position_primitive=np.array(config.position_primitive),
        orientation_primitive=np.array(config.orientation_primitive),
        surface=np.array(config.surface),
        period=np.array(config.period),
        ramp_time=np.array(config.ramp_time),
        position_amplitude_x=np.array(config.position_amplitude_x),
        position_amplitude_y=np.array(config.position_amplitude_y),
        position_amplitude_z=np.array(config.position_amplitude_z),
        line_amplitude=np.array(config.line_amplitude),
        circle_radius=np.array(config.circle_radius),
        helix_radius=np.array(config.helix_radius),
        helix_pitch=np.array(config.helix_pitch),
        orientation_amplitude=np.array(config.orientation_amplitude),
        surface_radius=np.array(config.surface_radius),
        t1=np.array(run.t1),
        solver_dt=np.array(run.solver_dt),
        save_dt=np.array(run.save_dt),
        t_traj=np.asarray(run.t_traj),
        q_traj=np.asarray(run.q_traj),
        qd_traj=np.asarray(run.qd_traj),
        u_traj=np.asarray(run.u_traj),
        x_traj_full=np.asarray(run.x_traj_full),
        x_des_traj_full=np.asarray(run.x_des_traj_full),
        **{
            f"metrics_{name}": np.array(value, dtype=float)
            for name, value in metrics.items()
        },
    )
    return path


def load_run_npz(path: str | Path) -> PCSImpedanceRun:
    """Load rollout data from an ``.npz`` file."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        config = TaskSpaceTrajectoryConfig(
            tracking=_load_str(data, "tracking"),
            position_primitive=_load_str(data, "position_primitive"),
            orientation_primitive=_load_str(data, "orientation_primitive"),
            surface=_load_str(data, "surface"),
            period=float(data["period"]),
            ramp_time=float(data["ramp_time"]),
            position_amplitude_x=float(data["position_amplitude_x"]),
            position_amplitude_y=float(data["position_amplitude_y"]),
            position_amplitude_z=float(data["position_amplitude_z"]),
            line_amplitude=float(data["line_amplitude"]),
            circle_radius=float(data["circle_radius"]),
            helix_radius=float(data["helix_radius"]),
            helix_pitch=float(data["helix_pitch"]),
            orientation_amplitude=float(data["orientation_amplitude"]),
            surface_radius=float(data["surface_radius"]),
        )
        metrics = {
            key.removeprefix("metrics_"): float(data[key])
            for key in data.files
            if key.startswith("metrics_")
        }
        x_traj_full = jnp.asarray(data["x_traj_full"])
        x_des_traj_full = jnp.asarray(data["x_des_traj_full"])
        if not metrics:
            metrics = _scalar_operational_metrics(
                compute_operational_metrics(x_traj_full, x_des_traj_full)
            )
        return PCSImpedanceRun(
            trajectory_config=config,
            t1=float(data["t1"]),
            solver_dt=float(data["solver_dt"]),
            save_dt=float(data["save_dt"]),
            t_traj=jnp.asarray(data["t_traj"]),
            q_traj=jnp.asarray(data["q_traj"]),
            qd_traj=jnp.asarray(data["qd_traj"]),
            u_traj=jnp.asarray(data["u_traj"]),
            x_traj_full=x_traj_full,
            x_des_traj_full=x_des_traj_full,
            metrics=metrics,
        )


def plot_run(
    run: PCSImpedanceRun,
    problem: PCSImpedanceProblem,
    *,
    output_prefix: str,
    no_show: bool,
    force: bool = False,
) -> tuple[str, str, Array, Array | None]:
    """Plot tracking and configuration results for a saved rollout."""
    configure_matplotlib()
    tracking_output, tracking_error_pos, pose_error = _plot_tracking_results(
        run,
        problem,
        output_prefix=output_prefix,
        no_show=no_show,
        force=force,
    )
    config_output = _plot_configuration_results(
        run,
        problem,
        output_prefix=output_prefix,
        no_show=no_show,
        force=force,
    )
    return tracking_output, config_output, tracking_error_pos, pose_error


def print_tracking_summary(run: PCSImpedanceRun) -> None:
    """Print the axis-independent pose metrics reported in Section V.C."""
    metrics = run.metrics or _scalar_operational_metrics(
        compute_operational_metrics(run.x_traj_full, run.x_des_traj_full)
    )
    print("\nOperational-space tracking metrics:")
    print(f"  Position RMSE: {1e3 * metrics['position_rmse']:.6f} mm")
    print(
        f"  Position reference span: {1e3 * metrics['position_reference_span']:.6f} mm"
    )
    print(f"  Position NRMSE: {metrics['position_nrmse_percent']:.6f} %")
    print(f"  Orientation RMSE: {np.rad2deg(metrics['orientation_rmse']):.6f} deg")
    print(
        "  Orientation reference span: "
        f"{np.rad2deg(metrics['orientation_reference_span']):.6f} deg"
    )
    print(f"  Orientation NRMSE: {metrics['orientation_nrmse_percent']:.6f} %")


def render_run(
    run: PCSImpedanceRun,
    problem: PCSImpedanceProblem,
    *,
    viser_port: int,
    open_browser: bool,
    robot_opacity: float,
    video_output: str | None = None,
    record_every_n: int = 1,
    record_client_timeout: float = 10.0,
    record_frame_timeout: float = 10.0,
    camera_config: CameraConfig | None = None,
    blocking: bool = True,
) -> None:
    """Render a saved rollout with the example-local Viser science renderer."""
    if ViserRenderer is None:
        raise ImportError(
            "Viser is required to render this example. Install the Viser extras "
            "or omit --render."
        )

    render_operational_space_tracking(
        robot=problem.robot,
        osd=problem.osd,
        q0=problem.q0,
        trajectory_config=run.trajectory_config,
        t_traj=run.t_traj,
        q_traj=run.q_traj,
        x_traj_full=run.x_traj_full,
        x_des_traj_full=run.x_des_traj_full,
        viser_renderer_cls=ViserRenderer,
        port=viser_port,
        open_browser=open_browser,
        robot_opacity=robot_opacity,
        video_output=video_output,
        record_every_n=record_every_n,
        record_client_timeout=record_client_timeout,
        record_frame_timeout=record_frame_timeout,
        camera_config=camera_config,
        blocking=blocking,
    )


def operational_tracking_data(
    run: PCSImpedanceRun,
    problem: PCSImpedanceProblem,
) -> tuple[Array, Array, Array, Array | None]:
    """Return position trajectories, position error, and geometric pose error."""
    pos_indices = _position_indices(problem.osd)
    x_traj_pos = run.x_traj_full[:, pos_indices]
    x_des_traj_pos = run.x_des_traj_full[:, pos_indices]
    tracking_error_pos = x_des_traj_pos - x_traj_pos
    pose_error = None
    if run.trajectory_config.tracking == "pose":
        pose_error = jax.vmap(problem.osd.compute_pose_error)(
            run.x_traj_full,
            run.x_des_traj_full,
        )
    return x_traj_pos, x_des_traj_pos, tracking_error_pos, pose_error


def plot_operational_tracking_axes(
    run: PCSImpedanceRun,
    problem: PCSImpedanceProblem,
    axes: np.ndarray,
) -> tuple[Array, Array | None]:
    """Plot position/orientation tracking and errors into caller-owned axes."""
    x_pos, x_des_pos, position_error, pose_error = operational_tracking_data(
        run, problem
    )
    t = np.asarray(run.t_traj)
    coordinate_labels = ("x", "y", "z")[: x_pos.shape[1]]

    for component, label in enumerate(coordinate_labels):
        color = COORDINATE_COLORS[component]
        axes[0].plot(
            t,
            x_pos[:, component],
            color=color,
            linewidth=1.3,
            label=rf"$p_{label}$",
        )
        axes[0].plot(
            t,
            x_des_pos[:, component],
            color=color,
            linestyle="--",
            linewidth=2.0,
            label=rf"$p_{{{label},\mathrm{{des}}}}$",
            zorder=5,
        )
        axes[1].plot(
            t,
            np.asarray(position_error[:, component]) * 1000.0,
            color=color,
            linewidth=1.3,
            label=rf"$e_{{p_{label}}}$",
        )
    axes[0].set_ylabel(r"Position $\mathrm{[m]}$")
    axes[1].set_ylabel(r"Position error $\mathrm{[mm]}$")
    axes[1].axhline(0.0, color="0.25", linewidth=0.7, alpha=0.7, zorder=0)

    if run.trajectory_config.tracking == "pose":
        assert pose_error is not None
        orientation_dim = problem.osd.n_orientation_dim
        orientation_error = pose_error[:, : problem.osd.n_angular_velocity_dim]
        for component, label in enumerate(("x", "y", "z")[:orientation_dim]):
            color = COORDINATE_COLORS[component]
            axes[2].plot(
                t,
                run.x_traj_full[:, component],
                color=color,
                linewidth=1.3,
                label=rf"$r_{label}$",
            )
            axes[2].plot(
                t,
                run.x_des_traj_full[:, component],
                color=color,
                linestyle="--",
                linewidth=2.0,
                label=rf"$r_{{{label},\mathrm{{des}}}}$",
                zorder=5,
            )
            axes[3].plot(
                t,
                np.rad2deg(np.asarray(orientation_error[:, component])),
                color=color,
                linewidth=1.3,
                label=rf"$e_{{r_{label}}}$",
            )
        axes[2].set_ylabel(r"Orientation $\mathrm{[rad]}$")
        axes[3].set_ylabel(r"Orientation error $\mathrm{[deg]}$")
        axes[3].axhline(0.0, color="0.25", linewidth=0.7, alpha=0.7, zorder=0)

    for ax in axes:
        ax.set_xlim(float(t[0]), float(t[-1]))
    return position_error, pose_error


def tracking_legend_handles() -> tuple[list[Line2D], list[str]]:
    """Return decoupled coordinate and desired/actual legend handles."""
    handles: list[Line2D] = [
        Line2D([], [], color=color, marker="o", linestyle="none", markersize=4)
        for color in COORDINATE_COLORS
    ]
    labels = [r"$x$", r"$y$", r"$z$"]
    handles.extend(
        [
            Line2D([], [], color="black", linestyle="-", linewidth=1.3),
            Line2D([], [], color="black", linestyle="--", linewidth=2.0),
        ]
    )
    labels.extend(["Actual", "Desired"])
    return handles, labels


def _plot_tracking_results(
    run: PCSImpedanceRun,
    problem: PCSImpedanceProblem,
    *,
    output_prefix: str,
    no_show: bool,
    force: bool,
) -> tuple[str, Array, Array | None]:
    config = run.trajectory_config
    n_rows = 5 if config.tracking == "pose" else 3
    height_cm = 16.5 if config.tracking == "pose" else 10.5
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=cm_to_inches((16.5, height_cm)),
        sharex=True,
        constrained_layout=True,
    )
    tracking_axes = axes[:4] if config.tracking == "pose" else axes[:2]
    tracking_error_pos, pose_error = plot_operational_tracking_axes(
        run, problem, tracking_axes
    )

    ax = axes[-1]
    num_actuators_to_plot = min(problem.robot.num_actuators, 6)
    for i in range(num_actuators_to_plot):
        ax.plot(
            run.t_traj,
            run.u_traj[:, i],
            label=f"$u_{i}$",
            linewidth=1.1,
        )
    ax.set_xlabel(r"Time $\mathrm{[s]}$")
    ax.set_ylabel("Generalized actuation")
    ax.legend(loc="upper right", ncol=3)
    handles, labels = tracking_legend_handles()
    fig.legend(handles, labels, loc="outside upper center", ncol=5)
    output = _output_path(
        output_prefix,
        f"{config.tracking}_{config.position_primitive}_{config.orientation_primitive}_results",
    )
    _ensure_writable(output, force=force)
    fig.savefig(output, bbox_inches=None)
    if no_show:
        plt.close(fig)
    else:
        plt.show()

    return output, tracking_error_pos, pose_error


def _plot_configuration_results(
    run: PCSImpedanceRun,
    problem: PCSImpedanceProblem,
    *,
    output_prefix: str,
    no_show: bool,
    force: bool,
) -> str:
    config = run.trajectory_config
    fig, axes = plt.subplots(
        2,
        1,
        figsize=cm_to_inches((16.5, 7.5)),
        sharex=True,
        constrained_layout=True,
    )
    num_dofs_to_plot = min(problem.num_dofs, 6)

    ax = axes[0]
    for i in range(num_dofs_to_plot):
        ax.plot(run.t_traj, run.q_traj[:, i], label=f"$q_{i}$", linewidth=1.2)
    ax.set_ylabel("Generalized strain")
    ax.legend(loc="upper right", ncol=3)

    ax = axes[1]
    for i in range(num_dofs_to_plot):
        ax.plot(
            run.t_traj,
            run.qd_traj[:, i],
            label=f"$\\dot{{q}}_{i}$",
            linewidth=1.2,
        )
    ax.set_xlabel(r"Time $\mathrm{[s]}$")
    ax.set_ylabel("Generalized strain rate")
    ax.legend(loc="upper right", ncol=3)
    output = _output_path(
        output_prefix,
        f"{config.tracking}_{config.position_primitive}_{config.orientation_primitive}_config",
    )
    _ensure_writable(output, force=force)
    fig.savefig(output, bbox_inches=None)
    if no_show:
        plt.close(fig)
    else:
        plt.show()
    return output


def _position_indices(osd: OperationalSpaceDynamics) -> Array:
    pos_start = osd.n_orientation_dim
    pos_dim = 2 if osd.is_planar else 3
    return jnp.arange(pos_start, pos_start + pos_dim)


def _output_path(prefix: str, suffix: str) -> str:
    path = Path(prefix)
    suffix = suffix.replace("_", "-")
    return str(path.with_name(f"{path.name}_{suffix}.pdf"))


def _ensure_writable(path: str | Path, *, force: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output}. Pass --force to replace it."
        )


def _load_str(data: np.lib.npyio.NpzFile, key: str) -> str:
    return str(data[key].item())


__all__ = [
    "DEFAULT_SAVE_DT",
    "DEFAULT_SOLVER_DT",
    "DEFAULT_TRAJECTORY_OUTPUT",
    "OUTPUTS_DIR",
    "PCSImpedanceProblem",
    "PCSImpedanceRun",
    "build_problem",
    "configure_matplotlib",
    "cm_to_inches",
    "load_run_npz",
    "operational_tracking_data",
    "plot_operational_tracking_axes",
    "plot_run",
    "print_problem_summary",
    "print_tracking_summary",
    "render_run",
    "run_impedance_simulation",
    "save_run_npz",
    "tracking_legend_handles",
]
