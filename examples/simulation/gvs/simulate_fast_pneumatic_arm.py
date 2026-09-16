"""Simulate the fast circular motion of a hanging pneumatic soft arm.

The geometry and motion envelope are based on robot #1 in:

    Haggerty, D. A. et al. (2023). Control of soft robots with inertial
    dynamics. Science Robotics, 8(81), eadd6864.
    https://doi.org/10.1126/scirobotics.add6864

The paper reports a 45 cm arm with four longitudinal pneumatic muscles, a
6.25 cm maximum diameter, and approximately 110 degrees of curvature. Its
tracking tests span 0.1--1.1 Hz. Tracking the tip in the attached reference
clip shows a sweep from approximately 0.66 to 1.07 Hz. The commanded frequency
range and eased ramp below account for the simulated arm's dynamic phase lag.

This is a physics-based GVS analogue, not a parameter identification of the
experimental robot. The effective stiffness, damping, density, muscle routing,
and force waveform are chosen to reproduce the visible motion envelope. The
two orthogonal bending strains are driven in quadrature, rotating the arm's
bending direction so its tip circles the mounting axis in a plane parallel to
the base. The script simulates two warm-up cycles before returning the requested
sequence so the rendered motion starts in its periodic regime.

Run the interactive Viser studio scene from the repository root:

    python examples/simulation/gvs/simulate_fast_pneumatic_arm.py

Record a browser-rendered MP4 after opening the printed Viser URL:

    python examples/simulation/gvs/simulate_fast_pneumatic_arm.py --record-viser

Run only the simulation and print its motion statistics:

    python examples/simulation/gvs/simulate_fast_pneumatic_arm.py --no-viser
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.rendering import (
    ActuatorStyleConfig,
    BackboneColorConfig,
    CameraConfig,
    GeometryConfig,
    RendererColorConfig,
    RendererConfig,
    RenderOutputConfig,
    SceneConfig,
    ViserRenderer,
)
from soromox.systems import (
    GVS,
    GVSSegment,
    JointSpec,
    LinkSpec,
    StrainBasisSpec,
    SystemState,
)
from soromox.utils.geometry.poses import spatial_mounting_pose

jax.config.update("jax_enable_x64", True)

ROBOT_LENGTH = 0.45
BACKBONE_RADIUS = 0.018
MUSCLE_ROUTE_RADIUS = 0.0245
MUSCLE_RENDER_RADIUS = 0.00675
MAXIMUM_DIAMETER = 2 * (MUSCLE_ROUTE_RADIUS + MUSCLE_RENDER_RADIUS)

DEFAULT_START_FREQUENCY = 0.65
DEFAULT_END_FREQUENCY = 1.12
DEFAULT_FREQUENCY_RAMP_POWER = 1.5
DEFAULT_DURATION = 6.9
DEFAULT_FRAME_RATE = 30.0
DEFAULT_RENDER_SIZE = 1080
DEFAULT_PRELOAD = 7.5
DEFAULT_AMPLITUDE = 6.25
DEFAULT_AMPLITUDE_TAPER = 0.06
DEFAULT_WARMUP_CYCLES = 2.0

TRACKER_POSITIONS = jnp.asarray([0.105, 0.215, 0.325, 0.44])
TRACKER_SURFACE_OFFSET = jnp.asarray(
    [0.0, -(MUSCLE_ROUTE_RADIUS + MUSCLE_RENDER_RADIUS + 0.001), 0.0]
)
DEFAULT_VIDEO_OUTPUT = (
    Path(__file__).resolve().parent / "videos" / "simulate_fast_pneumatic_arm_viser.mp4"
)


def make_robot() -> GVS:
    """Construct the four-muscle hanging GVS arm.

    The two active generalized coordinates are constant bending strains about
    the local y and z axes. Rotational bases are scaled by length, so each
    coordinate is a total bend contribution in radians.

    Returns:
        A 45 cm fixed-base GVS arm with four routed muscle actuators.
    """
    link = LinkSpec.circular(
        length=ROBOT_LENGTH,
        radius=BACKBONE_RADIUS,
        density=650.0,
        young_modulus=4.5e5,
        poisson_ratio=0.45,
        material_damping_coefficient=2.5e4,
        reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )
    basis = StrainBasisSpec(
        type="monomial",
        strain_selector=("kappa_y", "kappa_z"),
        basis_order=0,
    )

    muscle_angles = jnp.arange(4) * (0.5 * jnp.pi)
    offsets = MUSCLE_ROUTE_RADIUS * jnp.stack(
        [
            jnp.zeros_like(muscle_angles),
            jnp.cos(muscle_angles),
            jnp.sin(muscle_angles),
        ],
        axis=-1,
    )
    routing = ThreadlikeRouting.linear(
        intercept=offsets,
        slope=jnp.zeros_like(offsets),
        start_segment_index=0,
        end_segment_index=0,
    )
    muscles = ThreadlikeActuator.muscles(
        routing,
        labels=("local +y", "local +z", "local -y", "local -z"),
    )
    return GVS.from_segments(
        [
            GVSSegment(
                link=link,
                joint=JointSpec(type="fixed"),
                basis=basis,
                num_gauss_points=9,
            )
        ],
        gravity=jnp.asarray([0.0, 0.0, -9.81]),
        base_pose=spatial_mounting_pose("hanging"),
        actuators=muscles,
        scale_rotational_basis_by_length=True,
    )


def muscle_tensions(
    t: Array,
    *,
    start_frequency: float = DEFAULT_START_FREQUENCY,
    end_frequency: float = DEFAULT_END_FREQUENCY,
    frequency_ramp_power: float = DEFAULT_FREQUENCY_RAMP_POWER,
    sweep_duration: float = DEFAULT_DURATION,
    preload: float = DEFAULT_PRELOAD,
    amplitude: float = DEFAULT_AMPLITUDE,
    amplitude_taper: float = DEFAULT_AMPLITUDE_TAPER,
) -> Array:
    """Return four nonnegative muscle tensions for one circular sweep.

    Both antagonistic muscle pairs are active. Their sinusoidal tension
    differences are 90 degrees out of phase, which rotates the bending direction
    around the undeformed arm axis.

    Args:
        t: Simulation time in seconds.
        start_frequency: Initial sweep frequency in hertz.
        end_frequency: Final sweep frequency in hertz.
        frequency_ramp_power: Exponent controlling when the frequency increase
            occurs. Values above one emphasize acceleration later in the sweep.
        sweep_duration: Duration of the frequency sweep in seconds.
        preload: Shared muscle pretension in newtons.
        amplitude: Initial sinusoidal tension amplitude in newtons.
        amplitude_taper: Fractional amplitude reduction across the visible sweep.

    Returns:
        Muscle tensions ordered ``[+y, +z, -y, -z]`` in newtons.
    """
    visible_time = jnp.clip(t, 0.0, sweep_duration)
    progress = visible_time / sweep_duration
    phase_cycles = start_frequency * t + (
        (end_frequency - start_frequency)
        * sweep_duration
        * progress ** (frequency_ramp_power + 1.0)
        / (frequency_ramp_power + 1.0)
    )
    phase = 2.0 * jnp.pi * phase_cycles
    tapered_amplitude = amplitude * (
        1.0 - amplitude_taper * visible_time / sweep_duration
    )
    sweep_y = tapered_amplitude * jnp.sin(phase)
    sweep_z = tapered_amplitude * jnp.cos(phase)
    return jnp.asarray(
        [
            preload + sweep_y,
            preload + sweep_z,
            preload - sweep_y,
            preload - sweep_z,
        ],
        dtype=jnp.float64,
    )


def simulate_motion(
    robot: GVS,
    *,
    duration: float = DEFAULT_DURATION,
    start_frequency: float = DEFAULT_START_FREQUENCY,
    end_frequency: float = DEFAULT_END_FREQUENCY,
    frequency_ramp_power: float = DEFAULT_FREQUENCY_RAMP_POWER,
    frame_rate: float = DEFAULT_FRAME_RATE,
    solver_dt: float = 1e-3,
    preload: float = DEFAULT_PRELOAD,
    amplitude: float = DEFAULT_AMPLITUDE,
    amplitude_taper: float = DEFAULT_AMPLITUDE_TAPER,
    warmup_cycles: float = DEFAULT_WARMUP_CYCLES,
) -> SystemState:
    """Simulate the periodic muscle command and return the visible sequence.

    Args:
        robot: Arm returned by :func:`make_robot`.
        duration: Visible sequence duration in seconds.
        start_frequency: Initial sweep frequency in hertz.
        end_frequency: Final sweep frequency in hertz.
        frequency_ramp_power: Exponent controlling when the frequency increase
            occurs.
        frame_rate: Saved trajectory rate in frames per second.
        solver_dt: Maximum fixed integration step in seconds.
        preload: Shared muscle pretension in newtons.
        amplitude: Initial sinusoidal tension amplitude in newtons.
        amplitude_taper: Fractional amplitude reduction across the visible sweep.
        warmup_cycles: Periods simulated before the first returned frame.

    Returns:
        Simulated state with timestamps beginning at zero.

    Raises:
        ValueError: If a duration, rate, step, tension, or warm-up is invalid.
    """
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if start_frequency <= 0.0 or end_frequency <= 0.0:
        raise ValueError("frequencies must be positive")
    if frequency_ramp_power <= 0.0:
        raise ValueError("frequency_ramp_power must be positive")
    if frame_rate <= 0.0:
        raise ValueError("frame_rate must be positive")
    if solver_dt <= 0.0:
        raise ValueError("solver_dt must be positive")
    if preload < 0.0 or amplitude < 0.0 or amplitude > preload:
        raise ValueError("tensions must satisfy 0 <= amplitude <= preload")
    if not 0.0 <= amplitude_taper < 1.0:
        raise ValueError("amplitude_taper must satisfy 0 <= amplitude_taper < 1")
    if warmup_cycles < 0.0:
        raise ValueError("warmup_cycles must be nonnegative")

    frame_count = int(np.floor(duration * frame_rate)) + 1
    save_ts = jnp.arange(frame_count, dtype=jnp.float64) / frame_rate
    warmup_duration = warmup_cycles / start_frequency
    initial_state = SystemState(
        t=-warmup_duration,
        y=jnp.zeros((robot.state_size,), dtype=jnp.float64),
    )

    def controller(state: SystemState) -> tuple[Array, None]:
        return (
            muscle_tensions(
                state.t,
                start_frequency=start_frequency,
                end_frequency=end_frequency,
                frequency_ramp_power=frequency_ramp_power,
                sweep_duration=duration,
                preload=preload,
                amplitude=amplitude,
                amplitude_taper=amplitude_taper,
            ),
            None,
        )

    return robot.rollout_closed_loop_to(
        initial_state=initial_state,
        controller=controller,
        t1=duration,
        solver_dt=solver_dt,
        save_ts=save_ts,
        max_steps=None,
    )


def configurations(robot: GVS, trajectory: SystemState) -> Array:
    """Extract generalized configurations from a simulated state sequence."""
    q_ts, _, _ = robot.split_state(trajectory.y)
    return q_ts


def points_along_arm(robot: GVS, q_ts: Array, positions: Array) -> Array:
    """Evaluate world positions along the arm for every saved frame.

    Args:
        robot: Simulated arm.
        q_ts: Configuration trajectory with shape ``(T, num_coordinates)``.
        positions: Arc-length coordinates with shape ``(N,)`` in metres.

    Returns:
        World positions with shape ``(N, T, 3)`` for Viser dynamic markers.
    """
    positions = jnp.asarray(positions)

    def frame_positions(q: Array) -> Array:
        return jax.vmap(lambda s: robot.forward_kinematics(q, s)[:3, 3])(positions)

    return jax.vmap(frame_positions)(q_ts).transpose(1, 0, 2)


def motion_statistics(robot: GVS, trajectory: SystemState) -> dict[str, float]:
    """Compute bend, orbit radius, tip speed, and tip acceleration maxima."""
    q_ts = configurations(robot, trajectory)
    tip_positions = points_along_arm(robot, q_ts, jnp.asarray([robot.length]))[0]
    base_rotation = jnp.asarray(robot.base_transform)[:3, :3]

    def tip_bend_angle(q: Array) -> Array:
        tip_rotation = robot.forward_kinematics(q, robot.length)[:3, :3]
        relative_rotation = base_rotation.T @ tip_rotation
        cosine = jnp.clip((jnp.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
        return jnp.arccos(cosine)

    bend_angles = np.asarray(jax.vmap(tip_bend_angle)(q_ts))
    times = np.asarray(trajectory.t)
    tips = np.asarray(tip_positions)
    edge_order = 2 if len(times) > 2 else 1
    tip_velocity = np.gradient(tips, times, axis=0, edge_order=edge_order)
    tip_acceleration = np.gradient(tip_velocity, times, axis=0, edge_order=edge_order)
    base_transform = np.asarray(robot.base_transform)
    base_frame_tips = (tips - base_transform[:3, 3]) @ base_transform[:3, :3]
    orbit_radii = np.linalg.norm(base_frame_tips[:, 1:3], axis=1)
    return {
        "maximum_bend_deg": float(np.rad2deg(np.max(bend_angles))),
        "maximum_orbit_radius_m": float(np.max(orbit_radii)),
        "orbit_plane_peak_to_peak_m": float(np.ptp(base_frame_tips[:, 0])),
        "maximum_tip_speed_m_s": float(np.max(np.linalg.norm(tip_velocity, axis=1))),
        "maximum_tip_acceleration_m_s2": float(
            np.max(np.linalg.norm(tip_acceleration, axis=1))
        ),
    }


def make_renderer_config(
    style: str = "neutral",
    *,
    width: int = DEFAULT_RENDER_SIZE,
    height: int = DEFAULT_RENDER_SIZE,
) -> RendererConfig:
    """Create the hanging studio appearance used by the Viser sequence."""
    scene = SceneConfig.studio(style=style, scene_extent=0.8)
    scene.ground.normal = (0.0, 0.0, -1.0)
    scene.ground.height_reference = "base_mounting_face"
    return RendererConfig(
        scene=scene,
        camera=CameraConfig(
            fov=32.0,
            position=(0.0, -1.45, -0.22),
            look_at=(0.0, 0.0, -0.22),
            up=(0.0, 0.0, 1.0),
            exposure_ev100=14.3,
        ),
        colors=RendererColorConfig(
            backbone=BackboneColorConfig(
                segment_palette=None,
                segment_colors=[(0.055, 0.075, 0.13)],
            ),
            base_plate_color=(0.12, 0.13, 0.15),
            actuators=ActuatorStyleConfig(
                default_color=(0.09, 0.12, 0.22),
                kind_colors={"muscle": (0.08, 0.11, 0.22)},
                kind_radii={"muscle": MUSCLE_RENDER_RADIUS},
            ),
        ),
        geometry=GeometryConfig(
            num_points=81,
            cross_section_resolution=24,
            base_plate_radius_scale=1.8,
            base_plate_thickness=0.018,
        ),
        output=RenderOutputConfig(width=width, height=height),
    )


def render_motion(
    robot: GVS,
    trajectory: SystemState,
    *,
    port: int = 8085,
    style: str = "neutral",
    record_path: Path | None = None,
    snapshot_path: Path | None = None,
    open_browser: bool = True,
    width: int = DEFAULT_RENDER_SIZE,
    height: int = DEFAULT_RENDER_SIZE,
) -> None:
    """Render the simulated arm, muscles, and tracker lights in Viser."""
    if ViserRenderer is None:
        raise ImportError("Install viser to render this example")
    q_ts = configurations(robot, trajectory)
    tracker_trajectories = (
        points_along_arm(robot, q_ts, TRACKER_POSITIONS)
        + TRACKER_SURFACE_OFFSET[None, None, :]
    )
    snapshot_paths = None
    if snapshot_path is not None:
        snapshot_paths = {len(trajectory.t) // 8: snapshot_path}
    renderer = ViserRenderer(
        robot,
        config=make_renderer_config(style, width=width, height=height),
        host="127.0.0.1",
        port=port,
        open_browser=open_browser,
    )
    renderer.render_sequence(
        trajectory.t,
        q_ts,
        playback_speed=1.0,
        autoplay=True,
        loop=record_path is None,
        record_path=None if record_path is None else str(record_path),
        snapshot_paths=snapshot_paths,
        stop_when_recording_done=record_path is not None,
        record_client_timeout=300.0,
        record_frame_timeout=30.0,
        dynamic_spheres_positions=tracker_trajectories,
        dynamic_spheres_radii=np.full((len(TRACKER_POSITIONS),), 0.006),
        dynamic_spheres_colors=np.tile((1.0, 0.04, 0.025), (len(TRACKER_POSITIONS), 1)),
        plot_configurations=True,
        plot_actuator_positions=True,
        robot_name="Fast pneumatic soft arm",
    )


def parse_args() -> argparse.Namespace:
    """Parse simulation and Viser rendering options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument(
        "--start-frequency", type=float, default=DEFAULT_START_FREQUENCY
    )
    parser.add_argument("--end-frequency", type=float, default=DEFAULT_END_FREQUENCY)
    parser.add_argument(
        "--frequency-ramp-power", type=float, default=DEFAULT_FREQUENCY_RAMP_POWER
    )
    parser.add_argument("--frame-rate", type=float, default=DEFAULT_FRAME_RATE)
    parser.add_argument("--solver-dt", type=float, default=1e-3)
    parser.add_argument("--preload", type=float, default=DEFAULT_PRELOAD)
    parser.add_argument("--amplitude", type=float, default=DEFAULT_AMPLITUDE)
    parser.add_argument(
        "--amplitude-taper", type=float, default=DEFAULT_AMPLITUDE_TAPER
    )
    parser.add_argument("--warmup-cycles", type=float, default=DEFAULT_WARMUP_CYCLES)
    parser.add_argument("--no-viser", action="store_true")
    parser.add_argument("--record-viser", action="store_true")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--viser-port", type=int, default=8085)
    parser.add_argument(
        "--studio-style", choices=("neutral", "bright", "dark"), default="neutral"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_VIDEO_OUTPUT)
    parser.add_argument("--width", type=int, default=DEFAULT_RENDER_SIZE)
    parser.add_argument("--height", type=int, default=DEFAULT_RENDER_SIZE)
    return parser.parse_args()


def main() -> None:
    """Construct, simulate, summarize, and optionally render the arm."""
    args = parse_args()
    robot = make_robot()
    print(
        f"Arm: length={robot.length:.3f} m, maximum diameter={MAXIMUM_DIAMETER:.4f} m, "
        f"DOFs={robot.num_internal_dofs}, muscles={robot.num_actuators}"
    )
    trajectory = simulate_motion(
        robot,
        duration=args.duration,
        start_frequency=args.start_frequency,
        end_frequency=args.end_frequency,
        frequency_ramp_power=args.frequency_ramp_power,
        frame_rate=args.frame_rate,
        solver_dt=args.solver_dt,
        preload=args.preload,
        amplitude=args.amplitude,
        amplitude_taper=args.amplitude_taper,
        warmup_cycles=args.warmup_cycles,
    )
    print(f"Simulation complete: {len(trajectory.t)} frames")
    for name, value in motion_statistics(robot, trajectory).items():
        print(f"  {name}: {value:.3f}")

    if args.no_viser:
        return
    record_path = args.output if args.record_viser else None
    snapshot_path = args.output.with_suffix(".png") if args.snapshot else None
    if record_path is not None:
        record_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path is not None:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    render_motion(
        robot,
        trajectory,
        port=args.viser_port,
        style=args.studio_style,
        record_path=record_path,
        snapshot_path=snapshot_path,
        open_browser=True,
        width=args.width,
        height=args.height,
    )
    if record_path is not None:
        print(f"Saved {record_path}")
    if snapshot_path is not None:
        print(f"Saved {snapshot_path}")


if __name__ == "__main__":
    main()
