"""Batched rendering demo for tendon-actuated PCS robots.

This example vmaps `robot.rollout_to` over randomly sampled, constant tendon
tensions to generate multiple trajectories from the same initial condition,
and visualizes them in a grid using Matplotlib, Open3D, and Viser renderers.
"""

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.rendering import (
    ActuatorStyleConfig,
    BackboneColorConfig,
    MatplotlibRenderer,
    Open3DRenderer,
    RendererColorConfig,
    ViserRenderer,
)
from soromox.systems import (
    PCS,
    LinkSpec,
    PCSStructure,
    SystemState,
)

DEFAULT_OPEN3D_VIDEO_PATH = (
    Path(__file__).resolve().parent
    / "videos"
    / "simulate_batched_tendon_actuated_pcs_open3d.mp4"
)
DEFAULT_VISER_VIDEO_PATH = (
    Path(__file__).resolve().parent
    / "videos"
    / "simulate_batched_tendon_actuated_pcs_viser.mp4"
)


def build_robot() -> PCS:
    """Construct a simple tendon-actuated PCS robot."""
    num_segments = 2
    rho = 1070 * jnp.ones((num_segments,))
    segment_lengths = 1e-1 * jnp.ones((num_segments,))
    material_damping_coefficient = 362.0
    body_params = PCS.params_from_links(
        [
            LinkSpec.circular(
                length=float(segment_lengths[index]),
                radius=2e-2,
                density=float(rho[index]),
                young_modulus=2e3,
                shear_modulus=1e3,
                material_damping_coefficient=material_damping_coefficient,
                reference_strain=[0, 0, 0, 1, 0, 0],
            )
            for index in range(num_segments)
        ],
        base_pose=jnp.array([0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 0.0]),
        gravity=jnp.array([0.0, 0.0, 9.81]),
    )
    active_tendon_routing = ThreadlikeRouting.linear(
        intercept=jnp.array(
            [[0.0, 2e-2, 0.0], [0.0, -2e-2, 0.0], [0.0, 1e-2, 0.0], [0.0, -1e-2, 0.0]]
        ),
        start_segment_index=0,
        end_segment_index=(0, 0, 1, 1),
    )

    return PCS(
        params=body_params,
        structure=PCSStructure(),
        actuators=ThreadlikeActuator.tendons(active_tendon_routing),
    )


def simulate_batched(robot: PCS, num_robots: int, rng_key: jax.Array):
    """Roll out multiple trajectories with vmap and random constant tendon tensions."""
    q0 = jnp.zeros((num_robots, robot.num_coordinates))
    # # sample different configurations per robot in the batch
    # q0 = q0 + jax.random.uniform(
    #     rng_key, (num_robots, robot.num_coordinates), minval=-0.2, maxval=0.2
    # )
    print("q0 batch:\n", q0)

    qd0 = jnp.zeros_like(q0)
    initial_states = SystemState(
        t=jnp.zeros((num_robots,)), y=jnp.concatenate([q0, qd0], axis=1)
    )

    # Sample positive tendon tensions per robot.
    u_batch = jax.random.uniform(
        rng_key,
        (num_robots, robot.num_actuators),
        minval=0.0,
        maxval=1.0,
    )
    print("u_batch:\n", u_batch)

    # specify save time steps
    solver_dt = 1e-4
    save_ts = robot._compute_save_times(0.0, 1.5, solver_dt=1e-4, save_dt=0.02)

    def rollout_single(initial_state, u_const):
        return robot.rollout_to(
            initial_state=initial_state,
            u=u_const,
            solver_dt=solver_dt,
            save_ts=save_ts,
            max_steps=None,
        )

    trajectories = jax.vmap(rollout_single)(initial_states, u_batch)

    # All trajectories share the same time grid
    ts = trajectories.t[0]
    q_ts, _ = jnp.split(trajectories.y, 2, axis=2)  # (N, T, DOF)
    return ts, q_ts


def main(*, open3d_video_output: Path, viser_video_output: Path):
    if Open3DRenderer is None:
        raise ImportError(
            "Open3D is required for this example. Install with `pip install open3d`."
        )

    robot = build_robot()
    ts, q_ts_batched = simulate_batched(
        robot, num_robots=9, rng_key=jax.random.PRNGKey(0)
    )

    grid_spacing = (0.3, 0.3)

    # animate using the MatplotlibRenderer
    matplotlib_renderer = MatplotlibRenderer(
        robot,
        num_points=60,
        line_width=2.5,
        grid_spacing=grid_spacing,
        actuator_line_width=2.0,
        color_config=RendererColorConfig(
            backbone=BackboneColorConfig(robot_palette="plasma"),
            actuators=ActuatorStyleConfig(default_color=(0.9, 0.1, 0.1)),
        ),
    )
    # q_ts_batched is (N, T, DOF); animate in slider mode for quick inspection
    matplotlib_renderer.animate(ts=ts, q_ts=q_ts_batched, mode="slider", show=True)

    # render using the Open3DRenderer
    renderer = Open3DRenderer(
        robot,
        grid_spacing=grid_spacing,
        color_config=RendererColorConfig(
            backbone=BackboneColorConfig(robot_palette="viridis")
        ),
    )
    renderer.render_sequence(
        ts=ts,
        q_ts=q_ts_batched,
        record_path=str(open3d_video_output),
    )

    # render using the ViserRenderer
    if ViserRenderer is None:
        print("ViserRenderer is unavailable. Install with `pip install viser`.")
        return

    viser_renderer = ViserRenderer(
        robot,
        grid_spacing=grid_spacing,
        color_config=RendererColorConfig(
            backbone=BackboneColorConfig(robot_palette="magma")
        ),
    )
    viser_renderer.render_sequence(
        ts=ts,
        q_ts=q_ts_batched,
        playback_speed=1.0,
        autoplay=True,
        loop=True,
        render_actuators=True,
        record_path=str(viser_video_output),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--open3d-video-output", type=Path, default=DEFAULT_OPEN3D_VIDEO_PATH
    )
    parser.add_argument(
        "--viser-video-output", type=Path, default=DEFAULT_VISER_VIDEO_PATH
    )
    args = parser.parse_args()
    main(
        open3d_video_output=args.open3d_video_output,
        viser_video_output=args.viser_video_output,
    )
