"""Batched Open3D rendering demo for tendon-actuated PCS robots.

This example vmaps `robot.rollout_to` over randomly sampled, constant tendon
tensions to generate multiple trajectories from the same initial condition,
and visualizes them in a grid using Open3DRenderer.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from soromox.rendering import MatplotlibRenderer, Open3DRenderer
from soromox.systems import SystemState, TendonActuatedPCS


def build_robot() -> TendonActuatedPCS:
    """Construct a simple tendon-actuated PCS robot."""
    num_segments = 2
    rho = 1070 * jnp.ones((num_segments,))
    params = {
        "p0": jnp.array([jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, 0.0]),
        "L": 1e-1 * jnp.ones((num_segments,)),
        "r": 2e-2 * jnp.ones((num_segments,)),
        "rho": rho,
        "g": jnp.array([0.0, 0.0, 9.81]),
        "E": 2e3 * jnp.ones((num_segments,)),
        "G": 1e3 * jnp.ones((num_segments,)),
    }
    params["D"] = 1e-3 * jnp.diag(
        (
            jnp.repeat(
                jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), num_segments, axis=0
            )
            * params["L"][:, None]
        ).flatten()
    )
    active_tendon_routing_params = {
        "ry": jnp.array([1e-2, -1e-2, 5e-3, -5e-3]),
        "rz": jnp.zeros((4,)),
        "my": jnp.array([0.0, 0.0, 0.0, 0.0]),
        "mz": jnp.array([0.0, 0.0, 0.0, 0.0]),
        "idx_seg_att": jnp.array([0, 0, 1, 1]),
    }

    return TendonActuatedPCS(
        num_segments=num_segments,
        params=params,
        active_tendon_routing_params=active_tendon_routing_params,
    )


def simulate_batched(robot: TendonActuatedPCS, num_robots: int, rng_key: jax.Array):
    """Roll out multiple trajectories with vmap and random constant tendon tensions."""
    q0 = jnp.repeat(
        jnp.array([0.0, 0.0, 5.0 * jnp.pi, 0.1, 0.2, 0.0])[None, :],
        robot.num_segments,
        axis=0,
    ).flatten()
    qd0 = jnp.zeros_like(q0)

    initial_state = SystemState(t=jnp.array(0.0), y=jnp.concatenate([q0, qd0]))

    # Sample constant tendon tensions per robot in [-1, 0]
    u_batch = jax.random.uniform(
        rng_key,
        (num_robots, robot.num_actuators),
        minval=-1.0,
        maxval=0.0,
    )
    print("u_batch:\n", u_batch)

    def rollout_single(u_const):
        return robot.rollout_to(
            initial_state=initial_state,
            u=u_const,
            t1=1.5,
            solver_dt=1e-4,
            save_dt=0.02,
            max_steps=None,
        )

    trajectories = jax.vmap(rollout_single)(u_batch)

    # All trajectories share the same time grid
    ts = trajectories.t[0]
    q_ts, _ = jnp.split(trajectories.y, 2, axis=2)  # (N, T, DOF)
    return ts, q_ts


def main():
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
        robot_colors="plasma",
        tendon_color=(0.9, 0.1, 0.1),
        tendon_line_width=2.0,
    )
    # q_ts_batched is (N, T, DOF); animate in slider mode for quick inspection
    matplotlib_renderer.animate(ts=ts, q_ts=q_ts_batched, mode="slider", show=True)

    # render using the Open3DRenderer
    renderer = Open3DRenderer(
        robot,
        grid_spacing=grid_spacing,
        robot_colors="viridis",
    )
    renderer.render_sequence(
        ts=ts, q_ts=q_ts_batched, record_path="videos/batched_tendon_actuated_pcs.mp4"
    )


if __name__ == "__main__":
    main()
