from functools import partial
from pathlib import Path

import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp

from soromox.rendering import OpenCVPlanarRenderer, ViserRenderer
from soromox.systems import Pendulum, SystemState

jax.config.update("jax_enable_x64", True)  # double precision

num_links = 2
params = {
    "m": jnp.array([10.0, 6.0]),
    "I": jnp.array([3.0, 2.0]),
    "L": jnp.array([2.0, 1.0]),
    "Lc": jnp.array([1.0, 0.5]),
    "g": jnp.array([0.0, -9.81]),
}

# define initial configuration
q0 = jnp.zeros((num_links,))
q0 = jnp.array([jnp.pi / 8, -jnp.pi / 4])

# set simulation parameters
solver_dt = 1e-4  # time step
t0 = jnp.array(0.0)
t1 = jnp.array(5.0)
save_dt = jnp.array(0.01)

# video settings
video_width, video_height = 700, 700  # img height and width
video_path = Path("videos") / f"pendulum_nl-{num_links}.mp4"


if __name__ == "__main__":
    # Instantiate the pendulum model directly
    robot = Pendulum(params)

    # initialize velocities and actuation
    qd0 = jnp.zeros_like(q0)  # initial velocities for simulation
    u = jnp.zeros_like(q0)  # torques (actuation)

    # call the forward dynamics
    yd = robot.forward_dynamics(t0, jnp.concatenate([q0, qd0]), (u,))
    print("yd0:\n", yd)

    # Integrate using the model's built-in solver
    initial_state = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))
    trajectory = robot.rollout_to(
        initial_state=initial_state,
        u=u,
        t1=t1,
        solver_dt=solver_dt,
        save_dt=save_dt,
    )
    ts = trajectory.t
    q_ts, qd_ts = jnp.split(trajectory.y, 2, axis=1)
    video_ts = ts
    print("Final configuration:\n", q_ts[-1])

    # =====================================================
    # End-effector position upon time
    # =====================================================
    chi_ee_ts = jax.vmap(
        robot.forward_kinematics_tips,
    )(q_ts)[:, -1, :]

    plt.figure()
    for link_idx in range(num_links):
        plt.plot(
            ts,
            q_ts[:, link_idx] / jnp.pi * 180,
            label=r"$q_{" + str(link_idx + 1) + "}$ [deg]",
        )
    plt.xlabel("Time [s]")
    plt.ylabel("Configuration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # plot end-effector position vs time
    plt.figure()
    plt.plot(ts, chi_ee_ts[:, 1], label="End-effector x [m]")
    plt.plot(ts, chi_ee_ts[:, 2], label="End-effector y [m]")
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector position [m]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()

    # end effector orientation vs. time
    plt.figure()
    plt.plot(
        ts,
        chi_ee_ts[:, 0] / jnp.pi * 180,
        label=r"End-effector Orientation $\theta$ [deg]",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector Orientation [deg]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()

    # plot the end-effector position in the x-y plane as a scatter plot with the time as the color
    plt.figure()
    plt.scatter(chi_ee_ts[:, 1], chi_ee_ts[:, 2], c=ts, cmap="viridis")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("End-effector x [m]")
    plt.ylabel("End-effector y [m]")
    plt.colorbar(label="Time [s]")
    plt.tight_layout()
    plt.show()

    # =====================================================
    # Energy computation upon time
    # =====================================================
    U_ts = jax.vmap(jax.jit(partial(robot.potential_energy)))(q_ts)
    T_ts = jax.vmap(jax.jit(partial(robot.kinetic_energy)))(q_ts, qd_ts)

    plt.figure()
    plt.plot(ts, U_ts, label="Potential Energy")
    plt.plot(ts, T_ts, label="Kinetic Energy")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.legend()
    plt.title("Energy over Time")
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()

    # create video
    video_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = OpenCVPlanarRenderer(
        robot,
        width=video_width,
        height=video_height,
        backbone_color=(0, 0, 0),
        length_scale=2.5,
    )
    renderer.render_sequence(video_ts, q_ts, record_path=str(video_path))
    print(f"Video saved to {video_path}")

    # =====================================================
    # Viser web-based visualization with plotly plots
    # =====================================================
    # Note: ViserRenderer is designed for 3D soft robots, so 3D visualization
    # may not work well for this planar pendulum system. However, we can still
    # use it to display plotly plots in the GUI.
    # Plotly plots are automatically added to the GUI at the end of the sidebar
    viser_renderer = ViserRenderer(robot, num_points=50, backbone_style="discrete")
    viser_renderer.render_sequence(
        ts,
        q_ts,
        playback_speed=1.0,
        loop=True,
        autoplay=True,
        plot_configurations=True,
        robot_name="Pendulum",
    )
