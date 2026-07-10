from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from diffrax import Tsit5

jax.config.update("jax_enable_x64", True)  # double precision
from soromox.rendering import (
    ActuatorStyleConfig,
    BackboneColorConfig,
    MatplotlibRenderer,
    Open3DRenderer,
    RendererColorConfig,
    ViserRenderer,
    get_color_theme,
)
from soromox.systems import PCS, PCSParams, SystemState

jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)


if __name__ == "__main__":
    num_segments = 2
    rho = 1070 * jnp.ones(
        (num_segments,)
    )  # Volumetric density of Dragon Skin 20 [kg/m^3]
    segment_lengths = 1e-1 * jnp.ones((num_segments,))
    material_damping_coefficient = 362.0
    params = PCSParams(
        base_pose=jnp.array(
            [0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0]
        ),  # Initial position and orientation
        length=segment_lengths,
        radius=2e-2 * jnp.ones((num_segments,)),
        density=rho,
        gravity=jnp.array([0.0, 0.0, 9.81]),  # Gravity vector [m/s^2]
        young_modulus=2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
        shear_modulus=1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
        material_damping_coefficient=material_damping_coefficient,
        reference_strain=jnp.tile(
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), num_segments
        ),
    )

    # ======================================================
    # Robot initialization
    # ======================================================
    robot = PCS(params=params)

    # =====================================================
    # Simulation upon time
    # =====================================================
    # Initial configuration
    q0 = jnp.repeat(
        jnp.array([0.0, 0.0, 5.0 * jnp.pi, 0.1, 0.2, 0.0])[None, :],
        num_segments,
        axis=0,
    ).flatten()
    # Initial velocities
    qd0 = jnp.zeros_like(q0)

    # Actuation parameters
    u = jnp.zeros_like(q0)

    # Simulation time parameters
    t0 = 0.0
    t1 = 2.0
    solver_dt = 1e-4
    save_dt = 0.01

    # Solver
    solver = Tsit5()  # Runge-Kutta 5(4) method

    initial_state = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))
    trajectory = robot.rollout_to(
        initial_state=initial_state,
        u=u,
        t1=t1,
        solver_dt=solver_dt,
        save_dt=save_dt,
        solver=solver,
        max_steps=None,
    )
    ts = trajectory.t
    q_ts, qd_ts = jnp.split(trajectory.y, 2, axis=1)

    # =====================================================
    # End-effector position upon time
    # =====================================================
    forward_kinematics_end_effector = jax.jit(
        partial(
            robot.forward_kinematics,
            s=jnp.sum(robot.L),  # end-effector position
        )
    )
    g_ee_ts = jax.vmap(forward_kinematics_end_effector)(q_ts)

    plt.figure()
    plt.plot(ts, g_ee_ts[:, 0, 3], label="End-effector x [m]")
    plt.plot(ts, g_ee_ts[:, 1, 3], label="End-effector y [m]")
    plt.plot(ts, g_ee_ts[:, 2, 3], label="End-effector z [m]")
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector position [m]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(
        g_ee_ts[:, 0, 3], g_ee_ts[:, 1, 3], g_ee_ts[:, 2, 3], c=ts, cmap="viridis"
    )
    ax.axis("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("End-effector trajectory (3D)")
    fig.colorbar(p, ax=ax, label="Time [s]")
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

    # =====================================================
    # Animate the robot motion
    # =====================================================
    q_demo = q_ts[len(ts) // 2]

    # Color scheme demos (built-in palettes + themes)
    demo_renderer = MatplotlibRenderer(robot, num_points=50)
    demo_renderer.show(
        q_demo,
        color_config=RendererColorConfig(
            backbone=BackboneColorConfig(segment_palette="viridis")
        ),
    )
    demo_renderer.show(q_demo, color_config=get_color_theme("soromox:paper"))
    demo_renderer.show(
        q_demo,
        color_config=RendererColorConfig(
            backbone=BackboneColorConfig(point_palette="soromox:glacier"),
            base_plate_color=(0.15, 0.15, 0.15),
            actuators=ActuatorStyleConfig(default_color=(0.2, 0.4, 0.8)),
        ),
    )

    renderer = MatplotlibRenderer(robot, num_points=50)
    renderer.animate(ts=ts, q_ts=q_ts, interval=100, mode="slider")
    renderer = Open3DRenderer(robot, num_points=50)
    renderer.render_sequence(ts, q_ts)

    # =====================================================
    # Viser web-based visualization (opens in browser)
    # =====================================================
    # ViserRenderer provides interactive 3D visualization in the browser
    # with GUI controls for playback, speed, and looping.
    # Plotly plots are automatically added to the GUI at the end of the sidebar
    viser_renderer = ViserRenderer(robot, num_points=50)

    # Create custom strain plots for PCS
    # Reshape to (T, num_segments, 6)
    q_reshaped = q_ts.reshape(len(ts), num_segments, 6)

    # Rotational strains (first 3 components)
    fig_rot = go.Figure()
    for seg in range(num_segments):
        for i, label in enumerate(["κx", "κy", "κz"]):
            fig_rot.add_trace(
                go.Scatter(
                    x=ts,
                    y=q_reshaped[:, seg, i],
                    mode="lines",
                    name=f"Seg {seg} - {label}",
                )
            )
    fig_rot.update_layout(
        title="PCS - Rotational Strains vs Time",
        xaxis_title="Time [s]",
        yaxis_title="Rotational Strain [rad/m]",
        height=400,
        margin={"l": 50, "r": 50, "t": 50, "b": 50},
    )

    # Linear strains (last 3 components)
    fig_lin = go.Figure()
    for seg in range(num_segments):
        for i, label in enumerate(["σx", "σy", "σz"]):
            fig_lin.add_trace(
                go.Scatter(
                    x=ts,
                    y=q_reshaped[:, seg, i + 3],
                    mode="lines",
                    name=f"Seg {seg} - {label}",
                )
            )
    fig_lin.update_layout(
        title="PCS - Linear Strains vs Time",
        xaxis_title="Time [s]",
        yaxis_title="Linear Strain [-]",
        height=400,
        margin={"l": 50, "r": 50, "t": 50, "b": 50},
    )

    viser_renderer.render_sequence(
        ts,
        q_ts,
        playback_speed=1.0,
        loop=True,
        autoplay=True,
        plot_configurations=False,
        custom_plots={
            "Rotational Strains": (fig_rot, 2.0),
            "Linear Strains": (fig_lin, 2.0),
        },
        robot_name="PCS",
    )
