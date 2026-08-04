import argparse
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from diffrax import Tsit5

from soromox.actuation import (
    ThreadlikeActuator,
    ThreadlikeImpedance,
    ThreadlikeRouting,
)
from soromox.rendering import (
    ActuatorStyleConfig,
    MatplotlibRenderer,
    Open3DRenderer,
    RendererColorConfig,
    ViserRenderer,
)
from soromox.systems import (
    PCS,
    LinkSpec,
    SystemState,
)

jax.config.update("jax_enable_x64", True)  # double precision

jnp.set_printoptions(precision=4, suppress=True)

DEFAULT_VIDEO_PATH = (
    Path(__file__).resolve().parent / "videos" / "simulate_tendon_actuated_pcs.mp4"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate a tendon-actuated PCS robot."
    )
    parser.add_argument("--video-output", type=Path, default=DEFAULT_VIDEO_PATH)
    args = parser.parse_args()
    num_segments = 2
    rho = 1070 * jnp.ones(
        (num_segments,)
    )  # Volumetric density of Dragon Skin 20 [kg/m^3]
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
        base_pose=jnp.array([0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0]),
        gravity=jnp.array([0.0, 0.0, 9.81]),
    )
    active_tendon_routing = ThreadlikeRouting.linear(
        intercept=2e-2 * jnp.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]),
        start_segment_index=0,
        end_segment_index=(1, 0),
    )
    passive_tendon_routing = ThreadlikeRouting.linear(
        intercept=2e-2 * jnp.array([0.0, 0.0, 1.0]),
        start_segment_index=0,
        end_segment_index=(1,),
    )
    passive_tendon = ThreadlikeImpedance(
        routing=passive_tendon_routing,
        stiffness=jnp.array([0.6]),
        damping=jnp.array([0.1]),
        rest_length=jnp.array([jnp.sum(segment_lengths) - 0.3]),
    )

    # ======================================================
    # Robot initialization
    # ======================================================
    robot = PCS(
        params=body_params,
        actuators=ThreadlikeActuator.tendons(active_tendon_routing),
        passive_elements=(passive_tendon,),
    )

    # =====================================================
    # Simulation upon time
    # =====================================================
    # Initial configuration
    q0 = jnp.repeat(
        jnp.array([0.0, 0.0, 5.0 * jnp.pi, 0.1, 0.2, 0.0])[None, :],
        num_segments,
        axis=0,
    ).flatten()
    # q0 = jnp.zeros_like(q0)

    # Initial velocities
    qd0 = jnp.zeros_like(q0)

    # Actuation parameters
    u = jnp.array([0.5, 1.0])
    print("u =\n", u)

    # Actuation matrix
    A = robot.actuation_matrix(q0)
    print("A =\n", A.shape)
    print(A)

    # Coupling matrix of the passive tendons
    P = jax.jacrev(lambda q: passive_tendon.path_lengths(robot, q))(q0)
    print("P =\n", P.shape)
    print(P)

    # Active tendon lengths
    la = robot.actuators[0].path_lengths(robot, q0)
    print("la =\n", la.shape)
    print(la)

    # Passive tendon lengths
    lp = passive_tendon.path_lengths(robot, q0)
    print("lp =\n", lp.shape)
    print(lp)

    # Active tendons' position
    ta_s = robot.actuators[0].path_poses(robot, q0, robot.L_cum[-1])
    print("ta_s =\n", ta_s.shape)
    print(ta_s)

    # Passive tendons' position
    tp_s = passive_tendon.path_poses(robot, q0, robot.L_cum[-1])
    print("tp_s =\n", tp_s.shape)
    print(tp_s)

    # Simulation time parameters
    t0 = 0.0
    t1 = 2.0
    solver_dt = 1e-4
    save_dt = 0.01

    # Solver
    solver = Tsit5()  # Runge-Kutta 5(4) method

    initial_time = time.time()
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
    end_time = time.time()
    print(
        f"Total simulation time = {round(end_time - initial_time, ndigits=3)} s  |  t0 = {t0}, t1 = {t1}, solver_dt0 = {solver_dt}"
    )

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

    # animate using the MatplotlibRenderer
    matplotlib_renderer = MatplotlibRenderer(
        robot,
        num_points=60,
        line_width=2.5,
        actuator_line_width=2.0,
        color_config=RendererColorConfig(
            actuators=ActuatorStyleConfig(default_color=(0.9, 0.1, 0.1))
        ),
    )
    # q_ts is (T, DOF); animate in slider mode for quick inspection
    matplotlib_renderer.animate(
        ts=ts, q_ts=q_ts, mode="slider", show=True, render_actuators=True
    )

    # render using the Open3DRenderer
    renderer = Open3DRenderer(robot)
    renderer.render_sequence(
        ts=ts,
        q_ts=q_ts,
        record_path=str(args.video_output),
    )

    # =====================================================
    # Viser web-based visualization (opens in browser)
    # =====================================================
    # ViserRenderer provides interactive 3D visualization in the browser
    # with GUI controls for playback, speed, and looping.
    # Plotly plots are automatically added to the GUI at the end of the sidebar
    viser_renderer = ViserRenderer(robot, num_points=50)

    # Create custom strain plots for Tendon-Actuated PCS
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
        title="Tendon-Actuated PCS - Rotational Strains vs Time",
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
        title="Tendon-Actuated PCS - Linear Strains vs Time",
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
        render_actuators=True,
        plot_configurations=False,
        plot_actuator_positions=True,
        custom_plots={
            "Rotational Strains": (fig_rot, 2.0),
            "Linear Strains": (fig_lin, 2.0),
        },
        robot_name="Tendon-Actuated PCS",
    )
