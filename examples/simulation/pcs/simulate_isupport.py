"""
Simulates an Additive Manufacturing (AM) I-SUPPORT manipulator. The system parameters are adapted from

Alessi, Carlo, Egidio Falotico, and Alessandro Lucantonio.
"Ablation study of a dynamic model for a 3d-printed pneumatic soft robotic arm." IEEE Access 11 (2023): 37840-37853.
https://ieeexplore.ieee.org/abstract/document/10098800
"""

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)  # double precision
from soromox.rendering import MatplotlibRenderer
from soromox.systems import ISupport, ISupportParams, SystemState

if __name__ == "__main__":
    num_segments = 2

    # Elastic modulus and poisson ratio
    E = 1.6464 * 1e6  # Elastic modulus [Pa]
    poisson_ratio = 0.5
    G = E / (
        2 * (1 + poisson_ratio)
    )  # Shear modulus from elastic modulus and poisson ratio

    segment_lengths = 190 * 1e-3 * jnp.ones((num_segments,))

    # damping coefficient
    # these values are from the paper but they seem way too large
    # gamma_t = 806  # translational damping constant [1/s]
    # gamma_r = 1.9416 * 10**(-4)  # rotational damping constant [m^2/s]
    gamma_t = 806 * 1e-3  # translational damping constant [1/s]
    gamma_r = 1.0 * 1e-3  # rotational damping constant [m^2/s]
    # Damping is specified per unit backbone length and must be integrated over
    # each segment, matching the strain-space stiffness assembly. Without this
    # length scaling the velocity term makes the two-segment fixed-step rollout
    # unnecessarily stiff and can drive the explicit solver to NaNs.
    damping_matrix = jnp.diag(
        (
            jnp.repeat(
                jnp.array([[gamma_r, gamma_r, gamma_r, gamma_t, gamma_t, gamma_t]]),
                num_segments,
                axis=0,
            )
            * segment_lengths[:, None]
        ).flatten()
    )
    params = ISupportParams(
        base_pose=jnp.array([jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, 0.0]),
        length=segment_lengths,
        radius=35.6 * 1e-3 * jnp.ones((num_segments,)),
        density=1104 * jnp.ones((num_segments,)),
        gravity=jnp.array([0.0, 0.0, 9.81]),
        young_modulus=E * jnp.ones((num_segments,)),
        shear_modulus=G * jnp.ones((num_segments,)),
        damping_matrix=damping_matrix,
        reference_strain=jnp.tile(
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), num_segments
        ),
        chamber_inner_radius=6.39 * 1e-3 * jnp.ones((num_segments,)),
        chamber_outer_radius=7.79 * 1e-3 * jnp.ones((num_segments,)),
        chamber_distance=20 * 1e-3 * jnp.ones((num_segments,)),
        chamber_angle_offset=jnp.zeros((num_segments,)),
    )

    # ======================================================
    # Robot initialization
    # ======================================================
    robot = ISupport(params=params)

    # =====================================================
    # Simulation upon time
    # =====================================================
    # Initial configuration
    q0 = jnp.repeat(
        jnp.array([0.0, 0.0, -1.0 * jnp.pi, 0.1, 0.2, 0.0])[None, :],
        num_segments,
        axis=0,
    ).flatten()
    # Initial velocities
    qd0 = jnp.zeros_like(q0)

    # # compute the actuation matrix at q0
    # A = robot.actuation_matrix(q0)
    # print("A:\n", A)

    # Actuation pressures
    u = (
        jnp.repeat(jnp.array([2e-1, 0.0, 0.0])[None, :], num_segments, axis=0).flatten()
        * 1e5
    )

    # Simulation time parameters
    t0 = 0.0
    t1 = 2.0
    solver_dt = 5e-5
    save_dt = 0.01

    initial_state = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))
    trajectory = robot.rollout_to(
        initial_state=initial_state,
        u=u,
        t1=t1,
        solver_dt=solver_dt,
        save_dt=save_dt,
        max_steps=None,
    )
    ts = trajectory.t
    q_ts, qd_ts = jnp.split(trajectory.y, 2, axis=1)

    q1 = q_ts[-1]
    print("q1:\n", q1)

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
    # Plot the robot configuration upon time
    # =====================================================
    renderer = MatplotlibRenderer(robot, num_points=50)
    renderer.animate(ts=ts, q_ts=q_ts, interval=100, mode="slider")
