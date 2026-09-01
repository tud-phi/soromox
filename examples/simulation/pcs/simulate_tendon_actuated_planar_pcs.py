from functools import partial

import jax
import matplotlib.pyplot as plt
from diffrax import Tsit5
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)  # double precision
from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.rendering import MatplotlibRenderer
from soromox.systems import (
    LinkSpec,
    PlanarPCS,
    PlanarPCSStructure,
    SystemState,
)

if __name__ == "__main__":
    num_segments = 2  # number of segments in the robot
    rho = 1070 * jnp.ones(
        (num_segments,)
    )  # Volumetric density of Dragon Skin 20 [kg/m^3]
    segment_lengths = 1e-1 * jnp.ones((num_segments,))
    material_damping_coefficient = 318.0
    body = PlanarPCS.params_from_links(
        [
            LinkSpec.circular(
                length=float(segment_lengths[index]),
                radius=2e-2,
                density=float(rho[index]),
                young_modulus=5e3,
                shear_modulus=1e3,
                material_damping_coefficient=material_damping_coefficient,
                reference_strain=[0, 1, 0],
            )
            for index in range(num_segments)
        ],
        gravity=0 * jnp.array([0.0, 9.81]),
    )
    tendon_offsets = 2e-2 * jnp.array([[1.0, -1.0]]).repeat(num_segments, axis=0)
    active_tendon_routing = ThreadlikeRouting.linear(
        intercept=jnp.stack(
            (
                jnp.zeros((2 * num_segments,)),
                tendon_offsets.reshape(-1),
                jnp.zeros((2 * num_segments,)),
            ),
            axis=-1,
        ),
        start_segment_index=0,
        end_segment_index=tuple(
            int(index) for index in jnp.repeat(jnp.arange(num_segments), 2)
        ),
    )

    # activate all strains (i.e. bending, shear, and axial)
    strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

    # ======================================================
    # Robot initialization
    # ======================================================
    robot = PlanarPCS(
        params=body,
        base_pose=jnp.array([jnp.pi / 2, 0.0, 0.0]),
        structure=PlanarPCSStructure(strain_selector=strain_selector),
        actuators=ThreadlikeActuator.tendons(active_tendon_routing),
    )

    # =====================================================
    # Simulation upon time
    # =====================================================
    # Initial configuration
    # q0 = jnp.repeat(
    #     jnp.array([5.0 * jnp.pi, 0.2, 0.1])[None, :], num_segments, axis=0
    # ).flatten()
    # q0 = jnp.zeros_like(q0)
    # randomly sample initial configuration
    key = jax.random.PRNGKey(4)
    q0 = (
        jax.random.normal(key, shape=int(robot.num_active_strains))
        * jnp.repeat(
            jnp.array([5.0 * jnp.pi, 0.1, 0.05])[None, :], num_segments, axis=0
        ).flatten()
    )
    print("q0 =\n", q0)

    # # visualize initial configuration
    # cv2.imshow(
    #     "Robot",
    #     rendering_fn(q0),
    # )
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Initial velocities
    qd0 = jnp.zeros_like(q0)

    # Actuation parameters
    # u = jnp.zeros((robot.num_actuators,))  # no actuation
    # u = (
    #     jnp.array([0.0, -0.3])[None].repeat(num_segments, axis=0).flatten()
    # )  # tendon tensions
    # randomly sample actuation
    # u = jax.random.uniform(
    #     key,
    #     shape=(robot.num_actuators,),
    #     minval=-0.5,
    #     maxval=0.0,
    # )
    # alternating tendon tensions
    base_tension = jnp.array(0.3, dtype=q0.dtype)
    segment_ids = jnp.arange(num_segments)
    segment_tensions = base_tension / (segment_ids.astype(q0.dtype) + 1.0)
    even_segments = (segment_ids % 2) == 0
    tendon_pattern = jnp.zeros((num_segments, 2), dtype=q0.dtype)
    tendon_pattern = tendon_pattern.at[:, 0].set(
        jnp.where(even_segments, segment_tensions, 0.0)
    )
    tendon_pattern = tendon_pattern.at[:, 1].set(
        jnp.where(~even_segments, segment_tensions, 0.0)
    )
    u = tendon_pattern.reshape(-1)
    print("u =\n", u)

    # call the actuation mapping function
    A = robot.actuation_matrix(
        q0,
    )
    print("A =\n", A)

    # Simulation time parameters
    t0 = 0.0
    t1 = 10.0
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
    print("Final configuration q(t1) =\n", q_ts[-1])

    # =====================================================
    # End-effector position upon time
    # =====================================================
    forward_kinematics_end_effector = jax.jit(
        partial(
            robot.forward_kinematics,
            s=jnp.sum(robot.L),  # end-effector position
        )
    )
    chi_ee_ts = jax.vmap(forward_kinematics_end_effector)(q_ts)

    plt.figure()
    for segment_idx in range(num_segments):
        plt.plot(
            ts,
            q_ts[:, 3 * segment_idx + 0],
            label=r"$\kappa_\mathrm{be," + str(segment_idx + 1) + "}$ [rad/m]",
        )
        plt.plot(
            ts,
            q_ts[:, 3 * segment_idx + 2],
            label=r"$\sigma_\mathrm{ax," + str(segment_idx + 1) + "}$ [-]",
        )
        plt.plot(
            ts,
            q_ts[:, 3 * segment_idx + 1],
            label=r"$\sigma_\mathrm{sh," + str(segment_idx + 1) + "}$ [-]",
        )
    plt.xlabel("Time [s]")
    plt.ylabel("Configuration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.plot(ts, chi_ee_ts[:, 1], label="End-effector x [m]")
    plt.plot(ts, chi_ee_ts[:, 2], label="End-effector y [m]")
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector position [m]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()

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

    # plot the end-effector position in the x-y plane as a scatter plot with the time as the color
    plt.figure()
    plt.scatter(chi_ee_ts[:, 1], chi_ee_ts[:, 2], c=ts, cmap="viridis")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("End-effector x [m]")
    plt.ylabel("End-effector y [m]")
    plt.colorbar(label="Time [s]")
    plt.tight_layout()

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
