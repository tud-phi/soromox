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
from soromox.rendering import (
    ISupportViserRenderer,
    ISupportVisualConfig,
    MatplotlibRenderer,
)
from soromox.systems import (
    PCS,
    ISupport,
    ISupportParams,
    ISupportStructure,
    LinkSpec,
    SystemState,
)

if __name__ == "__main__":
    num_pneumatic_segments = 2

    # Elastic modulus and poisson ratio
    E = 1.6464 * 1e6  # Elastic modulus [Pa]
    poisson_ratio = 0.5
    G = E / (
        2 * (1 + poisson_ratio)
    )  # Shear modulus from elastic modulus and poisson ratio

    # Physical layout: base interface, pneumatic section, middle interface,
    # pneumatic section, tip interface.
    rigid_segment_selector = (True, False, True, False, True)
    physical_segment_lengths = jnp.array([41e-3, 190e-3, 27e-3, 180e-3, 6e-3])
    # Alessi et al. (2023) report a 30 mm circular cross-section radius for
    # the complete arm, including its pneumatic sections and terminal plates.
    physical_segment_radii = 30e-3 * jnp.ones((len(rigid_segment_selector),))
    # The compiled reference model stores 1210 kg/m^3 for every rigid
    # interface and 1104 kg/m^3 for each pneumatic section.
    physical_segment_densities = jnp.array([1210.0, 1104.0, 1210.0, 1104.0, 1210.0])
    # Topology: how many PCS segments approximate each pneumatic segment.
    pcs_segment_counts = (2, 3)
    # Parameters: metric PCS segment lengths, flattened by pneumatic segment.
    # The first two entries sum to 190 mm; the last three sum to 180 mm.
    # Set this to None to divide each pneumatic segment equally according to
    # pcs_segment_counts.
    pcs_segment_lengths = jnp.array([95e-3, 95e-3, 60e-3, 60e-3, 60e-3])
    # Preserve the calibrated generalized damping used by the reference model.
    gamma_rotational = 1.0e-3
    gamma_translational = 806e-3
    links = [
        LinkSpec.circular(
            length=float(physical_segment_lengths[index]),
            radius=float(physical_segment_radii[index]),
            density=float(physical_segment_densities[index]),
            young_modulus=E,
            shear_modulus=G,
            damping=float(physical_segment_lengths[index])
            * jnp.diag(
                jnp.array(
                    [
                        gamma_rotational,
                        gamma_rotational,
                        gamma_rotational,
                        gamma_translational,
                        gamma_translational,
                        gamma_translational,
                    ]
                )
            ),
            reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )
        for index in range(len(rigid_segment_selector))
    ]
    params = ISupportParams(
        gravity=jnp.array([0.0, 0.0, -9.81]),
        link=PCS.params_from_links(links).link,
        chamber_inner_radius=6.39 * 1e-3 * jnp.ones((num_pneumatic_segments,)),
        chamber_outer_radius=7.79 * 1e-3 * jnp.ones((num_pneumatic_segments,)),
        chamber_distance=20 * 1e-3 * jnp.ones((num_pneumatic_segments,)),
        # Optional: override the default effective area
        # pi * (chamber_outer_radius**2 - chamber_inner_radius**2).
        # Explicit [0, 2*pi/3, 4*pi/3] = [0, 120, 240] degrees for each
        # pneumatic segment. Array index is the corresponding pressure channel.
        chamber_azimuth_angles=jnp.tile(
            2.0 * jnp.pi * jnp.arange(3) / 3,
            (num_pneumatic_segments, 1),
        ),
        pcs_segment_lengths=pcs_segment_lengths,
    )

    # ======================================================
    # Robot initialization
    # ======================================================
    robot = ISupport(
        params=params,
        base_pose=jnp.array([0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0]),
        structure=ISupportStructure(
            pcs_segment_counts=pcs_segment_counts,
            rigid_segment_selector=rigid_segment_selector,
        ),
    )

    # =====================================================
    # Simulation upon time
    # =====================================================
    # Initial configuration
    q0_pneumatic = jnp.repeat(
        jnp.array([0.0, 0.0, -1.0 * jnp.pi, 0.1, 0.2, 0.0])[None, :],
        num_pneumatic_segments,
        axis=0,
    )
    q0 = jnp.concatenate(
        [
            jnp.tile(q0_pneumatic[i], pcs_segment_counts[i])
            for i in range(num_pneumatic_segments)
        ]
    )
    # Initial velocities
    qd0 = jnp.zeros_like(q0)

    # # compute the actuation matrix at q0
    # A = robot.actuation_matrix(q0)
    # print("A:\n", A)

    # Actuation pressures [Pa]. The threadlike transmission uses the
    # pressure-conjugate coordinate V = effective_area * chamber_path_length.
    u = (
        jnp.repeat(
            jnp.array([2e-1, 0.0, 0.0])[None, :],
            num_pneumatic_segments,
            axis=0,
        ).flatten()
        * 1e5
    )

    # Simulation time parameters
    t0 = 0.0
    t1 = 2.0
    solver_dt = 2e-5
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

    # =====================================================
    # Viser web-based visualization
    # =====================================================
    if ISupportViserRenderer is None:
        print("ISupportViserRenderer is unavailable. Install with `pip install viser`.")
    else:
        viser_renderer = ISupportViserRenderer(
            robot,
            num_points=50,
            visual_config=ISupportVisualConfig(pressure_range=(0.0, float(jnp.max(u)))),
        )
        viser_renderer.render_sequence(
            ts=ts,
            q_ts=q_ts,
            pressures=u,
            playback_speed=1.0,
            autoplay=True,
            loop=True,
            render_actuators=True,
            plot_configurations=True,
            robot_name="I-SUPPORT",
        )
