from functools import partial

import jax
import matplotlib.pyplot as plt
import numpy as np
from diffrax import Tsit5
from jax import numpy as jnp
from jax import vmap

jax.config.update("jax_enable_x64", True)  # double precision
from soromox.rendering import MatplotlibRenderer
from soromox.systems import (
    PlanarPCSStructure,
    PneumaticActuatedPlanarPCS,
    PneumaticActuatedPlanarPCSParams,
    SystemState,
)


def sweep_actuation_mapping(
    robot: PneumaticActuatedPlanarPCS,
):
    # Actuation mapping for a straight backbone
    q = jnp.zeros(
        (robot.num_active_strains,)
    )  # straight backbone, no bending or axial strain
    A = robot.actuation_matrix(q)
    print("Evaluating actuation matrix for straight backbone: A =\n", A)

    # Curvature and axial strain points to evaluate the actuation mapping
    kappa_be_pts = jnp.linspace(-3 * jnp.pi, 3 * jnp.pi, 500)
    sigma_ax_pts = jnp.zeros_like(kappa_be_pts)  # no axial strain
    q_pts = jnp.stack([kappa_be_pts, sigma_ax_pts], axis=-1)
    A_pts = vmap(robot.actuation_matrix)(q_pts)

    # mark the points that are not controllable as the u1 and u2 terms share the same sign
    non_controllable_selector = A_pts[..., 0, 0] * A_pts[..., 0, 1] >= 0.0
    non_controllable_indices = jnp.where(non_controllable_selector)[0]
    non_controllable_boundary_indices = jnp.where(
        non_controllable_selector[:-1] != non_controllable_selector[1:]
    )[0]
    # plot the mapping on the bending strain for various bending strains
    fig, ax = plt.subplots(
        num="pneumatic_planar_pcs_actuation_mapping_bending_torque_vs_bending_strain"
    )
    plt.title(r"Actuation mapping from $u$ to $\tau_\mathrm{be}$")
    # # shade the region where the actuation mapping is negative as we are not able to bend the robot further
    # ax.axhspan(A_pts[:, 0, 0:2].min(), 0.0, facecolor='red', alpha=0.2)
    for idx in non_controllable_indices:
        ax.axvspan(kappa_be_pts[idx], kappa_be_pts[idx + 1], facecolor="red", alpha=0.2)
    ax.plot(
        kappa_be_pts,
        A_pts[:, 0, 0],
        linewidth=2,
        label=r"$\frac{\partial \tau_\mathrm{be}}{\partial u_1}$",
    )
    ax.plot(
        kappa_be_pts,
        A_pts[:, 0, 1],
        linewidth=2,
        label=r"$\frac{\partial \tau_\mathrm{ax}}{\partial u_2}$",
    )
    ax.set_xlabel(r"$\kappa_\mathrm{be}$ [rad/m]")
    ax.set_ylabel(r"$\frac{\partial \tau_\mathrm{be}}{\partial u_1}$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # plot the actuation mapping of u1 vs. the bending strain for various segment radii
    r_pts = jnp.linspace(1e-2, 1e-1, 10)
    r_chamber_out_pts = r_pts - 2e-3
    fig, ax = plt.subplots(
        num="pneumatic_planar_pcs_actuation_mapping_bending_torque_vs_bending_strain_4segment_radii"
    )
    for r, r_chamber_out in zip(r_pts, r_chamber_out_pts):
        updated_robot = robot.update_params(
            radius=r * jnp.ones((robot.num_segments,)),
            chamber_outer_radius=r_chamber_out * jnp.ones((robot.num_segments,)),
        )
        A_pts = vmap(updated_robot.actuation_matrix)(q_pts)
        ax.plot(kappa_be_pts, A_pts[:, 0, 0], label=r"$R = " + str(r) + "$")
    ax.set_xlabel(r"$\kappa_\mathrm{be}$ [rad/m]")
    ax.set_ylabel(r"$\frac{\partial \tau_\mathrm{be}}{\partial u_1}$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # create grid for bending and axial strains
    kappa_be_grid, sigma_ax_grid = jnp.meshgrid(
        jnp.linspace(-jnp.pi, jnp.pi, 20),
        jnp.linspace(-0.2, 0.2, 20),
    )
    q_pts = jnp.stack([kappa_be_grid.flatten(), sigma_ax_grid.flatten()], axis=-1)

    # evaluate the actuation mapping on the grid
    A_pts = vmap(robot.actuation_matrix)(q_pts)
    # reshape A_pts to match the grid shape
    A_grid = A_pts.reshape(kappa_be_grid.shape[:2] + A_pts.shape[-2:])

    # plot the mapping on the bending strain
    fig, ax = plt.subplots(
        num="pneumatic_planar_pcs_actuation_mapping_bending_torque_vs_axial_vs_bending_strain"
    )
    plt.title(r"Actuation mapping from $u_1$ to $\tau_\mathrm{be}$")
    # contourf plot
    c = ax.contourf(kappa_be_grid, sigma_ax_grid, A_grid[..., 0, 0], levels=100)
    fig.colorbar(c, ax=ax, label=r"$\frac{\partial \tau_\mathrm{be}}{\partial u_1}$")
    # contour plot
    ax.contour(
        kappa_be_grid,
        sigma_ax_grid,
        A_grid[..., 0, 0],
        levels=20,
        colors="k",
        linewidths=0.5,
    )
    ax.set_xlabel(r"$\kappa_\mathrm{be}$ [rad/m]")
    ax.set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
    plt.tight_layout()
    plt.show()

    # plot the mapping on the axial strain
    fig, ax = plt.subplots(
        num="pneumatic_planar_pcs_actuation_mapping_axial_torque_vs_axial_vs_bending_strain"
    )
    plt.title(r"Actuation mapping from $u_1$ to $\tau_\mathrm{ax}$")
    # contourf plot
    c = ax.contourf(kappa_be_grid, sigma_ax_grid, A_grid[..., 1, 0], levels=100)
    fig.colorbar(c, ax=ax, label=r"$\frac{\partial \tau_\mathrm{ax}}{\partial u_1}$")
    # contour plot
    ax.contour(
        kappa_be_grid,
        sigma_ax_grid,
        A_grid[..., 1, 0],
        levels=20,
        colors="k",
        linewidths=0.5,
    )
    ax.set_xlabel(r"$\kappa_\mathrm{be}$ [rad/m]")
    ax.set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    num_segments = 1
    rho = 1070 * jnp.ones(
        (num_segments,)
    )  # Volumetric density of Dragon Skin 20 [kg/m^3]
    segment_lengths = 1e-1 * jnp.ones((num_segments,))
    damping_matrix = 5e-4 * jnp.diag(
        (
            jnp.repeat(jnp.array([[1e0, 1e3, 1e3]]), num_segments, axis=0)
            * segment_lengths[:, None]
        ).flatten()
    )
    params = PneumaticActuatedPlanarPCSParams(
        base_pose=jnp.array([jnp.pi / 2, 0.0, 0.0]),
        length=segment_lengths,
        radius=2e-2 * jnp.ones((num_segments,)),
        density=rho,
        gravity=jnp.array([0.0, 9.81]),
        young_modulus=2e3 * jnp.ones((num_segments,)),
        shear_modulus=1e3 * jnp.ones((num_segments,)),
        damping_matrix=damping_matrix,
        reference_strain=jnp.tile(jnp.array([0.0, 1.0, 0.0]), num_segments),
        chamber_inner_radius=5e-3 * jnp.ones((num_segments,)),
        chamber_outer_radius=(2e-2 - 2e-3) * jnp.ones((num_segments,)),
        chamber_angle=jnp.pi / 2 * jnp.ones((num_segments,)),
        chamber_distance=jnp.zeros((num_segments,)),
    )

    strain_selector = (
        jnp.array([True, True, False])[None, :].repeat(num_segments, axis=0).flatten()
    )  # Select bending and axial strains, no shear strain

    # ======================================================
    # Robot initialization
    # ======================================================
    robot = PneumaticActuatedPlanarPCS(
        params=params,
        structure=PlanarPCSStructure(strain_selector=strain_selector),
        chamber_cross_section_geometry="concentric",
    )

    print("A=", robot.actuation_matrix(q=jnp.zeros(robot.num_active_strains)))

    # ======================================================
    # Sweep the actuation mapping
    # ======================================================

    # sweep_actuation_mapping(robot)

    # =====================================================
    # Simulation upon time
    # =====================================================
    # Initial configuration
    q0 = jnp.repeat(
        jnp.array([-5.0 * jnp.pi, -0.2])[None, :], robot.num_segments, axis=0
    ).flatten()

    # Draw the initial configuration
    renderer = MatplotlibRenderer(robot, num_points=100)
    curve = np.array(renderer.compute_backbone_curve(q0))
    plt.figure()
    plt.plot(curve[:, 0], curve[:, 1], lw=4, color="blue")
    plt.plot(curve[0, 0], curve[0, 1], "o", color="blue", label="Proximal end")
    plt.plot(curve[-1, 0], curve[-1, 1], "o", color="red", label="Distal end")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Initial robot configuration")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Initial velocities
    qd0 = jnp.zeros_like(q0)

    # Actuation parameters
    u = jnp.array(
        [1.2e3, 0e0]
    )  # control inputs (pressures in the right and left chambers)

    print("u =\n", u)

    # Simulation time parameters
    t0 = 0.0
    t1 = 7.0
    solver_dt = 5e-5
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
    chi_ee_ts = jax.vmap(forward_kinematics_end_effector)(q_ts)

    # plot the configuration vs time
    plt.figure()
    for segment_idx in range(robot.num_segments):
        plt.plot(
            ts,
            q_ts[:, 2 * segment_idx + 0],
            label=r"$\kappa_\mathrm{be," + str(segment_idx + 1) + "}$ [rad/m]",
        )
        plt.plot(
            ts,
            q_ts[:, 2 * segment_idx + 1],
            label=r"$\sigma_\mathrm{ax," + str(segment_idx + 1) + "}$ [-]",
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

    # =====================================================
    # Plot the robot configuration upon time
    # =====================================================
    renderer.animate(ts=ts, q_ts=q_ts, interval=100, mode="slider")
