from functools import partial
import jax

jax.config.update("jax_enable_x64", True)  # double precision
from diffrax import diffeqsolve, Euler, ODETerm, SaveAt, Tsit5
from jax import Array, vmap
from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path
from typing import Callable, Dict

import soromox
from soromox import ode_factory
from soromox.systems import pneumatically_actuated_planar_pcs

num_segments = 1

# filepath to symbolic expressions
sym_exp_filepath = (
    Path(soromox.__file__).parent
    / "symbolic_expressions"
    / f"planar_pcs_ns-{num_segments}.dill"
)

# set parameters
rho = 1070 * jnp.ones((num_segments,))  # Volumetric density of Dragon Skin 20 [kg/m^3]
params = {
    "th0": jnp.array(0.0),  # initial orientation angle [rad]
    "l": 1e-1 * jnp.ones((num_segments,)),
    "r": 2e-2 * jnp.ones((num_segments,)),
    "rho": rho,
    "g": jnp.array([0.0, 9.81]), # gravitational acceleration [m/s^2] UP!
    "E": 2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
    "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
    "r_cham_in": 5e-3 * jnp.ones((num_segments,)),
    "r_cham_out": 2e-2 - 2e-3 * jnp.ones((num_segments,)),
    "varphi_cham": jnp.pi/2 * jnp.ones((num_segments,)),
}
params["D"] = 5e-4 * jnp.diag(
    (jnp.repeat(
        jnp.array([[1e0, 1e3, 1e3]]), num_segments, axis=0
    ) * params["l"][:, None]).flatten()
)

# activate all strains (i.e. bending, shear, and axial)
# strain_selector = jnp.ones((3 * num_segments,), dtype=bool)
strain_selector = jnp.array([True, False, True])[None, :].repeat(num_segments, axis=0).flatten()

B_xi, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = (
    pneumatically_actuated_planar_pcs.factory(
        num_segments, sym_exp_filepath, strain_selector, # simplified_actuation_mapping=True
    )
)
# jit the functions
dynamical_matrices_fn = jax.jit(dynamical_matrices_fn)
actuation_mapping_fn = partial(
    auxiliary_fns["actuation_mapping_fn"],
    forward_kinematics_fn,
    auxiliary_fns["jacobian_fn"],
)
print("A=", actuation_mapping_fn(params, B_xi, jnp.zeros((2 * num_segments,))))


def sweep_actuation_mapping():
    # evaluate the actuation matrix for a straight backbone
    q = jnp.zeros((2 * num_segments,))
    A = actuation_mapping_fn(params, B_xi, q)
    print("Evaluating actuation matrix for straight backbone: A =\n", A)

    kappa_be_pts = jnp.linspace(-3*jnp.pi, 3*jnp.pi, 500)
    sigma_ax_pts = jnp.zeros_like(kappa_be_pts)
    q_pts = jnp.stack([kappa_be_pts, sigma_ax_pts], axis=-1)
    A_pts = vmap(actuation_mapping_fn, in_axes=(None, None, 0))(params, B_xi, q_pts)
    # mark the points that are not controllable as the u1 and u2 terms share the same sign
    non_controllable_selector = A_pts[..., 0, 0] * A_pts[..., 0, 1] >= 0.0
    non_controllable_indices = jnp.where(non_controllable_selector)[0]
    non_controllable_boundary_indices = jnp.where(non_controllable_selector[:-1] != non_controllable_selector[1:])[0]
    # plot the mapping on the bending strain for various bending strains
    fig, ax = plt.subplots(num="pneumatic_planar_pcs_actuation_mapping_bending_torque_vs_bending_strain")
    plt.title(r"Actuation mapping from $u$ to $\tau_\mathrm{be}$")
    # # shade the region where the actuation mapping is negative as we are not able to bend the robot further
    # ax.axhspan(A_pts[:, 0, 0:2].min(), 0.0, facecolor='red', alpha=0.2)
    for idx in non_controllable_indices:
        ax.axvspan(kappa_be_pts[idx], kappa_be_pts[idx+1], facecolor='red', alpha=0.2)
    ax.plot(kappa_be_pts, A_pts[:, 0, 0], linewidth=2, label=r"$\frac{\partial \tau_\mathrm{be}}{\partial u_1}$")
    ax.plot(kappa_be_pts, A_pts[:, 0, 1], linewidth=2, label=r"$\frac{\partial \tau_\mathrm{ax}}{\partial u_2}$")
    ax.set_xlabel(r"$\kappa_\mathrm{be}$ [rad/m]")
    ax.set_ylabel(r"$\frac{\partial \tau_\mathrm{be}}{\partial u_1}$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # plot the actuation mapping of u1 vs. the bending strain for various segment radii
    r_pts = jnp.linspace(1e-2, 1e-1, 10)
    r_cham_out_pts = r_pts - 2e-3
    fig, ax = plt.subplots(num="pneumatic_planar_pcs_actuation_mapping_bending_torque_vs_bending_strain_4segment_radii")
    for r, r_cham_out in zip(r_pts, r_cham_out_pts):
        _params = params.copy()
        _params["r"] = r * jnp.ones((num_segments,))
        _params["r_cham_out"] = r_cham_out * jnp.ones((num_segments,))
        A_pts = vmap(actuation_mapping_fn, in_axes=(None, None, 0))(_params, B_xi, q_pts)
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
    A_pts = vmap(actuation_mapping_fn, in_axes=(None, None, 0))(params, B_xi, q_pts)
    # reshape A_pts to match the grid shape
    A_grid = A_pts.reshape(kappa_be_grid.shape[:2] + A_pts.shape[-2:])

    # plot the mapping on the bending strain
    fig, ax = plt.subplots(num="pneumatic_planar_pcs_actuation_mapping_bending_torque_vs_axial_vs_bending_strain")
    plt.title(r"Actuation mapping from $u_1$ to $\tau_\mathrm{be}$")
    # contourf plot
    c = ax.contourf(kappa_be_grid, sigma_ax_grid, A_grid[..., 0, 0], levels=100)
    fig.colorbar(c, ax=ax, label=r"$\frac{\partial \tau_\mathrm{be}}{\partial u_1}$")
    # contour plot
    ax.contour(kappa_be_grid, sigma_ax_grid, A_grid[..., 0, 0], levels=20, colors="k", linewidths=0.5)
    ax.set_xlabel(r"$\kappa_\mathrm{be}$ [rad/m]")
    ax.set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
    plt.tight_layout()
    plt.show()

    # plot the mapping on the axial strain
    fig, ax = plt.subplots(num="pneumatic_planar_pcs_actuation_mapping_axial_torque_vs_axial_vs_bending_strain")
    plt.title(r"Actuation mapping from $u_1$ to $\tau_\mathrm{ax}$")
    # contourf plot
    c = ax.contourf(kappa_be_grid, sigma_ax_grid, A_grid[..., 1, 0], levels=100)
    fig.colorbar(c, ax=ax, label=r"$\frac{\partial \tau_\mathrm{ax}}{\partial u_1}$")
    # contour plot
    ax.contour(kappa_be_grid, sigma_ax_grid, A_grid[..., 1, 0], levels=20, colors="k", linewidths=0.5)
    ax.set_xlabel(r"$\kappa_\mathrm{be}$ [rad/m]")
    ax.set_ylabel(r"$\sigma_\mathrm{ax}$ [-]")
    plt.tight_layout()
    plt.show()


def simulate_robot():
    # define initial configuration
    q0 = jnp.repeat(jnp.array([-5.0 * jnp.pi, -0.2])[None, :], num_segments, axis=0).flatten()
    # number of generalized coordinates
    n_q = q0.shape[0]

    # set simulation parameters
    dt = 1e-3  # time step
    sim_dt = 5e-5  # simulation time step
    ts = jnp.arange(0.0, 7.0, dt)  # time steps

    x0 = jnp.concatenate([q0, jnp.zeros_like(q0)])  # initial condition
    u = jnp.array([1.2e3, 0e0])  # control inputs (pressures in the right and left chambers)

    ode_fn = ode_factory(dynamical_matrices_fn, params, u)
    term = ODETerm(ode_fn)

    sol = diffeqsolve(
        term,
        solver=Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=sim_dt,
        y0=x0,
        max_steps=None,
        saveat=SaveAt(ts=ts),
    )

    print("sol.ys =\n", sol.ys)
    # the evolution of the generalized coordinates
    q_ts = sol.ys[:, :n_q]
    # the evolution of the generalized velocities
    qd_ts = sol.ys[:, n_q:]

    # evaluate the forward kinematics along the trajectory
    chi_ee_ts = vmap(forward_kinematics_fn, in_axes=(None, 0, None))(
        params, q_ts, jnp.array([jnp.sum(params["l"])])
    )
    # plot the configuration vs time
    plt.figure()
    for segment_idx in range(num_segments):
        plt.plot(
            ts, q_ts[:, 2 * segment_idx + 0],
            label=r"$\kappa_\mathrm{be," + str(segment_idx + 1) + "}$ [rad/m]"
        )
        plt.plot(
            ts, q_ts[:, 2 * segment_idx + 1],
            label=r"$\sigma_\mathrm{ax," + str(segment_idx + 1) + "}$ [-]"
        )
    plt.xlabel("Time [s]")
    plt.ylabel("Configuration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # plot end-effector position vs time
    plt.figure()
    plt.plot(ts, chi_ee_ts[:, 0], label="x")
    plt.plot(ts, chi_ee_ts[:, 1], label="y")
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector Position [m]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()
    # plot the end-effector position in the x-y plane as a scatter plot with the time as the color
    plt.figure()
    plt.scatter(chi_ee_ts[:, 0], chi_ee_ts[:, 1], c=ts, cmap="viridis")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("End-effector x [m]")
    plt.ylabel("End-effector y [m]")
    plt.colorbar(label="Time [s]")
    plt.tight_layout()
    plt.show()
    # plt.figure()
    # plt.plot(chi_ee_ts[:, 0], chi_ee_ts[:, 1])
    # plt.axis("equal")
    # plt.grid(True)
    # plt.xlabel("End-effector x [m]")
    # plt.ylabel("End-effector y [m]")
    # plt.tight_layout()
    # plt.show()

    # plot the energy along the trajectory
    kinetic_energy_fn_vmapped = vmap(
        partial(auxiliary_fns["kinetic_energy_fn"], params)
    )
    potential_energy_fn_vmapped = vmap(
        partial(auxiliary_fns["potential_energy_fn"], params)
    )
    U_ts = potential_energy_fn_vmapped(q_ts)
    T_ts = kinetic_energy_fn_vmapped(q_ts, qd_ts)
    plt.figure()
    plt.plot(ts, U_ts, label="Potential energy")
    plt.plot(ts, T_ts, label="Kinetic energy")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [J]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sweep_actuation_mapping()
    simulate_robot()
