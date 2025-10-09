from functools import partial
import jax

from diffrax import Tsit5
from jax import Array
from jax import numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
import numpy as onp

from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from matplotlib.widgets import Slider

jax.config.update("jax_enable_x64", True)  # double precision
from soromox.systems.pneumatic_actuated_planar_pcs import (
    PneumaticActuatedPlanarPCS,
)


def draw_robot(
    robot: PneumaticActuatedPlanarPCS,
    q: Array,
    num_points: int = 50,
):
    batched_forward_kinematics = jax.vmap(
        robot.forward_kinematics, in_axes=(None, 0), out_axes=-1
    )
    L_max = jnp.sum(robot.L)

    s_ps = jnp.linspace(0, L_max, num_points)
    chi_ps = batched_forward_kinematics(q, s_ps)

    curve = onp.array(chi_ps[1:, :], dtype=onp.float64).T

    return curve  # (N, 2)


def animate_robot_matplotlib(
    robot: PneumaticActuatedPlanarPCS,
    t_list: Array,  # shape (T,)
    q_list: Array,  # shape (T, DOF)
    num_points: int = 50,
    interval: int = 50,
    slider: bool = None,
    animation: bool = None,
    show: bool = True,
):
    if slider is None and animation is None:
        raise ValueError("Either 'slider' or 'animation' must be set to True.")
    if animation and slider:
        raise ValueError(
            "Cannot use both animation and slider at the same time. Choose one."
        )

    width = jnp.linalg.norm(robot.L) * 3
    height = width

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])  # [left, bottom, width, height]

    # Base
    def draw_base(ax, robot, L=robot.L[0] / 2):
        angle1 = robot.th0 - jnp.pi / 2
        angle2 = robot.th0 + jnp.pi / 2
        x1, y1 = L * jnp.cos(angle1), L * jnp.sin(angle1)
        x2, y2 = L * jnp.cos(angle2), L * jnp.sin(angle2)
        ax.plot([x1, x2], [y1, y2], color="black", linestyle="-", linewidth=2)

    if animation:
        (line,) = ax.plot([], [], lw=4, color="blue")
        ax.set_xlim(-width / 2, width / 2)
        ax.set_ylim(0, height)
        title_text = ax.set_title("t = 0.00 s")

        def init():
            line.set_data([], [])
            title_text.set_text("t = 0.00 s")
            return line, title_text

        def update(frame_idx):
            q = q_list[frame_idx]
            t = t_list[frame_idx]
            draw_base(ax, robot, L=0.1)
            curve = draw_robot(robot, q, num_points)
            line.set_data(curve[:, 0], curve[:, 1])
            title_text.set_text(f"t = {t:.2f} s")
            return line, title_text

        ani = FuncAnimation(
            fig,
            update,
            frames=len(q_list),
            init_func=init,
            blit=False,
            interval=interval,
        )

        if show:
            plt.show()
        plt.close(fig)
        return HTML(ani.to_jshtml())

    elif slider:

        def update_plot(frame_idx):
            ax.cla()  # Clear current axes
            ax.set_xlim(-width / 2, width / 2)
            ax.set_ylim(0, height)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_title(f"t = {t_list[frame_idx]:.2f} s")
            draw_base(ax, robot, L=0.1)
            q = q_list[frame_idx]
            curve = draw_robot(robot, q, num_points)
            ax.plot(curve[:, 0], curve[:, 1], lw=4, color="blue")
            fig.canvas.draw_idle()

        # Create slider
        slider = Slider(
            ax=ax_slider,
            label="Frame",
            valmin=0,
            valmax=len(t_list) - 1,
            valinit=0,
            valstep=1,
        )
        slider.on_changed(update_plot)

        update_plot(0)  # Initial plot

        if show:
            plt.show()

        plt.close(fig)
        return HTML(
            "Slider animation not implemented in HTML format. Use matplotlib directly to view the slider."
        )  # Slider cannot be converted to HTML


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
        _params = {}
        _params["r"] = r * jnp.ones((robot.num_segments,))
        _params["r_chamber_out"] = r_chamber_out * jnp.ones((robot.num_segments,))
        updated_robot = robot.update_params(_params)
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
    params = {
        "th0": jnp.array(jnp.pi / 2),  # initial orientation angle [rad]
        "L": 1e-1 * jnp.ones((num_segments,)),
        "r": 2e-2 * jnp.ones((num_segments,)),
        "rho": rho,
        "g": jnp.array([0.0, 9.81]),  # gravity vector [m/s^2] UP!
        "E": 2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
        "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
        "r_chamber_in": 5e-3 * jnp.ones((num_segments,)),
        "r_chamber_out": 2e-2 - 2e-3 * jnp.ones((num_segments,)),
        "phi_chamber": jnp.pi / 2 * jnp.ones((num_segments,)),
    }
    params["D"] = 5e-4 * jnp.diag(
        (
            jnp.repeat(jnp.array([[1e0, 1e3, 1e3]]), num_segments, axis=0)
            * params["L"][:, None]
        ).flatten()
    )

    strain_selector = (
        jnp.array([True, True, False])[None, :].repeat(num_segments, axis=0).flatten()
    )  # Select bending and axial strains, no shear strain

    # ======================================================
    # Robot initialization
    # ======================================================
    robot = PneumaticActuatedPlanarPCS(
        num_segments=num_segments,
        params=params,
        strain_selector=strain_selector,
        chamber_cross_section_geometry="concentric"
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

    # Dessiner la configuration initiale
    curve = draw_robot(robot, q0, num_points=100)
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
    dt = 5e-5
    save_dt = 0.01

    # Solver
    solver = Tsit5()  # Runge-Kutta 5(4) method

    ts, q_ts, qd_ts = robot.resolve_upon_time(
        q0=q0,
        qd0=qd0,
        u=u,
        t0=t0,
        t1=t1,
        dt=dt,
        save_dt=save_dt,
        solver=solver,
        max_steps=None,
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
    animate_robot_matplotlib(
        robot=robot,
        t_list=ts,  # shape (T,)
        q_list=q_ts,  # shape (T, DOF)
        num_points=50,
        interval=100,  # ms
        slider=True,
    )
