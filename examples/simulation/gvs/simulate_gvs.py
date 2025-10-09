from functools import partial
from IPython.display import HTML
import jax
from jax import Array
import jax.numpy as jnp
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as onp
from typing import List

jax.config.update("jax_enable_x64", True)  # double precision
from soromox.systems.gvs import *

jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)


def draw_robot_curve(
    robot: GVS,
    q: Array,
    num_points: int = 50,
):
    batched_forward_kinematics = jax.vmap(robot.forward_kinematics, in_axes=(None, 0))
    L_max = jnp.sum(robot.V_L)

    s_ps = jnp.linspace(0, L_max, num_points)
    g_ps = batched_forward_kinematics(q, s_ps)[:, :3, 3]

    # q_gathered = robot._min_size_gathered(q)
    # V_g = robot._forward_kinematics_gauss(q_gathered)

    # g_ps = jnp.concatenate(V_g[:, :, :-1, -1:], axis=0)

    curve = onp.array(g_ps, dtype=onp.float64)
    return curve  # (N, 3)


def animate_robot_matplotlib(
    robot: GVS,
    t_list: Array,  # shape (T,)
    q_list: Array,  # shape (T, DOF)
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

    width = jnp.linalg.norm(robot.V_L) * 3
    height = width

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])  # [left, bottom, width, height]

    if animation:
        (line,) = ax.plot([], [], [], lw=4, color="blue")
        ax.set_xlim(-width / 2, width / 2)
        ax.set_ylim(-width / 2, width / 2)
        ax.set_zlim(0, height)
        title_text = ax.set_title("t = 0.00 s")

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            title_text.set_text("t = 0.00 s")
            return line, title_text

        def update(frame_idx):
            q = q_list[frame_idx]
            t = t_list[frame_idx]
            curve = draw_robot_curve(robot, q)
            line.set_data(curve[:, 0], curve[:, 1])
            line.set_3d_properties(curve[:, 2])
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
            ax.set_ylim(-width / 2, width / 2)
            ax.set_zlim(0, height)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_zlabel("Z [m]")
            ax.set_title(f"t = {t_list[frame_idx]:.2f} s")
            q = q_list[frame_idx]
            curve = draw_robot_curve(robot, q)
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], lw=4, color="blue")
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


if __name__ == "__main__":
    # Define model inputs
    List_links: List[LinkAttributes] = []
    List_joints: List[JointAttributes] = []
    List_basis: List[BasisAttributes] = []
    List_nGauss: List[int] = []

    link1 = LinkAttributes(
        section="Circular", E=1e6, nu=0.5, rho=1000, eta=1e4, L=0.3, r_i=0.03, r_f=0.03
    )
    List_links.append(link1)
    joint1 = JointAttributes(jointtype="Fixed")
    List_joints.append(joint1)
    basis1 = BasisAttributes(
        basistype="Legendre",
        Bdof=[0, 1, 1, 0, 0, 0],
        Bodr=[0, 0, 0, 0, 0, 0],
        xi_ref=[0, 0, 0, 1, 0, 0],
    )
    List_basis.append(basis1)
    List_nGauss.append(5)  # Number of Gauss points for the first link

    link2 = LinkAttributes(
        section="Circular", E=1e6, nu=0.5, rho=1000, eta=1e4, L=0.3, r_i=0.03, r_f=0.03
    )
    List_links.append(link2)
    joint2 = JointAttributes(jointtype="Fixed")
    # joint2 = JointAttributes(jointtype='Revolute', axis='z')
    List_joints.append(joint2)
    basis2 = BasisAttributes(
        basistype="Monomial",
        Bdof=[1, 1, 0, 0, 0, 0],
        Bodr=[0, 0, 0, 0, 0, 0],
        xi_ref=[0, 0, 0, 1, 0, 0],
    )
    List_basis.append(basis2)
    List_nGauss.append(6)  # Number of Gauss points for the second link

    # link3 = LinkAttributes(
    #     section='Elliptical',  # Section type
    #     E=1e7,                # Young's modulus in Pascals
    #     nu=0.4,               # Poisson's ratio [-1, 0.5]
    #     rho=1050,             # Density [kg/m^3]
    #     eta=1e4,              # Damping coefficient
    #     L=0.3,                # Length in meters
    #     a_i=0.04,             # Initial semi-major axis in meters
    #     a_f=0.04,             # Final semi-major axis in meters
    #     b_i=0.02,             # Initial semi-minor axis in meters
    #     b_f=0.02              # Final semi-minor axis in meters
    # )
    # List_links.append(link3)
    # joint3 = JointAttributes(
    #     jointtype='Revolute',  # Prismatic joint
    #     axis='z',               # Axis of translation
    # )
    # List_joints.append(joint3)
    # basis3 = BasisAttributes(
    #     basistype='Chebychev',       # Type of basis
    #     Bdof=[0, 1, 0, 1, 0, 0],    # Degrees of freedom for each deformation type
    #     Bodr=[0, 0, 0, 0, 0, 0],    # Order of basis functions for each deformation type
    #     xi_ref=[0, 0, 0, 1, 0, 0], # Reference strain values as vector
    # )
    # List_basis.append(basis3)
    # List_nGauss.append(5)  # Number of Gauss points for the third link

    # ======================================================
    # Robot initialization
    # ======================================================
    robot = GVS(
        links_list=List_links,
        joints_list=List_joints,
        basis_list=List_basis,
        n_gauss_list=List_nGauss,
        gravity_vector=[0, 0, 9.81],
    )

    print(f"System initialized with {robot.num_segments} segments")
    print(f"max DOFs: {robot.max_dof}, max Gauss points: {robot.max_nGauss}.")
    print(f"Total DOFs: {robot.dof_tot_system} (chosen), {robot.dof_tot_max} (real)")

    # =====================================================
    # Simulation upon time
    # =====================================================
    # Initial configuration
    q0 = jnp.zeros(robot.dof_tot_system)
    # Initial velocities
    qd0 = jnp.zeros_like(q0)

    # Plot the initial configuration
    curve = draw_robot_curve(robot, q0)

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "3d"})
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], lw=4, color="blue")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Initial configuration")
    ax.axis("equal")
    plt.show()

    # Actuation parameters
    u = jnp.zeros_like(q0)

    # Simulation time parameters
    t0 = 0.0
    t1 = 2.0
    dt = 1e-4
    skip_step = 100  # how many time steps to skip in between video frames

    ts, q_ts, qd_ts = robot.resolve_upon_time(
        q0=q0,
        qd0=qd0,
        u=u,
        t0=t0,
        t1=t1,
        dt=dt,
        skip_steps=skip_step,
        max_steps=None,
    )
    print(f"Simulation completed with {len(ts)} time steps.")

    # =====================================================
    # End-effector position upon time
    # =====================================================
    forward_kinematics_end_effector = jax.jit(
        partial(
            robot.forward_kinematics,
            s=jnp.sum(robot.V_L),  # end-effector position
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
    # Plot the robot configuration upon time
    # =====================================================
    animate_robot_matplotlib(
        robot,
        t_list=ts,  # shape (T,)
        q_list=q_ts,  # shape (T, DOF)
        interval=100,  # ms
        slider=True,
    )
