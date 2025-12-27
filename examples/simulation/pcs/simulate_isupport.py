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
import numpy as onp
from diffrax import Tsit5
from IPython.display import HTML
from jax import Array
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

jax.config.update("jax_enable_x64", True)  # double precision
from soromox.systems import ISupport, SystemState

jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)


def draw_robot_curve(robot: ISupport, L_max: float, q: Array, num_points: int = 50):
    s_ps = jnp.linspace(0, L_max, num_points)
    g_ps = robot.forward_kinematics_batched(q, s_ps)[:, :3, 3]

    curve = onp.array(g_ps, dtype=onp.float64)
    return curve  # (N, 3)


def animate_robot_matplotlib(
    robot: ISupport,
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

    L_max = jnp.sum(robot.L)

    width = jnp.linalg.norm(robot.L) * 1.5
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
            curve = draw_robot_curve(robot, L_max, q, num_points)
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
            curve = draw_robot_curve(robot, L_max, q, num_points)
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
    num_segments = 1

    # Elastic modulus and poisson ratio
    E = 1.6464 * 1e6  # Elastic modulus [Pa]
    poisson_ratio = 0.5
    G = E / (
        2 * (1 + poisson_ratio)
    )  # Shear modulus from elastic modulus and poisson ratio

    params = {
        "p0": jnp.array(
            [jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, 0.0]
        ),  # Initial position and orientation
        "L": 190 * 1e-3 * jnp.ones((num_segments,)),
        "r": 35.6 * 1e-3 * jnp.ones((num_segments,)),
        "rho": 1104
        * jnp.ones((num_segments,)),  # material density of TPU 80 A LF by BASF [kg/m^3]
        "g": jnp.array([0.0, 0.0, 9.81]),  # Gravity vector [m/s^2]
        "E": E * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
        "G": G * jnp.ones((num_segments,)),  # Shear modulus [Pa]
        "r_chamber_in": 6.39
        * 1e-3
        * jnp.ones(
            (num_segments,)
        ),  # inner radius of each segment's pneumatic chamber [m]
        "r_chamber_out": 7.79
        * 1e-3
        * jnp.ones(
            (num_segments,)
        ),  # outer radius of each segment's pneumatic chamber [m]
        "d_chamber": 20
        * 1e-3
        * jnp.ones(
            (num_segments,)
        ),  # radial distance of the center of the chambers from the centerline of the backbone [m]
    }

    # damping coefficient
    # these values are from the paper but they seem way too large
    # gamma_t = 806  # translational damping constant [1/s]
    # gamma_r = 1.9416 * 10**(-4)  # rotational damping constant [m^2/s]
    gamma_t = 806 * 1e-3  # translational damping constant [1/s]
    gamma_r = 1.0 * 1e-4  # rotational damping constant [m^2/s]
    params["D"] = jnp.diag(
        (
            jnp.repeat(
                jnp.array([[gamma_r, gamma_r, gamma_r, gamma_t, gamma_t, gamma_t]]),
                num_segments,
                axis=0,
            )
        ).flatten()
    )

    # ======================================================
    # Robot initialization
    # ======================================================
    robot = ISupport(
        num_segments=num_segments,
        params=params,
    )

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
    u = jnp.array([2e-1, 0.0, 0.0]) * 1e5

    # Simulation time parameters
    t0 = 0.0
    t1 = 2.0
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
    animate_robot_matplotlib(
        robot,
        t_list=ts,  # shape (T,)
        q_list=q_ts,  # shape (T, DOF)
        num_points=50,
        interval=100,  # ms
        slider=True,
    )
