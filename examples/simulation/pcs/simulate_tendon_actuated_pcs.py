from diffrax import Tsit5

import jax
from jax import Array
import jax.numpy as jnp

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

import numpy as onp
from typing import Callable, Dict
from functools import partial

from soromox.systems.tendon_actuated_pcs import TendonActuatedPCS

jax.config.update("jax_enable_x64", True)  # double precision


jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)
jnp.set_printoptions(precision=4, suppress=True)

def draw_robot_curve(
    batched_forward_kinematics: Callable,
    L_max: float,
    q: Array,
    num_points: int = 50,
):
    s_ps = jnp.linspace(0, L_max, num_points)
    g_ps = batched_forward_kinematics(q, s_ps)[:, :3, 3]

    curve = jnp.array(g_ps, dtype=jnp.float64)
    return curve  # (N, 3)


def draw_tendon_curves(
    batched_forward_kinematics_tendons: Callable,
    L_max: float,
    q: Array,
    num_points: int = 50,
):
    s_ps = jnp.linspace(0, L_max, num_points)
    ps = batched_forward_kinematics_tendons(q, s_ps)

    curves = jnp.array(ps, dtype=jnp.float64)
    return curves  # (n_tendons, N, 3)


def animate_robot_tendons_matplotlib(
    robot,
    t_list: jnp.ndarray,  # shape (T,)
    q_list: jnp.ndarray,  # shape (T, DOF)
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

    batched_forward_kinematics = jax.vmap(robot.forward_kinematics, in_axes=(None, 0))
    batched_forward_kinematics_tendons = jax.vmap(
        robot.forward_kinematics_tendons, in_axes=(None, 0), out_axes=1
    )
    L_max = jnp.sum(robot.L)

    width = jnp.linalg.norm(robot.L) * 3
    height = width

    # backbone linewidth
    backbone_lw = 4  # float(jnp.max(robot.r) * 1.25e3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])  # [left, bottom, width, height]

    # ---------------------------------------------------
    # Precompute all trajectories
    # ---------------------------------------------------
    batched_robot_curve = jax.vmap(
        lambda q: draw_robot_curve(batched_forward_kinematics, L_max, q, num_points)
    )
    batched_tendon_curve = jax.vmap(
        lambda q: draw_tendon_curves(
            batched_forward_kinematics_tendons, L_max, q, num_points
        )
    )

    all_robot_curves = batched_robot_curve(q_list)  # (T, num_points, 3)
    all_tendon_curves = batched_tendon_curve(q_list)  # (T, n_actuators, num_points, 3)

    # ---------------------------------------------------
    # Animation branch
    # ---------------------------------------------------
    if animation:
        # initialize line objects once
        (line_robot,) = ax.plot([], [], [], lw=backbone_lw, color="blue", alpha=0.45)
        lines = [line_robot]
        for _ in range(robot.num_actuators):
            (ll,) = ax.plot([], [], [], lw=2, color="red")
            lines.append(ll)

        ax.set_xlim(-width / 2, width / 2)
        ax.set_ylim(-width / 2, width / 2)
        ax.set_zlim(0, height)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        title_text = ax.set_title("t = 0.00 s")

        def init():
            for l in lines:
                l.set_data([], [])
                l.set_3d_properties([])
            title_text.set_text("t = 0.00 s")
            return lines + [title_text]

        def update(frame_idx):
            curve_r = all_robot_curves[frame_idx]  # (num_points, 3)
            curves_t = all_tendon_curves[frame_idx]  # (n_actuators, num_points, 3)

            # update robot backbone
            lines[0].set_data(curve_r[:, 0], curve_r[:, 1])
            lines[0].set_3d_properties(curve_r[:, 2])

            # update tendons
            for i in range(robot.num_actuators):
                lines[i + 1].set_data(curves_t[i, :, 0], curves_t[i, :, 1])
                lines[i + 1].set_3d_properties(curves_t[i, :, 2])

            title_text.set_text(f"t = {t_list[frame_idx]:.2f} s")
            return lines + [title_text]

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

    # ---------------------------------------------------
    # Slider branch
    # ---------------------------------------------------
    elif slider:
        (line_robot,) = ax.plot([], [], [], lw=backbone_lw, color="blue", alpha=0.45)
        lines = [line_robot]
        for _ in range(robot.num_actuators):
            (ll,) = ax.plot([], [], [], lw=2, color="red")
            lines.append(ll)

        def update_plot(frame_idx):
            curve_r = all_robot_curves[frame_idx]
            curves_t = all_tendon_curves[frame_idx]

            ax.set_xlim(-width / 2, width / 2)
            ax.set_ylim(-width / 2, width / 2)
            ax.set_zlim(0, height)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_zlabel("Z [m]")
            ax.set_title(f"t = {t_list[frame_idx]:.2f} s")

            # robot backbone
            lines[0].set_data(curve_r[:, 0], curve_r[:, 1])
            lines[0].set_3d_properties(curve_r[:, 2])

            # tendons
            for i in range(robot.num_actuators):
                lines[i + 1].set_data(curves_t[i, :, 0], curves_t[i, :, 1])
                lines[i + 1].set_3d_properties(curves_t[i, :, 2])

            fig.canvas.draw_idle()

        slider_widget = Slider(
            ax=ax_slider,
            label="Frame",
            valmin=0,
            valmax=len(t_list) - 1,
            valinit=0,
            valstep=1,
        )
        slider_widget.on_changed(update_plot)

        update_plot(0)  # Initial frame

        if show:
            plt.show()
        plt.close(fig)

        return HTML(
            "Slider animation not implemented in HTML format. Use matplotlib directly to view the slider."
        )


if __name__ == "__main__":
    num_segments = 2
    rho = 1070 * jnp.ones(
        (num_segments,)
    )  # Volumetric density of Dragon Skin 20 [kg/m^3]
    params = {
        "p0": jnp.array([jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, 0.0]),  # Initial position and orientation
        "L": 1e-1 * jnp.ones((num_segments,)),  # default: 1e-1
        "r": 2e-2 * jnp.ones((num_segments,)),  # default: 2e-2
        "rho": rho,
        "g": jnp.array([0.0, 0.0, 9.81]),  # Gravity vector [m/s^2]
        "E": 2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
        "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
    }
    params["D"] = 1e-3 * jnp.diag(
        (
            jnp.repeat(
                jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), num_segments, axis=0
            )
            * params["L"][:, None]
        ).flatten()
    )
    tendon_routing_params = {
        "ry": 2e-2
        * jnp.array(
            [1.0, -1.0]
        ),  # y-coordinate of the pulling point of the tendons [m]
        "rz": 2e-2
        * jnp.array([0.0, 0.0]),  # z-coordinate of the pulling point of the tendons [m]
        "my": jnp.array(
            [0.0, 0.0]
        ),  # slope coefficient in the x-y plane of the tendons [-]
        "mz": jnp.array(
            [0.0, 0.0]
        ),  # slope coefficient in the x-z plane of the tendons [-]
        "idx_seg_att": jnp.array(
            [1, 0]
        ),  # length of the tendons = x-coordinate of the attachment points [m]
    }
    # tendon_routing_params = {
    #     'ry': jnp.array([0., -0.0139, 0.0139, -0.0139, 0., 0.0139]),
    #     'rz': jnp.array([0.016, -0.008, -0.008, 0.008, -0.016, 0.008]),
    #     'my': jnp.zeros(6),
    #     'mz': jnp.zeros(6),
    #     'idx_seg_att': jnp.array([0, 0, 0, 1, 1, 1]),
    # }

    # ======================================================
    # Robot initialization
    # ======================================================
    robot = TendonActuatedPCS(
        num_segments=num_segments,
        params=params,
        tendon_routing_params=tendon_routing_params,
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
    #q0 = jnp.array([0.9143,-0.0292,0.6006,-0.7162,-0.1565,0.8315,0.5844,0.9190,0.3115,-0.9286,0.6983,0.8680])

    # Initial velocities
    qd0 = jnp.zeros_like(q0)

    # Actuation parameters
    u = jnp.array([-1.0, 0.0])
    print("u =\n", u)

    # Actuation matrix
    A = robot.actuation_matrix(q0)
    print("A =\n", A.shape)
    print(A)

    # Tendons' position
    t_s = robot.forward_kinematics_tendons(q0, robot.L_cum[-1])
    print("t_s =\n", t_s.shape)
    print(t_s)
    
    # Simulation time parameters
    t0 = 0.0
    t1 = 2.0
    dt = 1e-4
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
    animate_robot_tendons_matplotlib(
        robot,
        t_list=ts,  # shape (T,)
        q_list=q_ts,  # shape (T, DOF)
        num_points=50,
        interval=100,  # ms
        animation=False,
        slider=True,
    )  # '''
