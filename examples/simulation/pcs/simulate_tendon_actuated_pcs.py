from diffrax import Tsit5
from functools import partial
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import numpy as np

from soromox.rendering import MatplotlibRenderer
from soromox.systems.tendon_actuated_pcs import TendonActuatedPCS
from soromox.systems.system_state import SystemState

jax.config.update("jax_enable_x64", True)  # double precision

jnp.set_printoptions(precision=4, suppress=True)


def animate_robot_tendons_matplotlib(
    robot: TendonActuatedPCS,
    t_list: jnp.ndarray,
    q_list: jnp.ndarray,
    num_points: int = 50,
    interval: int = 50,
    mode: str = "slider",
    show: bool = True,
):
    """Animate robot with tendon visualization.

    This is a specialized animation function that shows both the backbone
    and the active tendons. For backbone-only visualization, use MatplotlibRenderer.
    """
    if mode not in ("slider", "animation"):
        raise ValueError("mode must be 'slider' or 'animation'")

    # Set up FK functions
    batched_fk_backbone = jax.vmap(robot.forward_kinematics, in_axes=(None, 0))
    batched_fk_tendons = jax.vmap(
        robot.forward_kinematics_active_tendons, in_axes=(None, 0), out_axes=1
    )
    L_max = float(jnp.sum(robot.L))
    s_ps = jnp.linspace(0, L_max, num_points)

    width = L_max * 3
    backbone_lw = 4

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])

    # Precompute all trajectories
    def compute_curves(q):
        backbone = batched_fk_backbone(q, s_ps)[:, :3, 3]
        tendons = batched_fk_tendons(q, s_ps)
        return backbone, tendons

    all_curves = jax.vmap(compute_curves)(q_list)
    all_robot_curves = np.array(all_curves[0])  # (T, num_points, 3)
    all_tendon_curves = np.array(all_curves[1])  # (T, n_actuators, num_points, 3)

    # Initialize line objects
    (line_robot,) = ax.plot([], [], [], lw=backbone_lw, color="blue", alpha=0.45)
    lines = [line_robot]
    for _ in range(robot.num_actuators):
        (ll,) = ax.plot([], [], [], lw=2, color="red")
        lines.append(ll)

    ax.set_xlim(-width / 2, width / 2)
    ax.set_ylim(-width / 2, width / 2)
    ax.set_zlim(0, width)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    if mode == "animation":
        title_text = ax.set_title("t = 0.00 s")

        def init():
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            title_text.set_text("t = 0.00 s")
            return lines + [title_text]

        def update(frame_idx):
            curve_r = all_robot_curves[frame_idx]
            curves_t = all_tendon_curves[frame_idx]

            lines[0].set_data(curve_r[:, 0], curve_r[:, 1])
            lines[0].set_3d_properties(curve_r[:, 2])

            for i in range(robot.num_actuators):
                lines[i + 1].set_data(curves_t[i, :, 0], curves_t[i, :, 1])
                lines[i + 1].set_3d_properties(curves_t[i, :, 2])

            title_text.set_text(f"t = {t_list[frame_idx]:.2f} s")
            return lines + [title_text]

        ani = FuncAnimation(
            fig, update, frames=len(q_list), init_func=init, blit=False, interval=interval
        )

        if show:
            plt.show()
        plt.close(fig)
        return ani

    else:  # slider mode
        def update_plot(frame_idx):
            frame_idx = int(frame_idx)
            curve_r = all_robot_curves[frame_idx]
            curves_t = all_tendon_curves[frame_idx]

            ax.set_title(f"t = {t_list[frame_idx]:.2f} s")

            lines[0].set_data(curve_r[:, 0], curve_r[:, 1])
            lines[0].set_3d_properties(curve_r[:, 2])

            for i in range(robot.num_actuators):
                lines[i + 1].set_data(curves_t[i, :, 0], curves_t[i, :, 1])
                lines[i + 1].set_3d_properties(curves_t[i, :, 2])

            fig.canvas.draw_idle()

        slider_widget = Slider(
            ax=ax_slider, label="Frame", valmin=0, valmax=len(t_list) - 1, valinit=0, valstep=1
        )
        slider_widget.on_changed(update_plot)
        update_plot(0)

        if show:
            plt.show()
        plt.close(fig)
        return None


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
    active_tendon_routing_params = {
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
    passive_tendon_routing_params = {
        "ry": 2e-2
        * jnp.array(
            [0.0]
        ),  # y-coordinate of the pulling point of the tendons [m]
        "rz": 2e-2
        * jnp.array([1.0]),  # z-coordinate of the pulling point of the tendons [m]
        "my": jnp.array(
            [0.0]
        ),  # slope coefficient in the x-y plane of the tendons [-]
        "mz": jnp.array(
            [0.0]
        ),  # slope coefficient in the x-z plane of the tendons [-]
        "idx_seg_att": jnp.array(
            [1]
        ),  # length of the tendons = x-coordinate of the attachment points [m]
    }
    passive_tendon_params = {
        "k_pt": jnp.array([0.6]),
        "d_pt": jnp.array([0.1]),
        "l_pt0": jnp.array([-0.3]),
    }

    # ======================================================
    # Robot initialization
    # ======================================================
    robot = TendonActuatedPCS(
        num_segments=num_segments,
        params=params,
        active_tendon_routing_params=active_tendon_routing_params,
        passive_tendon_routing_params=passive_tendon_routing_params,
        passive_tendon_params=passive_tendon_params
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
    u = jnp.array([-0.5, -1.0])
    print("u =\n", u)

    # Actuation matrix
    A = robot.actuation_matrix(q0)
    print("A =\n", A.shape)
    print(A)

    # Coupling matrix of the passive tendons
    P = robot.jacobian_passive_tendon(q0).T
    print("P =\n", P.shape)
    print(P)

    # Active tendon lengths
    la = robot.active_tendon_length(q0)
    print("la =\n", la.shape)
    print(la)

    # Passive tendon lengths
    lp = robot.passive_tendon_length(q0)
    print("lp =\n", lp.shape)
    print(lp)

    # Active tendons' position
    ta_s = robot.forward_kinematics_active_tendons(q0, robot.L_cum[-1])
    print("ta_s =\n", ta_s.shape)
    print(ta_s)

    # Passive tendons' position
    tp_s = robot.forward_kinematics_passive_tendons(q0, robot.L_cum[-1])
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

    # =====================================================
    # Plot the robot configuration upon time
    # =====================================================
    animate_robot_tendons_matplotlib(
        robot,
        t_list=ts,
        q_list=q_ts,
        num_points=50,
        interval=100,
        mode="slider",
    )
