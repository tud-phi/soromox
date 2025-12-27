import cv2
from functools import partial
import jax

from diffrax import Tsit5
from jax import Array
from jax import numpy as jnp
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as onp
from pathlib import Path


jax.config.update("jax_enable_x64", True)  # double precision
from soromox.rendering.animation import animate_cv2
from soromox.rendering.planar_pcs.opencv_renderer import render_planar_pcs
from soromox.systems import TendonActuatedPlanarPCS, SystemState


videos_dir = Path("videos")
videos_dir.mkdir(parents=True, exist_ok=True)


def draw_robot(
    robot: TendonActuatedPlanarPCS,
    q: Array,
    num_points: int = 50,
):
    L_max = jnp.sum(robot.L)

    s_ps = jnp.linspace(0, L_max, num_points)
    chi_ps = robot.forward_kinematics_batched(q, s_ps)

    curve = onp.array(chi_ps[:, 1:], dtype=onp.float64)

    return curve  # (N, 2)


def animate_robot_matplotlib(
    robot: TendonActuatedPlanarPCS,
    t_ts: Array,  # shape (T,)
    q_ts: Array,  # shape (T, DOF)
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
    if slider:
        ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])  # [left, bottom, width, height]

    # Base
    def draw_base(ax, robot, L=robot.L[0] / 2):
        angle1 = robot.th0 - jnp.pi / 2
        angle2 = robot.th0 + jnp.pi / 2
        x1, y1 = L * jnp.cos(angle1), L * jnp.sin(angle1)
        x2, y2 = L * jnp.cos(angle2), L * jnp.sin(angle2)
        ax.plot([x1, x2], [y1, y2], color="black", linestyle="-", linewidth=2)

    if animation:
        (line,) = ax.plot([], [], lw=10, color="blue")
        ax.set_xlim(-width / 2, width / 2)
        ax.set_ylim(0, height)
        title_text = ax.set_title("t = 0.00 s")

        def init():
            line.set_data([], [])
            title_text.set_text("t = 0.00 s")
            return line, title_text

        def update(frame_idx):
            q = q_ts[frame_idx]
            t = t_ts[frame_idx]
            draw_base(ax, robot, L=0.1)
            curve = draw_robot(robot, q, num_points)
            line.set_data(curve[:, 0], curve[:, 1])
            title_text.set_text(f"t = {t:.2f} s")
            return line, title_text

        ani = FuncAnimation(
            fig,
            update,
            frames=len(q_ts),
            init_func=init,
            blit=False,
            interval=interval,
        )

        if show:
            plt.show()

        # Save animation as video
        print("Saving animation as video...")
        ani.save(videos_dir / "tendon_actuated_planar_pcs_animation.mp4", writer="ffmpeg", dpi=200)

        plt.close(fig)
        return HTML(ani.to_jshtml())

    elif slider:

        def update_plot(frame_idx):
            ax.cla()  # Clear current axes
            ax.set_xlim(-width / 2, width / 2)
            ax.set_ylim(0, height)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_title(f"t = {t_ts[frame_idx]:.2f} s")
            draw_base(ax, robot, L=0.1)
            q = q_ts[frame_idx]
            curve = draw_robot(robot, q, num_points)
            ax.plot(curve[:, 0], curve[:, 1], lw=4, color="blue")
            fig.canvas.draw_idle()

        # Create slider
        slider = Slider(
            ax=ax_slider,
            label="Frame",
            valmin=0,
            valmax=len(t_ts) - 1,
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
    num_segments = 2  # number of segments in the robot
    rho = 1070 * jnp.ones(
        (num_segments,)
    )  # Volumetric density of Dragon Skin 20 [kg/m^3]
    params = {
        "th0": jnp.array(jnp.pi / 2),  # initial orientation angle [rad]
        "L": 1e-1 * jnp.ones((num_segments,)),
        "r": 2e-2 * jnp.ones((num_segments,)),
        "rho": rho,
        "g": 0 * jnp.array([0.0, 9.81]),  # gravitational acceleration [m/s^2] UP!
        "E": 5e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
        "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
        "d": 2e-2
        * jnp.array([[1.0, -1.0]]).repeat(
            num_segments, axis=0
        ),  # distance of tendons from the central axis [m]
    }
    params["D"] = 1e-3 * jnp.diag(
        (
            jnp.repeat(jnp.array([[1e0, 1e3, 1e3]]), num_segments, axis=0)
            * params["L"][:, None]
        ).flatten()
    )

    # activate all strains (i.e. bending, shear, and axial)
    strain_selector = jnp.ones((3 * num_segments,), dtype=bool)
    # actuation selector for the segments
    segment_actuation_selector = jnp.ones((num_segments,), dtype=bool)

    # ======================================================
    # Robot initialization
    # ======================================================
    robot = TendonActuatedPlanarPCS(
        num_segments=num_segments,
        params=params,
        strain_selector=strain_selector,
        segment_actuation_selector=segment_actuation_selector,
    )

    w, h = 400, 400  # image height and width
    rendering_fn = partial(
        render_planar_pcs,
        robot,
        width=w,
        height=h,
        length_scale=2.0,
        num_points=100,
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
    q0 = jax.random.normal(key, shape=int(robot.num_active_strains)) * jnp.repeat(
        jnp.array([5.0 * jnp.pi, 0.1, 0.05])[None, :], num_segments, axis=0
    ).flatten()
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
    base_tension = jnp.array(-0.3, dtype=q0.dtype)
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
    u = jnp.take(
        tendon_pattern,
        robot.segment_indices_to_actuate,
        axis=0,
    ).reshape(-1)
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
    animate_robot_matplotlib(
        robot=robot,
        t_ts=ts,  # shape (T,)
        q_ts=q_ts,  # shape (T, DOF)
        num_points=50,
        interval=100,  # ms
        slider=True,
    )
    animate_cv2(
        rendering_fn=rendering_fn,
        t_ts=onp.array(ts),
        q_ts=onp.array(q_ts),
        filepath=videos_dir / "tendon_actuated_planar_pcs_cv2.mp4",
        width=w,
        height=h,
        speed_up=1.0,
        rgb_to_bgr=False,
    )
