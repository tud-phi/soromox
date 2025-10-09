import cv2  # importing cv2
import jax

jax.config.update("jax_enable_x64", True)  # double precision
from jax import Array, lax, vmap
from jax import numpy as jnp
import numpy as onp
from pathlib import Path
from typing import Callable, Dict

from functools import partial

import soromox
from soromox.systems.tendon_actuated_pendulum import TendonActuatedPendulum

import matplotlib.pyplot as plt


num_links = 2

params = {
    "m": jnp.array([10.0, 6.0]),
    "I": jnp.array([3.0, 2.0]),
    "L": jnp.array([2.0, 1.0]),
    "Lc": jnp.array([1.0, 0.5]),
    "g": jnp.array([0.0, -9.81]),
    "K": jnp.diag(jnp.array([1.5, 2.0]))
}
tendon_params = {
    "R_at": jnp.array([[0.2, -0.25]]),
    "R_pt": jnp.array([[-0.15, 0.3]]),
    "k_pt": jnp.array([2.75]),
    "l_pt0": jnp.array([0.5])
}

# define initial configuration
q0 = jnp.zeros((num_links,))
q0 = jnp.array([jnp.pi / 8, -jnp.pi / 4])
#q0 = jnp.array([-jnp.pi / 2, 0.0])

# set simulation parameters
dt = 1e-4  # time step
ts = jnp.arange(0.0, 5, dt)  # time steps
save_dt = 0.01

# video settings
video_width, video_height = 700, 700  # img height and width
video_path = Path("videos") / f"pendulum_nl-{num_links}.mp4"


def draw_robot(
    robot: TendonActuatedPendulum,
    q: Array,
    width: int,
    height: int,
) -> onp.ndarray:
    # Color cables
    colors_p = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]  # BGR

    # Plot in OpenCV
    h, w = height, width  # img height and width
    ppm = h / (2.5 * jnp.sum(params["L"]))  # pixel per meter
    robot_color = (0, 0, 0)  # black robot_color in BGR

    # Poses along the robot of shape (N, 3)
    chi_ls = robot.forward_kinematics_tips(q)
    N = chi_ls.shape[0]

    # Initialize image background and origin of the pixel frame
    img = 255 * onp.ones((w, h, 3), dtype=jnp.uint8)  # initialize background to white
    curve_og = onp.array([w // 2, h // 2], dtype=onp.int32)  # in x-y pixel coordinates

    # Transform robot poses to pixel coordinates. Should be of shape (N, 2)
    curve = onp.array((curve_og + chi_ls[:, 1:] * ppm), dtype=onp.int32)
    curve = onp.vstack([curve_og, curve])  # (N + 1, 2)

    # Invert the y pixel coordinate
    curve[:, 1] = h - curve[:, 1]
    for i in range(curve.shape[0] - 1):
        pts = onp.array([curve[i, :], curve[i + 1, :]]).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=False, color=colors_p[i], thickness=10)

    # Retrieve angles of the links
    angles = chi_ls[:, 0] + onp.pi / 2  # (N,)

    # Pulleys and cables
    if not onp.array_equal(onp.asarray(robot.R_at), onp.eye(N)):
        chi = onp.vstack([jnp.zeros(3), chi_ls])
        act = robot.A_at
        for i in range(act.shape[1]):
            cable_pts = []
            for j in range(act.shape[0]):
                cv2.circle(
                    img,
                    center=tuple(curve[j, :]),
                    radius=int(abs(act[j, i] * ppm)),
                    color=colors_p[i],
                    thickness=2,
                )
                idx = max(0, j - 1)
                cable_pt = chi[j, 1:] + act[j, i] * onp.array(
                    [onp.cos(angles[idx]), onp.sin(angles[idx])]
                )
                cable_pts.append(curve_og + cable_pt * ppm)
            cable_pts = onp.array(cable_pts, dtype=onp.int32)
            cable_pts[:, 1] = h - cable_pts[:, 1]
            cv2.polylines(
                img, [cable_pts], isClosed=False, color=colors_p[i], thickness=2
            )

    return img


if __name__ == "__main__":
    # Instantiate the pendulum model directly
    robot = TendonActuatedPendulum(params, tendon_params)

    # initialize velocities and actuation
    qd0 = jnp.zeros_like(q0)  # initial velocities for simulation
    u = jnp.array([0.0])  # torques (actuation)
    #u = jnp.zeros_like(q0)

    # compute the operational space matrices
    Lambda, mu, J, Jd, JB_inv = robot.operational_space_dynamical_matrices(
        q0, qd0, link_idx=1
    )
    print("Lambda:\n", Lambda)

    # call the forward dynamics
    yd = robot.forward_dynamics(ts[0], jnp.concatenate([q0, qd0]), (u,))
    print("yd0:\n", yd)

    # check tendons' contributions
    print("R_at =", robot.R_at.shape)
    print(robot.R_at)
    print("A_at =", robot.A_at.shape)
    print(robot.A_at)
    print("R_pt =", robot.R_pt.shape)
    print(robot.R_pt)
    print("A_pt =", robot.A_pt.shape)
    print(robot.A_pt)
    print("K_pt =", robot.K_pt.shape)
    print(robot.K_pt)
    print("D_pt =", robot.D_pt.shape)
    print(robot.D_pt)
    print("l_pt0 =", robot.l_pt0.shape)
    print(robot.l_pt0)

    # Integrate using the model's built-in solver
    ts_out, qs, qds = robot.resolve_upon_time(
        q0=q0,
        qd0=qd0,
        u=u,
        t0=ts[0],
        t1=ts[-1],
        dt=dt,
        save_dt=save_dt,
    )
    video_ts = ts_out
    print("Final configuration:\n", qs[-1])


    # =====================================================
    # Energy computation upon time
    # =====================================================
    U_ts_out = jax.vmap(jax.jit(partial(robot.potential_energy)))(qs)
    T_ts_out = jax.vmap(jax.jit(partial(robot.kinetic_energy)))(qs, qds)
    E_ts_out = jax.vmap(jax.jit(partial(robot.total_energy)))(qs, qds)

    plt.figure()
    plt.plot(ts_out, U_ts_out, label="Potential Energy")
    plt.plot(ts_out, T_ts_out, label="Kinetic Energy")
    plt.plot(ts_out, E_ts_out, label="Total Energy")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.legend()
    plt.title("Energy over Time")
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()


    # =====================================================
    # Create video
    # =====================================================
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(
        str(video_path),
        fourcc,
        1 / (save_dt * dt),  # fps
        (video_width, video_height),
    )

    for time_idx, t in enumerate(video_ts):
        img = draw_robot(
            robot,
            qs[time_idx],
            video_width,
            video_height,
        )
        video.write(img)

    video.release()
    print(f"Video saved to {video_path}")  # '''
