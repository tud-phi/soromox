import cv2  # importing cv2
import jax

jax.config.update("jax_enable_x64", True)  # double precision
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as onp
from jax import Array
from jax import numpy as jnp

from soromox.systems import SystemState, TendonActuatedPendulum

num_links = 2

params = {
    "m": jnp.array([10.0, 6.0]),
    "I": jnp.array([3.0, 2.0]),
    "L": jnp.array([2.0, 1.0]),
    "Lc": jnp.array([1.0, 0.5]),
    "g": jnp.array([0.0, -9.81]),
    "K": jnp.diag(jnp.array([1.5, 2.0])),
}
tendon_params = {
    "R_at": jnp.array([[0.2, -0.25]]),
    "R_pt": jnp.array([[-0.15, 0.3]]),
    "k_pt": jnp.array([2.75]),
    "l_pt0": jnp.array([0.5]),
}

# define initial configuration
q0 = jnp.zeros((num_links,))
# q0 = jnp.array([jnp.pi / 8, -jnp.pi / 4])

# set simulation parameters
solver_dt = 1e-4  # time step
t0 = 0.0
t1 = 5.0
save_dt = solver_dt * 100

# video settings
video_width, video_height = 1080, 1080  # img height and width
video_path = Path("videos") / f"tendon_actuated_pendulum_nl-{num_links}.mp4"


def draw_robot(
    robot: TendonActuatedPendulum,
    q: Array,
    width: int,
    height: int,
    target_position: Array | None = None,
    draw_active_tendons: bool = True,
    draw_passive_tendons: bool = True,
) -> onp.ndarray:
    # Colors
    # BGR: [tab:blue, tab:orange, tab:green]
    color_links = [(14, 127, 255), (180, 119, 31), (44, 160, 44)]
    # BGR: [tab:red, black, tab:purple, tab:brown, tab:cyan]
    a_color_cables = [
        (40, 39, 214),
        (0, 0, 0),
        (75, 86, 140),
        (189, 103, 148),
        (207, 190, 23),
    ]
    p_color_cables = list(reversed(a_color_cables))

    # Plot in OpenCV
    h, w = height, width  # img height and width
    ppm = h / (2.5 * jnp.sum(robot.L))  # pixel per meter

    # Initialize image background and origin of the pixel frame
    img = 255 * onp.ones((w, h, 3), dtype=jnp.uint8)  # initialize background to white
    origin = onp.array([w // 2, h // 2], dtype=onp.int32)  # in x-y pixel coordinates

    # Target position
    if target_position is not None:
        target = onp.array(origin + target_position * ppm, dtype=onp.int32)
        target[1] = h - target[1]
        cv2.drawMarker(
            img=img,
            position=target,
            color=(0, 0, 0),
            markerType=cv2.MARKER_STAR,
            markerSize=20,
            thickness=2,
        )

    # Poses along the robot of shape (n, 3)
    chi_ls = robot.forward_kinematics_tips(q)

    # Actuation matrix
    A_at = onp.asarray(robot.actuation_matrix(q))
    N, Na = A_at.shape

    # Passive tendons
    try:
        A_pt = onp.asarray(robot.A_pt)
        Np = A_pt.shape[1]
    except:
        A_pt = onp.zeros((N, N))

    # Transform robot poses to pixel coordinates. Should be of shape (N, 2)
    curve = onp.array((origin + chi_ls[:, 1:] * ppm), dtype=onp.int32)
    curve = onp.vstack([origin, curve])  # (N + 1, 2)

    # Invert the y pixel coordinate
    curve[:, 1] = h - curve[:, 1]
    for i in range(curve.shape[0] - 1):
        pts = onp.array([curve[i, :], curve[i + 1, :]]).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=False, color=color_links[i], thickness=10)

    # Angles of the links
    angles = chi_ls[:, 0] + onp.pi / 2  # (N,)

    # Active pulleys and cables
    if not onp.array_equal(A_at, onp.eye(N)) and draw_active_tendons:
        chi = onp.vstack([jnp.zeros(3), chi_ls])
        for i in range(Na):
            cable_pts = []
            for j in range(N):
                cv2.circle(
                    img,
                    center=tuple(curve[j, :]),
                    radius=int(abs(A_at[j, i] * ppm)),
                    color=a_color_cables[i],
                    thickness=2,
                )
                idx = max(0, j - 1)
                if A_at[j, i] != 0:
                    cable_pt = chi[j, 1:] + A_at[j, i] * onp.array(
                        [onp.cos(angles[idx]), onp.sin(angles[idx])]
                    )
                    cable_pts.append(origin + cable_pt * ppm)
                    if j > 0 and j < A_at.shape[0] - 1:
                        cable_pt = chi[j, 1:] + A_at[j, i] * onp.array(
                            [onp.cos(angles[idx + 1]), onp.sin(angles[idx + 1])]
                        )
                        cable_pts.append(origin + cable_pt * ppm)
            cable_pts = onp.array(cable_pts, dtype=onp.int32)
            cable_pts[:, 1] = h - cable_pts[:, 1]
            cable_pairs = onp.split(cable_pts, max(1, int(len(cable_pts) / 2)), axis=0)
            for pair in cable_pairs:
                cv2.polylines(
                    img, [pair], isClosed=False, color=a_color_cables[i], thickness=2
                )

    # Passive pulleys and cables
    if not onp.array_equal(A_pt, onp.zeros((N, N))) and draw_passive_tendons:
        chi = onp.vstack([jnp.zeros(3), chi_ls])
        for i in range(Np):
            cable_pts = []
            for j in range(N):
                cv2.circle(
                    img,
                    center=tuple(curve[j, :]),
                    radius=int(abs(A_pt[j, i] * ppm)),
                    color=p_color_cables[i],
                    thickness=2,
                )
                idx = max(0, j - 1)
                if A_pt[j, i] != 0:
                    cable_pt = chi[j, 1:] + A_pt[j, i] * onp.array(
                        [onp.cos(angles[idx]), onp.sin(angles[idx])]
                    )
                    cable_pts.append(origin + cable_pt * ppm)
                    if j > 0 and j < A_pt.shape[0] - 1:
                        cable_pt = chi[j, 1:] + A_pt[j, i] * onp.array(
                            [onp.cos(angles[idx + 1]), onp.sin(angles[idx + 1])]
                        )
                        cable_pts.append(origin + cable_pt * ppm)
            cable_pts = onp.array(cable_pts, dtype=onp.int32)
            cable_pts[:, 1] = h - cable_pts[:, 1]
            cable_pairs = onp.split(cable_pts, max(1, int(len(cable_pts) / 2)), axis=0)
            for pair in cable_pairs:
                cv2.polylines(
                    img, [pair], isClosed=False, color=p_color_cables[i], thickness=2
                )
    return img


if __name__ == "__main__":
    # Instantiate the pendulum model directly
    robot = TendonActuatedPendulum(params, tendon_params)

    # initialize velocities and actuation
    qd0 = jnp.zeros_like(q0)  # initial velocities for simulation
    u = jnp.array([0.0])  # torques (actuation)

    # call the forward dynamics
    yd = robot.forward_dynamics(t0, jnp.concatenate([q0, qd0]), (u,))
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
    print("l_tot_a at q0 =", robot.active_tendon_length(q0).shape)
    print(robot.active_tendon_length(q0))
    print("l_tot_p at q0 =", robot.passive_tendon_length(q0).shape)
    print(robot.passive_tendon_length(q0))

    # Integrate using the model's built-in solver
    initial_state = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))
    trajectory = robot.rollout_to(
        initial_state=initial_state,
        u=u,
        t1=t1,
        solver_dt=solver_dt,
        save_dt=save_dt,
    )
    ts = trajectory.t
    qs, qds = jnp.split(trajectory.y, 2, axis=1)
    video_ts = ts
    print("Final configuration:\n", qs[-1])

    # =====================================================
    # Energy computation upon time
    # =====================================================
    U_ts_out = jax.vmap(jax.jit(partial(robot.potential_energy)))(qs)
    T_ts_out = jax.vmap(jax.jit(partial(robot.kinetic_energy)))(qs, qds)
    E_ts_out = jax.vmap(jax.jit(partial(robot.total_energy)))(qs, qds)

    plt.figure()
    plt.plot(ts, U_ts_out, label="Potential Energy")
    plt.plot(ts, T_ts_out, label="Kinetic Energy")
    plt.plot(ts, E_ts_out, label="Total Energy")
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
        1 / save_dt,  # fps
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
    print(f"Video saved to {video_path}")
