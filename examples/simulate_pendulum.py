import cv2  # importing cv2
import jax

jax.config.update("jax_enable_x64", True)  # double precision
from jax import Array, lax, vmap
from jax import numpy as jnp
import numpy as onp
from pathlib import Path
from typing import Callable, Dict

import soromox
from soromox.systems import pendulum

num_links = 2

sym_exp_filepath = (
    Path(soromox.__file__).parent
    / "symbolic_expressions"
    / f"pendulum_nl-{num_links}.dill"
)
params = {
    "m": jnp.array([10.0, 6.0]),
    "I": jnp.array([3.0, 2.0]),
    "L": jnp.array([2.0, 1.0]),
    "Lc": jnp.array([1.0, 0.5]),
    "g": jnp.array([0.0, -9.81]),
}

# define initial configuration
q0 = jnp.zeros((num_links,))
q0 = jnp.array([jnp.pi / 8, -jnp.pi / 4])

# set simulation parameters
dt = 1e-4  # time step
ts = jnp.arange(0.0, 5, dt)  # time steps
skip_step = 100  # how many time steps to skip in between video frames

# video settings
video_width, video_height = 700, 700  # img height and width
video_path = Path(__file__).parent / "videos" / f"{sym_exp_filepath.stem}.mp4"


def draw_robot(
    robot: pendulum.Pendulum,
    q: Array,
    width: int,
    height: int,
) -> onp.ndarray:
    # plotting in OpenCV
    h, w = height, width  # img height and width
    ppm = h / (2.5 * jnp.sum(robot.L))  # pixel per meter
    robot_color = (0, 0, 0)  # black robot_color in BGR

    # poses along the robot of shape (N, 3)
    chi_ls = robot.forward_kinematics_tips(q)

    img = 255 * onp.ones((w, h, 3), dtype=jnp.uint8)  # initialize background to white
    curve_origin = onp.array(
        [w // 2, h // 2], dtype=onp.int32
    )  # in x-y pixel coordinates
    # transform robot poses to pixel coordinates
    # extract (px, py) which are now columns 1 and 2
    curve = onp.array((curve_origin + chi_ls[:, 1:] * ppm), dtype=onp.int32)
    # invert the v pixel coordinate
    curve[:, 1] = h - curve[:, 1]
    cv2.polylines(img, [curve], isClosed=False, color=robot_color, thickness=10)

    return img


if __name__ == "__main__":
    # Instantiate the pendulum model directly
    robot = pendulum.Pendulum(params)

    # initialize velocities and actuation
    qd0 = jnp.zeros_like(q0)  # initial velocities for simulation
    u = jnp.zeros_like(q0)  # torques (actuation)

    # compute the operational space matrices
    Lambda, mu, J, Jd, JB_inv = robot.operational_space_dynamical_matrices(
        q0, qd0, link_idx=1
    )
    print("Lambda:\n", Lambda)

    # call the forward dynamics
    yd = robot.forward_dynamics(ts[0], jnp.concatenate([q0, qd0]), (u,))
    print("yd0:\n", yd)

    # Integrate using the model's built-in solver
    ts_out, qs, qds = robot.resolve_upon_time(
        q0=q0,
        qd0=qd0,
        u=u,
        t0=ts[0],
        t1=ts[-1],
        dt=dt,
        skip_steps=skip_step,
    )
    video_ts = ts_out
    print("Final configuration:\n", qs[-1])

    # create video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(
        str(video_path),
        fourcc,
        1 / (skip_step * dt),  # fps
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
