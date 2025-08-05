import cv2  # importing cv2
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
from soromox.systems import planar_pcs_sym

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
    "g": jnp.array([0.0, -9.81]),
    "E": 2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
    "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
}
params["D"] = 1e-3 * jnp.diag(
    (
        jnp.repeat(jnp.array([[1e0, 1e3, 1e3]]), num_segments, axis=0)
        * params["l"][:, None]
    ).flatten()
)

# activate all strains (i.e. bending, shear, and axial)
strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

# define initial configuration
q0 = jnp.repeat(
    jnp.array([5.0 * jnp.pi, 0.1, 0.2])[None, :], num_segments, axis=0
).flatten()
# number of generalized coordinates
n_q = q0.shape[0]

# set simulation parameters
dt = 1e-4  # time step
ts = jnp.arange(0.0, 2, dt)  # time steps
skip_step = 10  # how many time steps to skip in between video frames
video_ts = ts[::skip_step]  # time steps for video

# video settings
video_width, video_height = 700, 700  # img height and width
video_path = Path(__file__).parent / "videos" / f"planar_pcs_ns-{num_segments}.mp4"


def draw_robot(
    batched_forward_kinematics_fn: Callable,
    params: Dict[str, Array],
    q: Array,
    width: int,
    height: int,
    num_points: int = 50,
) -> onp.ndarray:
    # plotting in OpenCV
    h, w = height, width  # img height and width
    ppm = h / (2.0 * jnp.sum(params["l"]))  # pixel per meter
    base_color = (0, 0, 0)  # black robot_color in BGR
    robot_color = (255, 0, 0)  # black robot_color in BGR

    # we use for plotting N points along the length of the robot
    s_ps = jnp.linspace(0, jnp.sum(params["l"]), num_points)

    # poses along the robot of shape (3, N)
    chi_ps = batched_forward_kinematics_fn(params, q, s_ps)

    img = 255 * onp.ones((w, h, 3), dtype=jnp.uint8)  # initialize background to white
    curve_origin = onp.array(
        [w // 2, 0.1 * h], dtype=onp.int32
    )  # in x-y pixel coordinates
    # draw base
    cv2.rectangle(img, (0, h - curve_origin[1]), (w, h), color=base_color, thickness=-1)
    # transform robot poses to pixel coordinates
    # should be of shape (N, 2)
    curve = onp.array((curve_origin + chi_ps[:2, :].T * ppm), dtype=onp.int32)
    # invert the v pixel coordinate
    curve[:, 1] = h - curve[:, 1]
    cv2.polylines(img, [curve], isClosed=False, color=robot_color, thickness=10)

    return img


if __name__ == "__main__":
    strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = (
        planar_pcs_sym.factory(sym_exp_filepath, strain_selector)
    )
    # jit the functions
    dynamical_matrices_fn = jax.jit(partial(dynamical_matrices_fn))
    batched_forward_kinematics = vmap(
        forward_kinematics_fn, in_axes=(None, None, 0), out_axes=-1
    )

    # import matplotlib.pyplot as plt
    # plt.plot(chi_ps[0, :], chi_ps[1, :])
    # plt.axis("equal")
    # plt.grid(True)
    # plt.xlabel("x [m]")
    # plt.ylabel("y [m]")
    # plt.show()

    # Displaying the image
    # window_name = f"Planar PCS with {num_segments} segments"
    # img = draw_robot(batched_forward_kinematics, params, q0, video_width, video_height)
    # cv2.namedWindow(window_name)
    # cv2.imshow(window_name, img)
    # cv2.waitKey()
    # cv2.destroyWindow(window_name)

    x0 = jnp.concatenate([q0, jnp.zeros_like(q0)])  # initial condition
    tau = jnp.zeros_like(q0)  # torques

    ode_fn = ode_factory(dynamical_matrices_fn, params, tau)
    # jit the ODE function
    ode_fn = jax.jit(ode_fn)
    # jit the ODE function
    ode_fn = jax.jit(ode_fn)
    term = ODETerm(ode_fn)

    sol = diffeqsolve(
        term,
        solver=Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=dt,
        y0=x0,
        max_steps=None,
        saveat=SaveAt(ts=video_ts),
    )

    print("sol.ys =\n", sol.ys)
    # the evolution of the generalized coordinates
    q_ts = sol.ys[:, :n_q]
    # the evolution of the generalized velocities
    q_d_ts = sol.ys[:, n_q:]    

    s_max = jnp.array([jnp.sum(params["l"])])

    forward_kinematics_fn_end_effector = partial(forward_kinematics_fn, params, s=s_max)
    forward_kinematics_fn_end_effector = jax.jit(forward_kinematics_fn_end_effector)
    forward_kinematics_fn_end_effector = vmap(forward_kinematics_fn_end_effector)

    # evaluate the forward kinematics along the trajectory
    chi_ee_ts = forward_kinematics_fn_end_effector(q_ts)
    # plot the configuration vs time
    plt.figure()
    for segment_idx in range(num_segments):
        plt.plot(
            video_ts,
            q_ts[:, 3 * segment_idx + 0],
            label=r"$\kappa_\mathrm{be," + str(segment_idx + 1) + "}$ [rad/m]",
        )
        plt.plot(
            video_ts,
            q_ts[:, 3 * segment_idx + 1],
            label=r"$\sigma_\mathrm{sh," + str(segment_idx + 1) + "}$ [-]",
        )
        plt.plot(
            video_ts,
            q_ts[:, 3 * segment_idx + 2],
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
    plt.plot(video_ts, chi_ee_ts[:, 0], label="x")
    plt.plot(video_ts, chi_ee_ts[:, 1], label="y")
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector Position [m]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()
    # plot the end-effector position in the x-y plane as a scatter plot with the time as the color
    plt.figure()
    plt.scatter(chi_ee_ts[:, 0], chi_ee_ts[:, 1], c=video_ts, cmap="viridis")
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
        partial(jax.jit(auxiliary_fns["kinetic_energy_fn"]), params)
    )
    potential_energy_fn_vmapped = vmap(
        partial(jax.jit(auxiliary_fns["potential_energy_fn"]), params)
    )
    U_ts = potential_energy_fn_vmapped(q_ts)
    T_ts = kinetic_energy_fn_vmapped(q_ts, q_d_ts)
    plt.figure()
    plt.plot(video_ts, U_ts, label="Potential energy")
    plt.plot(video_ts, T_ts, label="Kinetic energy")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [J]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()

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
        x = sol.ys[time_idx]
        img = draw_robot(
            batched_forward_kinematics,
            params,
            x[: (x.shape[0] // 2)],
            video_width,
            video_height,
        )
        video.write(img)

    video.release()
    print(f"Video saved at {video_path}")
