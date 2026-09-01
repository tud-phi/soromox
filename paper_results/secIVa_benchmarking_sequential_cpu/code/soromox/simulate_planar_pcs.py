import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from diffrax import Tsit5

from soromox.rendering import MatplotlibRenderer, OpenCVPlanarRenderer
from soromox.systems import LinkSpec, PlanarPCS, SystemState

jax.config.update("jax_enable_x64", True)  # double precision

jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _planar_position_from_pose(chi_ts: jax.Array) -> jax.Array:
    """Extract the two translational coordinates from ``[theta, x, y]`` poses."""
    if chi_ts.ndim != 2 or chi_ts.shape[1] != 3:
        raise ValueError(f"Expected planar poses with shape (T, 3), got {chi_ts.shape}")
    return chi_ts[:, 1:3]


if __name__ == "__main__":
    num_segments = 2
    rho = 1000 * jnp.ones(
        (num_segments,)
    )  # Volumetric density of Dragon Skin 20 [kg/m^3]
    segment_lengths = 3e-1 * jnp.ones((num_segments,))
    damping_matrix = 1e-2 * jnp.diag(
        (
            jnp.repeat(jnp.array([[2e0, 1e3, 1e3]]), num_segments, axis=0)
            * segment_lengths[:, None]
        ).flatten()
    )
    links = [
        LinkSpec.circular(
            length=float(segment_lengths[index]),
            radius=3e-2,
            density=float(rho[index]),
            young_modulus=1e6,
            shear_modulus=3.333333e5,
            damping=damping_matrix[
                3 * index : 3 * (index + 1),
                3 * index : 3 * (index + 1),
            ],
            reference_strain=[0.0, 1.0, 0.0],
        )
        for index in range(num_segments)
    ]
    params = PlanarPCS.params_from_links(
        links,
        gravity=jnp.array([-9.81 * 1, 9.81 * 0]),  # gravity vector [m/s^2] UP!
    )

    # ======================================================
    # Robot initialization
    # ======================================================
    robot = PlanarPCS(
        params=params,
        base_pose=jnp.array([jnp.pi / 2, 0.0, 0.0]),
    )

    J, Jd = robot.jacobian_and_time_derivative(
        q=jnp.zeros((3 * num_segments,)),
        qd=jnp.zeros((3 * num_segments,)),
        s=params.link.length[0],
    )

    # =====================================================
    # Simulation upon time
    # =====================================================
    # Initial configuration
    q0 = jnp.repeat(jnp.array([0.0, 0.0, 0.0])[None, :], num_segments, axis=0).flatten()
    # Initial velocities
    qd0 = jnp.zeros_like(q0)

    # Actuation parameters
    u = jnp.zeros_like(q0)

    # Simulation time parameters
    t0 = 0.0
    t1 = 3.0
    solver_dt = 1e-4
    save_dt = 1e-3

    # Solver
    solver = Tsit5()  # Runge-Kutta 5(4) method

    t_start = time.perf_counter()

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

    # Block until results are ready (important!)
    # jax.block_until_ready(trajectory.y)

    t_end = time.perf_counter()
    print(f"Rollout time: {t_end - t_start:.6f} s")

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
    p_ee_ts = _planar_position_from_pose(chi_ee_ts)

    # build [t, x, z]
    data_jnp = jnp.column_stack((ts, p_ee_ts))

    # convert to numpy before saving
    data_np = np.array(data_jnp)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        DATA_DIR / "end_effector_position_planar_pcs_soromox.csv",
        data_np,
        delimiter=",",
        header="t,x,z",
        comments="",
    )

    # plt.figure()
    # for segment_idx in range(num_segments):
    #     plt.plot(
    #         ts,
    #         q_ts[:, 3 * segment_idx + 0],
    #         label=r"$\kappa_\mathrm{be," + str(segment_idx + 1) + "}$ [rad/m]",
    #     )
    #     plt.plot(
    #         ts,
    #         q_ts[:, 3 * segment_idx + 2],
    #         label=r"$\sigma_\mathrm{sh," + str(segment_idx + 1) + "}$ [-]",
    #     )
    #     plt.plot(
    #         ts,
    #         q_ts[:, 3 * segment_idx + 1],
    #         label=r"$\sigma_\mathrm{ax," + str(segment_idx + 1) + "}$ [-]",
    #     )
    # plt.xlabel("Time [s]")
    # plt.ylabel("Configuration")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # plot end-effector position vs time
    plt.figure()

    plt.plot(ts, p_ee_ts[:, 0], label="End-effector x [m]")
    plt.plot(ts, p_ee_ts[:, 1], label="End-effector z [m]")
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector position [m]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()

    # end effector orientation vs. time
    # plt.figure()
    # plt.plot(
    #     ts,
    #     chi_ee_ts[:, 0] / jnp.pi * 180,
    #     label=r"End-effector Orientation $\theta$ [deg]",
    # )
    # plt.xlabel("Time [s]")
    # plt.ylabel("End-effector Orientation [deg]")
    # plt.legend()
    # plt.grid(True)
    # plt.box(True)
    # plt.tight_layout()
    # plt.show()

    # plot the end-effector position in the x-y plane as a scatter plot with the time as the color
    # plt.figure()
    # plt.scatter(chi_ee_ts[:, 1], chi_ee_ts[:, 2], c=ts, cmap="viridis")
    # plt.axis("equal")
    # plt.grid(True)
    # plt.xlabel("End-effector x [m]")
    # plt.ylabel("End-effector y [m]")
    # plt.colorbar(label="Time [s]")
    # plt.tight_layout()
    # plt.show()

    # =====================================================
    # Energy computation upon time
    # =====================================================
    U_ts = jax.vmap(jax.jit(partial(robot.potential_energy)))(q_ts)
    T_ts = jax.vmap(jax.jit(partial(robot.kinetic_energy)))(q_ts, qd_ts)

    # plt.figure()
    # plt.plot(ts, U_ts, label="Potential Energy")
    # plt.plot(ts, T_ts, label="Kinetic Energy")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Energy (J)")
    # plt.legend()
    # plt.title("Energy over Time")
    # plt.grid(True)
    # plt.box(True)
    # plt.tight_layout()
    # plt.show()

    # =====================================================
    # Plot the robot configuration upon time
    # =====================================================
    renderer = MatplotlibRenderer(robot, num_points=50)
    renderer.animate(ts=ts, q_ts=q_ts, interval=100, mode="slider")

    # =====================================================
    # OpenCV-based rendering example
    # =====================================================
    opencv_renderer = OpenCVPlanarRenderer(
        robot, num_points=50, width=700, height=700, length_scale=3.0
    )
    opencv_renderer.render_sequence(
        ts, q_ts, record_path="videos/planar_pcs_opencv.mp4"
    )
