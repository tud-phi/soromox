from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

from soromox.rendering import (
    BackboneColorConfig,
    Open3DRenderer,
    RendererColorConfig,
    ViserRenderer,
)
from soromox.systems import (
    GVS,
    GVSSegment,
    JointSpec,
    LinkSpec,
    StrainBasisSpec,
    SystemState,
)

jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)


if __name__ == "__main__":
    # Define model inputs
    links: list[LinkSpec] = []
    joints: list[JointSpec] = []
    bases: list[StrainBasisSpec] = []
    num_gauss_points: list[int] = []

    link1 = LinkSpec.circular(
        young_modulus=1e6,
        shear_modulus=1e6 / 3.0,
        density=1000,
        material_damping_coefficient=1e4,
        length=0.3,
        radius=0.03,
        reference_strain=[0, 0, 0, 1, 0, 0],
    )
    links.append(link1)
    joint1 = JointSpec(type="fixed")
    joints.append(joint1)
    basis1 = StrainBasisSpec(
        type="legendre",
        strain_selector=[0, 1, 1, 0, 0, 0],
        basis_order=[0, 0, 0, 0, 0, 0],
    )
    bases.append(basis1)
    num_gauss_points.append(5)  # Number of Gauss points for the first link

    link2 = LinkSpec.circular(
        young_modulus=1e6,
        shear_modulus=1e6 / 3.0,
        density=1000,
        material_damping_coefficient=1e4,
        length=0.3,
        radius=0.03,
        reference_strain=[0, 0, 0, 1, 0, 0],
    )
    links.append(link2)
    joint2 = JointSpec(type="fixed")
    # joint2 = JointSpec(type='revolute', axis='z')
    joints.append(joint2)
    basis2 = StrainBasisSpec(
        type="monomial",
        strain_selector=[1, 1, 0, 0, 0, 0],
        basis_order=[0, 0, 0, 0, 0, 0],
    )
    bases.append(basis2)
    num_gauss_points.append(6)  # Number of Gauss points for the second link

    # ======================================================
    # Robot initialization
    # ======================================================
    segments = [
        GVSSegment(link=link, joint=joint, basis=basis, num_gauss_points=n)
        for link, joint, basis, n in zip(links, joints, bases, num_gauss_points)
    ]
    robot = GVS.from_segments(
        segments,
        gravity=jnp.array([0.0, 0.0, -9.81]),
        max_dof=6,
    )

    print(f"System initialized with {robot.num_segments} segments")
    print(f"max DOFs: {robot.max_dof}, max Gauss points: {robot.max_num_gauss_points}.")
    print(f"Total DOFs: {robot.num_dofs} (chosen), {robot.num_padded_dofs} (real)")

    # =====================================================
    # Simulation upon time
    # =====================================================
    # Initial configuration
    q0 = jnp.zeros(robot.num_dofs)
    # Initial velocities
    qd0 = jnp.zeros_like(q0)

    if Open3DRenderer is None:
        raise ImportError("Open3DRenderer is unavailable. Install open3d to run this.")

    # Visualize the initial configuration using Open3DRenderer
    renderer = Open3DRenderer(robot, num_points=50)
    renderer.show(q0)

    # Actuation parameters
    rng = jax.random.PRNGKey(0)
    u = jax.random.uniform(rng, shape=q0.shape) * 1.0
    print("u =\n", u)

    # Simulation time parameters
    t0 = 0.0
    t1 = 2.0
    solver_dt = 1e-4
    skip_step = 100  # how many time steps to skip in between video frames
    save_dt = solver_dt * skip_step

    initial_state = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))
    trajectory = robot.rollout_to(
        initial_state=initial_state,
        u=u,
        t1=t1,
        solver_dt=solver_dt,
        save_dt=save_dt,
        max_steps=None,
    )
    ts = trajectory.t
    q_ts, qd_ts = jnp.split(trajectory.y, 2, axis=1)
    print(f"Simulation completed with {len(ts)} time steps.")

    # =====================================================
    # End-effector position upon time
    # =====================================================
    forward_kinematics_end_effector = jax.jit(
        partial(robot.forward_kinematics, s=robot.length)  # end-effector position
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
    renderer.render_sequence(ts=ts, q_ts=q_ts, playback_speed=1.0)

    # =====================================================
    # Viser web-based visualization (trajectory + transparent q0 overlay)
    # =====================================================
    if ViserRenderer is None:
        raise ImportError("ViserRenderer is unavailable. Install viser to run this.")

    q0_ts = jnp.repeat(q0[None, :], len(ts), axis=0)
    q_ts_overlay = jnp.stack([q_ts, q0_ts], axis=0)

    # Use robot alpha as the default opacity when point/segment colors omit alpha.
    color_config = RendererColorConfig(
        backbone=BackboneColorConfig(
            robot_colors=[
                [1.0, 1.0, 1.0, 1.0],  # dynamic trajectory (opaque)
                [0.2, 0.6, 0.9, 0.35],  # q0 overlay (transparent)
            ]
        )
    )
    viser_renderer = ViserRenderer(robot, num_points=50)
    viser_renderer.render_sequence(
        ts=ts,
        q_ts=q_ts_overlay,
        multi_robot_layout="overlay",
        playback_speed=1.0,
        autoplay=True,
        loop=True,
        color_config=color_config,
    )
