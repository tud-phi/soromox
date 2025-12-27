from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp

jax.config.update("jax_enable_x64", True)
from soromox.rendering import MatplotlibRenderer  # double precision
from soromox.systems import GVS, SystemState
from soromox.systems.gvs import BasisAttributes, JointAttributes, LinkAttributes

jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)


if __name__ == "__main__":
    # Define model inputs
    List_links: list[LinkAttributes] = []
    List_joints: list[JointAttributes] = []
    List_basis: list[BasisAttributes] = []
    List_nGauss: list[int] = []

    link1 = LinkAttributes(
        section="Circular", E=1e6, nu=0.5, rho=1000, eta=1e4, L=0.3, r_i=0.03, r_f=0.03
    )
    List_links.append(link1)
    joint1 = JointAttributes(jointtype="Fixed")
    List_joints.append(joint1)
    basis1 = BasisAttributes(
        basistype="Legendre",
        Bdof=[0, 1, 1, 0, 0, 0],
        Bodr=[0, 0, 0, 0, 0, 0],
        xi_ref=[0, 0, 0, 1, 0, 0],
    )
    List_basis.append(basis1)
    List_nGauss.append(5)  # Number of Gauss points for the first link

    link2 = LinkAttributes(
        section="Circular", E=1e6, nu=0.5, rho=1000, eta=1e4, L=0.3, r_i=0.03, r_f=0.03
    )
    List_links.append(link2)
    joint2 = JointAttributes(jointtype="Fixed")
    # joint2 = JointAttributes(jointtype='Revolute', axis='z')
    List_joints.append(joint2)
    basis2 = BasisAttributes(
        basistype="Monomial",
        Bdof=[1, 1, 0, 0, 0, 0],
        Bodr=[0, 0, 0, 0, 0, 0],
        xi_ref=[0, 0, 0, 1, 0, 0],
    )
    List_basis.append(basis2)
    List_nGauss.append(6)  # Number of Gauss points for the second link

    # link3 = LinkAttributes(
    #     section='Elliptical',  # Section type
    #     E=1e7,                # Young's modulus in Pascals
    #     nu=0.4,               # Poisson's ratio [-1, 0.5]
    #     rho=1050,             # Density [kg/m^3]
    #     eta=1e4,              # Damping coefficient
    #     L=0.3,                # Length in meters
    #     a_i=0.04,             # Initial semi-major axis in meters
    #     a_f=0.04,             # Final semi-major axis in meters
    #     b_i=0.02,             # Initial semi-minor axis in meters
    #     b_f=0.02              # Final semi-minor axis in meters
    # )
    # List_links.append(link3)
    # joint3 = JointAttributes(
    #     jointtype='Revolute',  # Prismatic joint
    #     axis='z',               # Axis of translation
    # )
    # List_joints.append(joint3)
    # basis3 = BasisAttributes(
    #     basistype='Chebychev',       # Type of basis
    #     Bdof=[0, 1, 0, 1, 0, 0],    # Degrees of freedom for each deformation type
    #     Bodr=[0, 0, 0, 0, 0, 0],    # Order of basis functions for each deformation type
    #     xi_ref=[0, 0, 0, 1, 0, 0], # Reference strain values as vector
    # )
    # List_basis.append(basis3)
    # List_nGauss.append(5)  # Number of Gauss points for the third link

    # ======================================================
    # Robot initialization
    # ======================================================
    robot = GVS(
        links_list=List_links,
        joints_list=List_joints,
        basis_list=List_basis,
        n_gauss_list=List_nGauss,
        gravity_vector=[0, 0, -9.81],
    )

    print(f"System initialized with {robot.num_segments} segments")
    print(f"max DOFs: {robot.max_dof}, max Gauss points: {robot.max_nGauss}.")
    print(f"Total DOFs: {robot.dof_tot_system} (chosen), {robot.dof_tot_max} (real)")

    # =====================================================
    # Simulation upon time
    # =====================================================
    # Initial configuration
    q0 = jnp.zeros(robot.dof_tot_system)
    # Initial velocities
    qd0 = jnp.zeros_like(q0)

    # Plot the initial configuration using MatplotlibRenderer
    renderer = MatplotlibRenderer(robot, num_points=50)
    curve = onp.array(renderer.compute_backbone_curve(q0))

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "3d"})
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], lw=4, color="blue")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Initial configuration")
    ax.axis("equal")
    plt.show()

    # Actuation parameters
    u = jnp.zeros_like(q0)

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
        partial(
            robot.forward_kinematics,
            s=jnp.sum(robot.V_L),  # end-effector position
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
    # Plot the robot configuration upon time
    # =====================================================
    renderer.animate(ts=ts, q_ts=q_ts, interval=100, mode="slider")
