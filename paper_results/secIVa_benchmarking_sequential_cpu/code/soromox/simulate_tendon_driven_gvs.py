import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optimistix as optx

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.systems import (
    GVS,
    GVSSegment,
    JointSpec,
    LinkSpec,
    StrainBasisSpec,
    SystemState,
)

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "gpu")  # or "cpu"

print("JAX default backend:", jax.default_backend())
print("JAX devices:", jax.devices())

### SOFT ROBOT UTILITIES FUNCTIONS ###


# STATIC EQUILIBRIUM EQUATION SOLVER
def solve_equilibrium(robot: GVS, u: jnp.ndarray, q0: jnp.ndarray):
    def statics_eq(q, args):
        u = args
        K = robot.stiffness_matrix()
        B = robot.actuation_matrix(q)
        G = robot.gravitational_force(q)
        return K @ q + G - B @ u

    solver = optx.Newton(rtol=1e-6, atol=1e-6)
    statics_eq_jit = jax.jit(statics_eq)
    return optx.root_find(statics_eq_jit, solver, q0, (u), max_steps=200)


jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

### BODY DEFINITION OF THE SOFT ROBOT ###

# 2 link version
# Link 1
link1 = LinkSpec.circular(
    length=0.3,
    radius=0.03,
    density=1000.0,
    young_modulus=1e6,
    shear_modulus=1e6 / (2.0 * (1.0 + 0.5)),
    material_damping_coefficient=1e4,
    reference_strain=[0, 0, 0, 1, 0, 0],
)

# Link 2
link2 = LinkSpec.circular(
    length=0.3,
    radius=0.03,
    density=1000.0,
    young_modulus=1e6,
    shear_modulus=1e6 / (2.0 * (1.0 + 0.5)),
    material_damping_coefficient=1e4,
    reference_strain=[0, 0, 0, 1, 0, 0],
)
joint1 = JointSpec(type="Fixed")
joint2 = JointSpec(type="Fixed")

basis1 = StrainBasisSpec(
    type="legendre",
    strain_selector=[1, 1, 1, 1, 0, 0],
    basis_order=[1, 1, 1, 1, 0, 0],
)
basis2 = StrainBasisSpec(
    type="legendre",
    strain_selector=[1, 1, 1, 1, 0, 0],
    basis_order=[1, 1, 1, 1, 0, 0],
)


num_gauss_points = [5, 5]
g = [0.0, 0.0, -9.81]

active_tendon_routing = ThreadlikeRouting.linear(
    intercept=jnp.array([[0.0, 0.0, 0.02], [0.0, 0.02, 0.0]]),
    start_segment_index=0,
    end_segment_index=(1, 1),
)


# attention: 0DEG -> Y+, 180DEG -> Y-, 90DEG -> Z+, 270DEG -> Z-


p0 = jnp.array([0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0])

# 2 link version
segments = [
    GVSSegment(
        link=link1, joint=joint1, basis=basis1, num_gauss_points=num_gauss_points[0]
    ),
    GVSSegment(
        link=link2, joint=joint2, basis=basis2, num_gauss_points=num_gauss_points[1]
    ),
]
robot = GVS.from_segments(
    segments,
    gravity=jnp.asarray(g),
    base_pose=p0,
    actuators=ThreadlikeActuator.tendons(active_tendon_routing),
    # max_dof=6,
    scale_rotational_basis_by_length=True,
)

# debug: check g0
# g0_test = lie.exp_SE3(p0)
# print("g0_test:\n", g0_test[:3,:3])

### MATRICES CHECKING AND PRINTING ###

print("num_segments:", int(robot.num_segments))
print("V_dof (joint, link) per segment:\n", robot.dofs_per_segment)
print("max_dof: \n", robot.max_dof)
# max_dof = robot.max_dof
# padded = 2 * max_dof


#### DEBUG: CHECKING MATRICES #####

# xi_reference = jax.device_get(robot.V_xi_ref_Xs)
# print("xi_reference shape:", onp.array(xi_reference).shape)
# print(onp.array(xi_reference))
# xi_ref_b1gp0 = xi_reference[0, 0]
# print("xi_ref segment 0, gauss 0:", onp.array(xi_ref_b1gp0))

# Bq printing per segment
for j in range(robot.num_segments):
    dof_joint = int(robot.dofs_per_segment[j, 0])
    dof_link = int(robot.dofs_per_segment[j, 1])

#     n_gauss_seg = int(robot.V_nip[j])
#     print(f"segment {j} -> dof_joint={dof_joint}, dof_link={dof_link}")


na = robot.num_actuators
print("num_actuators:", na)
dof = sum(robot.dofs_per_segment.reshape(-1))
print("total dof:", dof)

q0 = jnp.zeros((dof,))
q0dot = jnp.zeros((dof,))
L_cum = jax.device_get(robot.segment_end_positions)
total_length = float(L_cum[-1])
s_end = float(total_length)
g_end = robot.forward_kinematics(q0, s_end)

J_end = robot.jacobian_bodyframe(q0, s_end)
M = robot.inertia_matrix(q0)
K = robot.stiffness_matrix()
F = robot.gravitational_force(q0)
D = robot.damping_matrix(q0)
B = robot.actuation_matrix(q0)
C = robot.coriolis_matrix(q0, q0dot)

# Threadlike tendon controls are nonnegative tensions.
u = jnp.asarray([1, 1], dtype=q0.dtype)
tau = robot.actuation_force(q0, u)


# print("body lengths per segment:", robot.V_L)
# print("L_cum:", L_cum)
# print("total length:", total_length)
# print("s_end:", s_end)
# print("g(s_end) SE(3):\n", g_end)
# print("Jacobian at end (6 x dof):\n", J_end)
# print("Inertia matrix:", M)
# print("Stiffness matrix:", K)
# print("Gravitational force:", F)
# print("Damping matrix:", D)
# print("Actuation matrix:", B )
# print("Coriolis matrix:", C)
# print("Actuation matrix shape:", B.shape)
# print("Input u:", u)
# print("Input u shape:", u.shape)
# print("Actuation force (tau):", tau)
# print("Actuation force shape:", tau.shape)

# y = jnp.concatenate([q0, q0dot])
tau_ext = 0 * jnp.ones((dof,))

l_tendons = robot.actuators[0].path_lengths(robot, q0)
# print("tendon lengths (m):", jax.device_get(l_tendons))


# if Open3DRenderer is None:
#     raise ImportError("Open3DRenderer is unavailable. Install open3d to run this.")

# # Create renderer for visualization
# renderer = Open3DRenderer(robot, num_points=50)

# =====================================================
# Static equilibrium (solve statics) and plot its shape
# =====================================================
# Solve for static equilibrium q* given current actuation u
res_stat = solve_equilibrium(robot, u, q0)
q_stat = res_stat.value  # equilibrium generalized coordinates
# print("q* =", q_stat)

# renderer.show(q_stat)
# renderer.show(q0)


# =====================================================
# Simulation upon time
# =====================================================

# Simulation time parameters
t0 = 0.0
t1 = 3.0
solver_dt = 1e-4
skip_step = 100  # how many time steps to skip in between video frames
save_dt = solver_dt * skip_step

t_start = time.perf_counter()

initial_state = SystemState(t=t0, y=jnp.concatenate([q0, q0dot]))
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

t_end = time.perf_counter()
print(f"Rollout time: {t_end - t_start:.6f} s")

# =====================================================
# End-effector position upon time
# =====================================================
forward_kinematics_end_effector = jax.jit(
    partial(robot.forward_kinematics, s=robot.length)  # end-effector position
)
g_ee_ts = jax.vmap(forward_kinematics_end_effector)(q_ts)


# build [t, x, y, z]
data_jnp = jnp.column_stack(
    (
        ts,
        g_ee_ts[:, 0, 3],
        g_ee_ts[:, 1, 3],
        g_ee_ts[:, 2, 3],
    )
)

# convert to numpy before saving
data_np = np.array(data_jnp)

DATA_DIR.mkdir(parents=True, exist_ok=True)
np.savetxt(
    DATA_DIR / "end_effector_position_tendon_driven_gvs_soromox.csv",
    data_np,
    delimiter=",",
    header="t,x,y,z",
    comments="",
)


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
