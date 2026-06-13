from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optimistix as optx

from soromox.rendering import Open3DRenderer
from soromox.systems import (
    CrossSectionGeometry,
    LinearTendonRoutingParams,
    SystemState,
    TendonActuatedGVS,
)
from soromox.systems.gvs import GVSSegment, JointSpec, LinkSpec, StrainBasisSpec

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "gpu")  # or "cpu"

print("JAX default backend:", jax.default_backend())
print("JAX devices:", jax.devices())

### SOFT ROBOT UTILITIES FUNCTIONS ###


# STATIC EQUILIBRIUM EQUATION SOLVER
def solve_equilibrium(robot: TendonActuatedGVS, u: jnp.ndarray, q0: jnp.ndarray):
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


### BODY DEFINITION OF THE SOFT ROBOT ###

# 2 link version
# Link 1
link1 = LinkSpec(
    cross_section_geometry=CrossSectionGeometry.CIRCULAR,
    E=3.04e5,
    nu=0.45,
    rho=1310.0,
    eta=1e4,
    L=0.0250 + 0.2550 + 0.0250,
    r_i=0.01541,
    r_f=0.00642,
)

# Link 2
link2 = LinkSpec(
    cross_section_geometry=CrossSectionGeometry.CIRCULAR,
    E=3.04e5,
    nu=0.45,
    rho=1310.0,
    eta=1e4,
    L=0.0550,
    r_i=0.00642,
    r_f=0.00480,
)
joint1 = JointSpec(type="fixed")
joint2 = JointSpec(type="fixed")

basis1 = StrainBasisSpec(
    type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[1, 1, 1, 1, 0, 0]
)
basis2 = StrainBasisSpec(
    type="monomial", active=[0, 1, 1, 0, 0, 0], orders=[0, 0, 0, 0, 0, 0]
)


num_gauss_points = [8, 8]
g = [0.0, 0.0, -9.81]


active_tendon_routing = LinearTendonRoutingParams(
    y_intercept=jnp.array(
        [0.0114 * jnp.cos(jnp.pi / 180 * 30), 0.0114 * jnp.cos(jnp.pi / 180 * 150)]
    ),
    y_slope=jnp.array(
        [-0.0295 * jnp.cos(jnp.pi / 180 * 30), -0.0295 * jnp.cos(jnp.pi / 180 * 150)]
    ),
    z_intercept=jnp.array(
        [0.0114 * jnp.sin(jnp.pi / 180 * 30), 0.0114 * jnp.sin(jnp.pi / 180 * 150)]
    ),
    z_slope=jnp.array(
        [-0.0295 * jnp.sin(jnp.pi / 180 * 30), -0.0295 * jnp.sin(jnp.pi / 180 * 150)]
    ),
    attachment_segment_index=jnp.array([0, 0]),
)
# attention: 0DEG -> Y+, 180DEG -> Y-, 90DEG -> Z+, 270DEG -> Z-

p0 = jnp.array([jnp.sqrt(0.5), 0.0, jnp.sqrt(0.5), 0.0, 0.0, 0.0, 0.0])

# 2 link version
segments = [
    GVSSegment(
        link=link1, joint=joint1, basis=basis1, num_gauss_points=num_gauss_points[0]
    ),
    GVSSegment(
        link=link2, joint=joint2, basis=basis2, num_gauss_points=num_gauss_points[1]
    ),
]
robot = TendonActuatedGVS.from_segments(
    segments,
    gravity=jnp.asarray(g),
    base_pose=p0,
    active_tendon_routing=active_tendon_routing,
    max_dof=6,
    scale_rotational_basis_by_length=True,
)
# debug: check g0
# from soromox.utils.geometry import poses
# g0_test = poses.quaternion_pose_to_transform(p0)
# print("g0_test:\n", g0_test[:3,:3])

### MATRICES CHECKING AND PRINTING ###

print("num_segments:", int(robot.num_segments))
print("V_dof (joint, link) per segment:\n", robot.dofs_per_segment)
print("max_dof: \n", robot.max_dof)
# max_dof = robot.max_dof
# padded = 2 * max_dof


#### DEBUG: CHECKING MATRICES #####

# xi_reference = jax.device_get(robot.xi_ref_Xs)
# print("xi_reference shape:", onp.array(xi_reference).shape)
# print(onp.array(xi_reference))
# xi_ref_b1gp0 = xi_reference[0, 0]
# print("xi_ref segment 0, gauss 0:", onp.array(xi_ref_b1gp0))

# Bq printing per segment
for j in range(robot.num_segments):
    dof_joint = int(robot.dofs_per_segment[j, 0])
    dof_link = int(robot.dofs_per_segment[j, 1])

#     n_gauss_seg = int(robot.num_integration_points[j])
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

u = jnp.asarray([-1, -0.00], dtype=q0.dtype)
tau = robot.actuation_force(q0, u)


print("total body length:", robot.length)
# print("L_cum:", L_cum)
# print("total length:", total_length)
print("s_end:", s_end)
print("g(s_end) SE(3):\n", g_end)
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

l_tendons = robot.tendon_length(q0)  # jax.Array (num_actuators,)
print("tendon lengths (m):", jax.device_get(l_tendons))


if Open3DRenderer is None:
    raise ImportError("Open3DRenderer is unavailable. Install open3d to run this.")

# Create renderer for visualization
renderer = Open3DRenderer(robot, num_points=50)

# =====================================================
# Static equilibrium (solve statics) and plot its shape
# =====================================================
# Solve for static equilibrium q* given current actuation u
res_stat = solve_equilibrium(robot, u, q0)
q_stat = res_stat.value  # equilibrium generalized coordinates
print("q* =", q_stat)

renderer.show(q_stat)
renderer.show(q0)


# =====================================================
# Simulation upon time
# =====================================================

# Simulation time parameters
t0 = 0.0
t1 = 2.0
solver_dt = 1e-4
skip_step = 100  # how many time steps to skip in between video frames
save_dt = solver_dt * skip_step


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
print(f"Simulation completed with {len(ts)} time steps.")

# =====================================================
# End-effector position upon time
# =====================================================
forward_kinematics_end_effector = jax.jit(
    partial(robot.forward_kinematics, s=robot.length)  # end-effector position
)
g_ee_ts = jax.vmap(forward_kinematics_end_effector)(q_ts)


### MARKERS POSITIONS UPON TIME ###
forward_kinematics_marker_1 = jax.jit(
    partial(
        robot.forward_kinematics,
        s=0.1291,
    )
)
g_marker_1_ts = jax.vmap(forward_kinematics_marker_1)(q_ts)
p_marker_1_ts = g_marker_1_ts[:, :3, 3] + g_marker_1_ts[:, :3, :3] @ jnp.array(
    [0.0, 0.0, 0.025]
)


forward_kinematics_marker_2 = jax.jit(
    partial(
        robot.forward_kinematics,
        s=0.2195,
    )
)
g_marker_2_ts = jax.vmap(forward_kinematics_marker_2)(q_ts)
p_marker_2_ts = g_marker_2_ts[:, :3, 3] + g_marker_2_ts[:, :3, :3] @ jnp.array(
    [0.0, 0.0, -0.021]
)


forward_kinematics_marker_3 = jax.jit(
    partial(
        robot.forward_kinematics,
        s=0.2800,
    )
)
g_marker_3_ts = jax.vmap(forward_kinematics_marker_3)(q_ts)
p_marker_3_ts = g_marker_3_ts[:, :3, 3] + g_marker_3_ts[:, :3, :3] @ jnp.array(
    [0.0, 0.0, 0.02]
)


forward_kinematics_marker_4 = jax.jit(
    partial(
        robot.forward_kinematics,
        s=robot.length,
    )
)
g_marker_4_ts = jax.vmap(forward_kinematics_marker_4)(q_ts)
p_marker_4_ts = g_marker_4_ts[:, :3, 3] + g_marker_4_ts[:, :3, :3] @ jnp.array(
    [0.008, 0.0, 0.0]
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
