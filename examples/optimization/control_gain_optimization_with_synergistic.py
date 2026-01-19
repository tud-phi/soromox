import os
import pickle
import time

import jax
import matplotlib.pyplot as plt
import numpy as onp
import optax
import scipy.io as sio
from jax import Array, jit, value_and_grad, vmap
from jax import numpy as jnp

from soromox.control import (
    OperationalSpaceSynergisticController,
    PIDControl,
    PIDControllerState,
    ReferenceTrajectory,
)
from soromox.coordinate_transformations import OperationalSpaceDynamics
from soromox.rendering import Open3DRenderer
from soromox.systems import SystemState, TendonActuatedPCS

jax.config.update("jax_enable_x64", True)  # double precision


def evaluate_closed_loop_system(
    opt_vars: dict[str, Array],
    controller: OperationalSpaceSynergisticController,
) -> tuple[Array, dict[str, Array]]:
    # Update optimization parameters
    ctr_params = opt_vars["opt_ctr_params"]
    controller = controller.update_gains(ctr_params)

    # Run simulation
    trajectory = robot.rollout_closed_loop_to(
        initial_state=initial_state,
        controller=controller,
        t1=t1,
        solver_dt=solver_dt,
        save_ts=save_ts,
        max_steps=num_ts + 1,
    )

    # Extract results
    t_ts = trajectory.t
    q_ts, qd_ts = jnp.split(trajectory.y, 2, axis=1)

    # Compute loss
    settling_idx = int(len(t_ts) / 2)

    e_ts = jnp.linalg.norm(x_des_ts - q_ts, axis=1)

    loss1 = jnp.trapezoid(e_ts[:settling_idx] ** 2, t_ts[:settling_idx])
    loss2 = jnp.trapezoid(e_ts[settling_idx:] ** 2, t_ts[settling_idx:])

    loss = 1e3 * loss1 + 5e3 * loss2

    # Store auxiliary information
    aux = {"t_ts": t_ts, "q_ts": q_ts, "qd_ts": qd_ts, "u_ts": trajectory.u}

    return loss, aux


# =========================================================================
# Setup
# =========================================================================
num_segments = 1
radius = 2e-2  # Segment radius [m]
rho = 1070 * jnp.ones((num_segments,))  # Volumetric density [kg/m^3]

params = {
    # Base pose: robot pointing upward in z-direction
    "p0": jnp.array([jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, 0.0]),
    "L": 1e-1 * jnp.ones((num_segments,)),  # Segment length [m]
    "r": radius * jnp.ones((num_segments,)),  # Segment radius [m]
    "rho": rho,
    "g": jnp.array([0.0, 0.0, 9.81]),  # Gravity vector [m/s^2]
    "E": 2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
    "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
}
# Damping matrix
params["D"] = 1e-3 * jnp.diag(
    (
        jnp.repeat(jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), num_segments, axis=0)
        * params["L"][:, None]
    ).flatten()
)

# Tendon routing: 3 tendons at 120 degrees apart, parallel to backbone
# Tendon positions at angles: 0°, 120°, 240° from the +z axis
# d_y = r * sin(theta), d_z = r * cos(theta)
tendon_distance = 0.8 * radius  # Slightly inside the robot radius
angles_deg = jnp.array([0.0, 120.0, 240.0])
angles_rad = jnp.deg2rad(angles_deg)

active_tendon_routing_params = {
    "ry": tendon_distance * jnp.sin(angles_rad),  # y-coordinate of tendons
    "rz": tendon_distance * jnp.cos(angles_rad),  # z-coordinate of tendons
    "my": jnp.zeros(3),  # Parallel to backbone (no y-slope)
    "mz": jnp.zeros(3),  # Parallel to backbone (no z-slope)
    "idx_seg_att": jnp.array([0, 0, 0]),  # All attached to segment 0 (the only segment)
}

robot = TendonActuatedPCS(
    num_segments=num_segments,
    params=params,
    active_tendon_routing_params=active_tendon_routing_params,
)

num_dofs = robot.num_active_strains
num_actuators = robot.num_actuators
total_length = float(jnp.sum(params["L"]))
print(f"Number of DOFs: {num_dofs}")
print(f"Number of actuators: {num_actuators}")
print(f"Total robot length: {total_length:.3f} m")

# =========================================================================
# Operational Space Definition
# =========================================================================
# Define the operational space as the end-effector position (x, y, z)
# We place the task point at the tip of the robot (s = L_total)
s_ps = jnp.array([total_length])

# For PCS robots, the Jacobian has shape (6, num_dofs) for each point:
# [omega_x, omega_y, omega_z, v_x, v_y, v_z]
# We select only the linear velocity components (position tracking)
# task_selector selects [v_x, v_y, v_z] which are indices 3, 4, 5
task_selector = jnp.array([False, False, False, True, True, True])

# Create operational space dynamics
osd = OperationalSpaceDynamics(
    robot=robot,
    s_ps=s_ps,
    task_selector=task_selector,
)

print(f"Operational space dimension (velocity): {osd.n_operational_space}")
print(f"Operational space dimension (pose): {osd.n_pose_operational_space}")
print(f"Full pose dimension: {osd.n_points * osd.n_pose_dim}")

# =========================================================================
# Find Statically-Feasible Setpoint
# =========================================================================
print("\nFinding steady-state configuration under constant actuation...")

# Apply constant tendon tensions (negative = pulling)
# Use asymmetric tensions to create an interesting configuration
u_constant = jnp.array([-0.5, -0.2, -0.1])
print(f"  Applied tendon tensions: {u_constant}")

# Define simulation parameters
sim_duration = 10.0
solver_dt = 1e-4

# Start from rest configuration
q0 = jnp.zeros(num_dofs)
qd0 = jnp.zeros(num_dofs)
y0 = jnp.concatenate([q0, qd0])

initial_state = SystemState(t=jnp.array(0.0), y=y0)

# Simulate with constant actuation
trajectory = robot.rollout_to(
    initial_state=initial_state,
    u=u_constant,
    t1=sim_duration,
    solver_dt=solver_dt,
    save_dt=0.1,  # Save less frequently for efficiency
)

# Extract final state
q_final, qd_final = jnp.split(trajectory.y[-1], 2)

# Check that velocities are small (steady-state reached)
velocity_norm = jnp.linalg.norm(qd_final)
print(f"  Final velocity norm: {velocity_norm:.2e}")
if velocity_norm > 1e-3:
    print(
        "  Warning: Steady-state may not have been reached. "
        "Consider increasing sim_duration."
    )

q_des = q_final

print(f"  Steady-state configuration (q_des): {q_des}")

# Get the initial FULL end-effector pose for reference
# IMPORTANT: The reference trajectory must provide FULL poses (all components),
# even though only some components (e.g., position) are part of the task.
# The controller will ignore components not in the task.
x0_full = osd.operational_space_poses(q0)  # shape: (n_points * n_pose_dim,)
x_des_full = osd.operational_space_poses(q_des)  # shape: (n_points * n_pose_dim,)

# Compute the corresponding tendon lengths at steady-state
print(f"  Initial end-effector pose (full): {x0_full}")
print(f"  Desired end-effector pose (full): {x_des_full}")

# Time parameters for closed-loop control
t0, t1 = 0.0, 5.0

# =========================================================================
# Define the Constant Reference
# =========================================================================
save_ts = robot._compute_save_times(t0, t1, solver_dt=solver_dt, save_dt=solver_dt)
num_ts = save_ts.shape[0]

reference_trajectory = ReferenceTrajectory(
    ts=save_ts,
    x_des_fn=lambda t: x_des_full,
    rotation_representation=osd.rotation_representation,
    n_points=osd.n_points,
    is_planar=osd.is_planar,
)
x_des_ts = reference_trajectory.x_des_ts

# =========================================================================
# Define the Controller
# =========================================================================
Kp = 10.0 * jnp.ones((osd.n_operational_space,))
Ki = 2.0 * jnp.ones((osd.n_operational_space,))
Kd = 0.25 * jnp.ones((osd.n_operational_space,))

# Define PID controller
pid_control = PIDControl(Kp, Ki, Kd)

# Create the synergistic controller
controller = OperationalSpaceSynergisticController(
    operational_space_dynamics=osd,
    reference_trajectory=reference_trajectory,
    pid_control=pid_control,
)

# =========================================================================
# Run Simulation
# =========================================================================
# Initial control state (sized for actuated coordinates)
control_state_0 = PIDControllerState.zero(osd.n_operational_space)

# Create initial system state
initial_state = SystemState(
    t=jnp.array(t0),
    y=y0,
    u=jnp.zeros(num_actuators),
    control_state=control_state_0,
)

# =====================================================
# Optimization
# =====================================================
# Control parameters
control_params = {"Kp": Kp, "Ki": Ki, "Kd": Kd}

# Initialize optimization variables
opt_keys_ctr = ["Kp", "Ki", "Kd"]
opt_keys_atr = []

opt_ctr_params = {k: control_params[k] for k in opt_keys_ctr}
opt_atr_params = {k: active_tendon_routing_params[k] for k in opt_keys_atr}

opt_vars = {"opt_ctr_params": opt_ctr_params, "opt_atr_params": opt_atr_params}

var_to_label_map = {
    "Kp": "P_gain",
    "Ki": "I_gain",
    "Kd": "D_gain",
}
param_labels = {
    "opt_ctr_params": {k: var_to_label_map[k] for k in opt_keys_ctr},
    "opt_atr_params": {k: var_to_label_map[k] for k in opt_keys_atr},
}
optimizer = optax.multi_transform(
    {
        "P_gain": optax.chain(
            optax.clip_by_global_norm(10.0), optax.yogi(learning_rate=0.5)
        ),
        "I_gain": optax.chain(
            optax.clip_by_global_norm(10.0), optax.yogi(learning_rate=0.25)
        ),
        "D_gain": optax.chain(
            optax.clip_by_global_norm(10.0), optax.yogi(learning_rate=0.05)
        ),
    },
    param_labels,
)
opt_state = optimizer.init(opt_vars)

# Gradient function
gradient_fn = jit(value_and_grad(evaluate_closed_loop_system, 0, has_aux=True))


# Update function of the optimization variables
def update(opt_vars, opt_state):
    # Compute gradient
    (loss, aux), grad = gradient_fn(opt_vars, controller)

    # Compute the update of the optimization variables
    updates, opt_state = optimizer.update(grad, opt_state, params=opt_vars)

    # Apply the update
    opt_vars = optax.apply_updates(opt_vars, updates)

    return opt_vars, opt_state, loss, aux


print("Optimization variables = ", list(opt_vars.keys()))

# Optimization loop
num_iters = 2

# Optimization-wise variables
time_iter = []
loss_tot = []
opt_vars_tot = []

# Simulation-wise variables
t_ts_tot = []
q_ts_tot = []
qd_ts_tot = []
u_ts_tot = []

print("\nStarting optimization...")

# Track time
compile_start = time.time()

# Step
opt_vars, opt_state, loss, aux = update(opt_vars, opt_state)
jax.block_until_ready(aux)

compile_end = time.time()
print(
    f"\n[INFO] Compilation + first execution time = {compile_end - compile_start:.2f} s"
)

# Store data
time_iter.append(compile_end - compile_start)
loss_tot.append(loss)
t_ts_tot.append(aux["t_ts"])
q_ts_tot.append(aux["q_ts"])
qd_ts_tot.append(aux["qd_ts"])
u_ts_tot.append(aux["u_ts"])
opt_vars_tot.append(opt_vars)

i = 1
while i < num_iters:
    # Start of iteration
    iter_start = time.time()

    # Step
    opt_vars, opt_state, loss, aux = update(opt_vars, opt_state)
    jax.block_until_ready(aux)

    # End of iteration
    iter_end = time.time()
    iter_duration = iter_end - iter_start

    # Check validity
    if jnp.isnan(aux["q_ts"]).any():
        print(f"\n[WARNING] NaN detected at iteration {i}. Stopping.")
        break

    # Store data
    time_iter.append(iter_duration)
    loss_tot.append(loss)
    t_ts_tot.append(aux["t_ts"])
    q_ts_tot.append(aux["q_ts"])
    qd_ts_tot.append(aux["qd_ts"])
    u_ts_tot.append(aux["u_ts"])
    opt_vars_tot.append(opt_vars)

    # Log
    t_left = (num_iters - i - 1) * sum(time_iter[1:]) / len(time_iter[1:])  # [s]
    eta = time.localtime(time.time() + t_left)
    print(
        f"Completion: {100 * (i + 1) / num_iters:3.1f} %  |  iteration {i + 1:>4d} of "
        + f"{num_iters:<4d}  |  iter time = {iter_duration:>.2f} s  |  ETA = {eta.tm_mday:02d}"
        + f"/{eta.tm_mon:02d}/{eta.tm_year} {eta.tm_hour:02d}:{eta.tm_min:02d}",
        end="\r",
    )

    # Update counter
    i += 1

# Log
print(
    f"Completion: {100 * i / num_iters:3.1f} %  |  iteration {i:>4d} of {num_iters:<4d}"
    "  |  iter time = {iter_duration:>.4f} s"
)

# Cast to Array type
time_iter = jnp.array(time_iter)  # (num_iters,)
loss_tot = jnp.array(loss_tot)  # (num_iters,)
t_ts_tot = jnp.array(t_ts_tot)  # (num_iters, num_ts)
q_ts_tot = jnp.array(q_ts_tot)  # (num_iters, num_ts, num_dofs)
qd_ts_tot = jnp.array(qd_ts_tot)  # (num_iters, num_ts, num_dofs)
u_ts_tot = jnp.array(u_ts_tot)  # (num_iters, num_ts, num_actuators)

# Chosen iteration to display
best_idx_opt = int(jnp.argmin(loss_tot))
print(
    "Optimization Finished. Best optimization variables:\n", opt_vars_tot[best_idx_opt]
)

# Extract results for comparison
t_ts = t_ts_tot[0, :]
q_ts_init = q_ts_tot[0, :, :]
qd_ts_init = qd_ts_tot[0, :, :]
u_ts_init = u_ts_tot[0, :, :]
q_ts_best = q_ts_tot[best_idx_opt, :, :]
qd_ts_best = qd_ts_tot[best_idx_opt, :, :]
u_ts_best = u_ts_tot[best_idx_opt, :, :]

# Compute FULL operational space coordinates along the trajectory (for visualization)
x_ts_init = vmap(osd.operational_space_poses)(q_ts_init)
x_ts_best = vmap(osd.operational_space_poses)(q_ts_best)

# Extract position components for plotting (indices 3, 4, 5 for 3D PCS)
# For 3D robots with ROTATION_VECTOR: full pose = [rot_x, rot_y, rot_z, p_x, p_y, p_z]
pos_indices = jnp.arange(6)  # All components

pos_ts_init = x_ts_init[:, pos_indices]
pos_ts_best = x_ts_best[:, pos_indices]


# =====================================================
# Compute metrics
# =====================================================
def compute_metrics(t: Array, x: Array, x_des: Array, pos_indices: list) -> dict:
    """Compute performance metrics for the simulation results."""

    # Compute tracking error
    tracking_error = x_des[None, :] - x

    # Root mean square error for each strain
    rmse = jnp.sqrt(jnp.mean(tracking_error[:, pos_indices] ** 2, axis=0))

    # Maximum absolute error
    max_error = jnp.max(jnp.abs(tracking_error[:, pos_indices]), axis=0)

    # Steady-state error (last 10% of simulation)
    n_ss = max(1, len(t) // 10)
    ss_error = jnp.mean(jnp.abs(tracking_error[-n_ss:, pos_indices]), axis=0)

    return {
        "rmse": rmse,
        "max_error": max_error,
        "ss_error": ss_error,
        "tracking_error": tracking_error,
    }


# Analyze all operational-space variables (for one point: 6 degrees of freedom)
pos_indices = list(pos_indices)

metrics_init = compute_metrics(t_ts, x_ts_init, x_des_full, pos_indices)
metrics_best = compute_metrics(t_ts, x_ts_best, x_des_full, pos_indices)

# =====================================================
# Plot
# =====================================================
savefigs = False

# Create figure for loss
plt.figure()
plt.plot(loss_tot, linewidth=2)
plt.grid(True)
plt.xlabel("Iterations [-]")
plt.ylabel("Loss")
plt.xlim([0, len(loss_tot)])
plt.tight_layout()
if savefigs:
    plt.savefig("loss.pdf", dpi=200, bbox_inches="tight")
    print("\nPlots saved to:")
    print("  - loss.pdf")
plt.show()

# Select key operational variable to plot (tip positions)
plot_task_indices = [3, 4, 5]  # p_x, p_y, p_z
plot_task_names = [
    r"$p_x$ [m]",
    r"$p_y$ [m]",
    r"$p_z$ [m]",
]

# Create figure for strain tracking comparison
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
color = "#0F0CBB"

for idx, (task_idx, task_name) in enumerate(zip(plot_task_indices, plot_task_names)):
    ax = axes[idx]

    # Plot desired setpoint
    ax.axhline(
        y=x_des_full[task_idx],
        color=color,
        linestyle="--",
        linewidth=2,
        label="Desired",
        alpha=0.7,
    )

    # Plot trajectory of initial system
    ax.plot(
        t_ts,
        x_ts_init[:, task_idx],
        color=color,
        linestyle="-.",
        linewidth=1.5,
        label="Initial",
    )

    # Plot trajectory of optimized system
    ax.plot(
        t_ts,
        x_ts_best[:, task_idx],
        color=color,
        linestyle="-",
        linewidth=1.2,
        label="Optimized",
    )

    ax.set_ylabel(f"{task_name}")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t0, t1])

    if idx == 0:
        ax.legend(loc="upper right", fontsize=9)

axes[-1].set_xlabel("Time [s]")
axes[0].set_title("Operational-Space Setpoint Regulation: Position Tracking")

plt.tight_layout()
if savefigs:
    plt.savefig(
        "operational_space_regulation_tracking.pdf", dpi=200, bbox_inches="tight"
    )
    print("  - operational_space_regulation_tracking.pdf")
plt.show()

# Create figure for tracking errors
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
color = "#DA0909"

for idx, (task_idx, task_name) in enumerate(zip(plot_task_indices, plot_task_names)):
    ax = axes[idx]

    tracking_error_init = metrics_init["tracking_error"]
    ax.plot(
        t_ts,
        tracking_error_init[:, task_idx],
        linestyle="-.",
        color=color,
        linewidth=1.5,
        label="Initial",
    )

    tracking_error_best = metrics_best["tracking_error"]
    ax.plot(
        t_ts,
        tracking_error_best[:, task_idx],
        linestyle="-",
        color=color,
        linewidth=1.2,
        label="Optimized",
    )

    ax.set_ylabel(f"Error {task_name}")
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t0, t1])

    if idx == 0:
        ax.legend(loc="upper right", fontsize=9)

axes[-1].set_xlabel("Time [s]")
axes[0].set_title("Operational-Space Setpoint Regulation: Tracking Error Comparison")

plt.tight_layout()
if savefigs:
    plt.savefig("operational_space_regulation_errors.pdf", dpi=200, bbox_inches="tight")
    print("  - operational_space_regulation_errors.pdf")
plt.show()

# Create figure for control inputs (tendon tensions)
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
tendon_names = ["Tendon 1 (0°)", "Tendon 2 (120°)", "Tendon 3 (240°)"]
color = "#0DAA54"

for idx in range(num_actuators):
    ax = axes[idx]

    # Plot reference steady-state actuation
    ax.axhline(
        y=u_constant[idx],
        color=color,
        linestyle="--",
        linewidth=2,
        label="Reference (steady-state)" if idx == 0 else None,
        alpha=0.7,
    )

    ax.plot(
        t_ts,
        u_ts_init[:, idx],
        color=color,
        linestyle="-.",
        linewidth=1.5,
        label="Initial",
    )

    ax.plot(
        t_ts,
        u_ts_best[:, idx],
        color=color,
        linestyle="-",
        linewidth=1.2,
        label="Optimized",
    )

    ax.set_ylabel(f"{tendon_names[idx]} [N]")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t0, t1])

    if idx == 0:
        ax.legend(loc="upper right", fontsize=9)

axes[-1].set_xlabel("Time [s]")
axes[0].set_title(
    "Operational-Space Setpoint Regulation: Control Inputs (Tendon Tensions)"
)

plt.tight_layout()
if savefigs:
    plt.savefig("operational_space_regulation_inputs.pdf", dpi=200, bbox_inches="tight")
    print("  - operational_space_regulation_inputs.pdf")
plt.show()

# Create bar chart for RMSE comparison
fig, ax = plt.subplots(figsize=(10, 6))

names = ["Initial", "Best"]
x = jnp.arange(len(names))
width = 0.25

for i, (task_idx, task_name) in enumerate(zip(plot_task_indices, plot_task_names)):
    rmse_values = [metrics_init["rmse"][task_idx], metrics_best["rmse"][task_idx]]
    bars = ax.bar(x + i * width, rmse_values, width, label=task_name)

ax.set_xlabel("Optimization stage")
ax.set_ylabel("RMSE")
ax.set_title("Operational-Space Setpoint Regulation: RMSE Comparison")
ax.set_xticks(x + width)
ax.set_xticklabels(names, rotation=15, ha="right")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
if savefigs:
    plt.savefig("operational_space_regulation_rmse.pdf", dpi=200, bbox_inches="tight")
    print("  - operational_space_regulation_rmse.pdf")
plt.show()

if Open3DRenderer is None:
    print("Open3DRenderer unavailable. Install open3d to view the animation.")
else:
    render_name = "Optimized System"
    renderer = Open3DRenderer(robot, num_points=50)
    renderer.render_sequence(
        ts=t_ts,
        q_ts=q_ts_best,
        playback_speed=1.0,
        window_name=f"Operational-Space Regulation ({render_name})",
    )

# =====================================================
# Save
# =====================================================
save_dir = os.path.join("data", "trial_1")
os.makedirs(save_dir, exist_ok=True)

save_path = str(save_dir)


def to_np(x):
    return onp.array(x)


mat_data = {}


# Optimization
def flatten_params_to_dict(params_tree, prefix=""):
    """
    Helper: Flatten params_tree.

    :param params_tree: Description
    :param prefix: Description
    """
    flat_dict = {}
    leaves, _ = jax.tree_util.tree_flatten_with_path(params_tree)
    for path, val in leaves:
        key_str = "_".join([str(p.key) if hasattr(p, "key") else str(p) for p in path])
        full_key = f"{prefix}{key_str}"
        flat_dict[full_key] = to_np(val)
    return flat_dict


mat_data["history_loss"] = to_np(loss_tot)
mat_data["history_time"] = to_np(time_iter)
stacked_history_tree = jax.tree_util.tree_map(
    lambda *xs: jnp.stack(xs, axis=0), *opt_vars_tot
)
mat_data.update(
    flatten_params_to_dict(stacked_history_tree, prefix="history_opt_vars_")
)

# Simulation
mat_data["t_ts"] = to_np(t_ts)

mat_data["q_ts_init"] = to_np(q_ts_init)
mat_data["qd_ts_init"] = to_np(qd_ts_init)
mat_data["u_ts_init"] = to_np(u_ts_init)
mat_data["x_ts_init"] = to_np(x_ts_init)

mat_data["q_ts_best"] = to_np(q_ts_best)
mat_data["qd_ts_best"] = to_np(qd_ts_best)
mat_data["u_ts_best"] = to_np(u_ts_best)
mat_data["x_ts_best"] = to_np(x_ts_best)

# Metadata
mat_data["num_iters"] = to_np(num_iters)
mat_data["x_des_ts"] = to_np(x_des_ts)
mat_data["best_idx_opt"] = to_np(best_idx_opt)

# Save main summary .mat file
mat_filename = os.path.join(save_dir, "optimization_results.mat")
print(f"Saving data to {mat_filename}")
sio.savemat(mat_filename, mat_data, do_compression=True)

# Save Data for Python Animation (Pickle)
animation_dict = {
    # Robot parameters
    "params": params,
    "active_tendon_routing_params": active_tendon_routing_params,
    "passive_tendon_routing_params": {},
    "passive_tendon_params": {},
    "num_segments": num_segments,
    # Optimized Parameters
    "best_opt_vars": opt_vars_tot[best_idx_opt],
    # Trajectory Data
    "t_ts": t_ts,
    "q_ts": q_ts_best,
    # Target
    "x_des_ts": x_des_ts,
}

anim_filename = os.path.join(save_dir, "animation_data.pkl")
with open(anim_filename, "wb") as f:
    pickle.dump(animation_dict, f)

print("Save complete.")  # '''
