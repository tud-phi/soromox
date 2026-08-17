import math
import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import optax
from jax import Array, jit, value_and_grad, vmap
from jax import numpy as jnp

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from control_gain_optimization_common import (  # noqa: E402
    finish_figure,
    parse_args,
    prepare_output_dirs,
    save_optimization_outputs,
)
from gain_optimization_loop import run_gain_optimization  # noqa: E402

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.control import PIDControl, PIDControllerState, ReferenceTrajectory
from soromox.control.actuation_space import PotentialCompensationRegulator
from soromox.coordinate_transformations import ActuationSpaceDynamics
from soromox.rendering import Open3DRenderer
from soromox.systems import (
    PCS,
    PCSParams,
    SystemState,
)

jax.config.update("jax_enable_x64", True)  # double precision

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
OUTPUTS_ROOT = CASE_DIR / "outputs"

ARGS = parse_args(
    description="Optimize collocated control gains for the Section Vd study.",
    default_result_dir=DATA_DIR / "collocated",
    default_output_dir=OUTPUTS_ROOT / "collocated_diagnostics",
)
RESULT_DIR, OUTPUTS_DIR = prepare_output_dirs(ARGS)
if (
    not math.isfinite(ARGS.integral_error_saturation_scale)
    or ARGS.integral_error_saturation_scale <= 0.0
):
    raise ValueError(
        "--integral-error-saturation-scale must be finite and strictly positive"
    )


def evaluate_closed_loop_system(
    opt_vars: dict[str, Array],
    controller: PotentialCompensationRegulator,
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

    # Track the configuration-space setpoint the controller regulates
    e_sq_ts = jnp.sum((x_des_ts - q_ts) ** 2, axis=1)

    loss1 = jnp.trapezoid(e_sq_ts[:settling_idx], t_ts[:settling_idx])
    loss2 = jnp.trapezoid(e_sq_ts[settling_idx:], t_ts[settling_idx:])

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

segment_lengths = 1e-1 * jnp.ones((num_segments,))
# Damping matrix
damping_matrix = 1e-3 * jnp.diag(
    (
        jnp.repeat(jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), num_segments, axis=0)
        * segment_lengths[:, None]
    ).flatten()
)
body_params = PCSParams(
    base_pose=jnp.array([0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0]),
    length=segment_lengths,
    radius=radius * jnp.ones((num_segments,)),
    density=rho,
    gravity=jnp.array([0.0, 0.0, 9.81]),
    young_modulus=2e3 * jnp.ones((num_segments,)),
    shear_modulus=1e3 * jnp.ones((num_segments,)),
    damping_matrix=damping_matrix,
    reference_strain=jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), num_segments),
)

# Tendon routing: 3 tendons at 120 degrees apart, parallel to backbone
# Tendon positions at angles: 0°, 120°, 240° from the +z axis
# d_y = r * sin(theta), d_z = r * cos(theta)
tendon_offset_radius = 0.8 * radius  # Slightly inside the robot radius
angles_deg = jnp.array([0.0, 120.0, 240.0])
angles_rad = jnp.deg2rad(angles_deg)

active_tendon_routing = ThreadlikeRouting.linear(
    intercept=tendon_offset_radius
    * jnp.stack(
        (jnp.zeros_like(angles_rad), jnp.sin(angles_rad), jnp.cos(angles_rad)),
        axis=-1,
    ),
    start_segment_index=0,
    end_segment_index=(0, 0, 0),
)

robot = PCS(
    params=body_params,
    actuators=ThreadlikeActuator.tendons(active_tendon_routing),
)
params = robot.params
total_length = float(jnp.sum(segment_lengths))
num_dofs = robot.num_active_strains
num_actuators = robot.num_actuators

print(f"Number of DOFs: {num_dofs}")
print(f"Number of actuators: {num_actuators}")

# Create actuation space dynamics transformer
asd = ActuationSpaceDynamics(robot)

# =========================================================================
# Find Statically-Feasible Setpoint
# =========================================================================
print("\nFinding steady-state configuration under constant actuation...")

# Apply positive tendon tensions.
# Use asymmetric tensions to create an interesting configuration
u_constant = jnp.array([0.5, 0.2, 0.1])
print(f"  Applied tendon tensions: {u_constant}")

# Define simulation parameters
sim_duration = 10.0
solver_dt = 1e-4

# Start from rest configuration
q0 = jnp.zeros((num_dofs,))
qd0 = jnp.zeros((num_dofs,))
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
    raise RuntimeError(
        "Setpoint generation did not reach steady state "
        f"(final velocity norm {float(velocity_norm):.3e}). Refusing to optimize "
        "gains for a transient target; increase sim_duration or choose an "
        "actuation away from routing degeneracies."
    )

q_des = q_final

print(f"  Steady-state configuration (q_des): {q_des}")

# Compute the corresponding tendon lengths at steady-state
y_des = asd.actuated_unactuated_coordinates(q_des)
print(f"  Actuation-space coordinates (y_des): {y_des}")

# A collapsed tendon has a nondifferentiable length coordinate: its first
# derivative is direction-dependent and its second derivative grows like the
# inverse path length. Reverse-mode sensitivities can therefore overflow even
# while the forward trajectory stays finite. Keep the generated reference well
# away from that geometric singularity.
minimum_tendon_length = jnp.min(jnp.abs(y_des[:num_actuators]))
minimum_tendon_length_threshold = 1e-2 * total_length
if minimum_tendon_length <= minimum_tendon_length_threshold:
    raise RuntimeError(
        "Setpoint generation placed a tendon too close to path collapse "
        f"(minimum length {float(minimum_tendon_length):.3e} m; required above "
        f"{minimum_tendon_length_threshold:.3e} m). Choose smaller reference "
        "tensions before differentiating the rollout."
    )

# Time parameters for closed-loop control
t0, t1 = 0.0, 5.0

# =========================================================================
# Define the Constant Reference
# =========================================================================
save_ts = robot._compute_save_times(t0, t1, solver_dt=solver_dt, save_dt=solver_dt)
num_ts = save_ts.shape[0]

reference_trajectory = ReferenceTrajectory(ts=save_ts, x_des_fn=lambda t: q_des)
x_des_ts = reference_trajectory.x_des_ts

# =========================================================================
# Define the Controller
# =========================================================================
# Units: tendon lengths are in meters
Kp = 5e1 * jnp.ones((num_actuators,))
Ki = 5e0 * jnp.ones((num_actuators,))
Kd = 1e0 * jnp.ones((num_actuators,))

# The committed legacy trajectories start from tendon lengths of -100 mm and
# target [-85.49, -96.41, -100.05] mm. The largest initial error is 14.51 mm,
# whereas all other observed initial/transient errors remain below 3.85 mm.
# A 10 mm error scale therefore limits integral accumulation during the large
# reference step while leaving the ordinary error regime nearly linear.
tendon_error_saturation_scale = ARGS.integral_error_saturation_scale  # [m]
tendon_error_gamma = 1.0 / tendon_error_saturation_scale  # [1/m]

pid_control = PIDControl(
    Kp=Kp,
    Ki=Ki,
    Kd=Kd,
    saturation_fn="tanh",
    gamma=tendon_error_gamma,
)

controller = PotentialCompensationRegulator(
    actuation_space_dynamics=asd,
    reference_trajectory=reference_trajectory,
    pid_control=pid_control,
)

# =========================================================================
# Run Simulation
# =========================================================================
# Initial control state (sized for actuated coordinates)
control_state_0 = PIDControllerState.zero(num_actuators)

# Create initial system state
initial_state = SystemState(
    t=jnp.array(t0),
    y=y0,
    u=jnp.zeros((num_actuators,)),
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
opt_atr_params = {}

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

# Gradient function
gradient_fn = jit(value_and_grad(evaluate_closed_loop_system, 0, has_aux=True))

print("Optimization variables = ", list(opt_vars.keys()))
print("\nStarting optimization...")

# Optimization loop
history = run_gain_optimization(
    gradient_fn=lambda variables: gradient_fn(variables, controller),
    optimizer=optimizer,
    opt_vars=opt_vars,
    num_iters=ARGS.num_iters,
    progress_label="Collocated",
)

num_iters = len(history)

if num_iters == 0:
    print(f"\n[ERROR] {history.stop_reason}")
    print("No finite iterate was recorded, so there is nothing to save or plot.")
    sys.exit(1)

# Cast to Array type
time_iter = jnp.array(history.time_iter)  # (num_iters,)
loss_tot = jnp.array(history.loss)  # (num_iters,)
t_ts_tot = history.stacked_aux("t_ts")  # (num_iters, num_ts)
q_ts_tot = history.stacked_aux("q_ts")  # (num_iters, num_ts, num_dofs)
qd_ts_tot = history.stacked_aux("qd_ts")  # (num_iters, num_ts, num_dofs)
u_ts_tot = history.stacked_aux("u_ts")  # (num_iters, num_ts, num_actuators)
opt_vars_tot = history.opt_vars

# Chosen iteration to display
best_idx_opt = history.best_index()
print(
    "Optimization Finished. Best optimization variables:\n",
    opt_vars_tot[best_idx_opt],
)

# Extract results for comparison
t_ts = t_ts_tot[0, :]
q_ts_init = q_ts_tot[0, :, :]
qd_ts_init = qd_ts_tot[0, :, :]
u_ts_init = u_ts_tot[0, :, :]
q_ts_best = q_ts_tot[best_idx_opt, :, :]
qd_ts_best = qd_ts_tot[best_idx_opt, :, :]
u_ts_best = u_ts_tot[best_idx_opt, :, :]

# End-effector trajectory
x_ts_init = vmap(robot.forward_kinematics, in_axes=(0, None))(q_ts_init, total_length)[
    :, :3, 3
]  # (p, 3)
x_ts_best = vmap(robot.forward_kinematics, in_axes=(0, None))(q_ts_best, total_length)[
    :, :3, 3
]  # (p, 3)

# =====================================================
# Save
# =====================================================
save_optimization_outputs(
    RESULT_DIR,
    loss_tot=loss_tot,
    time_iter=time_iter,
    opt_vars_tot=opt_vars_tot,
    t_ts=t_ts,
    q_ts_init=q_ts_init,
    qd_ts_init=qd_ts_init,
    u_ts_init=u_ts_init,
    x_ts_init=x_ts_init,
    q_ts_best=q_ts_best,
    qd_ts_best=qd_ts_best,
    u_ts_best=u_ts_best,
    x_ts_best=x_ts_best,
    num_iters=num_iters,
    x_des_ts=x_des_ts,
    best_idx_opt=best_idx_opt,
    params=params,
    active_tendon_routing=active_tendon_routing,
    passive_elements=robot.passive_elements,
    num_segments=num_segments,
)


# =====================================================
# Compute metrics
# =====================================================
def compute_metrics(t: Array, q: Array, q_des: Array, strain_indices: list) -> dict:
    """Compute performance metrics for the simulation results."""

    # Compute tracking error
    tracking_error = q_des[None, :] - q

    # Root mean square error for each strain
    rmse = jnp.sqrt(jnp.mean(tracking_error[:, strain_indices] ** 2, axis=0))

    # Maximum absolute error
    max_error = jnp.max(jnp.abs(tracking_error[:, strain_indices]), axis=0)

    # Steady-state error (last 10% of simulation)
    n_ss = max(1, len(t) // 10)
    ss_error = jnp.mean(jnp.abs(tracking_error[-n_ss:, strain_indices]), axis=0)

    return {
        "rmse": rmse,
        "max_error": max_error,
        "ss_error": ss_error,
        "tracking_error": tracking_error,
    }


# Analyze all strains (for a 1-segment robot: 6 strains)
strain_indices = list(range(num_dofs))

metrics_init = compute_metrics(t_ts, q_ts_init, q_des, strain_indices)
metrics_best = compute_metrics(t_ts, q_ts_best, q_des, strain_indices)

# =====================================================
# Plot
# =====================================================
# Create figure for loss
plt.figure()
plt.plot(loss_tot, linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel("Iterations [-]")
plt.ylabel("Loss")
plt.xlim([0, len(loss_tot)])
plt.tight_layout()
finish_figure(ARGS, OUTPUTS_DIR, "loss.pdf")

# Select key strains to plot (curvatures and axial strain)
plot_strain_indices = [1, 2, 3]  # kappa_y, kappa_z, sigma_x
plot_strain_names = [
    r"$\kappa_y$ [rad/m]",
    r"$\kappa_z$ [rad/m]",
    r"$\sigma_x$ [-]",
]

# Create figure for strain tracking comparison
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
color = "#0F0CBB"

for idx, (strain_idx, strain_name) in enumerate(
    zip(plot_strain_indices, plot_strain_names)
):
    ax = axes[idx]

    # Plot desired setpoint
    ax.axhline(
        y=q_des[strain_idx],
        color=color,
        linestyle="--",
        linewidth=2,
        label="Desired",
        alpha=0.7,
    )

    # Plot trajectory of initial system
    ax.plot(
        t_ts,
        q_ts_init[:, strain_idx],
        color=color,
        linestyle="-.",
        linewidth=1.5,
        label="Initial",
    )

    # Plot trajectory of optimized system
    ax.plot(
        t_ts,
        q_ts_best[:, strain_idx],
        color=color,
        linestyle="-",
        linewidth=1.2,
        label="Optimized",
    )

    ax.set_ylabel(f"{strain_name}")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t0, t1])

    if idx == 0:
        ax.legend(loc="upper right", fontsize=9)

axes[-1].set_xlabel("Time [s]")
axes[0].set_title("Actuation-Space Setpoint Regulation: Strain Tracking Comparison")

plt.tight_layout()
finish_figure(ARGS, OUTPUTS_DIR, "actuation_space_regulation_tracking.pdf")

# Create figure for tracking errors
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
color = "#DA0909"

for idx, (strain_idx, strain_name) in enumerate(
    zip(plot_strain_indices, plot_strain_names)
):
    ax = axes[idx]

    tracking_error_init = metrics_init["tracking_error"]
    ax.plot(
        t_ts,
        tracking_error_init[:, strain_idx],
        linestyle="-.",
        color=color,
        linewidth=1.5,
        label="Initial",
    )

    tracking_error_best = metrics_best["tracking_error"]
    ax.plot(
        t_ts,
        tracking_error_best[:, strain_idx],
        linestyle="-",
        color=color,
        linewidth=1.2,
        label="Optimized",
    )

    ax.set_ylabel(f"Error {strain_name}")
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t0, t1])

    if idx == 0:
        ax.legend(loc="upper right", fontsize=9)

axes[-1].set_xlabel("Time [s]")
axes[0].set_title("Actuation-Space Setpoint Regulation: Tracking Error Comparison")

plt.tight_layout()
finish_figure(ARGS, OUTPUTS_DIR, "actuation_space_regulation_errors.pdf")

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
    "Actuation-Space Setpoint Regulation: Control Inputs (Tendon Tensions)"
)

plt.tight_layout()
finish_figure(ARGS, OUTPUTS_DIR, "actuation_space_regulation_inputs.pdf")

# Create bar chart for RMSE comparison
fig, ax = plt.subplots(figsize=(10, 6))

names = ["Initial", "Best"]
x = jnp.arange(len(names))
width = 0.25

for i, (strain_idx, strain_name) in enumerate(
    zip(plot_strain_indices, plot_strain_names)
):
    rmse_values = [metrics_init["rmse"][strain_idx], metrics_best["rmse"][strain_idx]]
    bars = ax.bar(x + i * width, rmse_values, width, label=strain_name)

ax.set_xlabel("Optimization stage")
ax.set_ylabel("RMSE")
ax.set_title("Actuation-Space Setpoint Regulation: RMSE Comparison")
ax.set_xticks(x + width)
ax.set_xticklabels(names, rotation=15, ha="right")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
finish_figure(ARGS, OUTPUTS_DIR, "actuation_space_regulation_rmse.pdf")

if ARGS.no_render:
    pass
elif Open3DRenderer is None:
    print("Open3DRenderer unavailable. Install open3d to view the animation.")
else:
    render_name = "Optimized System"
    renderer = Open3DRenderer(robot, num_points=50)
    renderer.render_sequence(
        ts=t_ts,
        q_ts=q_ts_best,
        playback_speed=1.0,
        window_name=f"Actuation-Space Regulation ({render_name})",
    )
