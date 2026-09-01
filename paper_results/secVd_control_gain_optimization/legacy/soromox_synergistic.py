import time

import jax
from jax import Array, lax, vmap, jit, value_and_grad
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)

from soromox.systems.tendon_actuated_pcs import TendonActuatedPCS

from functools import partial
from typing import Dict, Tuple

from scipy.spatial.transform import Rotation as R
import numpy as onp

import pickle
import random
import scipy.io as sio
import os

import optax

from utils import algebra as alg
from utils import controllers as ctr


# =====================================================
# Functions
# =====================================================
def saturate_tanh(u, u_max=1.0, gain=1.0):
    return u_max * jnp.tanh(gain * u / u_max)

# ------------------------------
def synergistic_control(
    q: Array,
    qd: Array,
    error_state: Array,
    x_des: Array,
    dt: float,
    Gain_p: Array,
    Gain_i: Array,
    Gain_d: Array,
    robot: TendonActuatedPCS,
) -> Tuple[Array, Array, dict[str, Array]]:
    
    # Synergistic mapping
    A = robot.actuation_matrix(q) # (n, m)
    J = robot.jacobian(q, L_cum)[3:, :] # (3, n)
    M_inv = jnp.linalg.inv(robot.inertia_matrix(q)) # (n, n)

    # 
    alpha0 = 2.
    mu0 = 0.1
    mu = alg.manip_index(J @ M_inv @ A)
    alpha = jnp.where(mu >= mu0, 0.0, alpha0 * (1 - mu / mu0)**2)
    # alpha = 0.0

    P_AM = jnp.linalg.inv(J @ M_inv @ A + alpha * jnp.eye(m)) @ J @ M_inv # (3, n)
    
    # Feed-back action: PD on actuated coordinates
    x = robot.forward_kinematics(q, L_cum)[:3, 3] # (3)
    xd = J @ qd # (1, )
    f, error_state_next = ctr.PID(x, xd, x_des, dt, error_state, Gain_p, Gain_i, Gain_d)
    tau_fb = P_AM @ J.T @ f

    # Total control action
    tau = saturate_tanh(tau_fb, u_max=tau_max)  # (m,)

    # Conditioning number
    k = alg.cond_number(J @ M_inv @ A)

    # Instantaneous power
    ld = A.T @ qd
    P_t = tau @ ld

    # Store auxiliary data
    aux = {"f": f, "k": k, "P_t": P_t}

    return tau, error_state_next, aux

# ------------------------------
def evaluate_closed_loop_system(
        opt_vars: dict[str, Array],
        robot: TendonActuatedPCS,
    ) -> Tuple[Array, Dict[str, Array]]:
    # Update parameters of the robot
    ctr_params = opt_vars["opt_ctr_params"]

    # Control law
    Kp = jnp.diag(ctr_params.get("Kp_", Kp_))
    Ki = jnp.diag(ctr_params.get("Ki_", Ki_))
    Kd = jnp.diag(ctr_params.get("Kd_", Kd_))
    control_fn = partial(synergistic_control, dt=ctrl_dt, Gain_p=Kp, Gain_i=Ki, Gain_d=Kd, robot=robot)

    # Initial condition
    x0 = jnp.concatenate((q0, qd_0))

    def scan_fn(carry, scan_input):
        # Current state
        x, error_state = carry["x_next"], carry["error_state"]
        q, qd = x[:n], x[n:]

        # Current input
        t, x_des = scan_input

        # Control action
        u, error_state_next, aux_ctrl = control_fn(q, qd, error_state, x_des)

        # Solves the system within [t, t + ctrl_dt]
        save_ts = jnp.linspace(t, t + ctrl_dt - sim_dt, num)
        t_ts, q_ts, qd_ts = robot.resolve_upon_time(q0=q, qd0=qd, u=u, t0=t, t1=t + ctrl_dt, dt=sim_dt, saveat_ts=save_ts,  max_steps=max_steps)
        
        # Update next initial point
        x_next = jnp.concatenate([q_ts[-1, :], qd_ts[-1, :]])
        carry["x_next"] = x_next
        carry["error_state"] = error_state_next

        # Store auxiliary information
        k = aux_ctrl["k"]
        P_t = aux_ctrl["P_t"]
        u_ts = jnp.tile(u, (t_ts.shape[0], 1)).squeeze()
        k_ts = jnp.tile(k, (t_ts.shape[0], 1)).squeeze()
        P_t_ts = jnp.tile(P_t, (t_ts.shape[0], 1)).squeeze()

        output = {"t_ts": t_ts, "q_ts": q_ts, "qd_ts": qd_ts, "u_ts": u_ts, "k_ts": k_ts, "P_t_ts": P_t_ts}

        return carry, output

    # Loops over control_ts -> solves the system for the whole sequence
    ts = jnp.arange(t0, t1, ctrl_dt)
    init_carry = dict(x_next=x0, error_state=error_state_init)
    scan_xs = (ts, x_des_ts)
    _, sol = lax.scan(scan_fn, init_carry, scan_xs)

    # Reshape solution
    t_ts = sol['t_ts'].flatten()
    q_ts = sol['q_ts'].reshape(-1, n)
    qd_ts = sol['qd_ts'].reshape(-1, n)
    u_ts = sol['u_ts'].reshape(-1, m)
    k_ts = sol['k_ts'].flatten()
    P_t_ts = sol['P_t_ts'].flatten()

    # Compute loss
    idx = int(len(t_ts) / 2)

    x_ts = vmap(robot.forward_kinematics, in_axes=(0, None))(q_ts, L_cum)[
        :, :3, 3
    ]  # (3, )
    e_ts = jnp.linalg.norm(x_des - x_ts, axis=1)

    loss1 = jnp.trapezoid(e_ts[:idx] ** 2, t_ts[:idx])
    loss2 = jnp.trapezoid(e_ts[idx:] ** 2, t_ts[idx:])

    P_tau = jnp.trapezoid(P_t_ts ** 2, t_ts) # ()
    loss3 = P_tau

    loss = 1e3 * (loss1 + 5 * loss2) + 2.5 * loss3

    # Store auxiliary information
    aux = {"t_ts": t_ts, "q_ts": q_ts, "qd_ts": qd_ts, "u_ts": u_ts, "k_ts": k_ts,
           "P_t_ts": P_t_ts, "loss_components": jnp.array([loss1, loss2, loss3])}

    return loss, aux


# =====================================================
# Robot definition
# =====================================================
num_segments = 2
rho = 1070 * jnp.ones(
    (num_segments,)
)  # Volumetric density of Dragon Skin 20 [kg/m^3]
L_seg = 0.10    # [m]
r_seg = 0.013   # [m]
r_ten = 0.012   # [m]
E = 14e4        # [Pa]
G = E / 3       # [Pa]
D = 1e-4
params = {
    "p0": jnp.array(
        [jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, 0.0]
    ),  # Initial position and orientation
    "L": L_seg * jnp.ones((num_segments,)),
    "r": r_seg * jnp.ones((num_segments,)),
    "rho": rho,
    "g": jnp.array([0.0, 0.0, 9.81]),       # Gravity vector [m/s^2]
    "E": E * jnp.ones((num_segments,)),     # Elastic modulus [Pa] # 8e4
    "G": G * jnp.ones((num_segments,)),     # Shear modulus [Pa] # 1e3
}
params["D"] = D * jnp.diag(                 # 4e-4
    (
        jnp.repeat(
            jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), num_segments, axis=0
        )
        * params["L"][:, None]
    ).flatten()
)
theta = jnp.pi / 32
L = jnp.sum(params["L"])
active_tendon_routing_params = {
    "ry": r_ten * jnp.array([jnp.cos(theta), jnp.cos(theta + 2 * jnp.pi / 3), jnp.cos(theta + 4 * jnp.pi / 3)]),
    "rz": r_ten * jnp.array([jnp.sin(theta), jnp.sin(theta + 2 * jnp.pi / 3), jnp.sin(theta + 4 * jnp.pi / 3)]),
    "my": jnp.zeros(3),
    "mz": jnp.zeros(3),
    "idx_seg_att": jnp.array([1, 1, 1]),
}


# ======================================================
# Robot initialization
# ======================================================
robot = TendonActuatedPCS(
    num_segments=num_segments,
    params=params,
    active_tendon_routing_params=active_tendon_routing_params,
)

# Total length of the robot
L_cum = jnp.sum(robot.L)

# Dimension of the configuration space
n = 6 * num_segments
m = len(active_tendon_routing_params["idx_seg_att"])
o = 3


# =====================================================
# Simulation parameters
# =====================================================
# Initial configuration
q0 = jnp.zeros(n)

# Initial velocities
qd_0 = jnp.zeros_like(q0)

# Initial end-effector position
x0 = robot.forward_kinematics(q0, L)[:3, 3] # (3,)

# =====================================================
# Time
# =====================================================
t0 = 0.0
t1 = 5.0
sim_dt = 1e-4
ctrl_dt = 1e-2

# Steps to save
save_every_n_steps = 1
num = int(ctrl_dt / (sim_dt * save_every_n_steps))

x_des = jnp.array([-0.055, -0.094, 0.118])
x_des_ts = jnp.tile(x_des, (int((t1 - t0) / ctrl_dt), 1))
x_des_ts_full = x_des

# Max steps for auto-differentiation
max_steps = int((t1 - t0) / sim_dt) + 1


# =====================================================
# Closed-loop simulation
# =====================================================
# PID gains
Kp_ = jnp.ones(m) * 40.
Ki_ = jnp.ones(m) * 20.
Kd_ = jnp.ones(m) * 2.

# Inital state of the PID error
error_state_init = jnp.zeros(m)

# Maximum allowed torque
tau_max = 30.


# =====================================================
# Optimization parameters
# =====================================================
# Control parameters
control_params = {"Kp_": Kp_, "Ki_": Ki_, "Kd_": Kd_}

# Initialize optimization variables
opt_keys_ctr = ["Kp_", "Ki_", "Kd_"]
opt_keys_atr = []

opt_ctr_params = {k: control_params[k] for k in opt_keys_ctr}
opt_atr_params = {k: active_tendon_routing_params[k] for k in opt_keys_atr}

opt_vars_physical = {
    "opt_ctr_params": opt_ctr_params, "opt_atr_params": opt_atr_params
}

# Define box constraints
threshold_box_maps = {
    "opt_ctr_params": {
        "Kp_": jnp.array([0.0, 200.0]),
        "Ki_": jnp.array([0.0, 100.0]),
        "Kd_": jnp.array([0.0, 10.0]),
    },
    "opt_atr_params": {
        "ry0": jnp.array([-r_seg, r_seg]),
        "rz0": jnp.array([-r_seg, r_seg]),
        "ry1": jnp.array([-r_seg, r_seg]),
        "rz1": jnp.array([-r_seg, r_seg]),
    }
}

def _get_bound_from_path(bounds, path):
    """Helper to navigate the bounds tree using a path from the params tree."""
    node = bounds
    try:
        for p in path:
            # Handle different JAX versions for DictKey/SequenceKey
            key = p.key if hasattr(p, 'key') else (p.idx if hasattr(p, 'idx') else p)
            node = node[key]
        return node
    except (KeyError, TypeError, IndexError, AttributeError):
        return None

def scale_tree(params, bounds):
    """
    Scales params to [0, 1] based on bounds. 
    Robust to 'bounds' containing extra keys that 'params' does not have.
    """
    leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
    new_leaves = []
    
    for path, val in leaves:
        bound = _get_bound_from_path(bounds, path)
        
        if bound is not None:
            # Apply Min-Max scaling: (x - min) / (max - min)
            min_v, max_v = bound[0], bound[1]
            scaled_val = (val - min_v) / (max_v - min_v)
            new_leaves.append(scaled_val)
        else:
            # If no bound is found, return the value as-is (or raise error)
            new_leaves.append(val)
            
    return jax.tree_util.tree_unflatten(treedef, new_leaves)

def unscale_tree(params, bounds):
    """
    Unscales params from [0, 1] back to original domain.
    Robust to 'bounds' containing extra keys that 'params' does not have.
    """
    leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
    new_leaves = []
    
    for path, val in leaves:
        bound = _get_bound_from_path(bounds, path)
        
        if bound is not None:
            # Apply Inverse Scaling: x * (max - min) + min
            min_v, max_v = bound[0], bound[1]
            unscaled_val = val * (max_v - min_v) + min_v
            new_leaves.append(unscaled_val)
        else:
            new_leaves.append(val)
            
    return jax.tree_util.tree_unflatten(treedef, new_leaves)

# Batch initialization (Nominal + Random)
def generate_mixed_batch(key, batch_size, nominal_vars, limits):
    """
    Generates a batch based strictly on the structure of 'nominal_vars'.
    
    1. Iterates only keys present in 'nominal_vars'.
    2. Looks up corresponding bounds in 'limits' using the PyTree path.
    3. If a key is in 'nominal_vars' but missing from 'limits', 
       randomization range defaults to [0, 0] (no noise).
    4. Keys in 'limits' that are not in 'nominal_vars' are ignored.
    """
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1")

    # 1. Flatten nominal_vars to define the authoritative structure
    leaves_val, treedef = jax.tree_util.tree_flatten_with_path(nominal_vars)
    
    # 2. Prepare random keys matching the structure
    keys = jax.random.split(key, len(leaves_val))
    
    new_leaves = []
    
    # 3. Iterate over the defined nominal variables
    for (path, val), subkey in zip(leaves_val, keys):
        
        # --- Lookup Limit logic ---
        # Traverse 'limits' using the exact path found in 'nominal_vars'
        current_node = limits
        try:
            for node in path:
                # Handle JAX DictKey/SequenceKey vs standard keys
                # .key attribute exists for DictKey, .idx for SequenceKey in newer JAX
                # Fallback to the node itself if attributes are missing
                if hasattr(node, 'key'):
                    key_idx = node.key
                elif hasattr(node, 'idx'):
                    key_idx = node.idx
                else:
                    key_idx = node
                
                current_node = current_node[key_idx]
            
            # Found matching limit
            low, high = current_node[0], current_node[1]
            
        except (KeyError, TypeError, IndexError, AttributeError):
            # Fallback: Variable exists in nominal, but no limit defined.
            # Use 0.0 to prevent NaNs and keep value nominal.
            low, high = 0.0, 0.0

        # --- Batch Generation ---
        # 1. Nominal (1, ...)
        val_nominal_batched = val[None, ...] 

        # 2. Random (BATCH-1, ...)
        if batch_size > 1:
            k_sub = jax.random.split(subkey, batch_size - 1)
            
            def sample_leaf(k):
                # If low=0, high=0, this adds 0.0 noise (returns val)
                return jax.random.uniform(k, shape=val.shape, minval=low, maxval=high)
                
            val_random_batched = jax.vmap(sample_leaf)(k_sub)
            batch_leaf = jnp.concatenate([val_nominal_batched, val_random_batched], axis=0)
        else:
            batch_leaf = val_nominal_batched
            
        new_leaves.append(batch_leaf)
        
    # Reconstruct using nominal_vars structure
    return jax.tree_util.tree_unflatten(treedef, new_leaves)

batch_size = 6
random_seed = 15
key = jax.random.PRNGKey(random_seed)

print(f"Generating batch of {batch_size} (1 Nominal + {batch_size - 1} Random)...")

init_box_maps = threshold_box_maps.copy()
init_box_maps["opt_ctr_params"] = {"Kp_": jnp.array([20.0, 60.0]), "Ki_": jnp.array([10.0, 30.0]), "Kd_": jnp.array([2.0, 3.0])}

opt_vars_physical_batch = generate_mixed_batch(key, batch_size, opt_vars_physical, init_box_maps)
# Normalize opt_vars to [0, 1]
opt_vars = scale_tree(opt_vars_physical_batch, threshold_box_maps)

# Mask gradients of variables that are initialized to 0, so that they stay zero
init_mask = jax.tree_util.tree_map(lambda x: (x != 0).astype(x.dtype), opt_vars_physical)

# Optimizer
optimizer = optax.chain(
    optax.zero_nans(),
    optax.clip_by_global_norm(1.0),
    optax.yogi(learning_rate=4e-3)
)

opt_state = vmap(optimizer.init)(opt_vars)

# Gradient function
def evaluate_closed_loop_wrapper(opt_vars_norm, robot):
    opt_vars_phys = unscale_tree(opt_vars_norm, threshold_box_maps)
    return evaluate_closed_loop_system(opt_vars_phys, robot)

gradient_fn = value_and_grad(evaluate_closed_loop_wrapper, 0, has_aux=True)

# Update function of the optimization variables
def update_single(opt_vars, opt_state):
    # Compute gradient
    (loss, aux), grad = gradient_fn(opt_vars, robot)

    # Apply mask
    grad = jax.tree_util.tree_map(lambda g, m: g * m, grad, init_mask)

    # Update
    updates, opt_state = optimizer.update(grad, opt_state, params=opt_vars)
    opt_vars = optax.apply_updates(opt_vars, updates)

    # Apply constraints
    opt_vars = jax.tree_util.tree_map(lambda x: jnp.clip(x, 0.0, 1.0), opt_vars)

    # Store data
    aux['grad'] = grad
    aux['update'] = updates

    return opt_vars, opt_state, loss, aux

# Batched Update function
# in_axes: (batch, batch) -> (batch, batch, batch)
batch_update = jit(vmap(update_single, in_axes=(0, 0)))

# Optimization-wise variables
time_iter = []
loss_tot = []
loss_components_tot = []
opt_vars_tot = []
grad_tot = []
update_tot = []

# Simulation-wise variables
init_sim_data = None
best_sim_data = None


# =====================================================
# Optimization loop
# =====================================================
# Number of iterations
num_iters = 100

# Initial log
print('\nOptimization variables = ', list(opt_vars_physical.keys()))
print('mask                   = ', init_mask)
print(f"\nStarting optimization...")

# Track time
compile_start = time.time()

# First optimization step
# opt_vars, updated_opt_state, loss, aux = update(opt_vars, opt_state)
opt_vars, opt_state, loss, aux = batch_update(opt_vars, opt_state)
jax.block_until_ready(aux)

compile_end = time.time()
print(f"[INFO] Compilation + first execution time = {compile_end - compile_start:.2f} s")

# Store data
global_best_loss = jnp.min(loss)
init_sim_data = {
    't_ts': aux['t_ts'],
    'q_ts': aux['q_ts'],
    'qd_ts': aux['qd_ts'],
    'u_ts': aux['u_ts'],
    'k_ts': aux['k_ts'],
    'P_t_ts': aux['P_t_ts'],
}
time_iter.append(compile_end - compile_start)
loss_tot.append(loss)
loss_components_tot.append(aux['loss_components'])
opt_vars_tot.append(unscale_tree(opt_vars, threshold_box_maps))
grad_tot.append(aux['grad'])
update_tot.append(aux['update'])

i = 1
while i < num_iters:
    # Start of iteration
    iter_start = time.time()

    # Step
    opt_vars, updated_opt_state, loss, aux = batch_update(opt_vars, opt_state)
    jax.block_until_ready(aux)

    # End of iteration
    iter_end = time.time()
    iter_duration = iter_end - iter_start

    # Check validity
    if jnp.isnan(aux['q_ts']).any():
        print(f"\n[WARNING] NaN detected at iteration {i}. Stopping.")
        break

    # Store history
    time_iter.append(iter_duration)
    loss_tot.append(loss)
    loss_components_tot.append(aux['loss_components'])
    opt_vars_tot.append(unscale_tree(opt_vars, threshold_box_maps))
    grad_tot.append(aux['grad'])
    update_tot.append(aux['update'])

    # Update best data
    if jnp.min(loss) < global_best_loss:
        global_best_loss = jnp.min(loss)
        best_sim_data = {
            't_ts': aux['t_ts'],
            'q_ts': aux['q_ts'],
            'qd_ts': aux['qd_ts'],
            'u_ts': aux['u_ts'],
            'k_ts': aux['k_ts'],
            'P_t_ts': aux['P_t_ts'],
        }

    # Log
    t_left = (num_iters - i - 1) * sum(time_iter[1:]) / len(time_iter[1:]) # [s]
    eta = time.localtime(time.time() + t_left)
    print(
        f"Completion: {100 * (i + 1) / num_iters:3.1f} %  |  iteration {i + 1:>4d} of "
        + f"{num_iters:<4d}  |  iter time = {iter_duration:>.2f} s  |  ETA = {eta.tm_mday:02d}"
        + f"/{eta.tm_mon:02d}/{eta.tm_year} {eta.tm_hour:02d}:{eta.tm_min:02d}  ",
        end="\r",
    )
    
    # Update counter
    i += 1

# Log
print(
    f"Completion: {100 * i / num_iters:3.1f} %  |  iteration {i:>4d} of {num_iters:<4d}"
    + f"  |  iter time = {iter_duration:>.2f} s"
)
print("\nOptimization Finished.")


# =====================================================
# Extract data
# =====================================================
best_idx_opt, best_idx_batch = jnp.unravel_index(jnp.argmin(jnp.stack(loss_tot)), (jnp.stack(loss_tot)).shape)

t_ts = init_sim_data['t_ts']            # (b, t,)

q_ts_init = init_sim_data['q_ts']       # (b, t, n)
qd_ts_init = init_sim_data['qd_ts']     # (b, t, n)
u_ts_init = init_sim_data['u_ts']       # (b, t, m)
k_ts_init = init_sim_data['k_ts']       # (b, t,)
P_t_ts_init = init_sim_data['P_t_ts']   # (b, t,)

q_ts_best = best_sim_data['q_ts']       # (b, t, n)
qd_ts_best = best_sim_data['qd_ts']     # (b, t, n)
u_ts_best = best_sim_data['u_ts']       # (b, t, m)
k_ts_best = best_sim_data['k_ts']       # (b, t,)
P_t_ts_best = best_sim_data['P_t_ts']   # (b, t,)

x_ts_init = vmap(vmap(robot.forward_kinematics, in_axes=(0, None)), in_axes=(0, None))(q_ts_init, L_cum)[:, :, :3, 3]  # (b, t, o)
x_ts_best = vmap(vmap(robot.forward_kinematics, in_axes=(0, None)), in_axes=(0, None))(q_ts_best, L_cum)[:, :, :3, 3]  # (b, t, o)


# =====================================================
# Log
# =====================================================
print("\n----------\nLOG")
print("num_iters            =", num_iters)
print("batch_size           =", batch_size)
print("\ntime_iter            =", jnp.stack(time_iter).shape)
print("loss_tot             =", jnp.stack(loss_tot).shape)
print("loss_components_tot  =", jnp.stack(loss_components_tot).shape)
print("opt_vars_tot         =", len(opt_vars_tot))
print("\nt_ts                 =", t_ts.shape)
print("q_ts_best            =", q_ts_best.shape)
print("qd_ts_best           =", qd_ts_best.shape)
print("u_ts_best            =", u_ts_best.shape)
print("k_ts_best            =", k_ts_best.shape)
print("P_t_ts_best          =", P_t_ts_best.shape)
print("x_ts_best            =", x_ts_best.shape)


# =====================================================
# Save
# =====================================================
def to_np(x): return onp.array(x)

def stack_history(history_list):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *history_list)

def flatten_params_to_dict(params_tree, prefix=""):
    flat_dict = {}
    leaves, _ = jax.tree_util.tree_flatten_with_path(params_tree)
    for path, val in leaves:
        # Construct name from path, e.g., "opt_atr_params_ry0"
        key_str = "_".join([str(p.key) if hasattr(p, 'key') else str(p) for p in path])
        full_key = f"{prefix}{key_str}"
        flat_dict[full_key] = to_np(val)
    return flat_dict

# Path
cwd = os.getcwd()
save_dir = os.path.join(cwd, "data", os.path.basename(__file__).split('.')[0].lower())
os.makedirs(save_dir, exist_ok=True)
save_path = str(save_dir)

mat_data = {}

# Optimization history
mat_data['history_loss'] = to_np(jnp.stack(loss_tot))
mat_data['history_loss_components'] = to_np(jnp.stack(loss_components_tot))
mat_data['history_time'] = to_np(jnp.array(time_iter))

# Initial simulation
if init_sim_data:
    mat_data['t_ts'] = to_np(t_ts)
    mat_data['q_ts_init'] = to_np(q_ts_init)
    mat_data['qd_ts_init'] = to_np(qd_ts_init)
    mat_data['u_ts_init'] = to_np(u_ts_init)
    mat_data['k_ts_init'] = to_np(k_ts_init)
    mat_data['P_t_ts_init'] = to_np(P_t_ts_init)
    mat_data['x_ts_init'] = to_np(x_ts_init)

# Optimized simulation
if best_sim_data:
    mat_data['q_ts_best'] = to_np(q_ts_best)
    mat_data['qd_ts_best'] = to_np(qd_ts_best)
    mat_data['u_ts_best'] = to_np(u_ts_best)
    mat_data['k_ts_best'] = to_np(k_ts_best)
    mat_data['P_t_ts_best'] = to_np(P_t_ts_best)
    mat_data['x_ts_best'] = to_np(x_ts_best)

# Optimization variables
stacked_opt_vars = stack_history(opt_vars_tot)
mat_data.update(flatten_params_to_dict(stacked_opt_vars, prefix="hist_"))

# Gradients
stacked_grads = stack_history(grad_tot)
mat_data.update(flatten_params_to_dict(stacked_grads, prefix="hist_grad_"))

# Updates
stacked_updates = stack_history(update_tot)
mat_data.update(flatten_params_to_dict(stacked_updates, prefix="hist_upd_"))

# Metadata
mat_data['num_iters'] = to_np(num_iters)
mat_data['batch_size'] = to_np(batch_size)
mat_data['opt_vars_idx'] = onp.arange(i)
mat_data['x_des_ts'] = to_np(x_des_ts_full[jnp.newaxis, :])
mat_data["best_idx_opt"] = best_idx_opt

# Save main summary .mat file
mat_filename = os.path.join(save_dir, "optimization_results.mat")
print(f"Saving data to {mat_filename}")
sio.savemat(mat_filename, mat_data, do_compression=True)

animation_dict = {
    # Robot parameters
    "params": params,
    "active_tendon_routing_params": active_tendon_routing_params,
    "passive_tendon_routing_params": {},
    "passive_tendon_params": {},
    "num_segments": num_segments,
    
    # Optimized parameters
    "best_opt_vars": opt_vars_tot[best_idx_opt],

    # Trajectory
    "t_ts": best_sim_data['t_ts'],
    "q_ts": best_sim_data['q_ts'],
    
    # Target
    "x_des_ts": x_des_ts_full
}

anim_filename = os.path.join(save_dir, "animation_data.pkl")
with open(anim_filename, "wb") as f:
    pickle.dump(animation_dict, f)

print("Save complete.") # '''