import jax
import pandas as pd
import jax.numpy as jnp
from soromox.systems.gvs.attributes import LinkAttributes, JointAttributes, BasisAttributes
#from soromox.systems.gvs.core import GVS
from soromox.systems.gvs.tendon_actuated_gvs import TendonActuatedGVS
from functools import partial
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import optimistix as optx
import equinox as eqx
import optax
from jax import Array, lax
import numpy as onp
from dataclasses import replace
from jax.tree_util import tree_leaves

jax.config.update("jax_enable_x64", True)

# jax.config.update("jax_platform_name", "gpu")  # or "cpu"

# print("JAX default backend:", jax.default_backend())
# print("JAX devices:", jax.devices())

### MOCAP DATA TRANSFORMATION FUNCTIONS ###

def _transform_points(pb: jnp.ndarray, p1: jnp.ndarray, p2: jnp.ndarray, p3: jnp.ndarray, p4: jnp.ndarray,
                      Rot: jnp.ndarray, R_alg1: jnp.ndarray, R_alg2: jnp.ndarray, offset: jnp.ndarray):
    offset_col = offset.reshape(3, 1)
    offset_col = jnp.broadcast_to(offset_col, pb.shape)
    pb_gl = Rot @ pb
    p1_gl = Rot @ p1
    p2_gl = Rot @ p2
    p3_gl = Rot @ p3
    p4_gl = Rot @ p4

    pb_lcl = -offset_col
    p1_lcl = p1_gl - (pb_gl + offset_col)
    p2_lcl = p2_gl - (pb_gl + offset_col)
    p3_lcl = p3_gl - (pb_gl + offset_col)
    p4_lcl = p4_gl - (pb_gl + offset_col)

    R_total = R_alg1 @ R_alg2
    pb_tr = R_total @ pb_lcl
    p1_tr = R_total @ p1_lcl
    p2_tr = R_total @ p2_lcl
    p3_tr = R_total @ p3_lcl
    p4_tr = R_total @ p4_lcl

    return {"pb_tr": pb_tr, "p1_tr": p1_tr, "p2_tr": p2_tr, "p3_tr": p3_tr, "p4_tr": p4_tr}

### EXCEL DATA READING ###
# Rot = jnp.array([[1.0, 0.0, 0.0],
#                  [0.0, 0.0, -1.0],
#                  [0.0, 1.0, 0.0]], dtype=jnp.float64)
# R_alg1 = jnp.array([[-1.0, 0.0, 0.0],
                    # [0.0, -1.0, 0.0],
                    # [0.0, 0.0, 1.0]], dtype=jnp.float64)
# R_alg2 = jnp.array([[1.0, 0.0, 0.0],
                    # [0.0, -1.0, 0.0],
                    # [0.0, 0.0, -1.0]], dtype=jnp.float64)
Rot = jnp.array([[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]], dtype=jnp.float64)
R_alg1 = jnp.array([[1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0]], dtype=jnp.float64)
R_alg2 = jnp.array([[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]], dtype=jnp.float64)
offset = jnp.array([-0.03, -0.013, 0.0], dtype=jnp.float64)



import pandas as pd
import jax.numpy as jnp
import os

def load_dynamic_dataset(file1, file2, concat_rules):
    """Legge due file CSV mocap e concatena i dati dinamici, adattandosi ai tre casi (u1, u4, u14)."""
    base_name = os.path.basename(file1)

    # --- SCELTA CASO IN BASE AL NOME FILE ---
    if "_u1_" in base_name:
        # Corrisponde a m1234_u1_0001
        idx_main = dict(pb=(9,12), p1=(6,9), p2=(12,15), p3=(3,6), p4=(0,3))
        idx_cont = dict(pbc=(9,12), p1c=(6,9), p2c=(12,15), p3c=(3,6), p4c=(0,3))
        offset_main, offset_cont = 2, 17

    elif "_u4_" in base_name:
        # Corrisponde a m1234_u4_0001
        idx_main = dict(pb=(9,12), p1=(3,6), p2=(12,15), p3=(0,3), p4=(6,9))
        idx_cont = dict(pbc=(9,12), p1c=(6,9), p2c=(12,15), p3c=(3,6), p4c=(0,3))
        offset_main, offset_cont = 2, 17

    elif "_u14_" in base_name:
        # Corrisponde a m1234_u14_00020001
        idx_main = dict(pb=(9,12), p1=(3,6), p2=(12,15), p3=(0,3), p4=(6,9))
        idx_cont = dict(pbc=(3,6), p1c=(0,3), p2c=(12,15), p3c=(6,9), p4c=(9,12))
        offset_main, offset_cont = 2, 17

    else:
        raise ValueError(f"Tipo di file non riconosciuto per: {base_name}")

    # --- LETTURA PRIMO FILE ---
    data = pd.read_csv(file1, header=None).iloc[:, offset_main:].to_numpy(dtype=float)
    pb  = data[:, idx_main['pb'][0]:idx_main['pb'][1]].T
    p1  = data[:, idx_main['p1'][0]:idx_main['p1'][1]].T
    p2  = data[:, idx_main['p2'][0]:idx_main['p2'][1]].T
    p3  = data[:, idx_main['p3'][0]:idx_main['p3'][1]].T
    p4  = data[:, idx_main['p4'][0]:idx_main['p4'][1]].T

    # --- LETTURA FILE CONTINUE ---
    data2 = pd.read_csv(file2, header=None).iloc[:, offset_cont:].to_numpy(dtype=float)
    pbc = data2[:, idx_cont['pbc'][0]:idx_cont['pbc'][1]].T
    p1c = data2[:, idx_cont['p1c'][0]:idx_cont['p1c'][1]].T
    p2c = data2[:, idx_cont['p2c'][0]:idx_cont['p2c'][1]].T
    p3c = data2[:, idx_cont['p3c'][0]:idx_cont['p3c'][1]].T
    p4c = data2[:, idx_cont['p4c'][0]:idx_cont['p4c'][1]].T

    # --- CONCATENAZIONE TEMPORALE (replica MATLAB) ---
    def cat_two(a, b, A, Bc):
        # a: primo limite matlab (es. 606) -> Python slice :a  OK
        # b: inizio nel continue matlab (es. 607) -> inizio python = b-1
        start_bc = max(0, b - 1)
        return jnp.concatenate([A[:, :a], Bc[:, start_bc:]], axis=1)

    if "_u4_" in base_name:
        # caso speciale: p4 è intercalato in più pezzi secondo il tuo matlab originale:
        # p4 = [p4(:,1:561), p4c(:,562:575), p4(:,576:577), p4c(:,578:end)];
        # traduciamo esattamente le slice (MATLAB indices)
        a1, b1 = 561, 562
        a2_start, a2_end = 576, 577
        b2 = 578
        part1 = p4[:, :a1]
        part2 = p4c[:, (b1-1):(575)]     # b1-1 .. 574 (python end exclusive -> 575)
        part3 = p4[:, (a2_start-1):a2_end]   # 575:577 in python -> a2_end is inclusive in matlab
        part4 = p4c[:, (b2-1):]
        p4 = jnp.concatenate([part1, part2, part3, part4], axis=1)

        # per le altre variabili usiamo cat_two standard con i concat_rules forniti
        p3 = cat_two(concat_rules['p3'][0], concat_rules['p3'][1], p3, p3c)
        p2 = cat_two(concat_rules['p2'][0], concat_rules['p2'][1], p2, p2c)
        p1 = cat_two(concat_rules['p1'][0], concat_rules['p1'][1], p1, p1c)
        pb = cat_two(concat_rules['pb'][0], concat_rules['pb'][1], pb, pbc)

    else:
        # casi u1, u14: concatenazione semplice
        p4 = cat_two(concat_rules['p4'][0], concat_rules['p4'][1], p4, p4c)
        p3 = cat_two(concat_rules['p3'][0], concat_rules['p3'][1], p3, p3c)
        p2 = cat_two(concat_rules['p2'][0], concat_rules['p2'][1], p2, p2c)
        p1 = cat_two(concat_rules['p1'][0], concat_rules['p1'][1], p1, p1c)
        pb = cat_two(concat_rules['pb'][0], concat_rules['pb'][1], pb, pbc)
    return pb, p1, p2, p3, p4

concat_rules_u1 = dict(p4=(606, 607), p3=(607, 608), p2=(817, 818), p1=(817, 818), pb=(817, 818))
pb, p1, p2, p3, p4 = load_dynamic_dataset(
    "SoftRob_tests/m1234_u1_0001.csv",
    "SoftRob_tests/m1234_u1_0001_continue.csv",
    concat_rules_u1
)
pb = pb[:, :3600]
p1 = p1[:, :3600]
p2 = p2[:, :3600]
p3 = p3[:, :3600]
p4 = p4[:, :3600]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_u1 = res["pb_tr"]
p1_u1 = res["p1_tr"]
p2_u1 = res["p2_tr"]
p3_u1 = res["p3_tr"]
p4_u1 = res["p4_tr"]

concat_rules_u4 = dict(p4=(561, 562), p3=(649, 650), p2=(782, 783), p1=(782, 783), pb=(782, 783))
pb, p1, p2, p3, p4 = load_dynamic_dataset(
    "SoftRob_tests/m1234_u4_0001.csv",
    "SoftRob_tests/m1234_u4_0001_continue.csv",
    concat_rules_u4
)
pb = pb[:, :3600]
p1 = p1[:, :3600]
p2 = p2[:, :3600]
p3 = p3[:, :3600]
p4 = p4[:, :3600]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_u2 = res["pb_tr"]
p1_u2 = res["p1_tr"]
p2_u2 = res["p2_tr"]
p3_u2 = res["p3_tr"]
p4_u2 = res["p4_tr"]
concat_rules_u14 = dict(p4=(594, 595), p3=(594, 595), p2=(594, 595), p1=(342, 343), pb=(342, 343))
pb, p1, p2, p3, p4 = load_dynamic_dataset(
    "SoftRob_tests/m1234_u14_00020001.csv",
    "SoftRob_tests/m1234_u14_00020001_continue.csv",
    concat_rules_u14
)
pb = pb[:, :3600]
p1 = p1[:, :3600]
p2 = p2[:, :3600]
p3 = p3[:, :3600]
p4 = p4[:, :3600]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_u12 = res["pb_tr"]
p1_u12 = res["p1_tr"]
p2_u12 = res["p2_tr"]
p3_u12 = res["p3_tr"]
p4_u12 = res["p4_tr"]


print("MOCAP data loaded.")

print("pb_u1 shape:", pb_u1.shape)
print("pb_u1:", pb_u1)
print("p4_u1 shape:", p4_u1.shape)
print("p4_u1:", p4_u1)

print("pb_u2 shape:", pb_u2.shape)
print("pb_u2:", pb_u2)
print("p4_u2 shape:", p4_u2.shape)
print("p4_u2:", p4_u2)

print("pb_u12 shape:", pb_u12.shape)
print("pb_u12:", pb_u12)
print("p4_u12 shape:", p4_u12.shape)
print("p4_u12:", p4_u12)




# ### SOFT ROBOT UTILITIES FUNCTIONS ###

# # ---- Marker prediction for a fixed q  ----
def markers_from_q(robot: TendonActuatedGVS, q: jnp.ndarray) -> jnp.ndarray:
    """
    Output [p1; p2; p3; p4] (12,), where:
      p1 @ s=0.1291 with offset [0,0, 0.025]
      p2 @ s=0.2195 with offset [0,0,-0.021]
      p3 @ s=0.2800 with offset [0,0, 0.020]
      p4 @ s=sum(L) with offset [0.008,0,0]
    """
    def tip_at_s(s_val, offset):
        G = robot.forward_kinematics(q, s_val)     # 4x4
        p = G[:3, 3]; R = G[:3, :3]
        return p + R @ offset

    p1 = tip_at_s(0.1291, jnp.array([0.0, 0.0,  0.025]))
    p2 = tip_at_s(0.2195, jnp.array([0.0, 0.0, -0.021]))
    p3 = tip_at_s(0.2800, jnp.array([0.0, 0.0,  0.020]))
    p4 = tip_at_s(jnp.sum(robot.V_L), jnp.array([0.008, 0.0, 0.0]))
    return jnp.concatenate([p1, p2, p3, p4], axis=0)  # (12,)

def integrate_markers(robot, q0, qd0, u_traj, tau, t_grid):
    mk_hist = []
    q = q0; qd = qd0; t_prev = float(t_grid[0])
    mk_hist.append(markers_from_q(robot, q))
    for k in range(len(t_grid) - 1):
        t1 = float(t_grid[k + 1])
        ts, q_ts, qd_ts = robot.resolve_upon_time(
            q0=q, qd0=qd,
            u=u_traj[k],
            t0=t_prev, t1=t1,
            dt=1e-4,
            tau_ext=tau,
            save_dt=t1 - t_prev,
        )
        q = q_ts[-1]; qd = qd_ts[-1]
        mk_hist.append(markers_from_q(robot, q))
        t_prev = t1
    return jnp.stack(mk_hist)

def dyn_loss(robot, q0, qd0, u_traj, t_grid, markers_traj, lambda_reg):
    def loss_tau(tau):
        mk_pred = integrate_markers(robot, q0, qd0, u_traj, tau, t_grid)   # (T, 12)
        err = mk_pred - markers_traj                                       # (T, 12)
        return 0.5 * jnp.mean(jnp.sum(err**2, axis=1)) + 0.5 * lambda_reg * jnp.sum(tau**2)
    return loss_tau

# ### BODY DEFINITION OF THE SOFT ROBOT ###

# #2 link version 
# # Link 1
link1 = LinkAttributes(
    section="Circular",
    E=3.04e5,
    nu=0.45,
    rho=1310.0,
    eta=1e4,
    L=0.0250+0.2550+0.0250,
    r_i=0.01541,
    r_f=0.00642,
)

# # Link 2
link2 = LinkAttributes(
    section="Circular",
    E=3.04e5,
    nu=0.45,
    rho=1310.0,
    eta=1e4,
    L=0.0550,
    r_i=0.00642,
    r_f=0.00480,
)
joint1 = JointAttributes(jointtype="Fixed")
joint2 = JointAttributes(jointtype="Fixed")
basis1 = BasisAttributes(basistype="Monomial", Bdof=[1, 1, 1, 1, 0, 0], Bodr=[0, 1, 1, 1, 0, 0])
basis2 = BasisAttributes(basistype="Monomial", Bdof=[0, 1, 1, 0, 0, 0], Bodr=[0, 0, 0, 0, 0, 0])

n_gauss_list = [8, 8]
gravity_vector = [0.0, 0.0, -9.81]

tendon_routing_params = {
     "ry": jnp.array([0.0114*jnp.cos(jnp.pi/180*30), 0.0114*jnp.cos(jnp.pi/180*150)]),
     "my": jnp.array([-0.0295*jnp.cos(jnp.pi/180*30),-0.0295*jnp.cos(jnp.pi/180*150)]),
     "rz": jnp.array([0.0114*jnp.sin(jnp.pi/180*30), 0.0114*jnp.sin(jnp.pi/180*150)]),
     "mz": jnp.array([-0.0295*jnp.sin(jnp.pi/180*30), -0.0295*jnp.sin(jnp.pi/180*150)]),
     "idx_seg_att": jnp.array([0, 0]),
}

robot = TendonActuatedGVS(
    links_list=[link1, link2],
    joints_list=[joint1, joint2],
    basis_list=[basis1, basis2],
    n_gauss_list=n_gauss_list,
    gravity_vector=gravity_vector,
    tendon_routing_params=tendon_routing_params,
)

# Initialize parameters
n_links = int(robot.num_segments)
radius = 0.0325
dof = int(sum(robot.V_dof.reshape(-1)))

markers_dyn_u1_batch = jnp.concatenate([p1_u1.T, p2_u1.T, p3_u1.T, p4_u1.T], axis=1)  # (T,12)
markers_dyn_u2_batch = jnp.concatenate([p1_u2.T, p2_u2.T, p3_u2.T, p4_u2.T], axis=1)
markers_dyn_u12_batch = jnp.concatenate([p1_u12.T, p2_u12.T, p3_u12.T, p4_u12.T], axis=1)





# === Snapshot-based tau* over full 15s (tau=0 elsewhere) ===

fs = 240.0
dt_frame = 1.0 / fs
T_full = int(markers_dyn_u12_batch.shape[0])  # 3600
num_samples = 10                              # 10 punti (ogni 1.5s)
frames_step = T_full // num_samples           # 360
assert frames_step * num_samples == T_full, "T_full deve essere multiplo di num_samples"

# istanti di valutazione: 360, 720, ..., 3600 (indici 0-based)
idx_eval = jnp.arange(frames_step, T_full + 1, frames_step)  # shape (10,)
# user intention: arrivare a t_eval-dt, poi 1 step fino t_eval
# quindi la prima integrazione 'libera' è di (frames_step-1) frame


# #U_BATCH NEEDS TO BE CORRECTED ACCORDING TO DYNAMIC DATASET!!!!
# u_const = jnp.array([-0.02 / radius, -0.01 / radius])      # caso u12 (adatta se cambi dataset)
# Build u(t) like the MATLAB snippet (T_full=3600, fs=240 Hz)
u_traj_full_np = onp.zeros((T_full, 2), dtype=onp.float64)  # u(1,:) = [0,0]
for k in range(1, T_full):                                  # k = 1..3599
    if (k * dt_frame) <= 2.0:                                # first 2 seconds
        u_traj_full_np[k] = u_traj_full_np[k-1] - onp.array([0.1*dt_frame, 0.05*dt_frame])
    else:
        u_traj_full_np[k] = u_traj_full_np[k-1]
u_traj_full = jnp.asarray(u_traj_full_np)                   # shape (T_full, 2)
u_traj_full = u_traj_full / radius

q_cur = jnp.zeros(dof)
qd_cur = jnp.zeros(dof)
t_cur = 0.0

tau_star_list = []
q_snap_list = []
mk_pred_list = []
mk_true_list = []

# def integrate_chunk_free(q0, qd0, dt_chunk):
#     """Integra per dt_chunk con tau_ext=0, u=u_const, ritorna (qT, qdT)."""
#     if dt_chunk <= 0:
#         print("Warning: dt_chunk <= 0 in integrate_chunk_free.")
#         return q0, qd0
        
#     ts, q_ts, qd_ts = robot.resolve_upon_time(
#         q0=q0, qd0=qd0,
#         u=u_const,
#         t0=0.0, t1=float(dt_chunk),
#         dt=1e-4,
#         tau_ext=jnp.zeros(dof),
#         save_dt=float(dt_chunk),  # salva solo l’ultimo stato
#     )
#     return q_ts[-1], qd_ts[-1]
def integrate_chunk_free(q0, qd0, start_idx: int, n_frames: int):
    """Integra n_frames passi con tau_ext=0 e u(t) variabile."""
    q, qd = q0, qd0
    for kk in range(n_frames):
        u_vec = u_traj_full[start_idx + kk]
        ts, q_ts, qd_ts = robot.resolve_upon_time(
            q0=q, qd0=qd,
            u=u_vec,
            t0=0.0, t1=float(dt_frame),
            dt=1e-4,
            tau_ext=jnp.zeros(dof),
            save_dt=float(dt_frame),
        )
        q, qd = q_ts[-1], qd_ts[-1]
    return q, qd

# def one_step_with_tau(q0, qd0, tau, dt_frame):
#     """Un singolo step dinamico di durata dt con tau esterno costante."""
#     ts, q_ts, qd_ts = robot.resolve_upon_time(
#         q0=q0, qd0=qd0,
#         u=u_const,
#         t0=0.0, t1=float(dt_frame),
#         dt=1e-4,
#         tau_ext=tau,
#         save_dt=float(dt_frame),
#     )
#     q1 = q_ts[-1]; qd1 = qd_ts[-1]
#     mk1 = markers_from_q(robot, q1)  # (12,)
#     return q1, qd1, mk1
def one_step_with_tau(q0, qd0, tau, u_vec, dt_frame):
     """Un singolo step dinamico di durata dt con tau esterno costante."""
     ts, q_ts, qd_ts = robot.resolve_upon_time(
         q0=q0, qd0=qd0,
         u=u_vec,
         t0=0.0, t1=float(dt_frame),
         dt=1e-4,
         tau_ext=tau,
         save_dt=float(dt_frame),
     )
     q1 = q_ts[-1]; qd1 = qd_ts[-1]
     mk1 = markers_from_q(robot, q1)  # (12,)
     return q1, qd1, mk1
# def one_step_free(q0, qd0, dt_frame):
#     return one_step_with_tau(q0, qd0, jnp.zeros(dof), dt_frame)
def one_step_free(q0, qd0, u_vec, dt_frame):
    return one_step_with_tau(q0, qd0, jnp.zeros(dof), u_vec, dt_frame)

def optimize_tau_snapshot(q_in, qd_in, mk_target, tau_init=None,
                          steps=120, lr=5e-3, lambda_reg=1e-6):
    """Ottimizza tau su un solo frame per matchare mk_target."""
    tau = jnp.zeros(dof) if tau_init is None else tau_init

    def loss_fn(tau_vec):
        # q1, qd1, mk1 = one_step_with_tau(q_in, qd_in, tau_vec, dt_frame)
        q1, qd1, mk1 = one_step_with_tau(q_in, qd_in, tau_vec, u_eval, dt_frame)
        err = mk1 - mk_target
        return 0.5 * jnp.sum(err * err) + 0.5 * lambda_reg * jnp.sum(tau_vec * tau_vec)

    val_grad = jax.value_and_grad(loss_fn)
    opt = optax.adam(lr)
    opt_state = opt.init(tau)

    for _ in range(steps):
        val, g = val_grad(tau)
        updates, opt_state = opt.update(g, opt_state, tau)
        tau = optax.apply_updates(tau, updates)

    # q_star, qd_star, mk_star = one_step_with_tau(q_in, qd_in, tau, dt_frame)
   
    q_star, qd_star, mk_star = one_step_with_tau(q_in, qd_in, tau, u_eval, dt_frame)
    return tau, q_star, qd_star, mk_star

prev_eval_idx = 0
tau_prev = None

for k, i_eval in enumerate(onp.asarray(idx_eval)):
    # 1) integra liberamente fino a t_eval - dt_frame (tau=0)
    frames_free = int(i_eval - prev_eval_idx) - 1  # es. 359
    # dt_free = frames_free * dt_frame
    # q_cur, qd_cur = integrate_chunk_free(q_cur, qd_cur, dt_free)
    # t_cur += dt_free
    q_cur, qd_cur = integrate_chunk_free(q_cur, qd_cur, start_idx=prev_eval_idx, n_frames=frames_free)
    t_cur += frames_free * dt_frame


    # 2) ottimizza tau solo sul micro-step per arrivare a t_eval
    mk_target = markers_dyn_u12_batch[i_eval-1]  # target al tempo t_cur + dt_frame (= t_eval)
    # tau_star_i, q_star_i, qd_star_i, mk_pred_star = optimize_tau_snapshot(q_cur, qd_cur, mk_target, tau_init=tau_prev,
    #                                    steps=150, lr=5e-3, lambda_reg=1e-6)
    u_eval = u_traj_full[i_eval-1]               # u al frame di arrivo
    tau_star_i, q_star_i, qd_star_i, mk_pred_star = optimize_tau_snapshot(
        q_cur, qd_cur, mk_target, tau_init=tau_prev,
        steps=150, lr=5e-3, lambda_reg=1e-6)
   
   
   
    # 3) applica lo step con tau*, salva risultati
    # q_next, qd_next, mk_pred_free = one_step_free(q_cur, qd_cur, dt_frame)
    # t_cur += dt_frame
    # t_cur += dt_frame
    q_next, qd_next, mk_pred_free = one_step_free(q_cur, qd_cur, u_eval, dt_frame)
    t_cur += dt_frame
    
    
    tau_star_list.append(tau_star_i)
    q_snap_list.append(q_star_i)
    mk_pred_list.append(mk_pred_star)
    mk_true_list.append(mk_target)

    # 4) prepara per il prossimo ciclo
    q_cur, qd_cur = q_next, qd_next
    prev_eval_idx = i_eval
    tau_prev = tau_star_i  # warm-start

tau_star_batch_10 = jnp.stack(tau_star_list, axis=1)  # (dof, 10)
q_snap_array = jnp.stack(q_snap_list, axis=0)         # (10, dof)
mk_pred_array = jnp.stack(mk_pred_list, axis=0)       # (10, 12)
mk_true_array = jnp.stack(mk_true_list, axis=0)       # (10, 12)

# Metriche
err_snap = mk_pred_array - mk_true_array              # (10,12)
rmse_markers = jnp.sqrt(jnp.mean(err_snap.reshape(-1,4,3)**2, axis=(0,2)))  # (4,)

print("tau* snapshots shape:", tau_star_batch_10.shape)
print("markers RMSE (per marker):", rmse_markers)

# Plot rapido RMSE per marker
plt.figure(figsize=(5,3))
plt.bar(onp.arange(4), onp.asarray(rmse_markers), alpha=0.85)
plt.xticks(onp.arange(4), [f"m{i}" for i in range(1,5)])
plt.ylabel("RMSE [m]")
plt.title("RMSE markers (10 snapshot tau*)")
plt.tight_layout()
plt.savefig("dyn_snapshots_marker_rmse.svg", bbox_inches="tight")
plt.show()


# in this way I have the set of tau* snapshots to be used in the training
##PENSO CHE DA QUA IN POI TU POSSA COPIARE DAGLI ALTRI FILE UNA VOLTA CHE LI HAI CORRETTI COME SI DEVE

#training nn


# dynamic simulation with learned tau*= f(q,u)


