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

print("JAX default backend:", jax.default_backend())
print("JAX devices:", jax.devices())

### MOCAP DATA TRANSFORMATION FUNCTIONS ###

def _transform_points(pb: jnp.ndarray, p1: jnp.ndarray, p2: jnp.ndarray, p3: jnp.ndarray, p4: jnp.ndarray,
                      Rot: jnp.ndarray, R_alg1: jnp.ndarray, R_alg2: jnp.ndarray, offset: jnp.ndarray):
    # equivalente del blocco richiesto
    pb_gl = Rot @ pb
    p1_gl = Rot @ p1
    p2_gl = Rot @ p2
    p3_gl = Rot @ p3
    p4_gl = Rot @ p4

    pb_lcl = -offset
    p1_lcl = p1_gl - (pb_gl + offset)
    p2_lcl = p2_gl - (pb_gl + offset)
    p3_lcl = p3_gl - (pb_gl + offset)
    p4_lcl = p4_gl - (pb_gl + offset)

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

df = pd.read_csv("SoftRob_tests/ss_NL2.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)     # (N, M-2)
filtered = jnp.asarray(vals.mean(axis=0))       # (M-2,)
# mapping 1-based MATLAB -> 0-based Python (end esclusivo)
pb = filtered[0:3]
p1 = filtered[6:9]
p2 = filtered[3:6]
p3 = filtered[12:15]
p4 = filtered[9:12]

res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_NL = res["pb_tr"]
p1_NL = res["p1_tr"]
p2_NL = res["p2_tr"]
p3_NL = res["p3_tr"]
p4_NL = res["p4_tr"]

# print("pb_NL:", pb_tr)
# print("p1_NL:", p1_tr)
# print("p2_NL:", p2_tr)
# print("p3_NL:", p3_tr)
# print("p4_NL:", p4_tr)


df = pd.read_csv("SoftRob_tests/t2m1_u01.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)     
filtered = jnp.asarray(vals.mean(axis=0))    
pb = filtered[12:15]
p1 = filtered[0:3]
p2 = filtered[9:12]
p3 = filtered[3:6]
p4 = filtered[6:9]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m1u01 = res["pb_tr"]
p1_m1u01 = res["p1_tr"]
p2_m1u01 = res["p2_tr"]
p3_m1u01 = res["p3_tr"]
p4_m1u01 = res["p4_tr"]

df = pd.read_csv("SoftRob_tests/t2m1_u015.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)     
filtered = jnp.asarray(vals.mean(axis=0))    
pb = filtered[9:12]
p1 = filtered[6:9]
p2 = filtered[12:15]
p3 = filtered[3:6]
p4 = filtered[0:3]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m1u015 = res["pb_tr"]
p1_m1u015 = res["p1_tr"]
p2_m1u015 = res["p2_tr"]
p3_m1u015 = res["p3_tr"]
p4_m1u015 = res["p4_tr"]

df = pd.read_csv("SoftRob_tests/t2m1_u02.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)     
filtered = jnp.asarray(vals.mean(axis=0))    
pb = filtered[9:12]
p1 = filtered[6:9]
p2 = filtered[12:15]
p3 = filtered[3:6]
p4 = filtered[0:3]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m1u02 = res["pb_tr"]
p1_m1u02 = res["p1_tr"]
p2_m1u02 = res["p2_tr"]
p3_m1u02 = res["p3_tr"]
p4_m1u02 = res["p4_tr"]

df = pd.read_csv("SoftRob_tests/t2m1_u025.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)     
filtered = jnp.asarray(vals.mean(axis=0))    
pb = filtered[12:15]
p1 = filtered[9:12]
p2 = filtered[0:3]
p3 = filtered[6:9]
p4 = filtered[3:6]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m1u025 = res["pb_tr"]
p1_m1u025 = res["p1_tr"]
p2_m1u025 = res["p2_tr"]
p3_m1u025 = res["p3_tr"]
p4_m1u025 = res["p4_tr"]

df = pd.read_csv("SoftRob_tests/t2m4_u01.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)     
filtered = jnp.asarray(vals.mean(axis=0))    
pb = filtered[6:9]
p1 = filtered[0:3]
p2 = filtered[12:15]
p3 = filtered[9:12]
p4 = filtered[3:6]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m2u01 = res["pb_tr"]
p1_m2u01 = res["p1_tr"]
p2_m2u01 = res["p2_tr"]
p3_m2u01 = res["p3_tr"]
p4_m2u01 = res["p4_tr"]

df = pd.read_csv("SoftRob_tests/t2m4_u015.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)     
filtered = jnp.asarray(vals.mean(axis=0))    
pb = filtered[9:12]
p1 = filtered[6:9]
p2 = filtered[12:15]
p3 = filtered[3:6]
p4 = filtered[0:3]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m2u015 = res["pb_tr"]
p1_m2u015 = res["p1_tr"]
p2_m2u015 = res["p2_tr"]
p3_m2u015 = res["p3_tr"]
p4_m2u015 = res["p4_tr"]

df = pd.read_csv("SoftRob_tests/t2m4_u02.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)     
filtered = jnp.asarray(vals.mean(axis=0))    
pb = filtered[9:12]
p1 = filtered[6:9]
p2 = filtered[12:15]
p3 = filtered[3:6]
p4 = filtered[0:3]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m2u02 = res["pb_tr"]
p1_m2u02 = res["p1_tr"]
p2_m2u02 = res["p2_tr"]
p3_m2u02 = res["p3_tr"]
p4_m2u02 = res["p4_tr"]

df = pd.read_csv("SoftRob_tests/t2m4_u025.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)     
filtered = jnp.asarray(vals.mean(axis=0))    
pb = filtered[6:9]
p1 = filtered[3:6]
p2 = filtered[9:12]
p3 = filtered[0:3]
p4 = filtered[12:15]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m2u025 = res["pb_tr"]
p1_m2u025 = res["p1_tr"]
p2_m2u025 = res["p2_tr"]
p3_m2u025 = res["p3_tr"]
p4_m2u025 = res["p4_tr"]


def compute_marker_errors(robot, u_batch, q0, measured_markers_batch, tau_batch):
    """
    Restituisce gli errori (M, 12) = differenze per ogni marker e ogni esperimento
    """
    u_T = u_batch.T
    meas_T = measured_markers_batch.T
    M = u_T.shape[0]
    if tau_batch.ndim == 1:
        tau_mat = jnp.tile(tau_batch.reshape(dof, 1), (1, M))     # (dof, M)
    elif tau_batch.shape == (dof, M):
        tau_mat = tau_batch                                        # (dof, M)
    elif tau_batch.shape == (M, dof):
        tau_mat = tau_batch.T                                      # (dof, M)
    else:
        raise ValueError(f"Unexpected tau_batch shape {tau_batch.shape}; expected (dof,M), (M,dof) o (dof,) with dof={dof}, M={M}")

    diffs = []
    for m in range(M):
        res = solve_equilibrium_with_tau(robot, u_T[m], tau_mat[:, m], q0)
        q_star = res.value
        pred = markers_from_q(robot, q_star)
        diffs.append(pred - meas_T[m])
    diffs = jnp.stack(diffs)  # (M, 12)
    return diffs


### SOFT ROBOT UTILITIES FUNCTIONS ###

# ---- Marker prediction for a fixed q  ----
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

#alternatively, u can u see data from a continuous movement and use dynamics(more useful for the whole simulation I think)
def solve_equilibrium_with_tau(robot: TendonActuatedGVS,
                               u: jnp.ndarray,          # (na,)
                               tau_ext: jnp.ndarray,    # (dof,)
                               q0: jnp.ndarray):        # (dof,)
    """
    Solve static equilibrium with an additive generalized external force tau_ext:
      K(q) q + G(q) - B(q) u - tau_ext = 0
    Returns: res (optimistix result with .value = q*)
    """
    def statics_eq(q, args):
        u_m, tau_ext = args
        K = robot.stiffness_matrix()       # (dof,dof)
        B = robot.actuation_matrix(q)      # (dof, na)
        G = robot.gravitational_force(q)   # (dof,)
        return K @ q + G - B @ u_m - tau_ext

    solver = optx.Newton(rtol=1e-6, atol=1e-6)
    statics_eq_jit = jax.jit(statics_eq)   # robot in closure (statico)
    return optx.root_find(statics_eq_jit, solver, q0, (u, tau_ext), max_steps=200)


def make_tau_loss_for_sample(robot: TendonActuatedGVS,
                             u_i: jnp.ndarray,           # (na,)
                             p_meas_i: jnp.ndarray,      # (12,)
                             q0: jnp.ndarray,
                             lambda_reg: float):
 
    def loss_tau(tau):
        res = solve_equilibrium_with_tau(robot, u_i, tau, q0)
        q_star = res.value
        pred = markers_from_q(robot, q_star)           # (12,)
        err = pred - p_meas_i
        return 0.5 * (err @ err) + 0.5 * lambda_reg * (tau @ tau)
    return loss_tau

#MODIFIED IN ORDER TO RETURN q_star AS WELL
def solve_tau_star_for_batch(robot: TendonActuatedGVS,
                             u_batch: jnp.ndarray,                 # (na, M)
                             measured_markers_batch: jnp.ndarray,  # (12, M)
                             q0: jnp.ndarray,
                             lambda_reg: float = 1e-6,
                             steps: int = 50,
                             lr: float = 1e-2):

    M = u_batch.shape[1]
    dof = q0.shape[0]
    tau_stars = []
    q_stars = []
    tau = jnp.zeros((dof,), dtype=jnp.float64)  # cold-start

    for m in range(M):
        u_i = u_batch[:, m]
        p_meas_i = measured_markers_batch[:, m]
        loss_tau = make_tau_loss_for_sample(robot, u_i, p_meas_i, q0, lambda_reg)
        val_and_grad = eqx.filter_value_and_grad(loss_tau)

        opt = optax.adam(lr)
        opt_state = opt.init(tau)

        best_tau = tau
        best_val = jnp.inf


        for step in range(steps):
            val, g = val_and_grad(tau)
            updates, opt_state = opt.update(g, opt_state, tau)
            
            print("Current step:", step)
            print(f"[Sample {m:02d}] Current tau loss={float(val):.6e}")
            print(f"[Sample {m:02d}] Current tau={tau}")
            
            tau = optax.apply_updates(tau, updates)

            if val < best_val:
                best_val = val
                best_tau = tau

            
        res_best = solve_equilibrium_with_tau(robot, u_i, best_tau, q0)
        q_star_best = res_best.value
        tau_stars.append(best_tau)
        q_stars.append(q_star_best)
        print(f"[Sample {m:02d}] Identified tau* with loss={float(best_val):.6e}")

    tau_star_batch = jnp.stack(tau_stars, axis=1)  # (dof, M)
    q_star_batch = jnp.stack(q_stars, axis=1)      # (dof, M)
    return tau_star_batch, q_star_batch


# ========================= NN to fit tau*(u) =========================

class TauNN(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, in_dim: int, out_dim: int, width: int = 64, depth: int = 2, *, key: Array):
        self.mlp = eqx.nn.MLP(in_dim, out_dim, width, depth, key=key)

    def __call__(self, x):
        return self.mlp(x)

def train_tau_nn(u_batch: jnp.ndarray,                 # (na, M)
                 tau_star_batch: jnp.ndarray,          # (dof, M)
                 q_star_batch: jnp.ndarray,            # (dof, M)
                 gamma_wd: float = 1e-4,
                 steps: int = 2000,
                 lr: float = 1e-3,
                 width: int = 64,
                 depth: int = 2,
                 key: Array | None = None):
    """
    Allena NN: x = [t; u] -> tau_ext (dof,)
    Loss: mean 0.5||NN(x_i)-tau*_i||^2 + 0.5*gamma*||theta||^2
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    M = u_batch.shape[1]
    na = u_batch.shape[0]
    dof = tau_star_batch.shape[0]

    # Build features X: (M, na)
    X = u_batch.T                             # (M, na)
    # X = jnp.concatenate([u_batch.T, q_star_batch.T], axis=1)  # (M, na+dof)
    Y = tau_star_batch.T                      # (M, dof)

    model = TauNN(in_dim=X.shape[1], out_dim=dof, width=width, depth=depth, key=key)
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    def loss_model(m: TauNN):
        pred = jax.vmap(lambda x: m(x))(X)               # (M, dof)
        diff = pred - Y
        data_term = 0.5 * jnp.mean(jnp.sum(diff * diff, axis=1))
        # weight decay su tutti i tensori trainabili
        # l2 = sum(jnp.sum(p**2) for p in jax.tree_leaves(eqx.filter(m, eqx.is_inexact_array)))
        # l2 = sum(jnp.sum(p**2) for p in tree_leaves(eqx.filter(m, eqx.is_inexact_array)))
        l2 = sum(jnp.sum(p**2) for p in jax.tree.leaves(eqx.filter(m, eqx.is_inexact_array)))
        return data_term + 0.5 * gamma_wd * l2
        
    loss_and_grad = eqx.filter_value_and_grad(loss_model)

    for step in range(steps):
        loss_val, grads = loss_and_grad(model)
        updates, opt_state = opt.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        # if step % 200 == 0:
        # print(f"[NN step {step:04d}] loss={float(loss_val):.6e}")

    return model

#function of both u and q
# def tau_nn_eval_fn(tau_nn_model: TauNN):
#     # ritorna una funzione tau(u,q) = NN([u;q])
#     def _eval(u: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
#         x = jnp.concatenate([u, q], axis=0)
#         return tau_nn_model(x)
#     return jax.jit(_eval)

def tau_nn_eval_fn(tau_nn_model: TauNN):
    # ritorna una funzione tau(u) = NN([u])
    def _eval(u: jnp.ndarray) -> jnp.ndarray:
        return tau_nn_model(u)
    return jax.jit(_eval)


def simulate_piecewise_dynamics(robot: TendonActuatedGVS,
                                    t: jnp.ndarray,          # (T,)
                                    u_traj: jnp.ndarray,      # (T, na)
                                    q0: jnp.ndarray,
                                    qd0: jnp.ndarray,
                                    tau_mode: str,            # "zero" | "nn"
                                    tau_nn_model: TauNN | None = None):

    assert tau_mode in ("zero", "nn")
    T = int(t.shape[0])
    dof = int(q0.shape[0])
    assert u_traj.shape[0] == T

    if tau_mode == "nn":
        assert tau_nn_model is not None
        tau_eval = tau_nn_eval_fn(tau_nn_model)

    t_hist = []
    q_hist = []
    qd_hist = []
    p_hist = []

    # stato corrente
    q = q0
    qd = qd0

    # primo punto
    t_hist.append(float(t[0]))
    q_hist.append(q)
    qd_hist.append(qd)
    p_hist.append(markers_from_q(robot, q))

    # integrazione a tratti
    for k in range(T - 1):
        t0 = float(t[k]); t1 = float(t[k + 1])
        dt_seg = t1 - t0
        u_k = u_traj[k]

        if tau_mode == "zero":
            tau_k = jnp.zeros((dof,), dtype=q.dtype)
        else:
            # tau_k = tau_eval(u_k, q)  # dipende da u e q correnti
            # print(f"Segment {k:02d}: Computed tau_k={tau_k} using NN at q={q} and u={u_k}")
            tau_k = tau_eval(u_k)
            print(f"Segment {k:02d}: Computed tau_k={tau_k} using NN at u={u_k}")
        dt = 1e-4
        
        # integra da t0 a t1 con u e tau_k costanti
        ts, q_ts, qd_ts = robot.resolve_upon_time(
            # q0=q, qd0=qd, u=u_k, t0=t0, t1=t1, dt=dt, tau_ext=tau_k, save_dt=dt_seg/10.0, saveat_ts=jnp.array([t1-dt]), max_steps=None
            q0=q, qd0=qd, u=u_k, t0=t0, t1=t1, dt=dt, tau_ext=tau_k, save_dt=dt_seg/10.0, max_steps=None
        )
        print(f"Segment {k:02d}: Integrated from t={t0:.4f} to t={t1:.4f} with dt={dt:.4f}, got {ts.shape[0]} steps.")
        print(f"Segment {k:02d}:Final state q={q_ts[-1]}, qd={qd_ts[-1]}")
        print("q_all:", q_ts)
        
        # accumula evitando il duplicato del primo campione
        t_hist.append(float(ts[-1]))
        q_hist.append(onp.asarray(q_ts[-1]))
        qd_hist.append(onp.asarray(qd_ts[-1]))
        p_hist.append(onp.asarray(markers_from_q(robot, q_ts[-1])))
        # aggiorna stato finale del tratto
        q = q_ts[-1]; qd = qd_ts[-1]
        print("p_all:", p_hist)
        
    # t_hist = jnp.asarray(onp.asarray(t_hist))
    # q_hist = jnp.asarray(onp.asarray(q_hist))
    # qd_hist = jnp.asarray(onp.asarray(qd_hist))
    # p_hist = jnp.asarray(onp.asarray(p_hist))
    t_hist = jnp.array(t_hist)        
    q_hist = jnp.stack(q_hist, axis=0)   
    qd_hist = jnp.stack(qd_hist, axis=0) 
    p_hist = jnp.stack(p_hist, axis=0)   
    return t_hist, q_hist, qd_hist, p_hist

### BODY DEFINITION OF THE SOFT ROBOT ###

#2 link version 
# Link 1
link1 = LinkAttributes(
    section="Circular",
    # E=3.04e5,
    E=4.25e5,
    nu=0.45,
    # rho=1310.0,
    rho=1209.0,
    eta=1e4,
    L=0.0250+0.2550+0.0250,
    r_i=0.01541,
    r_f=0.00642,
)

# Link 2
link2 = LinkAttributes(
    section="Circular",
    # E=3.04e5,
    E=4.25e5,
    nu=0.45,
    # rho=1310.0,
    rho=1209.0,
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


# Prepare batch data (measured coordinates and inputs)
def _pack_markers(p1, p2, p3, p4):
    return jnp.concatenate([p1, p2, p3, p4], axis=0)  # (12,)

measured_list = [
    _pack_markers(p1_m1u01,  p2_m1u01,  p3_m1u01,  p4_m1u01),
    _pack_markers(p1_m1u015, p2_m1u015, p3_m1u015, p4_m1u015),
    _pack_markers(p1_m1u02,  p2_m1u02,  p3_m1u02,  p4_m1u02),
    _pack_markers(p1_m1u025, p2_m1u025, p3_m1u025, p4_m1u025),
    _pack_markers(p1_m2u01,  p2_m2u01,  p3_m2u01,  p4_m2u01),
    _pack_markers(p1_m2u015, p2_m2u015, p3_m2u015, p4_m2u015),
    _pack_markers(p1_m2u02,  p2_m2u02,  p3_m2u02,  p4_m2u02),
    _pack_markers(p1_m2u025, p2_m2u025, p3_m2u025, p4_m2u025),
]
measured_markers_batch = jnp.stack(measured_list, axis=1) 
radius = 0.0325
levels = jnp.array([-0.01, -0.015, -0.02, -0.025], dtype=jnp.float64) / radius  # (4,)

u_m1 = jnp.stack([jnp.array([lvl, 0.0], dtype=jnp.float64) for lvl in levels], axis=1)  # (2,4)
u_m2 = jnp.stack([jnp.array([0.0, lvl], dtype=jnp.float64) for lvl in levels], axis=1)  # (2,4)
u_batch = jnp.concatenate([u_m1, u_m2], axis=1)  # shape (2, 8)


print("u_batch:", u_batch)
print("measured_markers_batch:", measured_markers_batch)

M = measured_markers_batch.shape[1]
na = u_batch.shape[0]
dof = int(sum(robot.V_dof.reshape(-1)))
assert u_batch.shape == (na, M)
assert measured_markers_batch.shape == (12, M)

q0 = jnp.zeros((dof,), dtype=jnp.float64)
qd0 = jnp.zeros((dof,), dtype=jnp.float64)

# 1) Identify tau*_i for each sample
lambda_reg = 1e-6   # regularization on tau
# print("Starting identification of tau* for each sample ===")
# tau_star_batch, q_star_batch = solve_tau_star_for_batch(
#     robot, u_batch, measured_markers_batch, q0,
#     lambda_reg=lambda_reg, steps=200, lr=1e-2
# )  # (dof, M)


# Save
# onp.save("tau_star_batch_uq.npy", onp.asarray(tau_star_batch))
# onp.save("q_star_batch_uq.npy", onp.asarray(q_star_batch))

# tau_star_batch = jnp.array(onp.load("tau_star_batch.npy"))
# q_star_list = []
# for m in range(M):
#     res_m = solve_equilibrium_with_tau(robot, u_batch[:, m], tau_star_batch[:, m], q0)
#     q_star_list.append(res_m.value)
# q_star_batch = jnp.stack(q_star_list, axis=1)  # (dof, M)
# onp.save("tau_star_batch_uq.npy", onp.asarray(tau_star_batch))
# onp.save("q_star_batch_uq.npy", onp.asarray(q_star_batch))


# Load (later)
tau_star_batch = jnp.array(onp.load("tau_star_batch_uq.npy"))# tau_star_batch = jnp.array(onp.load("tau_star_batch_uq.npy"))
q_star_batch = jnp.array(onp.load("q_star_batch_uq.npy"))


print("Identified tau* shape:", tau_star_batch.shape)
print("Identified q* shape:", q_star_batch.shape)

# # Save in a text file
# onp.savetxt("tau_star_batch.txt", onp.asarray(tau_star_batch))

# # Load (shape must be known afterward)
# data = onp.loadtxt("tau_star_batch.txt")
# tau_star_loaded = jnp.array(data)
# tau_star_loaded = tau_star_loaded.reshape(tau_star_batch.shape) # if necessary


tau_nn = train_tau_nn(
    u_batch, tau_star_batch, q_star_batch=q_star_batch,
    gamma_wd=1e-4, steps=3000, lr=1e-3, width=64, depth=2, key=jax.random.PRNGKey(0)
)
print("Trained tau_NN model.")

### DYNAMIC SIMULATION COMPARISON ###

# Costruisci una sequenza temporale ripetendo gli 8 input per finestre di durata hold_time
hold_time = 0.1
dt_dyn = 0.01
steps_per_hold = int(hold_time / dt_dyn)
u_seq = onp.repeat(onp.asarray(u_batch.T), steps_per_hold, axis=0)  # (8*steps_per_hold, na)
t_seq = onp.arange(u_seq.shape[0]) * dt_dyn
u_traj = jnp.asarray(u_seq, dtype=jnp.float64)
t_traj = jnp.asarray(t_seq, dtype=jnp.float64)
qd0_dyn = jnp.zeros_like(q0)


# #Caso B: tau_ext = tau_NN([u; q])
# Caso B: tau_ext = tau_NN([u])


# tN_hist, qN_hist, qdN_hist, mkN_hist = simulate_piecewise_dynamics(
    # robot, t_traj, u_traj, q0, qd0_dyn, tau_mode="nn", tau_nn_model=tau_nn
# )#this fails :()

u_k = u_traj[0]
t0 = 0.0
t1 = 2.0
dt = 1e-4
tau_k = tau_nn(u_k)
print(f"Computed tau_k={tau_k} using NN at u={u_k}")
        
# integra da t0 a t1 con u e tau_k costanti
ts, q_ts, qd_ts = robot.resolve_upon_time(
    q0=q0, qd0=qd0, u=u_k, t0=t0, t1=t1, dt=dt, tau_ext=tau_k, max_steps=None
)
print("Completed dynamic simulation with tau=NN(u).")
mkN_hist = jax.vmap(markers_from_q, in_axes=(None, 0))(robot, q_ts)

print(f"Integrated from t={t0:.4f} to t={t1:.4f} with dt={dt:.4f}, got {ts.shape[0]} steps.")
print(f"Final state q={q_ts[-1]}, qd={qd_ts[-1]}")
print("q_all:", q_ts)

# # Caso A: tau_ext = 0
# t0_hist, q0_hist, qd0_hist, mk0_hist = simulate_piecewise_dynamics(
#     robot, t_traj, u_traj, q0, qd0_dyn, tau_mode="zero"
# )
# # print("Completed dynamic simulation with tau=0.")
# # print("t0_hist:", t0_hist)
# # print("q0_hist:", q0_hist)
# # print("mk0_hist:", mk0_hist)

# # Plot rapido: traiettoria marker p4 (x,y,z)
# p4_zero = onp.asarray(mk0_hist[:, 9:12])
# p4_nn   = onp.asarray(mkN_hist[:, 9:12])
# tt = onp.asarray(t_traj)

# fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
# for i, comp in enumerate(["x","y","z"]):
#     axs[i].plot(tt, p4_zero[:, i], "b--", label="tau=0")
#     axs[i].plot(tt, p4_nn[:, i],   "r-.", label="tau=NN(u,q)")
#     axs[i].set_ylabel(f"p4 {comp} [m]")
#     axs[i].grid(True, linestyle="--", alpha=0.4)
# axs[-1].set_xlabel("time [s]")
# axs[0].legend()
# fig.suptitle("Dynamic simulation via resolve_upon_time: tau=0 vs tau=NN(u,q)")
# fig.tight_layout(rect=[0,0.03,1,0.95])
# fig.savefig("dyn_p4_resolve_compare.svg", bbox_inches="tight")
# fig.savefig("dyn_p4_resolve_compare.pdf", bbox_inches="tight")
# fig.savefig("dyn_p4_resolve_compare.jpg", dpi=300, bbox_inches="tight")
# plt.show()