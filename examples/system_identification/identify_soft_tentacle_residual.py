import jax
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optimistix as optx
import equinox as eqx
import optax
from jax import Array
import numpy as onp
import seaborn as sns

from soromox.systems import (
    CrossSectionGeometry,
    GVSSegment,
    JointSpec,
    LinkSpec,
    LinearTendonRoutingParams,
    StrainBasisSpec,
    TendonActuatedGVS,
)

jax.config.update("jax_enable_x64", True)
# print("JAX default backend:", jax.default_backend())
# print("JAX devices:", jax.devices())

### MOCAP DATA TRANSFORMATION FUNCTIONS ###

def _transform_points(pb: jnp.ndarray, p1: jnp.ndarray, p2: jnp.ndarray, p3: jnp.ndarray, p4: jnp.ndarray,
                      Rot: jnp.ndarray, R_alg1: jnp.ndarray, R_alg2: jnp.ndarray, offset: jnp.ndarray, dyn_sel: bool):
    if dyn_sel:
        offset_col = offset.reshape(3, 1)
        offset_col = jnp.broadcast_to(offset_col, pb.shape)
    else:
        offset_col = offset

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


def compute_marker_errors(robot, u_batch, q0, measured_markers_batch, tau_batch):
    """
    Returns marker errors (M, 12) = difference between each marker and corresponding experiment
    """
    dof = q0.shape[0]
    u_T = u_batch.T
    meas_T = measured_markers_batch.T
    M = u_T.shape[0]
    if tau_batch.ndim == 1:
        tau_mat = jnp.tile(tau_batch.reshape(dof, 1), (1, M))      # (dof, M)
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


###drawing plots
def draw_robot_curve(
    robot: TendonActuatedGVS,
    q: Array,
    num_points: int = 50,
):
    batched_forward_kinematics = jax.vmap(robot.forward_kinematics, in_axes=(None, 0))
    L_max = robot.length

    s_ps = jnp.linspace(0, L_max, num_points)
    g_ps = batched_forward_kinematics(q, s_ps)[:, :3, 3]

    curve = onp.array(g_ps, dtype=onp.float64)
    return curve  # (N, 3)


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
    p4 = tip_at_s(robot.length, jnp.array([0.008, 0.0, 0.0]))
    return jnp.concatenate([p1, p2, p3, p4], axis=0)  # (12,)

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

        tau_min = jnp.array([-1,-1,-1,-1,-1,-1,-1,-0.01,-0.01])
        tau_max = jnp.array([1, 1, 1, 1, 1, 1, 1, 0.01, 0.01])

        res = solve_equilibrium_with_tau(robot, u_i, tau, q0)
        q_star = res.value
        pred = markers_from_q(robot, q_star)           # (12,)
        err = pred - p_meas_i

        penalty = jnp.sum(jnp.square(jnp.maximum(0.0, tau - tau_max))) \
                + jnp.sum(jnp.square(jnp.maximum(0.0, tau_min - tau)))


        return 0.5 * (err @ err) \
               + 0.5 * lambda_reg * (tau @ tau) \
               + 1e1 * penalty


    return loss_tau


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
    tau0 = jnp.zeros((dof,), dtype=jnp.float64)  # cold-start

    for m in range(M):
        u_i = u_batch[:, m]
        p_meas_i = measured_markers_batch[:, m]
        loss_tau = make_tau_loss_for_sample(robot, u_i, p_meas_i, q0, lambda_reg)
        val_and_grad = eqx.filter_value_and_grad(loss_tau)

        opt = optax.adam(lr)
        opt_state = opt.init(tau0)

        best_tau = tau0
        best_val = jnp.inf

        tau = tau0

        for step in range(steps):
            val, g = val_and_grad(tau)

            print("Current step:", step)
            print(f"[Sample {m:02d}] Current tau loss={float(val):.6e}")
            print(f"[Sample {m:02d}] Current tau={tau}")

            if val < best_val:
                best_val = val
                best_tau = tau

            updates, opt_state = opt.update(g, opt_state, tau)
            tau = optax.apply_updates(tau, updates)

        tau_stars.append(best_tau)

        print(f"[Sample {m:02d}] Identified tau* with loss={float(best_val):.6e}")

    tau_star_batch = jnp.stack(tau_stars, axis=1)  # (dof, M)
    return tau_star_batch


# ========================= NN to fit tau*(u) =========================

class TauNN(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, in_dim: int, out_dim: int, width: int = 64, depth: int = 2, *, key: Array):
        self.mlp = eqx.nn.MLP(in_dim, out_dim, width, depth, key=key)

    def __call__(self, x):
        return self.mlp(x)

def train_tau_nn(u_batch: jnp.ndarray,                 # (na, M)
                 tau_star_batch: jnp.ndarray,          # (dof, M)
                 gamma_wd: float = 1e-4,
                 steps: int = 2000,
                 lr: float = 1e-3,
                 width: int = 64,
                 depth: int = 2,
                 key: Array | None = None):

    if key is None:
        key = jax.random.PRNGKey(0)
    M = u_batch.shape[1]
    na = u_batch.shape[0]
    dof = tau_star_batch.shape[0]

    # Build features X: (M, na)
    X = u_batch.T                             # (M, na)
    Y = tau_star_batch.T                      # (M, dof)

    model = TauNN(in_dim=na, out_dim=dof, width=width, depth=depth, key=key)

    #base version
    # opt = optax.adam(lr)

    #cosine decay version
    lr_schedule = optax.cosine_decay_schedule(
        init_value=1e-3,
        decay_steps=15000,
        alpha=0.05
    )
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr_schedule)
    )

    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    def loss_model(m: TauNN):
        pred = jax.vmap(lambda x: m(x))(X)               # (M, dof)
        diff = pred - Y
        data_term = 0.5 * jnp.mean(jnp.sum(diff * diff, axis=1))

        l2 = sum(jnp.sum(p**2) for p in jax.tree.leaves(eqx.filter(m, eqx.is_inexact_array)))
        return data_term + 0.5 * gamma_wd * l2

    loss_and_grad = eqx.filter_value_and_grad(loss_model)

    losses = []
    for step in range(steps):
        loss_val, grads = loss_and_grad(model)
        updates, opt_state = opt.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        losses.append(float(loss_val))

        # DEBUG PRINT
        # if step % 200 == 0:
        # print(f"[NN step {step:04d}] loss={float(loss_val):.6e}")

    return model, losses





if __name__ == "__main__":
    ### DATA LOADING AND PROCESSING ###
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

    df = pd.read_csv("data/m1_u01.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[12:15]
    p1 = filtered[0:3]
    p2 = filtered[9:12]
    p3 = filtered[3:6]
    p4 = filtered[6:9]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m1u01 = res["pb_tr"]
    p1_m1u01 = res["p1_tr"]
    p2_m1u01 = res["p2_tr"]
    p3_m1u01 = res["p3_tr"]
    p4_m1u01 = res["p4_tr"]

    df = pd.read_csv("data/m1_u015.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[9:12]
    p1 = filtered[6:9]
    p2 = filtered[12:15]
    p3 = filtered[3:6]
    p4 = filtered[0:3]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m1u015 = res["pb_tr"]
    p1_m1u015 = res["p1_tr"]
    p2_m1u015 = res["p2_tr"]
    p3_m1u015 = res["p3_tr"]
    p4_m1u015 = res["p4_tr"]

    df = pd.read_csv("data/m1_u02.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[9:12]
    p1 = filtered[6:9]
    p2 = filtered[12:15]
    p3 = filtered[3:6]
    p4 = filtered[0:3]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m1u02 = res["pb_tr"]
    p1_m1u02 = res["p1_tr"]
    p2_m1u02 = res["p2_tr"]
    p3_m1u02 = res["p3_tr"]
    p4_m1u02 = res["p4_tr"]

    df = pd.read_csv("data/m1_u025.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[12:15]
    p1 = filtered[9:12]
    p2 = filtered[0:3]
    p3 = filtered[6:9]
    p4 = filtered[3:6]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m1u025 = res["pb_tr"]
    p1_m1u025 = res["p1_tr"]
    p2_m1u025 = res["p2_tr"]
    p3_m1u025 = res["p3_tr"]
    p4_m1u025 = res["p4_tr"]

    df = pd.read_csv("data/m2_u01.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[6:9]
    p1 = filtered[0:3]
    p2 = filtered[12:15]
    p3 = filtered[9:12]
    p4 = filtered[3:6]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m2u01 = res["pb_tr"]
    p1_m2u01 = res["p1_tr"]
    p2_m2u01 = res["p2_tr"]
    p3_m2u01 = res["p3_tr"]
    p4_m2u01 = res["p4_tr"]

    df = pd.read_csv("data/m2_u015.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[9:12]
    p1 = filtered[6:9]
    p2 = filtered[12:15]
    p3 = filtered[3:6]
    p4 = filtered[0:3]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m2u015 = res["pb_tr"]
    p1_m2u015 = res["p1_tr"]
    p2_m2u015 = res["p2_tr"]
    p3_m2u015 = res["p3_tr"]
    p4_m2u015 = res["p4_tr"]

    df = pd.read_csv("data/m2_u02.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[9:12]
    p1 = filtered[6:9]
    p2 = filtered[12:15]
    p3 = filtered[3:6]
    p4 = filtered[0:3]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m2u02 = res["pb_tr"]
    p1_m2u02 = res["p1_tr"]
    p2_m2u02 = res["p2_tr"]
    p3_m2u02 = res["p3_tr"]
    p4_m2u02 = res["p4_tr"]

    df = pd.read_csv("data/m2_u025.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[6:9]
    p1 = filtered[3:6]
    p2 = filtered[9:12]
    p3 = filtered[0:3]
    p4 = filtered[12:15]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m2u025 = res["pb_tr"]
    p1_m2u025 = res["p1_tr"]
    p2_m2u025 = res["p2_tr"]
    p3_m2u025 = res["p3_tr"]
    p4_m2u025 = res["p4_tr"]

    df = pd.read_csv("data/m1_u005.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[9:12]
    p1 = filtered[3:6]
    p2 = filtered[6:9]
    p3 = filtered[12:15]
    p4 = filtered[0:3]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m1_005, p1_m1_005, p2_m1_005, p3_m1_005, p4_m1_005 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

    df = pd.read_csv("data/m1_u012.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[9:12]
    p1 = filtered[6:9]
    p2 = filtered[12:15]
    p3 = filtered[3:6]
    p4 = filtered[0:3]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m1_012, p1_m1_012, p2_m1_012, p3_m1_012, p4_m1_012 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

    df = pd.read_csv("data/m1_u018.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[6:9]
    p1 = filtered[3:6]
    p2 = filtered[9:12]
    p3 = filtered[0:3]
    p4 = filtered[12:15]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m1_018, p1_m1_018, p2_m1_018, p3_m1_018, p4_m1_018 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

    df = pd.read_csv("data/m2_u005.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[6:9]
    p1 = filtered[3:6]
    p2 = filtered[9:12]
    p3 = filtered[0:3]
    p4 = filtered[12:15]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m2_005, p1_m2_005, p2_m2_005, p3_m2_005, p4_m2_005 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

    df = pd.read_csv("data/m2_u012.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[6:9]
    p1 = filtered[3:6]
    p2 = filtered[9:12]
    p3 = filtered[12:15]
    p4 = filtered[0:3]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m2_012, p1_m2_012, p2_m2_012, p3_m2_012, p4_m2_012 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

    df = pd.read_csv("data/m2_u018.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[3:6]
    p1 = filtered[0:3]
    p2 = filtered[6:9]
    p3 = filtered[9:12]
    p4 = filtered[12:15]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m2_018, p1_m2_018, p2_m2_018, p3_m2_018, p4_m2_018 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

    df = pd.read_csv("data/m12_u001.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[6:9]
    p1 = filtered[12:15]
    p2 = filtered[9:12]
    p3 = filtered[0:3]
    p4 = filtered[3:6]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m12_01, p1_m12_01, p2_m12_01, p3_m12_01, p4_m12_01 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

    df = pd.read_csv("data/m12_u002.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[9:12]
    p1 = filtered[6:9]
    p2 = filtered[12:15]
    p3 = filtered[3:6]
    p4 = filtered[0:3]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m12_02, p1_m12_02, p2_m12_02, p3_m12_02, p4_m12_02 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

    df = pd.read_csv("data/m12_u001002.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[9:12]
    p1 = filtered[6:9]
    p2 = filtered[3:6]
    p3 = filtered[0:3]
    p4 = filtered[12:15]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m12_0102, p1_m12_0102, p2_m12_0102, p3_m12_0102, p4_m12_0102 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

    df = pd.read_csv("data/m12_u0015002.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[12:15]
    p1 = filtered[3:6]
    p2 = filtered[9:12]
    p3 = filtered[6:9]
    p4 = filtered[0:3]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m12_01502, p1_m12_01502, p2_m12_01502, p3_m12_01502, p4_m12_01502 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

    df = pd.read_csv("data/m12_u002001.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[12:15]
    p1 = filtered[6:9]
    p2 = filtered[9:12]
    p3 = filtered[3:6]
    p4 = filtered[0:3]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m12_0201, p1_m12_0201, p2_m12_0201, p3_m12_0201, p4_m12_0201 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

    df = pd.read_csv("data/m12_u0020015.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[9:12]
    p1 = filtered[3:6]
    p2 = filtered[6:9]
    p3 = filtered[0:3]
    p4 = filtered[12:15]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m12_02015, p1_m12_02015, p2_m12_02015, p3_m12_02015, p4_m12_02015 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

    df = pd.read_csv("data/m12_u00100005.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[9:12]
    p1 = filtered[6:9]
    p2 = filtered[12:15]
    p3 = filtered[3:6]
    p4 = filtered[0:3]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m12_01005, p1_m12_01005, p2_m12_01005, p3_m12_01005, p4_m12_01005 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

    df = pd.read_csv("data/m12_u00050010.csv", header=None)
    vals = df.iloc[:, 2:].to_numpy(dtype=float)
    filtered = jnp.asarray(vals.mean(axis=0))
    pb = filtered[9:12]
    p1 = filtered[6:9]
    p2 = filtered[12:15]
    p3 = filtered[3:6]
    p4 = filtered[0:3]
    res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset, dyn_sel=False)
    pb_m12_00501, p1_m12_00501, p2_m12_00501, p3_m12_00501, p4_m12_00501 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

    ### END DATA LOADING AND PROCESSING ###

    ### BODY DEFINITION OF THE SOFT ROBOT ###
    # Link 1
    link1 = LinkSpec(
        cross_section_geometry=CrossSectionGeometry.CIRCULAR,
        E=3.15e5,
        nu=0.45,
        rho=1320.0,
        eta=1e4,
        L=0.0250 + 0.2550 + 0.0250,
        r_i=0.01541,
        r_f=0.00642,
    )

    # Link 2
    link2 = LinkSpec(
        cross_section_geometry=CrossSectionGeometry.CIRCULAR,
        E=3.15e5,
        nu=0.45,
        rho=1320.0,
        eta=1e4,
        L=0.0550,
        r_i=0.00642,
        r_f=0.00480,
    )
    joint1 = JointSpec(type="fixed")
    joint2 = JointSpec(type="fixed")
    basis1 = StrainBasisSpec(
        type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[0, 1, 1, 1, 0, 0]
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
    p0 = jnp.array([jnp.sqrt(0.5), 0.0, jnp.sqrt(0.5), 0.0, 0.0, 0.0, 0.0])

    robot = TendonActuatedGVS.from_segments(
        [
            GVSSegment(
                link=link1, joint=joint1, basis=basis1, num_gauss_points=num_gauss_points[0]
            ),
            GVSSegment(
                link=link2, joint=joint2, basis=basis2, num_gauss_points=num_gauss_points[1]
            ),
        ],
        gravity=jnp.asarray(g),
        base_pose=p0,
        active_tendon_routing=active_tendon_routing,
        scale_rotational_basis_by_length=True,
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

        _pack_markers(p1_m1_005, p2_m1_005, p3_m1_005, p4_m1_005),
        _pack_markers(p1_m1_012, p2_m1_012, p3_m1_012, p4_m1_012),
        _pack_markers(p1_m1_018, p2_m1_018, p3_m1_018, p4_m1_018),
        _pack_markers(p1_m2_005, p2_m2_005, p3_m2_005, p4_m2_005),
        _pack_markers(p1_m2_012, p2_m2_012, p3_m2_012, p4_m2_012),
        _pack_markers(p1_m2_018, p2_m2_018, p3_m2_018, p4_m2_018),

        _pack_markers(p1_m12_01,    p2_m12_01,    p3_m12_01,    p4_m12_01),
        _pack_markers(p1_m12_02,    p2_m12_02,    p3_m12_02,    p4_m12_02),
        _pack_markers(p1_m12_0102,  p2_m12_0102,  p3_m12_0102,  p4_m12_0102),
        _pack_markers(p1_m12_01502, p2_m12_01502, p3_m12_01502, p4_m12_01502),
        _pack_markers(p1_m12_0201,  p2_m12_0201,  p3_m12_0201,  p4_m12_0201),
        _pack_markers(p1_m12_02015, p2_m12_02015, p3_m12_02015, p4_m12_02015),
        _pack_markers(p1_m12_01005,  p2_m12_01005,  p3_m12_01005,  p4_m12_01005),
        _pack_markers(p1_m12_00501,  p2_m12_00501,  p3_m12_00501,  p4_m12_00501),
    ]
    measured_markers_batch = jnp.stack(measured_list, axis=1)
    radius = 0.0325

    def _neg(val):
        return -float(val) / float(radius)

    # Define mapping from sample name to u
    sample_to_u = {
        "m1u01":      jnp.array([_neg(0.1), 0.0]),
        "m1u015":     jnp.array([_neg(0.15), 0.0]),
        "m1u02":      jnp.array([_neg(0.2), 0.0]),
        "m1u025":     jnp.array([_neg(0.25), 0.0]),
        "m2u01":      jnp.array([0.0, _neg(0.1)]),
        "m2u015":     jnp.array([0.0, _neg(0.15)]),
        "m2u02":      jnp.array([0.0, _neg(0.2)]),
        "m2u025":     jnp.array([0.0, _neg(0.25)]),

        "m1_005":      jnp.array([_neg(0.05), 0.0]),
        "m1_012":      jnp.array([_neg(0.12), 0.0]),
        "m1_018":      jnp.array([_neg(0.18), 0.0]),
        "m2_005":      jnp.array([0.0, _neg(0.05)]),
        "m2_012":      jnp.array([0.0, _neg(0.12)]),
        "m2_018":      jnp.array([0.0, _neg(0.18)]),

        "m12_001":         jnp.array([_neg(0.1), _neg(0.1)]),
        "m12_002":         jnp.array([_neg(0.2), _neg(0.2)]),
        "m12_001002":      jnp.array([_neg(0.1), _neg(0.2)]),
        "m12_0015002":     jnp.array([_neg(0.15), _neg(0.2)]),
        "m12_002001":      jnp.array([_neg(0.2), _neg(0.1)]),
        "m12_0020015":     jnp.array([_neg(0.2), _neg(0.15)]),
        "m12_00100005":    jnp.array([_neg(0.1), _neg(0.05)]),
        "m12_00050010":    jnp.array([_neg(0.05), _neg(0.1)]),
    }

    u_list = [
        sample_to_u["m1u01"],
        sample_to_u["m1u015"],
        sample_to_u["m1u02"],
        sample_to_u["m1u025"],
        sample_to_u["m2u01"],
        sample_to_u["m2u015"],
        sample_to_u["m2u02"],
        sample_to_u["m2u025"],

        sample_to_u["m1_005"],
        sample_to_u["m1_012"],
        sample_to_u["m1_018"],
        sample_to_u["m2_005"],
        sample_to_u["m2_012"],
        sample_to_u["m2_018"],

        sample_to_u["m12_001"],
        sample_to_u["m12_002"],
        sample_to_u["m12_001002"],
        sample_to_u["m12_0015002"],
        sample_to_u["m12_002001"],
        sample_to_u["m12_0020015"],
        sample_to_u["m12_00100005"],
        sample_to_u["m12_00050010"],
    ]
    u_batch = jnp.stack(u_list, axis=1)
    # print("u_batch:", u_batch)
    # print("measured_markers_batch:", measured_markers_batch)

    M = measured_markers_batch.shape[1]
    na = u_batch.shape[0]
    dof = sum(robot.dofs_per_segment.reshape(-1))
    assert u_batch.shape == (na, M)
    assert measured_markers_batch.shape == (12, M)

    q0 = jnp.zeros((dof,), dtype=jnp.float64)


    # 1) Identify tau*_i for each sample
    # lambda_reg = 1e-8   # regularization on tau
    # print("Starting identification of tau* for each sample ===")

    # tau_star_batch = solve_tau_star_for_batch(
    #     robot, u_batch, measured_markers_batch, q0,
    #     lambda_reg=lambda_reg, steps=300, lr=1e-2
    # )  # (dof, M)

    # # # Save
    # onp.save("data/tau_star_batch.npy", onp.asarray(tau_star_batch))
    # onp.savetxt("data/tau_star_batch.txt", onp.asarray(tau_star_batch))

    # Load (later)
    tau_star_batch_1act1 = jnp.array(onp.load("data/tau_star_batch_1actpt1.npy"))
    tau_star_batch_1act2 = jnp.array(onp.load("data/tau_star_batch_1actpt2.npy"))
    tau_star_batch_2act = jnp.array(onp.load("data/tau_star_batch_2act.npy"))
    tau_star_batch = jnp.concatenate([tau_star_batch_1act1,tau_star_batch_1act2, tau_star_batch_2act], axis=1)

    print("Identified tau* shape:", tau_star_batch.shape)
    # print("Identified tau* samples:", tau_star_batch)

    # Evaluate errors before and after tau* identification
    tau_zero_batch = jnp.zeros((dof, M), dtype=jnp.float64)
    errors_before = compute_marker_errors(robot, u_batch, q0, measured_markers_batch, tau_zero_batch)

    def marker_rmse(errors):
        # errors shape: (M, 12)
        errors_reshaped = errors.reshape(errors.shape[0], 4, 3)  # (M, 4, 3)
        rms_per_marker = jnp.sqrt(jnp.mean(jnp.sum(errors_reshaped**2, axis=2), axis=0))
        return rms_per_marker  # (4,)

    rmse_before = marker_rmse(errors_before)
    print("RMSE per marker before residual tau* identification:", rmse_before)

    # Evaluate errors after tau* identification
    errors_tau_star = compute_marker_errors(robot, u_batch, q0, measured_markers_batch, tau_star_batch)
    rmse_tau_star = marker_rmse(errors_tau_star)
    print("RMSE per marker with residual tau* identification:", rmse_tau_star)


    # ===== Plot: robot shape for each tau* =====
    q_star_list = []
    curves_list = []
    for m in range(M):
        u_i = u_batch[:, m]
        tau_i = tau_star_batch[:, m]

        res = solve_equilibrium_with_tau(robot, u_i, tau_i, q0)
        q_star = res.value
        q_star_list.append(onp.asarray(q_star))
        curves_list.append(draw_robot_curve(robot, q_star, num_points=80))

    all_pts = onp.concatenate(curves_list, axis=0)  # (sum N_i, 3)
    xmin, ymin, zmin = all_pts.min(axis=0)
    xmax, ymax, zmax = all_pts.max(axis=0)
    pad = 0.05 * max(xmax - xmin, ymax - ymin, zmax - zmin)
    lims = ( (xmin - pad, xmax + pad), (ymin - pad, ymax + pad), (zmin - pad, zmax + pad) )


    # Compute predicted marker positions from q_star
    predicted_markers_batch = []
    for m in range(M):
        q_star = q_star_list[m]
        pred_markers = markers_from_q(robot, q_star)  # (12,)
        predicted_markers_batch.append(pred_markers)
    predicted_markers_batch = jnp.stack(predicted_markers_batch, axis=1)  # (12, M)

    all_pts = []
    for curve in curves_list:
        all_pts.append(onp.asarray(curve))
    for m in range(M):
        meas = onp.asarray(measured_markers_batch[:, m]).reshape(4, 3)
        pred = onp.asarray(predicted_markers_batch[:, m]).reshape(4, 3)
        all_pts.append(meas)
        all_pts.append(pred)
    all_pts = onp.vstack(all_pts)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    ranges = maxs - mins
    max_range = ranges.max()
    center = 0.5 * (maxs + mins)

    fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={"projection": "3d"})

    alpha_hat = 1.0    # full opacity

    for m in range(M):
        curve = curves_list[m]

        # plot backbone
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], lw=3, alpha=alpha_hat)

        # predicted markers: red
        pred = onp.asarray(predicted_markers_batch[:, m]).reshape(4, 3)
        # measured markers: blue
        meas = onp.asarray(measured_markers_batch[:, m]).reshape(4, 3)

        # per-marker alpha gradient from marker1 -> marker4 (0.25 -> 1.0)
        grad_alphas = onp.linspace(0.25, 1.0, 4)
        red_col = (0.8, 0.1, 0.1)
        blue_col = (0.0, 0.2, 0.8)

        for i in range(4):
            ax.scatter(pred[i, 0], pred[i, 1], pred[i, 2], c=[red_col], s=90, marker='x', alpha=grad_alphas[i])
            ax.scatter(meas[i, 0], meas[i, 1], meas[i, 2], facecolors="none", edgecolors=blue_col, s=90, marker='o', alpha=grad_alphas[i])

    # Set axis labels and title (larger font)
    ax.set_xlabel("X [m]", fontsize=14)
    ax.set_ylabel("Y [m]", fontsize=14)
    ax.set_zlabel("Z [m]", fontsize=14)
    ax.view_init(elev=20, azim=15)

    # enforce equal aspect ratio for 3D plot
    ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
    ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
    ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)
    ax.set_box_aspect([1, 1, 1])

    # create a custom legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color='k', lw=3.0, linestyle='-', alpha=alpha_hat, label='Backbone'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markeredgecolor='red', markersize=8, label='Predicted markers'),
        Line2D([0], [0], marker='o', color='w', markeredgecolor='blue', markerfacecolor='none', markersize=8, label='Measured markers'),
    ]
    ax.legend(handles=legend_elems, loc='upper left', fontsize=14)

    plt.tight_layout()
    plt.tick_params(axis='both', labelsize=12)
    # plt.savefig("MARKERS_CONFIG_residual.pdf", format="pdf", bbox_inches="tight")
    # plt.savefig("MARKERS_CONFIG_residual.jpg", format="jpg", dpi=300, bbox_inches="tight")
    plt.show()


    # # # 2) Train NN: x=[u] -> tau*
    train_idx = jnp.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21])
    test_idx = jnp.setdiff1d(jnp.arange(M), train_idx) # 1,16 test

    # split data
    u_train = u_batch[:, train_idx]
    tau_star_train = tau_star_batch[:, train_idx]

    u_test = u_batch[:, test_idx]
    tau_star_test = tau_star_batch[:, test_idx]

    # model, losses = train_tau_nn(
    #     u_train, tau_star_train,
    #     gamma_wd=1e-5, steps=12000, lr=1e-3, width=512, depth=4, key=jax.random.PRNGKey(0)
    # )
    # # Save model
    # onp.save("tau_nn_model.npy", model)
    # eqx.tree_serialise_leaves("tau_nn_model_idxfix_test116.eqx", model)

    # # Plot training loss curve
    # plt.figure(figsize=(8, 5))
    # plt.plot(losses, label='Training Loss')
    # plt.xlabel('Training Step')
    # plt.ylabel('Loss')
    # plt.title('Neural Network Training Loss Curve')
    # plt.yscale('log')
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # # plt.savefig('training_loss_curve.svg', bbox_inches='tight')
    # # plt.savefig('training_loss_curve.pdf', bbox_inches='tight')
    # # plt.savefig('training_loss_curve.jpg', dpi=300, bbox_inches='tight')
    # plt.show()

    # # Load model
    model = TauNN(in_dim=na, out_dim=dof, width=512, depth=4, key=jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves("data/tau_nn_model.eqx", model)

    # Evaluate marker errors with tau_NN prediction
    tau_pred = []
    for m in range(M):
        u_i = u_batch[:, m]
        tau_pred_i = model(u_i)  # (dof,)
        tau_pred.append(tau_pred_i)

    tau_pred_batch = jnp.stack(tau_pred, axis=1)  # (dof, M)
    errors_tau_pred = compute_marker_errors(robot, u_batch, q0, measured_markers_batch, tau_pred_batch)
    rmse_tau_pred = marker_rmse(errors_tau_pred)
    print("RMSE per marker with tau* NN prediction:", rmse_tau_pred)


    # # # # === Plots: Random_id vs Opt_id vs Residual vs Residual_NN ===

    # Per-marker error per configuration (4 subplot)
    errs_zero = onp.asarray(errors_before).reshape(-1, 4, 3)      # (M, 4, 3)
    errs_star = onp.asarray(errors_tau_star).reshape(-1, 4, 3)    # (M, 4, 3)
    errs_pred = onp.asarray(errors_tau_pred).reshape(-1, 4, 3)    # (M, 4, 3)
    errnorms_zero = onp.linalg.norm(errs_zero, axis=2)            # (M, 4)
    errnorms_star = onp.linalg.norm(errs_star, axis=2)            # (M, 4)
    errnorms_pred = onp.linalg.norm(errs_pred, axis=2)            # (M, 4)


    # Spider plots: Error per marker across all configurations
    # Calculate RMS error for each marker across all configurations
    errnorm_zero_per_marker = onp.mean(errnorms_zero, axis=0)  # RMS for each marker
    errnorm_star_per_marker = onp.mean(errnorms_star, axis=0)  # RMS for each marker


    # Prepare data for plotting
    errnorm_zero_plot = errnorm_zero_per_marker.tolist()
    errnorm_zero_plot += errnorm_zero_plot[:1]
    errnorm_star_plot = errnorm_star_per_marker.tolist()
    errnorm_star_plot += errnorm_star_plot[:1]

    # Calculate RMS error for each marker across all configurations
    errnorm_pred_per_marker = onp.mean(errnorms_pred, axis=0)  # RMS for each marker

    # Prepare data for plotting
    errnorm_pred_plot = errnorm_pred_per_marker.tolist()
    errnorm_pred_plot += errnorm_pred_plot[:1]


    # Calculate RMS error per marker for initial parameters
    errors_before_id = onp.asarray(onp.load("data/errors_before.npy"))
    errs_before_id = onp.asarray(errors_before_id).reshape(-1, 4, 3)  # (M, 4, 3)
    errnorms_before_id = onp.linalg.norm(errs_before_id, axis=2)      # (M, 4)
    errnorm_before_id_per_marker = onp.mean(errnorms_before_id, axis=0)  # RMS for each marker

    # Prepare data for plotting
    errnorm_before_id_plot = errnorm_before_id_per_marker.tolist()
    errnorm_before_id_plot += errnorm_before_id_plot[:1]


    # Setup for 4-axis radar plot (one for each marker)
    angles_markers = onp.linspace(0, 2 * onp.pi, 4, endpoint=False).tolist()
    angles_markers += angles_markers[:1]  # Complete the circle
    marker_labels = [f'$M_{{{i+1}}}$' for i in range(4)]


    # Additional per-marker radar plot: include initial param + all cases
    fig, ax = plt.subplots(figsize=(12, 9), subplot_kw=dict(projection='polar'))

    # Plot
    ax.plot(angles_markers, errnorm_before_id_plot, 'o-', linewidth=2.5, label=r'Initial parameters', color='lightgray', markersize=8)
    ax.fill(angles_markers, errnorm_before_id_plot, alpha=0.25, color='lightgray')
    ax.plot(angles_markers, errnorm_zero_plot, 'o-', linewidth=2.5, label=r'Optimal parameters', color='gray', markersize=8)
    ax.fill(angles_markers, errnorm_zero_plot, alpha=0.25, color='gray')
    ax.plot(angles_markers, errnorm_star_plot, 's-', linewidth=2.5, label=r'Optimal residual', color='tab:blue', markersize=8)
    ax.fill(angles_markers, errnorm_star_plot, alpha=0.25, color='tab:blue')
    ax.plot(angles_markers, errnorm_pred_plot, '^--', linewidth=2.5, label=r'NN-predicted residual', color='skyblue', markersize=8)
    ax.fill(angles_markers, errnorm_pred_plot, alpha=0.25, color='skyblue')

    ax.set_xticks(angles_markers[:-1])
    ax.set_xticklabels(marker_labels, fontsize=16)
    ax.set_ylabel('Position RMSE [m]', labelpad=30, fontsize=16)
    ax.legend(loc='upper center', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True)

    plt.tight_layout()
    # plt.savefig("radar_error_per_marker_all_with_initial_center.pdf", format="pdf", bbox_inches="tight")
    # plt.savefig("radar_error_per_marker_all_with_initial_center.jpg", format="jpg", dpi=300, bbox_inches="tight")
    plt.show()


    # Violin plot: 12 violins (4 markers × 3 tau conditions)
    # Distribution of error for each marker across all 22 configurations for each tau value
    data_list = []
    tau_labels = [r'$\tau_{ext} = 0$', r'$\tau_{ext} = \tau_{ext}^*$', r'$\tau_{ext} = \tau_{ext,NN}$']
    tau_names = ['Zero', 'Star', 'NN']

    # Collect errors for each marker-tau combination across all configurations
    for marker_idx in range(4):
        # tau = 0
        for config_idx in range(M):
            data_list.append({
                'Marker': f'M{marker_idx + 1}',
                'Tau Condition': tau_labels[0],
                'Error': errnorms_zero[config_idx, marker_idx],
            })
        # tau = tau*
        for config_idx in range(M):
            data_list.append({
                'Marker': f'M{marker_idx + 1}',
                'Tau Condition': tau_labels[1],
                'Error': errnorms_star[config_idx, marker_idx],
            })
        # tau = tau_NN
        for config_idx in range(M):
            data_list.append({
                'Marker': f'M{marker_idx + 1}',
                'Tau Condition': tau_labels[2],
                'Error': errnorms_pred[config_idx, marker_idx],
            })

    df_violin = pd.DataFrame(data_list)
    # print("violin plot data:", df_violin.to_string())

    # Create violin plot with 12 violins
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.violinplot(data=df_violin, x='Marker', y='Error', hue='Tau Condition', cut=0, ax=ax,
                   palette=['tab:orange', 'tab:blue', 'tab:green'], split=False)
    ax.set_ylabel('Position Error Norm Distribution [m]', fontsize=18)
    ax.set_xlabel('Markers', fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.legend(loc='upper left', fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    plt.tight_layout()
    # plt.savefig("violin_error_distribution_12markers.svg", format="svg", bbox_inches="tight")
    # plt.savefig("violin_error_distribution_12markers.pdf", format="pdf", bbox_inches="tight")
    # plt.savefig("violin_error_distribution_12markers.jpg", format="jpg", dpi=300, bbox_inches="tight")
    plt.show()
