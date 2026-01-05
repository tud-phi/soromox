import jax
import pandas as pd
import jax.numpy as jnp
from soromox.systems import CrossSectionGeometry, TendonActuatedGVS
from soromox.systems.gvs import LinkAttributes, JointAttributes, BasisAttributes
from functools import partial
import matplotlib.pyplot as plt
import optimistix as optx
import equinox as eqx
import optax
import numpy as onp

jax.config.update("jax_enable_x64", True)

# jax.config.update("jax_platform_name", "gpu")  # or "cpu"

print("JAX default backend:", jax.default_backend())
print("JAX devices:", jax.devices())

### MOCAP DATA TRANSFORMATION FUNCTIONS ###
def _transform_points(
    pb: jnp.ndarray,
    p1: jnp.ndarray,
    p2: jnp.ndarray,
    p3: jnp.ndarray,
    p4: jnp.ndarray,
    Rot: jnp.ndarray,
    R_alg1: jnp.ndarray,
    R_alg2: jnp.ndarray,
    offset: jnp.ndarray,
):
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

    return {
        "pb_tr": pb_tr,
        "p1_tr": p1_tr,
        "p2_tr": p2_tr,
        "p3_tr": p3_tr,
        "p4_tr": p4_tr,
    }


### EXCEL DATA READING ###
Rot = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float64)
R_alg1 = jnp.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=jnp.float64
)
R_alg2 = jnp.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float64
)
offset = jnp.array([-0.03, -0.013, 0.0], dtype=jnp.float64)


df = pd.read_csv("data/m1_u01.csv", header=None)
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

df = pd.read_csv("data/m1_u015.csv", header=None)
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

df = pd.read_csv("data/m1_u02.csv", header=None)
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

df = pd.read_csv("data/m1_u025.csv", header=None)
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

df = pd.read_csv("data/m2_u01.csv", header=None)
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

df = pd.read_csv("data/m2_u015.csv", header=None)
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

df = pd.read_csv("data/m2_u02.csv", header=None)
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

df = pd.read_csv("data/m2_u025.csv", header=None)
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


def compute_marker_errors(robot, u_batch, q0, measured_markers_batch, E, nu, rho):
    """
    Restituisce gli errori (M, 12) = differenze per ogni marker e ogni esperimento
    """
    u_T = u_batch.T
    meas_T = measured_markers_batch.T
    M = u_T.shape[0]
    diffs = []
    for m in range(M):
        res = solve_equilibrium_Enurho(robot, u_T[m], q0, E, nu, rho)
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
        G = robot.forward_kinematics(q, s_val)  # 4x4
        p = G[:3, 3]
        R = G[:3, :3]
        return p + R @ offset

    p1 = tip_at_s(0.1291, jnp.array([0.0, 0.0, 0.025]))
    p2 = tip_at_s(0.2195, jnp.array([0.0, 0.0, -0.021]))
    p3 = tip_at_s(0.2800, jnp.array([0.0, 0.0, 0.020]))
    p4 = tip_at_s(jnp.sum(robot.V_L), jnp.array([0.008, 0.0, 0.0]))
    return jnp.concatenate([p1, p2, p3, p4], axis=0)  # (12,)


# STATIC EQUILIBRIUM EQUATION SOLVER with UPDATED E,rho AND NU
def solve_equilibrium_Enurho(
    robot: TendonActuatedGVS,
    u: jnp.ndarray,
    q0: jnp.ndarray,
    E: jnp.ndarray,
    nu: jnp.ndarray,
    rho: jnp.ndarray,
):
    def statics_eq(q, args):
        u_m, E_new, nu_new, rho_new = args

        K = robot.stiffness_matrix()
        B = robot.actuation_matrix(q)
        G = robot.gravitational_force(q)

        E_old = E0
        nu_old = nu0
        rho_old = rho0
        G_old = E_old / (2 * (1 + nu_old))
        factor_E = E_new / E_old
        G_new = E_new / (2 * (1 + nu_new))
        factor_G = G_new / G_old
        factor_rho = rho_new / rho_old

        # THIS UPDATE STRICTLY DEPENDS ON THE CHOSEN DOF -> HAS TO BE ADAPTED IF THE ROBOT STRUCTURE CHANGES
        # 8 + 2 DOf
        # K = K.at[:2, :].multiply(factor_G)
        # K = K.at[2:, :].multiply(factor_E)
        # 7 + 2 DOf
        K = K.at[:1, :].multiply(factor_G)
        K = K.at[1:, :].multiply(factor_E)
        G = G * factor_rho

        return K @ q + G - B @ u_m

    solver = optx.Newton(rtol=1e-6, atol=1e-6)
    statics_eq_jit = jax.jit(statics_eq)
    return optx.root_find(statics_eq_jit, solver, q0, (u, E, nu, rho), max_steps=200)


def make_loss_fn(
    u_batch: jnp.ndarray, q0: jnp.ndarray, measured_markers_batch: jnp.ndarray
):
    """
    measured_markers: shape (12,) = [p1; p2; p3; p4]
    measured_markers_batch : shape (12, M)
    u_batch : shape (na, M)
    M = number of measurements in the batch
    """

    u_T = u_batch.T  # (M, na)
    measmark_T = measured_markers_batch.T  # (M, 12)

    def loss_fn(params):
        raw_E, raw_nu, raw_rho = params
        # vincoli: E>0, nu in (-0.5, 0.5) ~ materiali elastomerici ~ 0.45
        E_min = 1e4
        E_max = 1e6
        # E_log_min = jnp.log(1e4)
        # E_log_max = jnp.log(1e6)
        nu_min = 0.4
        nu_max = 0.5
        rho_min = 1000.0
        rho_max = 2000.0
        rho = 0.5 * (rho_min + rho_max) + 0.5 * (rho_max - rho_min) * jnp.tanh(
            raw_rho
        )  # (1000, 2000)
        E = 0.5 * (E_min + E_max) + 0.5 * (E_max - E_min) * jnp.tanh(
            raw_E
        )  # (1e4, 1e6)
        # E = jnp.exp(0.5 * (E_log_max + E_log_min) +0.5 * (E_log_max - E_log_min) * jnp.tanh(raw_E))
        nu = 0.5 * (nu_min + nu_max) + 0.5 * (nu_max - nu_min) * jnp.tanh(
            raw_nu
        )  # (0.3, 0.5)

        M = u_T.shape[0]
        err_sum = 0.0
        for m in range(M):
            res = solve_equilibrium_Enurho(robot, u_T[m], q0, E, nu, rho)
            q_star = res.value
            pred = markers_from_q(robot, q_star)  # (12,)
            diff = pred - measmark_T[m]
            err_sum = err_sum + jnp.dot(diff, diff)
        return 0.5 * err_sum / M
        # return err_sum

    return loss_fn

### BODY DEFINITION OF THE SOFT ROBOT ###

# 2 link version
# Link 1
link1 = LinkAttributes(
    cross_section_geometry=CrossSectionGeometry.CIRCULAR,
    E=5.05e5,
    nu=0.45,
    rho=1500.0,
    eta=1e4,
    L=0.0250 + 0.2550 + 0.0250,
    r_i=0.01541,
    r_f=0.00642,
)

# Link 2
link2 = LinkAttributes(
    cross_section_geometry=CrossSectionGeometry.CIRCULAR,
    E=5.05e5,
    nu=0.45,
    rho=1500.0,
    eta=1e4,
    L=0.0550,
    r_i=0.00642,
    r_f=0.00480,
)
joint1 = JointAttributes(jointtype="Fixed")
joint2 = JointAttributes(jointtype="Fixed")
basis1 = BasisAttributes(
    basistype="Monomial", Bdof=[1, 1, 1, 1, 0, 0], Bodr=[0, 1, 1, 1, 0, 0]
)
basis2 = BasisAttributes(
    basistype="Monomial", Bdof=[0, 1, 1, 0, 0, 0], Bodr=[0, 0, 0, 0, 0, 0]
)

n_gauss_list = [8, 8]
gravity_vector = [0.0, 0.0, -9.81]

tendon_routing_params = {
    "ry": jnp.array(
        [0.0114 * jnp.cos(jnp.pi / 180 * 30), 0.0114 * jnp.cos(jnp.pi / 180 * 150)]
    ),
    "my": jnp.array(
        [-0.0295 * jnp.cos(jnp.pi / 180 * 30), -0.0295 * jnp.cos(jnp.pi / 180 * 150)]
    ),
    "rz": jnp.array(
        [0.0114 * jnp.sin(jnp.pi / 180 * 30), 0.0114 * jnp.sin(jnp.pi / 180 * 150)]
    ),
    "mz": jnp.array(
        [-0.0295 * jnp.sin(jnp.pi / 180 * 30), -0.0295 * jnp.sin(jnp.pi / 180 * 150)]
    ),
    "idx_seg_att": jnp.array([0, 0]),
}
p0 = jnp.array([-jnp.pi / 2, jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0])

robot = TendonActuatedGVS(
    links_list=[link1, link2],
    joints_list=[joint1, joint2],
    basis_list=[basis1, basis2],
    n_gauss_list=n_gauss_list,
    gravity_vector=gravity_vector,
    tendon_routing_params=tendon_routing_params,
    p0=p0,
    scale_rotational_strain_basis=True,
)

# Initialize parameters
n_links = int(robot.num_segments)

# E and nu initial values (same for all links)
E0 = float(link1.E)
nu0 = float(link1.nu)
rho0 = float(link1.rho)
E_init = jnp.asarray(E0, dtype=jnp.float64)  # shape: ()
nu_init = jnp.asarray(nu0, dtype=jnp.float64)  # shape: ()
rho_init = jnp.asarray(rho0, dtype=jnp.float64)  # shape: ()


# E_log_min = jnp.log(1e4)
# E_log_max = jnp.log(1e6)
E_min = 1e4
E_max = 1e6

nu_min, nu_max = 0.4, 0.5
rho_min, rho_max = 1000.0, 2000.0


# helper to avoid going outside (-1,1)
def _atanh_clipped(x, eps=1e-6):
    return jnp.arctanh(jnp.clip(x, -1 + eps, 1 - eps))


# inverse E mapping
# mu_E = 0.5 * (E_log_max + E_log_min)
# r_E  = 0.5 * (E_log_max - E_log_min)
# x_E  = (jnp.log(E_init) - mu_E) / r_E
mu_E = 0.5 * (E_max + E_min)
r_E = 0.5 * (E_max - E_min)
x_E = (E_init - mu_E) / r_E
raw_E0 = _atanh_clipped(x_E)

# inverse nu mapping
mu_nu = 0.5 * (nu_min + nu_max)
r_nu = 0.5 * (nu_max - nu_min)
x_nu = (nu_init - mu_nu) / r_nu
raw_nu0 = _atanh_clipped(x_nu)

# inverse rho mapping
mu_rho = 0.5 * (rho_min + rho_max)
r_rho = 0.5 * (rho_max - rho_min)
x_rho = (rho_init - mu_rho) / r_rho
raw_rho0 = _atanh_clipped(x_rho)

params = (raw_E0, raw_nu0, raw_rho0)


# Prepare batch data (measured coordinates and inputs)
def _pack_markers(p1, p2, p3, p4):
    return jnp.concatenate([p1, p2, p3, p4], axis=0)  # (12,)


measured_list = [
    _pack_markers(p1_m1u01, p2_m1u01, p3_m1u01, p4_m1u01),
    _pack_markers(p1_m1u015, p2_m1u015, p3_m1u015, p4_m1u015),
    _pack_markers(p1_m1u02, p2_m1u02, p3_m1u02, p4_m1u02),
    _pack_markers(p1_m1u025, p2_m1u025, p3_m1u025, p4_m1u025),
    _pack_markers(p1_m2u01, p2_m2u01, p3_m2u01, p4_m2u01),
    _pack_markers(p1_m2u015, p2_m2u015, p3_m2u015, p4_m2u015),
    _pack_markers(p1_m2u02, p2_m2u02, p3_m2u02, p4_m2u02),
    _pack_markers(p1_m2u025, p2_m2u025, p3_m2u025, p4_m2u025),
]
measured_markers_batch = jnp.stack(measured_list, axis=1)
radius = 0.0325
levels = jnp.array([-0.1, -0.15, -0.2, -0.25], dtype=jnp.float64) / radius  # (4,)

u_m1 = jnp.stack(
    [jnp.array([lvl, 0.0], dtype=jnp.float64) for lvl in levels], axis=1
)  # (2,4)
u_m2 = jnp.stack(
    [jnp.array([0.0, lvl], dtype=jnp.float64) for lvl in levels], axis=1
)  # (2,4)
u_batch = jnp.concatenate([u_m1, u_m2], axis=1)  # shape (2, 8)

print("u_batch:", u_batch)
print("measured_markers_batch:", measured_markers_batch)

dof = sum(robot.V_dof.reshape(-1))
q0 = jnp.zeros((dof,), dtype=jnp.float64)


### GRADIENT-BASED OPTIMIZATION LOOP ###
loss_fn = make_loss_fn(
    u_batch=u_batch, q0=q0, measured_markers_batch=measured_markers_batch
)
loss_and_grad = eqx.filter_value_and_grad(loss_fn)

opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-2))
opt_state = opt.init(params)


# best checkpoint
best_loss = jnp.inf
best_params = params
best_step = -1
loss_history = []


for step in range(100):
    loss_val, grads = loss_and_grad(params)
    loss_history.append(float(loss_val))
    updates, opt_state = opt.update(grads, opt_state, params)

    print(f"[step {step:03d}] loss={float(loss_val):.6e}")

    raw_E_hat, raw_nu_hat, raw_rho_hat = params
    # E_hat = jnp.exp(mu_E + r_E * jnp.tanh(raw_E_hat))
    E_hat = mu_E + r_E * jnp.tanh(raw_E_hat)
    nu_hat = mu_nu + r_nu * jnp.tanh(raw_nu_hat)
    rho_hat = mu_rho + r_rho * jnp.tanh(raw_rho_hat)
    print("Current values:")
    print("E_hat:", E_hat)
    print("nu_hat:", nu_hat)
    print("rho_hat:", rho_hat)

    if float(loss_val) < float(best_loss):
        best_loss = loss_val
        best_params = params
        best_step = step

    params = optax.apply_updates(params, updates)


# Final parameter values
params = best_params
loss_val, grads = loss_and_grad(params)
raw_E_hat, raw_nu_hat, raw_rho_hat = params
# E_hat = jnp.exp(mu_E + r_E * jnp.tanh(raw_E_hat))
E_hat = mu_E + r_E * jnp.tanh(raw_E_hat)
nu_hat = mu_nu + r_nu * jnp.tanh(raw_nu_hat)
rho_hat = mu_rho + r_rho * jnp.tanh(raw_rho_hat)
print("Final estimated parameters:")
print("E_hat:", E_hat)
print("nu_hat:", nu_hat)
print("rho_hat:", rho_hat)
print("Final loss:", best_loss)
print("best_step:", best_step)


# onp.save("E_hat.npy", onp.asarray(E_hat))
# onp.save("nu_hat.npy", onp.asarray(nu_hat))
# onp.save("rho_hat.npy", onp.asarray(rho_hat))


# ### PLOTTING RESULTS ###

errors_before = compute_marker_errors(
    robot, u_batch, q0, measured_markers_batch, E0, nu0, rho0
)
errors_after = compute_marker_errors(
    robot, u_batch, q0, measured_markers_batch, E_hat, nu_hat, rho_hat
)


def marker_rmse(errors):
    # errors shape: (M, 12)
    errors_reshaped = errors.reshape(errors.shape[0], 4, 3)  # (M, 4, 3)
    rms_per_marker = jnp.sqrt(jnp.mean(jnp.sum(errors_reshaped**2, axis=2), axis=0))
    return rms_per_marker  # (4,)


rmse_before = marker_rmse(errors_before)
rmse_after = marker_rmse(errors_after)

marker_names = [f"marker {i + 1}" for i in range(4)]
x = onp.arange(len(marker_names))
width = 0.35

plt.figure(figsize=(8, 5))
plt.plot(loss_history, "o-", color="blue", label="Loss")
plt.xlabel("Optimization step")
plt.ylabel("Loss")
plt.title("Loss evolution during optimization")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
# plt.savefig("loss_over_time.svg", format="svg", bbox_inches="tight")
# plt.savefig("loss_over_time.pdf", format="pdf", bbox_inches="tight")
# plt.savefig("loss_over_time.jpg", format="jpg", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, rmse_before, width, label="Before optimization", alpha=0.7)
plt.bar(x + width / 2, rmse_after, width, label="After optimization", alpha=0.7)
plt.xticks(x, marker_names)
plt.ylabel("RMSE [m]")
plt.title("Marker position errors before vs after optimization")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
# plt.savefig("rmse_per_marker.svg", format="svg", bbox_inches="tight")
# plt.savefig("rmse_per_marker.pdf", format="pdf", bbox_inches="tight")
# plt.savefig("rmse_per_marker.jpg", format="jpg", dpi=300, bbox_inches="tight")
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(jnp.sqrt(jnp.sum(errors_before**2, axis=1)), "o-", label="Before")
plt.plot(jnp.sqrt(jnp.sum(errors_after**2, axis=1)), "s-", label="After")
plt.xlabel("Sample index")
plt.ylabel("Total position error norm [m]")
plt.title("Total marker error per configuration")
plt.legend()
plt.grid(True)
# plt.savefig("error_per_configuration.svg", format="svg", bbox_inches="tight")
# plt.savefig("error_per_configuration.pdf", format="pdf", bbox_inches="tight")
# plt.savefig("error_per_configuration.jpg", format="jpg", dpi=300, bbox_inches="tight")
plt.show()
