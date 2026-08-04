import argparse
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import optax
import optimistix as optx
import pandas as pd

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.systems import (
    GVS,
    GVSSegment,
    JointSpec,
    LinearProfile,
    LinkSpec,
    StrainBasisSpec,
)

jax.config.update("jax_enable_x64", True)

CASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = CASE_DIR / "data"
DEFAULT_OUTPUT_DATA_DIR = DEFAULT_DATA_DIR / "parameter_identification"
DATA_DIR = DEFAULT_DATA_DIR
OUTPUT_DATA_DIR = DEFAULT_OUTPUT_DATA_DIR
GENERATED_FILENAMES = (
    "E_hat.npy",
    "nu_hat.npy",
    "rho_hat.npy",
    "loss_history.npy",
    "rmse_before.npy",
    "errors_before.npy",
    "rmse_after.npy",
    "errors_after.npy",
    "curves_orig.npy",
    "curves_hat.npy",
    "markers_orig.npy",
    "markers_hat.npy",
    "measured_pts.npy",
)

# jax.config.update("jax_platform_name", "gpu")  # or "cpu"

print("JAX default backend:", jax.default_backend())
print("JAX devices:", jax.devices())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify soft-tentacle parameters from the Section Va data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the committed motion-capture CSV inputs.",
    )
    parser.add_argument(
        "--output-data-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DATA_DIR,
        help="Directory for generated NumPy result arrays.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=int(os.environ.get("SOROMOX_SYSID_STEPS", "100")),
        help="Number of parameter-optimization steps.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Close diagnostic figures instead of opening windows.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace generated arrays that already exist.",
    )
    return parser.parse_args()


def finish_figure(*, show: bool) -> None:
    if show:
        plt.show()
    else:
        plt.close()


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


def _load_marker_data():
    ### EXCEL DATA READING ###
    Rot = jnp.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float64
    )
    R_alg1 = jnp.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=jnp.float64
    )
    R_alg2 = jnp.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float64
    )
    offset = jnp.array([-0.03, -0.013, 0.0], dtype=jnp.float64)

    df = pd.read_csv(DATA_DIR / "m1_u01.csv", header=None)
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

    df = pd.read_csv(DATA_DIR / "m1_u015.csv", header=None)
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

    df = pd.read_csv(DATA_DIR / "m1_u02.csv", header=None)
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

    df = pd.read_csv(DATA_DIR / "m1_u025.csv", header=None)
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

    df = pd.read_csv(DATA_DIR / "m2_u01.csv", header=None)
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

    df = pd.read_csv(DATA_DIR / "m2_u015.csv", header=None)
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

    df = pd.read_csv(DATA_DIR / "m2_u02.csv", header=None)
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

    df = pd.read_csv(DATA_DIR / "m2_u025.csv", header=None)
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

    return locals()


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


def draw_robot_curve(
    robot: GVS,
    q: jnp.ndarray,
    num_points: int = 50,
):
    batched_forward_kinematics = jax.vmap(robot.forward_kinematics, in_axes=(None, 0))
    L_max = robot.length

    s_ps = jnp.linspace(0, L_max, num_points)
    g_ps = batched_forward_kinematics(q, s_ps)[:, :3, 3]

    # q_gathered = robot._min_size_gathered(q)
    # V_g = robot._forward_kinematics_gauss(q_gathered)

    # g_ps = jnp.concatenate(V_g[:, :, :-1, -1:], axis=0)

    curve = onp.array(g_ps, dtype=onp.float64)
    return curve  # (N, 3)


# ---- Marker prediction for a fixed q  ----
def markers_from_q(robot: GVS, q: jnp.ndarray) -> jnp.ndarray:
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
    p4 = tip_at_s(jnp.sum(robot.length), jnp.array([0.008, 0.0, 0.0]))
    return jnp.concatenate([p1, p2, p3, p4], axis=0)  # (12,)


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


# STATIC EQUILIBRIUM EQUATION SOLVER with UPDATED E,rho AND NU
def solve_equilibrium_Enurho(
    robot: GVS,
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


if __name__ == "__main__":
    args = parse_args()
    DATA_DIR = args.data_dir.resolve()
    OUTPUT_DATA_DIR = args.output_data_dir.resolve()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    existing_outputs = [
        OUTPUT_DATA_DIR / filename
        for filename in GENERATED_FILENAMES
        if (OUTPUT_DATA_DIR / filename).exists()
    ]
    if existing_outputs and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite {existing_outputs[0]}; pass --force"
        )
    globals().update(_load_marker_data())

    ### BODY DEFINITION OF THE SOFT ROBOT ###

    # 2 link version
    # Link 1
    link1 = LinkSpec.circular(
        length=0.0250 + 0.2550 + 0.0250,
        radius=LinearProfile(base=0.01541, tip=0.00642),
        density=1500.0,
        young_modulus=5.05e5,
        shear_modulus=5.05e5 / (2.0 * (1.0 + 0.45)),
        material_damping_coefficient=1e4,
        reference_strain=[0, 0, 0, 1, 0, 0],
    )

    # Link 2
    link2 = LinkSpec.circular(
        length=0.0550,
        radius=LinearProfile(base=0.00642, tip=0.00480),
        density=1500.0,
        young_modulus=5.05e5,
        shear_modulus=5.05e5 / (2.0 * (1.0 + 0.45)),
        material_damping_coefficient=1e4,
        reference_strain=[0, 0, 0, 1, 0, 0],
    )
    joint1 = JointSpec(type="fixed")
    joint2 = JointSpec(type="fixed")
    basis1 = StrainBasisSpec(
        type="monomial",
        strain_selector=[1, 1, 1, 1, 0, 0],
        basis_order=[0, 1, 1, 1, 0, 0],
    )
    basis2 = StrainBasisSpec(
        type="monomial",
        strain_selector=[0, 1, 1, 0, 0, 0],
        basis_order=[0, 0, 0, 0, 0, 0],
    )

    num_gauss_points = [8, 8]
    g = [0.0, 0.0, -9.81]

    tendon_angles = jnp.deg2rad(jnp.array([30.0, 150.0]))
    active_tendon_routing = ThreadlikeRouting.linear(
        intercept=0.0114
        * jnp.stack(
            (
                jnp.zeros_like(tendon_angles),
                jnp.cos(tendon_angles),
                jnp.sin(tendon_angles),
            ),
            axis=-1,
        ),
        slope=-0.0295
        * jnp.stack(
            (
                jnp.zeros_like(tendon_angles),
                jnp.cos(tendon_angles),
                jnp.sin(tendon_angles),
            ),
            axis=-1,
        ),
        start_segment_index=0,
        end_segment_index=(0, 0),
    )
    p0 = jnp.array([jnp.sqrt(0.5), 0.0, jnp.sqrt(0.5), 0.0, 0.0, 0.0, 0.0])

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
        scale_rotational_basis_by_length=True,
    )

    # Initialize parameters
    n_links = int(robot.num_segments)

    # E and nu initial values (same for all links)
    E0 = float(link1.young_modulus)
    nu0 = 0.45
    rho0 = float(link1.density)
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
    marker_data = _load_marker_data()

    def _pack_markers(p1, p2, p3, p4):
        return jnp.concatenate([p1, p2, p3, p4], axis=0)  # (12,)

    marker_samples = (
        "m1u01",
        "m1u015",
        "m1u02",
        "m1u025",
        "m2u01",
        "m2u015",
        "m2u02",
        "m2u025",
    )
    measured_list = [
        _pack_markers(
            marker_data[f"p1_{sample}"],
            marker_data[f"p2_{sample}"],
            marker_data[f"p3_{sample}"],
            marker_data[f"p4_{sample}"],
        )
        for sample in marker_samples
    ]
    measured_markers_batch = jnp.stack(measured_list, axis=1)
    radius = 0.0325
    # The tendon preset embeds y_a = -length, so positive controls are tensions.
    levels = jnp.array([0.1, 0.15, 0.2, 0.25], dtype=jnp.float64) / radius  # (4,)

    u_m1 = jnp.stack(
        [jnp.array([lvl, 0.0], dtype=jnp.float64) for lvl in levels], axis=1
    )  # (2,4)
    u_m2 = jnp.stack(
        [jnp.array([0.0, lvl], dtype=jnp.float64) for lvl in levels], axis=1
    )  # (2,4)
    u_batch = jnp.concatenate([u_m1, u_m2], axis=1)  # shape (2, 8)

    print("u_batch:", u_batch)
    print("measured_markers_batch:", measured_markers_batch)

    dof = sum(robot.dofs_per_segment.reshape(-1))
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

    num_optimization_steps = args.steps
    for step in range(num_optimization_steps):
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

    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    onp.save(OUTPUT_DATA_DIR / "E_hat.npy", onp.asarray(E_hat))
    onp.save(OUTPUT_DATA_DIR / "nu_hat.npy", onp.asarray(nu_hat))
    onp.save(OUTPUT_DATA_DIR / "rho_hat.npy", onp.asarray(rho_hat))
    onp.save(OUTPUT_DATA_DIR / "loss_history.npy", onp.asarray(loss_history))

    # load results
    # E_hat = jnp.array(onp.load("data/E_hat.npy"))
    # nu_hat = jnp.array(onp.load("data/nu_hat.npy"))
    # rho_hat = jnp.array(onp.load("data/rho_hat.npy"))
    # loss_history = onp.load("data/loss_history.npy").tolist()

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
    onp.save(OUTPUT_DATA_DIR / "rmse_before.npy", onp.asarray(rmse_before))
    onp.save(OUTPUT_DATA_DIR / "errors_before.npy", onp.asarray(errors_before))
    onp.save(OUTPUT_DATA_DIR / "rmse_after.npy", onp.asarray(rmse_after))
    onp.save(OUTPUT_DATA_DIR / "errors_after.npy", onp.asarray(errors_after))

    marker_names = [f"M{i + 1}" for i in range(4)]
    x = onp.arange(len(marker_names))
    width = 0.35

    plt.figure(figsize=(12, 9))
    plt.plot(loss_history, "o-", color="blue", label="Loss")
    plt.xlabel("Optimization step", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    # plt.title("Loss evolution during optimization")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=16)
    plt.tick_params(axis="both", labelsize=16)
    plt.tight_layout()
    # plt.savefig("loss_over_time.svg", format="svg", bbox_inches="tight")
    # plt.savefig("loss_over_time.pdf", format="pdf", bbox_inches="tight")
    # plt.savefig("loss_over_time.jpg", format="jpg", dpi=300, bbox_inches="tight")
    finish_figure(show=not args.no_show)

    plt.figure(figsize=(12, 9))
    plt.bar(x - width / 2, rmse_before, width, label="Initial param.", alpha=0.8)
    plt.bar(x + width / 2, rmse_after, width, label="Optimized param.", alpha=0.8)
    plt.xticks(x, marker_names, fontsize=16)
    plt.ylabel("RMS Position Error [m]", fontsize=16)
    # plt.title("Marker position errors before vs after optimization")
    plt.legend(fontsize=16)
    plt.tick_params(axis="both", labelsize=16)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    # plt.savefig("rmse_per_marker.svg", format="svg", bbox_inches="tight")
    # plt.savefig("rmse_per_marker.pdf", format="pdf", bbox_inches="tight")
    # plt.savefig("rmse_per_marker.jpg", format="jpg", dpi=300, bbox_inches="tight")
    finish_figure(show=not args.no_show)

    plt.figure(figsize=(12, 9))
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
    finish_figure(show=not args.no_show)

    # Markers comparison visualization
    link1_hat = LinkSpec.circular(
        length=0.0250 + 0.2550 + 0.0250,
        radius=LinearProfile(base=0.01541, tip=0.00642),
        density=float(rho_hat),
        young_modulus=float(E_hat),
        shear_modulus=float(E_hat / (2.0 * (1.0 + nu_hat))),
        material_damping_coefficient=1e4,
        reference_strain=[0, 0, 0, 1, 0, 0],
    )
    link2_hat = LinkSpec.circular(
        length=0.0550,
        radius=LinearProfile(base=0.00642, tip=0.00480),
        density=float(rho_hat),
        young_modulus=float(E_hat),
        shear_modulus=float(E_hat / (2.0 * (1.0 + nu_hat))),
        material_damping_coefficient=1e4,
        reference_strain=[0, 0, 0, 1, 0, 0],
    )
    robot_hat = GVS.from_segments(
        [
            GVSSegment(
                link=link1_hat,
                joint=joint1,
                basis=basis1,
                num_gauss_points=num_gauss_points[0],
            ),
            GVSSegment(
                link=link2_hat,
                joint=joint2,
                basis=basis2,
                num_gauss_points=num_gauss_points[1],
            ),
        ],
        gravity=jnp.asarray(g),
        base_pose=p0,
        actuators=ThreadlikeActuator.tendons(active_tendon_routing),
        scale_rotational_basis_by_length=True,
    )

    # For each input in u_batch, solve statics and plot original vs optimized shapes
    M = int(u_batch.shape[1])

    # Precompute solutions, curves and markers for all samples
    curves_orig = []
    curves_hat = []
    markers_orig = []
    markers_hat = []
    measured_pts = []

    for m in range(M):
        # solve static equilibrium for this input
        res0 = solve_equilibrium(robot, u_batch[:, m], q0)
        res_hat = solve_equilibrium(robot_hat, u_batch[:, m], q0)

        q_stat0_m = res0.value
        q_statHat_m = res_hat.value

        # sample backbone curves (use a finer resolution for nicer plots)
        curve0 = draw_robot_curve(robot, q_stat0_m, num_points=200)
        curve_hat = draw_robot_curve(robot_hat, q_statHat_m, num_points=200)

        curves_orig.append(onp.asarray(curve0))
        curves_hat.append(onp.asarray(curve_hat))

        # predicted marker positions
        markers0 = onp.asarray(markers_from_q(robot, q_stat0_m).reshape(4, 3))
        markersH = onp.asarray(markers_from_q(robot_hat, q_statHat_m).reshape(4, 3))
        markers_orig.append(markers0)
        markers_hat.append(markersH)

        # measured markers from your dataset: measured_markers_batch shape (12, M)
        meas = onp.asarray(measured_markers_batch[:, m].reshape(4, 3))
        measured_pts.append(meas)

    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    onp.save(OUTPUT_DATA_DIR / "curves_orig.npy", onp.asarray(curves_orig))
    onp.save(OUTPUT_DATA_DIR / "curves_hat.npy", onp.asarray(curves_hat))
    onp.save(OUTPUT_DATA_DIR / "markers_orig.npy", onp.asarray(markers_orig))
    onp.save(OUTPUT_DATA_DIR / "markers_hat.npy", onp.asarray(markers_hat))
    onp.save(OUTPUT_DATA_DIR / "measured_pts.npy", onp.asarray(measured_pts))

    # compute plot bounds from all points
    all_pts = []
    for c0, ch, m0, mh, mm in zip(
        curves_orig, curves_hat, markers_orig, markers_hat, measured_pts
    ):
        all_pts.append(c0)
        all_pts.append(ch)
        all_pts.append(m0)
        all_pts.append(mh)
        all_pts.append(mm)
    all_pts = onp.vstack(all_pts)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    ranges = maxs - mins
    max_range = ranges.max()
    center = 0.5 * (maxs + mins)

    fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={"projection": "3d"})

    # choose colors for each sample but avoid pure red/blue for the backbone shapes
    tab10 = plt.get_cmap("tab10")
    allowed_indices = [1, 2, 4, 5, 6, 7, 8, 9]  # skip 0 (blue) and 3 (red)
    colors = [tab10(allowed_indices[i % len(allowed_indices)]) for i in range(M)]

    alpha_orig = 0.25  # transparency for original (less prominent)
    alpha_hat = 1.0  # optimized more opaque

    for m in range(M):
        c = colors[m]
        c_rgb = (c[0], c[1], c[2])

        curve0 = curves_orig[m]
        curve_hat = curves_hat[m]

        # plot backbones with same color but different alpha
        ax.plot(
            curve0[:, 0],
            curve0[:, 1],
            curve0[:, 2],
            lw=3,
            color=c_rgb,
            alpha=alpha_orig,
        )
        ax.plot(
            curve_hat[:, 0],
            curve_hat[:, 1],
            curve_hat[:, 2],
            lw=3,
            color=c_rgb,
            alpha=alpha_hat,
        )

        # predicted markers: use orange for original, red for optimized
        m0 = markers_orig[m]
        mh = markers_hat[m]
        # measured markers
        mm = measured_pts[m]

        # per-marker alpha gradient from marker1 -> marker4 (0.25 -> 1.0)
        grad_alphas = onp.linspace(0.25, 1.0, 4)
        red_col = (0.8, 0.1, 0.1)
        orange_col = (1.0, 0.55, 0.0)
        blue_col = (0.0, 0.2, 0.8)

        for i in range(4):
            a = float(grad_alphas[i])
            # original predicted marker (orange), scaled by original alpha factor
            ax.scatter(
                m0[i, 0],
                m0[i, 1],
                m0[i, 2],
                marker="x",
                color=orange_col,
                s=50,
                alpha=a,
            )
            # optimized predicted marker (red)
            ax.scatter(
                mh[i, 0], mh[i, 1], mh[i, 2], marker="x", color=red_col, s=90, alpha=a
            )
            # measured marker (blue circle) with gradient alpha
            ax.scatter(
                mm[i, 0],
                mm[i, 1],
                mm[i, 2],
                marker="o",
                facecolors="none",
                edgecolors=blue_col,
                s=90,
                alpha=a,
            )

    # Set axis labels and title (larger font)
    ax.set_xlabel("X [m]", fontsize=14)
    ax.set_ylabel("Y [m]", fontsize=14)
    ax.set_zlabel("Z [m]", fontsize=14)
    ax.view_init(elev=20, azim=15)
    # ax.set_title("Shape comparison across inputs", fontsize=14)

    # enforce equal aspect ratio for 3D plot
    ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
    ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
    ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)
    ax.set_box_aspect([1, 1, 1])

    # create a custom legend: show division Original vs Optimized plus markers
    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D(
            [0],
            [0],
            color="k",
            lw=2.5,
            linestyle="-",
            alpha=alpha_orig,
            label="Original Backbone",
        ),
        Line2D(
            [0],
            [0],
            color="k",
            lw=3.0,
            linestyle="-",
            alpha=alpha_hat,
            label="Optimized Backbone",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="w",
            markerfacecolor="orange",
            markeredgecolor="orange",
            markersize=8,
            label="Predicted markers (Original)",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="w",
            markerfacecolor="red",
            markeredgecolor="red",
            markersize=8,
            label="Predicted markers (Optimized)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markeredgecolor="blue",
            markerfacecolor="none",
            markersize=8,
            label="Measured markers",
        ),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=14)

    plt.tight_layout()
    plt.tick_params(axis="both", labelsize=12)

    # plt.savefig("MARKERS_CONFIG_ss.pdf", format="pdf", bbox_inches="tight")
    # plt.savefig("MARKERS_CONFIG_ss.jpg", format="jpg", dpi=300, bbox_inches="tight")
    finish_figure(show=not args.no_show)
