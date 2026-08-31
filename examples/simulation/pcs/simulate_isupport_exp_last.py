"""AM-I Support: soromox rollout of the exp_veronica pre-impact TAIL fit (W6).

Sibling of `simulate_isupport_exp.py`, restricted to ONE run: the
`W6_veronica_tail_t11` MATLAB fit -- the pre-impact tail (t=[11, 13.06] s) of
the longest exp_veronica bag (`Eco50_3bar_default_intermadiate_170006`), perm
[2 1 3] / roll -7 deg / clock 0, Eta+gains refit on the tail. Every parameter
is read from the `.mat` exported by `python_exporter_veronica_last.m` -- no
model literals here either.

x0 is TRUE REST here (all-zero, verified: MATLAB checked tip position holds to
<0.6 mm over the first 0.19 s of the window, well before the actuation step at
~11.66 s) -- same convention as the square/chirp export. An earlier version of
this run (W5) warm-started x0 from the model's OWN simulated state at t=12 s
instead of moving the window back to a genuine rest point; that was flagged on
review as not an independent measurement and dropped. The export still carries
`x0` and this script still splits it into q0/qd0 for structural symmetry with
that contract, but it is a no-op here (q0 = qd0 = 0).

Usage (inside the WSL venv, from examples/simulation/pcs):

    python simulate_isupport_exp_last.py
    python simulate_isupport_exp_last.py --ladder   # rung 0 only, no rollout

Known model limitation, expected to reproduce identically on both sides: the
physical robot has a resonance at 1.408 Hz (Q ~ 4.3) that the model does not
have. Both simulations therefore miss the ring-up envelope and the 5th
harmonic in the same way. That is not a port defect -- only a DIFFERENCE
between the two simulations is.

System parameters ultimately trace to:
Alessi, Carlo, Egidio Falotico, and Alessandro Lucantonio.
"Ablation study of a dynamic model for a 3d-printed pneumatic soft robotic arm."
IEEE Access 11 (2023): 37840-37853.
"""

from __future__ import annotations

import argparse
import os
from functools import partial

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.io as sio
from diffrax import Kvaerno5, PIDController, Tsit5

jax.config.update("jax_enable_x64", True)  # double precision

from soromox.systems import ISupport, ISupportParams, ISupportStructure, SystemState

from amisupport_metrics import parity_stats, print_parity, print_tip_metrics, tip_metrics

# ---------------------------------------------------------------------------
# Paths. WSL sees the MATLAB repo directly -- nothing is copied.
# ---------------------------------------------------------------------------
MATLAB_REPO = "/mnt/c/Users/danie/OneDrive/Documents/GitHub/am_isupportGVS"
EXPORT_DIR = os.path.join(MATLAB_REPO, "dataset", "python_exported")
LOCAL_DIR = "amisupport_dataset"
RESULTS_DIR = os.path.join(LOCAL_DIR, "results")

BAGS = {
    "last": "Eco50_3bar_default_intermadiate_170006",
}

# python_exporter_veronica_last.m writes
# "Eco50_3bar_default_intermadiate_170006_py_last.mat", i.e. tag "_last" under
# the existing {bagname}_py{tag}.mat convention -- no new loading path needed.
DEFAULT_TAG = "_last"

# Acceptance for simulation-vs-simulation agreement.
PARITY_THRESHOLD_MM = 1.0


# ---------------------------------------------------------------------------
# Export loading
# ---------------------------------------------------------------------------
def load_export(bag_key: str, tag: str = ""):
    """Load the exported .mat, preferring the live MATLAB repo over the local copy.

    `tag` selects a geometry variant written by `python_exporter.m` with
    `AMI_EXPORT_TAG` (e.g. "_g35s178"); empty loads the baseline export.
    """
    bagname = BAGS[bag_key]
    fname = f"{bagname}_py{tag}.mat"
    for d in (EXPORT_DIR, LOCAL_DIR):
        path = os.path.join(d, fname)
        if os.path.isfile(path):
            data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
            print(f"Loaded export: {path}")
            return data, bagname
    raise FileNotFoundError(
        f"{fname} not found in {EXPORT_DIR} or {LOCAL_DIR}. Regenerate with "
        f'`matlab -sd {MATLAB_REPO} -batch "python_exporter"`.'
    )


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """SO(3) -> scalar-first quaternion [w, x, y, z] (Shepperd's method)."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array(
            [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        )
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array(
            [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        )
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array(
            [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
        )
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# Layout: MATLAB SoRoSim link chain -> soromox PCS segment layout
# ---------------------------------------------------------------------------
def build_layout(geom):
    """Collapse the MATLAB link chain into soromox's alternating rigid/soft layout.

    MATLAB builds `r s r r s r`: two 6 mm short interfaces bracketing each
    180 mm soft module, plus a 21 mm long interface between the modules. Two
    consecutive rigid links (the 6 mm + 21 mm pair in the middle) are one rigid
    body, so they merge into a single 27 mm PCS segment.

    The `g_ini` translation (36 mm along the base-frame x axis) is folded into
    the FIRST rigid segment instead of being carried as a base position. This is
    exact, not an approximation: `g_ini`'s rotation is a roll about x, so it
    leaves the x axis invariant and the 36 mm offset is collinear with the first
    link. The extra mass this gives the base segment has no dynamic effect --
    the segment is proximal to every strain DOF, so nothing can move it.
    """
    link_type = str(geom.link_type)
    L = np.atleast_1d(np.asarray(geom.link_length, dtype=float))
    radius = np.atleast_1d(np.asarray(geom.link_radius, dtype=float))
    rho = np.atleast_1d(np.asarray(geom.link_rho, dtype=float))
    E_link = np.atleast_1d(np.asarray(geom.link_E, dtype=float))
    eta_link = np.atleast_1d(np.asarray(geom.link_Eta, dtype=float))
    poi_link = np.atleast_1d(np.asarray(geom.link_Poi, dtype=float))

    segs: list[dict] = []
    for i, kind in enumerate(link_type):
        is_rigid = kind == "r"
        if segs and is_rigid and segs[-1]["rigid"]:
            prev = segs[-1]
            # Merged rigid bodies must be geometrically identical, otherwise a
            # single PCS segment cannot represent them.
            assert np.isclose(prev["radius"], radius[i]), "merged rigid radii differ"
            assert np.isclose(prev["rho"], rho[i]), "merged rigid densities differ"
            prev["L"] += L[i]
            continue
        segs.append(
            {
                "rigid": is_rigid,
                "L": float(L[i]),
                "radius": float(radius[i]),
                "rho": float(rho[i]),
                "E": float(E_link[i]),
                "eta": float(eta_link[i]),
                "poi": float(poi_link[i]),
            }
        )

    g_ini = np.asarray(geom.g_ini, dtype=float)
    x_offset = float(g_ini[0, 3])
    assert segs[0]["rigid"], "expected the chain to start with a rigid interface"
    assert np.allclose(g_ini[1:3, 3], 0.0), (
        "g_ini has an off-axis translation; it cannot be folded into segment 0"
    )
    segs[0]["L"] += x_offset

    # Material constants: the rigid segments carry no DOF, so their E/Eta never
    # enter the dynamics. Fill them from the soft link so the arrays are finite.
    soft = [s for s in segs if not s["rigid"]]
    assert soft, "no soft segment found in the link chain"
    E_soft = soft[0]["E"]
    eta_soft = soft[0]["eta"]
    poi_soft = soft[0]["poi"]
    for s in segs:
        if not np.isfinite(s["E"]):
            s["E"] = E_soft
        if not np.isfinite(s["eta"]):
            s["eta"] = eta_soft
        if not np.isfinite(s["poi"]):
            s["poi"] = poi_soft

    total = sum(s["L"] for s in segs)
    assert np.isclose(total, float(geom.s_tip), atol=1e-9), (
        f"layout length {total:.6f} != MATLAB s_tip {float(geom.s_tip):.6f}"
    )
    return segs, E_soft, eta_soft, poi_soft


def snap_triad(angles_deg: np.ndarray, tol_deg: float = 0.01):
    """Snap a nominally uniform chamber triad onto an exactly uniform one.

    The MATLAB `Linkage.dc` triad is uniform only to about 7e-4 deg (and its
    radii to 0.4 um) -- a construction artefact baked into `robot_linkage.mat`,
    not round-off from the clock rotation. soromox validates uniformity to
    1e-8 rad and refuses anything looser, so the triad is replaced by the exact
    one at its circular-mean clock. The deviation is asserted, not ignored: at
    7e-4 deg on a 20 mm moment arm the resulting actuation error is ~2e-5
    relative, i.e. microns on a 300 mm tip swing.

    Returns (snapped angles [deg], per-segment clock [deg], max deviation [deg]).
    """
    angles_deg = np.asarray(angles_deg, dtype=float)
    n_seg, n_ch = angles_deg.shape
    nominal = 360.0 * np.arange(n_ch) / n_ch
    # Circular mean of the per-chamber clock offsets, so no chamber is favoured.
    resid = np.deg2rad(angles_deg - nominal[None, :])
    clock = np.rad2deg(
        np.arctan2(np.sin(resid).mean(axis=1), np.cos(resid).mean(axis=1))
    )
    snapped = np.mod(clock[:, None] + nominal[None, :], 360.0)
    dev = np.abs(((angles_deg - snapped + 180.0) % 360.0) - 180.0)
    max_dev = float(dev.max())
    assert max_dev < tol_deg, (
        f"chamber triad is not uniform to {tol_deg} deg (max deviation "
        f"{max_dev:.6f} deg) -- refusing to snap"
    )
    return snapped, clock, max_dev


def build_robot(data):
    """Construct the soromox ISupport model from the exported geometry block."""
    geom = data["geom"]
    segs, E_soft, eta_soft, poi_soft = build_layout(geom)

    rigid_selector = tuple(bool(s["rigid"]) for s in segs)
    lengths = jnp.asarray([s["L"] for s in segs])
    radii = jnp.asarray([s["radius"] for s in segs])
    densities = jnp.asarray([s["rho"] for s in segs])
    n_seg = len(segs)
    n_pneu = sum(1 for s in segs if not s["rigid"])

    E = jnp.asarray([s["E"] for s in segs])
    poisson = float(poi_soft)
    G = E / (2.0 * (1.0 + poisson))
    damping = jnp.asarray([s["eta"] for s in segs])

    # Base pose: rotation from g_ini (the x translation is already folded into
    # segment 0 by build_layout).
    g_ini = np.asarray(geom.g_ini, dtype=float)
    quat = rotation_to_quaternion(g_ini[:3, :3])
    base_pose = jnp.asarray(np.concatenate([quat, np.zeros(3)]))

    # Gravity: MATLAB stores the screw [omega; v]; the linear part is what acts.
    G_screw = np.atleast_1d(np.asarray(geom.gravity, dtype=float))
    gravity = jnp.asarray(G_screw[3:6] if G_screw.size == 6 else G_screw[:3])

    # Chamber triad. MATLAB reports atan2(z, y) of Linkage.dc, which is exactly
    # soromox's convention (right-handed azimuth about local +x, from +y toward
    # +z). Whatever chamber clock the MATLAB run used is already baked into
    # these angles: the T1_fc28 / U0_ctrl configuration clocks the triad +15 deg
    # and reports 105/225/345, while the roll-only U1 configuration leaves it at
    # the nominal 90/210/330 and carries the geometry fix in the g_ini roll
    # instead (-13 deg rather than -30). Nothing here needs to know which.
    cham_deg_raw = np.atleast_1d(np.asarray(geom.chamber_angle_deg, dtype=float))
    n_ch = cham_deg_raw.size // n_pneu
    cham_deg, clock_deg, azi_dev = snap_triad(cham_deg_raw.reshape(n_pneu, n_ch))
    azimuths = jnp.asarray(np.deg2rad(cham_deg))

    cham_r_raw = np.atleast_1d(np.asarray(geom.chamber_radius, dtype=float)).reshape(
        n_pneu, n_ch
    )
    # soromox carries ONE chamber distance per pneumatic segment, so per-chamber
    # radii cannot be represented; the spread is sub-micron (see snap_triad).
    rad_dev = float(np.max(np.abs(cham_r_raw - cham_r_raw.mean(axis=1, keepdims=True))))
    assert rad_dev < 1e-5, f"chamber radii differ by {1e3 * rad_dev:.4f} mm within a segment"
    chamber_distance = jnp.asarray(cham_r_raw.mean(axis=1))

    r_in = float(data["r_in"])
    r_out = float(data["r_out"])
    chamber_area = float(data["chamber_area"])

    params = ISupportParams(
        base_pose=base_pose,
        length=lengths,
        radius=radii,
        density=densities,
        gravity=gravity,
        young_modulus=E,
        shear_modulus=G,
        material_damping_coefficient=damping,
        reference_strain=jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), n_seg),
        chamber_inner_radius=r_in * jnp.ones((n_pneu,)),
        chamber_outer_radius=r_out * jnp.ones((n_pneu,)),
        chamber_distance=chamber_distance,
        chamber_azimuth_angles=azimuths,
        pcs_segment_lengths=None,
        chamber_effective_pressure_area=chamber_area * jnp.ones((n_pneu,)),
    )

    robot = ISupport(
        params=params,
        structure=ISupportStructure(
            pcs_segment_counts=(1,) * n_pneu,
            rigid_segment_selector=rigid_selector,
            scale_rotational_basis_by_length=True,
        ),
    )

    info = {
        "segments": segs,
        "n_pneu": n_pneu,
        "E": E_soft,
        "Eta": eta_soft,
        "Poi": poisson,
        "azimuth_deg": cham_deg.ravel(),
        "azimuth_deg_raw": cham_deg_raw,
        "azimuth_snap_dev_deg": azi_dev,
        "chamber_clock_deg": clock_deg,
        "s_tip": float(geom.s_tip),
        "s_mid": float(geom.s_mid),
    }
    return robot, info


# ---------------------------------------------------------------------------
# Actuation: recorded pressures -> filtered chamber pressures
# ---------------------------------------------------------------------------
class RecordedInput(eqx.Module):
    """Recorded pressure command with a first-order lag on the PRESSURE.

    The lag sits UPSTREAM of the force map, matching `simulate_robot_lp.m`:
    `tau * d(u_filt)/dt = u_raw - u_filt`, and the force is `A_c * u_filt`.
    The filtered pressure is carried as `control_state`, so diffrax integrates
    the filter together with the mechanics.

    `act_values` is already the fully scaled pressure in pascals -- permutation,
    air-leak factors, per-actuator gains and bar->Pa are applied once, up front,
    because every one of them is a constant linear factor and therefore commutes
    with the filter.
    """

    act_time: jax.Array  # (Mw,)
    act_values: jax.Array  # (nact, Mw), [Pa], model actuator order, gains applied
    tau: jax.Array  # () filter time constant [s]

    def __call__(self, state: SystemState):
        u_raw = jax.vmap(lambda row: jnp.interp(state.t, self.act_time, row))(
            self.act_values
        )
        u_filt = state.control_state
        return u_filt, (u_raw - u_filt) / self.tau


def build_actuation(data):
    """Reproduce the MATLAB actuation chain exactly, from exported quantities.

    MATLAB (force mode):
        u_force = air_leaks_sf * A_c * bar2pa * (P * u_raw)   [N], pre-gain
        F_i     = g_i * u_force_i                             (gain applied
                                                               after the filter)

    soromox multiplies the applied chamber pressure by A_c internally, so the
    pressure handed to the solver is `g_i * air_leaks_i * bar2pa * (P u_raw)_i`
    and the resulting force is identical. Gain and filter are both linear and
    constant, so applying the gain before the lag is equivalent.
    """
    exp_time = np.atleast_1d(np.asarray(data["exp_time"], dtype=float)).ravel()
    u_raw = np.atleast_2d(np.asarray(data["u_raw_win"], dtype=float))  # (6, Mw) [bar]
    P = np.asarray(data["perm_matrix"], dtype=float)
    air_leaks = np.asarray(data["air_leaks_sf"], dtype=float)
    gains = np.atleast_1d(np.asarray(data["gains_hat"], dtype=float)).ravel()
    bar2pa = float(data["bar2pa"])
    tau = float(data["tau"])

    u_model_bar = P @ u_raw
    # The export also ships the permuted signal; assert rather than trust.
    ref = np.atleast_2d(np.asarray(data["u_model_bar"], dtype=float))
    assert np.allclose(u_model_bar, ref, atol=1e-12), (
        "permutation applied here does not reproduce the exported u_model_bar"
    )

    u_pa = np.diag(gains) @ air_leaks @ u_model_bar * bar2pa

    # Cross-check the force against the exported pre-gain force.
    chamber_area = float(data["chamber_area"])
    u_force_ref = np.atleast_2d(np.asarray(data["u_force"], dtype=float))
    assert np.allclose(
        u_pa * chamber_area, np.diag(gains) @ u_force_ref, atol=1e-9
    ), "force map does not reproduce the exported u_force"

    controller = RecordedInput(
        act_time=jnp.asarray(exp_time),
        act_values=jnp.asarray(u_pa),
        tau=jnp.asarray(tau),
    )
    return controller, exp_time, u_pa, u_model_bar


# ---------------------------------------------------------------------------
# Ladder rung 0: rest pose, no gravity integration, no actuation
# ---------------------------------------------------------------------------
def check_rest_pose(robot, info, data) -> bool:
    """Compare FK at q = 0 against the MATLAB rest pose (parity ladder rung 0)."""
    geom = data["geom"]
    q0 = jnp.zeros((robot.num_active_strains,))
    p_tip = np.asarray(robot.forward_kinematics(q0, s=info["s_tip"]))[0:3, 3]
    p_mid = np.asarray(robot.forward_kinematics(q0, s=info["s_mid"]))[0:3, 3]
    ref_tip = np.atleast_1d(np.asarray(geom.p_tip_rest, dtype=float)).ravel()
    ref_mid = np.atleast_1d(np.asarray(geom.p_mid_rest, dtype=float)).ravel()

    d_tip = float(np.linalg.norm(p_tip - ref_tip))
    d_mid = float(np.linalg.norm(p_mid - ref_mid))
    print("\n===== LADDER RUNG 0 : rest pose (q = 0) =====")
    print(f"tip soromox : [{p_tip[0] * 1e3:9.4f} {p_tip[1] * 1e3:9.4f} {p_tip[2] * 1e3:9.4f}] mm")
    print(f"tip MATLAB  : [{ref_tip[0] * 1e3:9.4f} {ref_tip[1] * 1e3:9.4f} {ref_tip[2] * 1e3:9.4f}] mm")
    print(f"  |diff|    : {d_tip * 1e3:.6e} mm")
    print(f"mid soromox : [{p_mid[0] * 1e3:9.4f} {p_mid[1] * 1e3:9.4f} {p_mid[2] * 1e3:9.4f}] mm")
    print(f"mid MATLAB  : [{ref_mid[0] * 1e3:9.4f} {ref_mid[1] * 1e3:9.4f} {ref_mid[2] * 1e3:9.4f}] mm")
    print(f"  |diff|    : {d_mid * 1e3:.6e} mm")
    ok = d_tip < 1e-9 and d_mid < 1e-9
    print(f"VERDICT     : {'PASS' if ok else 'FAIL'} (geometry / tip definition)")
    print("=" * 45)
    return ok


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def run_rollout(robot, info, data, rtol=1e-3, atol=1e-4, tmax=None, solver=None):
    controller, exp_time, u_pa, u_model_bar = build_actuation(data)

    if tmax is not None:
        keep = exp_time <= exp_time[0] + tmax
        exp_time = exp_time[keep]
        u_pa = u_pa[:, keep]
        u_model_bar = u_model_bar[:, keep]
        print(f"[tmax] window truncated to {exp_time[-1] - exp_time[0]:.3f} s "
              f"({exp_time.size} samples)")

    n_q = robot.num_active_strains

    # x0 is TRUE REST here (all-zero, verified on the MATLAB side -- see the
    # module docstring), so this split is a no-op in practice. Kept generic
    # (reads x0 from the export rather than hardcoding zeros) for structural
    # symmetry with the W5 warm-start contract, and so a future export with a
    # genuine nonzero x0 does not need this function touched again.
    x0_mat = np.asarray(data["x0"], dtype=float).ravel()
    assert x0_mat.size == 2 * n_q, (
        f"exported x0 has {x0_mat.size} entries, expected 2*ndof = {2 * n_q}"
    )
    q0 = jnp.asarray(x0_mat[:n_q])
    qd0 = jnp.asarray(x0_mat[n_q:])
    rest_tag = "true rest" if bool(jnp.all(x0_mat == 0.0)) else "NONZERO warm start"
    print(f"[initial state] |q0| = {float(jnp.linalg.norm(q0)):.4e} | "
          f"|qd0| = {float(jnp.linalg.norm(qd0)):.4e}  ({rest_tag})")

    t0 = float(exp_time[0])
    t1 = float(exp_time[-1])

    # Filter initial state: MATLAB starts the lag at the raw command evaluated
    # at the window start (zero-transient initialisation).
    u_filt0 = jnp.asarray(u_pa[:, 0])

    initial_state = SystemState(
        t=t0,
        y=jnp.concatenate([q0, qd0]),
        u=jnp.zeros((robot.num_actuators,)),
        control_state=u_filt0,
    )

    # Save exactly on the MATLAB/VICON grid so no interpolation enters the
    # parity comparison.
    save_ts = jnp.asarray(exp_time)
    save_dt = float(np.mean(np.diff(exp_time)))

    print(
        f"[rollout] t = {t0:.3f} -> {t1:.3f} s | {save_ts.size} save points | "
        f"solver {type(solver).__name__ if solver is not None else 'Tsit5'} | "
        f"rtol {rtol:g} atol {atol:g}",
        flush=True,
    )

    trajectory = robot.rollout_closed_loop_to(
        initial_state=initial_state,
        controller=controller,
        t1=t1,
        solver_dt=1e-3,
        save_dt=save_dt,
        save_ts=save_ts,
        solver=Tsit5() if solver is None else solver,
        stepsize_controller=PIDController(rtol=rtol, atol=atol),
        max_steps=None,
    )

    ts = np.asarray(trajectory.t)
    q_ts, qd_ts = jnp.split(trajectory.y, 2, axis=1)

    fk_tip = jax.jit(partial(robot.forward_kinematics, s=info["s_tip"]))
    fk_mid = jax.jit(partial(robot.forward_kinematics, s=info["s_mid"]))
    g_tip = np.asarray(jax.vmap(fk_tip)(q_ts))
    g_mid = np.asarray(jax.vmap(fk_mid)(q_ts))

    return {
        "t": ts,
        "q": np.asarray(q_ts),
        "qd": np.asarray(qd_ts),
        "u_applied": np.asarray(trajectory.u),
        "u_model_bar": u_model_bar,
        "p_tip": g_tip[:, 0:3, 3].T,  # (3, M)
        "p_mid": g_mid[:, 0:3, 3].T,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(bag_key, data, roll, stats_tip, met_sx, met_mat, outdir):
    import matplotlib

    matplotlib.use("Agg" if not os.environ.get("DISPLAY") else matplotlib.get_backend())
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )

    t = roll["t"]
    p_sx = 1e3 * roll["p_tip"]
    p_mat = 1e3 * np.asarray(data["p_tip_sim"], dtype=float).reshape(3, -1)
    p_vic = 1e3 * np.asarray(data["p_tip_meas"], dtype=float).reshape(3, -1)
    axis_names = ["x", "y", "z"]
    C = {"mat": "#1f4e79", "sx": "#d1495b", "vic": "#3f3f3f"}
    figs = {}

    # --- 1. tip traces: MATLAB vs soromox vs VICON -------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, p_vic[i], color=C["vic"], lw=1.4, alpha=0.55, label="VICON")
        ax.plot(t, p_mat[i], color=C["mat"], lw=1.8, label="MATLAB SoRoSim")
        ax.plot(t, p_sx[i], color=C["sx"], lw=1.2, ls="--", label="soromox")
        ax.set_ylabel(f"tip {axis_names[i]} [mm]")
    axes[0].legend(ncol=3, loc="upper right")
    axes[-1].set_xlabel("time since window start [s]")
    axes[0].set_title(
        f"AM-I Support tip, {bag_key} ({data['bagname']}), "
        f"{float(data['t_start']):.2f}-{float(data['t_end']):.2f} s\n"
        f"vs VICON: MATLAB {1e3 * met_mat.rmse_full:.2f} mm | "
        f"soromox {1e3 * met_sx.rmse_full:.2f} mm  --  "
        f"sim-vs-sim RMS {1e3 * stats_tip['rms']:.3f} mm"
    )
    fig.tight_layout()
    figs[f"{bag_key}_tip_traces"] = fig

    # --- 2. per-axis error against VICON -----------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    for i, ax in enumerate(axes):
        ax.axhline(0, color="k", lw=0.6)
        ax.plot(t, p_mat[i] - p_vic[i], color=C["mat"], lw=1.4, label="MATLAB - VICON")
        ax.plot(t, p_sx[i] - p_vic[i], color=C["sx"], lw=1.0, ls="--", label="soromox - VICON")
        ax.set_ylabel(f"{axis_names[i]} error [mm]")
    axes[0].legend(ncol=2, loc="upper right")
    axes[-1].set_xlabel("time since window start [s]")
    axes[0].set_title(
        f"Per-axis tracking error vs VICON, {bag_key}  "
        f"(both models share the same missing 1.408 Hz resonance)"
    )
    fig.tight_layout()
    figs[f"{bag_key}_axis_error"] = fig

    # --- 3. parity residual: soromox - MATLAB ------------------------------
    d = p_sx - p_mat
    n = np.linalg.norm(d, axis=0)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, height_ratios=[2, 1])
    for i in range(3):
        axes[0].plot(t, d[i], lw=1.1, label=f"{axis_names[i]}")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("soromox - MATLAB [mm]")
    axes[0].legend(ncol=3, loc="upper right")
    axes[0].set_title(
        f"Port parity residual, {bag_key}: RMS {1e3 * stats_tip['rms']:.4f} mm, "
        f"max {1e3 * stats_tip['max']:.4f} mm on a "
        f"{1e3 * stats_tip['ptp_reference']:.0f} mm peak-to-peak swing "
        f"(threshold {PARITY_THRESHOLD_MM:.1f} mm)"
    )
    axes[1].plot(t, n, color=C["sx"], lw=1.1)
    axes[1].axhline(PARITY_THRESHOLD_MM, color="k", ls=":", lw=1.0, label="threshold")
    axes[1].set_ylabel("|difference| [mm]")
    axes[1].set_xlabel("time since window start [s]")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    figs[f"{bag_key}_parity_residual"] = fig

    # --- 4. 3D overlay -----------------------------------------------------
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(p_vic[0], p_vic[1], p_vic[2], color=C["vic"], alpha=0.5, lw=1.0, label="VICON")
    ax.plot(p_mat[0], p_mat[1], p_mat[2], color=C["mat"], lw=1.5, label="MATLAB SoRoSim")
    ax.plot(p_sx[0], p_sx[1], p_sx[2], color=C["sx"], lw=1.0, ls="--", label="soromox")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    # Equal aspect: the trajectory is a physical path, so an unequal aspect
    # distorts the bending plane and makes the out-of-plane error look larger or
    # smaller than it is.
    ax.set_aspect("equal")
    ax.set_title(f"Tip trajectory overlay, {bag_key}")
    ax.legend()
    fig.tight_layout()
    figs[f"{bag_key}_tip_3d"] = fig

    os.makedirs(outdir, exist_ok=True)
    for name, f in figs.items():
        path = os.path.join(outdir, f"{name}.png")
        f.savefig(path, bbox_inches="tight")
        print(f"figure -> {path}")
    return figs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def refigure(bag_key, data, info, args, rung0) -> dict:
    """Regenerate metrics and figures from a saved CSV -- no rollout."""
    csv_path = os.path.join(args.outdir, f"{bag_key}{args.tag}_tip_positions.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"{csv_path} not found -- run the rollout first.")
    arr = np.genfromtxt(csv_path, delimiter=",", names=True)
    t = arr["time"]
    p_sx = np.vstack([arr["sx_x"], arr["sx_y"], arr["sx_z"]])
    p_mat = np.vstack([arr["mat_x"], arr["mat_y"], arr["mat_z"]])
    p_vic = np.vstack([arr["vicon_x"], arr["vicon_y"], arr["vicon_z"]])
    print(f"Refiguring from {csv_path} ({t.size} samples, no rollout)")

    _, _, _, u_model_bar = build_actuation(data)
    n = t.size
    u_model_bar = u_model_bar[:, :n]
    f0 = float(data["f0"])

    met_sx = tip_metrics(t, p_sx, p_vic, u_model_bar, f0=f0)
    met_mat = tip_metrics(t, p_mat, p_vic, u_model_bar, f0=f0)
    print_tip_metrics(met_sx, "TIP (soromox vs VICON)")
    print_tip_metrics(met_mat, "TIP (MATLAB vs VICON)")

    stats_tip = parity_stats(p_mat, p_sx)
    ok_tip = print_parity(stats_tip, f"{bag_key} TIP", PARITY_THRESHOLD_MM)

    # make_figures reads p_tip_sim / p_tip_meas off `data`; point them at the
    # CSV contents so a truncated or re-run trace stays self-consistent.
    data = dict(data)
    data["p_tip_sim"] = p_mat
    data["p_tip_meas"] = p_vic
    roll = {"t": t, "p_tip": p_sx, "u_model_bar": u_model_bar}
    make_figures(bag_key + args.tag, data, roll, stats_tip, met_sx, met_mat, args.outdir)

    return {
        "bag": bag_key,
        "rung0": rung0,
        "met_sx": met_sx,
        "met_mat": met_mat,
        "stats_tip": stats_tip,
        "parity_ok": bool(ok_tip),
    }


def run_bag(bag_key: str, args) -> dict:
    print("\n" + "#" * 70)
    print(f"# BAG: {bag_key}")
    print("#" * 70)

    data, bagname = load_export(bag_key, args.tag)
    print(
        f"source result : {str(data['source_result'])}\n"
        f"run id        : {str(data['source_run'])} (hash {str(data['source_hash'])})"
    )

    robot, info = build_robot(data)
    segs = info["segments"]
    print("\n--- soromox layout (from the exported linkage geometry) ---")
    print(f"{'seg':>4} {'kind':>6} {'L [mm]':>10} {'r [mm]':>9} {'rho':>10}")
    for i, s in enumerate(segs):
        print(
            f"{i:>4} {'rigid' if s['rigid'] else 'soft':>6} "
            f"{1e3 * s['L']:>10.3f} {1e3 * s['radius']:>9.3f} {s['rho']:>10.1f}"
        )
    print(
        f"E = {info['E']:.6e} Pa | Eta = {info['Eta']:.6e} Pa*s | Poi = {info['Poi']:.3f}"
    )
    print(
        "chamber azimuths [deg] : "
        + ", ".join(f"{v:.3f}" for v in info["azimuth_deg"])
        + f"  (clock {float(data['clock_deg']):+.1f} deg, "
        f"snapped from MATLAB dc, max dev {info['azimuth_snap_dev_deg']:.2e} deg)"
    )
    print(
        f"gains   : [{', '.join(f'{v:.4f}' for v in np.atleast_1d(data['gains_hat']))}]"
    )
    print(
        f"perm    : {np.atleast_1d(data['perm_idx']).astype(int).tolist()} (1-based) | "
        f"fc = {float(data['fc']):.2f} Hz (tau = {float(data['tau']) * 1e3:.3f} ms)"
    )
    print(f"ndof    : {robot.num_active_strains} | actuators : {robot.num_actuators}")
    print(f"s_tip   : {1e3 * info['s_tip']:.4f} mm | s_mid : {1e3 * info['s_mid']:.4f} mm")

    rung0 = check_rest_pose(robot, info, data)
    if args.ladder:
        return {"bag": bag_key, "rung0": rung0}

    if args.refigure:
        # Rebuild the figures and metrics from a previously saved CSV, without
        # repeating the rollout (which costs hours).
        return refigure(bag_key, data, info, args, rung0)

    import time as _time

    t_wall = _time.perf_counter()
    solver = {"tsit5": Tsit5, "kvaerno5": Kvaerno5}[args.solver]()
    rtol = float(data["ode_reltol"]) if args.rtol is None else args.rtol
    atol = float(data["ode_abstol"]) if args.atol is None else args.atol
    roll = run_rollout(
        robot,
        info,
        data,
        rtol=rtol,
        atol=atol,
        tmax=args.tmax,
        solver=solver,
    )
    jax.block_until_ready(roll["p_tip"])
    print(f"\nrollout (incl. JIT compile) : {_time.perf_counter() - t_wall:.1f} s", flush=True)

    if args.tmax is not None:
        # Truncated diagnostic run: the exported reference covers the full
        # window, so trim it to the same samples before scoring.
        n_keep = roll["t"].size
        for key in ("p_tip_meas", "p_mid_meas", "p_tip_sim", "p_mid_sim"):
            data[key] = np.asarray(data[key], dtype=float).reshape(3, -1)[:, :n_keep]

    t = roll["t"]
    p_vic_tip = np.asarray(data["p_tip_meas"], dtype=float).reshape(3, -1)
    p_vic_mid = np.asarray(data["p_mid_meas"], dtype=float).reshape(3, -1)
    f0 = float(data["f0"])
    u_ref = roll["u_model_bar"]

    met_sx = tip_metrics(t, roll["p_tip"], p_vic_tip, u_ref, f0=f0)
    print_tip_metrics(met_sx, "TIP (soromox vs VICON)")
    met_sx_mid = tip_metrics(t, roll["p_mid"], p_vic_mid, u_ref, f0=f0)

    result = {"bag": bag_key, "rung0": rung0, "met_sx": met_sx, "roll": roll, "data": data}

    if float(data["has_reference"]) != 1.0:
        print("\nNo MATLAB reference rollout in the export -- parity not evaluated.")
        result["parity_ok"] = None
        return result

    p_mat_tip = np.asarray(data["p_tip_sim"], dtype=float).reshape(3, -1)
    p_mat_mid = np.asarray(data["p_mid_sim"], dtype=float).reshape(3, -1)
    met_mat = tip_metrics(t, p_mat_tip, p_vic_tip, u_ref, f0=f0)
    print_tip_metrics(met_mat, "TIP (MATLAB vs VICON)")

    stats_tip = parity_stats(p_mat_tip, roll["p_tip"])
    stats_mid = parity_stats(p_mat_mid, roll["p_mid"])
    ok_tip = print_parity(stats_tip, f"{bag_key} TIP", PARITY_THRESHOLD_MM)
    ok_mid = print_parity(stats_mid, f"{bag_key} MID", PARITY_THRESHOLD_MM)

    print(
        f"\nvs VICON: MATLAB tip RMSE {1e3 * met_mat.rmse_full:.3f} mm | "
        f"soromox tip RMSE {1e3 * met_sx.rmse_full:.3f} mm | "
        f"delta {1e3 * (met_sx.rmse_full - met_mat.rmse_full):+.3f} mm"
    )

    result.update(
        {
            "met_mat": met_mat,
            "met_sx_mid": met_sx_mid,
            "stats_tip": stats_tip,
            "stats_mid": stats_mid,
            "parity_ok": bool(ok_tip and ok_mid),
        }
    )

    if not args.no_figs:
        make_figures(bag_key + args.tag, data, roll, stats_tip, met_sx, met_mat, args.outdir)

    # Per-sample CSV so the traces are inspectable outside Python.
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, f"{bag_key}{args.tag}_tip_positions.csv")
    header = "time,sx_x,sx_y,sx_z,mat_x,mat_y,mat_z,vicon_x,vicon_y,vicon_z"
    np.savetxt(
        csv_path,
        np.column_stack([t, roll["p_tip"].T, p_mat_tip.T, p_vic_tip.T]),
        delimiter=",",
        header=header,
        comments="",
    )
    print(f"csv    -> {csv_path}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bag", choices=["last"], default="last")
    ap.add_argument("--ladder", action="store_true", help="rung 0 only, no rollout")
    ap.add_argument(
        "--refigure",
        action="store_true",
        help="rebuild metrics and figures from the saved CSV, skipping the rollout",
    )
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--show", action="store_true", help="display figures (blocks)")
    ap.add_argument("--outdir", default=RESULTS_DIR)
    ap.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help="suffix of the exported .mat and of every output file "
        "(default: '_last', matching python_exporter_veronica_last.m)",
    )
    # Left as None, the solver tolerances are taken from the export
    # (cfg.ode_reltol / cfg.ode_abstol, i.e. MATLAB's own ode15s settings) so
    # the two engines are compared at the same accuracy instead of one being
    # held to a much tighter standard than the reference it is scored against.
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--atol", type=float, default=None)
    ap.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="truncate the fit window to this many seconds (timing diagnostic)",
    )
    ap.add_argument(
        "--solver",
        choices=["tsit5", "kvaerno5"],
        default="tsit5",
        help="kvaerno5 is implicit -- use it if the explicit solver stalls on stiffness",
    )
    args = ap.parse_args()

    results = [run_bag(args.bag, args)]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        if r.get("parity_ok") is None:
            print(f"{r['bag']:>8} : rung0 {'PASS' if r['rung0'] else 'FAIL'} | no reference")
            continue
        print(
            f"{r['bag']:>8} : rung0 {'PASS' if r['rung0'] else 'FAIL'} | "
            f"parity {'PASS' if r['parity_ok'] else 'FAIL'} "
            f"(tip RMS {1e3 * r['stats_tip']['rms']:.4f} mm) | "
            f"vs VICON MATLAB {1e3 * r['met_mat'].rmse_full:.2f} mm, "
            f"soromox {1e3 * r['met_sx'].rmse_full:.2f} mm"
        )

    if args.show and not args.no_figs:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
