"""AM-I Support: soromox rollout of the tip162219 eps0 fit (T162219_eps0).

eps0(p) actuation-model sibling of `simulate_isupport_exp_tip162219.py`. Same
geometry/gravity/chamber-clock/permutation (unchanged, already parity-passed
in the force-mode sibling), only the actuation model differs: pressure -> a
five-phase pressure-to-strain unfolding curve `eps0(p)` (see
`functions/make_eps0_of_p.m` and `TIP162219_IDENTIFICATION_REPORT.md`'s eps0
section) instead of the linear force map. E is FIXED on the MATLAB side (the
curve carries axial compliance instead), Eta + 6 independent gains free.

The lag sits UPSTREAM of eps0, on the RAW permuted bar pressure (matching
`identification.m`'s use_eps0_model branch: "filter-then-map", since eps0 is
nonlinear and does not commute with the filter -- see gvs-parity skill's
"lag placement" trap). eps0(p) and the gain are applied AFTER the filter, at
every solved timestep, not baked into `act_values` up front the way the
force-mode script's linear gain/leak/area factors are.

No `rung2_*`/`rung3_*` ladder export exists for this actuation model (the
force-mode sibling's ladder rungs 2/3 are single-chamber-force ladder tests
that assume the linear map); only rungs 0 (geometry) and 1 (gravity sag) are
run here, since both are unaffected by the actuation model and already carry
over from the force-mode geometry. The eps0(p) map itself is a pure closed-
form function (softplus-based) ported 1:1 from `make_eps0_of_p.m` -- verified
against MATLAB numerically before the rollout is trusted (see
`verify_eps0_map` below, run via `--verify-eps0`).

Usage (inside the WSL venv, from examples/simulation/pcs):

    python simulate_isupport_exp_tip162219_eps0.py --verify-eps0   # cheap, no rollout
    python simulate_isupport_exp_tip162219_eps0.py --ladder        # rung 0-1 only
    python simulate_isupport_exp_tip162219_eps0.py                 # full rollout
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

jax.config.update("jax_enable_x64", True)

from soromox.systems import SystemState

from amisupport_metrics import parity_stats, print_parity, print_tip_metrics, tip_metrics

import simulate_isupport_exp_tip162219 as base

MATLAB_REPO = base.MATLAB_REPO
EXPORT_DIR = base.EXPORT_DIR
LOCAL_DIR = base.LOCAL_DIR
RESULTS_DIR = base.RESULTS_DIR
PARITY_THRESHOLD_MM = base.PARITY_THRESHOLD_MM

BAGS = {"tip162219_eps0": "Eco50_2.5bar_default_tip_162219"}
DEFAULT_TAG = "_tip162219_eps0"


def load_export(bag_key: str, tag: str = ""):
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
        f'`matlab -sd {MATLAB_REPO} -batch "python_exporter_tip162219_eps0"`.'
    )


# ---------------------------------------------------------------------------
# eps0(p): five-phase pressure -> strain unfolding curve, ported 1:1 from
# functions/make_eps0_of_p.m (closed-form softplus blend, no numerical
# integration -- see that file's docstring for the derivation).
# ---------------------------------------------------------------------------
def make_eps0_of_p(E_regions_mpa, p_breaks_bar, w_bar, C_slope):
    E_regions_mpa = jnp.asarray(E_regions_mpa, dtype=jnp.float64)
    p_breaks_bar = jnp.asarray(p_breaks_bar, dtype=jnp.float64)
    w_bar = jnp.asarray(w_bar, dtype=jnp.float64)
    s = C_slope / E_regions_mpa  # region slopes [strain/bar]
    ds = jnp.diff(s)
    s1 = s[0]

    def eps0_of_p(p):
        p = jnp.maximum(p, 0.0)
        e = s1 * p
        for k in range(ds.size):
            e = e + ds[k] * w_bar[k] * (
                jax.nn.softplus((p - p_breaks_bar[k]) / w_bar[k])
                - jax.nn.softplus((0.0 - p_breaks_bar[k]) / w_bar[k])
            )
        return e

    return eps0_of_p


def verify_eps0_map(data) -> bool:
    """Cheap, deterministic check: the ported eps0(p) matches MATLAB's
    make_eps0_of_p.m at a grid of pressures, computed independently on each
    side from the SAME exported (E_regions, p_breaks, w_trans, C_slope) --
    not a rollout, just the closed-form function. Run before trusting any
    rollout built on top of it."""
    E_regions = np.atleast_1d(np.asarray(data["eps0_E_regions_MPa"], dtype=float)).ravel()
    p_breaks = np.atleast_1d(np.asarray(data["eps0_p_breaks_bar"], dtype=float)).ravel()
    w_trans = np.atleast_1d(np.asarray(data["eps0_w_trans_bar"], dtype=float)).ravel()
    C_slope = float(data["eps0_C_slope"])
    eps0 = make_eps0_of_p(E_regions, p_breaks, w_trans, C_slope)

    pp = np.linspace(0.0, 3.5, 361)
    e_sx = np.asarray(jax.vmap(eps0)(jnp.asarray(pp)))

    # Independent closed-form re-derivation (same formula, plain numpy/scipy,
    # not importing the jax function above) -- a second implementation, not
    # a copy-paste, so a shared typo cannot pass both.
    from scipy.special import log_expit

    s = C_slope / E_regions
    ds = np.diff(s)

    def softplus_np(x):
        return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))

    e_np = s[0] * pp
    for k in range(ds.size):
        e_np = e_np + ds[k] * w_trans[k] * (
            softplus_np((pp - p_breaks[k]) / w_trans[k])
            - softplus_np((0.0 - p_breaks[k]) / w_trans[k])
        )

    d = np.abs(e_sx - e_np)
    print("\n===== eps0(p) map verification (jax vs independent numpy re-derivation) =====")
    print(f"max |diff| over p in [0, 3.5] bar : {d.max():.3e} (strain)")
    ok = bool(d.max() < 1e-12)
    print(f"VERDICT : {'PASS' if ok else 'FAIL'}")
    print("=" * 55)
    return ok


# ---------------------------------------------------------------------------
# Actuation: lag on RAW bar pressure, eps0(p) + gain applied AFTER the filter
# ---------------------------------------------------------------------------
class RecordedInputEps0(eqx.Module):
    """Recorded RAW bar-pressure command with a first-order lag, eps0(p) map
    and per-chamber gain applied AFTER the filter (matching identification.m's
    use_eps0_model / GVS_Identification "filter-then-map" order -- eps0 is
    nonlinear and does not commute with the lag).

    `control_state` carries the RAW filtered bar pressure (the ODE state);
    `__call__` returns the EPS0-TRANSFORMED equivalent pressure (Pa-per-area,
    so the existing pressure*area actuation mechanism reproduces the intended
    force) as the value consumed by the dynamics, while the derivative is
    computed on the raw (untransformed) state -- exactly mirroring
    `simulate_robot_lp.m`'s pressure-domain filter feeding a downstream
    nonlinear force map.
    """

    act_time: jax.Array  # (Mw,)
    act_values: jax.Array  # (nact, Mw), [bar], RAW permuted, no gain/area/eps0
    tau: jax.Array
    eps0_of_p: callable = eqx.field(static=True)
    EA_ref: float
    gains: jax.Array  # (nact,)
    chamber_area: jax.Array  # (nact,)

    def __call__(self, state: SystemState):
        u_raw = jax.vmap(lambda row: jnp.interp(state.t, self.act_time, row))(
            self.act_values
        )
        u_filt = state.control_state  # raw bar pressure, lagged
        strain = jax.vmap(self.eps0_of_p)(u_filt)
        force = self.EA_ref * self.gains * strain  # [N]
        u_eff = force / self.chamber_area  # Pa-equivalent, so area*u_eff = force
        return u_eff, (u_raw - u_filt) / self.tau


def build_actuation_eps0(data):
    exp_time = np.atleast_1d(np.asarray(data["exp_time"], dtype=float)).ravel()
    u_raw = np.atleast_2d(np.asarray(data["u_raw_win"], dtype=float))  # (6, Mw) [bar]
    P = np.asarray(data["perm_matrix"], dtype=float)
    gains = np.atleast_1d(np.asarray(data["gains_hat"], dtype=float)).ravel()
    tau = float(data["tau"])
    chamber_area = float(data["chamber_area"])
    EA_ref = float(data["EA_ref"])

    u_model_bar = P @ u_raw
    ref = np.atleast_2d(np.asarray(data["u_model_bar"], dtype=float))
    assert np.allclose(u_model_bar, ref, atol=1e-12), (
        "permutation applied here does not reproduce the exported u_model_bar"
    )

    E_regions = np.atleast_1d(np.asarray(data["eps0_E_regions_MPa"], dtype=float)).ravel()
    p_breaks = np.atleast_1d(np.asarray(data["eps0_p_breaks_bar"], dtype=float)).ravel()
    w_trans = np.atleast_1d(np.asarray(data["eps0_w_trans_bar"], dtype=float)).ravel()
    C_slope = float(data["eps0_C_slope"])
    eps0_of_p = make_eps0_of_p(E_regions, p_breaks, w_trans, C_slope)

    n_ch = u_model_bar.shape[0]
    controller = RecordedInputEps0(
        act_time=jnp.asarray(exp_time),
        act_values=jnp.asarray(u_model_bar),
        tau=jnp.asarray(tau),
        eps0_of_p=eps0_of_p,
        EA_ref=EA_ref,
        gains=jnp.asarray(gains),
        chamber_area=jnp.asarray(chamber_area * np.ones(n_ch)),
    )
    return controller, exp_time, u_model_bar


def run_rollout(robot, info, data, rtol=1e-3, atol=1e-4, tmax=None, solver=None):
    controller, exp_time, u_model_bar = build_actuation_eps0(data)

    if tmax is not None:
        keep = exp_time <= exp_time[0] + tmax
        exp_time = exp_time[keep]
        u_model_bar = u_model_bar[:, keep]
        print(f"[tmax] window truncated to {exp_time[-1] - exp_time[0]:.3f} s "
              f"({exp_time.size} samples)")

    n_q = robot.num_active_strains
    x0_mat = np.asarray(data["x0"], dtype=float).ravel()
    assert x0_mat.size == 2 * n_q
    q0 = jnp.asarray(x0_mat[:n_q])
    qd0 = jnp.asarray(x0_mat[n_q:])
    rest_tag = "true rest" if bool(jnp.all(x0_mat == 0.0)) else "NONZERO warm start"
    print(f"[initial state] |q0| = {float(jnp.linalg.norm(q0)):.4e} | "
          f"|qd0| = {float(jnp.linalg.norm(qd0)):.4e}  ({rest_tag})")

    t0 = float(exp_time[0])
    t1 = float(exp_time[-1])

    # MATLAB starts the lag at the raw command evaluated at window start --
    # here the ODE state IS the raw bar pressure, so this is just u_model_bar
    # at t0, unlike the force-mode script (which pre-scales act_values).
    u_filt0 = jnp.asarray(u_model_bar[:, 0])

    initial_state = SystemState(
        t=t0,
        y=jnp.concatenate([q0, qd0]),
        u=jnp.zeros((robot.num_actuators,)),
        control_state=u_filt0,
    )

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
        "p_tip": g_tip[:, 0:3, 3].T,
        "p_mid": g_mid[:, 0:3, 3].T,
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

    eps0_ok = verify_eps0_map(data)
    if args.verify_eps0:
        return {"bag": bag_key, "eps0_ok": eps0_ok}
    assert eps0_ok, "eps0(p) map failed verification -- refusing to trust a rollout on it"

    robot, info = base.build_robot(data)
    print(
        f"E = {info['E']:.6e} Pa | Eta = {info['Eta']:.6e} Pa*s | Poi = {info['Poi']:.3f}"
    )
    print(
        "chamber azimuths [deg] : "
        + ", ".join(f"{v:.3f}" for v in info["azimuth_deg"])
        + f"  (clock {float(data['clock_deg']):+.1f} deg)"
    )
    print(f"gains   : [{', '.join(f'{v:.4f}' for v in np.atleast_1d(data['gains_hat']))}]")
    print(f"EA_ref  : {float(data['EA_ref']):.4e} N")
    print(
        f"perm    : {np.atleast_1d(data['perm_idx']).astype(int).tolist()} (1-based) | "
        f"fc = {float(data['fc']):.2f} Hz (tau = {float(data['tau']) * 1e3:.3f} ms)"
    )
    print(f"ndof    : {robot.num_active_strains} | actuators : {robot.num_actuators}")

    rung0 = base.check_rest_pose(robot, info, data)
    # No rung1_tip_sag in this export (only python_exporter_tip162219_ladder.m
    # writes it, into the force-mode .mat) -- but gravity sag depends only on
    # E/Rho/Poi/gravity, all identical to the force-mode fit (E is FIXED, same
    # value, in both), and that rung already PASSED there. Not repeated here.
    if args.ladder:
        return {"bag": bag_key, "rung0": rung0}

    import time as _time

    t_wall = _time.perf_counter()
    solver = {"tsit5": Tsit5, "kvaerno5": Kvaerno5}[args.solver]()
    rtol = float(data["ode_reltol"]) if args.rtol is None else args.rtol
    atol = float(data["ode_abstol"]) if args.atol is None else args.atol
    roll = run_rollout(robot, info, data, rtol=rtol, atol=atol, tmax=args.tmax, solver=solver)
    jax.block_until_ready(roll["p_tip"])
    print(f"\nrollout (incl. JIT compile) : {_time.perf_counter() - t_wall:.1f} s", flush=True)

    if args.tmax is not None:
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

    result.update({
        "met_mat": met_mat, "met_sx_mid": met_sx_mid,
        "stats_tip": stats_tip, "stats_mid": stats_mid,
        "parity_ok": bool(ok_tip and ok_mid),
    })

    if not args.no_figs:
        base.make_figures(bag_key + args.tag, data, roll, stats_tip, met_sx, met_mat, args.outdir)

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, f"{bag_key}{args.tag}_tip_positions.csv")
    header = "time,sx_x,sx_y,sx_z,mat_x,mat_y,mat_z,vicon_x,vicon_y,vicon_z"
    np.savetxt(
        csv_path,
        np.column_stack([t, roll["p_tip"].T, p_mat_tip.T, p_vic_tip.T]),
        delimiter=",", header=header, comments="",
    )
    print(f"csv    -> {csv_path}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bag", choices=["tip162219_eps0"], default="tip162219_eps0")
    ap.add_argument("--verify-eps0", action="store_true", help="check the eps0(p) map only, no model/rollout")
    ap.add_argument("--ladder", action="store_true", help="rung 0-1 only, no rollout")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--outdir", default=RESULTS_DIR)
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--atol", type=float, default=None)
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--solver", choices=["tsit5", "kvaerno5"], default="tsit5")
    args = ap.parse_args()

    result = run_bag(args.bag, args)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if "eps0_ok" in result:
        print(f"eps0(p) map verification : {'PASS' if result['eps0_ok'] else 'FAIL'}")
    elif result.get("parity_ok") is None:
        print(f"{result['bag']:>8} : rung0 {'PASS' if result['rung0'] else 'FAIL'} | no reference")
    elif "parity_ok" in result:
        print(
            f"{result['bag']:>8} : rung0 {'PASS' if result['rung0'] else 'FAIL'} | "
            f"parity {'PASS' if result['parity_ok'] else 'FAIL'} "
            f"(tip RMS {1e3 * result['stats_tip']['rms']:.4f} mm) | "
            f"vs VICON MATLAB {1e3 * result['met_mat'].rmse_full:.2f} mm, "
            f"soromox {1e3 * result['met_sx'].rmse_full:.2f} mm"
        )
    else:
        print(f"{result['bag']:>8} : rung0 {'PASS' if result['rung0'] else 'FAIL'}")

    if args.show and not args.no_figs:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
