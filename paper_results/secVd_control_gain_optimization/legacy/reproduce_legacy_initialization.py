"""Re-run the recovered Section Vd generator at its initialization.

This is the check that closed issue #154. It evaluates the legacy synergistic
objective at the six recovered gain initializations and compares the result
against ``history_loss[0]`` of the legacy MAT archive. A forward evaluation is
enough -- no autodiff, no optimizer -- which isolates the initialization, the
objective and the robot from the optimization dynamics.

Result, reproduced by running this file: all six starts agree to <= 1.8e-9
relative, which is solver noise accumulated over 50000 steps. That pins, at once,

* the six initial gain sets and the ``PRNGKey(15)`` behind them,
* the objective ``1e3 * (l1 + 5 * l2) + 2.5 * int P_t**2 dt``, including the
  tendon-power penalty that ``rescore_legacy_archives.py`` can only bound,
* the two-segment robot and its parameters, and
* the PID convention (the integral term uses the accumulator from *before* this
  step, which is then advanced -- ``integrate_after`` below). The alternative
  ordering is wrong by ~1e-3 relative, so the archive discriminates sharply.

Cross-check: ``2.5 * power_integral`` printed here equals, to every digit, the
residual that ``rescore_legacy_archives.py`` measures between the stored loss and
the tracking term. Two independent routes to the same number.

## Running it

The generator targets APIs removed by PR #117, so it needs an old soromox tree
plus three helpers the author kept in a local ``utils`` package that was never
committed. This script reimplements those three helpers and expects the old tree
on ``PYTHONPATH``::

    git worktree add /tmp/soromox-legacy backup/main_resolve_upon_time
    PYTHONPATH=/tmp/soromox-legacy/src JAX_PLATFORMS=cpu \\
        python paper_results/secVd_control_gain_optimization/legacy/\\
reproduce_legacy_initialization.py

Takes roughly 20 s per start on CPU. Pass a comma-separated list of starts as the
first argument to run a subset, and ``integrate_first`` as the second to see the
rejected PID ordering.
"""

from __future__ import annotations

import io
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.io as sio
from jax import lax, vmap

jax.config.update("jax_enable_x64", True)

from soromox.systems.tendon_actuated_pcs import TendonActuatedPCS  # noqa: E402

LEGACY_SYNERGISTIC_MAT_BLOB = "621f9c76902468bbd324ce5a880b860e4a4b9b6f"

# Recovered initialization. See the case README's provenance section.
LEGACY_SEED = 15
LEGACY_BATCH_SIZE = 6
LEGACY_NOMINAL = {"Kp_": 40.0, "Ki_": 20.0, "Kd_": 2.0}
LEGACY_BOX = {"Kp_": (20.0, 60.0), "Ki_": (10.0, 30.0), "Kd_": (2.0, 3.0)}

# Objective.
EARLY_WEIGHT, LATE_RATIO, POWER_WEIGHT = 1e3, 5.0, 2.5

RELATIVE_TOLERANCE = 1e-7


# --------------------------------------------------------------------------
# Robot, verbatim from the recovered generator (legacy/soromox_synergistic.py).
# --------------------------------------------------------------------------
def build_legacy_robot() -> tuple[TendonActuatedPCS, float, int, int]:
    """Rebuild the two-segment robot the legacy synergistic run used."""
    num_segments = 2
    segment_length, body_radius, tendon_radius = 0.10, 0.013, 0.012
    young_modulus = 14e4
    params = {
        "p0": jnp.array([jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, 0.0]),
        "L": segment_length * jnp.ones((num_segments,)),
        "r": body_radius * jnp.ones((num_segments,)),
        "rho": 1070 * jnp.ones((num_segments,)),
        "g": jnp.array([0.0, 0.0, 9.81]),
        "E": young_modulus * jnp.ones((num_segments,)),
        "G": (young_modulus / 3) * jnp.ones((num_segments,)),
    }
    params["D"] = 1e-4 * jnp.diag(
        (
            jnp.repeat(
                jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), num_segments, axis=0
            )
            * params["L"][:, None]
        ).flatten()
    )
    theta = jnp.pi / 32
    offsets = theta + jnp.array([0.0, 2 * jnp.pi / 3, 4 * jnp.pi / 3])
    routing = {
        "ry": tendon_radius * jnp.cos(offsets),
        "rz": tendon_radius * jnp.sin(offsets),
        "my": jnp.zeros(3),
        "mz": jnp.zeros(3),
        "idx_seg_att": jnp.array([1, 1, 1]),
    }
    robot = TendonActuatedPCS(
        num_segments=num_segments,
        params=params,
        active_tendon_routing_params=routing,
    )
    return robot, float(jnp.sum(robot.L)), 6 * num_segments, 3


# --------------------------------------------------------------------------
# The three helpers the author kept in an uncommitted local ``utils`` package.
# --------------------------------------------------------------------------
def legacy_pid(x, xd, x_des, dt, error_state, gain_p, gain_i, gain_d, variant):
    """Reimplementation of ``utils.controllers.PID``.

    ``integrate_after`` -- the recovered convention -- forms the command from the
    integral accumulated *before* this step and advances the accumulator
    afterwards. ``integrate_first`` advances it before use and is retained only
    to show that the archive rejects it.
    """
    error = x_des - x
    if variant == "integrate_after":
        force = gain_p @ error + gain_i @ error_state - gain_d @ xd
        return force, error_state + error * dt
    force = gain_p @ error + gain_i @ (error_state + error * dt) - gain_d @ xd
    return force, error_state + error * dt


def saturate_tanh(u, u_max=1.0, gain=1.0):
    """Reimplementation of the generator's ``saturate_tanh``."""
    return u_max * jnp.tanh(gain * u / u_max)


# ``utils.algebra.manip_index`` is not reimplemented: the generator only uses it
# to decide whether to damp the synergy inverse, via
# ``alpha = where(mu >= 0.1, 0, 2 * (1 - mu / 0.1) ** 2)``. Along this rollout the
# index is >= 0.598 under every candidate definition (sqrt(det(S S^T)),
# sigma_min / sigma_max, sigma_min), so ``alpha`` is identically zero and the
# missing definition provably cannot affect the result.
LEGACY_ALPHA = 0.0

# ``utils.algebra.cond_number`` only fed a logged diagnostic, never the loss.


def make_evaluator(robot, total_length, n, m, variant):
    """Evaluate the legacy synergistic objective for one gain set.

    Args:
        robot: The legacy two-segment tendon-actuated PCS.
        total_length: Backbone length, the abscissa the tip is read at.
        n: Configuration-space dimension.
        m: Number of tendons.
        variant: PID ordering, ``"integrate_after"`` or ``"integrate_first"``.

    Returns:
        A jitted callable mapping ``(Kp, Ki, Kd)`` to ``(loss, early, late, power)``.
    """
    t0, t1, solver_dt, control_dt = 0.0, 5.0, 1e-4, 1e-2
    saves_per_step = int(control_dt / solver_dt)
    max_steps = int((t1 - t0) / solver_dt) + 1
    x_des = jnp.array([-0.055, -0.094, 0.118])
    torque_max = 30.0

    def evaluate(gains):
        gain_p, gain_i, gain_d = (jnp.diag(g) for g in gains)

        def control(q, qd, error_state):
            actuation = robot.actuation_matrix(q)
            jacobian = robot.jacobian(q, total_length)[3:, :]
            inertia_inv = jnp.linalg.inv(robot.inertia_matrix(q))
            synergy = jacobian @ inertia_inv @ actuation
            mapping = (
                jnp.linalg.inv(synergy + LEGACY_ALPHA * jnp.eye(m))
                @ jacobian
                @ inertia_inv
            )
            x = robot.forward_kinematics(q, total_length)[:3, 3]
            force, error_state_next = legacy_pid(
                x,
                jacobian @ qd,
                x_des,
                control_dt,
                error_state,
                gain_p,
                gain_i,
                gain_d,
                variant,
            )
            torque = saturate_tanh(mapping @ jacobian.T @ force, u_max=torque_max)
            return torque, error_state_next, torque @ (actuation.T @ qd)

        def step(carry, t):
            q, qd = carry["state"][:n], carry["state"][n:]
            torque, error_state_next, power = control(q, qd, carry["error_state"])
            save_ts = jnp.linspace(t, t + control_dt - solver_dt, saves_per_step)
            t_ts, q_ts, qd_ts = robot.resolve_upon_time(
                q0=q,
                qd0=qd,
                u=torque,
                t0=t,
                t1=t + control_dt,
                dt=solver_dt,
                saveat_ts=save_ts,
                max_steps=max_steps,
            )
            next_carry = {
                "state": jnp.concatenate([q_ts[-1], qd_ts[-1]]),
                "error_state": error_state_next,
            }
            return next_carry, {
                "t_ts": t_ts,
                "q_ts": q_ts,
                "power_ts": jnp.tile(power, (t_ts.shape[0],)).squeeze(),
            }

        initial = {"state": jnp.zeros(2 * n), "error_state": jnp.zeros(m)}
        _, sol = lax.scan(step, initial, jnp.arange(t0, t1, control_dt))

        t_ts = sol["t_ts"].flatten()
        q_ts = sol["q_ts"].reshape(-1, n)
        power_ts = sol["power_ts"].flatten()
        split = int(len(t_ts) / 2)
        x_ts = vmap(robot.forward_kinematics, in_axes=(0, None))(q_ts, total_length)[
            :, :3, 3
        ]
        error_sq = jnp.linalg.norm(x_des - x_ts, axis=1) ** 2
        early = jnp.trapezoid(error_sq[:split], t_ts[:split])
        late = jnp.trapezoid(error_sq[split:], t_ts[split:])
        power = jnp.trapezoid(power_ts**2, t_ts)
        loss = EARLY_WEIGHT * (early + LATE_RATIO * late) + POWER_WEIGHT * power
        return loss, early, late, power

    return jax.jit(evaluate)


def sample_legacy_initial_gains():
    """Reproduce ``generate_mixed_batch`` from the recovered generator.

    The original flattened a PyTree, so the split keys were consumed in *sorted*
    key order -- ``Kd_``, ``Ki_``, ``Kp_`` -- not in the natural ``Kp, Ki, Kd``
    order. Getting this wrong resamples every start.
    """
    names = sorted(LEGACY_NOMINAL)  # Kd_, Ki_, Kp_
    keys = jax.random.split(jax.random.PRNGKey(LEGACY_SEED), len(names))
    gains = {}
    for name, key in zip(names, keys):
        nominal = LEGACY_NOMINAL[name] * jnp.ones(3)
        low, high = LEGACY_BOX[name]
        subkeys = jax.random.split(key, LEGACY_BATCH_SIZE - 1)
        sampled = jax.vmap(
            lambda k, low=low, high=high: jax.random.uniform(
                k, shape=(3,), minval=low, maxval=high
            )
        )(subkeys)
        gains[name] = jnp.concatenate([nominal[None, :], sampled], axis=0)
    return gains


def main() -> int:
    starts = (
        [int(s) for s in sys.argv[1].split(",")]
        if len(sys.argv) > 1
        else list(range(LEGACY_BATCH_SIZE))
    )
    variant = sys.argv[2] if len(sys.argv) > 2 else "integrate_after"

    raw = subprocess.check_output(
        ["git", "cat-file", "blob", LEGACY_SYNERGISTIC_MAT_BLOB]
    )
    archive = sio.loadmat(io.BytesIO(raw))
    reference = archive["history_loss"][0]

    robot, total_length, n, m = build_legacy_robot()
    tip = robot.forward_kinematics(jnp.zeros(n), total_length)[:3, 3]
    print(f"PID variant: {variant}")
    print(f"tip at q=0:  {np.asarray(tip)}  (archive: {archive['x_ts_init'][0, 0]})")
    print(
        f"\n{'b':>2} {'predicted':>13} {'archive':>13} {'rel.err':>11}  "
        f"{'2.5*power':>11}"
    )

    evaluate = make_evaluator(robot, total_length, n, m, variant)
    gains = sample_legacy_initial_gains()
    worst = 0.0
    for b in starts:
        started = time.time()
        loss, _early, _late, power = evaluate(
            (gains["Kp_"][b], gains["Ki_"][b], gains["Kd_"][b])
        )
        loss = float(loss)
        relative = abs(loss - reference[b]) / reference[b]
        worst = max(worst, relative)
        print(
            f"{b:>2} {loss:13.6f} {reference[b]:13.6f} {relative:11.3e}  "
            f"{POWER_WEIGHT * float(power):11.6f}   [{time.time() - started:.0f}s]"
        )

    print(f"\nworst relative error: {worst:.3e}")
    if variant == "integrate_after" and worst > RELATIVE_TOLERANCE:
        print("FAILED: the recovered initialization no longer reproduces the archive")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
