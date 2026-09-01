"""The canonical Section Vd optimization objective.

Both generators import this, so "Loss" in the comparison figure means the same
thing in both panels. That was not previously true, and the follow-up on issue
#154 asked for it to be made so.

## What the legacy objectives were

Recovered from the archives (see `../legacy/`), the published runs used

```text
collocated   1e2*int_0^2.5 e^T W e dt + 2e2*int_2.5^5 e^T W e dt,
             W = diag([1, 1, 1, 100, 100, 100])
synergistic  1e3*int_0^2.5 ||e||^2 dt + 5e3*int_2.5^5 ||e||^2 dt + 2.5*int P_t^2 dt
```

Different global factors, different early/late ratios (1:2 against 1:5), one
with a strain weighting and one with a tendon-power penalty. The plotter then
divided the collocated loss by 100 purely to align the two y-axes. So the
plotted "Loss" was neither the optimized objective nor comparable across
controllers.

## What this module defines instead

The legacy collocated weighting turns out not to be arbitrary. With the segment
length ``L = 0.1 m``,

```text
W = diag([1, 1, 1, 100, 100, 100]) = (1 / L**2) * diag(L**2, L**2, L**2, 1, 1, 1)
```

so ``e^T W e = ||e_hat||**2 / L**2`` for ``e_hat = [L * e_kappa ; e_sigma]``. The
legacy objective was already the strain error **non-dimensionalized by segment
length**; only the leading constant was arbitrary. That fixes the characteristic
scale the follow-up asked for, from the data rather than by choice.

The canonical objective is therefore the time-averaged squared dimensionless
error,

```text
L = (1 / T) * int_0^T ||e_hat(t)||**2 dt
```

with

```text
collocated   e_hat = [L_seg * e_kappa ; e_sigma]     (strains)
synergistic  e_hat = [e_theta ; e_x / L_seg]         (pose)
```

Angular strains are ``1/m`` and shear/axial strains are already dimensionless;
task-space orientation error is in radians and position error in metres. Both
sides therefore come out dimensionless and directly comparable, with no global
factor and no per-controller divisor in the plotter.

**There is no early/late weighting.** The legacy runs disagreed on it (1:2
against 1:5) and today's scripts had harmonized on 1:5, which was never the
original on either side. Rather than bake a transient/steady-state preference
into the objective, the split is *reported* as a metric --
:func:`steady_state_report` -- so it can be inspected without silently steering
the optimizer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import Array

# The steady-state metrics are shared with the Section Vc study rather than
# reimplemented here.
_SECVC_CODE = Path(__file__).resolve().parents[2] / "secVc_model_based_control" / "code"
if str(_SECVC_CODE) not in sys.path:
    sys.path.insert(0, str(_SECVC_CODE))

from evaluation_metrics import (  # noqa: E402
    STEADY_STATE_FRACTION,
    componentwise_rmse,
    terminal_window_masks,
)

__all__ = [
    "OBJECTIVE_NAME",
    "STEADY_STATE_FRACTION",
    "configuration_space_scales",
    "operational_space_scales",
    "steady_state_report",
    "time_averaged_squared_error",
]

# Recorded in every archive so a stored loss can always be traced to its
# definition.
OBJECTIVE_NAME = "dimensionless_mean_squared_error_v1"

# PCS strains are laid out per segment as
# [kappa_x, kappa_y, kappa_z, sigma_x, sigma_y, sigma_z]: the first three are
# curvatures in 1/m, the last three are dimensionless shear/axial strains.
_ANGULAR_STRAINS_PER_SEGMENT = 3


def configuration_space_scales(segment_lengths: Array) -> Array:
    """Return the factors that make a PCS strain error dimensionless.

    Multiplying the angular strains by their segment length and leaving the
    shear/axial strains alone reproduces the legacy collocated weighting exactly,
    up to its arbitrary global constant.

    Args:
        segment_lengths: Per-segment lengths in metres, shape ``(num_segments,)``.

    Returns:
        Scales of shape ``(6 * num_segments,)``.
    """
    lengths = jnp.asarray(segment_lengths)
    per_segment = jnp.concatenate(
        [
            jnp.broadcast_to(
                lengths[:, None], (lengths.shape[0], _ANGULAR_STRAINS_PER_SEGMENT)
            ),
            jnp.ones((lengths.shape[0], _ANGULAR_STRAINS_PER_SEGMENT)),
        ],
        axis=1,
    )
    return per_segment.reshape(-1)


def operational_space_scales(
    task_selector: Array, characteristic_length: float
) -> Array:
    """Return the factors that make a task-space pose error dimensionless.

    Args:
        task_selector: Boolean mask of shape ``(6,)`` over
            ``[rotation xyz, translation xyz]``, selecting the controlled
            components.
        characteristic_length: Length scale in metres, normally the backbone
            length.

    Returns:
        Scales of shape ``(number of selected components,)``. Orientation entries
        are ``1`` (radians are already dimensionless); position entries are
        ``1 / characteristic_length``.
    """
    selector = jnp.asarray(task_selector, dtype=bool)
    full = jnp.concatenate([jnp.ones(3), jnp.full((3,), 1.0 / characteristic_length)])
    return full[selector]


def time_averaged_squared_error(error: Array, scales: Array, t: Array) -> Array:
    """Return the canonical dimensionless loss for one rollout.

    Args:
        error: Signed error of shape ``(timesteps, k)``, in physical units.
        scales: Per-component factors of shape ``(k,)`` making it dimensionless.
        t: Time grid of shape ``(timesteps,)``.

    Returns:
        Scalar ``(1 / T) * int ||scales * error||**2 dt``. Time-averaging rather
        than integrating keeps the value independent of the horizon, so runs of
        different length stay comparable.
    """
    squared = jnp.sum((error * scales) ** 2, axis=-1)
    horizon = t[-1] - t[0]
    return jnp.trapezoid(squared, t) / horizon


def steady_state_report(
    error: np.ndarray,
    scales: np.ndarray,
    t: np.ndarray,
    *,
    fraction: float = STEADY_STATE_FRACTION,
) -> dict[str, float]:
    """Report transient and steady-state error without weighting the objective.

    The legacy runs disagreed about how much to emphasize the steady state, so
    that preference is reported here instead of being folded into the loss.

    Args:
        error: Signed error of shape ``(timesteps, k)``, in physical units.
        scales: Per-component factors of shape ``(k,)``.
        t: Time grid of shape ``(timesteps,)``.
        fraction: Terminal fraction of the horizon treated as steady state.

    Returns:
        Dimensionless RMS error over the whole horizon and over the terminal
        window, plus the window boundary in seconds.
    """
    scaled = np.asarray(error) * np.asarray(scales)
    timestamps = np.asarray(t, dtype=float)
    # Section Vd holds one constant reference for the whole rollout, so there is
    # a single interval spanning it.
    interval = (float(timestamps[0]), float(timestamps[-1]))
    mask = np.array(terminal_window_masks(timestamps, [interval], fraction=fraction)[0])
    # The interval is half-open, which would drop the final sample. Widening the
    # bound instead would move the window boundary by a whole sample.
    mask[-1] = True
    return {
        "rms_overall": float(np.sqrt(np.mean(np.sum(scaled**2, axis=-1)))),
        "rms_steady_state": float(np.sqrt(np.mean(np.sum(scaled[mask] ** 2, axis=-1)))),
        "componentwise_rmse": componentwise_rmse(scaled).tolist(),
        "steady_state_starts_at": float(np.asarray(t)[mask][0]),
    }


def recompute_archive_losses(data: dict) -> dict[str, np.ndarray]:
    """Rescore an archive's stored trajectories with its own objective.

    This is the regression check the follow-up on issue #154 asked for. The
    legacy discrepancy -- stored losses an order of magnitude away from what the
    committed objective scored the same trajectories at -- went unnoticed for
    months because nothing ever recomputed one from the other.

    It works from the archive alone: ``objective_scales`` and ``objective_name``
    are stored precisely so the loss never again depends on knowing which robot
    and which weighting produced it.

    Only the initial and best rollouts are archived, so only those two losses per
    start can be checked; that is enough to catch a changed objective, a changed
    normalization, or a trajectory paired with the wrong loss.

    Args:
        data: A loaded schema-v2 archive.

    Returns:
        ``stored_init``, ``recomputed_init``, ``stored_best`` and
        ``recomputed_best``, each of shape ``(B,)``.

    Raises:
        ValueError: If the archive records an objective this module cannot score.
    """
    name = str(np.asarray(data["objective_name"]).item())
    if name != OBJECTIVE_NAME:
        raise ValueError(
            f"Archive was optimized with objective {name!r}, but this module "
            f"defines {OBJECTIVE_NAME!r}; rescoring would compare two "
            "different quantities"
        )
    scales = np.asarray(data["objective_scales"])
    t = np.asarray(data["t_ts"])
    loss = np.asarray(data["history_loss"])
    mask = np.asarray(data["history_finite_mask"], dtype=bool)
    best_iteration = np.asarray(data["best_iteration"])
    method = str(np.asarray(data["method"]).item())

    if method == "collocated":
        reference = np.asarray(data["q_des_ts"])
        initial = reference - np.asarray(data["q_ts_init"])
        best = reference - np.asarray(data["q_ts_best"])
    else:
        # The task error lives in the selected pose components, which is what
        # objective_scales is sized for.
        selected = slice(6 - scales.shape[0], 6)
        reference = np.asarray(data["x_des_ts"])[:, selected]
        initial = reference - np.asarray(data["x_ts_init"])[:, :, selected]
        best = reference - np.asarray(data["x_ts_best"])[:, :, selected]

    def score(errors: np.ndarray) -> np.ndarray:
        scaled = np.sum((errors * scales) ** 2, axis=-1)
        return np.trapezoid(scaled, t, axis=-1) / (t[-1] - t[0])

    batch = loss.shape[1]
    return {
        "stored_init": np.where(mask[0], loss[0], np.nan),
        "recomputed_init": score(initial),
        "stored_best": np.array([loss[best_iteration[b], b] for b in range(batch)]),
        "recomputed_best": score(best),
    }
