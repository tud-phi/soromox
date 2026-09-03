"""The canonical Section Vd optimization objective."""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import Array

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

OBJECTIVE_NAME = "dimensionless_mean_squared_error_v1"
_ANGULAR_STRAINS_PER_SEGMENT = 3


def configuration_space_scales(segment_lengths: Array) -> Array:
    """Return the factors that make a PCS strain error dimensionless.

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
    interval = (float(timestamps[0]), float(timestamps[-1]))
    mask = np.array(terminal_window_masks(timestamps, [interval], fraction=fraction)[0])
    mask[-1] = True
    return {
        "rms_overall": float(np.sqrt(np.mean(np.sum(scaled**2, axis=-1)))),
        "rms_steady_state": float(np.sqrt(np.mean(np.sum(scaled[mask] ** 2, axis=-1)))),
        "componentwise_rmse": componentwise_rmse(scaled).tolist(),
        "steady_state_starts_at": float(np.asarray(t)[mask][0]),
    }


def recompute_archive_losses(data: dict) -> dict[str, np.ndarray]:
    """Rescore an archive's stored trajectories with its own objective.

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
