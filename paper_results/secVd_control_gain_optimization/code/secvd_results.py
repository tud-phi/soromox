"""Canonical Section Vd result schema, validation, and serialization.

Schema version 3 carries a real multi-start batch axis of width ``B``, restoring
the six-start structure of the published results (issue #154). The axis appears
**only** on quantities that genuinely vary per start: the loss and gain
histories, each start's initial and best rollouts, and the selection indices. The
time grid and the reference trajectory are shared by every start and are stored
without one, so a batch axis in this archive always means something.

A start that produces a non-finite candidate is frozen rather than aborting the
whole run, and ``history_finite_mask`` records exactly which entries are real.
Every start must nevertheless attain at least one finite iterate: a start that
was non-finite from its first evaluation has no trajectory worth archiving, and
signals an initialization that should be resampled.

``init_Kp`` / ``init_Ki`` / ``init_Kd``, together with ``init_seed``,
``init_scheme`` and ``init_spread``, record the initialization directly rather
than leaving it to be inferred from the history. The legacy results were
unrecoverable from their archives precisely because this was never written; see
the provenance section of the case README.

**Per-iteration trajectory histories are deliberately absent.** Storing every
start's rollout at every iteration would reach several gigabytes at 100
iterations and six starts, so the archive keeps each start's initial and best
rollouts only. The loss and gain histories remain complete.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# v3 records the two case parameters that were previously implicit. Both now
# vary: the integration step is exposed on the generators, and the plant's
# material damping was changed from 3.6e2 to 7.2e1 (see secvd_case). Without
# them an archive cannot say which plant or which fidelity produced it, and a
# coarse run is indistinguishable from a full-fidelity one.
SCHEMA_VERSION = 3
RESULTS_FILENAME = "optimization_results.npz"
METHODS = ("collocated", "synergistic")
GAIN_NAMES = ("Kp", "Ki", "Kd")


def best_batch_from_loss(
    history_loss: np.ndarray, finite_mask: np.ndarray | None = None
) -> int:
    """Return the start that attained the lowest loss over all iterations.

    This is the single definition of "best batch". Issue #128 arose from the
    plotter deriving the index from one method's losses and reusing it for the
    other, so callers must apply this to each archive separately.

    Args:
        history_loss: Losses of shape ``(iterations, B)``.
        finite_mask: Optional boolean mask of the same shape. ``False`` entries
            are excluded from the comparison.

    Returns:
        Index into the batch axis.
    """
    loss = np.asarray(history_loss, dtype=float)
    if finite_mask is not None:
        loss = np.where(np.asarray(finite_mask, dtype=bool), loss, np.inf)
    return int(np.argmin(np.min(loss, axis=0)))


def best_iteration_per_batch(
    history_loss: np.ndarray, finite_mask: np.ndarray | None = None
) -> np.ndarray:
    """Return each start's own lowest-loss iteration, as ``(B,)``."""
    loss = np.asarray(history_loss, dtype=float)
    if finite_mask is not None:
        loss = np.where(np.asarray(finite_mask, dtype=bool), loss, np.inf)
    return np.argmin(loss, axis=0).astype(np.int64)


def improvement_over_initial_median(
    history_loss: np.ndarray, finite_mask: np.ndarray | None = None
) -> float:
    """Return the loss reduction the paper reports for Section Vd.

    Recovered from the legacy archives, where it reproduces the published
    62% / 57% exactly: the best final loss against the median loss of the starts
    at iteration zero.

    Args:
        history_loss: Losses of shape ``(iterations, B)``.
        finite_mask: Optional boolean mask of the same shape.

    Returns:
        Fractional improvement, e.g. ``0.6262`` for 62.62%.
    """
    loss = np.asarray(history_loss, dtype=float)
    if finite_mask is not None:
        loss = np.where(np.asarray(finite_mask, dtype=bool), loss, np.inf)
    return float(1.0 - np.min(loss[-1]) / np.median(loss[0]))


def _stack_gain_history(
    opt_vars_history: list[dict[str, Any]], gain: str
) -> np.ndarray:
    """Stack one gain across iterations, keeping the batch axis the loop produced."""
    return np.stack(
        [np.asarray(item["opt_ctr_params"][gain]) for item in opt_vars_history], axis=0
    )


def save_optimization_outputs(
    result_dir: Path,
    *,
    method: str,
    batch_size: int,
    history_loss: Any,
    history_time: Any,
    history_finite_mask: Any,
    history_grad_norm: Any,
    history_update_norm: Any,
    opt_vars_history: list[dict[str, Any]],
    init_gains: dict[str, Any],
    init_seed: int,
    init_scheme: str,
    init_spread: float,
    objective_name: str,
    objective_scales: Any,
    saturation_config: str,
    optimizer_metadata: str,
    solver_dt: float,
    material_damping_coefficient: float,
    t_ts: Any,
    q_ts_init: Any,
    q_ts_best: Any,
    qd_ts_init: Any,
    qd_ts_best: Any,
    u_ts_init: Any,
    u_ts_best: Any,
    x_ts_init: Any,
    x_ts_best: Any,
    x_des_ts: Any,
    best_iteration: Any,
    best_batch: int,
    q_des_ts: Any | None = None,
    is_placeholder: bool = False,
) -> Path:
    """Write and validate the one supported archive format.

    Trajectory arguments carry a leading batch axis of width ``batch_size``;
    ``t_ts`` and the references do not.

    Args:
        result_dir: Directory to write ``optimization_results.npz`` into.
        method: One of :data:`METHODS`.
        batch_size: Number of independently optimized starts ``B``.
        history_loss: Losses of shape ``(iterations, B)``.
        history_time: Wall-clock seconds per iteration, shape ``(iterations,)``.
        history_finite_mask: Boolean validity of shape ``(iterations, B)``.
        history_grad_norm: Per-start gradient L2 norms, ``(iterations, B)``.
        history_update_norm: Per-start update L2 norms, ``(iterations, B)``.
        opt_vars_history: Per-iteration optimization variables, whose
            ``opt_ctr_params`` leaves carry the batch axis.
        init_gains: The sampled initializations, each of shape ``(B, m)``.
        init_seed: PRNG seed behind ``init_gains``.
        init_scheme: Name of the sampling scheme behind ``init_gains``.
        init_spread: Scheme spread parameter.
        objective_name: Identifier of the loss definition the history was
            measured with.
        objective_scales: Per-component factors that made the error
            dimensionless, shape ``(k,)``. Storing them lets the loss be
            recomputed from the archive alone.
        saturation_config: Human-readable integral-error saturation setting.
        optimizer_metadata: Human-readable optimizer transforms and rates.
        t_ts: Shared time grid of shape ``(timesteps,)``.
        q_ts_init: Per-start initial configuration rollouts.
        q_ts_best: Per-start best configuration rollouts.
        qd_ts_init: Per-start initial velocity rollouts.
        qd_ts_best: Per-start best velocity rollouts.
        u_ts_init: Per-start initial control inputs.
        u_ts_best: Per-start best control inputs.
        x_ts_init: Per-start initial end-effector poses.
        x_ts_best: Per-start best end-effector poses.
        x_des_ts: Shared reference poses of shape ``(timesteps, 6)``.
        best_iteration: Each start's best iteration, shape ``(B,)``.
        best_batch: Index of the best start, from :func:`best_batch_from_loss`.
        q_des_ts: Shared configuration reference; required for collocated.
        is_placeholder: Mark the archive as development-only.

    Returns:
        The path written.

    Raises:
        ValueError: If ``method`` is unknown or the arrays fail validation.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown Section Vd method {method!r}")
    arr = np.asarray
    completed_iterations = len(history_loss)
    result_data: dict[str, np.ndarray] = {
        "schema_version": arr(SCHEMA_VERSION, dtype=np.int64),
        "method": arr(method),
        "rotation_representation": arr("rotation_vector"),
        "is_placeholder": arr(bool(is_placeholder)),
        "completed_iterations": arr(completed_iterations, dtype=np.int64),
        "batch_size": arr(int(batch_size), dtype=np.int64),
        "history_loss": arr(history_loss),
        "history_time": arr(history_time),
        "history_finite_mask": arr(history_finite_mask, dtype=bool),
        "solver_dt": arr(float(solver_dt)),
        "material_damping_coefficient": arr(float(material_damping_coefficient)),
        "history_grad_norm": arr(history_grad_norm),
        "history_update_norm": arr(history_update_norm),
        "init_seed": arr(int(init_seed), dtype=np.int64),
        "init_scheme": arr(str(init_scheme)),
        "init_spread": arr(float(init_spread)),
        "objective_name": arr(str(objective_name)),
        "objective_scales": arr(objective_scales, dtype=float),
        "saturation_config": arr(str(saturation_config)),
        "optimizer_metadata": arr(str(optimizer_metadata)),
        "t_ts": arr(t_ts),
        "q_ts_init": arr(q_ts_init),
        "q_ts_best": arr(q_ts_best),
        "qd_ts_init": arr(qd_ts_init),
        "qd_ts_best": arr(qd_ts_best),
        "u_ts_init": arr(u_ts_init),
        "u_ts_best": arr(u_ts_best),
        "x_ts_init": arr(x_ts_init),
        "x_ts_best": arr(x_ts_best),
        "x_des_ts": arr(x_des_ts),
        "best_iteration": arr(best_iteration, dtype=np.int64),
        "best_batch": arr(int(best_batch), dtype=np.int64),
    }
    for gain in GAIN_NAMES:
        result_data[f"history_{gain}"] = _stack_gain_history(opt_vars_history, gain)
        result_data[f"init_{gain}"] = arr(init_gains[gain])
    if q_des_ts is not None:
        result_data["q_des_ts"] = arr(q_des_ts)
    validate_results(result_data, expected_method=method)
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / RESULTS_FILENAME
    np.savez_compressed(path, **result_data)
    print(f"Saved {path}")
    return path


def load_results(
    path_or_directory: Path, *, expected_method: str | None = None
) -> dict[str, np.ndarray]:
    """Load and validate one archive.

    Args:
        path_or_directory: The archive, or a directory containing it.
        expected_method: Reject an archive from the other controller.

    Returns:
        The archive's arrays.

    Raises:
        FileNotFoundError: If the archive is missing.
        ValueError: If it does not match the canonical schema.
    """
    path = Path(path_or_directory)
    if path.is_dir():
        path = path / RESULTS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    validate_results(data, expected_method=expected_method)
    return data


def _scalar_string(data: dict[str, np.ndarray], key: str) -> str:
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"{key} must be a scalar string, got {value.shape}")
    return str(value.item())


_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "method",
        "rotation_representation",
        "is_placeholder",
        "completed_iterations",
        "batch_size",
        "history_loss",
        "history_time",
        "history_finite_mask",
        "history_grad_norm",
        "history_update_norm",
        "init_seed",
        "init_scheme",
        "init_spread",
        "objective_name",
        "objective_scales",
        "saturation_config",
        "optimizer_metadata",
        "solver_dt",
        "material_damping_coefficient",
        "t_ts",
        "q_ts_init",
        "q_ts_best",
        "qd_ts_init",
        "qd_ts_best",
        "u_ts_init",
        "u_ts_best",
        "x_ts_init",
        "x_ts_best",
        "x_des_ts",
        "best_iteration",
        "best_batch",
    }
    | {f"history_{gain}" for gain in GAIN_NAMES}
    | {f"init_{gain}" for gain in GAIN_NAMES}
)


def validate_results(
    data: dict[str, np.ndarray], *, expected_method: str | None = None
) -> None:
    """Reject anything other than the canonical schema, including old archives.

    Args:
        data: Archive contents.
        expected_method: Reject an archive from the other controller.

    Raises:
        ValueError: If the archive is not the current schema version or is
            internally inconsistent.
    """
    missing = _REQUIRED_FIELDS.difference(data)
    if missing:
        raise ValueError(
            "Unsupported Section Vd archive (schema-v1 and legacy MAT archives "
            f"are not supported); missing fields: {sorted(missing)}"
        )
    version = np.asarray(data["schema_version"])
    if version.shape != () or int(version) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Section Vd schema_version {version!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    method = _scalar_string(data, "method")
    if method not in METHODS or (
        expected_method is not None and method != expected_method
    ):
        raise ValueError(f"Archive method is {method!r}, expected {expected_method!r}")
    if _scalar_string(data, "rotation_representation") != "rotation_vector":
        raise ValueError("Section Vd poses must use rotation_vector orientation")

    loss = np.asarray(data["history_loss"])
    if loss.ndim != 2 or loss.shape[0] < 1 or loss.shape[1] < 1:
        raise ValueError(
            f"history_loss must have shape (iterations, B), got {loss.shape}"
        )
    iterations, batch = loss.shape
    declared_batch = np.asarray(data["batch_size"])
    if declared_batch.shape != () or int(declared_batch) != batch:
        raise ValueError(
            f"batch_size must equal the loss batch axis; got {declared_batch!r} "
            f"and {batch}"
        )
    completed = np.asarray(data["completed_iterations"])
    if completed.shape != () or int(completed) != iterations:
        raise ValueError(
            "completed_iterations must equal the history length; "
            f"got {completed!r} and {iterations}"
        )

    t = np.asarray(data["t_ts"])
    if t.ndim != 1 or t.shape[0] < 1:
        raise ValueError(f"t_ts must have shape (timesteps,), got {t.shape}")
    timesteps = t.shape[0]
    q_init = np.asarray(data["q_ts_init"])
    if q_init.ndim != 3 or q_init.shape[:2] != (batch, timesteps):
        raise ValueError(
            f"q_ts_init must have shape (B, timesteps, dofs), got {q_init.shape}"
        )
    dofs = q_init.shape[2]
    u_init = np.asarray(data["u_ts_init"])
    if u_init.ndim != 3 or u_init.shape[:2] != (batch, timesteps):
        raise ValueError(
            f"u_ts_init must have shape (B, timesteps, actuators), got {u_init.shape}"
        )
    actuators = u_init.shape[2]

    exact_shapes = {
        "history_time": (iterations,),
        "history_finite_mask": (iterations, batch),
        "history_grad_norm": (iterations, batch),
        "history_update_norm": (iterations, batch),
        "q_ts_best": (batch, timesteps, dofs),
        "qd_ts_init": (batch, timesteps, dofs),
        "qd_ts_best": (batch, timesteps, dofs),
        "u_ts_best": (batch, timesteps, actuators),
        "x_ts_init": (batch, timesteps, 6),
        "x_ts_best": (batch, timesteps, 6),
        "x_des_ts": (timesteps, 6),
        "best_iteration": (batch,),
    }
    for key, expected in exact_shapes.items():
        actual = np.asarray(data[key]).shape
        if actual != expected:
            raise ValueError(f"{key} must have shape {expected}, got {actual}")

    for gain in GAIN_NAMES:
        history = np.asarray(data[f"history_{gain}"])
        if history.ndim != 3 or history.shape[:2] != (iterations, batch):
            raise ValueError(
                f"history_{gain} must have shape (iterations, B, m), "
                f"got {history.shape}"
            )
        initial = np.asarray(data[f"init_{gain}"])
        if initial.shape != (batch, history.shape[2]):
            raise ValueError(f"init_{gain} must have shape (B, m), got {initial.shape}")

    if method == "collocated":
        if "q_des_ts" not in data:
            raise ValueError("Collocated archives require q_des_ts")
        if np.asarray(data["q_des_ts"]).shape != (timesteps, dofs):
            raise ValueError("q_des_ts must have shape (timesteps, dofs)")

    scales = np.asarray(data["objective_scales"])
    if scales.ndim != 1 or scales.size < 1:
        raise ValueError(f"objective_scales must have shape (k,), got {scales.shape}")

    mask = np.asarray(data["history_finite_mask"], dtype=bool)
    if not bool(np.all(np.any(mask, axis=0))):
        dead = [int(i) for i in np.flatnonzero(~np.any(mask, axis=0))]
        raise ValueError(
            f"Starts {dead} never produced a finite iterate; rerun with a "
            "different --init-seed or a narrower --init-spread"
        )

    # Frozen entries are allowed to be non-finite; nothing else is.
    for key, value in data.items():
        array = np.asarray(value)
        if array.dtype.kind not in "fc":
            continue
        if key in ("history_loss",) or array.shape[:2] == (iterations, batch):
            if not np.all(np.isfinite(array[mask])):
                raise ValueError(f"{key} has non-finite values at valid iterations")
        elif not np.all(np.isfinite(array)):
            raise ValueError(f"Section Vd archive has non-finite values in {key}")

    # Selection is supplied by the loop and re-derived here, so the two cannot
    # silently disagree.
    stored_best_iteration = np.asarray(data["best_iteration"])
    expected_best_iteration = best_iteration_per_batch(loss, mask)
    if not np.array_equal(stored_best_iteration, expected_best_iteration):
        raise ValueError(
            f"best_iteration {stored_best_iteration.tolist()} does not match the "
            f"loss history {expected_best_iteration.tolist()}"
        )
    stored_best_batch = np.asarray(data["best_batch"])
    if stored_best_batch.shape != () or int(stored_best_batch) != best_batch_from_loss(
        loss, mask
    ):
        raise ValueError(
            f"best_batch {stored_best_batch!r} does not match the loss history"
        )
