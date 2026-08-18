"""Canonical Section Vd result schema, validation, and serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
RESULTS_FILENAME = "optimization_results.npz"
METHODS = ("collocated", "synergistic")


def _stack_gain_history(
    opt_vars_history: list[dict[str, Any]], gain: str
) -> np.ndarray:
    return np.stack(
        [np.asarray(item["opt_ctr_params"][gain]) for item in opt_vars_history], axis=0
    )[:, None, ...]


def save_optimization_outputs(
    result_dir: Path,
    *,
    method: str,
    history_loss: Any,
    history_time: Any,
    opt_vars_history: list[dict[str, Any]],
    history_qd_ts: Any,
    history_u_ts: Any,
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
    best_iteration: int,
    q_des_ts: Any | None = None,
    is_placeholder: bool = False,
) -> Path:
    """Write and validate the one supported, explicitly batched archive format."""
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
        "history_loss": arr(history_loss)[:, None],
        "history_time": arr(history_time)[:, None],
        "history_Kp": _stack_gain_history(opt_vars_history, "Kp"),
        "history_Ki": _stack_gain_history(opt_vars_history, "Ki"),
        "history_Kd": _stack_gain_history(opt_vars_history, "Kd"),
        "history_qd_ts": arr(history_qd_ts)[:, None, ...],
        "history_u_ts": arr(history_u_ts)[:, None, ...],
        "t_ts": arr(t_ts)[None, ...],
        "q_ts_init": arr(q_ts_init)[None, ...],
        "q_ts_best": arr(q_ts_best)[None, ...],
        "qd_ts_init": arr(qd_ts_init)[None, ...],
        "qd_ts_best": arr(qd_ts_best)[None, ...],
        "u_ts_init": arr(u_ts_init)[None, ...],
        "u_ts_best": arr(u_ts_best)[None, ...],
        "x_ts_init": arr(x_ts_init)[None, ...],
        "x_ts_best": arr(x_ts_best)[None, ...],
        "x_des_ts": arr(x_des_ts)[None, ...],
        "best_iteration": arr([best_iteration], dtype=np.int64),
    }
    if q_des_ts is not None:
        result_data["q_des_ts"] = arr(q_des_ts)[None, ...]
    validate_results(result_data, expected_method=method)
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / RESULTS_FILENAME
    np.savez_compressed(path, **result_data)
    print(f"Saved {path}")
    return path


def load_results(
    path_or_directory: Path, *, expected_method: str | None = None
) -> dict[str, np.ndarray]:
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


def validate_results(
    data: dict[str, np.ndarray], *, expected_method: str | None = None
) -> None:
    """Reject anything other than the canonical schema, including old archives."""
    required = {
        "schema_version",
        "method",
        "rotation_representation",
        "is_placeholder",
        "completed_iterations",
        "history_loss",
        "history_time",
        "history_Kp",
        "history_Ki",
        "history_Kd",
        "history_qd_ts",
        "history_u_ts",
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
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(
            "Unsupported Section Vd archive (legacy archives are not supported); "
            f"missing fields: {sorted(missing)}"
        )
    version = np.asarray(data["schema_version"])
    if version.shape != () or int(version) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Section Vd schema_version {version!r}; expected 1"
        )
    method = _scalar_string(data, "method")
    if method not in METHODS or (
        expected_method is not None and method != expected_method
    ):
        raise ValueError(f"Archive method is {method!r}, expected {expected_method!r}")
    if _scalar_string(data, "rotation_representation") != "rotation_vector":
        raise ValueError("Section Vd poses must use rotation_vector orientation")

    loss = np.asarray(data["history_loss"])
    if loss.ndim != 2 or loss.shape[1] != 1 or loss.shape[0] < 1:
        raise ValueError(
            f"history_loss must have shape (iterations, 1), got {loss.shape}"
        )
    iterations = loss.shape[0]
    completed = np.asarray(data["completed_iterations"])
    if completed.shape != () or int(completed) != iterations:
        raise ValueError(
            "completed_iterations must equal the history length; "
            f"got {completed!r} and {iterations}"
        )
    t = np.asarray(data["t_ts"])
    if t.ndim != 2 or t.shape[0] != 1:
        raise ValueError(f"t_ts must have shape (1, timesteps), got {t.shape}")
    timesteps = t.shape[1]
    q_init = np.asarray(data["q_ts_init"])
    if q_init.ndim != 3 or q_init.shape[:2] != (1, timesteps):
        raise ValueError(
            f"q_ts_init must have shape (1, timesteps, dofs), got {q_init.shape}"
        )
    dofs = q_init.shape[2]
    u_init = np.asarray(data["u_ts_init"])
    if u_init.ndim != 3 or u_init.shape[:2] != (1, timesteps):
        raise ValueError(
            f"u_ts_init must have shape (1, timesteps, actuators), got {u_init.shape}"
        )
    actuators = u_init.shape[2]
    exact_shapes = {
        "history_time": (iterations, 1),
        "history_qd_ts": (iterations, 1, timesteps, dofs),
        "history_u_ts": (iterations, 1, timesteps, actuators),
        "q_ts_best": (1, timesteps, dofs),
        "qd_ts_init": (1, timesteps, dofs),
        "qd_ts_best": (1, timesteps, dofs),
        "u_ts_best": (1, timesteps, actuators),
        "x_ts_init": (1, timesteps, 6),
        "x_ts_best": (1, timesteps, 6),
        "x_des_ts": (1, timesteps, 6),
        "best_iteration": (1,),
    }
    for key, expected in exact_shapes.items():
        actual = np.asarray(data[key]).shape
        if actual != expected:
            raise ValueError(f"{key} must have shape {expected}, got {actual}")
    for key in ("history_Kp", "history_Ki", "history_Kd"):
        value = np.asarray(data[key])
        if value.ndim < 3 or value.shape[:2] != (iterations, 1):
            raise ValueError(
                f"{key} must begin with (iterations, 1), got {value.shape}"
            )
    if method == "collocated":
        if "q_des_ts" not in data:
            raise ValueError("Collocated archives require q_des_ts")
        if np.asarray(data["q_des_ts"]).shape != (1, timesteps, dofs):
            raise ValueError("q_des_ts must have shape (1, timesteps, dofs)")
    best = int(np.asarray(data["best_iteration"])[0])
    if not 0 <= best < iterations:
        raise ValueError(f"best_iteration {best} is outside [0, {iterations})")
    numeric = [
        np.asarray(v) for v in data.values() if np.asarray(v).dtype.kind in "biufc"
    ]
    if any(not np.all(np.isfinite(value)) for value in numeric):
        raise ValueError("Section Vd archive contains non-finite numeric values")
