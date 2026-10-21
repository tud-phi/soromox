"""Internal scalar and vector validation for editable rendering settings."""

import numpy as np

RGB = tuple[float, float, float]
Vector3 = tuple[float, float, float]


def _validate_vector(value, name, *, color=False, nonzero=False):
    """Validate a finite vector, optionally restricting its range or length.

    Args:
        value: Three numeric components.
        name: Field name used in errors.
        color: Require sRGB components in [0, 1].
        nonzero: Require a nonzero vector.

    Returns:
        None.

    Raises:
        ValueError: The vector violates a requested constraint.
    """
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite three-vector")
    if color and ((a < 0).any() or (a > 1).any()):
        raise ValueError(f"{name} must lie in [0, 1]")
    if nonzero and np.linalg.norm(a) == 0:
        raise ValueError(f"{name} must be nonzero")


def _nonnegative(value, name):
    """Validate a finite nonnegative scalar.

    Args:
        value: Numeric scalar.
        name: Field name used in errors.

    Returns:
        None.

    Raises:
        ValueError: The value is negative or nonfinite.
    """
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
