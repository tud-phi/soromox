"""Reproducible multi-start gain initialization for the Section Vd study."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax import Array


__all__ = [
    "GAIN_ORDER",
    "SECVD_BATCH_SIZE",
    "SECVD_INIT_SCHEME",
    "SECVD_INIT_SEED",
    "SECVD_INIT_SPREAD",
    "describe_initial_gains",
    "sample_initial_gains",
]

SECVD_BATCH_SIZE = 6

SECVD_INIT_SEED = 0
SECVD_INIT_SPREAD = 3.0
SECVD_INIT_SCHEME = "log_uniform_v1"
GAIN_ORDER = ("Kp", "Ki", "Kd")


def _validate(nominal: dict[str, Array], batch_size: int, spread: float) -> None:
    """Reject an initialization request that cannot produce finite gains.

    Args:
        nominal: Nominal gains keyed by :data:`GAIN_ORDER`.
        batch_size: Number of starts to produce.
        spread: Multiplicative half-width.

    Raises:
        ValueError: If any argument is outside its supported range.
    """
    import jax.numpy as jnp

    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if set(nominal) != set(GAIN_ORDER):
        raise ValueError(
            f"Nominal gains must be keyed by {GAIN_ORDER}, got {sorted(nominal)}"
        )
    if not spread > 1.0:
        raise ValueError(f"spread must be greater than 1.0, got {spread}")
    for name in GAIN_ORDER:
        value = jnp.asarray(nominal[name])
        if not bool(jnp.all(jnp.isfinite(value))):
            raise ValueError(f"Nominal {name} must be finite")
        if not bool(jnp.all(value > 0.0)):
            raise ValueError(f"Nominal {name} must be strictly positive")


def _sample_log_uniform(nominal, batch_size, seed, spread):
    """Scale each gain by one log-uniform factor per start."""
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(seed)
    gains: dict[str, Array] = {}
    for name in GAIN_ORDER:
        key, subkey = jax.random.split(key)
        value = jnp.asarray(nominal[name])
        if batch_size == 1:
            gains[name] = value[None, ...]
            continue
        half_width = jnp.log10(spread)
        factor = 10.0 ** jax.random.uniform(
            subkey,
            (batch_size - 1,) + (1,) * value.ndim,
            dtype=value.dtype,
            minval=-half_width,
            maxval=half_width,
        )
        gains[name] = jnp.concatenate([value[None, ...], value[None, ...] * factor])
    return gains


def sample_initial_gains(
    nominal: dict[str, Array],
    *,
    batch_size: int = SECVD_BATCH_SIZE,
    seed: int = SECVD_INIT_SEED,
    spread: float = SECVD_INIT_SPREAD,
) -> dict[str, Array]:
    """Build a batch of gain initializations whose first entry is the nominal set.

    Args:
        nominal: Nominal gains keyed by :data:`GAIN_ORDER`, each of shape ``(m,)``.
        batch_size: Number of starts ``B``. ``1`` returns the nominal alone.
        seed: PRNG seed. Recorded in the archive alongside the scheme name.
        spread: Multiplicative half-width.

    Returns:
        Gains keyed by :data:`GAIN_ORDER`, each of shape ``(B, m)``, with entry
        ``0`` equal to the corresponding nominal value.

    Raises:
        ValueError: If the request cannot produce finite gains.
    """
    _validate(nominal, batch_size, spread)
    return _sample_log_uniform(nominal, batch_size, seed, spread)


def describe_initial_gains(gains: dict[str, Array]) -> str:
    """Render the batch of initializations as the table issue #154 asks for.

    Args:
        gains: Output of :func:`sample_initial_gains`.

    Returns:
        A table with one row per start.
    """
    import jax.numpy as jnp

    columns = {
        name: [
            ", ".join(f"{float(v):.4f}" for v in jnp.asarray(gains[name])[index])
            for index in range(int(jnp.asarray(gains[GAIN_ORDER[0]]).shape[0]))
        ]
        for name in GAIN_ORDER
    }
    widths = {
        name: max(len(name), max(len(cell) for cell in cells))
        for name, cells in columns.items()
    }
    header = "  start  " + "  ".join(f"{name:>{widths[name]}s}" for name in GAIN_ORDER)
    rows = [header, "  " + "-" * (len(header) - 2)]
    for index in range(len(columns[GAIN_ORDER[0]])):
        cells = "  ".join(
            f"{columns[name][index]:>{widths[name]}s}" for name in GAIN_ORDER
        )
        rows.append(f"{index:>7d}" + (" *" if index == 0 else "  ") + cells)
    rows.append("  * nominal")
    return "\n".join(rows)
