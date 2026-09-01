"""Reproducible multi-start gain initialization for the Section Vd study.

The published Section Vd results came from six independently optimized gain
initializations. Issue #154 asked what those initializations were; they were
recovered in September 2026 from the original generator, archived under
[`../legacy/`](../legacy/). This module provides two schemes:

``log_uniform_v1``
    The default for new runs. Perturbs the caller's nominal gains
    multiplicatively, so it adapts to whatever controller it is applied to.

``legacy_mixed_v1``
    An exact reproduction of the recovered generator's ``generate_mixed_batch``,
    for regenerating or checking against the legacy results. Its sampling box is
    absolute and belongs to the legacy two-segment robot and its controller
    parametrization, so it is **not** appropriate for new work.

Batch entry ``0`` is always the caller's nominal gain set under both schemes,
which keeps a single-start run a strict subset of a batched one. The recovered
generator did the same.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax import Array

# JAX is imported inside the functions, not here. The Section Vd entrypoints read
# this module's constants to pick a device *before* JAX is imported, and JAX
# silently ignores JAX_PLATFORMS once it has been imported.

__all__ = [
    "GAIN_ORDER",
    "INIT_SCHEMES",
    "LEGACY_KEY_ORDER",
    "LEGACY_MIXED_BOX",
    "LEGACY_MIXED_NOMINAL",
    "LEGACY_MIXED_SEED",
    "SECVD_BATCH_SIZE",
    "SECVD_INIT_SCHEME",
    "SECVD_INIT_SEED",
    "SECVD_INIT_SPREAD",
    "describe_initial_gains",
    "sample_initial_gains",
]

# Number of independently optimized gain initializations, matching the batch
# dimension of the legacy archives.
SECVD_BATCH_SIZE = 6

SECVD_INIT_SEED = 0
SECVD_INIT_SPREAD = 3.0
SECVD_INIT_SCHEME = "log_uniform_v1"

# Iterated explicitly so that adding a fourth gain later cannot shift the keys
# consumed by the first three and silently resample them.
GAIN_ORDER = ("Kp", "Ki", "Kd")

# --- Recovered legacy initialization ---------------------------------------
# From legacy/soromox_synergistic.py: PRNGKey(15), nominal Kp/Ki/Kd = 40/20/2,
# and each non-nominal start drawn i.i.d. per component from an absolute box.
LEGACY_MIXED_SEED = 15
LEGACY_MIXED_NOMINAL = {"Kp": 40.0, "Ki": 20.0, "Kd": 2.0}
LEGACY_MIXED_BOX = {
    "Kp": (20.0, 60.0),
    "Ki": (10.0, 30.0),
    "Kd": (2.0, 3.0),
}

# Reproducibility-critical. The original called
# ``jax.random.split(PRNGKey(15), 3)`` and then handed the subkeys out in the
# order ``jax.tree_util.tree_flatten_with_path`` produced them -- which is
# *sorted* key order, "Kd_", "Ki_", "Kp_", not the natural Kp, Ki, Kd. Consuming
# them in GAIN_ORDER instead resamples every start.
LEGACY_KEY_ORDER = tuple(sorted(GAIN_ORDER))

INIT_SCHEMES = ("log_uniform_v1", "legacy_mixed_v1")


def _validate(
    nominal: dict[str, Array], batch_size: int, scheme: str, spread: float
) -> None:
    """Reject an initialization request that cannot produce finite gains.

    Args:
        nominal: Nominal gains keyed by :data:`GAIN_ORDER`.
        batch_size: Number of starts to produce.
        scheme: Name from :data:`INIT_SCHEMES`.
        spread: Multiplicative half-width for ``log_uniform_v1``.

    Raises:
        ValueError: If any argument is outside its supported range.
    """
    import jax.numpy as jnp

    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if scheme not in INIT_SCHEMES:
        raise ValueError(f"Unknown init scheme {scheme!r}; expected {INIT_SCHEMES}")
    if set(nominal) != set(GAIN_ORDER):
        raise ValueError(
            f"Nominal gains must be keyed by {GAIN_ORDER}, got {sorted(nominal)}"
        )
    if scheme == "log_uniform_v1" and not spread > 1.0:
        raise ValueError(f"spread must be greater than 1.0, got {spread}")
    for name in GAIN_ORDER:
        value = jnp.asarray(nominal[name])
        if not bool(jnp.all(jnp.isfinite(value))):
            raise ValueError(f"Nominal {name} must be finite")
        # log_uniform_v1 perturbs multiplicatively, so a zero or negative
        # nominal gain has no well-defined neighbourhood to sample from.
        # legacy_mixed_v1 draws from an absolute box and does not care.
        if scheme == "log_uniform_v1" and not bool(jnp.all(value > 0.0)):
            raise ValueError(
                f"Nominal {name} must be strictly positive for log_uniform_v1"
            )


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
        # One scalar per start, broadcast across actuators, so that each start
        # is fully described by three numbers.
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


def _sample_legacy_mixed(nominal, batch_size, seed):
    """Reproduce ``generate_mixed_batch`` from the recovered generator.

    Draws are i.i.d. **per component** from :data:`LEGACY_MIXED_BOX` and are
    independent of ``nominal``; only start ``0`` uses the nominal. Subkeys are
    consumed in :data:`LEGACY_KEY_ORDER`, and no ``dtype`` is passed to
    ``jax.random.uniform`` -- both matching the original exactly.
    """
    import jax
    import jax.numpy as jnp

    # The original ran with jax_enable_x64. jax.random.uniform consumes a
    # different number of PRNG bits per element in single precision, so a
    # float32 run silently returns a different -- and wrong -- batch.
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "legacy_mixed_v1 reproduces the recovered Section Vd initialization "
            "only in double precision; call "
            'jax.config.update("jax_enable_x64", True) first'
        )

    keys = jax.random.split(jax.random.PRNGKey(seed), len(LEGACY_KEY_ORDER))
    subkey_for = dict(zip(LEGACY_KEY_ORDER, keys))
    gains: dict[str, Array] = {}
    for name in GAIN_ORDER:
        value = jnp.asarray(nominal[name])
        if batch_size == 1:
            gains[name] = value[None, ...]
            continue
        low, high = LEGACY_MIXED_BOX[name]
        per_start = jax.random.split(subkey_for[name], batch_size - 1)
        sampled = jax.vmap(
            lambda k, shape=value.shape, low=low, high=high: jax.random.uniform(
                k, shape=shape, minval=low, maxval=high
            )
        )(per_start)
        gains[name] = jnp.concatenate([value[None, ...], sampled])
    return gains


def sample_initial_gains(
    nominal: dict[str, Array],
    *,
    batch_size: int = SECVD_BATCH_SIZE,
    seed: int = SECVD_INIT_SEED,
    scheme: str = SECVD_INIT_SCHEME,
    spread: float = SECVD_INIT_SPREAD,
) -> dict[str, Array]:
    """Build a batch of gain initializations whose first entry is the nominal set.

    ``log_uniform_v1`` scales the whole nominal vector for gain ``g`` by a factor
    drawn log-uniformly from ``[1 / spread, spread]``, which treats gains as the
    scale parameters they are and preserves any anisotropy already present in the
    nominal.

    ``legacy_mixed_v1`` reproduces the recovered generator bit-for-bit. To
    regenerate the legacy table, pass :data:`LEGACY_MIXED_NOMINAL` (broadcast to
    the actuator count) with :data:`LEGACY_MIXED_SEED`.

    Args:
        nominal: Nominal gains keyed by :data:`GAIN_ORDER`, each of shape ``(m,)``.
        batch_size: Number of starts ``B``. ``1`` returns the nominal alone.
        seed: PRNG seed. Recorded in the archive alongside the scheme.
        scheme: Name from :data:`INIT_SCHEMES`.
        spread: Multiplicative half-width for ``log_uniform_v1``; unused otherwise.

    Returns:
        Gains keyed by :data:`GAIN_ORDER`, each of shape ``(B, m)``, with entry
        ``0`` equal to the corresponding nominal value.

    Raises:
        ValueError: If the request cannot produce finite gains.
    """
    _validate(nominal, batch_size, scheme, spread)
    if scheme == "log_uniform_v1":
        return _sample_log_uniform(nominal, batch_size, seed, spread)
    return _sample_legacy_mixed(nominal, batch_size, seed)


def describe_initial_gains(gains: dict[str, Array]) -> str:
    """Render the batch of initializations as the table issue #154 asks for.

    Every component is shown. ``log_uniform_v1`` makes them identical within a
    gain by construction, but ``legacy_mixed_v1`` samples per component, and
    printing only the first would hide most of each start.

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
