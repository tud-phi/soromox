"""Finiteness diagnostics for JAX pytrees.

These helpers walk any pytree -- a gradient tree, a trajectory dictionary, a
parameter tree -- and name the leaves that went non-finite. Inspecting the
gradient as well as the trajectory distinguishes a diverging forward pass from
a forward pass that stays healthy while the reverse pass produces ``NaN``.

The helpers need concrete arrays, so call them outside ``jit`` -- typically once
per optimization iteration, where their cost is negligible.
"""

__all__ = [
    "nonfinite_leaves",
    "format_nonfinite_report",
    "assert_all_finite",
    "is_all_finite",
]

import jax
import jax.numpy as jnp


def _format_path(path) -> str:
    """Render a ``tree_flatten_with_path`` key path as a readable string.

    Args:
        path: Sequence of pytree key entries as produced by
            ``jax.tree_util.tree_flatten_with_path``.

    Returns:
        Dotted path string, for example ``opt_ctr_params.Kp``. An empty path
        renders as ``<root>``.
    """
    parts = [str(entry.key) if hasattr(entry, "key") else str(entry) for entry in path]
    return ".".join(parts) if parts else "<root>"


def nonfinite_leaves(tree) -> list[tuple[str, str]]:
    """Locate the non-finite leaves of a pytree.

    Args:
        tree: Any pytree whose array leaves are concrete, e.g. a gradient tree
            returned by ``jax.value_and_grad`` or an auxiliary output dict.

    Returns:
        List of ``(leaf_path, reason)`` pairs, where ``reason`` is ``"nan"``,
        ``"inf"`` or ``"nan+inf"``. The list is empty when every leaf is finite.
        Non-floating leaves (integers, booleans) are always finite and are
        skipped.

    Example:
        ```python
        import jax.numpy as jnp
        from soromox.utils import nonfinite_leaves

        nonfinite_leaves({"a": jnp.array([1.0, jnp.nan])})  # [("a", "nan")]
        ```
    """
    report: list[tuple[str, str]] = []

    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        array = jnp.asarray(leaf)
        if not jnp.issubdtype(array.dtype, jnp.inexact):
            continue

        has_nan = bool(jnp.isnan(array).any())
        has_inf = bool(jnp.isinf(array).any())
        if not (has_nan or has_inf):
            continue

        if has_nan and has_inf:
            reason = "nan+inf"
        elif has_nan:
            reason = "nan"
        else:
            reason = "inf"
        report.append((_format_path(path), reason))

    return report


def format_nonfinite_report(label: str, report: list[tuple[str, str]]) -> str:
    """Render a :func:`nonfinite_leaves` result as a single-line message.

    Args:
        label: Short description of what was inspected, e.g. ``"gradient"``.
        report: Pairs returned by :func:`nonfinite_leaves`.

    Returns:
        Human-readable summary. Returns a message stating that everything is
        finite when ``report`` is empty.
    """
    if not report:
        return f"{label}: all leaves finite"
    details = ", ".join(f"{path} ({reason})" for path, reason in report)
    return f"{label}: non-finite at {details}"


def assert_all_finite(tree, label: str) -> None:
    """Raise if any leaf of ``tree`` is non-finite.

    Args:
        tree: Pytree with concrete array leaves.
        label: Short description used in the error message, e.g. ``"gradient"``.

    Raises:
        FloatingPointError: If any leaf contains ``NaN`` or ``inf``. The message
            names every offending leaf path.
    """
    report = nonfinite_leaves(tree)
    if report:
        raise FloatingPointError(format_nonfinite_report(label, report))


def is_all_finite(tree) -> bool:
    """Return whether every leaf of ``tree`` is finite.

    Args:
        tree: Pytree with concrete array leaves.

    Returns:
        ``True`` when no leaf contains ``NaN`` or ``inf``.
    """
    return not nonfinite_leaves(tree)
