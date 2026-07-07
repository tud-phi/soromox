__all__ = ["PCSStructure", "PlanarPCSStructure", "ISupportStructure"]

from collections.abc import Sequence
from typing import Any

import equinox as eqx
from jax import Array


def _tolist(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _is_scalar(value: Any) -> bool:
    value = _tolist(value)
    return not isinstance(value, Sequence) or isinstance(value, (str, bytes))


def _float_tuple(value: Any, *, name: str) -> tuple[float, ...]:
    value = _tolist(value)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of numbers.")
    return tuple(float(entry) for entry in value)


def _pcs_segment_lengths(
    value: Any,
) -> tuple[tuple[float, ...], ...] | None:
    if value is None:
        return None
    value = _tolist(value)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("pcs_segment_lengths must be a nested sequence of numbers.")
    if len(value) == 0:
        return ()
    if _is_scalar(value[0]):
        return (_float_tuple(value, name="pcs_segment_lengths"),)
    return tuple(
        _float_tuple(segment_lengths, name="pcs_segment_lengths")
        for segment_lengths in value
    )


def _rigid_connector_lengths(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    return _float_tuple(value, name="rigid_connector_lengths")


class PCSStructure(eqx.Module):
    """Static PCS layout that determines JAX compilation structure."""

    num_gauss_points: int = eqx.field(static=True, default=5)
    strain_selector: Array | None = None
    scale_rotational_basis_by_length: bool = eqx.field(static=True, default=False)


class PlanarPCSStructure(eqx.Module):
    """Static planar PCS layout."""

    num_gauss_points: int = eqx.field(static=True, default=5)
    strain_selector: Array | None = None
    scale_rotational_basis_by_length: bool = eqx.field(static=True, default=False)


class ISupportStructure(PCSStructure):
    """Static I-SUPPORT PCS layout.

    ``pcs_segment_lengths`` partitions each physical pneumatic segment into one
    or more constant-strain PCS segments. ``rigid_connector_lengths`` has one
    entry more than the number of pneumatic segments and follows the fixed
    physical sequence:

    ``rigid_connector[0]``,
    ``pneumatic_segment[0]``,
    ``rigid_connector[1]``,
    ...,
    ``pneumatic_segment[n - 1]``,
    ``rigid_connector[n]``.

    Zero-length rigid connectors are omitted from the internal PCS model. If
    ``strain_selector`` is provided, it is interpreted on the expanded PCS
    segment layout; rigid connector strains are always deactivated.
    """

    pcs_segment_lengths: tuple[tuple[float, ...], ...] | None = eqx.field(
        static=True, default=None, converter=_pcs_segment_lengths
    )
    rigid_connector_lengths: tuple[float, ...] | None = eqx.field(
        static=True, default=None, converter=_rigid_connector_lengths
    )
