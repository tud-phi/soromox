__all__ = ["PCSStructure", "PlanarPCSStructure", "ISupportStructure"]

from typing import Any

import equinox as eqx
from jax import Array


def _tolist(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _int_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    value = _tolist(value)
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        return (int(value),)
    return tuple(int(entry) for entry in value)


def _bool_tuple(value: Any, *, name: str) -> tuple[bool, ...]:
    value = _tolist(value)
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        return (bool(value),)
    return tuple(bool(entry) for entry in value)


def _pcs_segment_counts(
    value: Any,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    return _int_tuple(value, name="pcs_segment_counts")


def _rigid_connector_selector(value: Any) -> tuple[bool, ...] | None:
    if value is None:
        return None
    return _bool_tuple(value, name="rigid_connector_selector")


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

    ``pcs_segment_counts`` defines how many constant-strain PCS segments are
    used for each physical pneumatic segment. A scalar count applies the same
    count to every pneumatic segment. The actual PCS segment lengths live in
    ``ISupportParams.pcs_segment_lengths``.

    ``rigid_connector_selector`` has one entry more than the number of
    pneumatic segments, with ``True`` marking a rigid connector PCS segment in
    the fixed physical sequence:

    ``rigid_connector[0]``,
    ``pneumatic_segment[0]``,
    ``rigid_connector[1]``,
    ...,
    ``pneumatic_segment[n - 1]``,
    ``rigid_connector[n]``.

    A scalar connector selector applies to all connector slots. The actual
    connector lengths live in ``ISupportParams.rigid_connector_lengths``. If
    ``strain_selector`` is provided, it is interpreted on the expanded PCS
    segment layout; rigid connector strains are always deactivated.
    """

    pcs_segment_counts: tuple[int, ...] | None = eqx.field(
        static=True, default=None, converter=_pcs_segment_counts
    )
    rigid_connector_selector: tuple[bool, ...] | None = eqx.field(
        static=True, default=None, converter=_rigid_connector_selector
    )
