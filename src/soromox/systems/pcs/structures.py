__all__ = [
    "PCSStructure",
    "PlanarPCSStructure",
    "PlanarHSAStructure",
    "ISupportStructure",
]

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


def _rigid_segment_selector(value: Any) -> tuple[bool, ...] | None:
    if value is None:
        return None
    value = _tolist(value)
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TypeError("rigid_segment_selector must be a boolean sequence.")
    return tuple(bool(entry) for entry in value)


class PCSStructure(eqx.Module):
    """Static spatial PCS layout that determines JAX compilation structure.

    Attributes:
        num_gauss_points: Number of Gauss points per constant-strain link.
        strain_selector: Optional boolean selector with six entries per link.
            ``None`` activates every strain coordinate.
        scale_rotational_basis_by_length: Whether rotational strain coordinates
            are normalized by their link length.
    """

    num_gauss_points: int = eqx.field(static=True, default=5)
    strain_selector: Array | None = None
    scale_rotational_basis_by_length: bool = eqx.field(static=True, default=False)


class PlanarPCSStructure(eqx.Module):
    """Static planar PCS layout.

    Attributes:
        num_gauss_points: Number of Gauss points per constant-strain link.
        strain_selector: Optional boolean selector with three entries per link.
            ``None`` activates every strain coordinate.
        scale_rotational_basis_by_length: Whether the rotational strain
            coordinate is normalized by its link length.
    """

    num_gauss_points: int = eqx.field(static=True, default=5)
    strain_selector: Array | None = None
    scale_rotational_basis_by_length: bool = eqx.field(static=True, default=False)


class PlanarHSAStructure(eqx.Module):
    """Static numerical layout choices for planar HSA PCS systems."""

    num_gauss_points: int = eqx.field(static=True, default=5)
    strain_selector: Array | None = None
    consider_underactuation: bool = eqx.field(static=True, default=True)
    consider_hysteresis: bool = eqx.field(static=True, default=False)
    eps: float = eqx.field(static=True, default=1e-6)


class ISupportStructure(PCSStructure):
    """Static I-SUPPORT PCS layout.

    ``pcs_segment_counts`` defines how many constant-strain PCS segments are
    used for each physical pneumatic segment, in pneumatic-segment order. A
    scalar count applies the same count to every pneumatic segment. Child
    lengths live in ``ISupportParams.pcs_segment_lengths``.

    ``rigid_segment_selector`` has one entry per physical segment in
    ``ISupportParams``. ``True`` marks a rigid segment and ``False`` a pneumatic
    segment. When omitted, segment types alternate from a rigid segment at index
    zero. If ``strain_selector`` is provided, it is interpreted on the expanded
    PCS layout; rigid-segment strains are always deactivated.

    Attributes:
        pcs_segment_counts: Optional number of PCS subdivisions for each
            pneumatic segment. A one-element tuple is broadcast.
        rigid_segment_selector: Optional physical-segment mask in which
            ``True`` denotes a rigid connector.
    """

    pcs_segment_counts: tuple[int, ...] | None = eqx.field(
        static=True, default=None, converter=_pcs_segment_counts
    )
    rigid_segment_selector: tuple[bool, ...] | None = eqx.field(
        static=True, default=None, converter=_rigid_segment_selector
    )
