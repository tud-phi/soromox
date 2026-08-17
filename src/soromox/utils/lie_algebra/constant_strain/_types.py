"""Result types shared by constant-strain Lie-algebra implementations."""

from typing import NamedTuple

from jax import Array


class ConstantStrainOperators(NamedTuple):
    """Related constant-strain Lie operators evaluated as one shared bundle.

    Attributes:
        adjoint: Forward adjoint ``exp(s * ad_xi)``.
        adjoint_inverse: Inverse of ``adjoint`` assembled from Lie-group
            blocks rather than a dense matrix inverse.
        tangent: Integrated adjoint
            ``integral_0^s exp(sigma * ad_xi) d sigma``.
        tangent_derivative: Analytic directional derivative
            ``D_xi tangent[xid]``. This is ``None`` when the bundle was
            requested without ``xid``.
    """

    adjoint: Array
    adjoint_inverse: Array
    tangent: Array
    tangent_derivative: Array | None
