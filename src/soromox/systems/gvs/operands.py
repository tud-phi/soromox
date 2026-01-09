__all__ = [
    "JointOperand",
    "GeometricOperand",
]

from dataclasses import dataclass

import jax
from jax import Array


@jax.tree_util.register_pytree_node_class
@dataclass
class JointOperand:
    """
    Compact container of joint parameters used to build joint basis matrices.

    Attributes
    ----------
    axis_idx : int
        Joint axis index for axis-based joints. Mapping: 0:'x', 1:'y', 2:'z'.
        Used by revolute, prismatic, helical, cylindrical joints. Dimensionless.
    pitch : float
        Pitch for helical joints [m/rad]. Ignored for other joint types.
    plane_idx : int
        Joint plane index for planar joints. Mapping: 0:'xy', 1:'yz', 2:'xz'.
        Dimensionless; ignored for non-planar joints.
    """

    axis_idx: int
    pitch: float
    plane_idx: int

    def tree_flatten(self):
        children = (self.axis_idx, self.pitch, self.plane_idx)
        aux_data = None  # no static field to exclude here
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass
class GeometricOperand:
    """
    Per-segment geometric operands evaluated along the normalized arc coordinate.

    Attributes
    ----------
    Xs : Array
        Gauss nodes on the normalized interval [0, 1]; shape (nip,).
    r_params : Tuple[Array, Array]
        Circular section radius parameters (r_i, r_f) [m]. Each element is a scalar Array.
    h_params : Tuple[Array, Array]
        Rectangular section height parameters (h_i, h_f) [m]. Scalars.
    w_params : Tuple[Array, Array]
        Rectangular section width parameters (w_i, w_f) [m]. Scalars.
    a_params : Tuple[Array, Array]
        Elliptical section semi-major axis parameters (a_i, a_f) [m]. Scalars.
    b_params : Tuple[Array, Array]
        Elliptical section semi-minor axis parameters (b_i, b_f) [m]. Scalars.

    Notes
    -----
    The geometric parameters are interpolated linearly along `Xs` to obtain the
    local cross-section A and second moments of area I_x, I_y, I_z.
    """

    Xs: Array
    r_params: tuple[Array, Array]
    h_params: tuple[Array, Array]
    w_params: tuple[Array, Array]
    a_params: tuple[Array, Array]
    b_params: tuple[Array, Array]

    def tree_flatten(self):
        children = (
            self.Xs,
            self.r_params,
            self.h_params,
            self.w_params,
            self.a_params,
            self.b_params,
        )
        aux_data = None  # no static field to exclude here
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
