__all__ = ["Basis", "Joint", "Link"]
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

from soromox.systems.gvs.operands import GeometricOperand, JointOperand
from soromox.systems.gvs.strain_bases import (
    dof_Chebychev,
    dof_Fourier,
    dof_Gaussian,
    dof_IMQ,
    dof_LegendrePolynomial,
    dof_Monomial,
)


class Basis:
    """
    Utilities for strain basis handling in the GVS model.

    Provides mappings from human-readable basis names to indices, references to
    functions that compute the number of link DOFs from `(Bdof, Bodr)`, and a
    helper to build vmap'ed basis evaluators padded to `max_dof`.

    Attributes
    ----------
    BASISTYPE_MAP : Dict[str, int]
        Mapping of basis type name to integer index used in `lax.switch`.
    DOF_BRANCHES : List[Callable]
        Branch functions returning the number of active link DOFs given
        (`Bdof`, `Bodr`). The selected branch depends on `BASISTYPE_MAP`.

    Notes
    -----
    - Basis evaluators produce matrices of shape (6, max_dof) at a single `x in [0,1]`.
      After vectorization over `integration_points` they yield (num_integration_points, 6, max_dof).
    - Basis columns map generalized coordinates to the 6D Cosserat strain twist per
      unit length [rad/m, rad/m, rad/m, 1/m, 1/m, 1/m].
    """

    BASISTYPE_MAP = {
        "monomial": 0,
        "legendre": 1,
        "chebyshev": 2,
        "fourier": 3,
        "gaussian": 4,
        "imq": 5,
    }

    DOF_BRANCHES = [
        lambda operand: dof_Monomial(operand[0], operand[1]),
        lambda operand: dof_LegendrePolynomial(operand[0], operand[1]),
        lambda operand: dof_Chebychev(operand[0], operand[1]),
        lambda operand: dof_Fourier(operand[0], operand[1]),
        lambda operand: dof_Gaussian(operand[0], operand[1]),
        lambda operand: dof_IMQ(operand[0], operand[1]),
    ]

    @staticmethod
    def make_B_branch(B_fn, max_dof):
        def padded_branch(operand):
            points, Bdof, Bodr = operand

            def apply_fn(x):
                return B_fn(x, Bdof, Bodr, max_dof)

            return jax.vmap(apply_fn)(points)

        return padded_branch


class Joint:
    """
    Utilities and maps for joint basis construction and metadata.

    Attributes
    ----------
    JOINTTYPE_MAP : Dict[str, int]
        Joint type to index mapping used in `lax.switch`.
    AXIS_MAP : Dict[str, int]
        Axis name to index mapping: 'x'->0, 'y'->1, 'z'->2.
    PLANE_MAP : Dict[str, int]
        Plane name to index mapping: 'xy'->0, 'yz'->1, 'xz'->2.
    DICT_JOINT_TYPE_DOF : Dict[str, int]
        Number of DOFs per joint type, used for padding/truncation.

    Methods
    -------
    make_B_branch(B_fn, dof_joint, max_dof)
        Wrap a joint basis generator so its (6, dof_joint) output is padded or
        truncated to (6, max_dof). Columns map joint coordinates to 6D twist.
    """

    JOINTTYPE_MAP = {
        "revolute": 0,
        "prismatic": 1,
        "helical": 2,
        "cylindrical": 3,
        "planar": 4,
        "spherical": 5,
        "free": 6,
        "fixed": 7,
    }
    AXIS_MAP = {
        "x": 0,  # x-axis
        "y": 1,  # y-axis
        "z": 2,  # z-axis
    }
    PLANE_MAP = {
        "xy": 0,  # xy-plane
        "yz": 1,  # yz-plane
        "xz": 2,  # xz-plane
    }
    DICT_JOINT_TYPE_DOF = {
        "revolute": 1,
        "prismatic": 1,
        "helical": 1,
        "cylindrical": 2,
        "planar": 3,
        "spherical": 3,
        "free": 6,
        "fixed": 0,
    }

    @staticmethod
    def make_B_branch(
        B_fn: Callable[[JointOperand], Array], dof_joint: int, max_dof: int
    ) -> Callable:
        def padded_branch(operand):
            B_unpadded = B_fn(operand)

            # Case 1: truncate if dof_joint > max_dof
            if dof_joint > max_dof:
                return B_unpadded[:, :max_dof]

            # Case 2: pad if dof_joint < max_dof
            return jnp.pad(
                B_unpadded, ((0, 0), (0, max_dof - dof_joint)), constant_values=0.0
            )

        return padded_branch


class Link:
    """
    Geometric utilities for link cross-sections and property interpolation.

    Attributes
    ----------
    cross_section_geometry : int
        Cross-section index using CrossSectionGeometry values.
    E : Array
        Young's modulus [N/m^2] (Pa). Scalar.
    nu : Array
        Poisson's ratio [-], typical range (-1, 0.5). Scalar.
    G : Array
        Shear modulus [N/m^2] (Pa). Scalar; internally G = E/(2(1+nu)).
    rho : Array
        Mass density [kg/m^3]. Scalar.
    eta : Array
        Material damping coefficient [N·s/m]. Scalar.
    l : Array
        Length of each division of the link (soft-link discretization) [m]. Scalar.

    r : Tuple[float, float]
        Initial and final radius for circular sections [m].
    h : Tuple[float, float]
        Initial and final height for rectangular sections [m].
    w : Tuple[float, float]
        Initial and final width for rectangular sections [m].
    a : Tuple[float, float]
        Initial and final semi-major axis for elliptical sections [m].
    b : Tuple[float, float]
        Initial and final semi-minor axis for elliptical sections [m].

    Notes
    -----
    - `interpolate_param` performs linear interpolation between initial and final
      geometric parameters along normalized arc length x in [0, 1].
    - `compute_*_params` return tuples (I_x, I_y, I_z, A) with shape (num_integration_points,):
        I_* [m^4], A [m^2]. These feed mass/stiffness/damping computations.
    """

    cross_section_geometry: int

    E: Array  # Young's Modulus [N/m²]
    nu: Array  # Poisson Ratio [-1, 0.5]
    G: Array  # Shear modulus [N/m²]
    rho: Array  # Density [kg/m³]
    eta: Array  # Material Damping [N·s/m]
    l: Array  # noqa: E741  # Length of each divisions of the link (soft link) [m]

    r: tuple[float, float]  # Initial and final value of the geometrical parameter
    h: tuple[float, float]  # Initial and final value of the geometrical parameter
    w: tuple[float, float]  # Initial and final value of the geometrical parameter
    a: tuple[float, float]  # Initial and final value of the geometrical parameter
    b: tuple[float, float]  # Initial and final value of the geometrical parameter

    @staticmethod
    def geometric_branches():
        """
        Returns a list of functions that compute the geometric parameters for different section types.
        Each function takes the same operand structure and returns the computed parameters.
        """
        return [
            Link.compute_circular_params,
            Link.compute_rectangular_params,
            Link.compute_elliptical_params,
        ]

    @staticmethod
    def interpolate_param(x, a, b):
        """
        Linearly interpolate a geometric parameter along normalized arc length.

        Args:
            x: Normalized coordinate in [0, 1]; shape compatible with broadcasting.
            a: Initial value at the base (x=0). Scalar/Array; units match the parameter.
            b: Final value at the tip (x=1). Scalar/Array; same units as `a`.

        Returns:
            Interpolated value with shape broadcast from inputs; units same as inputs.
        """
        return a + (b - a) * x

    @staticmethod
    def compute_rectangular_params(
        operand: GeometricOperand,
    ) -> tuple[Array, Array, Array, Array]:
        """
        Compute area and second moments for a rectangular section along the link.

        Uses linear interpolation of `h_params=(h_i, h_f)` and `w_params=(w_i, w_f)`
        over `integration_points` to obtain local height and width, then evaluates:
        - Iy = (1/12) h w^3 [m^4]
        - Iz = (1/12) w h^3 [m^4]
        - Ix = Iy + Iz [m^4] (polar about the center)
        - A  = h w [m^2]

        Args:
            operand: GeometricOperand with fields `integration_points`, `h_params`, `w_params`.

        Returns:
            Ix_p (Array): Polar second moment of area about the local x-axis [m^4], shape (num_integration_points,).
            Iy_p (Array): Second moment of area about the local y-axis [m^4], shape (num_integration_points,).
            Iz_p (Array): Second moment of area about the local z-axis [m^4], shape (num_integration_points,).
            A_p (Array): Cross-sectional area [m^2], shape (num_integration_points,).
        """
        normalized_points = operand.integration_points
        h_params = operand.h_params
        w_params = operand.w_params

        h_at_points = Link.interpolate_param(normalized_points, *h_params)
        w_at_points = Link.interpolate_param(normalized_points, *w_params)

        Iy_p = (1 / 12) * h_at_points * w_at_points**3
        Iz_p = (1 / 12) * w_at_points * h_at_points**3
        Ix_p = Iy_p + Iz_p
        A_p = h_at_points * w_at_points
        return Ix_p, Iy_p, Iz_p, A_p

    @staticmethod
    def compute_circular_params(
        operand: GeometricOperand,
    ) -> tuple[Array, Array, Array, Array]:
        """
        Compute area and second moments for a circular section along the link.

        Uses linear interpolation of `r_params=(r_i, r_f)` over `integration_points` to obtain local
        radius r, then evaluates:
        - Iy = π/4 r^4 [m^4]
        - Iz = π/4 r^4 [m^4]
        - Ix = Iy + Iz = π/2 r^4 [m^4] (polar moment)
        - A  = π r^2 [m^2]

        Args:
            operand: GeometricOperand with fields `integration_points`, `r_params`.

        Returns:
            Ix_p (Array): Polar second moment of area about the local x-axis [m^4], shape (num_integration_points,).
            Iy_p (Array): Second moment of area about the local y-axis [m^4], shape (num_integration_points,).
            Iz_p (Array): Second moment of area about the local z-axis [m^4], shape (num_integration_points,).
            A_p (Array): Cross-sectional area [m^2], shape (num_integration_points,).
        """
        normalized_points = operand.integration_points
        r_params = operand.r_params

        r_at_points = Link.interpolate_param(normalized_points, *r_params)

        Iy_p = jnp.pi / 4 * r_at_points**4
        Iz_p = Iy_p
        Ix_p = Iy_p + Iz_p
        A_p = jnp.pi * r_at_points**2
        return Ix_p, Iy_p, Iz_p, A_p

    @staticmethod
    def compute_elliptical_params(
        operand: GeometricOperand,
    ) -> tuple[Array, Array, Array, Array]:
        """
        Compute area and second moments for an elliptical section along the link.

        Uses linear interpolation of `a_params=(a_i, a_f)` and `b_params=(b_i, b_f)`
        over `integration_points` to obtain local semi-axes a, b, then evaluates:
        - Iy = π/4 a b^3 [m^4]
        - Iz = π/4 b a^3 [m^4]
        - Ix = Iy + Iz [m^4] (polar moment)
        - A  = π a b [m^2]

        Args:
            operand: GeometricOperand with fields `integration_points`, `a_params`, `b_params`.

        Returns:
            Ix_p (Array): Polar second moment of area about the local x-axis [m^4], shape (num_integration_points,).
            Iy_p (Array): Second moment of area about the local y-axis [m^4], shape (num_integration_points,).
            Iz_p (Array): Second moment of area about the local z-axis [m^4], shape (num_integration_points,).
            A_p (Array): Cross-sectional area [m^2], shape (num_integration_points,).
        """
        normalized_points = operand.integration_points
        a_params = operand.a_params
        b_params = operand.b_params

        a_at_points = Link.interpolate_param(normalized_points, *a_params)
        b_at_points = Link.interpolate_param(normalized_points, *b_params)

        Iy_p = jnp.pi / 4 * a_at_points * b_at_points**3
        Iz_p = jnp.pi / 4 * b_at_points * a_at_points**3
        Ix_p = Iy_p + Iz_p
        A_p = jnp.pi * a_at_points * b_at_points
        return Ix_p, Iy_p, Iz_p, A_p
