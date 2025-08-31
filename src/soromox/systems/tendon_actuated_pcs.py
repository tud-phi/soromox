__all__ = ["linear_routing", "linear_routing_derivative", "TendonActuatedPCS"]
import equinox as eqx
import jax
from jax import Array, lax, vmap
from jax import numpy as jnp
import numpy as onp
from typing import Callable, Dict, Tuple, Optional

import soromox.utils.lie_algebra as lie

from .pcs import PCS
from .utils import scale_gaussian_quadrature


def linear_routing(tendon_routing_params, s):
    """
    Function representing the routing of linear tendons as function of s. Specifically, d_s
    is the vector between the centerline of the robot and the tendon position, in the
    cross-sectional plane at abscissa s.

    Args:
            tendon_routing_params (Dict[Array]): parameters of the linear tendons (6,)
                ry: y-coordinate of the pulling point of the tendons [m]
                my: slope coefficient in the x-y plane of the tendons [-]
                rz: z-coordinate of the pulling point of the tendons [m]
                mz: slope coefficient in the x-z plane of the tendons [-]
            s (Array): abscissa points (num_gauss_points,)

        Returns:
            d_s (Array): position of the tendon at s wrt centerline, in homogeneous coordinates (4,)
    """

    ry = tendon_routing_params["ry"]
    my = tendon_routing_params["my"]
    rz = tendon_routing_params["rz"]
    mz = tendon_routing_params["mz"]
    d_s = jnp.array([0.0, ry + my * s, rz + mz * s, 1.0])
    return d_s


def linear_routing_derivative(tendon_routing_params, s):
    """
    Function representing the spatial derivative of the routing of linear tendons as function of s.
    Specifically, dd_s_ds is the derivative of d_s over s, at s.

    Args:
            tendon_routing_params (Dict[Array]): parameters of the linear tendons (6,)
                my: slope coefficient in the x-y plane of the tendons [-]
                mz: slope coefficient in the x-z plane of the tendons [-]
            s (Array): abscissa points (num_gauss_points,)

        Returns:
            dd_s_ds (Array): path of the tendon at s wrt centerline, in homogeneous coordinates (4,)
    """

    my = tendon_routing_params["my"]
    mz = tendon_routing_params["mz"]
    dd_s_ds = jnp.array([0.0, my, mz, 1.0])
    return dd_s_ds


class TendonActuatedPCS(PCS):
    """
    Piecewise Constant Strain (PCS) model for 3D soft continuum robots.

    This class implements the geometric and dynamic modeling of a 3D soft robot
    using the Cosserat rod theory and piecewise constant strain assumption.
    It supports computation of forward kinematics, Jacobians, dynamical matrices.

    Attributes:
    ----------
    num_segments : int
        Number of segments (constant strain sections) along the robot.
    num_actuators : int
        Number of actuators (control inputs) for the robot.
    g0 : Array
        Initial pose of the robot base as an SE(3) transformation matrix.
    g : Array
        Gravitational acceleration vector (embedded in a 6D vector).
        [0, 0, 0, g_x, g_y, g_z]
    L, r, E, G, rho, D : Array
        Physical properties of each segment (length, radius, elastic/shear modulus, etc.).
    num_active_strains : int
        Number of active strain components (based on strain_selector).
    num_strains : int
        Total number of strain components (6 * num_segments).
    B_xi : Array
        Basis matrix for projecting active strains.
    xi_ref : Array
        Reference strain (reference configuration) of the robot.
    num_gauss_points : int
        Number of points used for numerical integration.
        Corresponds to the order of Gauss-Legendre quadrature + 2 (for the endpoints).
    Xs, Ws : Array
        Gauss-Legendre quadrature nodes and weights for numerical integration.
    tendon_routing_params : Dict[str, Array]
        Dictionary of arrays of length n_actuators representing the tendon parameters.
    d_s : Callable
        Function that returns the homogeneous vector [d_x, d_y, d_z, 1] of the tendon
        position wrt the central backbone within the cross-sectional plane at a given
        abscissa point s. The first 3 entries represent the coordinate of the tendon
        position with respect to the local cross-sectional frame at s. For this reason,
        d_x is always equal to 0.

    dd_s_ds : Callable
        Function that returns the homogeneous vector (4D) of the derivative over s of
        the d_s.

    Notes:
    -----
    - The strain vector is composed of 6 components per segment:
      [kappa_x, kappa_y, kappa_z, sigma_x, sigma_y, sigma_z].
      By default, the rod is assumed to be straight and aligned with the x-axis,
        so the reference strain is set to [0, 0, 0, 1, 0, 0].
        Thus:   - kappa_x corresponds to torsion around the x-axis,
                - kappa_y corresponds to bending around the y-axis,
                - kappa_z corresponds to bending around the z-axis,
                - sigma_x corresponds to axial strain along the x-axis,
                - sigma_y corresponds to shear along the y-axis,
                - sigma_z corresponds to shear along the z-axis.

    """

    tendon_routing_params: Dict[str, Array]
    d_s: Callable
    dd_s_ds: Callable

    def __init__(
        self,
        num_segments: int,
        params: Dict[str, Array],
        *args,
        tendon_routing_basis: Optional[Dict[str, Callable]] = None,
        tendon_routing_params: Dict[str, Array],
        **kwargs,
    ):
        """
        Initialize the TendonActuatedPCS class.

        Args:
            num_segments (int):
                Number of segments in the robot.
            params (Dict[str, Array]):
                Dictionary containing the robot parameters:
                - "p0": (optional) List/Array of shape (6,)
                    Initial orientation and position in the inertial frame [rad, m]
                    [ψ, θ, φ, x0, y0, z0]
                        [ψ, θ, φ] are the Euler angles in the ZXZ convention:
                            ψ (psi): Rotation around Z axis (fixed axis)
                            θ (theta): Rotation around X' axis (after first rotation)
                            φ (phi): Rotation about Z' axis (after the first two rotations)
                        [x0, y0, z0]: Base position in the inertial frame
                    Defaults to [pi/2, pi/2, 0.0, 0.0, 0.0, 0.0].
                - "L": List/Array of num_segments floats
                    Length of each segment [m]
                - "r": List/Array of num_segments floats
                    Radius of each segment [m]
                - "rho": List/Array of num_segments floats
                    Density of each segment [kg/m^3]
                - "g": List/Array of 3 floats [gx, gy, gz]
                    Gravitational acceleration vector [m/s^2]
                - "E": List/Array of num_segments floats
                    Elastic modulus of each segment [Pa]
                - "G": List/Array of num_segments floats
                    Shear modulus of each segment [Pa]
                - "D": Array of shape (6*num_segments, 6*num_segments)
                    Damping matrix [Pa·s]
            order_gauss (int, optional):
                Order of the Gauss-Legendre quadrature for integration over each segment.
                Defaults to 5.
            strain_selector (Optional[Array], optional):
                Boolean array of shape (6 * num_segments,) specifying which strain components are active.
                Defaults to all strains active (i.e. all True).
            xi_ref (Optional[Array], optional):
                Reference strain of shape (6 * num_segments,).
                Defaults to 0.0 for bending and shear strains, and 1.0 for axial strain (along local x-axis).
            tendon_routing_basis (Optional[Dict[str, Callable]]):
                Dictionary with the tendon routing functions. If None, a linear routing is used.
                Expected keys and signatures:
                - "d_s": Callable[[Dict[str, Array], Array], Array]
                    Returns the homogeneous 4D vector [d_x, d_y, d_z, 1] giving the tendon position
                    with respect to the local cross-section at abscissa s. For typical routing,
                    d_x = 0 as the offset lies in the cross-sectional plane.
                - "dd_s_ds": Callable[[Dict[str, Array], Array], Array]
                    Returns the homogeneous 4D derivative over s of d_s.
            tendon_routing_params (Dict[str, Array]):
                Dictionary describing the tendon routing parameters for each actuator (length n_actuators).
                When using the default linear routing, the following keys are expected:
                - "ry": Array (n_actuators,)
                    y-intercept of the tendon line at the base [m].
                - "my": Array (n_actuators,)
                    Slope in the x–y plane [-].
                - "rz": Array (n_actuators,)
                    z-intercept of the tendon line at the base [m].
                - "mz": Array (n_actuators,)
                    Slope in the x–z plane [-].
                - "lt": Array (n_actuators,)
                    Attachment segment index for each tendon (0-based, inclusive). The tendon contributes
                    along the backbone up to and including this segment’s distal end.
        """
        super().__init__(
            num_segments,
            params,
            *args,
            **kwargs,
        )

        # Set default tendon routing basis to linear routing if not provided
        if tendon_routing_basis is None:
            tendon_routing_basis = {
                "d_s": linear_routing,
                "dd_s_ds": linear_routing_derivative,
            }
        self._set_tendon_routing_basis(tendon_routing_basis)
        self._set_tendon_routing_params(tendon_routing_params)

    def _set_tendon_routing_params(self, tendon_routing_params: Dict[str, Array]):
        if not isinstance(tendon_routing_params, (dict)):
            raise TypeError(
                "The parameter 'tendon_routing_params' must be a dictionary of jnp.ndarrays."
            )
        self.tendon_routing_params = tendon_routing_params
        self.num_actuators = len(self.tendon_routing_params["ry"])

    def _set_tendon_routing_basis(self, tendon_routing_basis: Dict[str, Callable]):
        self.d_s = tendon_routing_basis["d_s"]
        self.dd_s_ds = tendon_routing_basis["dd_s_ds"]

    def update_tendon_routing_params(
        self, tendon_routing_params: Dict[str, Array]
    ) -> "TendonActuatedPCS":
        updated_self = self

        # Recompute derived parameters
        tendon_routing_params = dict(tendon_routing_params)
        num_actuators = len(tendon_routing_params["ry"])

        # update all fields
        updated_self = eqx.tree_at(
            lambda x: x.tendon_routing_params, updated_self, tendon_routing_params
        )
        updated_self = eqx.tree_at(
            lambda x: x.num_actuators, updated_self, num_actuators
        )

        return updated_self

    @eqx.filter_jit
    def _local_actuation_matrix(self, q: Array, s: Array) -> Array:
        """
        Compute the local actuation matrix at current abscissa s.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,)
            s (Array): abscissa points (num_gauss_points,)

        Returns:
            A_s (Array): Actuation matrix of shape at s (num_active_strains, num_actuators)
        """

        def Phi_a_kj(tendon_routing_params, j):
            def get_block(xi, j, block_size=6):
                return lax.dynamic_slice(xi, (block_size * j,), (block_size,))

            attachment_segment_idx = tendon_routing_params["lt"]  # ()
            cond = attachment_segment_idx >= j  # ()

            xi_ = get_block(xi, j, block_size=6)  # strains of segment j (6,)
            d_s = self.d_s(tendon_routing_params, s)  # (4,)
            dd_s = self.dd_s_ds(tendon_routing_params, s)  # (4,)

            term = (dd_s + lie.hat_SE3(xi_) @ d_s)[:-1]  # (3,)
            norm = jnp.linalg.norm(term)  # ()
            t = term / norm  # (3,)

            return cond * jnp.hstack([lie.tilde_SE3(d_s[:-1]) @ t, t])  # (6,)

        xi = self.strain(q)  # strains (6*num_segments,)

        Phi_a = vmap(
            vmap(Phi_a_kj, in_axes=(0, None), out_axes=1), in_axes=(None, 0), out_axes=0
        )(
            self.tendon_routing_params, jnp.arange(self.num_segments)
        )  # (num_segments, 6, num_actuators)

        Phi_a = Phi_a.reshape(
            (6 * self.num_segments, self.num_actuators)
        )  # (6*num_segments, num_actuators)

        A_s = self.B_xi.T @ Phi_a  # (6*num_segments, num_actuators)

        return A_s

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Compute the actuation matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            A (Array): Actuation matrix of shape (num_active_strains, num_actuators).
        """

        def A_i(i):
            Xs_scaled, Ws_scaled = scale_gaussian_quadrature(
                self.Xs, self.Ws, self.L_cum[i], self.L_cum[i + 1]
            )

            def build_block(n1, n2, j, step):
                a = jnp.zeros((n1, n2), dtype=jnp.int32)
                block = jnp.ones((step, n2), dtype=jnp.int32)
                # j_clipped = jnp.clip(j, 0, n2 + step)
                return lax.dynamic_update_slice(
                    a, block, (jnp.array(j, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))
                )

            strain_mask = build_block(
                6 * self.num_segments, # numer of rows
                self.num_actuators, # number of columns
                6 * i, # starting index
                6, # number of rows set to 1 starting from starting index
            )

            def A_j(j):
                Xs_j = Xs_scaled[j]
                Ws_j = Ws_scaled[j]
                A_jj = self._local_actuation_matrix(q, Xs_j)
                return Ws_j * A_jj * strain_mask

            A_blocks_i = vmap(A_j)(jnp.arange(self.num_gauss_points))

            # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
            # A_blocks_i = jnp.stack([A_j(j) for j in range(self.num_gauss_points)], axis=0)
            # print('A_blocks_i =\n', A_blocks_i.shape)

            return A_blocks_i

        A_blocks_tot = vmap(A_i)(jnp.arange(self.num_segments))

        # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
        # A_blocks_tot = jnp.stack([A_i(i) for i in range(self.num_segments)], axis=0)
        # print('A_blocks_tot =\n', A_blocks_tot.shape)

        A_full = jnp.sum(
            A_blocks_tot, axis=(0, 1)
        )  # Sum over segments and Gauss points

        return A_full

    @eqx.filter_jit
    def forward_kinematics_tendons(
        self, tendon_routing_params: Dict[str, float], q: Array, s: Array
    ) -> Array:
        """
        Compute the forward kinematics of the robot at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L]. (n_s,)

        Returns:
            t (Array): 3D position of the tendons at s, shape (n_tendons, n_s, 3)
        """

        g_s = self.forward_kinematics(q, s)  # (4,4)
        t_s = g_s @ self.d_s(tendon_routing_params, s)  # (4,)
        t_s = t_s[:-1]  # (3,)

        lt = self.L_cum[tendon_routing_params["lt"] + 1]  # ()
        mask = s <= lt  # ()

        t_s = jnp.where(mask, t_s, jnp.nan)  # (3,)

        return t_s
