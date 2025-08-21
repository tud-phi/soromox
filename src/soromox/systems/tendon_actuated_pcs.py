import equinox as eqx
import jax
from jax import Array, lax, vmap
from jax import numpy as jnp
import numpy as onp
from typing import Callable, Dict, Tuple, Optional

from .utils import (
    compute_strain_basis,
    gauss_quadrature,
    scale_gaussian_quadrature,
)

from .pcs import PCS

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
    d: Array
        Distances of the straight tendons from the segment's backbone.
    alpha: Array
        Angles of the longitudinal planes where the tendons lie, 0 if aligned to the z-axis
    x_n: Array
        Depths of the attachment points of the tendons, along the x-axis
    segment_indices_to_actuate : Array
        Indices of the segments that are actuated.

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

    segment_indices_to_actuate: Array  # indices of the segments that are actuated, shape (num_actuated_segments,)
    ry: Array
    rz: Array
    my: Array
    mz: Array
    l_t: Array

    def __init__(
        self,
        num_segments: int,
        params: Dict[str, Array],
        order_gauss: int = 5,
        num_actuators: Optional[int] = None,
        strain_selector: Optional[Array] = None,
        xi_ref: Optional[Array] = None,
        segment_actuation_selector: Optional[Array] = None,
    ):
        super().__init__(
            num_segments=num_segments,
            params=params,
            order_gauss=order_gauss,
            num_actuators=num_actuators,
            strain_selector=strain_selector,
            xi_ref=xi_ref,
        )

        if segment_actuation_selector is None:
            segment_actuation_selector = jnp.ones(num_segments, dtype=bool)

        self.segment_indices_to_actuate = jnp.array(
            [i for i, act in enumerate(segment_actuation_selector) if act]
        )

        self._set_params(params)
        self.num_actuators = len(self.ry)

    def _set_params(self, params: Dict[str, Array]):
        """
        Set the parameters of the tendon-driven planar PCS.

        Args:
            params (Dict[str, Array]): Dictionary containing the parameters of the robot.
                Dictionary containing the robot parameters:
                - "th0": (optional) float
                    Initial orientation angle [rad]
                    Default is 90 degrees (1.57 radians).
                - "L": List/Array of num_segments floats
                    Length of each segment [m]
                - "r": List/Array of num_segments floats
                    Radius of each segment [m]
                - "rho": List/Array of num_segments floats
                    Density of each segment [kg/m^3]
                - "g": List/Array of 2 floats [gx, gy]
                    Gravitational acceleration vector [m/s^2]
                - "E": List/Array of num_segments floats
                    Elastic modulus of each segment [Pa]
                - "G": List/Array of num_segments floats
                    Shear modulus of each segment [Pa]
                - "D": List/Array of (num_segments x num_segments) floats
                    Damping matrix of each segment [Pa*s]
                - "d": List/Array of num_tendon floats
                    Distance of the tendons from the segment's backbone [m]
                - "alpha": List/Array of num_tendon floats
                    Angles of the longitudinal planes where the tendons lie [rad]
                - "x_n": List/Array of num_tendon floats
                    Depths of the attachment points of the tendons, along the x-axis [m]
        """
        super()._set_params(params)

        # 
        try:
            ry = params["ry"]
        except KeyError:
            raise KeyError(
                "The parameter 'ry' () is required for the tendon-driven PCS."
            )
        if not isinstance(ry, (list, jnp.ndarray)):
            raise TypeError("The parameter 'ry' must be a list or a jnp.ndarray.")
        self.ry = jnp.asarray(ry, dtype=jnp.float64)

        # 
        try:
            rz = params["rz"]
        except KeyError:
            raise KeyError(
                "The parameter 'rz' () is required for the tendon-driven PCS."
            )
        if not isinstance(rz, (list, jnp.ndarray)):
            raise TypeError("The parameter 'rz' must be a list or a jnp.ndarray.")
        self.rz = jnp.asarray(rz, dtype=jnp.float64)

        # 
        try:
            my = params["my"]
        except KeyError:
            raise KeyError(
                "The parameter 'my' () is required for the tendon-driven PCS."
            )
        if not isinstance(my, (list, jnp.ndarray)):
            raise TypeError("The parameter 'my' must be a list or a jnp.ndarray.")
        self.my = jnp.asarray(my, dtype=jnp.float64)

        # 
        try:
            mz = params["mz"]
        except KeyError:
            raise KeyError(
                "The parameter 'mz' () is required for the tendon-driven PCS."
            )
        if not isinstance(mz, (list, jnp.ndarray)):
            raise TypeError("The parameter 'mz' must be a list or a jnp.ndarray.")
        self.mz = jnp.asarray(mz, dtype=jnp.float64)

        # Depths of the attachment points of the tendons, along the x-axis
        try:
            l_t = params["l_t"]
        except KeyError:
            raise KeyError(
                "The parameter 'l_t' (depths of the attachment points of the tendons) is required for the tendon-driven PCS."
            )
        if not isinstance(l_t, (list, jnp.ndarray)):
            raise TypeError("The parameter 'l_t' must be a list or a jnp.ndarray.")
        self.l_t = jnp.asarray(l_t, dtype=jnp.float64)

    def update_params(self, params: Dict[str, Array]) -> "TendonActuatedPCS":
        """
        Update the parameters of the tendon-driven planar PCS.

        Args:
            params (Dict[str, Array]):
                Dictionary that contains the robot parameters to update:
                - "th0": (optional) float
                    Initial orientation angle [rad]
                - "L": List/Array of num_segments floats
                    Length of each segment [m]
                - "r": List/Array of num_segments floats
                    Radius of each segment [m]
                - "rho": List/Array of num_segments floats
                    Density of each segment [kg/m^3]
                - "g": List/Array of 2 floats [gx, gy]
                    Gravitational acceleration vector [m/s^2]
                - "E": List/Array of num_segments floats
                    Elastic modulus of each segment [Pa]
                - "G": List/Array of num_segments floats
                    Shear modulus of each segment [Pa]
                - "D": List/Array of (num_segments x num_segments) floats
                    Damping matrix of each segment [Pa*s]
                - "d": List/Array of num_segments floats
                    Distance of the tendons from the segment's backbone [m]

        Returns:
            updated_self (TendonActuatedPlanarPCS):
                A new instance of TendonActuatedPlanarPCS with updated parameters.
        """
        # Apply updates sequentially
        updated_self = super().update_params(params)

        if "ry" in params:
            ry = params["ry"]
            if not isinstance(ry, (list, jnp.ndarray)):
                raise TypeError("The parameter 'ry' must be a list or a jnp.ndarray.")
            updated_self = eqx.tree_at(
                lambda x: x.ry, updated_self, jnp.asarray(ry, dtype=jnp.float64)
            )

        if "rz" in params:
            rz = params["rz"]
            if not isinstance(rz, (list, jnp.ndarray)):
                raise TypeError("The parameter 'rz' must be a list or a jnp.ndarray.")
            updated_self = eqx.tree_at(
                lambda x: x.rz, updated_self, jnp.asarray(rz, dtype=jnp.float64)
            )

        if "my" in params:
            my = params["my"]
            if not isinstance(my, (list, jnp.ndarray)):
                raise TypeError("The parameter 'my' must be a list or a jnp.ndarray.")
            updated_self = eqx.tree_at(
                lambda x: x.my, updated_self, jnp.asarray(my, dtype=jnp.float64)
            )

        if "mz" in params:
            mz = params["mz"]
            if not isinstance(mz, (list, jnp.ndarray)):
                raise TypeError("The parameter 'mz' must be a list or a jnp.ndarray.")
            updated_self = eqx.tree_at(
                lambda x: x.mz, updated_self, jnp.asarray(mz, dtype=jnp.float64)
            )

        if "l_t" in params:
            l_t = params["l_t"]
            if not isinstance(l_t, (list, jnp.ndarray)):
                raise TypeError("The parameter 'l_t' must be a list or a jnp.ndarray.")
            updated_self = eqx.tree_at(
                lambda x: x.l_t, updated_self, jnp.asarray(l_t, dtype=jnp.float64)
            )

        return updated_self

    @eqx.filter_jit
    def _actuation_matrix(self, q: Array, s: Array) -> Array:
        """
        """

        def Phi_a_k(k, j):
            idx = 6*j
            column = jnp.array([
            (f[k]*s*self.my[k] + self.ry[k])*(f[k]*self.mz[k] + xi[idx]*(f[k]*s*self.my[k] + self.ry[k]) + xi[idx+5]) + \
             (-f[k]*s*self.mz[k] - self.rz[k])*(f[k]*self.my[k] - xi[idx]*(f[k]*s*self.mz[k] + self.rz[k]) + xi[idx+4]),
            (f[k]*s*self.mz[k] + self.rz[k])*(xi[idx+1]*(f[k]*s*self.mz[k] + self.rz[k]) - xi[idx+2]*(f[k]*s*self.my[k] + self.ry[k]) + xi[idx+3]),
            (-f[k]*s*self.my[k] - self.ry[k])*(xi[idx+1]*(f[k]*s*self.mz[k] + self.rz[k]) - xi[idx+2]*(f[k]*s*self.my[k] + self.ry[k]) + xi[idx+3]),
            xi[idx+1]*(f[k]*s*self.mz[k] + self.rz[k]) - xi[idx+2]*(f[k]*s*self.my[k] + self.ry[k]) + xi[idx+3],
            f[k]*self.my[k] - xi[idx]*(f[k]*s*self.mz[k] + self.rz[k]) + xi[idx+4],
            f[k]*self.mz[k] + xi[idx]*(f[k]*s*self.my[k] + self.ry[k]) + xi[idx+5]
            ])
            term = jnp.array([
                xi[idx+1]*(f[k]*s*self.mz[k] + self.rz[k]) - xi[idx+2]*(f[k]*s*self.my[k] + self.ry[k]) + xi[idx+3],
                f[k]*self.my[k] - xi[idx]*(f[k]*s*self.mz[k] + self.rz[k]) + xi[idx+4],
                f[k]*self.mz[k] + xi[idx]*(f[k]*s*self.my[k] + self.ry[k]) + xi[idx+5]
                ])
            norm = jnp.linalg.norm(term)
            return column / norm    # (6,)

        xi = self.strain(q)
        f = self.l_t / self.L_cum[-1]

        Phi_a = vmap(
            vmap(Phi_a_k, in_axes=(0, None), out_axes=1),
            in_axes=(None, 0),
            out_axes=0)(jnp.arange(self.num_actuators), jnp.arange(self.num_segments))  # (num_segments, 6, num_num_actuators)
        Phi_a = Phi_a.reshape((6*self.num_segments, self.num_actuators), order='C')     # (num_segments*6, num_num_actuators)

        A_local = self.B_xi.T @ Phi_a                                                   # (num_segments*6, num_num_actuators)

        return A_local

    def compute_actuation_matrix(self, q: Array) -> Array:
        def A_i(i):
            Xs_scaled, Ws_scaled = scale_gaussian_quadrature(self.Xs, self.Ws, self.L_cum[i], self.L_cum[i + 1])
            def A_j(j):
                Xs_j = Xs_scaled[j]
                Ws_j = Ws_scaled[j]
                A_jj = self._actuation_matrix(q, Xs_j)
                return Ws_j * A_jj

            A_blocks_i = vmap(A_j)(jnp.arange(self.num_gauss_points))
            #print('A_blocks_i', A_blocks_i.shape)

            # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
            # A_blocks_i = jnp.stack([A_j(j) for j in range(self.num_gauss_points)], axis=0)

            return A_blocks_i

        A_blocks_tot = vmap(A_i)(jnp.arange(self.num_segments))
        #print('A_blocks_tot', A_blocks_tot.shape)

        # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
        # A_blocks_tot = jnp.stack([A_i(i) for i in range(self.num_segments)], axis=0)

        A_full = jnp.sum(A_blocks_tot, axis=(0, 1))  # Sum over segments and Gauss points
        #print('A_full', A_full.shape)
        
        return A_full