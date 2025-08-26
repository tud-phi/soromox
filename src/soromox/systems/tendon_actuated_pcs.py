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
    tendon_params : Dict[str, Array]
        Dictionary of arrays of length n_actuators representing the tendon parameters.
    d_s : Callable
        Function that returns the homogeneous vector (4D) of the distance of the tendons 
        from the central backbone at a given abscissa point.
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

    tendon_params: Dict[str, Array]
    d_s: Callable
    dd_s_ds: Callable

    def __init__(
        self,
        num_segments: int,
        params: Dict[str, Array],
        tendon_params: Dict[str, Array],
        actuation_basis: Dict[str, Callable],
        order_gauss: int = 5,
        num_actuators: Optional[int] = None,
        strain_selector: Optional[Array] = None,
        xi_ref: Optional[Array] = None,
    ):
        super().__init__(
            num_segments=num_segments,
            params=params,
            order_gauss=order_gauss,
            num_actuators=num_actuators,
            strain_selector=strain_selector,
            xi_ref=xi_ref,
        )

        self._set_params(params)
        self._set_tendon_params(tendon_params)
        self._set_actuation_basis(actuation_basis)


    def _set_tendon_params(self, tendon_params: Dict[str, Array]):
        if not isinstance(tendon_params, (dict)):
            raise TypeError("The parameter 'tendon_params' must be a dictionary of jnp.ndarrays.")
        self.tendon_params = tendon_params
        self.tendon_params['f'] =   tendon_params['lt'] / self.L_cum[-1]
        self.num_actuators = len(self.tendon_params['ry'])

    def _set_actuation_basis(self, actuation_basis: Dict[str, Callable]):
        self.d_s = actuation_basis['d_s']
        self.dd_s_ds = actuation_basis['dd_s_ds']
    
    def update_tendon_params(self, tendon_params: Dict[str, Array]) -> "TendonActuatedPCS":
        updated_self = self

        # Recompute derived parameters
        tendon_params = dict(tendon_params)
        tendon_params["f"] = tendon_params["lt"] / self.L_cum[-1]
        num_actuators = len(tendon_params["ry"])

        # update all fields
        updated_self = eqx.tree_at(lambda x: x.tendon_params, updated_self, tendon_params)
        updated_self = eqx.tree_at(lambda x: x.num_actuators, updated_self, num_actuators)

        return updated_self

    
    @eqx.filter_jit
    def _actuation_matrix(self, q: Array, s: Array) -> Array:
        """
        Compute the local actuation matrix at current abscissa s.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,)
            s (Array): abscissa points (num_gauss_points,)

        Returns:
            A (Array): Actuation matrix of shape at s (num_active_strains, num_actuators)
        """

        def Phi_a_kj(tendon_params, j):
            def get_block(xi, j, block_size=6):
                return lax.dynamic_slice(xi, (block_size*j,), (block_size,))
            
            def tilde(vec):
                """
                (3,3)
                """
                return jnp.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])

            def hat(vec):
                """
                (4,4)
                """
                return jnp.vstack([jnp.column_stack([tilde(vec[:3]), vec[3:]]), jnp.zeros(4)])
            
            xi_  =  get_block(xi, j, block_size=6)  # strains of segment j (6,)
            d_s  =  self.d_s(tendon_params, s)      # (4,)
            dd_s =  self.dd_s_ds(tendon_params, s)  # (4,)

            term =  (dd_s + hat(xi_) @ d_s)[:-1]    # (3,)
            norm =  jnp.linalg.norm(term)           # ()
            t    =  term / norm                     # (3,)

            return jnp.hstack([tilde(d_s) @ t, t])  # (6,)

        xi = self.strain(q)

        Phi_a = vmap(
            vmap(Phi_a_kj, in_axes=(0, None), out_axes=1),
            in_axes=(None, 0),
            out_axes=0)(self.tendon_params, jnp.arange(self.num_segments))              # (num_segments, 6, num_actuators)
        Phi_a = Phi_a.reshape((6*self.num_segments, self.num_actuators), order='C')     # (num_segments*6, num_actuators)

        A_local = self.B_xi.T @ Phi_a                                                   # (num_segments*6, num_actuators)

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

            # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
            # A_blocks_i = jnp.stack([A_j(j) for j in range(self.num_gauss_points)], axis=0)

            return A_blocks_i

        A_blocks_tot = vmap(A_i)(jnp.arange(self.num_segments))

        # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
        # A_blocks_tot = jnp.stack([A_i(i) for i in range(self.num_segments)], axis=0)

        A_full = jnp.sum(A_blocks_tot, axis=(0, 1))  # Sum over segments and Gauss points
        
        return A_full
    
    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Compute the actuation matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            A (Array): Actuation matrix of shape (num_active_strains, num_actuators).
        """
        A = self.compute_actuation_matrix(q)
        return A

