import equinox as eqx
from jax import Array, lax, vmap
from jax import numpy as jnp
import numpy as onp
from typing import Callable, Dict, Tuple, Optional

from .utils import (
    compute_strain_basis,
    compute_planar_stiffness_matrix,
    gauss_quadrature,
    scale_gaussian_quadrature,
)
from soromox.math_utils import (
    blk_diag,
    compute_weighted_sums,
)
import soromox.utils.lie_algebra as lie

from diffrax import (
    diffeqsolve,
    ODETerm,
    SaveAt,
    Tsit5,
    PIDController,
    ConstantStepSize,
    AbstractSolver,
)

# Function to compute the stiffness matrix for all segments
compute_stiffness_matrix_for_all_segments_fn = vmap(compute_planar_stiffness_matrix)


class PlanarPCS(eqx.Module):
    """
    Planar Piecewise Constant Strain (PCS) model for 2D soft continuum robots.

    This class implements the geometric and dynamic modeling of a 2D soft robot
    using the Cosserat rod theory and piecewise constant strain assumption.
    It supports computation of forward kinematics, Jacobians, dynamical matrices.

    Attributes:
    ----------
    num_segments : int
        Number of segments (constant strain sections) along the robot.
    num_actuators : int
        Number of actuators (control inputs) for the robot.
    th0 : Array
        Initial orientation angle of the robot in radians.
    g : Array
        Gravitational acceleration vector (embedded in a 3D vector).
        [0, g_x, g_y]
    L, r, E, G, rho, D : Array
        Physical properties of each segment (length, radius, elastic/shear modulus, etc.).
    num_active_strains : int
        Number of active strain components (based on strain_selector).
    num_strains : int
        Total number of strain components (6 * num_segments).
    B_xi : Array
        Basis matrix for projecting active strains.
    xi_star : Array
        Rest strain (reference configuration) of the robot.
    num_gauss_points : int
        Number of points used for numerical integration.
        Corresponds to the order of Gauss-Legendre quadrature + 2 (for the endpoints).
    Xs, Ws : Array
        Gauss-Legendre quadrature nodes and weights for numerical integration.

    Notes:
    -----
    - The strain vector is composed of 3 components per segment:
      [kappa_z, sigma_x, sigma_y].
      By default, the rod is assumed to be straight and aligned with the y-axis,
        so the rest strain is set to [0, 0, 1].
        Thus:   - kappa_z corresponds to bending around the z-axis,
                - sigma_x corresponds to shear along the x-axis,
                - sigma_y corresponds to axial strain along the y-axis.

    """

    # Robot parameters
    th0: Array  # Initial orientation angle [rad]
    g: Array  # Gravitational acceleration vector

    L: Array  # Length of the segments
    L_cum: Array  # Cumulative length of the segments
    r: Array  # Radius of the segments
    rho: Array
    E: Array  # Young's modulus of the segments
    G: Array  # Shear modulus of the segments
    D: Array  # Damping coefficient of the segments

    global_eps: float = jnp.finfo(jnp.float64).eps

    num_segments: int = eqx.field(static=True)
    num_actuators: int = eqx.field(static=True)  # Number of actuators
    num_gauss_points: int = eqx.field(static=True)  #
    num_strains: int = eqx.field(static=True)  # Number of strains (3 * num_segments)

    xi_star: Array  # Rest configuration strain
    B_xi: Array  # Strain basis matrix
    num_active_strains: Array  # Number of selected strains

    Xs: Array  # Gauss nodes
    Ws: Array  # Gauss weights

    def __init__(
        self,
        num_segments: int,
        params: Dict[str, Array],
        num_actuators: Optional[int] = None,
        order_gauss: int = 5,
        strain_selector: Optional[Array] = None,
        xi_star: Optional[Array] = None,
    ):
        """
        Initialize the PlanarPCS class.

        Args:
            num_segments (int):
                Number of segments in the robot.
            params (Dict[str, Array]):
                Dictionary containing the robot parameters:
                - "th0": Initial orientation angle [rad]
                - "L": Length of each segment [m]
                - "r": Radius of each segment [m]
                - "rho": Density of each segment [kg/m^3]
                - "g": Gravitational acceleration vector [m/s^2]
                - "E": Elastic modulus of each segment [Pa]
                - "G": Shear modulus of each segment [Pa]
                - "D": Damping matrix of each segment
            num_actuators (Optional[int], optional):
                Number of actuators (control inputs) for the robot. If None, we default to a fully actuated robot (i.e. num_actuators = num_active_strains).
            order_gauss (int, optional):
                Order of the Gauss-Legendre quadrature for integration over each segment.
                Defaults to 5.
            strain_selector (Optional[Array], optional):
                Boolean array of shape (3 * num_segments,) specifying which strain components are active.
                Defaults to all strains active (i.e. all True).
            xi_star (Optional[Array], optional):
                Rest strain of shape (3 * num_segments,).
                Defaults to 0.0 for bending and shear strains, and 1.0 for axial strain (along local y-axis).

        """
        # Number of segments
        if not isinstance(num_segments, int):
            raise TypeError(
                f"num_segments must be an integer, got {type(num_segments).__name__}"
            )
        if num_segments < 1:
            raise ValueError(f"num_segments must be at least 1, got {num_segments}")
        self.num_segments = num_segments

        num_strains = 3 * num_segments
        self.num_strains = num_strains

        # ================================================================
        # Robot parameters
        self._set_params(params)

        # ================================================================
        # Order of Gauss-Legendre quadrature
        if not isinstance(order_gauss, int):
            raise TypeError(
                f"order_gauss must be an integer, got {type(order_gauss).__name__}"
            )
        if order_gauss < 1:
            raise ValueError(f"param_integration must be at least 1, got {order_gauss}")
        Xs, Ws, num_gauss_points = gauss_quadrature(order_gauss, a=0.0, b=1.0)
        self.Xs = Xs
        self.Ws = Ws
        self.num_gauss_points = num_gauss_points

        # ================================================================
        # Strain basis matrix
        if strain_selector is None:
            strain_selector = jnp.ones(self.num_strains, dtype=bool)
        else:
            if not isinstance(strain_selector, (list, jnp.ndarray)):
                raise TypeError(
                    f"strain_selector must be a list or an array, got {type(strain_selector).__name__}"
                )
            strain_selector = jnp.asarray(strain_selector)
            if not jnp.issubdtype(strain_selector.dtype, jnp.bool_):
                raise TypeError(
                    f"strain_selector must be a boolean array, got {strain_selector.dtype}"
                )
            if strain_selector.size != self.num_strains:
                raise ValueError(
                    f"strain_selector must have {self.num_strains} elements, got {strain_selector.size}"
                )
            strain_selector = strain_selector.reshape(self.num_strains)
        self.B_xi = compute_strain_basis(strain_selector)

        self.num_active_strains = jnp.sum(strain_selector)

        # Rest configuration strain
        if xi_star is None:
            xi_star = jnp.tile(
                jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64), (self.num_segments, 1)
            ).reshape(self.num_strains)
        else:
            if not isinstance(xi_star, (list, jnp.ndarray)):
                raise TypeError(
                    f"xi_star must be a list or an array, got {type(xi_star).__name__}"
                )
            xi_star = jnp.asarray(xi_star)
            if xi_star.size != self.num_strains:
                raise ValueError(
                    f"xi_star must have {self.num_strains} elements, got {xi_star.size}"
                )
            xi_star = xi_star.reshape(self.num_strains)
        self.xi_star = xi_star

        # Number of actuators
        if num_actuators is None:
            self.num_actuators = int(self.num_active_strains.item())
        else:
            self.num_actuators = num_actuators

    def _set_params(
        self,
        params: Dict[str, Array],
    ) -> None:
        """
        Set the parameters of the PCS model.

        Args:
            params (Dict[str, Array]):
                Dictionary containing the robot parameters:
                - "th0": Initial orientation angle [rad]
                - "L": Length of each segment [m]
                - "r": Radius of each segment [m]
                - "rho": Density of each segment [kg/m^3]
                - "g": Gravitational acceleration vector [m/s^2]
                - "E": Elastic modulus of each segment [Pa]
                - "G": Shear modulus of each segment [Pa]
                - "D": Damping matrix of each segment
        """
        # Initial orientation angle
        try:
            th0 = params["th0"]
        except KeyError:
            raise KeyError("Parameter 'th0' is required in params dictionary.")
        if not (isinstance(th0, (float, int, jnp.ndarray))):
            raise TypeError(
                f"th0 must be a float, int, or an array, got {type(th0).__name__}"
            )
        th0 = jnp.asarray(th0, dtype=jnp.float64)
        self.th0 = th0

        # Gravitational acceleration vector
        try:
            g = params["g"]
        except KeyError:
            raise KeyError("Parameter 'g' is required in params dictionary.")
        if not (isinstance(g, (list, jnp.ndarray))):
            raise TypeError(f"g must be a list or an array, got {type(g).__name__}")
        g = jnp.asarray(g, dtype=jnp.float64)
        if g.size != 2:
            raise ValueError(f"g must be a vector of shape (2,), got {g.size}")
        self.g = jnp.concatenate(
            [jnp.zeros(1), g]
        )  # Add a zero for the orientation angle

        # Lengths of the segments
        try:
            L = params["L"]
        except KeyError:
            raise KeyError("Parameter 'L' is required in params dictionary.")
        if not (isinstance(L, (list, jnp.ndarray))):
            raise TypeError(f"L must be a list or an array, got {type(L).__name__}")
        L = jnp.asarray(L, dtype=jnp.float64)
        if L.shape != (self.num_segments,):
            raise ValueError(f"L must have shape ({self.num_segments},), got {L.shape}")
        self.L = L

        L_cum = jnp.cumsum(jnp.concatenate([jnp.zeros(1), self.L]))
        self.L_cum = L_cum

        # Radius of the segments
        try:
            r = params["r"]
        except KeyError:
            raise KeyError("Parameter 'r' is required in params dictionary.")
        if not (isinstance(r, (list, jnp.ndarray))):
            raise TypeError(f"r must be a list or an array, got {type(r).__name__}")
        r = jnp.asarray(r, dtype=jnp.float64)
        if r.shape != (self.num_segments,):
            raise ValueError(f"r must have shape ({self.num_segments},), got {r.shape}")
        self.r = r

        # Densities of the segments
        try:
            rho = params["rho"]
        except KeyError:
            raise KeyError("Parameter 'rho' is required in params dictionary.")
        if not (isinstance(rho, (list, jnp.ndarray))):
            raise TypeError(f"rho must be a list or an array, got {type(rho).__name__}")
        rho = jnp.asarray(rho, dtype=jnp.float64)
        if rho.shape != (self.num_segments,):
            raise ValueError(f"rho must have shape ({self.num_segments},), got {rho.shape}")
        self.rho = rho

        # Elastic modulus of the segments
        try:
            E = params["E"]
        except KeyError:
            raise KeyError("Parameter 'E' is required in params dictionary.")
        if not (isinstance(E, (list, jnp.ndarray))):
            raise TypeError(f"E must be a list or an array, got {type(E).__name__}")
        E = jnp.asarray(E, dtype=jnp.float64)
        if E.shape != (self.num_segments,):
            raise ValueError(f"E must have shape ({self.num_segments},), got {E.shape}")
        self.E = E

        # Shear modulus of the segments
        try:
            G = params["G"]
        except KeyError:
            raise KeyError("Parameter 'G' is required in params dictionary.")
        if not (isinstance(G, (list, jnp.ndarray))):
            raise TypeError(f"G must be a list or an array, got {type(G).__name__}")
        G = jnp.asarray(G, dtype=jnp.float64)
        if G.shape != (self.num_segments,):
            raise ValueError(f"G must have shape ({self.num_segments},), got {G.shape}")
        self.G = G

        # Damping matrix of the robot
        try:
            D = params["D"]
        except KeyError:
            raise KeyError("Parameter 'D' is required in params dictionary.")
        if not (isinstance(D, (list, jnp.ndarray))):
            raise TypeError(f"D must be a list or an array, got {type(D).__name__}")
        D = jnp.asarray(D, dtype=jnp.float64)
        expected_D_shape = (self.num_strains, self.num_strains)
        if D.shape != expected_D_shape:
            raise ValueError(f"D must have shape {expected_D_shape}, got {D.shape}")
        self.D = D

    def update_params(self, params: Dict[str, Array]) -> "PlanarPCS":
        """
        Update the parameters of the PCS model.

        Args:
            params (Dict[str, Array]):
                Dictionary containing the robot parameters:
                - "th0": Initial orientation angle [rad]
                - "L": Length of each segment [m]
                - "r": Radius of each segment [m]
                - "rho": Density of each segment [kg/m^3]
                - "g": Gravitational acceleration vector [m/s^2]
                - "E": Elastic modulus of each segment [Pa]
                - "G": Shear modulus of each segment [Pa]
                - "D": Damping matrix of each segment
        """
        # Apply updates sequentially
        updated_self = self
        
        if "th0" in params:
            updated_self = eqx.tree_at(lambda m: m.th0, updated_self, jnp.asarray(params["th0"], dtype=jnp.float64))
        
        if "g" in params:
            g = jnp.asarray(params["g"], dtype=jnp.float64)
            updated_self = eqx.tree_at(lambda m: m.g, updated_self, jnp.concatenate([jnp.zeros(1), g]))
        
        if "L" in params:
            L = jnp.asarray(params["L"], dtype=jnp.float64)
            L_cum = jnp.cumsum(jnp.concatenate([jnp.zeros(1), L]))
            updated_self = eqx.tree_at(lambda m: (m.L, m.L_cum), updated_self, (L, L_cum))
        
        if "r" in params:
            updated_self = eqx.tree_at(lambda m: m.r, updated_self, jnp.asarray(params["r"], dtype=jnp.float64))
        
        if "rho" in params:
            updated_self = eqx.tree_at(lambda m: m.rho, updated_self, jnp.asarray(params["rho"], dtype=jnp.float64))
        
        if "E" in params:
            updated_self = eqx.tree_at(lambda m: m.E, updated_self, jnp.asarray(params["E"], dtype=jnp.float64))
        
        if "G" in params:
            updated_self = eqx.tree_at(lambda m: m.G, updated_self, jnp.asarray(params["G"], dtype=jnp.float64))
        
        if "D" in params:
            updated_self = eqx.tree_at(lambda m: m.D, updated_self, jnp.asarray(params["D"], dtype=jnp.float64))
        
        return updated_self

    @eqx.filter_jit
    def classify_segment(
        self,
        s: Array,
    ) -> Tuple[Array, Array]:
        """
        Classify the point along the robot to the corresponding segment.

        Args:
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            segment_idx (Array): index of the segment where the point is located
            s_segment (Array): point coordinate along the segment in the interval [0, l_segment]
        """

        # Classify the point along the robot to the corresponding segment
        segment_idx = jnp.clip(jnp.sum(s > self.L_cum) - 1, 0, self.num_segments - 1)

        # Compute the point coordinate along the segment in the interval [0, l_segment]
        s_local = s - self.L_cum[segment_idx]

        return segment_idx, s_local

    @eqx.filter_jit
    def strain(
        self,
        q: Array,
    ) -> Array:
        """
        Compute the strain vector from the generalized coordinates.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            xi (Array): strain vector of shape (num_active_strains,)
        """
        xi = self.B_xi @ q + self.xi_star

        return xi

    @eqx.filter_jit
    def chi(
        self,
        xi: Array,
        s: Array,
    ) -> Array:
        """
        Compute the forward kinematics of the robot.

        Args:
            xi (Array): strain vector of shape (3*num_segments,) where each row corresponds to a segment
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            chi_s (Array): forward kinematics of the robot at point s, shape (3,) : [theta, x, y]
        """
        xi = xi.reshape(self.num_segments, 3)

        segment_idx, s_local = self.classify_segment(s)

        chi_0 = jnp.concatenate(
            [self.th0[None], jnp.zeros(2)]
        )  # Initial configuration [theta, x, y]

        # Iteration function
        def chi_i(chi_prev: Array, i: int) -> Tuple[Array, Array]:
            th_prev = chi_prev[0]
            p_prev = chi_prev[1:]

            kappa_i = xi[i, 0]
            sigmas_i = xi[i, 1:]

            l_i = jnp.where(i == segment_idx, s_local, self.L[i])

            th = th_prev + kappa_i * l_i

            int_cos_th = jnp.where(
                jnp.abs(kappa_i) < self.global_eps,
                l_i * jnp.cos(th_prev),
                (jnp.sin(th) - jnp.sin(th_prev)) / kappa_i,
            )
            int_sin_th = jnp.where(
                jnp.abs(kappa_i) < self.global_eps,
                l_i * jnp.sin(th_prev),
                -(jnp.cos(th) - jnp.cos(th_prev)) / kappa_i,
            )

            R = jnp.stack(
                [
                    jnp.stack([int_cos_th, -int_sin_th]),
                    jnp.stack([int_sin_th, int_cos_th]),
                ]
            )

            p = p_prev + R @ sigmas_i

            chi = jnp.concatenate([th[None], p])

            return chi, chi

        _, chi_list = lax.scan(f=chi_i, init=chi_0, xs=jnp.arange(self.num_segments))

        chi_s = chi_list[segment_idx]

        return chi_s

    @eqx.filter_jit
    def forward_kinematics(
        self,
        q: Array,
        s: Array,
    ) -> Array:
        """
        Compute the forward kinematics of the robot at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            chi (Array): forward kinematics of the robot at point s, shape (3,) : [theta, x, y]
        """
        xi = self.strain(q)

        chi = self.chi(xi, s)

        return chi

    def _J_local(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            _J_local (Array): Jacobian of the forward kinematics at point s, shape (num_segments, 3, 3)
            where each row corresponds to a segment.
        """
        xi = self.strain(q).reshape(self.num_segments, 3)

        # Classify the point along the robot to the corresponding segment
        segment_idx, s_local = self.classify_segment(s)

        # Initial condition
        xi_0 = xi[0]
        L_0 = self.L[0]

        Ad_g0_inv_L0 = lie.Adjoint_gi_se2_inv(xi_0, L_0, eps=self.global_eps)
        Ad_g0_inv_s = lie.Adjoint_gi_se2_inv(xi_0, s_local, eps=self.global_eps)

        T_g0_L0 = lie.Tangent_gi_se2(xi_0, L_0, eps=self.global_eps)
        T_g0_s = lie.Tangent_gi_se2(xi_0, s_local, eps=self.global_eps)

        mat_0_L0 = Ad_g0_inv_L0 @ T_g0_L0
        mat_0_s = Ad_g0_inv_s @ T_g0_s

        J_0_L0 = jnp.concatenate(
            [mat_0_L0[None, :, :], jnp.zeros((self.num_segments - 1, 3, 3))], axis=0
        )
        J_0_s = jnp.concatenate(
            [mat_0_s[None, :, :], jnp.zeros((self.num_segments - 1, 3, 3))], axis=0
        )

        tuple_J_0 = (J_0_L0, J_0_s)

        # Iteration function
        def J_i(tuple_J_prev: Array, i: int) -> Tuple[Tuple[Array, Array], Array]:
            J_prev_Lprev, _ = tuple_J_prev

            xi_i = xi[i]

            Ad_gi_inv_Li = lie.Adjoint_gi_se2_inv(xi_i, self.L[i], eps=self.global_eps)
            Ad_gi_inv_s = lie.Adjoint_gi_se2_inv(xi_i, s_local, eps=self.global_eps)

            T_gi_Li = lie.Tangent_gi_se2(xi_i, self.L[i], eps=self.global_eps)
            T_gi_s = lie.Tangent_gi_se2(xi_i, s_local, eps=self.global_eps)

            mat_i_Li = Ad_gi_inv_Li @ T_gi_Li
            mat_i_s = Ad_gi_inv_s @ T_gi_s

            J_i_s = lax.dynamic_update_slice(
                jnp.einsum("ij, njk->nik", Ad_gi_inv_s, J_prev_Lprev),
                mat_i_s[jnp.newaxis, ...],
                (i, 0, 0),
            )
            J_i_Li = lax.dynamic_update_slice(
                jnp.einsum("ij, njk->nik", Ad_gi_inv_Li, J_prev_Lprev),
                mat_i_Li[jnp.newaxis, ...],
                (i, 0, 0),
            )

            return (J_i_Li, J_i_s), J_i_s

        indices_links = jnp.arange(1, self.num_segments)

        _, J_array = lax.scan(f=J_i, init=tuple_J_0, xs=indices_links)

        # Add the initial condition to the Jacobian array
        J_array = jnp.concatenate([J_0_s[jnp.newaxis, ...], J_array], axis=0)

        # Extract the Jacobian for the segment that contains the point s
        _J_local = lax.dynamic_index_in_dim(
            J_array, segment_idx, axis=0, keepdims=False
        )

        return _J_local

    def _final_size_jacobian(self, J_full: Array) -> Array:
        """
        Convert the Jacobian or its derivative from the full computation form to the selected strains form.

        Args:
            J_full (Array): Full Jacobian of shape (num_segments, 3, 3)

        Returns:
            J_selected (Array): Jacobian for the selected strains of shape (3, num_strains)
        """
        J_final = J_full.transpose(1, 0, 2).reshape(3, self.num_strains)

        return J_final

    def _jacobian_bodyframe_full(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a point s along the robot in the body frame (for every strains)

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_local (Array): Jacobian of the forward kinematics at point s in the body frame, shape (3, num_strains)
        """
        _J_local = self._J_local(q, s)

        J_local = self._final_size_jacobian(_J_local)

        return J_local

    @eqx.filter_jit
    def jacobian_bodyframe(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a point s along the robot in the body frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_local (Array): Jacobian of the forward kinematics at point s in the body frame, shape (3, num_active_strains)
        """
        _J_local = self._J_local(q, s)

        J_local = self._final_size_jacobian(_J_local) @ self.B_xi

        return J_local

    @eqx.filter_jit
    def jacobian_inertialframe(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a point s along the robot in the inertial frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_global (Array): Jacobian of the forward kinematics at point s in the inertial frame, shape (3, num_active_strains)
        """
        _J_local = self._J_local(q, s)

        chi = self.forward_kinematics(q, s)
        theta = chi[0]
        g_i = lie.exp_SE2(
            jnp.stack([theta, 0.0, 0.0])
        )  # SE(2) transformation at point s
        Adj_gi = lie.Adjoint_g_SE2(
            g_i
        )  # Adjoint representation of the SE(2) transformation

        _J_global = jnp.einsum(
            "ij, njk -> nik",
            Adj_gi,
            _J_local,
        )

        J_global = self._final_size_jacobian(_J_global) @ self.B_xi

        return J_global

    def _J_Jd(self, q: Array, qd: Array, s: Array) -> Tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            _J_local (Array): Jacobian of the forward kinematics at point s, shape (num_segments, 3, 3)
            _J_d_local (Array): Time-derivative of the Jacobian at point s, shape (num_segments, 3, 3)
        """
        xi_d = (self.B_xi @ qd).reshape(self.num_segments, 3)

        # Classify the point along the robot to the corresponding segment
        segment_idx, _ = self.classify_segment(s)

        _J_local = self._J_local(q, s)

        # =================================
        # Computation of the time-derivative of the Jacobian

        idx_range = jnp.arange(self.num_segments)
        J_i = vmap(
            lambda i: lax.dynamic_index_in_dim(_J_local, i, axis=0, keepdims=False)
        )(idx_range)  # shape: (num_segments, 3, 3)
        sum_Jj_xi_d_j = compute_weighted_sums(
            _J_local, xi_d, self.num_segments
        )  # shape: (num_segments, 3)
        adjoint_sum = vmap(lie.adjoint_se2)(
            sum_Jj_xi_d_j
        )  # shape: (num_segments, 3, 3)

        # Compute the time-derivative of the Jacobian
        _J_d_local = jnp.einsum(
            "ijk, ikl->ijl", adjoint_sum, J_i
        )  # shape: (num_segments, 3, 3)

        # Replace the elements of J_d_segment_SE2 for i > segment_idx by null matrices
        _J_d_local = jnp.where(
            jnp.arange(self.num_segments)[:, None, None] > segment_idx,
            jnp.zeros_like(_J_d_local),
            _J_d_local,
        )

        return _J_local, _J_d_local

    @eqx.filter_jit
    def jacobian_and_derivative_bodyframe(
        self, q: Array, qd: Array, s: Array
    ) -> Tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at a point s along the robot in the body frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_local (Array): Jacobian of the forward kinematics at point s in the body frame, shape (3, num_active_strains)
            J_d_local (Array): Time-derivative of the Jacobian at point s in the body frame, shape (3, num_active_strains)
        """
        _J_local, _J_d_local = self._J_Jd(q, qd, s)

        J_local = self._final_size_jacobian(_J_local) @ self.B_xi
        J_d_local = self._final_size_jacobian(_J_d_local) @ self.B_xi

        return J_local, J_d_local

    @eqx.filter_jit
    def jacobian_and_derivative_inertialframe(
        self, q: Array, qd: Array, s: Array
    ) -> Tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at a point s along the robot in the inertial frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_global (Array): Jacobian of the forward kinematics at point s in the inertial frame, shape (3, num_active_strains)
            J_d_global (Array): Time-derivative of the Jacobian at point s in the inertial frame, shape (3, num_active_strains)
        """
        _J_local, _J_d = self._J_Jd(q, qd, s)

        chi = self.forward_kinematics(q, s)
        theta = chi[0]
        g_i = lie.exp_SE2(
            jnp.stack([theta, 0.0, 0.0])
        )  # SE(2) transformation at point s
        Adj_gi = lie.Adjoint_g_SE2(g_i)

        _J_global = jnp.einsum(
            "ijk, ikl -> ijl",
            Adj_gi,
            _J_local,
        )
        _J_d = jnp.einsum(
            "ijk, ikl -> ijl",
            Adj_gi,
            _J_d,
        )

        J_global = self._final_size_jacobian(_J_global) @ self.B_xi
        J_d_global = self._final_size_jacobian(_J_d) @ self.B_xi

        return J_global, J_d_global

    @eqx.filter_jit
    def jacobian(
        self,
        q: Array,
        s: Array,
    ) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a point s along the robot in the inertial frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_global (Array): Jacobian of the forward kinematics at point s in the inertial frame, shape (3, num_active_strains)
        """
        J_global = self.jacobian_inertialframe(q, s)

        return J_global

    @eqx.filter_jit
    def jacobian_and_derivative(
        self,
        q: Array,
        qd: Array,
        s: Array,
    ) -> Tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at a point s along the robot in the inertial frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_global (Array): Jacobian of the forward kinematics at point s in the inertial frame, shape (3, num_active_strains)
            J_d_global (Array): Time-derivative of the Jacobian at point s in the inertial frame, shape (3, num_active_strains)
        """
        J_global, J_d_global = self.jacobian_and_derivative_inertialframe(q, qd, s)

        return J_global, J_d_global

    # ==========================================
    # Useful functions for the system

    def _local_cross_sectional_area(self, i: int) -> Array:
        """
        Compute the local cross-sectional area for the i-th segment.

        Args:
            i (int): index of the segment

        Returns:
            A_i (Array): local cross-sectional area of the i-th segment
        """
        A_i = jnp.pi * self.r[i] ** 2  # Cross-sectional area
        return A_i

    def _local_mass_matrix(self, i: int) -> Array:
        """
        Compute the local mass matrix for the i-th segment.

        Args:
            i (int): index of the segment
        Returns:
            M_i (Array): local mass matrix of shape (3, 3) for the i-th segment
        """
        rho_i = self.rho[i]
        A_i = self._local_cross_sectional_area(i)  # Cross-sectional area
        I_i = A_i**2 / (4 * jnp.pi)  # Second moment of area

        M_i = rho_i * jnp.diag(jnp.array([I_i, A_i, A_i]))
        return M_i

    # ===========================================
    # Dynamical matrices computation

    def _inertia_full_matrix(
        self,
        q: Array,
    ) -> Array:
        """
        Compute the full inertia matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            B_full (Array): Full inertia matrix of shape (num_strains, num_strains).
        """

        def B_i(i):
            Xs_scaled, Ws_scaled = scale_gaussian_quadrature(
                self.Xs, self.Ws, self.L_cum[i], self.L_cum[i + 1]
            )
            M_i = self._local_mass_matrix(i)

            def B_j(j):
                Xs_j = Xs_scaled[j]
                Ws_j = Ws_scaled[j]
                J_j = self._jacobian_bodyframe_full(q, Xs_j)
                return Ws_j * J_j.T @ M_i @ J_j

            B_blocks_i = vmap(B_j)(jnp.arange(self.num_gauss_points))

            # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
            # B_blocks_i = jnp.stack([B_j(j) for j in range(self.num_gauss_points)], axis=0)

            return B_blocks_i

        B_blocks_tot = vmap(B_i)(jnp.arange(self.num_segments))

        # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
        # B_blocks_tot = jnp.stack([B_i(i) for i in range(self.num_segments)], axis=0)

        B_full = jnp.sum(
            B_blocks_tot, axis=(0, 1)
        )  # Sum over segments and Gauss points

        return B_full

    @eqx.filter_jit
    def inertia_matrix(
        self,
        q: Array,
    ) -> Array:
        """
        Compute the inertia matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            B (Array): Inertia matrix of shape (num_active_strains, num_active_strains).
        """
        B_full = self._inertia_full_matrix(q)

        B = self.B_xi.T @ B_full @ self.B_xi

        return B

    def _coriolis_full_matrix(
        self,
        q: Array,
        qd: Array,
    ) -> Array:
        """
        Compute the full Coriolis matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).

        Returns:
            C_full (Array): Full Coriolis matrix of shape (num_strains, num_strains).
        """

        def C_i(i):
            Xs_scaled, Ws_scaled = scale_gaussian_quadrature(
                self.Xs, self.Ws, self.L_cum[i], self.L_cum[i + 1]
            )
            M_i = self._local_mass_matrix(i)

            def C_j(j):
                Xs_j = Xs_scaled[j]
                Ws_j = Ws_scaled[j]
                J_j, J_d_j = self.jacobian_and_derivative_bodyframe(q, qd, Xs_j)
                return Ws_j * (
                    J_j.T @ (M_i @ J_d_j + lie.coadjoint_se2(J_j @ qd) @ M_i @ J_j)
                )

            C_blocks_i = vmap(C_j)(jnp.arange(self.num_gauss_points))

            return C_blocks_i

        C_blocks_tot = vmap(C_i)(jnp.arange(self.num_segments))

        C_full = jnp.sum(C_blocks_tot, axis=(0, 1))

        return C_full

    @eqx.filter_jit
    def coriolis_matrix(
        self,
        q: Array,
        qd: Array,
    ) -> Array:
        """
        Compute the Coriolis matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).

        Returns:
            C (Array): Coriolis matrix of shape (num_active_strains, num_active_strains).
        """
        C_full = self._coriolis_full_matrix(q, qd)

        C = self.B_xi.T @ C_full @ self.B_xi

        return C

    def _gravitational_force_full(
        self,
        q: Array,
    ) -> Array:
        """
        Compute the full gravitational force acting on the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            G (Array): Full gravitational force of shape (num_strains,).
        """

        def G_i(i):
            Xs_scaled, Ws_scaled = scale_gaussian_quadrature(
                self.Xs, self.Ws, self.L_cum[i], self.L_cum[i + 1]
            )
            M_i = self._local_mass_matrix(i)

            def G_j(j):
                Xs_j = Xs_scaled[j]
                Ws_j = Ws_scaled[j]
                Ad_g_inv_j = lie.Adjoint_g_inv_SE2(
                    lie.exp_SE2(self.forward_kinematics(q, Xs_j))
                )
                J_j = self._jacobian_bodyframe_full(q, Xs_j)

                return Ws_j * J_j.T @ M_i @ Ad_g_inv_j @ self.g

            G_blocks_segment_i = vmap(G_j)(jnp.arange(self.num_gauss_points))

            # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
            # G_blocks_segment_i = jnp.stack(
            #     [G_j(j) for j in range(self.num_gauss_points)], axis=0
            # )

            return G_blocks_segment_i

        G_blocks_tot = vmap(G_i)(jnp.arange(self.num_segments))

        # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
        # G_blocks_tot = jnp.stack(
        #     [G_i(i) for i in range(self.num_segments)], axis=0
        # )

        G_full = jnp.sum(
            G_blocks_tot, axis=(0, 1)
        )  # Sum over links and quadrature points

        return G_full

    @eqx.filter_jit
    def gravitational_force(
        self,
        q: Array,
    ) -> Array:
        """
        Compute the gravitational force acting on the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            G (Array): Gravitational force of shape (num_active_strains,).
        """
        G_full = self._gravitational_force_full(q)

        G = self.B_xi.T @ G_full

        return G
    
    def _stiffness(
        self, formulate_in_strain_space: bool = False,
    ) -> Array:
        # cross-sectional area and second moment of area
        A = jnp.pi * self.r**2
        Ib = A**2 / (4 * jnp.pi)

        # stiffness matrix of shape (num_segments, 3, 3)
        S_sms = compute_stiffness_matrix_for_all_segments_fn(self.L, A, Ib, self.E, self.G)
        # we define the elastic matrix of shape (num_strains, num_strains) as K(xi) = K @ xi where K is equal to
        S = blk_diag(S_sms)

        if not formulate_in_strain_space:
            S = self.B_xi.T @ S @ self.B_xi

        return S

    def _stiffness_full_matrix(
        self,
    ) -> Array:
        """
        Compute the full stiffness matrix of the robot.

        Returns:
            K_full (Array): Full stiffness matrix of shape (num_strains, num_strains).
        """
        K_full = self._stiffness(formulate_in_strain_space=True)

        return K_full

    @eqx.filter_jit
    def stiffness_matrix(
        self,
    ) -> Array:
        """
        Compute the stiffness matrix of the robot.

        Returns:
            K (Array): Stiffness matrix of shape (num_active_strains, num_active_strains).
        """
        K = self._stiffness(formulate_in_strain_space=False)

        return K
    
    @eqx.filter_jit
    def elastic_force(self, q: Array) -> Array:
        """
        Compute the elastic forces of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            tau_el (Array): Elastic force of shape (num_active_strains,).
        """
        K = self.stiffness_matrix()
        tau_el = K @ q

        return tau_el

    def _damping_full_matrix(
        self,
    ) -> Array:
        """
        Compute the full damping matrix of the robot.

        Args:
            None

        Returns:
            D (Array): Full damping matrix of shape (num_strains, num_strains).
        """
        D_full = self.D

        return D_full

    @eqx.filter_jit
    def damping_matrix(
        self,
    ) -> Array:
        """
        Compute the damping matrix of the robot.

        Returns:
            D (Array): Damping matrix of shape (num_active_strains, num_active_strains).
        """
        D_full = self._damping_full_matrix()

        D = self.B_xi.T @ D_full @ self.B_xi

        return D

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Compute the actuation matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            A (Array): Actuation matrix of shape (num_active_strains, num_actuators).
        """
        A = jnp.identity(self.num_actuators)
        return A

    @eqx.filter_jit
    def actuation_force(
        self,
        q: Array,
        u: Array,
    ) -> Array:
        """
        Compute the actuation force acting on the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            u (Array): actuation/control input of shape (num_actuators,).

        Returns:
            tau_u (Array): Actuation force of shape (num_active_strains, ).
        """
        # evaluate the actuation matrix
        A = self.actuation_matrix(q)

        # compute the actuation mapping
        tau_u = A @ u

        return tau_u

    @eqx.filter_jit
    def kinetic_energy(
        self,
        q: Array,
        qd: Array,
    ) -> Array:
        """
        Compute the kinetic energy of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).

        Returns:
            T (float): Kinetic energy of the robot.
        """
        B = self.inertia_matrix(q)
        T = 0.5 * qd.T @ B @ qd

        return T

    @eqx.filter_jit
    def elastic_energy(
        self,
        q: Array,
    ) -> Array:
        """
        Compute the elastic energy of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            U_K (float): Elastic energy of the robot.
        """
        K_full = self._stiffness_full_matrix()
        U_K = 0.5 * (self.B_xi @ q).T @ K_full @ (self.B_xi @ q)

        return U_K

    @eqx.filter_jit
    def gravitational_energy(
        self,
        q: Array,
    ) -> Array:
        """
        Compute the gravitational energy of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            U_G (float): Gravitational energy of the robot.
        """

        def U_G_i(i):
            Xs_scaled, Ws_scaled = scale_gaussian_quadrature(
                self.Xs, self.Ws, self.L_cum[i], self.L_cum[i + 1]
            )
            rho_i = self.rho[i]
            A_i = self._local_cross_sectional_area(i)  # Cross-sectional area

            def U_G_j(j):
                Xs_j = Xs_scaled[j]
                Ws_j = Ws_scaled[j]
                p_j = (
                    self.forward_kinematics(q, Xs_j).at[0].set(0.0)
                )  # Set the orientation angle to 0 for gravitational energy computation
                return Ws_j * rho_i * A_i * jnp.dot(p_j, self.g)

            U_G_blocks_segment_i = vmap(U_G_j)(jnp.arange(self.num_gauss_points))

            return U_G_blocks_segment_i

        U_G_blocks_tot = vmap(U_G_i)(jnp.arange(self.num_segments))

        U_G = jnp.sum(U_G_blocks_tot, axis=(0, 1))  # Sum over segments and Gauss points

        return U_G

    @eqx.filter_jit
    def potential_energy(
        self,
        q: Array,
    ) -> Array:
        """
        Compute the potential energy of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            U (float): Potential energy of the robot.
        """
        U_K = self.elastic_energy(q)
        U_G = self.gravitational_energy(q)

        return U_K + U_G

    @eqx.filter_jit
    def total_energy(
        self,
        q: Array,
        qd: Array,
    ) -> Array:
        """
        Compute the total energy of the robot, which is the sum of kinetic and potential energy.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).

        Returns:
            E (float): Total energy of the robot.
        """
        T = self.kinetic_energy(q, qd)
        U = self.potential_energy(q)
        E = T + U
        return E

    @eqx.filter_jit
    def operational_space_dynamical_matrices(
        self,
        q: Array,
        qd: Array,
        s: Array,
        operational_space_selector: Tuple = (True, True, True),
    ) -> Tuple[Array, Array, Array, Array, Array]:
        """
        Compute the operational space dynamical matrices for the robot at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].
            operational_space_selector (Tuple): Selector for the operational space dimensions.
                Default is (True, True, True) for all dimensions.

        Returns:
            Lambda (Array): Inertia matrix in the operational space, shape (num_operational_space_dims, num_operational_space_dims).
            mu (Array): Coriolis and centrifugal matrix in the operational space, shape (num_operational_space_dims,).
            J (Array): Jacobian of the forward kinematics at point s in the body frame, shape (num_operational_space_dims, num_active_strains).
            J_d (Array): Time-derivative of the Jacobian at point s in the body frame, shape (num_operational_space_dims, num_active_strains).
            JB_pinv (Array): Dynamically-consistent pseudo-inverse of the Jacobian, shape (num_active_strains, num_operational_space_dims).
        """
        # classify the point along the robot to the corresponding segment
        _, s_local = self.classify_segment(s)

        # make operational_space_selector a boolean array
        operational_space_selector = onp.array(operational_space_selector, dtype=bool)

        # Jacobian and its time-derivative
        J, J_d = self.jacobian_and_derivative_bodyframe(q, qd, s_local)

        J = J[operational_space_selector, :]
        J_d = J_d[operational_space_selector, :]

        # inverse of the inertia matrix in the configuration space
        B = self.inertia_matrix(q)
        B_inv = jnp.linalg.inv(B)
        C = self.coriolis_matrix(q, qd)

        Lambda = jnp.linalg.inv(
            J @ B_inv @ J.T
        )  # inertia matrix in the operational space
        mu = Lambda @ (
            J @ B_inv @ C - J_d
        )  # coriolis and centrifugal matrix in the operational space

        JB_pinv = (
            B_inv @ J.T @ Lambda
        )  # dynamically-consistent pseudo-inverse of the Jacobian

        return Lambda, mu, J, J_d, JB_pinv

    @eqx.filter_jit
    def forward_dynamics(
        self,
        t: float,
        y: Array,
        actuation_args: Optional[Tuple] = None,
    ) -> Array:
        """
        Forward dynamics function.

        Args:
            t (float): Current time.
            y (Array): State vector containing configuration and velocity.
                Shape is (2 * num_strains,).
            actuation_args (Tuple, optional): Additional arguments for the actuation mapping function.
                Default is None.
        Returns:
            y_d: Time derivative of the state vector.
        """
        # Split the state vector into configuration and velocity
        q, qd = jnp.split(y, 2)

        # split the actuation arguments if provided
        if actuation_args is None:
            u, tau_ext = None, None
        elif len(actuation_args) == 1:
            u = actuation_args[0]
            tau_ext = None
        elif len(actuation_args) == 2:
            u, tau_ext = actuation_args
        else:
            raise ValueError("actuation_args must be a tuple of length 1 or 2.")
        
        if u is None:
            u = jnp.zeros((self.num_actuators, ))
        if tau_ext is None:
            tau_ext = jnp.zeros((q.shape[-1], ))

        # evaluate the dynamical matrices
        B = self.inertia_matrix(q)
        C = self.coriolis_matrix(q, qd)
        G = self.gravitational_force(q)
        D = self.damping_matrix()
        tau_el = self.elastic_force(q)
        tau_u = self.actuation_force(q, u)

        B_inv = jnp.linalg.inv(B)  # Inverse of the inertia matrix
        qdd = B_inv @ (tau_u + tau_ext - C @ qd - G - tau_el - D @ qd)  # Compute the acceleration

        y_d = jnp.concatenate([qd, qdd])

        return y_d

    @eqx.filter_jit
    def resolve_upon_time(
        self,
        q0: Array,
        qd0: Array,
        u: Optional[Array] = None,
        tau_ext: Optional[Array] = None,
        t0: Optional[float] = 0.0,
        t1: Optional[float] = 10.0,
        dt: Optional[float] = 1e-4,
        skip_steps: Optional[int] = 0,
        solver: Optional[AbstractSolver] = Tsit5(),
        stepsize_controller: Optional[PIDController] = ConstantStepSize(),
        max_steps: Optional[int] = None,
    ) -> Tuple[Array, Array, Array]:
        """
        Resolve the system dynamics over time using Diffrax.

        Args:
            q0 (Array): Initial configuration (strains).
            qd0 (Array): Initial velocity (strains).
            u (Array, optional): Actuation/control input.
                Default is None (no actuation).
            tau_ext (Array, optional): External forces/torques applied to the system.
            t0 (float, optionnal): Initial time.
                Default is 0.0.
            t1 (float, optionnal): Final time.
                Default is 10.0.
            dt (float, optionnal): Time step for the solver.
                Default is 1e-4.
            skip_steps (int, optionnal): Number of steps to skip in the output.
                This allows to reduce the number of saved time points.
                Default is 0.
            solver (AbstractSolver, optional): Solver to use for the ODE integration.
                Default is Tsit5() (Runge-Kutta 5(4) method).
            stepsize_controller (PIDController, optional): Stepsize controller for the solver.
                Default is ConstantStepSize().
            max_steps (int, optional): Maximum number of steps for the solver.
                Default is None (no limit).

        Returns:
            ts (Array): Time points at which the solution is saved.
            qs (Array): Configuration (strains) at the saved time points.
            qds (Array): Velocity (strains) at the saved time points.
        """
        y0 = jnp.concatenate([q0, qd0])  # Initial state vector
        if u is None:
            u = jnp.zeros((self.num_actuators, ))
        if tau_ext is None:
            tau_ext = jnp.zeros((q0.shape[-1], ))

        term = ODETerm(self.forward_dynamics)

        t = jnp.arange(t0, t1, dt)  # Time points for the solution
        saveat = SaveAt(ts=t[::skip_steps])  # Save at specified time points

        sol = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t[0],
            t1=t[-1],
            dt0=dt,
            y0=y0,
            args=(u, tau_ext),
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )

        ts = sol.ts
        # Extract the configuration and velocity from the solution
        y_out = sol.ys
        qs, qds = jnp.split(y_out, 2, axis=1)

        return ts, qs, qds
