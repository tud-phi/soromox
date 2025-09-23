__all__ = ["PlanarPCS"]
import equinox as eqx
import jax
from jax import Array, lax, vmap
from jax import numpy as jnp
import numpy as onp
from typing import Callable, Dict, Tuple, Optional, ClassVar

from soromox.systems.dynamical_system import DynamicalSystem
from soromox.utils.array_math import (
    blk_diag,
    compute_weighted_sums,
)
from soromox.utils.basic import (
    compute_strain_basis,
)
from soromox.utils.integration import (   
    gauss_quadrature,
    scale_gaussian_quadrature,
)
import soromox.utils.lie_algebra as lie


class PlanarPCS(DynamicalSystem):
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
    xi_ref : Array
        Reference strain (reference configuration) of the robot.
    num_gauss_points : int
        Number of points used for numerical integration.
        Corresponds to the order of Gauss-Legendre quadrature + 2 (for the endpoints).
    Xs, Ws : Array
        Gauss-Legendre quadrature nodes and weights for numerical integration.

    Notes:
    -----
    - The strain vector is composed of 3 components per segment:
      [kappa_z, sigma_x, sigma_y].
      By default, the rod is assumed to be straight and aligned with the x-axis,
        so the reference strain is set to [0, 1, 0].
        Thus:   - kappa_z corresponds to bending around the z-axis,
                - sigma_x corresponds to axial strain along the x-axis,
                - sigma_y corresponds to shear along the y-axis.

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

    # Not a dataclass field: avoid default/non-default ordering issues in subclasses
    global_eps: ClassVar[float] = float(jnp.finfo(jnp.float64).eps)

    num_segments: int = eqx.field(static=True)
    num_gauss_points: int = eqx.field(static=True)  #
    num_strains: int = eqx.field(static=True)  # Number of strains (3 * num_segments)

    xi_ref: Array  # Reference configuration strain
    B_xi: Array  # Strain basis matrix
    num_active_strains: Array  # Number of selected strains

    Xs: Array  # Gauss nodes
    Ws: Array  # Gauss weights

    def __init__(
        self,
        num_segments: int,
        params: Dict[str, Array],
        order_gauss: int = 5,
        strain_selector: Optional[Array] = None,
        xi_ref: Optional[Array] = None,
    ):
        """
        Initialize the PlanarPCS class.

        Args:
            num_segments (int):
                Number of segments in the robot.
            params (Dict[str, Array]):
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
            order_gauss (int, optional):
                Order of the Gauss-Legendre quadrature for integration over each segment.
                Defaults to 5.
            strain_selector (Optional[Array], optional):
                Boolean array of shape (3 * num_segments,) specifying which strain components are active.
                Defaults to all strains active (i.e. all True).
            xi_ref (Optional[Array], optional):
                Reference strain of shape (3 * num_segments,).
                Defaults to 0.0 for bending and shear strains, and 1.0 for axial strain (along local x-axis).

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

        # Reference configuration strain
        if xi_ref is None:
            xi_ref = jnp.tile(
                jnp.array([0.0, 1.0, 0.0], dtype=jnp.float64), (self.num_segments, 1)
            ).reshape(self.num_strains)
        else:
            if not isinstance(xi_ref, (list, jnp.ndarray)):
                raise TypeError(
                    f"xi_ref must be a list or an array, got {type(xi_ref).__name__}"
                )
            xi_ref = jnp.asarray(xi_ref)
            if xi_ref.size != self.num_strains:
                raise ValueError(
                    f"xi_ref must have {self.num_strains} elements, got {xi_ref.size}"
                )
            xi_ref = xi_ref.reshape(self.num_strains)
        self.xi_ref = xi_ref

        # Number of actuators
        self.num_actuators = int(self.num_active_strains.item())

    def _set_params(self, params: Dict[str, Array]) -> None:
        """
        Set the parameters of the PCS model.

        Args:
            params (Dict[str, Array]):
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
        """
        # Initial orientation angle
        th0 = params.get("th0", jnp.pi / 2.0)  # Default to 90 degrees
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
            raise ValueError(
                f"rho must have shape ({self.num_segments},), got {rho.shape}"
            )
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

        Returns:
            updated_self (PlanarPCS):
                A new instance of PlanarPCS with updated parameters.
        """
        # Apply updates sequentially
        updated_self = self

        if "th0" in params:
            th0 = params["th0"]
            if not (isinstance(th0, (float, int, jnp.ndarray))):
                raise TypeError(
                    f"th0 must be a float, int, or an array, got {type(th0).__name__}"
                )
            th0 = jnp.asarray(th0, dtype=jnp.float64)
            updated_self = eqx.tree_at(lambda m: m.th0, updated_self, th0)

        if "g" in params:
            g = params["g"]
            if not (isinstance(g, (list, jnp.ndarray))):
                raise TypeError(f"g must be a list or an array, got {type(g).__name__}")
            g = jnp.asarray(g, dtype=jnp.float64)
            if g.size != 2:
                raise ValueError(f"g must be a vector of shape (2,), got {g.size}")
            updated_self = eqx.tree_at(
                lambda m: m.g, updated_self, jnp.concatenate([jnp.zeros(1), g])
            )

        if "L" in params:
            L = params["L"]
            if not (isinstance(L, (list, jnp.ndarray))):
                raise TypeError(f"L must be a list or an array, got {type(L).__name__}")
            L = jnp.asarray(L, dtype=jnp.float64)
            if L.shape != (self.num_segments,):
                raise ValueError(
                    f"L must have shape ({self.num_segments},), got {L.shape}"
                )
            L_cum = jnp.cumsum(jnp.concatenate([jnp.zeros(1), L]))
            updated_self = eqx.tree_at(
                lambda m: (m.L, m.L_cum), updated_self, (L, L_cum)
            )

        if "r" in params:
            r = params["r"]
            if not (isinstance(r, (list, jnp.ndarray))):
                raise TypeError(f"r must be a list or an array, got {type(r).__name__}")
            r = jnp.asarray(r, dtype=jnp.float64)
            if r.shape != (self.num_segments,):
                raise ValueError(
                    f"r must have shape ({self.num_segments},), got {r.shape}"
                )
            updated_self = eqx.tree_at(lambda m: m.r, updated_self, r)

        if "rho" in params:
            rho = params["rho"]
            if not (isinstance(rho, (list, jnp.ndarray))):
                raise TypeError(
                    f"rho must be a list or an array, got {type(rho).__name__}"
                )
            rho = jnp.asarray(rho, dtype=jnp.float64)
            if rho.shape != (self.num_segments,):
                raise ValueError(
                    f"rho must have shape ({self.num_segments},), got {rho.shape}"
                )
            updated_self = eqx.tree_at(lambda m: m.rho, updated_self, rho)

        if "E" in params:
            E = params["E"]
            if not (isinstance(E, (list, jnp.ndarray))):
                raise TypeError(f"E must be a list or an array, got {type(E).__name__}")
            E = jnp.asarray(E, dtype=jnp.float64)
            if E.shape != (self.num_segments,):
                raise ValueError(
                    f"E must have shape ({self.num_segments},), got {E.shape}"
                )
            updated_self = eqx.tree_at(lambda m: m.E, updated_self, E)

        if "G" in params:
            G = params["G"]
            if not (isinstance(G, (list, jnp.ndarray))):
                raise TypeError(f"G must be a list or an array, got {type(G).__name__}")
            G = jnp.asarray(G, dtype=jnp.float64)
            if G.shape != (self.num_segments,):
                raise ValueError(
                    f"G must have shape ({self.num_segments},), got {G.shape}"
                )
            updated_self = eqx.tree_at(lambda m: m.G, updated_self, G)

        if "D" in params:
            D = params["D"]
            if not (isinstance(D, (list, jnp.ndarray))):
                raise TypeError(f"D must be a list or an array, got {type(D).__name__}")
            D = jnp.asarray(D, dtype=jnp.float64)
            expected_D_shape = (self.num_strains, self.num_strains)
            if D.shape != expected_D_shape:
                raise ValueError(f"D must have shape {expected_D_shape}, got {D.shape}")
            updated_self = eqx.tree_at(lambda m: m.D, updated_self, D)

        return updated_self

    @eqx.filter_jit
    def classify_segment(self, s: Array) -> Tuple[Array, Array]:
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
    def strain(self, q: Array) -> Array:
        """
        Compute the strain vector from the generalized coordinates.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            xi (Array): strain vector of shape (num_active_strains,)
        """
        xi = self.B_xi @ q + self.xi_ref

        return xi

    @eqx.filter_jit
    def chi(self, xi: Array, s: Array) -> Array:
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
        def chi_i(chi_prev: Array, i: Array) -> Tuple[Array, Array]:
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
    def forward_kinematics(self, q: Array, s: Array) -> Array:
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

    @eqx.filter_jit
    def _J_local(self, q: Array, s: Array) -> Array:
        """
        Compute the body frame Jacobian of the forward kinematics at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_local (Array): Jacobian of the forward kinematics at point s, shape (num_segments, 3, 3)
            where each row corresponds to a segment.
        """
        xi = self.strain(q).reshape(self.num_segments, 3)

        segment_idx, s_local = self.classify_segment(s)
        segment_idx = segment_idx.astype(jnp.int32)
        s_local = jnp.asarray(s_local, dtype=xi.dtype)

        zeros = jnp.zeros((self.num_segments, 3, 3), dtype=xi.dtype)

        def integrate_segment(J_prev: Array, xi_i: Array, arc_len: Array, idx: Array) -> Array:
            """
            Propagate the Jacobian one segment forward and overwrite the active slice.

            Args:
                J_prev (Array): Jacobian accumulated up to the previous segment, shape (num_segments, 3, 3).
                xi_i (Array): Strain parameters of the current segment, shape (3,).
                arc_len (Array): Integration length used for this evaluation (either segment length or `s_local`).
                idx (Array): Index of the segment being updated.

            Returns:
                J_next: Jacobian including the contribution of the current segment, shape (num_segments, 3, 3).
            """
            # Map the previous Jacobian into the current segment frame using the inverse adjoint.
            Ad_inv = lie.Adjoint_gi_se2_inv(xi_i, arc_len, eps=self.global_eps)
            # Integrate the local tangent contribution for this segment length.
            T = lie.Tangent_gi_se2(xi_i, arc_len, eps=self.global_eps)
            # Rotate the previous Jacobian into the current frame.
            J_rot = jnp.matmul(Ad_inv, J_prev)
            # Overwrite the slice associated with the active segment.
            J_next = J_rot.at[idx].set(Ad_inv @ T)
            return J_next

        zero_output = jnp.zeros((3, 3), dtype=xi.dtype)

        def scan_body(
            carry: Tuple[Array, Array, Array],
            i: Array,
        ) -> Tuple[Tuple[Array, Array, Array], Array]:
            """
            Advance the running Jacobian and capture the value once the target segment is reached.

            Args:
                carry (Tuple[Array, Array, Array]): Tuple storing the running Jacobian, the cached target result,
                    and a boolean flag indicating whether the target segment was already processed.
                i (Array): Current segment index handled by the scan.

            Returns:
                Tuple[Tuple[Array, Array, Array], Array]: Updated carry together with a dummy output required by `lax.scan`.
            """
            J_prev, J_target, done = carry

            def compute_branch(_: None) -> Tuple[Tuple[Array, Array, Array], Array]:
                """
                Integrate the current segment and update the cached result if this is the target.

                Args:
                    _: None: Unused placeholder to comply with `lax.switch` signature.

                Returns:
                    Tuple[Tuple[Array, Array, Array], Array]: Carry with the latest Jacobian state plus dummy scan output.
                """
                xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
                L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)
                arc_len = jnp.where(i == segment_idx, s_local, L_i)

                J_next = integrate_segment(J_prev, xi_i, arc_len, i)

                is_target = i == segment_idx
                J_target_next = jnp.where(is_target, J_next, J_target)
                done_next = jnp.logical_or(done, is_target)

                return (J_next, J_target_next, done_next), zero_output

            def skip_branch(_: None) -> Tuple[Tuple[Array, Array, Array], Array]:
                """
                Keep the cached Jacobian untouched once the target segment has been processed.

                Args:
                    _: None: Unused placeholder required by `lax.switch`.

                Returns:
                    Tuple[Tuple[Array, Array, Array], Array]: Original carry and dummy scan output.
                """
                return (J_prev, J_target, done), zero_output

            branch_index = lax.convert_element_type(done, jnp.int32)

            return lax.switch(
                branch_index,
                (compute_branch, skip_branch),
                operand=None,
            )

        carry_init = (
            zeros,
            zeros,
            jnp.array(False, dtype=jnp.bool_),
        )

        indices = jnp.arange(self.num_segments, dtype=segment_idx.dtype)
        (_, J_local, _), _ = lax.scan(scan_body, carry_init, indices)

        return J_local

    @eqx.filter_jit
    def _final_size_jacobian(self, J_full: Array) -> Array:
        """
        Convert the Jacobian or its derivative from the full computation form to the selected strains form.

        Args:
            J_full (Array): Full Jacobian of shape (num_segments, 3, 3)

        Returns:
            J_final (Array): Jacobian for the selected strains of shape (3, num_strains)
        """
        J_final = J_full.transpose(1, 0, 2).reshape(3, self.num_strains)

        return J_final

    @eqx.filter_jit
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
        J_local_ = self._J_local(q, s)

        J_local = self._final_size_jacobian(J_local_) @ self.B_xi

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
        J_local_ = self._J_local(q, s)

        chi = self.forward_kinematics(q, s)
        theta = chi[0]
        # SE(2) transformation at point s
        g = lie.exp_SE2(jnp.stack([theta, 0.0, 0.0]))
        # Adjoint representation of the SE(2) transformation
        Ad_g = lie.Adjoint_g_SE2(g)

        # Rotate the Jacobian to the inertial frame
        J_global_ = jnp.einsum(
            "ij, njk -> nik",
            Ad_g,
            J_local_,
        )

        # Reduce to the active strains
        J_global = self._final_size_jacobian(J_global_) @ self.B_xi

        return J_global

    @eqx.filter_jit
    def _J_Jd_local(self, q: Array, qd: Array, s: Array) -> Tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_local (Array): Jacobian of the forward kinematics at point s, shape (num_segments, 3, 3)
            Jd_local (Array): Time-derivative of the Jacobian at point s, shape (num_segments, 3, 3)
        """
        xi = self.strain(q).reshape(self.num_segments, 3)
        xid = (self.B_xi @ qd).reshape(self.num_segments, 3)

        segment_idx, s_local = self.classify_segment(s)
        segment_idx = segment_idx.astype(jnp.int32)
        s_local = jnp.asarray(s_local, dtype=xi.dtype)

        zeros = jnp.zeros((self.num_segments, 3, 3), dtype=xi.dtype)

        def integrate_segment(
            J_prev: Array,
            Jd_prev: Array,
            xi_i: Array,
            xid_i: Array,
            arc_len: Array,
            idx: Array,
        ) -> Tuple[Array, Array]:
            """
            Propagate Jacobian/Jacobian rate forward by one segment and insert the local contribution.

            Args:
                J_prev (Array): Jacobian accumulated up to the previous segment, shape (num_segments, 3, 3).
                Jd_prev (Array): Time derivative of the Jacobian up to the previous segment, shape (num_segments, 3, 3).
                xi_i (Array): Strain parameters for the current segment, shape (3,).
                xid_i (Array): Time derivative of the strain for the current segment, shape (3,).
                arc_len (Array): Integration length used for this evaluation (either segment length or `s_local`).
                idx (Array): Index of the segment being updated.

            Returns:
                J_next (Array): Updated Jacobian including the current segment, shape (num_segments, 3, 3).
                Jd_next (Array): Updated Jacobian rate including the current segment, shape (num_segments, 3, 3).
            """
            # Inverse adjoint transformation and tangent integration for the current segment
            Ad_inv = lie.Adjoint_gi_se2_inv(xi_i, arc_len, eps=self.global_eps)
            # Compute the tangent vector and its derivative
            T = lie.Tangent_gi_se2(xi_i, arc_len, eps=self.global_eps)
            Td = lie.Tangent_derivative_gi_se2(xi_i, xid_i, arc_len, eps=self.global_eps)

            # Rotate the previous Jacobian into the current frame and add the local contribution
            J_rot = jnp.matmul(Ad_inv, J_prev)
            # Overwrite the slice associated with the active segment
            J_next = J_rot.at[idx].set(Ad_inv @ T)

            # Compute the bodyframe velocity twist at the end of the current segment
            eta = jnp.matmul(J_next[idx], xid_i)
            # Time-derivative of the inverse adjoint
            Ad_inv_dot = -lie.adjoint_se2(eta) @ Ad_inv

            # Rotate the previous Jacobian rate into the current frame and add the convective term
            Jd_rot = jnp.matmul(Ad_inv, Jd_prev) + jnp.matmul(Ad_inv_dot, J_prev)
            # Overwrite the slice associated with the active segment
            Jd_next = Jd_rot.at[idx].set(Ad_inv_dot @ T + Ad_inv @ Td)

            return J_next, Jd_next

        zero_output = jnp.zeros((3, 3), dtype=xi.dtype)

        def scan_body(
            carry: Tuple[Array, Array, Array, Array, Array],
            i: Array,
        ) -> Tuple[Tuple[Array, Array, Array, Array, Array], Array]:
            """
            Advance Jacobian/Jacobian rate along segments and remember the target-segment tensors.

            Args:
                carry (Tuple[Array, Array, Array, Array, Array]): Running Jacobian, Jacobian rate, cached target tensors,
                    and processed flag for the target segment.
                i (Array): Current segment index handled by the scan.

            Returns:
                Tuple[Tuple[Array, Array, Array, Array, Array], Array]: Updated carry and dummy scan output.
            """
            J_prev, Jd_prev, J_target, Jd_target, done = carry

            def compute_branch(_: None) -> Tuple[Tuple[Array, Array, Array, Array, Array], Array]:
                """
                Integrate the current segment and cache the result if this corresponds to the query index.

                Args:
                    _: None: Unused placeholder required by `lax.switch`.

                Returns:
                    Tuple[Tuple[Array, Array, Array, Array, Array], Array]: Updated carry and dummy scan output.
                """
                xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
                xid_i = lax.dynamic_index_in_dim(xid, i, axis=0, keepdims=False)
                L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)
                arc_len = jnp.where(i == segment_idx, s_local, L_i)

                J_next, Jd_next = integrate_segment(
                    J_prev, Jd_prev, xi_i, xid_i, arc_len, i
                )

                is_target = i == segment_idx
                J_target_next = jnp.where(is_target, J_next, J_target)
                Jd_target_next = jnp.where(is_target, Jd_next, Jd_target)
                done_next = jnp.logical_or(done, is_target)

                return (
                    J_next,
                    Jd_next,
                    J_target_next,
                    Jd_target_next,
                    done_next,
                ), zero_output

            def skip_branch(_: None) -> Tuple[Tuple[Array, Array, Array, Array, Array], Array]:
                """
                Reuse the previously cached tensors after the target segment has been handled.

                Args:
                    _: None: Unused placeholder required by `lax.switch`.

                Returns:
                    Tuple[Tuple[Array, Array, Array, Array, Array], Array]: Original carry and dummy scan output.
                """
                return (J_prev, Jd_prev, J_target, Jd_target, done), zero_output

            branch_index = lax.convert_element_type(done, jnp.int32)

            return lax.switch(
                branch_index,
                (compute_branch, skip_branch),
                operand=None,
            )

        carry_init = (
            zeros,
            zeros,
            zeros,
            zeros,
            jnp.array(False, dtype=jnp.bool_),
        )

        indices = jnp.arange(self.num_segments, dtype=segment_idx.dtype)
        ( _, _, J_local, Jd_local, _ ), _ = lax.scan(scan_body, carry_init, indices)

        return J_local, Jd_local

    @eqx.filter_jit
    def _jacobian_and_derivative_bodyframe_full(
        self, q: Array, qd: Array, s: Array
    ) -> Tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at a point s along the robot in the body frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_local (Array): Jacobian of the forward kinematics at point s in the body frame, shape (6, num_active_strains)
            Jd_local (Array): Time-derivative of the Jacobian at point s in the body frame, shape (6, num_active_strains)
        """
        J_local_, Jd_local_ = self._J_Jd_local(q, qd, s)

        J_local = self._final_size_jacobian(J_local_)
        Jd_local = self._final_size_jacobian(Jd_local_)

        return J_local, Jd_local

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
            Jd_local (Array): Time-derivative of the Jacobian at point s in the body frame, shape (3, num_active_strains)
        """
        J_local_, Jd_local_ = self._J_Jd_local(q, qd, s)

        J_local = self._final_size_jacobian(J_local_) @ self.B_xi
        Jd_local = self._final_size_jacobian(Jd_local_) @ self.B_xi

        return J_local, Jd_local

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
            Jd_global (Array): Time-derivative of the Jacobian at point s in the inertial frame, shape (3, num_active_strains)
        """
        # Compute the strains and strain rates
        xid = (self.B_xi @ qd).reshape(self.num_segments, 3)

        # Compute the bodyframe Jacobian and its derivative
        J_local_, Jd_local_ = self._J_Jd_local(q, qd, s)  # shape (num_segments, 3, 3)

        # Compute the forward kinematics to get the pose at point s
        chi = self.forward_kinematics(q, s)
        # extract the orientation
        theta = chi[0]
        # SE(2) transformation at point s
        g = lie.exp_SE2(jnp.stack([theta, 0.0, 0.0]))
        # Rotational velocity twist at point s
        eta_rot = jnp.einsum("nj,nj->", J_local_[:, 0, :], xid)

        # Adjoint representation of the SE(2) transformation
        Ad_g = lie.Adjoint_g_SE2(g)
        # Derivative of the Adjoint of g
        Ad_g_dot = Ad_g @ lie.adjoint_se2(jnp.stack([eta_rot, 0.0, 0.0]))

        # Rotate the Jacobian to the inertial frame
        J_global_ = jnp.einsum(
            "ij, njk -> nik",
            Ad_g,
            J_local_,
        )

        # compute the inertial frame Jacobian derivative
        Jd_global_ = jnp.einsum(
            "ij, njk -> nik",
            Ad_g,
            Jd_local_,
        ) + jnp.einsum(
            "ij, njk -> nik",
            Ad_g_dot,
            J_local_,
        )

        # reduce to the active strains
        J_global = self._final_size_jacobian(J_global_) @ self.B_xi
        Jd_global = self._final_size_jacobian(Jd_global_) @ self.B_xi

        return J_global, Jd_global

    @eqx.filter_jit
    def jacobian(self, q: Array, s: Array) -> Array:
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
            Jd_global (Array): Time-derivative of the Jacobian at point s in the inertial frame, shape (3, num_active_strains)
        """
        J_global, Jd_global = self.jacobian_and_derivative_inertialframe(q, qd, s)

        return J_global, Jd_global

    # ==========================================
    # Useful functions for the system

    @eqx.filter_jit
    def _local_cross_sectional_area(self, i: Array) -> Array:
        """
        Compute the local cross-sectional area for the i-th segment.

        Args:
            i (Array): index of the segment

        Returns:
            A_i (Array): local cross-sectional area of the i-th segment
        """
        # Cross-sectional area
        A_i = jnp.pi * self.r[i] ** 2
        return A_i

    @eqx.filter_jit
    def _local_second_moment_of_area(self, i: Array) -> Array:
        """
        Compute the local second moment of area for the i-th segment.

        Args:
            i (Array): index of the segment

        Returns:
            I_i (Array): local second moment of area of the i-th segment
        """
        # Second moment of area
        I_i = jnp.pi * self.r[i] ** 4 / 4
        return I_i

    @eqx.filter_jit
    def _local_mass_matrix(self, i: Array) -> Array:
        """
        Compute the local mass matrix for the i-th segment.

        Args:
            i (Array): index of the segment
        Returns:
            M_i (Array): local mass matrix of shape (3, 3) for the i-th segment
        """
        rho_i = self.rho[i]
        A_i = self._local_cross_sectional_area(i)  # Cross-sectional area
        I_i = self._local_second_moment_of_area(i)  # Second moment of area

        M_i = rho_i * jnp.diag(jnp.array([I_i, A_i, A_i]))
        return M_i

    # ===========================================
    # Dynamical matrices computation

    @eqx.filter_jit
    def _inertia_full_matrix(self, q: Array) -> Array:
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
            # B_blocks_i = jnp.stack(
            #     [B_j(j) for j in range(self.num_gauss_points)], axis=0
            # )

            return B_blocks_i

        B_blocks_tot = vmap(B_i)(jnp.arange(self.num_segments))

        # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
        # B_blocks_tot = jnp.stack([B_i(i) for i in range(self.num_segments)], axis=0)

        B_full = jnp.sum(
            B_blocks_tot, axis=(0, 1)
        )  # Sum over segments and Gauss points

        return B_full

    @eqx.filter_jit
    def inertia_matrix(self, q: Array) -> Array:
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

    @eqx.filter_jit
    def _coriolis_full_matrix(self, q: Array, qd: Array) -> Array:
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
                J_j, Jd_j = self._jacobian_and_derivative_bodyframe_full(q, qd, Xs_j)
                return Ws_j * (
                    J_j.T
                    @ (M_i @ Jd_j + lie.coadjoint_se2(J_j @ self.B_xi @ qd) @ M_i @ J_j)
                )

            C_blocks_i = vmap(C_j)(jnp.arange(self.num_gauss_points))

            return C_blocks_i

        C_blocks_tot = vmap(C_i)(jnp.arange(self.num_segments))

        C_full = jnp.sum(C_blocks_tot, axis=(0, 1))

        return C_full

    @eqx.filter_jit
    def coriolis_matrix(self, q: Array, qd: Array) -> Array:
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

    @eqx.filter_jit
    def _gravitational_full_force(self, q: Array) -> Array:
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

                return -Ws_j * J_j.T @ M_i @ Ad_g_inv_j @ self.g

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
    def gravitational_force(self, q: Array) -> Array:
        """
        Compute the gravitational force acting on the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            G (Array): Gravitational force of shape (num_active_strains,).
        """
        G_full = self._gravitational_full_force(q)

        G = self.B_xi.T @ G_full

        return G

    @eqx.filter_jit
    def _local_stiffness_matrix(self, i: Array) -> Array:
        """
        Compute the local stiffness matrix of a planar system for a rod aligned along the x-axis.

        Args:
            i (Array): index of the segment

        Returns:
            S_i (Array): Local stiffness matrix of shape (3, 3) for the i-th segment.
        """
        I_i = self._local_second_moment_of_area(i)  # Second moment of area
        A_i = self._local_cross_sectional_area(i)  # Cross-sectional area

        S_i = self.L[i] * jnp.diag(
            jnp.stack(
                [
                    I_i * self.E[i],  # bending Z
                    A_i * self.E[i],  # axial X
                    A_i * self.G[i],  # shear Y
                ],
                axis=0,
            )
        )

        return S_i

    @eqx.filter_jit
    def _stiffness(self, formulate_in_strain_space: bool = False) -> Array:
        # stiffness matrix of shape (num_segments, 3, 3)
        S_sms = vmap(self._local_stiffness_matrix)(jnp.arange(self.num_segments))

        # we define the elastic matrix of shape (num_strains, num_strains) as K(xi) = K @ xi where K is equal to
        S = blk_diag(S_sms)

        if not formulate_in_strain_space:
            S = self.B_xi.T @ S @ self.B_xi

        return S

    @eqx.filter_jit
    def _stiffness_full_matrix(self) -> Array:
        """
        Compute the full stiffness matrix of the robot.

        Returns:
            K_full (Array): Full stiffness matrix of shape (num_strains, num_strains).
        """
        K_full = self._stiffness(formulate_in_strain_space=True)

        return K_full

    @eqx.filter_jit
    def stiffness_matrix(self) -> Array:
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

    @eqx.filter_jit
    def _damping_full_matrix(self) -> Array:
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
    def damping_matrix(self) -> Array:
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
    def actuation_force(self, q: Array, u: Array) -> Array:
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

        # compute the actuation force
        tau_u = A @ u

        return tau_u

    @eqx.filter_jit
    def kinetic_energy(self, q: Array, qd: Array) -> Array:
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
    def elastic_energy(self, q: Array) -> Array:
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
    def gravitational_energy(self, q: Array) -> Array:
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
                return -Ws_j * rho_i * A_i * jnp.dot(p_j, self.g)

            U_G_blocks_segment_i = vmap(U_G_j)(jnp.arange(self.num_gauss_points))

            # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
            # U_G_blocks_segment_i = jnp.stack(
            #     [U_G_j(j) for j in range(self.num_gauss_points)], axis=0
            # )

            return U_G_blocks_segment_i

        U_G_blocks_tot = vmap(U_G_i)(jnp.arange(self.num_segments))

        # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
        # U_G_blocks_tot = jnp.stack(
        #     [U_G_i(i) for i in range(self.num_segments)], axis=0
        # )

        U_G = jnp.sum(U_G_blocks_tot, axis=(0, 1))  # Sum over segments and Gauss points

        return U_G

    @eqx.filter_jit
    def potential_energy(self, q: Array) -> Array:
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
    def total_energy(self, q: Array, qd: Array) -> Array:
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
            Jd (Array): Time-derivative of the Jacobian at point s in the body frame, shape (num_operational_space_dims, num_active_strains).
            JB_pinv (Array): Dynamically-consistent pseudo-inverse of the Jacobian, shape (num_active_strains, num_operational_space_dims).
        """
        # make operational_space_selector a boolean array
        operational_space_selector = onp.array(operational_space_selector, dtype=bool)

        # Jacobian and its time-derivative
        J, Jd = self.jacobian_and_derivative_inertialframe(q, qd, s)

        J = J[operational_space_selector, :]
        Jd = Jd[operational_space_selector, :]

        # inverse of the inertia matrix in the configuration space
        B = self.inertia_matrix(q)
        B_inv = jnp.linalg.inv(B)
        C = self.coriolis_matrix(q, qd)

        Lambda = jnp.linalg.inv(
            J @ B_inv @ J.T
        )  # inertia matrix in the operational space
        mu = Lambda @ (
            J @ B_inv @ C - Jd
        )  # coriolis and centrifugal matrix in the operational space

        JB_pinv = (
            B_inv @ J.T @ Lambda
        )  # dynamically-consistent pseudo-inverse of the Jacobian

        return Lambda, mu, J, Jd, JB_pinv

    @eqx.filter_jit
    def forward_dynamics(
        self, t: Array, y: Array, actuation_args: Optional[Tuple] = None
    ) -> Array:
        """
        Forward dynamics function.

        Args:
            t (Array): Current time.
            y (Array): State vector containing configuration and velocity.
                Shape is (2 * num_strains,).
            actuation_args (Tuple, optional): Additional arguments for the actuation mapping function.
                Default is None.
        Returns:
            yd: Time derivative of the state vector.
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
            u = jnp.zeros((self.num_actuators,))
        if tau_ext is None:
            tau_ext = jnp.zeros((q.shape[-1],))

        # evaluate the dynamical matrices
        B = self.inertia_matrix(q)
        C = self.coriolis_matrix(q, qd)
        G = self.gravitational_force(q)
        D = self.damping_matrix()
        tau_el = self.elastic_force(q)
        tau_u = self.actuation_force(q, u)

        B_inv = jnp.linalg.inv(B)  # Inverse of the inertia matrix
        qdd = B_inv @ (
            tau_u + tau_ext - C @ qd - G - tau_el - D @ qd
        )  # Compute the acceleration

        yd = jnp.concatenate([qd, qdd])

        return yd
