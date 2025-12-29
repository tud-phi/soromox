__all__ = ["PCS"]


from typing import Any

import equinox as eqx
from jax import Array, lax, vmap
from jax import numpy as jnp

import soromox.utils.lie_algebra as lie
from soromox.systems.soft_robot import CrossSectionGeometry, SoftRobot
from soromox.utils.array_math import blk_diag
from soromox.utils.basic import (
    compute_strain_basis,
)
from soromox.utils.integration import (
    gauss_quadrature,
    scale_gaussian_quadrature,
)


class PCS(SoftRobot):
    """
    Piecewise Constant Strain (PCS) model for 3D soft continuum robots.

    This class implements the geometric and dynamic modeling of a 3D soft robot
    using the Cosserat rod theory and piecewise constant strain assumption.
    It supports computation of forward kinematics, Jacobians, dynamical matrices.

    Attributes:
        num_segments: Number of segments (constant strain sections) along the robot.
        num_actuators: Number of actuators (control inputs) for the robot.
        g0: Initial pose of the robot base as an SE(3) transformation matrix.
        g: Gravitational acceleration vector (embedded in a 6D vector).
            [0, 0, 0, g_x, g_y, g_z]
        L, r, E, G, rho, D: Physical properties of each segment (length, radius, elastic/shear modulus, etc.).
        num_active_strains: Number of active strain components (based on strain_selector).
        num_strains: Total number of strain components (6 * num_segments).
        B_xi: Basis matrix for projecting active strains (6 * num_segments, num_active_strains).
        xi_ref: Reference strain (reference configuration) of the robot.
        num_gauss_points: Number of points used for numerical integration.
            Corresponds to the order of Gauss-Legendre quadrature + 2 (for the endpoints).
        Xs, Ws: Gauss-Legendre quadrature nodes and weights for numerical integration.

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

    References:
        Renda, F., Boyer, F., Dias, J., & Seneviratne, L. (2018). Discrete Cosserat
        Approach for Multisection Soft Manipulator Dynamics. IEEE Transactions on
        Robotics, 34(6), 1518-1533.

    """

    # Robot parameters
    g0: Array  # Initial position and orientation of the rod
    g: Array  # Gravitational acceleration vector

    L: Array  # Length of the segments
    L_cum: Array  # Cumulative length of the segments
    r: Array  # Radius of the segments
    rho: Array
    E: Array  # Young's modulus of the segments
    G: Array  # Shear modulus of the segments
    D: Array  # Damping coefficient of the segments

    num_segments: int = eqx.field(static=True)
    num_gauss_points: int = eqx.field(static=True)  #
    num_strains: int = eqx.field(static=True)  # Number of strains (6 * num_segments)

    xi_ref: Array  # Reference configuration strain
    B_xi: Array  # Strain basis matrix
    num_active_strains: Array  # Number of selected strains

    Xs: Array  # Gauss nodes
    Ws: Array  # Gauss weights

    def __init__(
        self,
        num_segments: int,
        params: dict[str, Array],
        order_gauss: int = 5,
        strain_selector: Array | None = None,
        xi_ref: Array | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the PCS class.

        Args:
            num_segments (int):
                Number of segments in the robot.
            params (Dict[str, Array]):
                Dictionary containing the robot parameters:
                - "p0": (optional) List/Array of shape (6,)
                    Initial orientation angle and position in the inertial frame [rad, m]
                    [ψ, θ, φ, x0, y0, z0]
                        [ψ, θ, φ] are the Euler angles in the ZXZ convention:
                            ψ (psi) : Rotation around Z axis (fixed axis)
                            θ (theta) : Rotation around X' axis (movable axis after first rotation)
                            φ (phi) : Rotation about the Z' axis (movable axis after the first two rotations)
                        [x0, y0, z0] : Position of the robot in the inertial frame
                    Defaults to [pi/2, pi/2, 0.0, 0.0, 0.0, 0.0] (i.e. aligned with the z-axis and at the origin).
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
                Boolean array of shape (6 * num_segments,) specifying which strain components are active.
                Defaults to all strains active (i.e. all True).
            xi_ref (Optional[Array], optional):
                Reference strain of shape (6 * num_segments,).
                Defaults to 0.0 for bending and shear strains, and 1.0 for axial strain (along local x-axis).
            **kwargs: Additional keyword arguments for SoftRobot.__init__.
        """
        super().__init__(**kwargs)

        # Number of segments
        if not isinstance(num_segments, int):
            raise TypeError(
                f"num_segments must be an integer, got {type(num_segments).__name__}"
            )
        if num_segments < 1:
            raise ValueError(f"num_segments must be at least 1, got {num_segments}")
        self.num_segments = num_segments

        num_strains = 6 * num_segments
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
            strain_selector = jnp.ones(num_strains, dtype=bool)
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
            if strain_selector.size != num_strains:
                raise ValueError(
                    f"strain_selector must have {num_strains} elements, got {strain_selector.size}"
                )
            strain_selector = strain_selector.reshape(num_strains)
        self.B_xi = compute_strain_basis(strain_selector)

        self.num_active_strains = jnp.sum(strain_selector)
        self.num_dofs = int(self.num_active_strains.item())

        # Reference configuration strain
        if xi_ref is None:
            xi_ref = jnp.tile(
                jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=jnp.float64),
                (num_segments, 1),
            ).reshape(num_strains)
        else:
            if not isinstance(xi_ref, (list, jnp.ndarray)):
                raise TypeError(
                    f"xi_ref must be a list or an array, got {type(xi_ref).__name__}"
                )
            xi_ref = jnp.asarray(xi_ref)
            if xi_ref.size != num_strains:
                raise ValueError(
                    f"xi_ref must have {num_strains} elements, got {xi_ref.size}"
                )
            xi_ref = xi_ref.reshape(num_strains)
        self.xi_ref = xi_ref

        # Number of actuators
        self.num_actuators = int(self.num_active_strains.item())

    @property
    def is_planar(self) -> bool:
        """PCS is a spatial (3D) model."""
        return False

    @property
    def length(self) -> Array:
        """Total backbone length."""
        return jnp.sum(self.L)

    @property
    def segment_length(self) -> Array:
        """Per-segment backbone lengths."""
        return jnp.asarray(self.L)

    def cross_section_geometry(self, q: Array, s: Array) -> tuple[Array, Array]:
        """Circular cross-section with segment radius."""
        segment_idx, _ = self.classify_segment(s)
        radius = jnp.asarray(self.r)[segment_idx]
        tag = jnp.asarray(CrossSectionGeometry.CIRCULAR, dtype=jnp.int32)
        return tag, jnp.array([radius])

    def _set_params(self, params: dict[str, Array]) -> None:
        """
        Set the robot parameters from a dictionary.
        Args:
            params (Dict[str, Array]):
                Dictionary containing the robot parameters:
                - "p0": (optional) List/Array of shape (6,)
                    Initial orientation angle and position in the inertial frame [rad, m]
                    [ψ, θ, φ, x0, y0, z0]
                        [ψ, θ, φ] are the Euler angles in the ZXZ convention:
                            ψ (psi) : Rotation around Z axis (fixed axis)
                            θ (theta) : Rotation around X' axis (movable axis after first rotation)
                            φ (phi) : Rotation about the Z' axis (movable axis after the first two rotations)
                        [x0, y0, z0] : Position of the robot in the inertial frame
                    Defaults to [pi/2, pi/2, 0.0, 0.0, 0.0, 0.0] (i.e. aligned with the z-axis and at the origin).
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
        # Initial position and orientation angle
        p0 = params.get("p0", jnp.array([jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, 0.0]))
        if not (isinstance(p0, (list, jnp.ndarray))):
            raise TypeError(f"p0 must be a list or an array, got {type(p0).__name__}")
        p0 = jnp.asarray(p0, dtype=jnp.float64)
        if p0.size != 6:
            raise ValueError(f"p0 must have shape (6,), got {p0.size}")
        self.g0 = lie.exp_SE3(p0)

        # Gravitational acceleration vector
        try:
            g = params["g"]
        except KeyError:
            raise KeyError("Parameter 'g' is required in params dictionary.")
        if not (isinstance(g, (list, jnp.ndarray))):
            raise TypeError(f"g must be a list or an array, got {type(g).__name__}")
        g = jnp.asarray(g, dtype=jnp.float64)
        if g.size != 3:
            raise ValueError(f"g must be a vector of shape (3,), got {g.size}")
        self.g = jnp.concatenate(
            [jnp.zeros(3), g]
        )  # Add zeros for the orientation angles

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

    def update_params(self, params: dict[str, Array]) -> "PCS":
        """
        Update the parameters of the PCS model.

        Args:
            params (Dict[str, Array]):
                Dictionary that contains the robot parameters to update:
                - "p0": (optional) List/Array of shape (6,)
                    Initial orientation angle and position in the inertial frame [rad, m]
                    [ψ, θ, φ, x0, y0, z0]
                        [ψ, θ, φ] are the Euler angles in the ZXZ convention:
                            ψ (psi) : Rotation around Z axis (fixed axis)
                            θ (theta) : Rotation around X' axis (movable axis after first rotation)
                            φ (phi) : Rotation about the Z' axis (movable axis after the first two rotations)
                        [x0, y0, z0] : Position of the robot in the inertial frame
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
        # Apply updates sequentially
        updated_self = self

        if "p0" in params:
            p0 = params["p0"]
            if not (isinstance(p0, (list, jnp.ndarray))):
                raise TypeError(
                    f"p0 must be a list or an array, got {type(p0).__name__}"
                )
            p0 = jnp.asarray(p0, dtype=jnp.float64)
            updated_self = eqx.tree_at(lambda m: m.g0, updated_self, lie.exp_SE3(p0))

        if "g" in params:
            g = params["g"]
            if not (isinstance(g, (list, jnp.ndarray))):
                raise TypeError(f"g must be a list or an array, got {type(g).__name__}")
            g = jnp.asarray(g, dtype=jnp.float64)
            if g.size != 3:
                raise ValueError(f"g must be a vector of shape (3,), got {g.size}")
            updated_self = eqx.tree_at(
                lambda m: m.g, updated_self, jnp.concatenate([jnp.zeros(3), g])
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
    def classify_segment(self, s: Array) -> tuple[Array, Array]:
        """
        Classify the point along the robot to the corresponding segment.

        Args:
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            segment_idx (Array): index of the segment where the point is located
            s_segment (Array): point coordinate along the segment in the interval [0, l_segment]
        """

        # Classify the point along the robot to the corresponding segment
        segment_idx = jnp.clip(
            jnp.sum(s > self.L_cum) - 1, 0, self.num_segments - 1
        ).astype(jnp.int32)

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
    def forward_kinematics(self, q: Array, s: Array) -> Array:
        """
        Compute the forward kinematics of the robot at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            g_s (Array): forward kinematics of the robot at point s, shape (4, 4) :
                [[  R,      p],
                [0, 0, 0,   1]] where R is the rotation matrix and p is the position vector.
        """
        xi = self.strain(q).reshape(self.num_segments, 6)

        # Compute the point coordinate along the segment in the interval [0, L_i]
        segment_idx, s_local = self.classify_segment(s)

        def body_segment_i(g_base_i: Array, i: Array) -> tuple[Array, Array]:
            """
            Compute the forward kinematics of the robot for the i-th segment. Either until the tip of the segment or until the point s_local.
            Args:
                g_base_i (Array): transformation matrix of the base of the segment, shape (4, 4).
                i (Array): index of the segment.
            Returns:
                g_i (Array): transformation matrix of the point s_local if i == segment_idx, otherwise the tip of the segment, shape (4, 4).
                g_i (Array): transformation matrix of the point s_local if i == segment_idx, otherwise the tip of the segment, shape (4, 4).
            """
            length_i = jnp.where(i == segment_idx, s_local, self.L[i])
            xi_i = xi[i]

            # Magnus expansion
            Magnus_i = length_i * xi_i

            # Compute the transformation matrix from the base of the segment until length_i using the exponential map
            g_step = lie.exp_gn_SE3(Magnus_i, eps=self.global_eps)

            g_i = g_base_i @ g_step

            return g_i, g_i

        indices_link = jnp.arange(self.num_segments)

        g_ini = self.g0  # Initial position and orientation of the robot base

        _, g_ls = lax.scan(f=body_segment_i, init=g_ini, xs=indices_link)

        # # For debugging purposes, you can uncomment the following line to see the list of transformations
        # carry = g_ini
        # g_list = []
        # for i_segment in indices_link:
        #     g_tip_i, g_list_i = body_segment_i(carry, i_segment)
        #     carry = g_tip_i
        #     g_list.append(g_list_i)
        # g_list = jnp.array(g_list)

        g_s = g_ls[segment_idx]

        return g_s

    @eqx.filter_jit
    def forward_kinematics_tips(self, q: Array) -> Array:
        """
        Compute the forward kinematics of the robot at all segment tips.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            g_tips (Array): forward kinematics of the robot at all segment tips, shape (num_segments, 4, 4) :
                g_tip_i = [[  R,      p],
                           [0, 0, 0,   1]] where R is the rotation matrix and p is the position vector.
        """
        xi = self.strain(q).reshape(self.num_segments, 6)

        def body_segment_i(g_base_i: Array, i: Array) -> tuple[Array, Array]:
            # Magnus expansion
            Magnus_Li = self.L[i] * xi[i]

            # Compute the transformation matrix from the base until the tip of the segment using the exponential map
            g_Li_rel = lie.exp_gn_SE3(Magnus_Li, eps=self.global_eps)

            g_Li = g_base_i @ g_Li_rel

            return g_Li, g_Li

        indices_link = jnp.arange(self.num_segments)

        # Initial position and orientation of the robot base
        g_ini = self.g0

        _, g_tips = lax.scan(f=body_segment_i, init=g_ini, xs=indices_link)

        return g_tips

    @eqx.filter_jit
    def forward_kinematics_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute the forward kinematics of the robot at a batch of points s_ps along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s_ps (Array): point coordinates along the robot in the interval [0, L] of shape (N,).

        Returns:
            g_ps: forward kinematics of the robot at all points s_ps, shape (N, 4, 4) :
                g_si = [[  R,      p],
                        [0, 0, 0,   1]] where R is the rotation matrix and p is the position vector.
        """
        xi = self.strain(q).reshape(self.num_segments, 6)

        # compute the forward kinematics at the tips
        g_tips = self.forward_kinematics_tips(q)

        # Compute the point coordinates along the segment in the interval [0, L_i]
        segment_indices, s_local_ps = vmap(self.classify_segment)(s_ps)

        # collect the base transform for each segment (g0 for the first, previous tip otherwise)
        g_bases = jnp.concatenate([self.g0[None, :, :], g_tips[:-1]], axis=0)

        # compute the magnus expansion for each point
        magnus = s_local_ps.reshape(-1, 1) * xi[segment_indices]
        # compute the relative transform from the base of the segment to the point
        g_rel = vmap(lambda magnus_i: lie.exp_gn_SE3(magnus_i, eps=self.global_eps))(
            magnus
        )
        # get the base transform for each point
        g_base_points = g_bases[segment_indices]

        # combine to get the absolute transforms
        g_ps = vmap(lambda g_base_i, g_rel_i: g_base_i @ g_rel_i)(g_base_points, g_rel)

        return g_ps

    @eqx.filter_jit
    def _compute_relative_segment_transforms(self, g_tips: Array) -> Array:
        """Return the SE(3) transforms from each segment base to its distal tip.

        The first entry is measured from the global base pose ``g0``; subsequent entries
        use the distal pose of the previous segment, mirroring the serial chain geometry.

        Args:
            g_tips (Array): absolute tip transforms of shape (num_segments, 4, 4).

        Returns:
            g_rels (Array): relative transforms of shape (num_segments, 4, 4).
        """
        g_tips = jnp.asarray(g_tips).reshape(self.num_segments, 4, 4)

        # prepend the base transform so every segment has its proximal pose available
        g_prev = jnp.concatenate([self.g0[None, :, :], g_tips[:-1]], axis=0)

        def rel_transform(g_prev_i: Array, g_curr_i: Array) -> Array:
            """
            Compute ``g_prev_i^{-1} @ g_curr_i`` without explicitly inverting ``g_prev_i``.
            This is more numerically stable and efficient than computing the inverse directly.
            Args:
                g_prev_i (Array): previous segment tip transform of shape (4, 4).
                g_curr_i (Array): current segment tip transform of shape (4, 4).
            Returns:
                g_rel (Array): relative transform of shape (4, 4).
            """
            R_prev = g_prev_i[:3, :3]
            p_prev = g_prev_i[:3, 3]

            R_curr = g_curr_i[:3, :3]
            p_curr = g_curr_i[:3, 3]

            R_rel = R_prev.T @ R_curr
            p_rel = R_prev.T @ (p_curr - p_prev)

            upper = jnp.concatenate([R_rel, p_rel[:, None]], axis=1)
            lower = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=g_curr_i.dtype)[None, :]
            g_rel = jnp.concatenate([upper, lower], axis=0)
            return g_rel

        g_rels = vmap(rel_transform)(g_prev, g_tips)
        return g_rels

    @eqx.filter_jit
    def inverse_kinematics(self, g_tips: Array) -> Array:
        """
        Recover generalized coordinates from the absolute tip poses of each segment.

        The routine converts each tip pose into a body-relative transform, extracts the
        corresponding twist via the SE(3) logarithmic map, and normalises by segment length
        to obtain the constant strain representation used by the PCS model.

        Args:
            g_tips (Array): homogeneous tip transforms of shape (num_segments, 4, 4).

        Returns:
            q (Array): generalized coordinates ``q`` of shape (num_active_strains,).
        """
        # Relative pose of each segment distal tip with respect to its proximal base
        g_rel = self._compute_relative_segment_transforms(g_tips)

        def _segment_inverse(g_rel_i: Array, length_i: Array) -> Array:
            """
            Map the relative tip pose of a segment to its constant strain twist.
            Args:
                g_rel_i (Array): relative transform of shape (4, 4).
                length_i (Array): length of the segment.
            Returns:
                xi (Array): strain twist of shape (6,).
            """
            xi_mag = lie.log_SE3(g_rel_i, eps=self.global_eps)
            xi = jnp.where(
                jnp.abs(length_i) < self.global_eps,
                jnp.zeros_like(xi_mag),
                xi_mag / length_i,
            )
            return xi

        # Convert every relative transform to segment twist parameters
        xi_segments = vmap(_segment_inverse)(g_rel, self.L)
        xi = xi_segments.reshape(-1)

        # Project the strain difference onto the active strain basis to obtain ``q``
        q = jnp.linalg.pinv(self.B_xi) @ (xi - self.xi_ref)

        return q

    @eqx.filter_jit
    def _final_size_jacobian(self, J_full: Array) -> Array:
        """
        Convert the Jacobian or its derivative from the full computation form to the selected strains form.

        Args:
            J_full (Array): Full Jacobian of shape (num_segments, 6, 6)

        Returns:
            J_selected (Array): Jacobian for the selected strains of shape (6, num_active_strains)
        """
        J_final = J_full.transpose(1, 0, 2).reshape(6, self.num_strains)

        return J_final

    @eqx.filter_jit
    def _J_local(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_local (Array): Jacobian of the forward kinematics at point s, shape (6, num_strains)
            where each row corresponds to a segment.
        """
        xi = self.strain(q).reshape(self.num_segments, 6)

        segment_idx, s_local = self.classify_segment(s)

        zeros = jnp.zeros((self.num_segments, 6, 6), dtype=xi.dtype)
        zero_slice = jnp.zeros((6, 6), dtype=xi.dtype)

        def integrate_segment(
            J_prev: Array,
            i: Array,
            xi_i: Array,
            arc_len: Array,
        ) -> Array:
            Ad_inv = lie.Adjoint_gi_se3_inv(xi_i, arc_len, eps=self.global_eps)
            T = lie.Tangent_gi_se3(xi_i, arc_len, eps=self.tangent_eps)

            J_rot = jnp.einsum("ij, njk->nik", Ad_inv, J_prev)
            J_next = J_rot.at[i].set(Ad_inv @ T)

            return J_next

        def scan_body(
            carry: tuple[Array, Array, Array],
            i: Array,
        ) -> tuple[tuple[Array, Array, Array], Array]:
            J_prev, J_local, done = carry

            def compute_branch(_: None) -> tuple[tuple[Array, Array, Array], Array]:
                xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
                L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)
                arc_len = jnp.where(i == segment_idx, s_local, L_i)

                J_next = integrate_segment(J_prev, i, xi_i, arc_len)

                is_target = i == segment_idx
                J_target_next = jnp.where(is_target, J_next, J_local)
                done_next = jnp.logical_or(done, is_target)

                return (J_next, J_target_next, done_next), zero_slice

            def skip_branch(_: None) -> tuple[tuple[Array, Array, Array], Array]:
                return (J_prev, J_local, done), zero_slice

            return lax.cond(done, skip_branch, compute_branch, operand=None)

        carry_init = (
            zeros,
            zeros,
            jnp.array(False, dtype=jnp.bool_),
        )

        indices = jnp.arange(self.num_segments, dtype=segment_idx.dtype)
        (_, J_target, _), _ = lax.scan(scan_body, carry_init, indices)

        J_local = self._final_size_jacobian(J_target)

        return J_local

    @eqx.filter_jit
    def _J_local_tips(self, q: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at the tips of all segments.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            J_local_tips (Array): Jacobian of the forward kinematics the tips of all segments, shape (num_segments, 6, num_strains)
        """
        xi = self.strain(q).reshape(self.num_segments, 6)

        zeros = jnp.zeros((self.num_segments, 6, 6), dtype=xi.dtype)

        def scan_body(
            J_prev: Array,
            i: Array,
        ) -> tuple[Array, Array]:
            xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
            L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)

            Ad_inv = lie.Adjoint_gi_se3_inv(xi_i, L_i, eps=self.global_eps)
            T = lie.Tangent_gi_se3(xi_i, L_i, eps=self.tangent_eps)

            J_rot = jnp.einsum("ij, njk->nik", Ad_inv, J_prev)
            J_next = J_rot.at[i].set(Ad_inv @ T)

            return J_next, J_next

        indices = jnp.arange(self.num_segments, dtype=jnp.int32)
        _, J_target_tips = lax.scan(scan_body, zeros, indices)

        J_local_tips = vmap(self._final_size_jacobian)(J_target_tips)

        return J_local_tips

    @eqx.filter_jit
    def _J_local_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute the Jacobian for the forward kinematics at a batch of points s_ps along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s_ps (Array): point coordinates along the robot in the interval [0, L] of shape (N,).

        Returns:
            J_local_ps (Array): Jacobians evaluated at all points, shape (N, 6, num_strains)
        """
        # num points
        N = s_ps.shape[0]

        # compute the strain for each segment
        xi = self.strain(q).reshape(self.num_segments, 6)

        # classify all points along the robot to the corresponding segment and compute local coordinates
        segment_indices, s_local_ps = vmap(self.classify_segment)(s_ps)

        # compute the Jacobian at the tips
        J_tips = self._J_local_tips(q)  # shape (num_segments, 6, num_strains)

        # select the base Jacobian for each point (g0 for the first, previous tip otherwise)
        J_bases = jnp.concatenate([jnp.zeros_like(J_tips[:1]), J_tips[:-1]], axis=0)

        J_base_ps = J_bases[segment_indices]
        # select the other variables for each point
        xi_ps = xi[segment_indices]
        idx_ps = segment_indices

        # reshape for easier indexing
        J_base_ps = J_base_ps.reshape(N, 6, self.num_segments, 6).transpose(
            0, 2, 1, 3
        )  # shape (N, 6, num_segments, 6)

        def integrate_segment(
            i: Array,
            xi_i: Array,
            s_local: Array,
            J_base: Array,
        ) -> Array:
            Ad_inv = lie.Adjoint_gi_se3_inv(xi_i, s_local, eps=self.global_eps)
            T = lie.Tangent_gi_se3(xi_i, s_local, eps=self.tangent_eps)

            J_rot = jnp.einsum("ij, njk->nik", Ad_inv, J_base)
            J_next = J_rot.at[i].set(Ad_inv @ T)

            return J_next

        # vmap the segment integration over all points
        J_local_ps = vmap(integrate_segment)(idx_ps, xi_ps, s_local_ps, J_base_ps)

        # reshape back to (N, 6, num_strains)
        J_local_ps = vmap(self._final_size_jacobian)(J_local_ps)

        return J_local_ps

    @eqx.filter_jit
    def jacobian_bodyframe(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a point s along the robot in the body frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_local (Array): Jacobian of the forward kinematics at point s in the body frame, shape (6, num_active_strains)
        """
        J_local = self._J_local(q, s) @ self.B_xi

        return J_local

    @eqx.filter_jit
    def jacobian_inertialframe(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a point s along the robot in the inertial frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_global (Array): Jacobian of the forward kinematics at point s in the inertial frame, shape (6, num_active_strains)
        """
        # compute the Jacobian in the body frame
        J_local = self._J_local(q, s)

        g_s = self.forward_kinematics(q, s)
        # construct g with zero translation for the Adjoint transformation
        g_rot = jnp.block(
            [[g_s[:3, :3], jnp.zeros((3, 1))], [jnp.zeros((1, 3)), jnp.ones((1, 1))]]
        )
        Ad_g = lie.Adjoint_g_SE3(g_rot)

        J_global_ = jnp.einsum("ij, jk->ik", Ad_g, J_local)
        J_global = J_global_ @ self.B_xi

        return J_global

    @eqx.filter_jit
    def _J_Jd_local(self, q: Array, qd: Array, s: Array) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_local (Array): Jacobian of the forward kinematics at point s, shape (6, num_strains)
            Jd_local (Array): Time-derivative of the Jacobian at point s, shape (6, num_strains)
        """
        xi = self.strain(q).reshape(self.num_segments, 6)
        xid = (self.B_xi @ qd).reshape(self.num_segments, 6)

        segment_idx, s_local = self.classify_segment(s)

        zeros = jnp.zeros((self.num_segments, 6, 6), dtype=xi.dtype)
        zero_slice = jnp.zeros((6, 6), dtype=xi.dtype)

        def integrate_segment(
            J_prev: Array,
            Jd_prev: Array,
            i: Array,
            xi_i: Array,
            xid_i: Array,
            arc_len: Array,
        ) -> tuple[Array, Array]:
            Ad_inv = lie.Adjoint_gi_se3_inv(xi_i, arc_len, eps=self.global_eps)
            T = lie.Tangent_gi_se3(xi_i, arc_len, eps=self.tangent_eps)
            Td = lie.Tangent_derivative_gi_se3(
                xi_i, xid_i, arc_len, eps=self.tangent_eps
            )

            J_rot = jnp.einsum("ij, njk->nik", Ad_inv, J_prev)
            J_next = J_rot.at[i].set(Ad_inv @ T)

            J_next_slice = lax.dynamic_index_in_dim(J_next, i, axis=0, keepdims=False)
            eta = jnp.matmul(J_next_slice, xid_i)

            Ad_inv_dot = -lie.adjoint_se3(eta) @ Ad_inv

            Jd_rot = jnp.einsum("ij, njk->nik", Ad_inv, Jd_prev) + jnp.einsum(
                "ij, njk->nik", Ad_inv_dot, J_prev
            )
            Jd_next = Jd_rot.at[i].set(Ad_inv_dot @ T + Ad_inv @ Td)

            return J_next, Jd_next

        def scan_body(
            carry: tuple[Array, Array, Array, Array, Array],
            i: Array,
        ) -> tuple[tuple[Array, Array, Array, Array, Array], Array]:
            J_prev, Jd_prev, J_target, Jd_target, done = carry

            def compute_branch(
                _: None,
            ) -> tuple[tuple[Array, Array, Array, Array, Array], Array]:
                xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
                xid_i = lax.dynamic_index_in_dim(xid, i, axis=0, keepdims=False)
                L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)
                arc_len = jnp.where(i == segment_idx, s_local, L_i)

                J_next, Jd_next = integrate_segment(
                    J_prev,
                    Jd_prev,
                    i,
                    xi_i,
                    xid_i,
                    arc_len,
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
                ), zero_slice

            def skip_branch(
                _: None,
            ) -> tuple[tuple[Array, Array, Array, Array, Array], Array]:
                return (J_prev, Jd_prev, J_target, Jd_target, done), zero_slice

            return lax.cond(done, skip_branch, compute_branch, operand=None)

        carry_init = (
            zeros,
            zeros,
            zeros,
            zeros,
            jnp.array(False, dtype=jnp.bool_),
        )

        indices = jnp.arange(self.num_segments, dtype=segment_idx.dtype)
        (_, _, J_target, Jd_target, _), _ = lax.scan(scan_body, carry_init, indices)

        J_local = self._final_size_jacobian(J_target)
        Jd_local = self._final_size_jacobian(Jd_target)

        return J_local, Jd_local

    @eqx.filter_jit
    def _J_Jd_local_tips(self, q: Array, qd: Array) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at all segment tips.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).

        Returns:
            J_local_tips (Array): Jacobian of the forward kinematics at point s, shape (num_segments, 6, num_strains)
            Jd_local_tips (Array): Time-derivative of the Jacobian at point s, shape (num_segments, 6, num_strains)
        """
        # compute the strain and strain rate for each segment
        xi = self.strain(q).reshape(self.num_segments, 6)
        xid = (self.B_xi @ qd).reshape(self.num_segments, 6)

        # precompute the Lie algebra expressions for all segment tips
        Ad_inv_tips = vmap(
            lambda xi_i, L_i: lie.Adjoint_gi_se3_inv(xi_i, L_i, eps=self.global_eps)
        )(xi, self.L)
        T_tips = vmap(
            lambda xi_i, L_i: lie.Tangent_gi_se3(xi_i, L_i, eps=self.tangent_eps)
        )(xi, self.L)
        Td_tips = vmap(
            lambda xi_i, xid_i, L_i: lie.Tangent_derivative_gi_se3(
                xi_i, xid_i, L_i, eps=self.tangent_eps
            )
        )(xi, xid, self.L)

        # initialize zeros
        zeros = jnp.zeros((self.num_segments, 6, 6), dtype=xi.dtype)

        def scan_body(
            carry: tuple[Array, Array],
            i: Array,
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            J_prev, Jd_prev = carry

            # extract the current segment variables
            xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
            xid_i = lax.dynamic_index_in_dim(xid, i, axis=0, keepdims=False)
            Ad_inv_i = lax.dynamic_index_in_dim(Ad_inv_tips, i, axis=0, keepdims=False)
            T_i = lax.dynamic_index_in_dim(T_tips, i, axis=0, keepdims=False)
            Td_i = lax.dynamic_index_in_dim(Td_tips, i, axis=0, keepdims=False)

            J_rot = jnp.einsum("ij, njk->nik", Ad_inv_i, J_prev)
            J_next = J_rot.at[i].set(Ad_inv_i @ T_i)

            J_next_slice = lax.dynamic_index_in_dim(J_next, i, axis=0, keepdims=False)
            eta = jnp.matmul(J_next_slice, xid_i)

            Ad_inv_dot = -lie.adjoint_se3(eta) @ Ad_inv_i

            Jd_rot = jnp.einsum("ij, njk->nik", Ad_inv_i, Jd_prev) + jnp.einsum(
                "ij, njk->nik", Ad_inv_dot, J_prev
            )
            Jd_next = Jd_rot.at[i].set(Ad_inv_dot @ T_i + Ad_inv_i @ Td_i)

            return (J_next, Jd_next), (J_next, Jd_next)

        carry_init = (zeros, zeros)

        indices = jnp.arange(self.num_segments, dtype=jnp.int32)
        (_, _), (J_local_tips, Jd_local_tips) = lax.scan(scan_body, carry_init, indices)

        J_local_tips = vmap(self._final_size_jacobian)(J_local_tips)
        Jd_local_tips = vmap(self._final_size_jacobian)(Jd_local_tips)

        return J_local_tips, Jd_local_tips

    @eqx.filter_jit
    def _J_Jd_local_batched(
        self, q: Array, qd: Array, s_ps: Array
    ) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at a batch of points s_ps along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).
            s_ps (Array): point coordinates along the robot in the interval [0, L] of shape (N,).

        Returns:
            J_local_ps (Array): Jacobians evaluated at all points, shape (N, 6, num_strains)
            Jd_local_ps (Array): Time-derivative of the Jacobians, shape (N, 6, num_strains)
        """
        # num points
        N = s_ps.shape[0]

        # compute the strain and strain rate for each segment
        xi = self.strain(q).reshape(self.num_segments, 6)
        xid = (self.B_xi @ qd).reshape(self.num_segments, 6)

        # classify all points along the robot to the corresponding segment and compute local coordinates
        segment_indices, s_local_ps = vmap(self.classify_segment)(s_ps)

        # compute the Jacobian at the tips
        J_tips, Jd_tips = self._J_Jd_local_tips(
            q, qd
        )  # shape (num_segments, 6, num_strains)

        # select the base Jacobian for each point (g0 for the first, previous tip otherwise)
        J_bases = jnp.concatenate([jnp.zeros_like(J_tips[:1]), J_tips[:-1]], axis=0)
        Jd_bases = jnp.concatenate([jnp.zeros_like(Jd_tips[:1]), Jd_tips[:-1]], axis=0)

        J_base_ps = J_bases[segment_indices]
        Jd_base_ps = Jd_bases[segment_indices]
        # select the other variables for each point
        xi_ps, xid_ps = xi[segment_indices], xid[segment_indices]
        idx_ps = segment_indices

        # reshape for easier indexing
        J_base_ps = J_base_ps.reshape(N, 6, self.num_segments, 6).transpose(
            0, 2, 1, 3
        )  # shape (N, 6, num_segments, 6)
        Jd_base_ps = Jd_base_ps.reshape(N, 6, self.num_segments, 6).transpose(
            0, 2, 1, 3
        )  # shape (N, 6, num_segments, 6)

        def integrate_segment(
            i: Array,
            xi_i: Array,
            xid_i: Array,
            s_local: Array,
            J_base: Array,
            Jd_base: Array,
        ) -> tuple[Array, Array]:
            Ad_inv = lie.Adjoint_gi_se3_inv(xi_i, s_local, eps=self.global_eps)
            T = lie.Tangent_gi_se3(xi_i, s_local, eps=self.tangent_eps)
            Td = lie.Tangent_derivative_gi_se3(
                xi_i, xid_i, s_local, eps=self.tangent_eps
            )

            J_rot = jnp.einsum("ij, njk->nik", Ad_inv, J_base)
            J_next = J_rot.at[i].set(Ad_inv @ T)

            J_next_slice = lax.dynamic_index_in_dim(J_next, i, axis=0, keepdims=False)
            eta = jnp.matmul(J_next_slice, xid_i)

            Ad_inv_dot = -lie.adjoint_se3(eta) @ Ad_inv

            Jd_rot = jnp.einsum("ij, njk->nik", Ad_inv, Jd_base) + jnp.einsum(
                "ij, njk->nik", Ad_inv_dot, J_base
            )
            Jd_next = Jd_rot.at[i].set(Ad_inv_dot @ T + Ad_inv @ Td)

            return J_next, Jd_next

        # vmap the segment integration over all points
        J_local_ps, Jd_local_ps = vmap(integrate_segment)(
            idx_ps, xi_ps, xid_ps, s_local_ps, J_base_ps, Jd_base_ps
        )

        # reshape back to (N, 6, num_strains)
        J_local_ps = vmap(self._final_size_jacobian)(J_local_ps)
        Jd_local_ps = vmap(self._final_size_jacobian)(Jd_local_ps)

        return J_local_ps, Jd_local_ps

    @eqx.filter_jit
    def jacobian_and_derivative_bodyframe(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
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

        J_local = J_local_ @ self.B_xi
        Jd_local = Jd_local_ @ self.B_xi

        return J_local, Jd_local

    @eqx.filter_jit
    def jacobian_and_derivative_inertialframe(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at a point s along the robot in the inertial frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_global (Array): Jacobian of the forward kinematics at point s in the inertial frame, shape (6, num_active_strains)
            Jd_global (Array): Time-derivative of the Jacobian at point s in the inertial frame, shape (6, num_active_strains)
        """
        J_local_, Jd_local_ = self._J_Jd_local(q, qd, s)

        # compute the Adjoint transformation matrix at point s
        g_s = self.forward_kinematics(q, s)
        g_rot = jnp.block(
            [[g_s[:3, :3], jnp.zeros((3, 1))], [jnp.zeros((1, 3)), jnp.ones((1, 1))]]
        )
        Ad_g = lie.Adjoint_g_SE3(g_rot)

        # compute the body twist eta
        J_body = J_local_ @ self.B_xi
        eta_body = J_body @ qd

        omega = eta_body[:3]
        eta_rot = jnp.concatenate([omega, jnp.zeros(3, dtype=eta_body.dtype)])
        Ad_g_dot = Ad_g @ lie.adjoint_se3(eta_rot)

        # rotate both J and Jd to the inertial frame
        J_global_ = jnp.einsum("ij, jk->ik", Ad_g, J_local_)
        Jd_global_ = jnp.einsum("ij, jk->ik", Ad_g, Jd_local_) + jnp.einsum(
            "ij, jk->ik", Ad_g_dot, J_local_
        )

        J_global = J_global_ @ self.B_xi
        Jd_global = Jd_global_ @ self.B_xi

        return J_global, Jd_global

    @eqx.filter_jit
    def jacobian(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a point s along the robot in the inertial frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_global (Array): Jacobian of the forward kinematics at point s in the inertial frame, shape (6, num_active_strains)
        """
        J_global = self.jacobian_inertialframe(q, s)

        return J_global

    @eqx.filter_jit
    def jacobian_and_derivative(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at a point s along the robot in the inertial frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_global (Array): Jacobian of the forward kinematics at point s in the inertial frame, shape (6, num_active_strains)
            Jd_global (Array): Time-derivative of the Jacobian at point s in the inertial frame, shape (6, num_active_strains)
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
            i (Array): index of the segment as array of shape ()

        Returns:
            A_i (Array): local cross-sectional area of the i-th segment
        """
        # Cross-sectional area for a circular cross-section
        A_i = jnp.pi * self.r[i] ** 2
        return A_i

    @eqx.filter_jit
    def _local_second_moment_of_area(self, i: Array) -> Array:
        """
        Compute the local second moment of area for the i-th segment.

        Args:
            i (Array): index of the segment as array of shape ()

        Returns:
            I_i (Array): local second moment of area of the i-th segment as array of shape (3, ) where the entries correspond to [I_xx, I_yy, I_zz]
        """
        # Second moment of area for a circular cross-section
        I_i = jnp.array(
            [
                jnp.pi * self.r[i] ** 4 / 2,  # I_xx
                jnp.pi * self.r[i] ** 4 / 4,  # I_yy
                jnp.pi * self.r[i] ** 4 / 4,  # I_zz
            ]
        )
        return I_i

    @eqx.filter_jit
    def _local_mass_matrix(self, i: Array) -> Array:
        """
        Compute the local mass matrix for the i-th segment.

        Args:
            i (Array): index of the segment as array of shape ()
        Returns:
            M_i (Array): local mass matrix of shape (6, 6) for the i-th segment
        """
        rho_i = self.rho[i]
        A_i = self._local_cross_sectional_area(i)  # Cross-sectional area
        I_i = self._local_second_moment_of_area(
            i
        )  # Second moment of area as array of shape (3, )

        M_i = rho_i * jnp.diag(jnp.array([I_i[0], I_i[1], I_i[2], A_i, A_i, A_i]))
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
        # compute the gauss quadrature points and weights for each segment
        Xs_scaled, Ws_scaled = vmap(
            scale_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.Xs, self.Ws, self.L_cum[:-1], self.L_cum[1:]
        )  # shape (num_segments, num_gauss_points) for both Xs_scaled and Ws_scaled

        # compute the jacobian for each quadrature point
        J_ps = self._J_local_batched(
            q, Xs_scaled.flatten()
        )  # shape (num_segments * num_gauss_points, 6, num_active_strains)
        J_ps = J_ps.reshape(
            self.num_segments, self.num_gauss_points, *J_ps.shape[1:]
        )  # shape (num_segments, num_gauss_points, 6, num_active_strains)

        def B_i(i: Array) -> Array:
            M_i = self._local_mass_matrix(i)

            def B_ij(j: Array) -> Array:
                # select the j-th quadrature weight
                Ws_ij = Ws_scaled[i][j]
                # select the j-th jacobian
                J_ij = J_ps[i, j]

                B_ij = Ws_ij * J_ij.T @ M_i @ J_ij
                return B_ij

            # we can skip the first and last quadrature points since their weight is zero
            B_blocks_i = vmap(B_ij)(jnp.arange(1, self.num_gauss_points - 1))

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
        # compute the gauss quadrature points and weights for each segment
        Xs_scaled, Ws_scaled = vmap(
            scale_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.Xs, self.Ws, self.L_cum[:-1], self.L_cum[1:]
        )  # shape (num_segments, num_gauss_points) for both Xs_scaled and Ws_scaled

        # compute the jacobian and its time-derivative for each quadrature point
        J_ps, Jd_ps = self._J_Jd_local_batched(
            q, qd, Xs_scaled.flatten()
        )  # shape (num_segments * num_gauss_points, 6, num_active_strains)
        J_ps = J_ps.reshape(
            self.num_segments, self.num_gauss_points, *J_ps.shape[1:]
        )  # shape (num_segments, num_gauss_points, 6, num_active_strains)
        Jd_ps = Jd_ps.reshape(
            self.num_segments, self.num_gauss_points, *Jd_ps.shape[1:]
        )  # shape (num_segments, num_gauss_points, 6, num_active_strains)

        def C_i(i: Array) -> Array:
            M_i = self._local_mass_matrix(i)

            def C_ij(j: Array) -> Array:
                # select the j-th quadrature weight
                Ws_ij = Ws_scaled[i][j]
                # select the j-th jacobian and its time-derivative
                J_ij = J_ps[i, j]
                Jd_ij = Jd_ps[i, j]

                C_ij = (
                    Ws_ij
                    * J_ij.T
                    @ (
                        M_i @ Jd_ij
                        + lie.coadjoint_se3(J_ij @ self.B_xi @ qd) @ M_i @ J_ij
                    )
                )
                return C_ij

            # we can skip the first and last quadrature points since their weight is zero
            C_blocks_i = vmap(C_ij)(jnp.arange(1, self.num_gauss_points - 1))

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
        # compute the gauss quadrature points and weights for each segment
        Xs_scaled, Ws_scaled = vmap(
            scale_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.Xs, self.Ws, self.L_cum[:-1], self.L_cum[1:]
        )  # shape (num_segments, num_gauss_points) for both Xs_scaled and Ws_scaled

        # compute the forward kinematics for each quadrature point
        g_ps = self.forward_kinematics_batched(
            q, Xs_scaled.flatten()
        )  # shape (num_segments * num_gauss_points, 4, 4)
        g_ps = g_ps.reshape(
            self.num_segments, self.num_gauss_points, 4, 4
        )  # shape (num_segments, num_gauss_points, 4, 4)

        # compute the jacobian for each quadrature point
        J_ps = self._J_local_batched(
            q, Xs_scaled.flatten()
        )  # shape (num_segments * num_gauss_points, 6, num_active_strains)
        J_ps = J_ps.reshape(
            self.num_segments, self.num_gauss_points, *J_ps.shape[1:]
        )  # shape (num_segments, num_gauss_points, 6, num_active_strains)

        def G_i(i: Array) -> Array:
            M_i = self._local_mass_matrix(i)

            def G_ij(j: Array) -> Array:
                # select the j-th quadrature weight
                Ws_ij = Ws_scaled[i][j]
                # select the j-th Cartesian pose
                g_ij = g_ps[i, j]
                # select the j-th jacobian and its time-derivative
                J_ij = J_ps[i, j]

                # compute the inverse Adjoint transformation matrix at point s
                Ad_g_inv_ij = lie.Adjoint_g_inv_SE3(g_ij)

                G_ij = -Ws_ij * J_ij.T @ M_i @ Ad_g_inv_ij @ self.g
                return G_ij

            # we can skip the first and last quadrature points since their weight is zero
            G_blocks_segment_i = vmap(G_ij)(jnp.arange(1, self.num_gauss_points - 1))

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
            i (Array): index of the segment as array of shape ()

        Returns:
            S_i (Array): Local stiffness matrix of shape (6, 6) for the i-th segment.
        """
        I_i = self._local_second_moment_of_area(
            i
        )  # Second moment of area as array of shape (3, )
        A_i = self._local_cross_sectional_area(i)  # Cross-sectional area

        S_i = self.L[i] * jnp.diag(
            jnp.stack(
                [
                    self.G[i] * I_i[0],  # torsion X
                    self.E[i] * I_i[1],  # bending Y
                    self.E[i] * I_i[2],  # bending Z
                    A_i * self.E[i],  # axial X
                    A_i * self.G[i],  # shear Y
                    A_i * self.G[i],  # shear Z
                ],
                axis=0,
            )
        )

        return S_i

    @eqx.filter_jit
    def _stiffness(self, formulate_in_strain_space: bool = False) -> Array:
        # stiffness matrix of shape (num_segments, 6, 6)
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
    def damping_matrix(self, q: Array) -> Array:
        """
        Compute the damping matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

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
    def gravitational_energy(self, q: Array) -> Array:
        """
        Compute the gravitational energy of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            U_G (float): Gravitational energy of the robot.
        """
        # compute the gauss quadrature points and weights for each segment
        Xs_scaled, Ws_scaled = vmap(
            scale_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.Xs, self.Ws, self.L_cum[:-1], self.L_cum[1:]
        )  # shape (num_segments, num_gauss_points) for both Xs_scaled and Ws_scaled

        # compute the forward kinematics for each quadrature point
        g_ps = self.forward_kinematics_batched(
            q, Xs_scaled.flatten()
        )  # shape (num_segments * num_gauss_points, 4, 4)
        g_ps = g_ps.reshape(
            self.num_segments, self.num_gauss_points, 4, 4
        )  # shape (num_segments, num_gauss_points, 4, 4)

        def U_G_i(i: Array) -> Array:
            rho_i = self.rho[i]
            A_i = self._local_cross_sectional_area(i)  # Cross-sectional area

            def U_G_ij(j: Array) -> Array:
                # select the j-th quadrature weight
                Ws_ij = Ws_scaled[i][j]
                # select the j-th Cartesian pose
                g_ij = g_ps[i, j]

                p_j = jnp.concatenate(
                    [jnp.zeros(3), g_ij[:3, 3]]
                )  # Add zeros for the orientation angles
                U_G_ij = -Ws_ij * rho_i * A_i * jnp.dot(p_j, self.g)
                return U_G_ij

            # we can skip the first and last quadrature points since their weight is zero
            U_G_blocks_segment_i = vmap(U_G_ij)(
                jnp.arange(1, self.num_gauss_points - 1)
            )

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
    def forward_dynamics(
        self, t: Array, y: Array, actuation_args: tuple | None = None
    ) -> Array:
        """
        Forward dynamics function.

        Args:
            t (Array): Current time.
            y (Array): State vector containing configuration and velocity.
                Shape is (2 * num_strains,).
            actuation_args (tuple, optional): Additional arguments for the actuation mapping function.
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

        # compute the gauss quadrature points and weights for each segment
        Xs_scaled, Ws_scaled = vmap(
            scale_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.Xs, self.Ws, self.L_cum[:-1], self.L_cum[1:]
        )  # shape (num_segments, num_gauss_points) for both Xs_scaled and Ws_scaled

        # compute the forward kinematics for each quadrature point
        g_ps = self.forward_kinematics_batched(
            q, Xs_scaled.flatten()
        )  # shape (num_segments * num_gauss_points, 4, 4)
        g_ps = g_ps.reshape(
            self.num_segments, self.num_gauss_points, 4, 4
        )  # shape (num_segments, num_gauss_points, 4, 4)

        # compute the jacobian and its time-derivative for each quadrature point
        J_ps, Jd_ps = self._J_Jd_local_batched(
            q, qd, Xs_scaled.flatten()
        )  # shape (num_segments * num_gauss_points, 6, num_active_strains)
        J_ps = J_ps.reshape(
            self.num_segments, self.num_gauss_points, *J_ps.shape[1:]
        )  # shape (num_segments, num_gauss_points, 6, num_active_strains)
        Jd_ps = Jd_ps.reshape(
            self.num_segments, self.num_gauss_points, *Jd_ps.shape[1:]
        )  # shape (num_segments, num_gauss_points, 6, num_active_strains)

        def dynamical_matrices_i(i: Array) -> tuple[Array, Array, Array]:
            """
            Compute the integrand for the dynamical matrices at the i-th segment.
            Args:
                i (Array): index of the segment
            Returns:
                tuple[Array, Array, Array]: The inertia matrix, Coriolis matrix, and gravitational force integrands.
            """
            M_i = self._local_mass_matrix(i)

            def dynamical_matrices_ij(j: Array) -> tuple[Array, Array, Array]:
                """
                Compute the integrand for the dynamical matrices at the j-th quadrature point of the i-th segment.
                Args:
                    j (Array): index of the quadrature point
                Returns:
                    tuple[Array, Array, Array]: The inertia matrix, Coriolis matrix, and gravitational force integrands.
                """
                # select the j-th quadrature weight
                Ws_ij = Ws_scaled[i][j]
                # select the j-th Cartesian pose
                g_ij = g_ps[i, j]
                # select the j-th jacobian and its time-derivative
                J_ij = J_ps[i, j]
                Jd_ij = Jd_ps[i, j]

                # compute the lie algebra expressions.
                Ad_g_inv_ij = lie.Adjoint_g_inv_SE3(g_ij)

                # compute the inertia matrix integrand
                B_ij = Ws_ij * J_ij.T @ M_i @ J_ij

                # compute the coriolis matrix integrand
                C_ij = Ws_ij * (
                    J_ij.T
                    @ (
                        M_i @ Jd_ij
                        + lie.coadjoint_se3(J_ij @ self.B_xi @ qd) @ M_i @ J_ij
                    )
                )

                # compute the gravitational force integrand
                G_ij = -Ws_ij * J_ij.T @ M_i @ Ad_g_inv_ij @ self.g

                return B_ij, C_ij, G_ij

            # we can skip the first and last quadrature points since their weight is zero
            B_blocks_i, C_blocks_i, G_blocks_i = vmap(dynamical_matrices_ij)(
                jnp.arange(1, self.num_gauss_points - 1)
            )

            return B_blocks_i, C_blocks_i, G_blocks_i

        # compute the dynamical matrices for each segment
        B_blocks_tot, C_blocks_tot, G_blocks_tot = vmap(dynamical_matrices_i)(
            jnp.arange(self.num_segments)
        )

        # sum over segments and Gauss points
        B_full = jnp.sum(B_blocks_tot, axis=(0, 1))
        C_full = jnp.sum(C_blocks_tot, axis=(0, 1))
        G_full = jnp.sum(G_blocks_tot, axis=(0, 1))

        # construct the dynamical matrices
        B = self.B_xi.T @ B_full @ self.B_xi
        C = self.B_xi.T @ C_full @ self.B_xi
        G = self.B_xi.T @ G_full
        D = self.damping_matrix(q)
        tau_el = self.elastic_force(q)
        tau_u = self.actuation_force(q, u)

        B_inv = jnp.linalg.inv(B)  # Inverse of the inertia matrix
        qdd = B_inv @ (
            tau_u + tau_ext - C @ qd - G - tau_el - D @ qd
        )  # Compute the acceleration

        yd = jnp.concatenate([qd, qdd])

        return yd
