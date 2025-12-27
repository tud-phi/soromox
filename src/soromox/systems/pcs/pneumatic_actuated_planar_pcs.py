__all__ = ["PneumaticActuatedPlanarPCS"]

import equinox as eqx
import jax.numpy as jnp
from jax import Array, vmap

from .planar_pcs import PlanarPCS


class PneumaticActuatedPlanarPCS(PlanarPCS):
    """
    Pneumatically Actuated Planar Piecewise Constant Strain (PCS) model for 2D soft continuum robots.

    This class implements the geometric and dynamic modeling of a 2D soft robot
    using the Cosserat rod theory and piecewise constant strain assumption.
    It supports computation of forward kinematics, Jacobians, dynamical matrices.

    Attributes:
    ----------
    num_segments : int
        Number of segments (constant strain sections) along the robot.
    num_actuators : int
        Number of actuators (control inputs) for the robot (2 per actuated segment in the case of planar pneumatically-actuated PCS).
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
    r_chamber_in : Array
        Inner radius of each segment's pneumatic chamber.
    r_chamber_out : Array
        Outer radius of each segment's pneumatic chamber.
    phi_chamber : Array
        Sector angle of each segment's pneumatic chamber.
    d_chamber : Array
        Radial distance of the center of the chambers from the centerline of the backbone for each segment.
    num_chambers_per_segment : int
        Number of pneumatic chambers per segment (default is 2).
    actuation_basis : Array
        Actuation basis matrix for mapping control inputs to segment strains.
    simplified_actuation_mapping : bool
        If True, uses a simplified actuation mapping (default is False).
    chamber_cross_section_geometry : str
        The cross sectional geometry of the chambers.
    pneumatic_load_distribution_assumption : str
        Determines the assumption used to map the pneumatic forces into configuration space.

    Notes:
    -----
    - The strain vector is composed of 3 components per segment:
      [kappa_z, sigma_x, sigma_y].
      By default, the rod is assumed to be straight and aligned with the x-axis,
        so the reference strain is set to [0, 1, 0].
        Thus:   - kappa_z corresponds to bending around the z-axis,
                - sigma_x corresponds to axial strain along the x-axis,
                - sigma_y corresponds to shear along the y-axis.

    - The actuation mapping assumes that each segment contains two identical and symmetric pneumatic chambers with pressures p1 and p2, which correspond to the right and left chamber pressures, respectively.
        The control inputs u1 and u2 are mapped as follows to the pressures:
            - p1 = u1 (right chamber)
            - p2 = u2 (left chamber)

    """

    r_chamber_in: Array  # inner radius of each segment's chamber, shape (num_segments,)
    r_chamber_out: (
        Array  # outer radius of each segment's chamber, shape (num_segments,)
    )
    phi_chamber: Array  # sector angle of each segment's chamber, shape (num_segments,)
    d_chamber: Array  # radial distance of the center of the chambers from the centerline of the backbone, shape (num_segments,)

    actuation_basis: Array  # actuation basis, shape (num_segments * 2, num_actuators)

    # fields with default values
    num_chambers_per_segment: int = eqx.field(
        static=True, default=2
    )  # number of pneumatic chambers per segment
    chamber_cross_section_geometry: str = eqx.field(static=True, default="circular")
    pneumatic_load_distribution_assumption: str = eqx.field(
        static=True, default="infitesimal"
    )

    def __init__(
        self,
        num_segments: int,
        params: dict[str, Array],
        *args,
        segment_actuation_selector: Array | None = None,
        chamber_cross_section_geometry: str = "circular",
        pneumatic_load_distribution_assumption: str = "infitesimal",
        **kwargs,
    ):
        """
        Initialize the PneumaticallyActuatedPlanarPCS class
        Args:
            num_segments (int): number of segments in the robot
            params (Dict[str, Array]): dictionary containing the robot parameters
            order_gauss (int): order of the Gauss quadrature rule
            strain_selector (Optional[Array]): array to select which strains are active. All six strains are active by default.
            xi_ref (Optional[Array]): reference configuration for the robot
            segment_actuation_selector (Optional[Array]): array to select the segments to be actuated
            chamber_cross_section_geometry: the cross sectional geometry of the chambers. Options:
                - circular: assumes circular chamber cross sections with outside radius "r_chamber_out" and inside radius "r_chamber_in" with their center at a distance of "d_chamber" from the centerline of the backbone
                - concentric: assumes chambers that are concentric with the circular geometry of the soft robot cross sectional geometry with outside radius "r_chamber_out", inside radius "r_chamber_in", and sector angle "phi_chamber".
            pneumatic_load_distribution_assumption: determines the assumption used to map the pneumatic forces into configuration space. Options:
                - infitesimal: assumes infitesimally thin chambers (in terms of height) such that the pneumatic forces and torques directly act on the generalized coordinates
                - cap_bodyframe_jacobians: assumes that the pneumatic chambers contain two discrete caps at the proximal and distal end of the segment, respectively. Then, the bodyframe jacobian is used to map the wrenches on the generalized coordinates
        """
        super().__init__(
            num_segments,
            params,
            *args,
            **kwargs,
        )

        if segment_actuation_selector is None:
            segment_actuation_selector = jnp.ones(num_segments, dtype=bool)

        self.num_actuators = (
            int(jnp.sum(segment_actuation_selector)) * self.num_chambers_per_segment
        )  # each segment has two control inputs u1 and u2

        actuation_basis = jnp.zeros((2 * self.num_segments, self.num_actuators))
        actuation_basis_cumsum = jnp.cumsum(segment_actuation_selector)
        for i in range(self.num_segments):
            j = int(actuation_basis_cumsum[i].item()) - 1
            if segment_actuation_selector[i].item() is True:
                actuation_basis = actuation_basis.at[2 * i, j].set(1.0)
                actuation_basis = actuation_basis.at[2 * i + 1, j + 1].set(1.0)
        self.actuation_basis = actuation_basis

        self.chamber_cross_section_geometry = chamber_cross_section_geometry
        self.pneumatic_load_distribution_assumption = (
            pneumatic_load_distribution_assumption
        )

        self._set_params(params)

    def _set_params(self, params: dict[str, Array]):
        """
        Set the parameters of the PneumaticallyActuatedPlanarPCS.

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
                - "r_chamber_in" : Array of num_segments floats
                    Inner radius of each segment's pneumatic chamber [m]
                - "r_chamber_out" : Array of num_segments floats
                    Outer radius of each segment's pneumatic chamber [m]
                - "phi_chamber" : Array of num_segments floats
                    Sector angle of each segment's pneumatic chamber [rad]
                - "d_chamber" : Array of num_segments floats
                    Radial distance of the center of the chambers from the centerline of the backbone [m]

        """
        super()._set_params(params)

        # Pneumatic chamber parameters
        try:
            r_chamber_in = params["r_chamber_in"]
        except KeyError:
            raise KeyError(
                "The parameter 'r_chamber_in' (inner radius of each segment's pneumatic chamber) is required for the pneumatically actuated planar PCS."
            )
        if not isinstance(r_chamber_in, (list, jnp.ndarray)):
            raise TypeError(
                "The parameter 'r_chamber_in' must be a list or a jnp.ndarray."
            )
        if len(r_chamber_in) != self.num_segments:
            raise ValueError(
                f"The parameter 'r_chamber_in' must have the same length as the number of segments ({self.num_segments})."
            )
        self.r_chamber_in = jnp.asarray(r_chamber_in, dtype=jnp.float64)

        try:
            r_chamber_out = params["r_chamber_out"]
        except KeyError:
            raise KeyError(
                "The parameter 'r_chamber_out' (outer radius of each segment's pneumatic chamber) is required for the pneumatically actuated planar PCS."
            )
        if not isinstance(r_chamber_out, (list, jnp.ndarray)):
            raise TypeError(
                "The parameter 'r_chamber_out' must be a list or a jnp.ndarray."
            )
        if len(r_chamber_out) != self.num_segments:
            raise ValueError(
                f"The parameter 'r_chamber_out' must have the same length as the number of segments ({self.num_segments})."
            )
        self.r_chamber_out = jnp.asarray(r_chamber_out, dtype=jnp.float64)

        phi_chamber = params.get("phi_chamber", jnp.ones((self.num_segments,)))
        if not isinstance(phi_chamber, (list, jnp.ndarray)):
            raise TypeError(
                "The parameter 'phi_chamber' must be a list or a jnp.ndarray."
            )
        if len(phi_chamber) != self.num_segments:
            raise ValueError(
                f"The parameter 'phi_chamber' must have the same length as the number of segments ({self.num_segments})."
            )
        self.phi_chamber = jnp.asarray(phi_chamber, dtype=jnp.float64)

        d_chamber = params.get("d_chamber", jnp.ones((self.num_segments,)))
        if not isinstance(d_chamber, (list, jnp.ndarray)):
            raise TypeError(
                "The parameter 'd_chamber' must be a list or a jnp.ndarray."
            )
        if len(d_chamber) != self.num_segments:
            raise ValueError(
                f"The parameter 'd_chamber' must have the same length as the number of segments ({self.num_segments})."
            )
        self.d_chamber = jnp.asarray(d_chamber, dtype=jnp.float64)

    def update_params(self, params: dict[str, Array]) -> "PneumaticActuatedPlanarPCS":
        """
        Update the parameters of the PneumaticallyActuatedPlanarPCS.

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
                - "r_chamber_in" : Array of num_segments floats
                    Inner radius of each segment's pneumatic chamber [m]
                - "r_chamber_out" : Array of num_segments floats
                    Outer radius of each segment's pneumatic chamber [m]
                - "phi_chamber" : Array of num_segments floats
                    Sector angle of each segment's pneumatic chamber [rad]
                - "d_chamber" : Array of num_segments floats
                    Radial distance of the center of the chambers from the centerline of the backbone [m]
        Returns:
            updated_self (PneumaticallyActuatedPlanarPCS):
                A new instance of PneumaticallyActuatedPlanarPCS with updated parameters.
        """
        # Apply updates sequentially
        updated_self = super().update_params(params)

        if "r_chamber_in" in params:
            r_chamber_in = params["r_chamber_in"]
            if not isinstance(r_chamber_in, (list, jnp.ndarray)):
                raise TypeError(
                    "The parameter 'r_chamber_in' must be a list or a jnp.ndarray."
                )
            if len(r_chamber_in) != self.num_segments:
                raise ValueError(
                    f"The parameter 'r_chamber_in' must have the same length as the number of segments ({self.num_segments})."
                )
            updated_self = eqx.tree_at(
                lambda x: x.r_chamber_in,
                updated_self,
                jnp.asarray(r_chamber_in, dtype=jnp.float64),
            )

        if "r_chamber_out" in params:
            r_chamber_out = params["r_chamber_out"]
            if not isinstance(r_chamber_out, (list, jnp.ndarray)):
                raise TypeError(
                    "The parameter 'r_chamber_out' must be a list or a jnp.ndarray."
                )
            if len(r_chamber_out) != self.num_segments:
                raise ValueError(
                    f"The parameter 'r_chamber_out' must have the same length as the number of segments ({self.num_segments})."
                )
            updated_self = eqx.tree_at(
                lambda x: x.r_chamber_out,
                updated_self,
                jnp.asarray(r_chamber_out, dtype=jnp.float64),
            )

        if "phi_chamber" in params:
            phi_chamber = params["phi_chamber"]
            if not isinstance(phi_chamber, (list, jnp.ndarray)):
                raise TypeError(
                    "The parameter 'phi_chamber' must be a list or a jnp.ndarray."
                )
            if len(phi_chamber) != self.num_segments:
                raise ValueError(
                    f"The parameter 'phi_chamber' must have the same length as the number of segments ({self.num_segments})."
                )
            updated_self = eqx.tree_at(
                lambda x: x.phi_chamber,
                updated_self,
                jnp.asarray(phi_chamber, dtype=jnp.float64),
            )

        if "d_chamber" in params:
            d_chamber = params["d_chamber"]
            if not isinstance(d_chamber, (list, jnp.ndarray)):
                raise TypeError(
                    "The parameter 'd_chamber' must be a list or a jnp.ndarray."
                )
            if len(d_chamber) != self.num_segments:
                raise ValueError(
                    f"The parameter 'd_chamber' must have the same length as the number of segments ({self.num_segments})."
                )
            updated_self = eqx.tree_at(
                lambda x: x.d_chamber,
                updated_self,
                jnp.asarray(d_chamber, dtype=jnp.float64),
            )

        return updated_self

    @eqx.filter_jit
    def _local_chamber_cross_sectional_area(self, i: Array) -> Array:
        """
        Compute the local cross-sectional area of one pneumatic chamber for the i-th segment.

        Args:
            i (Array): index of the segment

        Returns:
            A_one_chamber_i (Array): local cross-sectional area of one pneumatic chamber of the i-th segment
        """
        match self.chamber_cross_section_geometry:
            case "circular":
                A_one_chamber_i = jnp.pi * (
                    self.r_chamber_out[i] ** 2 - self.r_chamber_in[i] ** 2
                )
            case "concentric":
                A_one_chamber_i = (
                    self.phi_chamber[i]
                    / 2
                    * (self.r_chamber_out[i] ** 2 - self.r_chamber_in[i] ** 2)
                )
            case _:
                raise NotImplementedError(
                    f"The chamber cross section geometry {self.chamber_cross_section_geometry} has not been implemented yet."
                )

        return A_one_chamber_i

    @eqx.filter_jit
    def _local_cross_sectional_area(self, i: Array) -> Array:
        """
        Compute the local cross-sectional area for the i-th segment.

        Args:
            i (Array): index of the segment

        Returns:
            A_i (Array): local cross-sectional area of the i-th segment
        """
        A_full_i = super()._local_cross_sectional_area(
            i
        )  # Full cross-sectional area of the i-th segment without chambers
        A_one_chamber_i = self._local_chamber_cross_sectional_area(i)
        A_i = (
            A_full_i - self.num_chambers_per_segment * A_one_chamber_i
        )  # Subtract the area of the four chambers

        return A_i

    @eqx.filter_jit
    def _local_chamber_second_moment_of_area(self, i: Array) -> Array:
        """
        Compute the local second moment of area of one pneumatic chamber for the i-th segment.

        Args:
            i (Array): index of the segment

        Returns:
            I_one_chamber_i (Array): local second moment of area of one pneumatic chamber of the i-th segment
        """
        match self.chamber_cross_section_geometry:
            case "circular":
                I_one_chamber_i = (
                    jnp.pi
                    / 4
                    * (
                        self.r_chamber_out[i] ** 4 - self.r_chamber_in[i] ** 4
                    )  # second moment of area of the annulus itself
                    + self._local_chamber_cross_sectional_area(i)
                    * self.d_chamber[i] ** 2  # parallel axis theorem
                )
            case "concentric":
                I_one_chamber_i = (
                    (self.r_chamber_out[i] ** 4 - self.r_chamber_in[i] ** 4)
                    * (
                        self.phi_chamber[i]
                        - 2
                        * jnp.sin(self.phi_chamber[i] / 2)
                        * jnp.cos(self.phi_chamber[i] / 2)
                    )
                    / 8
                )
            case _:
                raise NotImplementedError(
                    f"The chamber cross section geometry {self.chamber_cross_section_geometry} has not been implemented yet."
                )

        return I_one_chamber_i

    @eqx.filter_jit
    def _local_second_moment_of_area(self, i: Array) -> Array:
        """
        Compute the local second moment of area for the i-th segment.

        Args:
            i (Array): index of the segment

        Returns:
            I_i (Array): local second moment of area of the i-th segment
        """
        I_full_i = super()._local_second_moment_of_area(
            i
        )  # Full second moment of area of the i-th segment without chambers
        I_one_chamber_i = self._local_chamber_second_moment_of_area(i)
        I_i = (
            I_full_i - self.num_chambers_per_segment * I_one_chamber_i
        )  # Subtract the second moment of area of the four chambers

        return I_i

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Compute the actuation matrix of the robot.
        We assume that each segment contains two identical and symmetric pneumatic chambers with pressures,
            where p1 and p2 are the right and left chamber pressures respectively.
        We map the control inputs u1 and u2 as follows to the pressures:
            p1 = u1 (right chamber)
            p2 = u2 (left chamber)

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            A (Array): Actuation matrix of shape (num_active_strains, num_actuators)
        """

        def _actuation_matrix_segment_i(i: Array) -> Array:
            """
            Compute the actuation matrix for the i-th segment.
            Args:
                i (Array): index of the segment
            """
            # Area of one pneumatic chamber
            A_one_chamber = self._local_chamber_cross_sectional_area(i)

            match self.chamber_cross_section_geometry:
                case "circular":
                    r_center_of_pressure = self.d_chamber[i]
                case "concentric":
                    # Distance from the center of the segment to the center of pressure of one chamber
                    r_center_of_pressure = (
                        2
                        / 3
                        * jnp.sinc(self.phi_chamber[i] / 2)
                        * (self.r_chamber_out[i] ** 3 - self.r_chamber_in[i] ** 3)
                        / (self.r_chamber_out[i] ** 2 - self.r_chamber_in[i] ** 2)
                    )

            match self.pneumatic_load_distribution_assumption:
                case "infitesimal":
                    A_full_segment_i = jnp.array(
                        [
                            [
                                A_one_chamber * r_center_of_pressure,
                                -A_one_chamber * r_center_of_pressure,
                            ],  # actuation on the bending
                            [
                                A_one_chamber,
                                A_one_chamber,
                            ],  # actuation on the axial strain
                            [0.0, 0.0],  # actuation on the shear strain
                        ]
                    )

                    A_segment_i = self.B_xi.T @ A_full_segment_i
                case "cap_bodyframe_jacobians":
                    # Jacobians at the base and tip of the segment
                    J_base_i = self.jacobian_bodyframe(q, self.L_cum[i])
                    J_tip_i = self.jacobian_bodyframe(q, self.L_cum[i + 1])

                    # Contribution of the proximal end
                    A_segment_i_proximal_end = J_base_i.T @ jnp.array(
                        [
                            [
                                -A_one_chamber * r_center_of_pressure,
                                A_one_chamber * r_center_of_pressure,
                            ],
                            [-A_one_chamber, -A_one_chamber],
                            [0.0, 0.0],
                        ]
                    )
                    # Contribution of the distal end
                    A_segment_i_distal_end = J_tip_i.T @ jnp.array(
                        [
                            [
                                A_one_chamber * r_center_of_pressure,
                                -A_one_chamber * r_center_of_pressure,
                            ],
                            [A_one_chamber, A_one_chamber],
                            [0.0, 0.0],
                        ]
                    )

                    # sum the contributions of the distal and proximal ends
                    A_segment_i = A_segment_i_distal_end + A_segment_i_proximal_end
                case _:
                    raise NotImplementedError(
                        f"The pneumatic load distribution assumption {self.pneumatic_load_distribution_assumption} is not (fully) implemented yet."
                    )

            return A_segment_i

        A_blocks_tot = vmap(_actuation_matrix_segment_i)(
            jnp.arange(self.num_segments),
        )

        # # For debugging purposes, we can use a for loop instead of vmap
        # A_blocks_tot = jnp.stack(
        #     [A_segment_i(i) for i in range(self.num_segments)],
        #     axis=0
        # )

        # we need to sum the contributions of the actuation of each segment
        A = jnp.concatenate(A_blocks_tot, axis=-1)

        # apply the actuation_basis
        A = A @ self.actuation_basis

        return A
