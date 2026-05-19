from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax import Array, vmap

from soromox.systems.pcs.params import ISupportParams
from soromox.systems.pcs.structures import PCSStructure
from soromox.utils.array_math import blk_diag

from .pcs import PCS


class ISupport(PCS):
    """
    A kinematic and dynamic model for the (AM) I-Support robot based on the Piecewise Constant Strain shape parametrization.

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
        num_gauss_points: Requested nonzero Gauss-Legendre quadrature nodes.
        num_integration_points: Stored integration nodes, including zero-weight endpoints.
        integration_points, integration_weights: Quadrature nodes and weights.
        r_chamber_in: Inner radius of each segment's actuator [m].
        r_chamber_out: Outer radius of each segment's actuator [m].
        d_chamber: Radial distance of the center of the actuators from the centerline of the backbone [m].
        varphi_chamber_off: Angular offset of the first actuator from the local segment frame [rad].

    References:
        Arleo, L., Stano, G., Percoco, G., & Cianchetti, M. (2021). I-support soft
        arm for assistance tasks: a new manufacturing approach based on 3D printing
        and characterization. Progress in Additive Manufacturing, 6(2), 243-256.
        https://doi.org/10.1007/s40964-020-00158-y

        Alessi, C., Falotico, E., & Lucantonio, A. (2023). Ablation study of a
        dynamic model for a 3D-printed pneumatic soft robotic arm. IEEE Access, 11,
        37840-37853. https://doi.org/10.1109/ACCESS.2023.3265261

        Alessi, C., Bianchi, D., Stano, G., Cianchetti, M., & Falotico, E. (2024).
        Pushing with soft robotic arms via deep reinforcement learning. Advanced
        Intelligent Systems, 6(8), 2300899. https://doi.org/10.1002/aisy.202300899

    """

    params: ISupportParams

    r_chamber_in: Array  # inner radius of each segment's chamber, shape (num_segments,)
    r_chamber_out: (
        Array  # outer radius of each segment's chamber, shape (num_segments,)
    )
    d_chamber: Array  # radial distance of the center of the chambers from the centerline of the backbone, shape (num_segments,)
    varphi_chamber_off: Array  # angular offset of the first actuator from the local

    num_chambers_per_segment: int = eqx.field(
        static=True, default=3
    )  # number of pneumatic chambers per segment

    def __init__(
        self,
        params: ISupportParams,
        structure: PCSStructure | None = None,
        num_chambers_per_segment: int = 3,
        **kwargs: Any,
    ):
        """
        Initialize the ISupport class

        Args:
            params: Dynamic I-SUPPORT parameters.
            structure: Static PCS layout. If omitted, the default PCS structure
                is used.
            num_chambers_per_segment:
                Number of pneumatic chambers per segment. Defaults to 3.
            **kwargs: Additional keyword arguments.
        """
        if not isinstance(params, ISupportParams):
            raise TypeError("params must be an ISupportParams instance.")
        super().__init__(params, structure=structure, **kwargs)

        self.num_chambers_per_segment = num_chambers_per_segment
        self.num_actuators = (
            self.num_segments * self.num_chambers_per_segment
        )  # each segment has one pressure input per chamber

        self._set_params(params)

    def _set_params(self, params: ISupportParams):
        """
        Set the parameters of the ISupport class.

        Args:
            params (Dict[str, Array]): Dictionary containing the parameters of the robot.
                Dictionary containing the robot parameters:
                - "p0": (optional) List/Array of shape (6,)
                    Initial orientation angle and position in the inertial frame [rad, m]
                    [ψ, θ, φ, x0, y0, z0]
                        [ψ, θ, φ] are the Euler angles in the ZXZ convention:
                            ψ (psi) : Rotation around Z axis (fixed axis)
                            θ (thêta) : Rotation around X' axis (movable axis after first rotation)
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
                - "D": List/Array of (num_segments, num_segments) floats
                    Damping matrix of each segment [Pa*s]
                - "r_chamber_in" : Array of num_segments floats
                    Inner radius of each segment's pneumatic chamber [m]
                - "r_chamber_out" : Array of num_segments floats
                    Outer radius of each segment's pneumatic chamber [m]
                - "d_chamber" : Array of num_segments floats
                    Radial distance of the center of the chambers from the centerline of the backbone [m]
                - "varphi_chamber_off" : Array of num_segments floats
                    Angular offset of the first actuator from the local z-axis [rad]

        """
        super()._set_params(params)

        # Pneumatic chamber parameters
        r_chamber_in = jnp.asarray(params.chamber_inner_radius, dtype=jnp.float64)
        if r_chamber_in.shape != (self.num_segments,):
            raise ValueError(
                "chamber_inner_radius must have shape "
                f"({self.num_segments},), got {r_chamber_in.shape}."
            )
        self.r_chamber_in = r_chamber_in

        r_chamber_out = jnp.asarray(params.chamber_outer_radius, dtype=jnp.float64)
        if r_chamber_out.shape != (self.num_segments,):
            raise ValueError(
                "chamber_outer_radius must have shape "
                f"({self.num_segments},), got {r_chamber_out.shape}."
            )
        self.r_chamber_out = r_chamber_out

        d_chamber = jnp.asarray(params.chamber_distance, dtype=jnp.float64)
        if d_chamber.shape != (self.num_segments,):
            raise ValueError(
                "chamber_distance must have shape "
                f"({self.num_segments},), got {d_chamber.shape}."
            )
        self.d_chamber = d_chamber

        varphi_chamber_off = jnp.asarray(params.chamber_angle_offset, dtype=jnp.float64)
        if varphi_chamber_off.shape != (self.num_segments,):
            raise ValueError(
                "chamber_angle_offset must have shape "
                f"({self.num_segments},), got {varphi_chamber_off.shape}."
            )
        self.varphi_chamber_off = varphi_chamber_off

    def with_params(self, params: ISupportParams) -> "ISupport":
        """Return an updated copy with a full typed parameter object."""
        if not isinstance(params, ISupportParams):
            raise TypeError("params must be an ISupportParams instance.")
        chamber_arrays = (
            jnp.asarray(params.chamber_inner_radius, dtype=jnp.float64),
            jnp.asarray(params.chamber_outer_radius, dtype=jnp.float64),
            jnp.asarray(params.chamber_distance, dtype=jnp.float64),
            jnp.asarray(params.chamber_angle_offset, dtype=jnp.float64),
        )
        for name, value in zip(
            (
                "chamber_inner_radius",
                "chamber_outer_radius",
                "chamber_distance",
                "chamber_angle_offset",
            ),
            chamber_arrays,
        ):
            if value.shape != (self.num_segments,):
                raise ValueError(
                    f"{name} must have shape ({self.num_segments},), got {value.shape}."
                )
        updated_self = self._with_pcs_params(params)
        return eqx.tree_at(
            lambda model: (
                model.r_chamber_in,
                model.r_chamber_out,
                model.d_chamber,
                model.varphi_chamber_off,
            ),
            updated_self,
            chamber_arrays,
        )

    def update_params(self, **updates: Array) -> "ISupport":
        """Return an updated copy with selected typed parameter fields replaced."""
        return self.with_params(self.params.replace(**updates))

    @eqx.filter_jit
    def _local_actuator_polar_angles(self, i: Array) -> Array:
        """
        Compute the polar angles of the chambers in the i-th segment.

        Args:
            i (Array): index of the segment as array of shape ()

        Returns:
            varphi_chambers_i (Array): polar angles of the i-th pneumatic chamber as Array of shape (num_chambers_per_segment, )
        """
        varphi_chambers_i = self.varphi_chamber_off[i] + jnp.linspace(
            0, 2 * jnp.pi, self.num_chambers_per_segment, endpoint=False
        )
        return varphi_chambers_i

    @eqx.filter_jit
    def _local_actuator_centroid(self, i: Array, varphi_angle: Array) -> Array:
        """
        Compute the position of the centroid of one pneumatic actuator in the local reference frame of the i-th segment.

        Args:
            i (Array): index of the segment as array of shape ()
            varphi_angle (Array): polar angle of the actuator center from the local z-axis as Array of shape ()

        Returns:
            centroid_actuator (Array): position of the centroid of one pneumatic actuator in the local reference frame of the i-th segment as array of shape (3, )
        """
        centroid_actuator = jnp.array(
            [
                0.0,  # the actuator centroid is on the cross-section
                self.d_chamber[i] * jnp.sin(varphi_angle),  # up along local y-axis
                self.d_chamber[i] * jnp.cos(varphi_angle),  # right along local z-axis
            ]
        )
        return centroid_actuator

    @eqx.filter_jit
    def _local_actuator_cross_sectional_area(self, i: Array) -> Array:
        """
        Compute the local cross-sectional area of one actuator for the i-th segment.
        This should not be confused with the cross-sectional area of the pneumatic chambers that are located within the actuator and used to define the actuation matrix.
        Instead, it represents the cross-sectional area of the actuator itself that is considered for inertial and stiffness calculations.

        Args:
            i (Array): index of the segment as array of shape ()

        Returns:
            A_one_actuator_i (Array): local cross-sectional area of one actuator of the i-th segment
        """
        A_one_actuator_i = jnp.pi * (
            self.r_chamber_out[i] ** 2 - self.r_chamber_in[i] ** 2
        )

        return A_one_actuator_i

    @eqx.filter_jit
    def _local_cross_sectional_area(self, i: Array) -> Array:
        """
        Compute the local cross-sectional area for the i-th segment.

        Args:
            i (Array): index of the segment as array of shape ()

        Returns:
            A_i (Array): local cross-sectional area of the i-th segment as array of shape ()
        """
        A_one_actuator_i = self._local_actuator_cross_sectional_area(i)
        # we assume that the cross-section only consists of the actuators
        A_i = self.num_chambers_per_segment * A_one_actuator_i

        return A_i

    def _local_actuator_second_moment_of_area(
        self, i: Array, varphi_chamber: Array
    ) -> Array:
        """
        Compute the local second moment of area of one actuator for the i-th segment.
        This should not be confused with the second moment of area of the pneumatic chambers that are located within the actuator and used to define the actuation matrix.
        Instead, it represents the second moment of area of the actuator itself that is considered for inertial and stiffness calculations.

        Args:
            i (Array): index of the segment as array of shape ()
            varphi_chamber (Array): polar angle of the chamber center from the local z-axis as Array of shape ()

        Returns:
            I_one_actuator_i (Array): local second moment of area of one actuator of the i-th segment as array of shape (3, )
        """
        # second moment of area of one pneumatic chamber around its own centroid
        I0 = jnp.array(
            [
                jnp.pi
                * (self.r_chamber_out[i] ** 4 - self.r_chamber_in[i] ** 4)
                / 2,  # twist strain
                jnp.pi
                * (self.r_chamber_out[i] ** 4 - self.r_chamber_in[i] ** 4)
                / 4,  # bending strain around local y-axis
                jnp.pi
                * (self.r_chamber_out[i] ** 4 - self.r_chamber_in[i] ** 4)
                / 4,  # bending strain around local z-axis
            ]
        )

        # position of centroid of actuator in local reference frame
        centroid_actuator = self._local_actuator_centroid(i, varphi_chamber)
        # area of the the actuator
        A = self._local_actuator_cross_sectional_area(i)

        # apply the parallel axis theorem
        I_one_actuator_i = I0 + A * (
            jnp.linalg.norm(centroid_actuator) ** 2 * jnp.ones((3,))
            - centroid_actuator**2
        )

        return I_one_actuator_i

    @eqx.filter_jit
    def _local_second_moment_of_area(self, i: Array) -> Array:
        """
        Compute the local second moment of area for the i-th segment.

        Args:
            i (Array): index of the segment as array of shape ()

        Returns:
            I_i (Array): local second moment of area of the i-th segment as array of shape (3, )
        """

        # compute the polar angles of the actuators
        varphi_actuators_i = self._local_actuator_polar_angles(i)
        # compute the second moment of area of each actuator
        I_actuators_i = vmap(
            lambda varphi: self._local_actuator_second_moment_of_area(i, varphi)
        )(varphi_actuators_i)

        # the second moment of area is assumed to be the sum across all actuators
        I_i = jnp.sum(I_actuators_i, axis=0)

        return I_i

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Compute the actuation matrix of the robot.
        We assume that each segment contains self.num_chambers_per_segment identical and symmetric pneumatic chambers that are pressurized to p1=u1, p2=u2, p3=u3, etc.
        Furthermore, we consider the following geometrical arrangement:
            - The 1st chamber with pressure p1 is located at an angular offset of self.varphi_chamber_off
            - The 2nd chamber with pressure p2 is located at an angular offset of self.varphi_chamber_off + 1 * 2 * jnp.pi / self.num_chambers_per_segment
            - The 3rd chamber with pressure p3 is located at an angular offset of self.varphi_chamber_off + 2 * 2 * jnp.pi / self.num_chambers_per_segment
            - etc.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            A (Array): Actuation matrix of shape (num_active_strains, num_actuators)
        """

        def _actuation_matrix_one_chamber(i: Array, varphi: Array) -> Array:
            # Area of one pneumatic chamber
            A_one_chamber = jnp.pi * self.r_chamber_in[i] ** 2

            # force contribution of the chamber
            force_contrib = jnp.array(
                [A_one_chamber, 0.0, 0.0]
            )  # force along local x-axis

            # compute the centroids of the chambers
            centroid_chamber = self._local_actuator_centroid(i, varphi)

            # compute the contribution of the chamber on the (rotational) backbone torque
            torque_contrib = -jnp.cross(centroid_chamber, force_contrib)

            A_single_chamber = jnp.concatenate(
                [
                    torque_contrib,
                    jnp.array(
                        [
                            A_one_chamber,  # axial strain
                            0.0,  # shear strain local y-dir
                            0.0,  # shear strain local z-dir
                        ]
                    ),
                ]
            )

            return A_single_chamber

        def _actuation_matrix_segment_i(i: Array) -> Array:
            # compute the local polar chamber angles
            varphi_chambers = self._local_actuator_polar_angles(i)

            # build the actuation matrix
            A_columns_i = vmap(lambda varphi: _actuation_matrix_one_chamber(i, varphi))(
                varphi_chambers
            )
            A_segment_i = jnp.stack(
                A_columns_i, axis=-1
            )  # shape (6, num_chambers_per_segment)

            return A_segment_i

        A_blocks_tot = vmap(_actuation_matrix_segment_i)(
            jnp.arange(self.num_segments),
        )

        # assemble the contributions of the actuation of each segment
        A_full = blk_diag(
            A_blocks_tot
        )  # shape (6 * num_segments, num_segments * num_chambers_per_segment)
        A = self.B_xi.T @ A_full  # shape (num_active_strains, num_actuators)

        return A
