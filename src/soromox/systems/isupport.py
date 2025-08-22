from jax import Array, vmap
import jax.numpy as jnp
import numpy as onp
from typing import Dict, Optional

import equinox as eqx

from .pcs import PCS


class ISupport(PCS):
    """
    A kinematic and dynamic model for the (AM) I-Support robot based on the Piecewise Constant Strain shape parametrization.
    
    Attributes:
    ----------
    TODO

    Notes:
    -----
    TODO

    References:
    -----
    This implementation builds upon the findings of:

    Arleo et al. (2021): Arleo, L., Stano, G., Percoco, G., & Cianchetti, M. (2021). 
        I-support soft arm for assistance tasks: a new manufacturing approach based on 3D printing and characterization. 
        Progress in Additive Manufacturing, 6(2), 243-256.
        https://link.springer.com/article/10.1007/s40964-020-00158-y
    
    Alessi et al. (2023): Alessi, C., Falotico, E., & Lucantonio, A. (2023). 
        Ablation study of a dynamic model for a 3d-printed pneumatic soft robotic arm. IEEE Access, 11, 37840-37853.
        https://ieeexplore.ieee.org/abstract/document/10098800

    Alessi, C., Bianchi, D., Stano, G., Cianchetti, M., & Falotico, E. (2024). 
        Pushing with soft robotic arms via deep reinforcement learning. Advanced Intelligent Systems, 6(8), 2300899.
        https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aisy.202300899

    """
    num_actuators: int = eqx.field(static=True)  # number of actuators (control inputs) for the robot (2 per actuated segment in the case of planar pneumatically-actuated PCS)
    num_chambers_per_segment: int = eqx.field(static=True, default=3)  # number of pneumatic chambers per segment
    actuation_basis: Array  # actuation basis, shape (num_segments * 2, num_actuators)

    r_chamber_in: Array  # inner radius of each segment's chamber, shape (num_segments,)
    r_chamber_out: (
        Array  # outer radius of each segment's chamber, shape (num_segments,)
    )

    def __init__(
        self,
        num_segments: int,
        params: Dict[str, Array],
        *args,
        segment_actuation_selector: Optional[Array] = None,
        num_chambers_per_segment: int = 3,
        **kwargs,
    ):
        """
        Initialize the ISupport class
        Args:
            
        """
        super().__init__(num_segments, params, *args, **kwargs)

        if segment_actuation_selector is None:
            segment_actuation_selector = jnp.ones(self.num_segments, dtype=bool)
        self.num_chambers_per_segment = num_chambers_per_segment

        self.num_actuators = (
            int(jnp.sum(segment_actuation_selector)) * self.num_chambers_per_segment
        )  # each segment has three control inputs (u1, u2, u3)

        actuation_basis = jnp.zeros((2 * self.num_segments, self.num_actuators))
        actuation_basis_cumsum = jnp.cumsum(segment_actuation_selector)
        for i in range(self.num_segments):
            j = int(actuation_basis_cumsum[i].item()) - 1
            if segment_actuation_selector[i].item() is True:
                actuation_basis = actuation_basis.at[2 * i, j].set(1.0)
                actuation_basis = actuation_basis.at[2 * i + 1, j + 1].set(1.0)
        self.actuation_basis = actuation_basis

        self._set_params(params)

    def _set_params(self, params: Dict[str, Array]):
        """
        Set the parameters of the tendon-driven planar PCS.

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
                    Angular offset of the first chamber from the local z-axis [rad]

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

        try:
            d_chamber = params["d_chamber"]
        except KeyError:
            raise KeyError(
                "The parameter 'd_chamber' (radial distance of the center of the chambers from the centerline of the backbone) is required for the pneumatically actuated planar PCS."
            )
        if not isinstance(d_chamber, (list, jnp.ndarray)):
            raise TypeError(
                "The parameter 'd_chamber' must be a list or a jnp.ndarray."
            )
        if len(d_chamber) != self.num_segments:
            raise ValueError(
                f"The parameter 'd_chamber' must have the same length as the number of segments ({self.num_segments})."
            )
        self.d_chamber = jnp.asarray(d_chamber, dtype=jnp.float64)

        varphi_chamber_off = params.get("varphi_chamber_off", jnp.zeros(self.num_segments))
        if not isinstance(varphi_chamber_off, (list, jnp.ndarray)):
            raise TypeError(
                "The parameter 'varphi_chamber_off' must be a list or a jnp.ndarray."
            )
        if len(varphi_chamber_off) != self.num_segments:
            raise ValueError(
                f"The parameter 'varphi_chamber_off' must have the same length as the number of segments ({self.num_segments})."
            )
        self.varphi_chamber_off = jnp.asarray(varphi_chamber_off, dtype=jnp.float64)

    def update_params(
        self, params: Dict[str, Array]
    ) -> "ISupport":
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
                - "r_chamber_in" : Array of num_segments floats
                    Inner radius of each segment's pneumatic chamber [m]
                - "r_chamber_out" : Array of num_segments floats
                    Outer radius of each segment's pneumatic chamber [m]
                - "d_chamber" : Array of num_segments floats
                    Radial distance of the center of the chambers from the centerline of the backbone [m]
                - "varphi_chamber_off" : Array of num_segments floats
                    Angular offset of the first chamber from the local z-axis [rad]
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

        if "varphi_chamber_off" in params:
            varphi_chamber_off = params["varphi_chamber_off"]
            if not isinstance(varphi_chamber_off, (list, jnp.ndarray)):
                raise TypeError(
                    "The parameter 'varphi_chamber_off' must be a list or a jnp.ndarray."
                )
            if len(varphi_chamber_off) != self.num_segments:
                raise ValueError(
                    f"The parameter 'varphi_chamber_off' must have the same length as the number of segments ({self.num_segments})."
                )
            updated_self = eqx.tree_at(
                lambda x: x.varphi_chamber_off,
                updated_self,
                jnp.asarray(varphi_chamber_off, dtype=jnp.float64),
            )

        return updated_self
    
    @eqx.filter_jit
    def _local_chambers_polar_angles(self, i: int) -> Array:
        """
        Compute the polar angles of the chambers in the i-th segment.

        Args:
            i (int): index of the segment

        Returns:
            varphi_chambers_i (Array): polar angles of the i-th pneumatic chamber as Array of shape (num_chambers_per_segment, )
        """
        varphi_chambers_i = self.varphi_chamber_off[i] + jnp.linspace(0, 2 * jnp.pi, self.num_chambers_per_segment, endpoint=False)
        return varphi_chambers_i

    @eqx.filter_jit
    def _local_chamber_cross_sectional_area(self, i: int) -> Array:
        """
        Compute the local cross-sectional area of one pneumatic chamber for the i-th segment.

        Args:
            i (int): index of the segment

        Returns:
            A_one_chamber_i (Array): local cross-sectional area of one pneumatic chamber of the i-th segment
        """
        A_one_chamber_i = jnp.pi * (self.r_chamber_out[i] ** 2 - self.r_chamber_in[i] ** 2)

        return A_one_chamber_i

    @eqx.filter_jit
    def _local_cross_sectional_area(self, i: int) -> Array:
        """
        Compute the local cross-sectional area for the i-th segment.

        Args:
            i (int): index of the segment

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
    def _local_chamber_second_moment_of_area(self, i: int, varphi_chamber: Array) -> Array:
        """
        Compute the local second moment of area of one pneumatic chamber for the i-th segment.

        Args:
            i (int): index of the segment
            varphi_chamber (Array): polar angle of the chamber center from the local z-axis as Array of shape ()
        Returns:
            I_one_chamber_i (Array): local second moment of area of one pneumatic chamber of the i-th segment as array of shape (3, )
        """
        # second moment of area of one pneumatic chamber around its own centroid
        I0 = jnp.array([
            jnp.pi * (self.r_chamber_out[i] ** 4 - self.r_chamber_in[i] ** 4) / 2,  # twist strain
            jnp.pi * (self.r_chamber_out[i] ** 4 - self.r_chamber_in[i] ** 4) / 4,  # bending strain around local y-axis
            jnp.pi * (self.r_chamber_out[i] ** 4 - self.r_chamber_in[i] ** 4) / 4,  # bending strain around local z-axis
        ])

        # position of centroid of chamber in local reference frame
        centroid_chamber = jnp.array([
            0.0,  # the chamber centroid is on the cross-section
            self.d_chamber[i] * jnp.sin(varphi_chamber),  # up along local y-axis
            self.d_chamber[i] * jnp.cos(varphi_chamber),  # right along local z-axis
        ])
        # area of the the chamber
        A = self._local_chamber_cross_sectional_area(i)

        # apply the parallel axis theorem
        I_one_chamber_i = I0 + A * (jnp.linalg.norm(centroid_chamber) ** 2 * jnp.ones((3, )) - centroid_chamber ** 2)

        return I_one_chamber_i

    @eqx.filter_jit
    def _local_second_moment_of_area(self, i: int) -> Array:
        """
        Compute the local second moment of area for the i-th segment.

        Args:
            i (int): index of the segment

        Returns:
            I_i (Array): local second moment of area of the i-th segment
        """
        # full second moment of area of the i-th segment without chambers
        I_full_i = super()._local_second_moment_of_area(i)

        # compute the polar angles of the chambers
        varphi_chambers_i = self._local_chambers_polar_angles(i)
        # compute the second moment of area of each pneumatic chamber
        I_chambers_i = vmap(lambda varphi: self._local_chamber_second_moment_of_area(i, varphi))(varphi_chambers_i)

        I_i = (
            I_full_i - jnp.sum(I_chambers_i, axis=0)
        )  # Subtract the second moment of area of the four chambers

        return I_i

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Compute the actuation matrix of the robot.
        We assume that each segment contains self.num_chambers_per_segment identical and symmetric pneumatic chambers that are pressurized to p1=u1, p2=u2, p3=u3, etc.
        Furthermore, we consider the following geometrical arrangement:
            - The 1st chamber with pressure 1 is located at an angular offset of self.varphi_chamber_off 

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            A (Array): Actuation matrix of shape (num_active_strains, num_actuators)
        """

        def A_segment_i(i: int) -> Array:
            # Area of one pneumatic chamber
            A_one_chamber = self._local_chamber_cross_sectional_area(i)

            A_full_segment_i = jnp.array(
                [
                    [
                        A_one_chamber * self.d_chamber[i],
                        -A_one_chamber * self.d_chamber[i],
                    ],  # actuation on the bending
                    [
                        A_one_chamber,
                        A_one_chamber,
                    ],  # actuation on the axial strain
                    [0.0, 0.0],  # actuation on the shear strain
                ]
            )

            return A_segment_i

        A_blocks_tot = vmap(A_segment_i)(
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
