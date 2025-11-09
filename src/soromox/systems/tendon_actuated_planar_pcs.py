__all__ = ["TendonActuatedPlanarPCS"]
from jax import Array, vmap
import jax.numpy as jnp
from typing import Dict, Optional

import equinox as eqx

from soromox.systems.planar_pcs import PlanarPCS


class TendonActuatedPlanarPCS(PlanarPCS):
    """
    Tendon-driven Planar Piecewise Constant Strain (PCS) model for 2D soft continuum robots.

    This class implements the geometric and dynamic modeling of a 2D soft robot
    using the Cosserat rod theory and piecewise constant strain assumption.
    It supports computation of forward kinematics, Jacobians, dynamical matrices.

    Attributes:
    ----------
    num_segments : int
        Number of segments (constant strain sections) along the robot.
    num_actuators : int
        Number of actuators (control inputs) for the robot (2 per actuated segment in the case of planar tendon-driven robots).
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
        Basis matrix for projecting active strains (6 * num_segments, num_active_strains).
    xi_ref : Array
        Reference strain (reference configuration) of the robot.
    num_gauss_points : int
        Number of points used for numerical integration.
        Corresponds to the order of Gauss-Legendre quadrature + 2 (for the endpoints).
    Xs, Ws : Array
        Gauss-Legendre quadrature nodes and weights for numerical integration.
    d: Array
        Distances of the tendons from the segment's backbone.
    segment_indices_to_actuate : Array
        Indices of the segments that are actuated.

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

    d: Array  # distance of the tendons from the segment's backbone, shape (num_segments,)
    segment_indices_to_actuate: Array  # indices of the segments that are actuated, shape (num_actuated_segments,)

    def __init__(
        self,
        num_segments: int,
        params: Dict[str, Array],
        *args,
        segment_actuation_selector: Optional[Array] = None,
        **kwargs,
    ):
        """
        Initialize the TendonActuatedPlanarPCS class
        Args:
            num_segments (int): number of segments in the robot
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
                - "d": List/Array of num_segments floats
                    Distance of the tendons from the segment's backbone [m]
            order_gauss (int, optional):
                Order of the Gauss-Legendre quadrature for integration over each segment.
                Defaults to 5.
            strain_selector (Optional[Array], optional):
                Boolean array of shape (3 * num_segments,) specifying which strain components are active.
                Defaults to all strains active (i.e. all True).
            xi_ref (Optional[Array], optional):
                Reference strain of shape (3 * num_segments,).
                Defaults to 0.0 for bending and shear strains, and 1.0 for axial strain (along local x-axis).
            segment_actuation_selector (Optional[Array]): array to select the segments to be actuated
        """
        super().__init__(num_segments, params, *args, **kwargs)

        if segment_actuation_selector is None:
            segment_actuation_selector = jnp.ones(num_segments, dtype=bool)

        self.segment_indices_to_actuate = jnp.array(
            [i for i, act in enumerate(segment_actuation_selector) if act]
        )

        self.num_actuators = (
            int(jnp.sum(segment_actuation_selector)) * 2
        )  # each segment has two tendons

        self._set_params(params)

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
                - "d": List/Array of num_segments floats
                    Distance of the tendons from the segment's backbone [m]
        """
        super()._set_params(params)

        # Distance of the tendons from the segment's backbone
        try:
            d = params["d"]
        except KeyError:
            raise KeyError(
                "The parameter 'd' (distance of the tendons from the segment's backbone) is required for the tendon-driven planar PCS."
            )
        if not isinstance(d, (list, jnp.ndarray)):
            raise TypeError("The parameter 'd' must be a list or a jnp.ndarray.")
        if len(d) != self.num_segments:
            raise ValueError(
                f"The parameter 'd' must have the same length as the number of segments ({self.num_segments})."
            )
        self.d = jnp.asarray(d, dtype=jnp.float64)

    def update_params(self, params: Dict[str, Array]) -> "TendonActuatedPlanarPCS":
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

        if "d" in params:
            d = params["d"]
            if not isinstance(d, (list, jnp.ndarray)):
                raise TypeError("The parameter 'd' must be a list or a jnp.ndarray.")
            if len(d) != self.num_segments:
                raise ValueError(
                    f"The parameter 'd' must have the same length as the number of segments ({self.num_segments})."
                )
            updated_self = eqx.tree_at(
                lambda x: x.d, updated_self, jnp.asarray(d, dtype=jnp.float64)
            )

        return updated_self

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Compute the actuation matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            A (Array): Actuation matrix of shape (num_active_strains, num_actuators)
        """
        xi = self.strain(q)

        segment_indices = jnp.arange(self.num_segments)

        def compute_actuation_matrix_for_segment(
            segment_idx: int,
            d_sm: Array,
        ) -> Array:
            """
            Compute the actuation matrix for a single segment.
            We assume that each segment is actuated by num_segment_tendons that are routed at a distance of d from the segment's backbone,
            respectively, and attached to the segment's distal end. We assume that the motor is located at the base of the robot and that the
            tendons are routed through all proximal segments.
            The positive control inputs u1 and u2 are the tensions (i.e., forces) applied by the two tendons.
            At a straight configuration with a positive d1, a positive u1 and zero u2 should cause the bend negatively (to the right) and contract its length.

            Args:
                segment_idx: index of the segment
                d_sm: distance of the tendons from the segment's backbone (shape: (num_segment_tendons,))
            Returns:
                A_sm: actuation matrix of shape (n_xi, num_segment_tendons)
            """

            def compute_A_d(d: Array) -> Array:
                """
                Compute the actuation matrix for a single actuator/tendon with respect to the soft robot's strains.
                Args:
                    d: distance of the tendon from the centerline
                Returns:
                    A_d: actuation matrix of shape (n_xi, ) where n_xi is the number of strains
                """
                kappa_0 = xi[0]  # bending strain
                axial_0 = xi[1]  # axial strain
                shear_0 = xi[2]  # shear strain
                square_root_term = jnp.sqrt(shear_0**2 + (axial_0 + d * kappa_0) ** 2)

                def compute_A_d_wrt_xi_i(i: Array, L_i: Array, xi_i: Array) -> Array:
                    """
                    Compute the actuation matrix for a single actuator with respect to the strains of a single segment.
                    Args:
                        i: index of the segment
                        L_i: length of the segment
                        xi_i: strains for the segment
                    Returns:
                        A_d_segment: actuation matrix for the segment of shape (3, 3)
                    """
                    kappa_i = xi_i[0]  # bending strain
                    axial_i = xi_i[1]  # axial strain
                    shear_i = xi_i[2]  # shear strain

                    A_d_wrt_xi_i = -jnp.array(
                        [
                            L_i
                            * d
                            * (d * kappa_i + axial_i)
                            / square_root_term,  # actuation on the bending
                            L_i
                            * (d * kappa_i + axial_i)
                            / square_root_term,  # actuation on the axial strain
                            L_i
                            * shear_i
                            / square_root_term,  # actuation on the shear strain
                        ]
                    )

                    A_d_segment = jnp.where(
                        i * jnp.ones((3,)) <= segment_idx * jnp.ones((3,)),
                        A_d_wrt_xi_i,
                        jnp.zeros_like(A_d_wrt_xi_i),
                    )

                    return A_d_segment

                A_d = vmap(compute_A_d_wrt_xi_i)(
                    segment_indices, self.L, xi.reshape(-1, 3)
                ).reshape(-1)

                return A_d

            A_sm = vmap(compute_A_d, in_axes=0, out_axes=-1)(d_sm)

            return A_sm

        # compute the actuation matrix for all segments

        # (num_segments, n_xi, num_segment_tendons)
        A = vmap(compute_actuation_matrix_for_segment, in_axes=(0, 0), out_axes=0)(
            segment_indices, self.d
        )

        # deactivate the actuation for some segments
        # (num_actuated_segments, n_xi, num_segment_tendons)
        A = A[self.segment_indices_to_actuate]

        # reshape the actuation matrix to have shape (n_xi, n_act)
        A = jnp.concatenate(A, axis=1)  # concatenate along the second axis

        return A
