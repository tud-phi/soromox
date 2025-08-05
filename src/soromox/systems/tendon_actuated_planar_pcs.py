__all__ = ["factory", "stiffness_fn"]
from jax import Array, debug, lax, vmap
import jax.numpy as jnp
from soromox.math_utils import blk_diag
import numpy as onp
from typing import Callable, Dict, Optional, Tuple, Union

from .planar_pcs_sym import factory as planar_pcs_factory


def factory(
    num_segments: int,
    *args,
    segment_actuation_selector: Optional[Array] = None,
    **kwargs,
):
    """
    Factory function for the tendon-driven planar PCS.
    Args:
        num_segments: number of segments
        segment_actuation_selector: actuation selector for the segments as boolean array of shape (num_segments,)
            True entries signify that the segment is actuated, False entries signify that the segment is passive
    Returns:
    """
    if segment_actuation_selector is None:
        segment_actuation_selector = jnp.ones(num_segments, dtype=bool)

    def actuation_mapping_fn(
        forward_kinematics_fn: Callable,
        jacobian_fn: Callable,
        params: Dict[str, Array],
        B_xi: Array,
        xi_ref: Array,
        q: Array,
    ) -> Array:
        """
        Returns the actuation matrix that maps the actuation space to the configuration space.
        Args:
            forward_kinematics_fn: function to compute the forward kinematics
            jacobian_fn: function to compute the Jacobian
            params: dictionary with robot parameters
            B_xi: strain basis matrix
            xi_ref: reference strains as array of shape (n_xi,)
            q: configuration of the robot
        Returns:
            A: actuation matrix of shape (n_xi, n_act) where n_xi is the number of strains and
                n_act is the number of actuators
        """
        # extract the parameters
        l = params["l"]
        # map the configuration to the strains
        xi = xi_ref + B_xi @ q
        # segment indices
        segment_indices = jnp.arange(num_segments)

        def compute_actuation_matrix_for_segment(
            segment_idx,
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

                def compute_A_d_wrt_xi_i(i: Array, l_i: Array, xi_i: Array) -> Array:
                    """
                    Compute the actuation matrix for a single actuator with respect to the strains of a single segment.
                    Args:
                        i: index of the segment
                        l_i: length of the segment
                        xi_i: strains for the segment
                    Returns:
                        A_d_segment: actuation matrix for the segment of shape (3, 3)
                    """
                    square_root_term = jnp.sqrt(xi[1] ** 2 + (xi[2] + d * xi[0]) ** 2)
                    A_d_wrt_xi_i = -jnp.array(
                        [
                            l_i * d * (d * xi_i[0] + xi_i[2]) / square_root_term,
                            l_i * xi_i[1] / square_root_term,
                            l_i * (d * xi_i[0] + xi_i[2]) / square_root_term,
                        ]
                    )
                    return jnp.where(
                        i * jnp.ones((3,)) <= segment_idx * jnp.ones((3,)),
                        A_d_wrt_xi_i,
                        jnp.zeros_like(A_d_wrt_xi_i),
                    )

                A_d = vmap(compute_A_d_wrt_xi_i)(
                    segment_indices, l, xi.reshape(-1, 3)
                ).reshape(-1)

                return A_d

            A_sm = vmap(compute_A_d, in_axes=0, out_axes=-1)(d_sm)

            return A_sm

        # compute the actuation matrix for all segments
        # will have shape (num_segments, n_xi, num_segment_tendons)
        A = vmap(compute_actuation_matrix_for_segment, in_axes=(0, 0), out_axes=0)(
            segment_indices,
            params["d"],
        )

        # deactivate the actuation for some segments
        A = A[segment_actuation_selector]

        # reshape the actuation matrix to have shape (n_xi, n_act)
        A = A.transpose((1, 0, 2)).reshape(xi.shape[0], -1)

        return A

    return planar_pcs_factory(
        *args, actuation_mapping_fn=actuation_mapping_fn, **kwargs
    )
