__all__ = ["LinkRuntimeData", "SegmentRuntimeData"]

from dataclasses import dataclass

import jax
from jax import Array


@jax.tree_util.register_pytree_node_class
@dataclass
class SegmentRuntimeData:
    """Discretized per-segment data assembled from a public GVS segment spec."""

    L: Array
    num_integration_points: Array
    dofs_joint_link: Array
    strain_selector: Array
    integration_points: Array
    integration_weights: Array
    mass_matrices: Array
    stiffness_matrices: Array
    damping_matrices: Array
    B_joint: Array
    B_Xs: Array
    B_Z1: Array
    B_Z2: Array
    xi_ref_joint: Array
    xi_ref_Xs: Array
    xi_ref_Z1: Array
    xi_ref_Z2: Array
    joint_stiffness: Array

    def tree_flatten(self):
        children = (
            self.L,
            self.num_integration_points,
            self.dofs_joint_link,
            self.strain_selector,
            self.integration_points,
            self.integration_weights,
            self.mass_matrices,
            self.stiffness_matrices,
            self.damping_matrices,
            self.B_joint,
            self.B_Xs,
            self.B_Z1,
            self.B_Z2,
            self.xi_ref_joint,
            self.xi_ref_Xs,
            self.xi_ref_Z1,
            self.xi_ref_Z2,
            self.joint_stiffness,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass
class LinkRuntimeData:
    """Discretized per-link runtime data for link-only computations."""

    L: Array
    num_integration_points: Array
    strain_selector: Array
    integration_points: Array
    integration_weights: Array
    mass_matrices: Array
    stiffness_matrices: Array
    damping_matrices: Array
    B_Xs: Array
    B_Z1: Array
    B_Z2: Array
    xi_ref_Xs: Array
    xi_ref_Z1: Array
    xi_ref_Z2: Array
    dof_link: Array

    def tree_flatten(self):
        children = (
            self.L,
            self.num_integration_points,
            self.strain_selector,
            self.integration_points,
            self.integration_weights,
            self.mass_matrices,
            self.stiffness_matrices,
            self.damping_matrices,
            self.B_Xs,
            self.B_Z1,
            self.B_Z2,
            self.xi_ref_Xs,
            self.xi_ref_Z1,
            self.xi_ref_Z2,
            self.dof_link,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
