__all__ = ["LinkData", "SegmentData"]
from dataclasses import dataclass, field
import jax
from jax import Array


@jax.tree_util.register_pytree_node_class
@dataclass
class SegmentData:
    """
    Discretized per-segment data used by GVS for geometry, integration and dynamics.

    Attributes
    ----------
    L : Array
        Segment length [m], shape ().
    nip : Array
        Number of integration/evaluation points (= N_GQ + 2), shape ().
    dofs_joint_link : Array
        Degrees of freedom of the segment as `[dof_joint, dof_link]`, shape (2,), int.
    strain_selector : Array
        Boolean selector of active generalized strains, padded to shape (2*max_dof,).
        First `max_dof` entries correspond to joint DOFs, last `max_dof` to link DOFs.
    Xs : Array
        Gauss nodes on [0, 1], padded to shape (max_nip,).
    Ws : Array
        Gauss weights on [0, 1], padded to shape (max_nip,).
    Ms : Array
        Distributed mass/inertia diagonal matrices at nodes, shape (max_nip, 6, 6).
        Rotational entries ≈ ρ I_p [kg·m], translational entries ≈ ρ A [kg/m].
    Es : Array
        Distributed elastic stiffness diagonal matrices, shape (max_nip, 6, 6).
        Torsion/bending ≈ G Ix, E Iy, E Iz [N·m^2]; axial/shear ≈ E A, G A [N].
    Gs : Array
        Distributed damping diagonal matrices, shape (max_nip, 6, 6).
        Rotational entries ≈ η I_p [N·s·m^3]; translational ≈ η A [N·s·m].
    B_joint : Array
        Joint basis matrix mapping joint DOFs to 6D twist, shape (6, max_dof).
    B_Xs : Array
        Link basis matrices evaluated at Gauss nodes, shape (max_nip, 6, max_dof).
    B_Z1 : Array
        Link basis matrices at midpoints Z1, padded to (max_nip, 6, max_dof).
    B_Z2 : Array
        Link basis matrices at midpoints Z2, padded to (max_nip, 6, max_dof).
    xi_ref_joint : Array
        Joint reference strain/twist, shape (6,), units [rad/m, rad/m, rad/m, 1/m, 1/m, 1/m].
    xi_ref_Xs : Array
        Link reference strain at Gauss nodes, shape (max_nip, 6), same units as above.
    xi_ref_Z1 : Array
        Link reference strain at Z1 points, shape (max_nip, 6).
    xi_ref_Z2 : Array
        Link reference strain at Z2 points, shape (max_nip, 6).
    K_joint : Array
        Joint stiffness matrix padded to (max_dof, max_dof). Units follow active DOFs:
        rotational terms [N·m/rad], translational terms [N/m]; couplings accordingly.
    """
    L: Array  # Length of the segment
    nip: Array  # Number of integration points
    dofs_joint_link: Array  # Degrees of freedom of the segment as [dof_joint, dof_link]
    strain_selector: (
        Array  # Boolean array indicating which strain components are active
    )
    Xs: Array  # Integration points
    Ws: Array  # Weights for the integration points
    Ms: Array  # Mass matrices at integration points
    Es: Array  # Stiffness matrices at integration points
    Gs: Array  # Damping matrices at integration points
    B_joint: Array  # Joint basis matrix
    B_Xs: Array  # Basis matrix at integration points
    B_Z1: Array  # Basis matrix at Z1 points
    B_Z2: Array  # Basis matrix at Z2 points
    xi_ref_joint: Array  # Joint initial strain vector
    xi_ref_Xs: Array  # Initial strain vector at integration points
    xi_ref_Z1: Array  # Initial strain vector at Z1 points
    xi_ref_Z2: Array  # Initial strain vector at Z2 points
    K_joint: Array  # Joint stiffness matrix

    def tree_flatten(self):
        children = (
            self.L,
            self.nip,
            self.dofs_joint_link,
            self.Xs,
            self.Ws,
            self.Ms,
            self.Es,
            self.Gs,
            self.B_joint,
            self.B_Xs,
            self.B_Z1,
            self.B_Z2,
            self.xi_ref_joint,
            self.xi_ref_Xs,
            self.xi_ref_Z1,
            self.xi_ref_Z2,
            self.K_joint,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass
class LinkData:
    """
    Discretized per-link data (without the joint part), used for link-only operations.

    Attributes
    ----------
    L : Array
        Segment length [m], shape ().
    nip : Array
        Number of integration/evaluation points (= N_GQ + 2), shape ().
    strain_selector : Array
        Boolean selector of active link generalized strains, padded to (max_dof,).
    Xs : Array
        Gauss nodes on [0, 1], padded to shape (max_nip,).
    Ws : Array
        Gauss weights on [0, 1], padded to shape (max_nip,).
    Ms : Array
        Distributed mass/inertia diagonal matrices at nodes, shape (max_nip, 6, 6).
        Rotational entries ≈ ρ I_p [kg·m], translational entries ≈ ρ A [kg/m].
    Es : Array
        Distributed elastic stiffness diagonal matrices, shape (max_nip, 6, 6).
        Torsion/bending ≈ G Ix, E Iy, E Iz [N·m^2]; axial/shear ≈ E A, G A [N].
    Gs : Array
        Distributed damping diagonal matrices, shape (max_nip, 6, 6).
        Rotational entries ≈ η I_p [N·s·m^3]; translational ≈ η A [N·s·m].
    B_Xs : Array
        Link basis matrices evaluated at Gauss nodes, shape (max_nip, 6, max_dof).
    B_Z1 : Array
        Link basis matrices at midpoints Z1, padded to (max_nip, 6, max_dof).
    B_Z2 : Array
        Link basis matrices at midpoints Z2, padded to (max_nip, 6, max_dof).
    xi_ref_Xs : Array
        Link reference strain at Gauss nodes, shape (max_nip, 6), units [rad/m, rad/m, rad/m, 1/m, 1/m, 1/m].
    xi_ref_Z1 : Array
        Link reference strain at Z1 points, shape (max_nip, 6).
    xi_ref_Z2 : Array
        Link reference strain at Z2 points, shape (max_nip, 6).
    dof_link : Array
        Number of active link DOFs, shape (), int.
    """
    L: Array  # Length of the segment
    nip: Array  # Number of integration points
    strain_selector: (
        Array  # Boolean array indicating which strain components are active
    )
    Xs: Array  # Integration points
    Ws: Array  # Weights for the integration points
    Ms: Array  # Mass matrices at integration points
    Es: Array  # Stiffness matrices at integration points
    Gs: Array  # Damping matrices at integration points
    B_Xs: Array  # Basis matrix at integration points
    B_Z1: Array  # Basis matrix at Z1 points
    B_Z2: Array  # Basis matrix at Z2 points
    xi_ref_Xs: Array  # Initial strain vector at integration points
    xi_ref_Z1: Array  # Initial strain vector at Z1 points
    xi_ref_Z2: Array  # Initial strain vector at Z2 points
    dof_link: Array  # Degrees of freedom of the segment

    def tree_flatten(self):
        children = (
            self.L,
            self.nip,
            self.Xs,
            self.Ws,
            self.Ms,
            self.Es,
            self.Gs,
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
