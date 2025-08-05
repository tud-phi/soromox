__all__ = ["factory", "stiffness_fn"]
from jax import Array, vmap
import jax.numpy as jnp
from soromox.math_utils import blk_diag
import numpy as onp
from typing import Callable, Dict, Optional, Tuple, Union

from .planar_pcs_sym import factory as planar_pcs_factory


def factory(
    num_segments: int,
    *args,
    segment_actuation_selector: Optional[Array] = None,
    simplified_actuation_mapping: bool = False,
    **kwargs,
):
    """
    Factory function for the pneumatically-actuated planar PCS.
    Args:
        num_segments: number of segments
        segment_actuation_selector: actuation selector for the segments as boolean array of shape (num_segments,)
            True entries signify that the segment is actuated, False entries signify that the segment is passive
        simplified_actuation_mapping: flag to use a simplified actuation mapping (i.e., a constant actuation matrix)
    Returns:
    """
    if segment_actuation_selector is None:
        segment_actuation_selector = jnp.ones(num_segments, dtype=bool)

    # number of input pressures
    actuation_dim = segment_actuation_selector.sum() * 2

    # matrix that maps the (possibly) underactuated actuation space to a full actuation space
    actuation_basis = jnp.zeros((2 * num_segments, actuation_dim))
    actuation_basis_cumsum = jnp.cumsum(segment_actuation_selector)
    for i in range(num_segments):
        j = int(actuation_basis_cumsum[i].item()) - 1
        if segment_actuation_selector[i].item() is True:
            actuation_basis = actuation_basis.at[2 * i, j].set(1.0)
            actuation_basis = actuation_basis.at[2 * i + 1, j + 1].set(1.0)

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
        # all segment bases and tips
        sms = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(params["l"])], axis=0)

        # compute the poses of all segment tips
        chi_sms = vmap(forward_kinematics_fn, in_axes=(None, None, 0))(params, q, sms)

        # compute the Jacobian for all segment tips
        J_sms = vmap(jacobian_fn, in_axes=(None, None, 0))(params, q, sms)

        def compute_actuation_matrix_for_segment(
            r_cham_in: Array,
            r_cham_out: Array,
            varphi_cham: Array,
            chi_pe: Array,
            chi_de: Array,
            J_pe: Array,
            J_de: Array,
        ) -> Array:
            """
            Compute the actuation matrix for a single segment.
            We assume that each segment contains four identical and symmetric pneumatic chambers with pressures
            p1, p2, p3, and p4, where p1 and p3 are the right and left chamber pressures respectively, and
            p2 and p4 are the back and front chamber pressures respectively. The front and back chambers
            do not exert a level arm (i.e., a bending moment) on the segment.
            We map the control inputs u1 and u2 as follows to the pressures:
                p1 = u1 (right chamber)
                p2 = (u1 + u2) / 2
                p3 = u2 (left chamber)
                p4 = (u1 + u2) / 2

            Args:
                r_cham_in: inner radius of each segment chamber
                r_cham_out: outer radius of each segment chamber
                varphi_cham: sector angle of each segment chamber
                chi_pe: pose of the proximal end (i.e., the base) of the segment as array of shape (3,)
                chi_de: pose of the distal end (i.e., the tip) of the segment as array of shape (3,)
                J_pe: Jacobian of the proximal end of the segment as array of shape (3, n_q)
                J_de: Jacobian of the distal end of the segment as array of shape (3, n_q)
            Returns:
                A_sm: actuation matrix of shape (n_xi, 2)
            """
            # orientation of the proximal and distal ends of the segment
            th_pe, th_de = chi_pe[2], chi_de[2]

            # compute the area of each pneumatic chamber (we assume identical chambers within a segment)
            A_cham = 0.5 * varphi_cham * (r_cham_out**2 - r_cham_in**2)
            # compute the center of pressure of the pneumatic chamber
            r_cop = (
                2
                / 3
                * jnp.sinc(0.5 * varphi_cham)
                * (r_cham_out**3 - r_cham_in**3)
                / (r_cham_out**2 - r_cham_in**2)
            )

            if simplified_actuation_mapping:
                A_sm = B_xi.T @ jnp.array(
                    [
                        [A_cham * r_cop, -A_cham * r_cop],
                        [0.0, 0.0],
                        [2 * A_cham, 2 * A_cham],
                    ]
                )
            else:
                # compute the actuation matrix that collects the contributions of the pneumatic chambers in the given segment
                # first we consider the contribution of the distal end
                A_sm_de = J_de.T @ jnp.array(
                    [
                        [-2 * A_cham * jnp.sin(th_de), -2 * A_cham * jnp.sin(th_de)],
                        [2 * A_cham * jnp.cos(th_de), 2 * A_cham * jnp.cos(th_de)],
                        [A_cham * r_cop, -A_cham * r_cop],
                    ]
                )
                # then, we consider the contribution of the proximal end
                A_sm_pe = J_pe.T @ jnp.array(
                    [
                        [2 * A_cham * jnp.sin(th_pe), 2 * A_cham * jnp.sin(th_pe)],
                        [-2 * A_cham * jnp.cos(th_pe), -2 * A_cham * jnp.cos(th_pe)],
                        [-A_cham * r_cop, A_cham * r_cop],
                    ]
                )

                # sum the contributions of the distal and proximal ends
                A_sm = A_sm_de + A_sm_pe

            return A_sm

        A_sms = vmap(compute_actuation_matrix_for_segment)(
            params["r_cham_in"],
            params["r_cham_out"],
            params["varphi_cham"],
            chi_pe=chi_sms[:-1],
            chi_de=chi_sms[1:],
            J_pe=J_sms[:-1],
            J_de=J_sms[1:],
        )
        # we need to sum the contributions of the actuation of each segment
        A = jnp.sum(A_sms, axis=0)

        # apply the actuation_basis
        A = A @ actuation_basis

        return A

    return planar_pcs_factory(
        *args,
        stiffness_fn=stiffness_fn,
        actuation_mapping_fn=actuation_mapping_fn,
        **kwargs,
    )


def _compute_stiffness_matrix_for_segment(
    l: Array,
    r: Array,
    r_cham_in: Array,
    r_cham_out: Array,
    varphi_cham: Array,
    E: Array,
):
    # cross-sectional area [m²] of the material
    A_mat = (
        jnp.pi * r**2 + 2 * r_cham_in**2 * varphi_cham - 2 * r_cham_out**2 * varphi_cham
    )
    # second moment of area [m⁴] of the material
    Ib_mat = (
        jnp.pi * r**4 / 4
        + r_cham_in**4 * varphi_cham / 2
        - r_cham_out**4 * varphi_cham / 2
    )
    # poisson ratio of the material
    nu = 0.0
    # shear modulus
    G = E / (2 * (1 + nu))

    # bending stiffness [Nm²]
    Sbe = Ib_mat * E * l
    # shear stiffness [N]
    Ssh = 4 / 3 * A_mat * G * l
    # axial stiffness [N]
    Sax = A_mat * E * l

    S = jnp.diag(jnp.stack([Sbe, Ssh, Sax], axis=0))

    return S


def stiffness_fn(
    params: Dict[str, Array],
    B_xi: Array,
    formulate_in_strain_space: bool = False,
) -> Array:
    """
    Compute the stiffness matrix of the system.
    Args:
        params: Dictionary of robot parameters
        B_xi: Strain basis matrix
        formulate_in_strain_space: whether to formulate the elastic matrix in the strain space
    Returns:
        S: elastic matrix of shape (n_q, n_q) if formulate_in_strain_space is False or (n_xi, n_xi) otherwise
    """
    # stiffness matrix of shape (num_segments, 3, 3)
    S_sms = vmap(_compute_stiffness_matrix_for_segment)(
        params["l"],
        params["r"],
        params["r_cham_in"],
        params["r_cham_out"],
        params["varphi_cham"],
        params["E"],
    )
    # we define the elastic matrix of shape (n_xi, n_xi) as K(xi) = S @ xi where K is equal to
    S = blk_diag(S_sms)

    if not formulate_in_strain_space:
        S = B_xi.T @ S @ B_xi

    return S
