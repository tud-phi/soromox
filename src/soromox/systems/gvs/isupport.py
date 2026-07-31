from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax import Array, lax, vmap

from soromox.systems.gvs.params import GVSParams
from soromox.systems.gvs.structures import GVSStructure

from .core import GVS

__all__ = ["ISupportGVSParams", "ISupportGVS"]


class ISupportGVSParams(GVSParams):
    """Dynamic parameters for the I-SUPPORT spatial pneumatic GVS model.

    The GVS body parameters (per-link arrays, reference strain, joint stiffness)
    are inherited from :class:`GVSParams`. The additional chamber fields describe
    the per-segment pneumatic chamber geometry and the angular offset of the
    first chamber, each with leading shape ``(num_segments,)``.

    The chamber geometry is used only to build the actuation matrix; the inertia,
    stiffness, and damping matrices come from the standard GVS cross-section
    family carried by ``GVSLinkParams`` (``radius_*``, ``height_*``, etc.).
    """

    chamber_inner_radius: Array
    chamber_outer_radius: Array
    chamber_distance: Array
    chamber_angle_offset: Array

    def _validate_chamber_shapes(self, n_segments: int) -> None:
        for name in (
            "chamber_inner_radius",
            "chamber_outer_radius",
            "chamber_distance",
            "chamber_angle_offset",
        ):
            value = jnp.asarray(getattr(self, name))
            if value.shape != (n_segments,):
                raise ValueError(
                    f"{name} must have shape ({n_segments},), got {value.shape}."
                )

    def validate(self) -> None:
        super().validate()
        self._validate_chamber_shapes(self.link.length.shape[0])

    def validate_against_structure(self, structure) -> None:
        super().validate_against_structure(structure)
        self._validate_chamber_shapes(len(structure.segments))


class ISupportGVS(GVS):
    """
    Geometric Variable Strain (GVS) model for the (AM) I-Support pneumatic
    soft robot.

    This adapts the constant-strain (PCS) I-Support model to the GVS framework,
    in which the strain field within each segment is expanded on arbitrary basis
    functions and integrated over Gauss points along the segment.

    Only the pneumatic actuation is I-Support-specific here. Each segment carries
    ``num_chambers_per_segment`` symmetric chambers; a unit pressure in a chamber
    produces an axial force over the chamber's inner area ``pi * r_in**2`` and a
    backbone torque from the chamber's radial offset. The resulting actuation
    wrench is constant along the segment, so it is projected onto the link strain
    basis and integrated over the segment's Gauss points. The actuation columns
    are block-diagonal across segments, since each chamber pressure only acts on
    its own segment.

    The inertia, stiffness, and damping matrices are produced entirely by the
    standard GVS pipeline from the declared cross-section family
    (circular/rectangular/elliptical) in ``GVSLinkParams``; the chamber geometry
    does not modify them.

    Attributes (in addition to those inherited from GVS):
        r_chamber_in: Inner radius of each segment's chamber, shape (num_segments,).
        r_chamber_out: Outer radius of each segment's chamber, shape (num_segments,).
        d_chamber: Radial distance of the chamber center from the backbone
            centerline, shape (num_segments,).
        varphi_chamber_off: Angular offset of the first chamber from the local
            frame, shape (num_segments,).
        num_chambers_per_segment: Number of pneumatic chambers per segment.

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

    params: ISupportGVSParams

    r_chamber_in: Array
    r_chamber_out: Array
    d_chamber: Array
    varphi_chamber_off: Array

    num_chambers_per_segment: int = eqx.field(static=True, default=3)

    def __init__(
        self,
        params: ISupportGVSParams,
        structure: GVSStructure,
        num_chambers_per_segment: int = 3,
        **kwargs: Any,
    ):
        """
        Initialize the ISupportGVS class.

        Args:
            params: Typed dynamic I-SUPPORT GVS parameters. Inherits the GVS body
                parameters and adds per-segment pneumatic chamber geometry.
            structure: Static GVS segment structure and padded layout.
            num_chambers_per_segment: Number of pneumatic chambers per segment.
                Defaults to 3.
            **kwargs: Additional keyword arguments forwarded to ``GVS``.
        """
        if not isinstance(params, ISupportGVSParams):
            raise TypeError("params must be an ISupportGVSParams instance.")
        params.validate_against_structure(structure)
        # GVS.__init__ validates params against the structure and builds the
        # runtime arrays (including mass/stiffness/damping from the standard
        # cross-section family) and the precomputed caches.
        super().__init__(params, structure=structure, **kwargs)
        # Re-store the more specific typed params object.
        object.__setattr__(self, "params", params)

        object.__setattr__(self, "num_chambers_per_segment", num_chambers_per_segment)
        object.__setattr__(
            self, "num_actuators", self.num_segments * num_chambers_per_segment
        )

        self._set_chamber_params(params)

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------
    def _set_chamber_params(self, params: ISupportGVSParams) -> None:
        """Validate and store the per-segment pneumatic chamber parameters."""
        sources = {
            "r_chamber_in": ("chamber_inner_radius", params.chamber_inner_radius),
            "r_chamber_out": ("chamber_outer_radius", params.chamber_outer_radius),
            "d_chamber": ("chamber_distance", params.chamber_distance),
            "varphi_chamber_off": (
                "chamber_angle_offset",
                params.chamber_angle_offset,
            ),
        }
        for attr_name, (src_name, raw) in sources.items():
            value = jnp.asarray(raw, dtype=jnp.float64)
            if value.shape != (self.num_segments,):
                raise ValueError(
                    f"{src_name} must have shape "
                    f"({self.num_segments},), got {value.shape}."
                )
            object.__setattr__(self, attr_name, value)

    def with_params(self, params: ISupportGVSParams) -> "ISupportGVS":
        """Return an updated copy with a full typed parameter object.

        The GVS body geometry (including mass/stiffness/damping from the standard
        cross-section family) is rebuilt by ``GVS.with_params``; only the chamber
        arrays and the typed params object are swapped in afterwards.
        """
        if not isinstance(params, ISupportGVSParams):
            raise TypeError("params must be an ISupportGVSParams instance.")
        params.validate_against_structure(self.structure)

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

        updated_self = GVS.with_params(self, params)
        return eqx.tree_at(
            lambda model: (
                model.params,
                model.r_chamber_in,
                model.r_chamber_out,
                model.d_chamber,
                model.varphi_chamber_off,
            ),
            updated_self,
            (params, *chamber_arrays),
        )

    def update_params(self, **updates: Array) -> "ISupportGVS":
        """Return an updated copy with selected typed parameter fields replaced."""
        return self.with_params(self.params.replace(**updates))

    # ------------------------------------------------------------------
    # Pneumatic actuation
    # ------------------------------------------------------------------
    @eqx.filter_jit
    def _local_chamber_polar_angles(self, i: Array) -> Array:
        """Polar angles of the chambers in segment ``i``.

        Args:
            i (Array): segment index as a scalar array.

        Returns:
            Array of shape (num_chambers_per_segment,).
        """
        return self.varphi_chamber_off[i] + jnp.linspace(
            0.0, 2 * jnp.pi, self.num_chambers_per_segment, endpoint=False
        )

    @eqx.filter_jit
    def _local_chamber_centroid(self, i: Array, varphi: Array) -> Array:
        """Position of one chamber centroid in the local frame of segment ``i``.

        The centroid lies on the cross-section (local x = 0), offset radially in
        the local y/z plane, matching the PCS convention.

        Returns:
            Array of shape (3,).
        """
        return jnp.array(
            [
                0.0,
                self.d_chamber[i] * jnp.sin(varphi),
                self.d_chamber[i] * jnp.cos(varphi),
            ]
        )

    @eqx.filter_jit
    def _actuation_wrench_one_chamber(self, i: Array, varphi: Array) -> Array:
        """Local actuation wrench (6,) for unit pressure in one chamber.

        A unit pressure acts over the inner chamber area ``pi * r_in**2``,
        producing an axial force along the local x-axis and a backbone torque
        from the chamber offset. The wrench is laid out in the GVS strain
        ordering ``[kappa_x, kappa_y, kappa_z, sigma_x, sigma_y, sigma_z]``.
        """
        A_inner = jnp.pi * self.r_chamber_in[i] ** 2
        force = jnp.array([A_inner, 0.0, 0.0])  # axial force, local x
        centroid = self._local_chamber_centroid(i, varphi)
        torque = -jnp.cross(centroid, force)
        return jnp.concatenate([torque, jnp.array([A_inner, 0.0, 0.0])])

    @eqx.filter_jit
    def _local_actuation_basis_segment(self, i: Array) -> Array:
        """Constant actuation basis for all chambers of segment ``i``.

        Returns:
            Array of shape (6, num_chambers_per_segment). Independent of the
            strain state ``q`` and of the arclength, since the chamber wrench is
            constant for the I-Support pneumatic model.
        """
        varphis = self._local_chamber_polar_angles(i)
        cols = vmap(lambda varphi: self._actuation_wrench_one_chamber(i, varphi))(
            varphis
        )  # (num_chambers, 6)
        return jnp.stack(cols, axis=-1)  # (6, num_chambers)

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Compute the actuation matrix of the robot.

        Each segment has ``num_chambers_per_segment`` symmetric pneumatic
        chambers, pressurized to ``p1=u1, p2=u2, ...``. The angular layout
        matches the PCS model: the first chamber sits at ``varphi_chamber_off``
        and each subsequent chamber is offset by ``2*pi /
        num_chambers_per_segment``.

        Because each chamber wrench is constant along its segment and
        independent of the strain state, the actuation matrix projects that
        (state-independent) wrench onto the link strain basis ``B_Xs`` and
        integrates over the segment's interior Gauss points. Each segment's
        chambers only drive that segment, so the padded actuation matrix is
        block-diagonal across segments before projection onto active strains.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,). Unused,
                retained for interface compatibility.

        Returns:
            A (Array): Actuation matrix of shape (num_dofs, num_actuators).
        """
        del q  # actuation wrench is state-independent for this model
        num_chambers = self.num_chambers_per_segment

        def A_segment_i(i: Array) -> Array:
            """Padded (joint, link) actuation block for segment ``i``.

            Returns:
                Array of shape (2, max_dof, num_chambers), stacked as
                [joint_block, link_block].
            """
            length_i = self.segment_lengths[i]
            Ws_i = self.integration_weights[i]  # (max_num_integration_points,)
            B_Xs_i = self.B_Xs[i]  # (max_num_integration_points, 6, max_dof)
            Phi_a_i = self._local_actuation_basis_segment(i)  # (6, num_chambers)

            A_joint_i = jnp.zeros((self.max_dof, num_chambers))

            def A_point_j(j: Array) -> Array:
                B_Xs_j = B_Xs_i[j]
                if self.scale_rotational_basis_by_length:
                    B_Xs_j = B_Xs_j.at[:3, :].divide(length_i)
                return Ws_i[j] * (B_Xs_j.T @ Phi_a_i)  # (max_dof, num_chambers)

            # interior (nonzero-weight) Gauss points only
            A_link_i = (
                jnp.sum(
                    vmap(A_point_j)(
                        jnp.arange(1, self.max_num_integration_points - 1)
                    ),
                    axis=0,
                )
                * length_i
            )
            return jnp.stack([A_joint_i, A_link_i], axis=0)

        # (num_segments, 2, max_dof, num_chambers)
        A_blocks = vmap(A_segment_i)(jnp.arange(self.num_segments))

        # Place each segment's chamber columns into its own actuator slots,
        # producing a block-diagonal padded actuation matrix.
        block_size = 2 * self.max_dof

        def place_segment(carry: Array, i: Array) -> tuple[Array, None]:
            A_seg = A_blocks[i].reshape(block_size, num_chambers)
            row0 = i * block_size
            col0 = i * num_chambers
            carry = lax.dynamic_update_slice(carry, A_seg, (row0, col0))
            return carry, None

        A_full = jnp.zeros((self.num_padded_dofs, self.num_actuators))
        A_full, _ = lax.scan(place_segment, A_full, jnp.arange(self.num_segments))

        # Project onto active strains.
        A = self.active_dof_map.T @ A_full  # (num_dofs, num_actuators)
        return A

    @eqx.filter_jit
    def actuation_force(self, q: Array, u: Array) -> Array:
        """Generalized actuation force for pressure inputs ``u``.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            u (Array): chamber pressures of shape (num_actuators,).

        Returns:
            tau_u (Array): generalized actuation force of shape (num_dofs,).
        """
        return self.actuation_matrix(q) @ u