__all__ = ["TendonActuatedPlanarPCS"]

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax import Array, vmap

import soromox.actuation.tendon_actuation as act
from soromox.rendering.actuators import ActuatorVisualLayer
from soromox.systems.params import (
    BaseTendonRoutingParams,
    PassiveTendonParams,
)
from soromox.systems.pcs.params import TendonActuatedPlanarPCSParams
from soromox.systems.pcs.structures import PlanarPCSStructure
from soromox.utils.integration import scale_gaussian_quadrature

from .planar_pcs import PlanarPCS


class TendonActuatedPlanarPCS(PlanarPCS):
    """
    Tendon-driven planar Piecewise Constant Strain (PCS) model.

    This model uses the planar strain convention ``[kappa_z, sigma_x, sigma_y]``
    per segment and supports active and passive tendons. Tendons are described by
    the same batched routing parameter objects used by ``TendonActuatedPCS`` and
    ``TendonActuatedGVS``. Each tendon has an ``attachment_segment_index`` that
    determines which proximal segments it spans.

    Routing basis functions may be shared with spatial systems or written in a
    compact planar form. For each tendon and arc-length position ``s``, ``d_s``
    and ``dd_s_ds`` may return:

    - a scalar: interpreted as the planar offset ``d(s)`` or derivative
      ``d'(s)``;
    - shape ``(1,)``: first entry is interpreted as ``d(s)`` or ``d'(s)``;
    - shape ``(3,)``: PCS/GVS-compatible ``[0, y, z]`` routing, where ``y`` is
      the planar offset and ``z`` must be zero.

    Shape ``(2,)`` is intentionally invalid because it is ambiguous whether it
    means ``[x, y]`` or ``[y, z]``. The built-in ``LinearTendonRoutingParams``
    uses the PCS/GVS-compatible ``[0, y, z]`` convention; planar callers should
    set ``z_intercept = z_slope = 0``.
    """

    n_p: int
    params: TendonActuatedPlanarPCSParams
    active_tendon_routing: BaseTendonRoutingParams
    passive_tendon_routing: BaseTendonRoutingParams
    active_d_s: Callable = eqx.field(static=True)
    active_dd_s_ds: Callable = eqx.field(static=True)
    passive_d_s: Callable = eqx.field(static=True)
    passive_dd_s_ds: Callable = eqx.field(static=True)
    K_pt: Array
    D_pt: Array
    l_pt0: Array

    def __init__(
        self,
        params: TendonActuatedPlanarPCSParams,
        structure: PlanarPCSStructure | None = None,
        active_tendon_routing_basis: dict[str, Callable] | None = None,
        passive_tendon_routing_basis: dict[str, Callable] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize a tendon-actuated planar PCS model.

        Args:
            params: Typed dynamic parameters. ``params.body`` stores the planar
                PCS body, and active/passive routing fields store one leading
                entry per tendon.
            structure: Static planar PCS layout. Changing the number of
                segments, active strains, or quadrature layout requires
                reconstruction.
            active_tendon_routing_basis: Routing functions for active tendons.
                Defaults to linear routing. Custom functions may return scalar,
                ``(1,)``, or PCS/GVS-compatible ``(3,)`` values as documented in
                the class docstring.
            passive_tendon_routing_basis: Routing functions for passive tendons.
                The same planar return-shape rules apply.
            **kwargs: Additional keyword arguments for ``PlanarPCS``.
        """
        if not isinstance(params, TendonActuatedPlanarPCSParams):
            raise TypeError("params must be a TendonActuatedPlanarPCSParams instance.")
        params.validate()
        super().__init__(params.body, structure=structure, **kwargs)
        self.params = params

        if active_tendon_routing_basis is None:
            active_tendon_routing_basis = {
                "d_s": act.linear_routing,
                "dd_s_ds": act.linear_routing_arc_length_derivative,
            }
        self.active_d_s, self.active_dd_s_ds = self._set_tendon_routing_basis(
            active_tendon_routing_basis
        )
        self.active_tendon_routing = self._set_active_tendon_routing(
            params.active_tendon_routing
        )

        if passive_tendon_routing_basis is None:
            passive_tendon_routing_basis = {
                "d_s": act.linear_routing,
                "dd_s_ds": act.linear_routing_arc_length_derivative,
            }
        self.passive_d_s, self.passive_dd_s_ds = self._set_tendon_routing_basis(
            passive_tendon_routing_basis
        )
        self.passive_tendon_routing = self._set_passive_tendon_routing(
            params.passive_tendon_routing
        )
        self._set_passive_tendon(params.passive_tendon)

    def _set_tendon_routing_basis(
        self, tendon_routing_basis: dict[str, Callable]
    ) -> tuple[Callable, Callable]:
        """Store tendon routing and arc-length derivative callables."""
        return tendon_routing_basis["d_s"], tendon_routing_basis["dd_s_ds"]

    @staticmethod
    def _planar_component_from_routing_value(value: Array, name: str) -> Array:
        """
        Extract the scalar planar tendon offset from one routing-basis value.

        Accepted shapes are scalar, ``(1,)``, and PCS/GVS-compatible ``(3,)``.
        Shape ``(2,)`` is rejected because its coordinate convention is
        ambiguous.
        """
        value = jnp.asarray(value)
        if value.ndim == 0:
            return value
        if value.shape[-1] == 1:
            return value[..., 0]
        if value.shape[-1] == 3:
            return value[..., 1]
        if value.shape[-1] == 2:
            raise ValueError(
                f"{name} returned shape (..., 2), which is ambiguous for planar "
                "routing. Return a scalar, shape (1,), or PCS/GVS-compatible "
                "shape (3,) with zero z."
            )
        raise ValueError(
            f"{name} must return a scalar, shape (1,), or shape (3,), got "
            f"shape {value.shape}."
        )

    @staticmethod
    def _planar_z_magnitude_from_routing_value(value: Array, name: str) -> Array:
        """Return ``abs(z)`` for PCS/GVS-compatible planar routing samples."""
        value = jnp.asarray(value)
        if value.ndim == 0 or value.shape[-1] == 1:
            return jnp.asarray(0.0, dtype=value.dtype)
        if value.shape[-1] == 3:
            return jnp.abs(value[..., 2])
        if value.shape[-1] == 2:
            raise ValueError(
                f"{name} returned shape (..., 2), which is ambiguous for planar "
                "routing. Return a scalar, shape (1,), or PCS/GVS-compatible "
                "shape (3,) with zero z."
            )
        raise ValueError(
            f"{name} must return a scalar, shape (1,), or shape (3,), got "
            f"shape {value.shape}."
        )

    def _validate_planar_routing_basis(
        self,
        tendon_routing_params: BaseTendonRoutingParams,
        d_s: Callable,
        dd_s_ds: Callable,
        name: str,
    ) -> None:
        """Validate planar routing shape, zero out-of-plane routing, and body fit."""
        if tendon_routing_params.num_tendons == 0:
            return

        sample_s = jnp.linspace(0.0, self.L_cum[-1], 75)

        def radius_at_s(s: Array) -> Array:
            idx = jnp.sum(self.L_cum <= s) - 1
            idx = jnp.clip(idx, 0, self.num_segments - 1)
            return self.r[idx]

        def sample_tendon(
            tendon_routing_params_k: BaseTendonRoutingParams, s: Array
        ) -> tuple[Array, Array, Array]:
            d_value = d_s(tendon_routing_params_k, s)
            dd_value = dd_s_ds(tendon_routing_params_k, s)
            d = self._planar_component_from_routing_value(d_value, f"{name}.d_s")
            z = self._planar_z_magnitude_from_routing_value(d_value, f"{name}.d_s")
            dz = self._planar_z_magnitude_from_routing_value(
                dd_value, f"{name}.dd_s_ds"
            )
            return jnp.abs(d), z, dz

        d_abs, z_abs, dz_abs = vmap(
            vmap(sample_tendon, in_axes=(None, 0), out_axes=0),
            in_axes=(0, None),
            out_axes=0,
        )(tendon_routing_params, sample_s)
        r_body = vmap(radius_at_s)(sample_s)

        if bool(jnp.any(z_abs > 1e-12)) or bool(jnp.any(dz_abs > 1e-12)):
            raise ValueError(
                f"{name} returned nonzero z routing or z derivative, which is "
                "invalid for TendonActuatedPlanarPCS."
            )
        if bool(jnp.any(d_abs > r_body[None, :])):
            raise UserWarning(f"{name} tendon(s) exit the robot body.")

    def _set_active_tendon_routing(
        self, active_tendon_routing: BaseTendonRoutingParams
    ) -> BaseTendonRoutingParams:
        """Store and validate active tendon routing parameters."""
        if not isinstance(active_tendon_routing, BaseTendonRoutingParams):
            raise TypeError(
                "active_tendon_routing must be a BaseTendonRoutingParams instance."
            )
        active_tendon_routing.validate()
        self.num_actuators = active_tendon_routing.num_tendons
        active_tendon_routing.validate_attachment_segments(
            self.num_segments, "active_tendon_routing"
        )
        self._validate_planar_routing_basis(
            active_tendon_routing,
            self.active_d_s,
            self.active_dd_s_ds,
            "active_tendon_routing",
        )
        return active_tendon_routing

    def _set_passive_tendon_routing(
        self, passive_tendon_routing: BaseTendonRoutingParams
    ) -> BaseTendonRoutingParams:
        """Store and validate passive tendon routing parameters."""
        if not isinstance(passive_tendon_routing, BaseTendonRoutingParams):
            raise TypeError(
                "passive_tendon_routing must be a BaseTendonRoutingParams instance."
            )
        passive_tendon_routing.validate()
        self.n_p = passive_tendon_routing.num_tendons
        passive_tendon_routing.validate_attachment_segments(
            self.num_segments, "passive_tendon_routing"
        )
        self._validate_planar_routing_basis(
            passive_tendon_routing,
            self.passive_d_s,
            self.passive_dd_s_ds,
            "passive_tendon_routing",
        )
        return passive_tendon_routing

    def _set_passive_tendon(self, passive_tendon: PassiveTendonParams) -> None:
        """Store per-passive-tendon stiffness, damping, and rest-length offsets."""
        if not isinstance(passive_tendon, PassiveTendonParams):
            raise TypeError("passive_tendon must be a PassiveTendonParams instance.")
        passive_tendon.validate()
        if passive_tendon.num_tendons != self.n_p:
            raise ValueError(
                "passive_tendon length must match passive_tendon_routing length; "
                f"got {passive_tendon.num_tendons} and {self.n_p}."
            )
        self.K_pt = jnp.diag(jnp.asarray(passive_tendon.stiffness))
        self.D_pt = jnp.diag(jnp.asarray(passive_tendon.damping))
        self.l_pt0 = jnp.asarray(passive_tendon.rest_length_offset)

    def with_params(
        self, params: TendonActuatedPlanarPCSParams
    ) -> "TendonActuatedPlanarPCS":
        """Return an updated copy with a full typed parameter object."""
        if not isinstance(params, TendonActuatedPlanarPCSParams):
            raise TypeError("params must be a TendonActuatedPlanarPCSParams instance.")
        params.validate()
        if type(params.active_tendon_routing) is not type(
            self.params.active_tendon_routing
        ):
            raise ValueError(
                "Changing active_tendon_routing type requires reconstruction."
            )
        if type(params.passive_tendon_routing) is not type(
            self.params.passive_tendon_routing
        ):
            raise ValueError(
                "Changing passive_tendon_routing type requires reconstruction."
            )
        if (
            params.active_tendon_routing.num_tendons
            != self.params.active_tendon_routing.num_tendons
        ):
            raise ValueError(
                "Changing the number of active tendons requires reconstruction."
            )
        if (
            params.passive_tendon_routing.num_tendons
            != self.params.passive_tendon_routing.num_tendons
        ):
            raise ValueError(
                "Changing the number of passive tendons requires reconstruction."
            )
        self.params.active_tendon_routing.assert_same_attachment_segments(
            params.active_tendon_routing, "active_tendon_routing"
        )
        self.params.passive_tendon_routing.assert_same_attachment_segments(
            params.passive_tendon_routing, "passive_tendon_routing"
        )

        updated_self = self._with_planar_pcs_params(params.body, stored_params=params)
        updated_self._validate_planar_routing_basis(
            params.active_tendon_routing,
            self.active_d_s,
            self.active_dd_s_ds,
            "active_tendon_routing",
        )
        updated_self._validate_planar_routing_basis(
            params.passive_tendon_routing,
            self.passive_d_s,
            self.passive_dd_s_ds,
            "passive_tendon_routing",
        )
        return eqx.tree_at(
            lambda model: (
                model.params,
                model.active_tendon_routing,
                model.passive_tendon_routing,
                model.K_pt,
                model.D_pt,
                model.l_pt0,
            ),
            updated_self,
            (
                params,
                params.active_tendon_routing,
                params.passive_tendon_routing,
                jnp.diag(jnp.asarray(params.passive_tendon.stiffness)),
                jnp.diag(jnp.asarray(params.passive_tendon.damping)),
                jnp.asarray(params.passive_tendon.rest_length_offset),
            ),
        )

    def update_params(self, **updates: Any) -> "TendonActuatedPlanarPCS":
        """Return an updated copy with selected typed parameter fields replaced."""
        return self.with_params(self.params.replace(**updates))

    @eqx.filter_jit
    def _local_tendon_basis_single(
        self,
        i: Array,
        xi_i: Array,
        s: Array,
        tendon_routing_params_k: BaseTendonRoutingParams,
        d_s_fn: Callable,
        dd_s_ds_fn: Callable,
        attachment_segment_idx: Array,
    ) -> Array:
        """Return the local planar actuation basis for one tendon at ``s``."""
        is_tendon_active = attachment_segment_idx >= i
        d = self._planar_component_from_routing_value(
            d_s_fn(tendon_routing_params_k, s), "d_s"
        )
        dd_ds = self._planar_component_from_routing_value(
            dd_s_ds_fn(tendon_routing_params_k, s), "dd_s_ds"
        )

        kappa = xi_i[0]
        axial = xi_i[1] + d * kappa
        shear = xi_i[2] + dd_ds
        norm = jnp.sqrt(axial**2 + shear**2)
        basis = jnp.array([d * axial / norm, axial / norm, shear / norm])
        return is_tendon_active * basis

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """Return the active tendon actuation matrix."""
        return self._actuation_matrix(
            q, self.active_tendon_routing, self.active_d_s, self.active_dd_s_ds
        )

    @eqx.filter_jit
    def jacobian_passive_tendon(self, q: Array) -> Array:
        """Return the passive tendon length Jacobian in active coordinates."""
        return self._actuation_matrix(
            q,
            self.passive_tendon_routing,
            self.passive_d_s,
            self.passive_dd_s_ds,
        ).T

    @eqx.filter_jit
    def _actuation_matrix(
        self,
        q: Array,
        tendon_routing_params: BaseTendonRoutingParams,
        d_s: Callable,
        dd_s_ds: Callable,
    ) -> Array:
        """Compute the generalized tendon actuation matrix by quadrature."""
        xi = self.strain(q).reshape((self.num_segments, 3))
        num_tendons = tendon_routing_params.num_tendons
        if num_tendons == 0:
            return jnp.zeros((self.num_dofs, 0), dtype=q.dtype)

        def A_segment_i(i: Array) -> Array:
            xi_i = xi[i]

            def A_point_j(j: Array) -> Array:
                Xs_j = Xs_scaled[j]
                Ws_j = Ws_scaled[j]
                Phi_a_j = vmap(
                    self._local_tendon_basis_single,
                    in_axes=(None, None, None, 0, None, None, 0),
                    out_axes=-1,
                )(
                    i,
                    xi_i,
                    Xs_j,
                    tendon_routing_params,
                    d_s,
                    dd_s_ds,
                    tendon_routing_params.attachment_segment_index_array,
                )
                return Ws_j * Phi_a_j

            Xs_scaled, Ws_scaled = scale_gaussian_quadrature(
                self.integration_points,
                self.integration_weights,
                self.L_cum[i],
                self.L_cum[i + 1],
            )
            return vmap(A_point_j)(jnp.arange(self.num_integration_points))

        A_blocks = vmap(A_segment_i)(jnp.arange(self.num_segments))
        A = jnp.sum(A_blocks, axis=1).reshape((3 * self.num_segments, num_tendons))
        return self.B_xi.T @ A

    @eqx.filter_jit
    def active_tendon_length(self, q: Array) -> Array:
        """Return active tendon lengths."""
        return self._tendon_length(
            q, self.active_tendon_routing, self.active_d_s, self.active_dd_s_ds
        )

    @eqx.filter_jit
    def passive_tendon_length(self, q: Array) -> Array:
        """Return passive tendon lengths."""
        return self._tendon_length(
            q,
            self.passive_tendon_routing,
            self.passive_d_s,
            self.passive_dd_s_ds,
        )

    tendon_length = active_tendon_length
    actuated_coordinates = active_tendon_length

    @eqx.filter_jit
    def _tendon_length(
        self,
        q: Array,
        tendon_routing_params: BaseTendonRoutingParams,
        d_s: Callable,
        dd_s_ds: Callable,
    ) -> Array:
        """Compute tendon lengths by integrating local planar length density."""
        xi = self.strain(q).reshape((self.num_segments, 3))
        num_tendons = tendon_routing_params.num_tendons
        if num_tendons == 0:
            return jnp.zeros((0,), dtype=q.dtype)

        def length_density_segment_i(i: Array) -> Array:
            xi_i = xi[i]

            def length_density_point_j(j: Array) -> Array:
                Xs_j = Xs_scaled[j]
                Ws_j = Ws_scaled[j]

                def length_density_tendon_k(
                    tendon_routing_params_k: BaseTendonRoutingParams,
                    attachment_segment_idx: Array,
                ) -> Array:
                    is_tendon_active = attachment_segment_idx >= i
                    d = self._planar_component_from_routing_value(
                        d_s(tendon_routing_params_k, Xs_j), "d_s"
                    )
                    dd_ds = self._planar_component_from_routing_value(
                        dd_s_ds(tendon_routing_params_k, Xs_j), "dd_s_ds"
                    )
                    axial = xi_i[1] + d * xi_i[0]
                    shear = xi_i[2] + dd_ds
                    return is_tendon_active * jnp.sqrt(axial**2 + shear**2)

                dl_ds_j = vmap(length_density_tendon_k)(
                    tendon_routing_params,
                    tendon_routing_params.attachment_segment_index_array,
                )
                return Ws_j * dl_ds_j

            Xs_scaled, Ws_scaled = scale_gaussian_quadrature(
                self.integration_points,
                self.integration_weights,
                self.L_cum[i],
                self.L_cum[i + 1],
            )
            return vmap(length_density_point_j)(
                jnp.arange(self.num_integration_points)
            )

        dl_ds_blocks = vmap(length_density_segment_i)(jnp.arange(self.num_segments))
        return jnp.sum(dl_ds_blocks, axis=(0, 1))

    @eqx.filter_jit
    def forward_kinematics_active_tendons(self, q: Array, s: Array) -> Array:
        """Return active tendon Cartesian points at arc length ``s``."""
        return self._forward_kinematics_tendons(
            q, s, self.active_tendon_routing, self.active_d_s
        )

    @eqx.filter_jit
    def forward_kinematics_passive_tendons(self, q: Array, s: Array) -> Array:
        """Return passive tendon Cartesian points at arc length ``s``."""
        return self._forward_kinematics_tendons(
            q, s, self.passive_tendon_routing, self.passive_d_s
        )

    @eqx.filter_jit
    def forward_kinematics_tendons(self, q: Array, s: Array) -> Array:
        """Return active and passive tendon Cartesian points at arc length ``s``."""
        active_pos = self.forward_kinematics_active_tendons(q, s)
        passive_pos = self.forward_kinematics_passive_tendons(q, s)
        if active_pos.size == 0:
            return passive_pos
        if passive_pos.size == 0:
            return active_pos
        return jnp.concatenate([active_pos, passive_pos], axis=0)

    @eqx.filter_jit
    def _forward_kinematics_tendons(
        self,
        q: Array,
        s: Array,
        tendon_routing_params: BaseTendonRoutingParams,
        d_s: Callable,
    ) -> Array:
        """Compute Cartesian tendon positions for one routing family."""
        if tendon_routing_params.num_tendons == 0:
            return jnp.zeros((0, 2), dtype=q.dtype)

        def forward_kinematics_tendon_k(
            tendon_routing_params_k: BaseTendonRoutingParams,
            attachment_segment_idx: Array,
            q: Array,
            s: Array,
        ) -> Array:
            attachment_s = self.L_cum[attachment_segment_idx + 1]
            s_val = jnp.clip(s, 0.0, attachment_s)
            pose = self.forward_kinematics(q, s_val)
            theta = pose[0]
            normal = jnp.array([-jnp.sin(theta), jnp.cos(theta)])
            d = self._planar_component_from_routing_value(
                d_s(tendon_routing_params_k, s_val), "d_s"
            )
            return pose[1:3] + d * normal

        return vmap(forward_kinematics_tendon_k, in_axes=(0, 0, None, None))(
            tendon_routing_params,
            tendon_routing_params.attachment_segment_index_array,
            q,
            s,
        )

    def actuator_visual_layers(
        self,
        q: Array,
        s_points: Array,
        *,
        actuator_inputs: Array | None = None,
    ) -> tuple[ActuatorVisualLayer, ...]:
        """Return renderer-facing actuator geometry for active/passive tendons."""
        del actuator_inputs
        layers: list[ActuatorVisualLayer] = []
        if self.active_tendon_routing.num_tendons > 0:
            active_points = vmap(
                lambda s: self.forward_kinematics_active_tendons(q, s)
            )(s_points)
            layers.append(
                ActuatorVisualLayer(
                    name="active_tendons",
                    kind="tendon",
                    points=active_points.transpose(1, 0, 2),
                )
            )
        if self.passive_tendon_routing.num_tendons > 0:
            passive_points = vmap(
                lambda s: self.forward_kinematics_passive_tendons(q, s)
            )(s_points)
            layers.append(
                ActuatorVisualLayer(
                    name="passive_tendons",
                    kind="tendon",
                    points=passive_points.transpose(1, 0, 2),
                )
            )
        return tuple(layers)

    @eqx.filter_jit
    def elastic_force(self, q: Array) -> Array:
        """Return body elastic force plus passive tendon spring force."""
        tau_el = super().elastic_force(q)
        J_pt = self.jacobian_passive_tendon(q)
        l_pt = self.passive_tendon_length(q)
        return tau_el + J_pt.T @ self.K_pt @ (l_pt - self.l_pt0)

    @eqx.filter_jit
    def damping_matrix(self, q: Array) -> Array:
        """Return body damping plus passive tendon damping."""
        D = super().damping_matrix(q)
        J_pt = self.jacobian_passive_tendon(q)
        return D + J_pt.T @ self.D_pt @ J_pt

    @eqx.filter_jit
    def _elastic_energy(self, q: Array) -> Array:
        """Return body elastic energy plus passive tendon spring energy."""
        U_K = super()._elastic_energy(q)
        l_pt = self.passive_tendon_length(q)
        delta_l = l_pt - self.l_pt0
        return U_K + 0.5 * delta_l.T @ self.K_pt @ delta_l
