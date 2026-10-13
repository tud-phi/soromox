"""Helicoidal lattice arms expressed with geometric variable strains."""

from typing import Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from soromox.execution import (
    GVS_DYNAMICS,
    ExecutionBackend,
    dispatch_dynamics_terms,
    evaluate_forward_dynamics,
)
from soromox.systems.components import IsotropicMaterialParams
from soromox.systems.gvs.construction import project_sectional_matrices
from soromox.systems.gvs.core import GVS
from soromox.systems.gvs.params import HelicoidalParams, HelicoidalSectionParams
from soromox.systems.gvs.specs import GVSSegment
from soromox.systems.gvs.structures import GVSStructure
from soromox.systems.params import _contains_tracer

__all__ = ["Helicoidal"]


def _require_jax(backend: ExecutionBackend | None) -> None:
    if backend not in (None, "auto", "jax"):
        raise ValueError("Helicoidal currently supports only the JAX backend.")


class Helicoidal(GVS):
    """GVS arm with effective helicoidal stiffness and independent lattice inertia.

    The local x-axis is tangent to the backbone. Sectional strains are ordered
    ``[twist, bend_y, bend_z, extension, shear_y, shear_z]``. The analytic law
    ``EI(s) = 1 / (cA + cB * (L-s))`` is projected with each link's existing
    Gauss quadrature and strain basis by :meth:`with_sectional_law`.

    Canonical generalized stiffness and damping remain in ``params.link``.
    Changing length preserves those matrices; call ``with_sectional_law`` after
    a length change to project the constitutive law over the new geometry.
    Numeric updates preserve compiled methods when structure, shapes, and
    dtypes remain fixed. Rendering cross sections do not determine lattice mass.

    Platform masses are point masses at link tips. Tendons use the existing
    contiguous GVS routing, including compliant transitions and rigid links.
    The initial implementation supports fixed bases and JAX execution.

    References:
        Qin, Z., et al. (2026). Cosserat Modeling of Trimmed Helicoid Soft Arms
        with a Separated-Section Constitutive Law. arXiv:2609.25264.
        https://arxiv.org/abs/2609.25264
    """

    def __init__(
        self,
        params: HelicoidalParams,
        structure: GVSStructure,
        *,
        backend: ExecutionBackend = "jax",
        floating_base: bool = False,
        **kwargs: Any,
    ) -> None:
        """Construct from lattice parameters and an ordinary GVS structure.

        Args:
            params: Canonical mechanics, line inertia, and link-tip masses.
            structure: Static GVS link, joint, basis, and quadrature choices.
            backend: ``"jax"`` or ``"auto"``; both select JAX.
            floating_base: Must be false for the initial implementation.
            **kwargs: GVS constructor arguments, including mounting and actuators.

        Raises:
            TypeError: If params are not helicoidal parameters.
            ValueError: If Warp, a floating base, or invalid parameters are used.
        """
        if not isinstance(params, HelicoidalParams):
            raise TypeError("params must be a HelicoidalParams instance.")
        _require_jax(backend)
        if floating_base:
            raise ValueError("Helicoidal currently requires a fixed base.")
        super().__init__(params, structure, backend="jax", **kwargs)

    @classmethod
    def from_segments(
        cls,
        segments: list[GVSSegment] | tuple[GVSSegment, ...],
        *,
        mass_per_length: Array,
        rotational_inertia_per_length: Array,
        tip_mass: Array | None = None,
        sectional_law: HelicoidalSectionParams | None = None,
        gravity: Array | None = None,
        max_dof: int | None = None,
        max_num_gauss_points: int | None = None,
        scale_rotational_basis_by_length: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Construct from native segment specs and independent lattice inertia.

        Args:
            segments: Ordinary GVS segment specifications.
            mass_per_length: One constant line mass per link [kg/m].
            rotational_inertia_per_length: Principal local x/y/z inertias,
                shape ``(num_links, 3)`` [kg m].
            tip_mass: One point mass per link tip [kg]; defaults to zeros.
            sectional_law: Optional analytic law replacing the link stiffness
                and damping supplied by the segment specs.
            gravity: Optional world-frame acceleration, shape ``(3,)``.
            max_dof: Optional common padded link/joint matrix dimension.
            max_num_gauss_points: Optional common quadrature padding.
            scale_rotational_basis_by_length: Normalize rotational basis rows
                by link length when true.
            **kwargs: Constructor arguments, including base pose and actuators.

        Returns:
            A fully initialized helicoidal model.
        """
        params, structure = GVS.params_from_segments(
            segments,
            gravity=gravity,
            max_dof=max_dof,
            max_num_gauss_points=max_num_gauss_points,
            scale_rotational_basis_by_length=scale_rotational_basis_by_length,
        )
        params = HelicoidalParams(
            link=params.link,
            joint=params.joint,
            gravity=params.gravity,
            mass_per_length=mass_per_length,
            rotational_inertia_per_length=rotational_inertia_per_length,
            tip_mass=jnp.zeros_like(params.link.length)
            if tip_mass is None
            else tip_mass,
        )
        model = cls(params, structure, **kwargs)
        return (
            model if sectional_law is None else model.with_sectional_law(sectional_law)
        )

    def _lattice_mass_matrices(self) -> Array:
        """Broadcast constant line inertia to the padded GVS quadrature grid."""
        diagonal = jnp.concatenate(
            (
                self.params.rotational_inertia_per_length,
                jnp.repeat(self.params.mass_per_length[:, None], 3, axis=1),
            ),
            axis=1,
        )
        matrices = jax.vmap(jnp.diag)(diagonal)
        return jnp.broadcast_to(matrices[:, None], self.mass_matrices.shape)

    def precompute(self) -> None:
        """Initialize GVS caches using the independent lattice inertia."""
        object.__setattr__(self, "mass_matrices", self._lattice_mass_matrices())
        super().precompute()

    def with_params(self, params: HelicoidalParams) -> Self:
        """Refresh geometry and inertia while preserving canonical link matrices.

        Args:
            params: Replacement parameters with the same static GVS layout.

        Returns:
            A new model, retaining the helicoidal type and refreshed mass caches.
        """
        if not isinstance(params, HelicoidalParams):
            raise TypeError("params must be a HelicoidalParams instance.")
        updated = super().with_params(params)
        mass = updated._lattice_mass_matrices()
        inner_mass = mass[:, 1 : updated.max_num_integration_points - 1]
        diagonal = jnp.diagonal(inner_mass, axis1=-2, axis2=-1)
        return eqx.tree_at(
            lambda model: (
                model.mass_matrices,
                model.inner_mass_matrices,
                model.inner_mass_diagonals,
                model.inner_weighted_mass_diagonals,
            ),
            updated,
            (
                mass,
                inner_mass,
                diagonal,
                updated.inner_integration_weights[..., None] * diagonal,
            ),
        )

    def with_sectional_law(self, law: HelicoidalSectionParams) -> Self:
        """Project the analytic sectional law using the configured Gauss rules.

        This evaluates ``integral B.T @ diag(GJ, EI, EI, EA, GA, GA) @ B ds``
        (paper Eq. 25) directly at each link's quadrature nodes. No station table
        or interpolation is used. Damping is ``tau * K`` per link.

        Args:
            law: Effective constitutive coefficients, one row per link.

        Returns:
            A model with new canonical generalized stiffness and damping.

        Raises:
            TypeError: If the law has the wrong parameter type.
            ValueError: If the link count differs or bending compliance is not
                strictly positive throughout any link.
        """
        if not isinstance(law, HelicoidalSectionParams):
            raise TypeError("law must be a HelicoidalSectionParams instance.")
        law.validate_for_update()
        if law.bending_compliance.shape[0] != self.num_segments:
            raise ValueError("sectional law must contain one row per link.")
        if not _contains_tracer((law, self.segment_lengths)):
            with jax.ensure_compile_time_eval():
                proximal_compliance = (
                    law.bending_compliance[:, 0]
                    + law.bending_compliance[:, 1] * self.segment_lengths
                )
                if not bool(jnp.all(proximal_compliance > 0)):
                    raise ValueError(
                        "Bending compliance must be positive throughout each link."
                    )

        def sectional_matrix(index: int, s: Array) -> Array:
            c_a, c_b = law.bending_compliance[index]
            ei = 1 / (c_a + c_b * (self.segment_lengths[index] - s))

            def constant(value: Array) -> Array:
                return jnp.broadcast_to(value[index], s.shape)

            diagonal = jnp.stack(
                (
                    constant(law.torsional_stiffness),
                    ei,
                    ei,
                    constant(law.axial_stiffness),
                    constant(law.shear_stiffness),
                    constant(law.shear_stiffness),
                ),
                axis=-1,
            )
            return jax.vmap(jnp.diag)(diagonal)

        stiffness = project_sectional_matrices(
            self.params, self.structure, sectional_matrix
        )
        damping = law.damping_time_constant[:, None, None] * stiffness
        return self.update_link_params(stiffness=stiffness, damping=damping)

    def with_isotropic_material(self, material: IsotropicMaterialParams) -> Self:
        """Reject solid-section reprojection; use ``with_sectional_law`` instead."""
        raise ValueError("Use with_sectional_law for effective helicoidal mechanics.")

    def _tip_linear_jacobians(self, q: Array) -> Array:
        return self.jacobian_tips(q)[:, 3:, :]

    def _tip_mass_matrix(self, jacobians: Array) -> Array:
        return jnp.einsum("n,nai,naj->ij", self.params.tip_mass, jacobians, jacobians)

    def _tip_gravity_force(self, jacobians: Array) -> Array:
        return -jnp.einsum("n,nai,a->i", self.params.tip_mass, jacobians, self.g[3:])

    @eqx.filter_jit
    def inertia_matrix(self, q: Array) -> Array:
        """Return distributed and platform inertia in generalized coordinates."""
        return super().inertia_matrix(q) + self._tip_mass_matrix(
            self._tip_linear_jacobians(q)
        )

    @eqx.filter_jit
    def coriolis_matrix(self, q: Array, qd: Array) -> Array:
        """Return the distributed and platform velocity-dependent matrix."""
        jacobians, derivatives = jax.jvp(self._tip_linear_jacobians, (q,), (qd,))
        return super().coriolis_matrix(q, qd) + jnp.einsum(
            "n,nai,naj->ij", self.params.tip_mass, jacobians, derivatives
        )

    @eqx.filter_jit
    def _gravitational_force(self, q: Array) -> Array:
        return super()._gravitational_force(q) + self._tip_gravity_force(
            self._tip_linear_jacobians(q)
        )

    @eqx.filter_jit
    def _gravitational_energy(self, q: Array) -> Array:
        positions = self.forward_kinematics_tips(q)[:, :3, 3]
        return super()._gravitational_energy(q) - jnp.einsum(
            "n,na,a->", self.params.tip_mass, positions, self.g[3:]
        )

    @eqx.filter_jit
    def _assemble_dynamics_terms(
        self, q: Array, qd: Array
    ) -> tuple[Array, Array, Array]:
        mass, bias, gravity = super()._assemble_dynamics_terms(q, qd)
        jacobians, derivatives = jax.jvp(self._tip_linear_jacobians, (q,), (qd,))
        tip_bias = jnp.einsum(
            "n,nai,naj,j->i", self.params.tip_mass, jacobians, derivatives, qd
        )
        return (
            mass + self._tip_mass_matrix(jacobians),
            bias + tip_bias,
            gravity + self._tip_gravity_force(jacobians),
        )

    def actuation_matrix(
        self, q: Array, *, backend: ExecutionBackend | None = None
    ) -> Array:
        """Evaluate the existing contiguous tendon transmission with JAX."""
        _require_jax(backend)
        return super().actuation_matrix(q, backend="jax")

    def actuation_force(
        self,
        q: Array,
        u: Array,
        qd: Array | None = None,
        *,
        backend: ExecutionBackend | None = None,
    ) -> Array:
        """Evaluate native actuator forces with JAX."""
        _require_jax(backend)
        return super().actuation_force(q, u, qd, backend="jax")

    def dynamics_terms(
        self, q: Array, qd: Array, *, backend: ExecutionBackend | None = None
    ) -> tuple[Array, Array, Array]:
        """Assemble inertia, velocity bias, and gravity, with optional batching."""
        _require_jax(backend)
        return dispatch_dynamics_terms(
            self, q, qd, backend="jax", capabilities=GVS_DYNAMICS, warp_supported=False
        )

    def forward_dynamics(
        self,
        t: Array,
        y: Array,
        actuation_args: tuple | None = None,
        *,
        backend: ExecutionBackend | None = None,
    ) -> Array:
        """Evaluate JAX dynamics including lattice inertia and platform masses."""
        _require_jax(backend)
        return evaluate_forward_dynamics(self, t, y, actuation_args, "jax")
