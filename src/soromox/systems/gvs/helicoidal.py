"""Helicoidal lattice arms expressed with geometric variable strains."""

from typing import Any, Self

import jax
import jax.numpy as jnp
from jax import Array

from soromox.systems.components import IsotropicMaterialParams
from soromox.systems.gvs.construction import project_sectional_matrices
from soromox.systems.gvs.core import GVS
from soromox.systems.gvs.params import HelicoidalSectionParams
from soromox.systems.gvs.specs import GVSSegment
from soromox.systems.params import _contains_tracer

__all__ = ["Helicoidal"]


class Helicoidal(GVS):
    """Tendon-driven helicoidal lattice arm with geometric variable strains.

    An effective sectional law describes the lattice's bending, torsion,
    extension, and shear response. Bending stiffness varies along each link as
    ``EI(s) = 1 / (cA + cB * (L-s))``. Distributed lattice inertia and attached
    bodies at link tips account for the arm and its platforms. Tendons run
    continuously through compliant transitions and rigid links.

    The local x-axis follows the backbone, with strains ordered as
    ``[twist, bend_y, bend_z, extension, shear_y, shear_z]``. Generalized
    stiffness and damping are stored in ``params.link``. After changing link
    lengths, call ``with_sectional_law`` to project the law over the new geometry.
    Numeric parameter updates reuse compilation with fixed structure, shapes,
    and dtypes. Supports JAX and Warp dynamics through GVS.

    References:
        Guan, Qinghua, Francesco Stella, Cosimo Della Santina, Jinsong Leng,
        and Josie Hughes (2023). Trimmed helicoids: an architectured soft
        structure yielding soft robots with high precision, large workspace,
        and compliant interactions. npj Robotics, 1, Article 4.
        https://doi.org/10.1038/s44182-023-00004-7

        Qin, Zhihang, Linxin Hou, Zeyu Zhong, Yuchen Sun, Wenci Xin, Yueheng Zhang,
        Ji Qi, Jie Wang, Peiyi Wang, Muhammad Sunny Nazeer, Yu Jun Tan,
        Federico Renda, and Cecilia Laschi (2026). Cosserat Modeling of Trimmed
        Helicoid Soft Arms with a Separated-Section Constitutive Law.
        arXiv:2609.25264. https://arxiv.org/abs/2609.25264
    """

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
            segments: GVS segment specifications.
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
        diagonal = jnp.concatenate(
            (
                jnp.asarray(rotational_inertia_per_length),
                jnp.repeat(jnp.asarray(mass_per_length)[:, None], 3, axis=1),
            ),
            axis=1,
        )
        tip_mass = (
            jnp.zeros_like(params.link.length)
            if tip_mass is None
            else jnp.asarray(tip_mass)
        )
        tip_diagonal = jnp.concatenate(
            (jnp.zeros((len(segments), 3)), jnp.repeat(tip_mass[:, None], 3, axis=1)),
            axis=1,
        )
        params = params.replace(
            link=params.link.replace(
                sectional_inertia=jax.vmap(jnp.diag)(diagonal),
                tip_inertia=jax.vmap(jnp.diag)(tip_diagonal),
            )
        )
        model = cls(params, structure, **kwargs)
        return (
            model if sectional_law is None else model.with_sectional_law(sectional_law)
        )

    def with_sectional_law(self, law: HelicoidalSectionParams) -> Self:
        """Project the analytic sectional law using the configured Gauss rules.

        This evaluates ``integral B.T @ diag(1 / C(s)) @ B ds``
        (paper Eq. 25) at each link's quadrature nodes. Damping is ``tau * K``
        per link.

        Args:
            law: Effective constitutive coefficients, one row per link.

        Returns:
            A model with new canonical generalized stiffness and damping.

        Raises:
            TypeError: If the law has the wrong parameter type.
            ValueError: If the link count differs or sectional compliance is not
                strictly positive throughout any link.
        """
        if not isinstance(law, HelicoidalSectionParams):
            raise TypeError("law must be a HelicoidalSectionParams instance.")
        law.validate_for_update()
        if law.compliance.shape[0] != self.num_segments:
            raise ValueError("sectional law must contain one row per link.")
        if not _contains_tracer((law, self.segment_lengths)):
            with jax.ensure_compile_time_eval():
                proximal_compliance = (
                    law.compliance
                    + law.compliance_gradient * self.segment_lengths[:, None]
                )
                if not bool(jnp.all(proximal_compliance > 0)):
                    raise ValueError(
                        "Sectional compliance must be positive throughout each link."
                    )

        def sectional_matrix(index: int, s: Array) -> Array:
            compliance = (
                law.compliance[index]
                + (self.segment_lengths[index] - s[:, None])
                * law.compliance_gradient[index]
            )
            diagonal = 1 / compliance
            return jax.vmap(jnp.diag)(diagonal)

        stiffness = project_sectional_matrices(
            self.params, self.structure, sectional_matrix
        )
        damping = law.damping_time_constant[:, None, None] * stiffness
        return self.update_link_params(stiffness=stiffness, damping=damping)

    def with_isotropic_material(self, material: IsotropicMaterialParams) -> Self:
        """Require an effective sectional law for helicoidal lattice mechanics."""
        raise ValueError("Use with_sectional_law for effective helicoidal mechanics.")
