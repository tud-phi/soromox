"""Helicoidal lattice arms expressed with geometric variable strains."""

from collections.abc import Mapping
from typing import Any, Self

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from soromox.systems.components import IsotropicMaterialParams
from soromox.systems.gvs.construction import project_sectional_matrices
from soromox.systems.gvs.core import GVS
from soromox.systems.gvs.params import HelicoidalBackboneParams, HelicoidalHelixParams
from soromox.systems.gvs.specs import GVSSegment
from soromox.systems.params import _contains_tracer

__all__ = ["Helicoidal"]


class Helicoidal(GVS):
    """Tendon-driven helicoidal lattice arm with geometric variable strains.

    An effective sectional constitutive law describes the lattice's bending, torsion,
    extension, and shear response. Compliance varies linearly between the
    reciprocals of the endpoint sectional stiffnesses. Distributed lattice inertia
    and attached
    bodies at link tips account for the arm and its platforms. Tendons run
    continuously through compliant transitions and rigid links.

    Supply an effective ``HelicoidalBackboneParams`` law directly, or prepare it
    from strand geometry and material using ``HelicoidalHelixParams`` and
    ``with_constitutive_params``.

    The local x-axis follows the backbone, with strains ordered as
    ``[twist, bend_y, bend_z, extension, shear_y, shear_z]``. Generalized
    stiffness and damping are stored in ``params.link``. After changing link
    lengths, call ``with_constitutive_params`` to project the law over the new geometry.
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
        constitutive_params: Mapping[
            int, HelicoidalBackboneParams | HelicoidalHelixParams
        ]
        | None = None,
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
            constitutive_params: Optional mapping from zero-based segment indices
                to backbone or helix parameters. Replaces stiffness and damping
                for selected segments; see ``with_constitutive_params``.
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
            model
            if constitutive_params is None
            else model.with_constitutive_params(constitutive_params)
        )

    def with_constitutive_params(
        self,
        params: Mapping[int, HelicoidalBackboneParams | HelicoidalHelixParams],
    ) -> Self:
        """Project constitutive parameters onto selected segments' GVS bases.

        Each mapping entry describes one segment. Helix inputs are converted
        to backbone parameters using six unit loads at the soft segment tip.
        Shared helix objects are reduced once per call. Their reference lengths
        must match the selected GVS segments.

        Backbone compliance is evaluated at each segment's Gauss nodes and
        projected as ``integral B.T @ diag(1/C(s)) @ B ds`` (paper Eq. 25).
        Damping is ``tau * K``. Active basis strains must be allowed by the
        backbone parameters' ``admissible_strains``.

        Backbone inputs support JAX transformations and differentiation. Prepare
        helix inputs eagerly on the CPU; repeat conversion after geometry or
        material updates. Prepared models reuse compiled dynamics with fixed
        structure, shapes and dtypes.

        Args:
            params: Mapping from zero-based segment indices to backbone or helix
                parameters. Unselected segments retain their stiffness and damping.

        Returns:
            A model with updated canonical stiffness and damping in ``params.link``.

        Raises:
            TypeError: If a value has an unsupported parameter type.
            ValueError: If an index, helix length or constitutive fit is invalid,
                or a basis activates inadmissible strains. Use
                ``helix.to_backbone().admissible_strains`` to choose a compatible
                basis before construction.
        """
        if not params:
            return self
        laws: dict[int, HelicoidalBackboneParams] = {}
        reductions: dict[int, HelicoidalBackboneParams] = {}
        for index, value in params.items():
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < self.num_segments
            ):
                raise ValueError(f"Invalid constitutive segment index: {index!r}.")
            if isinstance(value, HelicoidalHelixParams):
                if _contains_tracer((value, self.segment_lengths)):
                    raise ValueError(
                        "Prepare helix inputs outside JAX transformations; "
                        "use to_backbone() before projecting inside JAX."
                    )
                if not np.isclose(
                    float(value.length),
                    float(self.segment_lengths[index]),
                    rtol=1e-6,
                    atol=0,
                ):
                    raise ValueError(
                        f"Helix length must match soft segment {index} length."
                    )
                key = id(value)
                if key not in reductions:
                    reductions[key] = value.to_backbone()
                law = reductions[key]
            elif isinstance(value, HelicoidalBackboneParams):
                law = value
            else:
                raise TypeError(
                    "constitutive_params values must be HelicoidalBackboneParams "
                    "or HelicoidalHelixParams instances."
                )
            law.validate_for_update()
            active = self.structure.segments[index].basis.strain_selector
            if any(
                a and not b for a, b in zip(active, law.admissible_strains, strict=True)
            ):
                raise ValueError(
                    f"Segment {index} activates inadmissible strains; "
                    "choose a basis compatible with the parameters' admissible_strains."
                )
            laws[index] = law

        def sectional_matrix(index: int, s: Array) -> Array:
            if index not in laws:
                return jnp.zeros((len(s), 6, 6), dtype=self.params.link.stiffness.dtype)
            law = laws[index]
            u = s[:, None] / self.segment_lengths[index]
            compliance = (1 - u) / law.stiffness_base + u / law.stiffness_tip
            return jax.vmap(jnp.diag)(1 / compliance)

        stiffness = project_sectional_matrices(
            self.params, self.structure, sectional_matrix
        )
        damping_times = jnp.stack(
            [
                laws[i].damping_time_constant
                if i in laws
                else jnp.zeros((), dtype=stiffness.dtype)
                for i in range(self.num_segments)
            ]
        )
        damping = damping_times[:, None, None] * stiffness
        mask = jnp.array([i in laws for i in range(self.num_segments)])[:, None, None]
        return self.update_link_params(
            stiffness=jnp.where(mask, stiffness, self.params.link.stiffness),
            damping=jnp.where(mask, damping, self.params.link.damping),
        )

    def with_isotropic_material(self, material: IsotropicMaterialParams) -> Self:
        """Require an effective sectional constitutive law for helicoidal lattice mechanics."""
        raise ValueError(
            "Use with_constitutive_params for effective helicoidal mechanics."
        )
