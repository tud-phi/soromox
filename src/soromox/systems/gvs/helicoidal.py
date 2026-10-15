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
    ``with_constitutive_params``. Helix inputs also derive sectional inertia from
    strand geometry and density; ordinary GVS segments use their own geometry
    and density.

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
        mass_per_length: Array | None = None,
        rotational_inertia_per_length: Array | None = None,
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
        """Construct from segment specs and optional helix material/geometry inputs.

        Args:
            segments: GVS segment specifications.
            mass_per_length: Optional constant line-mass override per link [kg/m].
                Supply together with ``rotational_inertia_per_length``. These
                overrides take precedence over geometry-derived inertia.
            rotational_inertia_per_length: Optional principal local x/y/z inertia
                override, shape ``(num_links, 3)`` [kg m]. By default, helix inputs
                derive strand inertia; other segments use their GVS geometry and
                density.
            tip_mass: One point mass per link tip [kg]; defaults to zeros.
            constitutive_params: Optional mapping from zero-based segment indices
                to backbone or helix parameters. Replaces stiffness and damping
                for selected segments. Helix inputs also derive distributed inertia;
                see ``with_constitutive_params``.
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
        if (mass_per_length is None) != (rotational_inertia_per_length is None):
            raise ValueError(
                "Supply mass_per_length and rotational_inertia_per_length together."
            )
        count = len(segments)
        points = structure.max_num_gauss_points or max(
            s.num_gauss_points for s in segments
        )
        dtype = params.link.stiffness.dtype
        tip_mass = (
            jnp.zeros(count, dtype=dtype) if tip_mass is None else jnp.asarray(tip_mass)
        )
        tip_diagonal = jnp.concatenate(
            (
                jnp.zeros((count, 3), dtype=dtype),
                jnp.repeat(tip_mass[:, None], 3, axis=1),
            ),
            axis=1,
        )
        params = params.replace(
            link=params.link.replace(
                sectional_inertia=jnp.zeros((count, points, 6, 6), dtype=dtype),
                sectional_inertia_mask=jnp.zeros(count, dtype=bool),
                tip_inertia=jax.vmap(jnp.diag)(tip_diagonal),
            )
        )
        model = cls(params, structure, **kwargs)
        if constitutive_params is not None:
            model = model.with_constitutive_params(constitutive_params)
        if mass_per_length is not None:
            mass = jnp.asarray(mass_per_length)
            rotation = jnp.asarray(rotational_inertia_per_length)
            if mass.shape != (count,) or rotation.shape != (count, 3):
                raise ValueError(
                    "Inertia overrides require shapes (num_links,) and (num_links, 3)."
                )
            diagonal = jnp.concatenate(
                (rotation, jnp.repeat(mass[:, None], 3, axis=1)), axis=1
            )
            model = model.update_link_params(
                sectional_inertia=jnp.broadcast_to(
                    jax.vmap(jnp.diag)(diagonal)[:, None], (count, points, 6, 6)
                ),
                sectional_inertia_mask=jnp.ones(count, dtype=bool),
            )
        return model

    def with_constitutive_params(
        self,
        params: Mapping[int, HelicoidalBackboneParams | HelicoidalHelixParams],
    ) -> Self:
        """Project constitutive parameters onto selected segments' GVS bases.

        Each mapping entry describes one segment. Helix inputs are converted
        to backbone parameters using six unit loads at the soft segment tip.
        Shared helix objects are reduced once per call. Their reference lengths
        must match the selected GVS segments. Their geometry and density also
        refresh the selected segments' distributed inertia. Backbone inputs
        update stiffness and damping. For density-only updates, use
        ``with_helix_inertia`` to prepare inertia directly.

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
            A model with updated stiffness, damping, and derived helix inertia
            in ``params.link``.

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
        helix_inputs = {
            i: value
            for i, value in params.items()
            if isinstance(value, HelicoidalHelixParams)
        }
        inertia_updates = (
            self._helix_inertia_updates(helix_inputs) if helix_inputs else {}
        )
        return self.update_link_params(
            stiffness=jnp.where(mask, stiffness, self.params.link.stiffness),
            damping=jnp.where(mask, damping, self.params.link.damping),
            **inertia_updates,
        )

    def with_helix_inertia(self, params: Mapping[int, HelicoidalHelixParams]) -> Self:
        """Derive distributed strand inertia for selected segments.

        Use this for density-only updates or with explicitly converted backbone
        laws. Geometry and density are evaluated at the existing Gauss points;
        strand contributions include cross-sectional inertia, offsets and the
        strand-to-backbone arc-length factor. Volumes add at fused crossings.
        Other segments retain their inertia selection, including ordinary GVS
        geometry/density evaluation. Attached bodies retain their own inertia.

        Args:
            params: Segment indices mapped to helix inputs with matching lengths.

        Returns:
            A model with updated sectional inertia. Numeric updates support JAX
            differentiation and reuse compilation at fixed shapes and dtypes.
        """
        if not params:
            return self
        return self.update_link_params(**self._helix_inertia_updates(params))

    def _helix_inertia_updates(
        self, params: Mapping[int, HelicoidalHelixParams]
    ) -> dict[str, Array]:
        shape = (self.num_segments, self.max_num_gauss_points, 6, 6)
        inertia = self.params.link.sectional_inertia
        if inertia is None:
            inertia = jnp.zeros(shape, dtype=self.params.link.stiffness.dtype)
            mask = jnp.zeros(self.num_segments, dtype=bool)
        else:
            inertia = jnp.broadcast_to(
                inertia[:, None] if inertia.ndim == 3 else inertia, shape
            )
            mask = self.params.link.sectional_inertia_mask
            if mask is None:
                mask = jnp.ones(self.num_segments, dtype=bool)
        for index, value in params.items():
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < self.num_segments
            ):
                raise ValueError(f"Invalid helix segment index: {index!r}.")
            if not isinstance(value, HelicoidalHelixParams):
                raise TypeError(
                    "Helix inertia inputs must be HelicoidalHelixParams instances."
                )
            value.validate_for_update()
            if not _contains_tracer(
                (value.length, self.segment_lengths)
            ) and not np.isclose(
                float(value.length),
                float(self.segment_lengths[index]),
                rtol=1e-6,
                atol=0,
            ):
                raise ValueError(
                    f"Helix length must match soft segment {index} length."
                )
            positions = (
                self.integration_points[index, 1:-1] * self.segment_lengths[index]
            )
            inertia = inertia.at[index].set(value.sectional_inertia(positions))
            mask = mask.at[index].set(True)
        return {"sectional_inertia": inertia, "sectional_inertia_mask": mask}

    def with_isotropic_material(self, material: IsotropicMaterialParams) -> Self:
        """Require an effective sectional constitutive law for helicoidal lattice mechanics."""
        raise ValueError(
            "Use with_constitutive_params for effective helicoidal mechanics."
        )
