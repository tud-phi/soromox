"""Typed dynamic parameters for the geometric variable-strain system."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import jax.numpy as jnp
from jax import Array

from soromox.systems.components import ContinuumLinkParams, JointParams
from soromox.systems.params import BaseSoftRobotParams, BaseSystemParams

if TYPE_CHECKING:
    from soromox.systems.gvs.structures import GVSStructure

__all__ = ["GVSParams", "HelicoidalParams", "HelicoidalSectionParams"]


class GVSParams(BaseSoftRobotParams):
    """Canonical dynamic parameters for a GVS model.

    Attributes:
        link: Batched continuum-link parameters. Link stiffness and damping use
            the padded generalized-coordinate dimension configured by the GVS
            structure.
        joint: Batched joint stiffness and damping matrices using the same
            padded generalized-coordinate dimension as ``link``.
        gravity: Gravity vector with shape ``(3,)`` in the world frame.
    """

    is_planar: ClassVar[bool] = False

    link: ContinuumLinkParams
    joint: JointParams

    def validate_structure(self) -> None:
        """Validate intrinsic GVS parameter structure.

        Returns:
            ``None`` after successful validation.

        Raises:
            ValueError: If a component shape is invalid, link and joint batch
                shapes disagree, reference strains do not have six components,
                or the gravity array has an invalid shape.
        """
        self.link.validate_structure()
        self.joint.validate_structure()
        num_segments = self.link.length.shape[0]
        if self.link.reference_strain.shape != (num_segments, 6):
            raise ValueError(
                "link.reference_strain must have shape "
                f"({num_segments}, 6), got {self.link.reference_strain.shape}."
            )
        if self.joint.stiffness.shape[0] != num_segments:
            raise ValueError("joint params must contain one matrix per segment.")
        if self.joint.stiffness.shape != self.link.stiffness.shape:
            raise ValueError(
                "GVS link and joint matrices must use the same padded shape."
            )
        gravity = jnp.asarray(self.gravity)
        if gravity.shape != (3,):
            raise ValueError(f"gravity must have shape (3,), got {gravity.shape}.")

    def validate_values(self) -> None:
        """Validate eager GVS component values."""
        self.link.validate_values()
        self.joint.validate_values()

    def validate_structure_compatibility(self, structure: GVSStructure) -> None:
        """Validate dynamic array shapes against a static padded GVS layout.

        Args:
            structure: Static :class:`GVSStructure` defining the number of
                segments, basis DOFs, cross-section coefficient packing, and
                quadrature padding.

        Returns:
            ``None`` after successful validation.

        Raises:
            TypeError: If ``structure`` is not a :class:`GVSStructure`.
            ValueError: If the structure is empty or any parameter batch,
                padded matrix, cross-section coefficient array, basis DOF, or
                quadrature setting is structurally inconsistent.
        """
        from soromox.systems.gvs.primitives import Joint
        from soromox.systems.gvs.structures import GVSStructure

        if not isinstance(structure, GVSStructure):
            raise TypeError("structure must be a GVSStructure instance.")
        if not structure.segments:
            raise ValueError("GVS requires at least one segment.")
        num_segments = len(structure.segments)
        if self.link.length.shape != (num_segments,):
            raise ValueError(f"link params must contain {num_segments} links.")
        expected_coefficients = max(
            sum(segment.link.cross_section_profile_parameter_counts)
            for segment in structure.segments
        )
        expected_cross_section_shape = (num_segments, expected_coefficients)
        if self.link.cross_section.coefficients.shape != expected_cross_section_shape:
            raise ValueError(
                "link.cross_section.coefficients must have shape "
                f"{expected_cross_section_shape}, got "
                f"{self.link.cross_section.coefficients.shape}."
            )
        joint_dofs = [
            Joint.DICT_JOINT_TYPE_DOF[segment.joint.type]
            for segment in structure.segments
        ]
        link_dofs = []
        for segment in structure.segments:
            multiplier = 2 if segment.basis.type == "fourier" else 1
            offset = 1
            link_dofs.append(
                sum(
                    int(active) * (multiplier * int(order) + offset)
                    for active, order in zip(
                        segment.basis.strain_selector,
                        segment.basis.basis_order,
                        strict=True,
                    )
                )
            )
        required_max_dof = max(joint_dofs + link_dofs)
        max_dof = (
            int(self.link.stiffness.shape[-1])
            if structure.max_dof is None
            else structure.max_dof
        )
        if max_dof < required_max_dof:
            raise ValueError(
                f"max_dof={max_dof} is smaller than required DOF {required_max_dof}."
            )
        expected = (num_segments, max_dof, max_dof)
        for owner, value in (
            ("link.stiffness", self.link.stiffness),
            ("link.damping", self.link.damping),
            ("joint.stiffness", self.joint.stiffness),
            ("joint.damping", self.joint.damping),
        ):
            if value.shape != expected:
                raise ValueError(
                    f"{owner} must have shape {expected}, got {value.shape}."
                )
        for index, segment in enumerate(structure.segments):
            if segment.num_gauss_points < 5:
                raise ValueError(
                    f"GVS segment {index} requires at least 5 Gauss points."
                )
            if (
                structure.max_num_gauss_points is not None
                and structure.max_num_gauss_points < segment.num_gauss_points
            ):
                raise ValueError(
                    "max_num_gauss_points must cover every segment quadrature rule."
                )

    def validate_value_compatibility(self, structure: GVSStructure) -> None:
        """Validate eager geometry values against the static GVS layout."""
        for index, segment in enumerate(structure.segments):
            count = sum(segment.link.cross_section_profile_parameter_counts)
            active_coefficients = self.link.cross_section.coefficients[index, :count]
            if not bool(jnp.all(active_coefficients > 0.0)):
                raise ValueError(
                    "Active cross-section coefficients must be strictly positive "
                    f"for GVS segment {index}."
                )


def _as_strong_array(value: object) -> Array:
    """Keep dtype while removing weak typing that could retrace numeric updates."""
    array = jnp.asarray(value)
    return jnp.asarray(array, dtype=array.dtype)


class HelicoidalParams(GVSParams):
    """GVS parameters with independent lattice inertia and platform masses.

    ``link.stiffness`` and ``link.damping`` remain canonical generalized
    matrices. Cross-section geometry describes the rendering envelope;
    ``link.density`` is not used to infer lattice inertia in this system.

    Attributes:
        mass_per_length: Constant line mass per link, shape ``(N,)`` [kg/m].
        rotational_inertia_per_length: Principal rotational inertia about the
            backbone in local x/y/z order, shape ``(N, 3)`` [kg m].
        tip_mass: Point mass at each link tip, shape ``(N,)`` [kg]. Use zero
            for unloaded tips. Geometric connector links may have zero
            distributed inertia, with platform mass accounted for here.
    """

    mass_per_length: Array
    rotational_inertia_per_length: Array
    tip_mass: Array

    def __check_init__(self) -> None:
        for name in ("mass_per_length", "rotational_inertia_per_length", "tip_mass"):
            object.__setattr__(self, name, _as_strong_array(getattr(self, name)))
        self.validate_for_update()

    def _normalize_replacement(self, name: str, value: object) -> object:
        if name in ("mass_per_length", "rotational_inertia_per_length", "tip_mass"):
            return _as_strong_array(value)
        return super()._normalize_replacement(name, value)

    def validate_structure(self) -> None:
        super().validate_structure()
        count = self.link.length.shape[0]
        for name, shape in (
            ("mass_per_length", (count,)),
            ("rotational_inertia_per_length", (count, 3)),
            ("tip_mass", (count,)),
        ):
            if jnp.shape(getattr(self, name)) != shape:
                raise ValueError(f"{name} must have shape {shape}.")

    def validate_values(self) -> None:
        super().validate_values()
        for name in ("mass_per_length", "rotational_inertia_per_length", "tip_mass"):
            value = _as_strong_array(getattr(self, name))
            if not bool(jnp.all(jnp.isfinite(value) & (value >= 0))):
                raise ValueError(f"{name} must be finite and nonnegative.")


class HelicoidalSectionParams(BaseSystemParams):
    """Analytic effective sectional law for GVS projection.

    These are effective coefficients, not solid-rod material moduli. Every
    array has one row per GVS link, including transitions and rigid connectors.
    This object is input to ``with_sectional_law`` and is not stored as a second
    source of the model's canonical generalized matrices.

    Attributes:
        bending_compliance: ``(cA, cB)`` per link, shape ``(N, 2)``, defining
            ``EI(s) = 1 / (cA + cB * (L - s))``. Units are reciprocal N m²
            and reciprocal N m³. Set ``cB=0`` for constant EI.
        torsional_stiffness: GJ per link, shape ``(N,)`` [N m²].
        axial_stiffness: EA per link, shape ``(N,)`` [N].
        shear_stiffness: GA for both transverse axes, shape ``(N,)`` [N].
        damping_time_constant: Kelvin-Voigt time constant per link,
            shape ``(N,)`` [s]; zero disables material damping.
    """

    bending_compliance: Array
    torsional_stiffness: Array
    axial_stiffness: Array
    shear_stiffness: Array
    damping_time_constant: Array

    def __check_init__(self) -> None:
        for name in HelicoidalSectionParams.__annotations__:
            object.__setattr__(self, name, _as_strong_array(getattr(self, name)))
        self.validate_for_update()

    def _normalize_replacement(self, name: str, value: object) -> object:
        return _as_strong_array(value)

    def validate_structure(self) -> None:
        compliance = jnp.asarray(self.bending_compliance)
        if compliance.ndim != 2 or compliance.shape[1] != 2 or not compliance.shape[0]:
            raise ValueError("bending_compliance must have shape (num_links, 2).")
        for name in (
            "torsional_stiffness",
            "axial_stiffness",
            "shear_stiffness",
            "damping_time_constant",
        ):
            if jnp.shape(getattr(self, name)) != (compliance.shape[0],):
                raise ValueError(f"{name} must have one value per link.")

    def validate_values(self) -> None:
        if not bool(jnp.all(jnp.isfinite(self.bending_compliance))):
            raise ValueError("bending_compliance must be finite.")
        if not bool(jnp.all(self.bending_compliance[:, 0] > 0)):
            raise ValueError("bending_compliance cA must be positive.")
        for name in (
            "torsional_stiffness",
            "axial_stiffness",
            "shear_stiffness",
            "damping_time_constant",
        ):
            value = _as_strong_array(getattr(self, name))
            positive = value >= 0 if name == "damping_time_constant" else value > 0
            if not bool(jnp.all(jnp.isfinite(value) & positive)):
                raise ValueError(f"{name} must be finite and in its physical domain.")
