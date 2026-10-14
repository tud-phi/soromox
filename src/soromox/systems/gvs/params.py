"""Typed dynamic parameters for the geometric variable-strain system."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self

import jax.numpy as jnp
from jax import Array, ensure_compile_time_eval

from soromox.systems.components import ContinuumLinkParams, JointParams
from soromox.systems.params import (
    BaseSoftRobotParams,
    BaseSystemParams,
    _contains_tracer,
)

if TYPE_CHECKING:
    from soromox.systems.gvs.structures import GVSStructure

__all__ = ["GVSParams", "HelicoidalBackboneParams"]


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


class HelicoidalBackboneParams(BaseSystemParams):
    """Endpoint sectional stiffnesses in the local backbone material frame.

    The coefficients describe the equivalent rod's effective lattice response.
    Local x is longitudinal and y/z are transverse, with component order
    ``[twist, bend_y, bend_z, extension, shear_y, shear_z]``. Individual helical
    strands' properties must first be reduced to this backbone constitutive law.

    Compliance varies linearly over normalized reference backbone arc length
    ``u = s/L``, measured from each link's proximal end:
    ``C(u) = (1-u)/stiffness_base + u/stiffness_tip``. The sectional stiffness
    is ``diag(1/C(u))``. Equal endpoint values give constant stiffness.
    All six endpoint entries must be finite and positive. Active strains are
    selected by the segment's strain basis.

    Pass this object to ``with_constitutive_law`` for Gauss projection.
    Reapplying it after a length change preserves the endpoint stiffnesses and
    stretches the profile over the new length. Numeric updates reuse compilation
    with fixed shapes and dtypes.

    Attributes:
        stiffness_base: Diagonal sectional stiffness at each link's proximal
            end, shape ``(num_links, 6)``. Rotational entries have units N m²;
            translational entries have units N.
        stiffness_tip: Diagonal sectional stiffness at each link's distal end,
            with the same shape, component order, frame, and units.
        damping_time_constant: Scalar Kelvin-Voigt time constant per link,
            shape ``(num_links,)`` [s]. It is frame independent and scales the
            backbone stiffness: ``D_section(u) = tau * K_section(u)`` and
            ``D_link = tau * K_link`` after projection. Zero disables damping.
    """

    stiffness_base: Array
    stiffness_tip: Array
    damping_time_constant: Array

    def __check_init__(self) -> None:
        for name in HelicoidalBackboneParams.__annotations__:
            object.__setattr__(self, name, _as_strong_array(getattr(self, name)))
        self.validate_for_update()

    def _normalize_replacement(self, name: str, value: object) -> object:
        return _as_strong_array(value)

    @classmethod
    def from_compliance(
        cls,
        *,
        compliance: Array,
        compliance_gradient: Array,
        lengths: Array,
        damping_time_constant: Array,
    ) -> Self:
        """Convert an affine compliance law to endpoint stiffnesses.

        Evaluates ``C(s) = compliance + (L-s)*compliance_gradient`` at both
        ends. The supplied lengths are used for conversion; the returned
        object stores endpoint stiffnesses. Subsequent length changes preserve
        those values when the law is reprojected.

        Args:
            compliance: Distal backbone-frame compliance, shape ``(num_links, 6)``.
                Rotational entries have units 1/(N m²), translational entries 1/N.
            compliance_gradient: Compliance variation per unit reference distance
                from the distal end, shape ``(num_links, 6)``. Units are compliance
                units per metre, and ``dC/ds = -compliance_gradient``.
            lengths: Positive reference backbone lengths, shape ``(num_links,)`` [m].
            damping_time_constant: Effective backbone time constants,
                shape ``(num_links,)`` [s].

        Returns:
            Endpoint parameters reproducing the supplied law at these lengths.

        Raises:
            ValueError: If shapes are incompatible, lengths are invalid, compliance
                is not finite and positive throughout a link, or damping is invalid.
        """
        compliance = _as_strong_array(compliance)
        compliance_gradient = _as_strong_array(compliance_gradient)
        lengths = _as_strong_array(lengths)
        if compliance.ndim != 2 or compliance.shape[1] != 6 or not compliance.shape[0]:
            raise ValueError("compliance must have shape (num_links, 6).")
        if compliance_gradient.shape != compliance.shape:
            raise ValueError(
                "compliance_gradient must have the same shape as compliance."
            )
        if lengths.shape != (compliance.shape[0],):
            raise ValueError("lengths must have shape (num_links,).")
        if not _contains_tracer((compliance, compliance_gradient, lengths)):
            with ensure_compile_time_eval():
                if not bool(jnp.all(jnp.isfinite(lengths) & (lengths > 0))):
                    raise ValueError("lengths must be finite and positive.")
                if not bool(jnp.all(jnp.isfinite(compliance_gradient))):
                    raise ValueError("compliance_gradient must be finite.")
                proximal = compliance + lengths[:, None] * compliance_gradient
                if not bool(
                    jnp.all(
                        jnp.isfinite(compliance)
                        & (compliance > 0)
                        & jnp.isfinite(proximal)
                        & (proximal > 0)
                    )
                ):
                    raise ValueError(
                        "Compliance must be finite and positive throughout each link."
                    )
        return cls(
            stiffness_base=1 / (compliance + lengths[:, None] * compliance_gradient),
            stiffness_tip=1 / compliance,
            damping_time_constant=damping_time_constant,
        )

    def validate_structure(self) -> None:
        if (
            self.stiffness_base.ndim != 2
            or self.stiffness_base.shape[1] != 6
            or not self.stiffness_base.shape[0]
        ):
            raise ValueError("stiffness_base must have shape (num_links, 6).")
        if self.stiffness_tip.shape != self.stiffness_base.shape:
            raise ValueError(
                "stiffness_tip must have the same shape as stiffness_base."
            )
        if self.damping_time_constant.shape != (self.stiffness_base.shape[0],):
            raise ValueError("damping_time_constant must contain one value per link.")

    def validate_values(self) -> None:
        for name in ("stiffness_base", "stiffness_tip"):
            values = getattr(self, name)
            if not bool(jnp.all(jnp.isfinite(values) & (values > 0))):
                raise ValueError(f"{name} must be finite and positive.")
        if not bool(
            jnp.all(
                jnp.isfinite(self.damping_time_constant)
                & (self.damping_time_constant >= 0)
            )
        ):
            raise ValueError("damping_time_constant must be finite and nonnegative.")
