"""Typed dynamic parameters for the geometric variable-strain system."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, ensure_compile_time_eval

from soromox.systems.components import ContinuumLinkParams, JointParams, spatial_inertia
from soromox.systems.params import (
    BaseSoftRobotParams,
    BaseSystemParams,
    _contains_tracer,
)

if TYPE_CHECKING:
    from soromox.systems.gvs.structures import GVSStructure

__all__ = [
    "GVSParams",
    "HelicoidalBackboneParams",
    "HelicoidalHelixParams",
]


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
    """Endpoint sectional stiffnesses for one segment in its backbone material frame.

    The coefficients describe the equivalent rod's effective lattice response.
    Local x is longitudinal and y/z are transverse, with component order
    ``[twist, bend_y, bend_z, extension, shear_y, shear_z]``. Individual helical
    strands' properties must first be reduced to this backbone constitutive law.

    Compliance varies linearly over normalized reference backbone arc length
    ``u = s/L``, measured from the segment's proximal end:
    ``C(u) = (1-u)/stiffness_base + u/stiffness_tip``. The sectional stiffness
    is ``diag(1/C(u))``. Equal endpoint values give constant stiffness.
    All six endpoint entries must be finite and positive. Active strains are
    selected by the segment's strain basis.

    Pass this object to ``with_constitutive_params`` for Gauss projection.
    Reapplying it after a length change preserves the endpoint stiffnesses and
    stretches the profile over the new length. Numeric updates reuse compilation
    with fixed shapes and dtypes.

    Attributes:
        stiffness_base: Diagonal sectional stiffness at the segment's proximal
            end, shape ``(6,)``. Rotational entries have units N m²;
            translational entries have units N.
        stiffness_tip: Diagonal sectional stiffness at the segment's distal end,
            with the same shape, component order, frame, and units.
        damping_time_constant: Kelvin-Voigt time constant [s], either a scalar
            shared by all strains or shape ``(6,)`` in backbone material-frame
            order ``[twist, bend_y, bend_z, extension, shear_y, shear_z]``.
            Relates sectional viscous damping to sectional stiffness for each
            strain component: ``D_i(s) = tau_i * K_i(s)``. Sectional damping is
            projected onto the GVS basis using Gauss quadrature. Zero disables
            damping for that component.
        admissible_strains: Six static flags in native strain order. Active basis
            strains must be admissible. Helix conversion marks inadmissible shear
            entries false; their endpoint values are finite padding. Numeric
            replacements preserve these restrictions.
    """

    stiffness_base: Array
    stiffness_tip: Array
    damping_time_constant: Array
    admissible_strains: tuple[bool, ...] = eqx.field(static=True, default=(True,) * 6)

    def __check_init__(self) -> None:
        for name in ("stiffness_base", "stiffness_tip", "damping_time_constant"):
            object.__setattr__(self, name, _as_strong_array(getattr(self, name)))
        self.validate_for_update()

    def _normalize_replacement(self, name: str, value: object) -> object:
        if name == "admissible_strains":
            raise ValueError(
                "admissible_strains is fixed when constructing the parameters."
            )
        return _as_strong_array(value)

    @classmethod
    def from_compliance(
        cls,
        *,
        compliance: Array,
        compliance_gradient: Array,
        length: Array,
        damping_time_constant: Array,
        admissible_strains: tuple[bool, ...] = (True,) * 6,
    ) -> Self:
        """Convert an affine compliance law to endpoint stiffnesses.

        Evaluates ``C(s) = compliance + (L-s)*compliance_gradient`` at both
        ends. The supplied length is used for conversion; the returned
        object stores endpoint stiffnesses. Subsequent length changes preserve
        those values when the law is reprojected.

        Args:
            compliance: Distal backbone-frame compliance, shape ``(6,)``.
                Rotational entries have units 1/(N m²), translational entries 1/N.
            compliance_gradient: Compliance variation per unit reference distance
                from the distal end, shape ``(6,)``. Units are compliance
                units per metre, and ``dC/ds = -compliance_gradient``.
            length: Positive scalar reference backbone length [m].
            damping_time_constant: Effective backbone time constant [s], either
                a scalar or shape ``(6,)`` in native backbone strain order.
            admissible_strains: Static flags identifying admissible basis strains.

        Returns:
            Endpoint parameters reproducing the supplied law at this length.

        Raises:
            ValueError: If shapes are incompatible, length is invalid, compliance
                is not finite and positive throughout a link, or damping is invalid.
        """
        compliance = _as_strong_array(compliance)
        compliance_gradient = _as_strong_array(compliance_gradient)
        length = _as_strong_array(length)
        if compliance.shape != (6,):
            raise ValueError("compliance must have shape (6,).")
        if compliance_gradient.shape != compliance.shape:
            raise ValueError(
                "compliance_gradient must have the same shape as compliance."
            )
        if length.shape != ():
            raise ValueError("length must be scalar.")
        if not _contains_tracer((compliance, compliance_gradient, length)):
            with ensure_compile_time_eval():
                if not bool(jnp.all(jnp.isfinite(length) & (length > 0))):
                    raise ValueError("length must be finite and positive.")
                if not bool(jnp.all(jnp.isfinite(compliance_gradient))):
                    raise ValueError("compliance_gradient must be finite.")
                proximal = compliance + length * compliance_gradient
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
            stiffness_base=1 / (compliance + length * compliance_gradient),
            stiffness_tip=1 / compliance,
            damping_time_constant=damping_time_constant,
            admissible_strains=admissible_strains,
        )

    def validate_structure(self) -> None:
        if self.stiffness_base.shape != (6,):
            raise ValueError("stiffness_base must have shape (6,).")
        if self.stiffness_tip.shape != self.stiffness_base.shape:
            raise ValueError(
                "stiffness_tip must have the same shape as stiffness_base."
            )
        if self.damping_time_constant.shape not in ((), (6,)):
            raise ValueError("damping_time_constant must be scalar or have shape (6,).")

        if (
            not isinstance(self.admissible_strains, tuple)
            or len(self.admissible_strains) != 6
            or any(type(flag) is not bool for flag in self.admissible_strains)
        ):
            raise ValueError("admissible_strains must be a tuple of six booleans.")

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


class HelicoidalHelixParams(BaseSystemParams):
    """Helix-domain material and geometry for one soft helicoidal segment.

    Two equally spaced winding families share a constant strand cross-section
    normal to each strand's tangent. The segment backbone is straight along local
    x in its printed reference configuration. Each strand frame has x along its
    tangent, y along the projected outward radial direction, and z completing a
    right-handed frame. All lengths are in metres and angles in radians.

    ``to_backbone()`` linearizes the unloaded lattice, solves six unit-load cases at
    the soft segment tip relative to its fixed base, then fits the reference
    toolkit's affine bending-compliance and constant torsion/extension law. Both
    compliance measurement points are the soft segment ends.
    This eager preparation step is independent of the runtime GVS strain basis.
    Recompute it after material or geometry updates; prepared matrices can be
    installed with compilation reuse when structure/shapes/dtypes stay fixed.
    Run the sparse reduction eagerly, outside JAX transformations.

    Attributes:
        length: Soft segment's reference backbone length [m].
        radius_base: Helix centreline radius at the base [m].
        radius_tip: Helix centreline radius at the tip [m]; linear taper in between.
        num_turns: Number of revolutions each strand makes around the backbone
            over ``length``. One turn is 2π radians; positive fractional values
            are supported. The two families wind in opposite directions.
        density: Strand material mass density [kg/m³]. Strand volumes are summed
            additively, neglecting overlap at fused crossings.
        young_modulus: Strand material Young's modulus [Pa].
        shear_modulus: Strand material shear modulus [Pa].
        strand_area: Strand-normal sectional area [m²].
        strand_second_moments: Principal area moments about local strand y/z,
            shape ``(2,)`` [m⁴]. The principal directions follow the strand frame.
        strand_torsion_constant: Strand-normal Saint-Venant torsion constant [m⁴].
        strand_shear_factors: Shear correction factors along strand y/z,
            shape ``(2,)``; the reference uses 0.9 for both.
        helices_per_direction: Positive number of strands in each winding family.
        phase: Base azimuth of the first negative-winding strand [rad].
        winding_family_phase_offset: Base azimuth of the first positive-winding
            strand relative to the first negative-winding strand [rad]. For
            example, π/6 separates the two families by 30° at the segment base.
        fusion_stiffness_multiplier: Dimensionless multiplier of the stiffness
            of fused strand crossings, computed from material moduli and a
            square contact patch of side sqrt(strand_area). Larger values
            make the connections stiffer; 1 uses the contact-patch estimate.
        tip_attachment_stiffness_multiplier: Stiffness of each strand's
            attachment to the tip ring relative to a fused crossing. The
            reference uses 10; strand bases are rigidly attached.
        damping_time_constant: Effective backbone Kelvin-Voigt time constant [s],
            either a scalar or shape ``(6,)`` ordered as
            ``[twist, bend_y, bend_z, extension, shear_y, shear_z]`` in the
            backbone material frame. Relates sectional viscous damping to fitted
            backbone sectional stiffness: ``D_i(s) = tau_i * K_i(s)``. Passed
            through the helix reduction unchanged; sectional damping is then
            projected onto the GVS basis using Gauss quadrature.

    References:
        CharlesQueen (2026). helicoid-toolkit, spec/identify.py,
        lattice/assembly2.py and rod/equiv_rod.py, revision 76c659d.
        https://github.com/CharlesQueen/helicoid-toolkit

        Qin et al. (2026). Cosserat Modeling of Trimmed Helicoid Soft Arms
        with a Separated-Section Constitutive Law. arXiv:2609.25264.
        https://arxiv.org/abs/2609.25264
    """

    length: Array
    radius_base: Array
    radius_tip: Array
    num_turns: Array
    density: Array
    young_modulus: Array
    shear_modulus: Array
    strand_area: Array
    strand_second_moments: Array
    strand_torsion_constant: Array
    strand_shear_factors: Array
    helices_per_direction: int = eqx.field(static=True, default=3)
    phase: Array = 0.0
    winding_family_phase_offset: Array = np.pi / 6
    fusion_stiffness_multiplier: Array = 4.0
    tip_attachment_stiffness_multiplier: Array = 10.0
    damping_time_constant: Array = 0.0

    def __check_init__(self) -> None:
        for name in HelicoidalHelixParams.__annotations__:
            if name != "helices_per_direction":
                object.__setattr__(self, name, _as_strong_array(getattr(self, name)))
        self.validate_for_update()

    def _normalize_replacement(self, name: str, value: object) -> object:
        return value if name == "helices_per_direction" else _as_strong_array(value)

    @classmethod
    def circular(cls, *, strand_radius: float, **kwargs) -> Self:
        """Construct circular strand sections from their radius [m].

        Remaining keyword arguments specify geometry, material and fusion as in
        the constructor. Shear correction defaults to the reference's 0.9.
        """
        radius = float(strand_radius)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("strand_radius must be finite and positive.")
        moment = np.pi * radius**4 / 4
        kwargs.setdefault("strand_shear_factors", jnp.full(2, 0.9))
        return cls(
            strand_area=np.pi * radius**2,
            strand_second_moments=jnp.full(2, moment),
            strand_torsion_constant=2 * moment,
            **kwargs,
        )

    def validate_structure(self) -> None:
        if (
            isinstance(self.helices_per_direction, bool)
            or not isinstance(self.helices_per_direction, int)
            or self.helices_per_direction < 1
        ):
            raise ValueError("helices_per_direction must be a positive integer.")
        for name in HelicoidalHelixParams.__annotations__:
            if name == "helices_per_direction":
                continue
            if name == "damping_time_constant":
                if self.damping_time_constant.shape not in ((), (6,)):
                    raise ValueError(
                        "damping_time_constant must be scalar or have shape (6,)."
                    )
                continue
            shape = (
                (2,)
                if name in {"strand_second_moments", "strand_shear_factors"}
                else ()
            )
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}.")

    def validate_values(self) -> None:
        for name in HelicoidalHelixParams.__annotations__:
            if name == "helices_per_direction":
                continue
            values = np.asarray(getattr(self, name))
            if not np.isfinite(values).all():
                raise ValueError(f"{name} must be finite.")
            if name in {"phase", "winding_family_phase_offset"}:
                continue
            if name == "damping_time_constant":
                valid = (values >= 0).all()
            else:
                valid = (values > 0).all()
            if not valid:
                raise ValueError(
                    f"{name} must be {'nonnegative' if name == 'damping_time_constant' else 'positive'}."
                )

    def sectional_inertia(self, s: Array) -> Array:
        """Evaluate strand inertia per reference backbone length at positions ``s``.

        Sum strand-normal section inertias after rotating into the backbone
        frame and shifting to its centreline. The arc-length factor converts
        strand length to backbone length. Each section moves with its backbone
        material frame. Strand volumes are additive at crossings.

        Args:
            s: Reference backbone arc positions [m], scalar or array, in [0, length].

        Returns:
            Angular-first spatial inertia, shape ``(*s.shape, 6, 6)``. Rotational,
            coupling and translational blocks have units kg m, kg and kg/m.
            Supports JAX differentiation and compilation.
        """
        self.validate_for_update()
        s = jnp.asarray(s)
        slope = (self.radius_tip - self.radius_base) / self.length
        radius = self.radius_base + slope * s
        angles = (
            self.phase
            + 2
            * jnp.pi
            * jnp.arange(self.helices_per_direction)
            / self.helices_per_direction
        )
        phases = jnp.concatenate((angles, angles + self.winding_family_phase_offset))
        rates = (
            2
            * jnp.pi
            * self.num_turns
            / self.length
            * jnp.repeat(jnp.array([-1.0, 1.0]), self.helices_per_direction)
        )
        theta = phases + s[..., None] * rates
        cosine, sine = jnp.cos(theta), jnp.sin(theta)
        zero, one = jnp.zeros_like(theta), jnp.ones_like(theta)
        radius = radius[..., None]
        radial = jnp.stack((zero, cosine, sine), axis=-1)
        tangent = jnp.stack(
            (
                one,
                slope * cosine - radius * rates * sine,
                slope * sine + radius * rates * cosine,
            ),
            axis=-1,
        )
        arc_factor = jnp.linalg.norm(tangent, axis=-1)
        tangent = tangent / arc_factor[..., None]
        normal = radial - jnp.sum(radial * tangent, axis=-1, keepdims=True) * tangent
        normal = normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)
        frame = jnp.stack((tangent, normal, jnp.cross(tangent, normal)), axis=-1)
        iy, iz = self.strand_second_moments
        # Polar area moment Iy+Iz determines mass inertia; Saint-Venant's
        # torsion constant describes elasticity and can differ for noncircles.
        section = jnp.diag(jnp.stack((iy + iz, iy, iz)))
        inertia = (
            self.density
            * arc_factor[..., None, None]
            * (frame @ section @ jnp.swapaxes(frame, -1, -2))
        )
        mass = self.density * self.strand_area * arc_factor
        center = radius[..., None] * radial
        return jnp.sum(spatial_inertia(mass, center, inertia), axis=-3)

    def to_backbone(self) -> HelicoidalBackboneParams:
        """Fit endpoint stiffnesses and admissible strains for this segment.

        Run eagerly on the CPU after material or geometry updates. The returned
        ``admissible_strains`` identifies which strains the fitted law supports;
        use it when choosing the segment's strain basis. Inadmissible residual
        shear constrains both shear strains. An inadmissible bending, torsion,
        or extension fit raises ``ValueError``.
        """
        from soromox.systems.gvs._helicoidal_reduction import reduce_helix

        return reduce_helix(self).backbone

    def reduction_diagnostics(self) -> dict[str, Array | float]:
        """Run the reduction and return diagnostics for inspecting the fit.

        Returns:
            A dictionary containing ``tip_compliance``, the symmetric 6×6
            compliance at the soft tip relative to its fixed base, in native
            ``[rotation_x/y/z, translation_x/y/z]`` order. Columns correspond
            to tip moments/forces; block units are 1/(N m), 1/N and m/N.
            ``shear_compliance_residual`` is the mean transverse tip compliance
            after subtracting fitted bending [m/N]; values <= 1e-9 constrain
            shear. ``relative_asymmetry`` is the relative antisymmetric norm
            before symmetrization, measuring numerical solve accuracy.
        """
        from soromox.systems.gvs._helicoidal_reduction import reduce_helix

        result = reduce_helix(self)
        return {
            "tip_compliance": result.tip_compliance,
            "shear_compliance_residual": result.shear_compliance_residual,
            "relative_asymmetry": result.relative_asymmetry,
        }
