"""N6 helicoidal module model for the Instron validation (``instron_static.py``).

One file holds everything about the specimen and how the two experiments load
it; ``instron_static.py`` holds the procedure (Instron data, static solver,
Viser, outputs).

N6 module (``N6ModuleSpec``)
    12 strands (6 per winding family, one turn each) on a 108 mm lattice
    between two rigid 7.5 mm end rings (specimen height 123 mm). The strand
    section is the committed N6 solid rectangle (12 mm radial width,
    2.926792 mm circumferential thickness at the strand mid-radius, inner and
    outer radius 18 / 30 mm). Young's modulus 2.48 MPa is confirmed;
    ``nu = 0.485`` (specified range 0.48-0.49) sets ``G = E / 2.97``. Density
    is a labelled placeholder the library requires; the gravity-free statics
    never use it. The spec turns into the strand-level inputs of the helix
    reduction through :meth:`N6ModuleSpec.helix_params`; the reduction, not
    this file, produces the linear GVS stiffness. Nothing here prescribes an
    effective stiffness: that quantity is an output of the comparison.

Mountings
    ``build_upright_n6_module`` stands the module on a fixed base with the
    backbone along world +z (axial compression). ``build_horizontal_n6_module``
    clamps one ring and lays the backbone along world +x with the material z
    axis up (tendon bending). Both use the same Legendre strain basis, the same
    helix reduction and a fixed base joint.

Tendon bending (``TendonBendingConfig``, ``TendonPath``, ``TendonLoad``)
    A string is anchored at a hole near the outer radius of the free ring,
    runs as a straight chord through the module to the matching hole of the
    clamped ring, exits at the clamped end and is pulled upward by the Instron.
    ``TendonPath`` gives the hole positions, string nodes, path length and its
    shortening (the string travel). ``TendonLoad`` turns the tension ``T`` at
    the anchor into the wrench on the free ring, about the backbone tip in
    world axes::

        W = [r(q) x T t(q); T t(q)]

    with ``t(q)`` the unit chord from the free-ring hole toward the clamped-
    ring hole and ``r(q)`` the moment arm from the backbone tip to the hole.
    The string is ideal: inextensible, frictionless, tension-only.
    ``RingMonitor`` reports the internal chord length and whether the free
    ring has run into the clamped ring, the validity limit of the model.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from jax import Array

try:
    from examples.simulation.gvs.simulate_helicoidal import MODULE_BASIS_ORDER
except ModuleNotFoundError:  # run as a script rather than from the checkout root
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from examples.simulation.gvs.simulate_helicoidal import MODULE_BASIS_ORDER
from soromox.systems import (
    GVSSegment,
    Helicoidal,
    HelicoidalHelixParams,
    JointSpec,
    LinkSpec,
    StrainBasisSpec,
)
from soromox.utils.lie_algebra import so3

__all__ = [
    "DENSITY_PLACEHOLDER_KG_PER_M3",
    "HORIZONTAL_BASE_POSE",
    "N6ModuleSpec",
    "RingMonitor",
    "TendonBendingConfig",
    "TendonExitGeometry",
    "TendonLoad",
    "TendonPath",
    "build_horizontal_n6_module",
    "build_upright_n6_module",
]

#: Strand density [kg/m^3], required by the library to build the module.
#: Not measured; it enters only the distributed mass (self-weight when gravity
#: is enabled), which the gravity-free statics ignore.
DENSITY_PLACEHOLDER_KG_PER_M3 = 1000.0

#: Fixed mounting of the horizontal module: scalar-last unit quaternion plus
#: translation, ``[qx, qy, qz, qw, x, y, z]``. The identity maps the material
#: frame onto the world frame, so the backbone runs along world +x, the
#: material y axis is world +y and the material z axis is world +z (up).
#: The upright module uses the library default, which rotates the backbone
#: onto world +z.
HORIZONTAL_BASE_POSE: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# N6 module specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class N6ModuleSpec:
    """Inputs of the theoretical N6 module in SI units.

    Attributes:
        length: Height of the deformable lattice along the backbone [m].
            Confirmed; excludes the rigid end rings.
        end_ring_heights: Heights of the (lower, upper) rigid end rings [m],
            confirmed 7.5 mm each. They are rigid, carry no strain and never
            enter the backbone length.
        inner_radius: Radial position of the strands' inner face [m].
        outer_radius: Radial position of the strands' outer face [m].
        strands_per_family: Strands in each winding family (12 strands total).
        turns_per_strand: Revolutions of each strand over ``length``.
        strand_width: Radial extent of the solid rectangular strand section [m].
        strand_thickness: Circumferential extent of the strand section [m],
            evaluated at the strand mid-radius.
        phase: Base azimuth of the first negative-winding strand [rad].
        family_phase: Base azimuth of the first positive-winding strand
            relative to the first negative-winding strand [rad].
        young_modulus: Strand Young's modulus [Pa]. Confirmed.
        poisson_ratio: Sets the shear modulus ``G = E / (2 (1 + nu))``; the
            default 0.485 is the midpoint of the specified 0.48-0.49 range
            (``G = E / 2.97``).
        density: Strand density [kg/m^3]. PLACEHOLDER.
        fusion_stiffness_multiplier: Fused-crossing stiffness multiplier of the
            helix reduction (reference toolkit heuristic).
        tip_attachment_stiffness_multiplier: Tip-ring attachment multiplier of
            the helix reduction (reference toolkit heuristic).
        gravity: Gravitational acceleration magnitude used by the static
            model [m/s^2]. Zero neglects self-weight.
    """

    length: float = 0.108
    end_ring_heights: tuple[float, float] = (0.0075, 0.0075)
    inner_radius: float = 0.018
    outer_radius: float = 0.030
    strands_per_family: int = 6
    turns_per_strand: float = 1.0
    strand_width: float = 0.012
    strand_thickness: float = 2.926792e-3
    phase: float = 0.0
    family_phase: float = 0.0
    young_modulus: float = 2.48e6
    poisson_ratio: float = 0.485
    density: float = DENSITY_PLACEHOLDER_KG_PER_M3
    fusion_stiffness_multiplier: float = 4.0
    tip_attachment_stiffness_multiplier: float = 10.0
    gravity: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            "length",
            "inner_radius",
            "outer_radius",
            "turns_per_strand",
            "strand_width",
            "strand_thickness",
            "young_modulus",
            "density",
            "fusion_stiffness_multiplier",
            "tip_attachment_stiffness_multiplier",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("phase", "family_phase"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")
        if (
            isinstance(self.strands_per_family, bool)
            or not isinstance(self.strands_per_family, int)
            or self.strands_per_family < 1
        ):
            raise ValueError("strands_per_family must be a positive integer.")
        if not (-1.0 < self.poisson_ratio <= 0.5):
            raise ValueError("poisson_ratio must lie in (-1, 0.5].")
        if self.gravity < 0:
            raise ValueError("gravity must be >= 0.")
        if self.outer_radius <= self.inner_radius:
            raise ValueError("outer_radius must exceed inner_radius.")
        rings = tuple(float(v) for v in self.end_ring_heights)
        if len(rings) != 2 or any(not math.isfinite(v) or v <= 0 for v in rings):
            raise ValueError("end_ring_heights must be two positive heights.")
        object.__setattr__(self, "end_ring_heights", rings)
        if not math.isclose(
            self.outer_radius - self.inner_radius, self.strand_width, rel_tol=1e-9
        ):
            raise ValueError(
                "strand_width must equal outer_radius - inner_radius "
                f"({self.outer_radius - self.inner_radius:.6g} m)."
            )

    @property
    def centerline_radius(self) -> float:
        """Helix centreline radius [m], midway between the strand faces."""
        return 0.5 * (self.inner_radius + self.outer_radius)

    @property
    def end_ring_height(self) -> float:
        """Combined height of the two rigid end rings [m] (15 mm)."""
        return sum(self.end_ring_heights)

    @property
    def specimen_height(self) -> float:
        """Unloaded specimen height, lattice plus both rings [m] (123 mm)."""
        return self.length + self.end_ring_height

    @property
    def total_strands(self) -> int:
        return 2 * self.strands_per_family

    @property
    def shear_modulus(self) -> float:
        """Isotropic shear modulus [Pa]; ``E / 2.97`` for the default ``nu = 0.485``."""
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    def helix_params(self) -> HelicoidalHelixParams:
        """Strand-level inputs of the helix reduction (solid rectangular section).

        ``HelicoidalHelixParams.rectangular_hollow`` builds a SOLID rectangle
        despite its name: ``width`` lies along the strand's radial axis and
        ``height`` along its circumferential axis.
        """
        return HelicoidalHelixParams.rectangular_hollow(
            width=self.strand_width,
            height=self.strand_thickness,
            length=self.length,
            radius_base=self.centerline_radius,
            radius_tip=self.centerline_radius,
            num_turns=self.turns_per_strand,
            helices_per_direction=self.strands_per_family,
            phase=self.phase,
            winding_family_phase_offset=self.family_phase,
            density=self.density,
            young_modulus=self.young_modulus,
            shear_modulus=self.shear_modulus,
            fusion_stiffness_multiplier=self.fusion_stiffness_multiplier,
            tip_attachment_stiffness_multiplier=(
                self.tip_attachment_stiffness_multiplier
            ),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str | bytes) -> N6ModuleSpec:
        return cls(**json.loads(text))


# --------------------------------------------------------------------------- #
# Module construction
# --------------------------------------------------------------------------- #
def _build_module(
    helix: HelicoidalHelixParams,
    *,
    gravity: float,
    base_pose: tuple[float, ...] | None,
    num_gauss_points: int,
) -> Helicoidal:
    """One base-clamped GVS segment carrying the projected helix law.

    The strain basis reuses the helicoidal example's Legendre orders and the
    admissible strains reported by the helix reduction. The zero stiffness and
    damping placeholders of the link are replaced by the helix law inside
    ``Helicoidal.from_segments``. The fixed base joint locks all six rigid-body
    degrees of freedom of the base ring; the tip ring carries no constraint.
    """
    backbone = helix.to_backbone()
    selector = list(backbone.admissible_strains)
    dof = sum(
        order + 1
        for order, active in zip(MODULE_BASIS_ORDER, selector, strict=True)
        if active
    )
    segment = GVSSegment(
        link=LinkSpec.circular(
            length=float(helix.length),
            radius=float(helix.radius_base),
            density=float(helix.density),
            reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            stiffness=jnp.zeros((dof, dof)),
            damping=jnp.zeros((dof, dof)),
        ),
        joint=JointSpec(type="fixed"),
        basis=StrainBasisSpec(
            type="legendre", strain_selector=selector, basis_order=MODULE_BASIS_ORDER
        ),
        num_gauss_points=num_gauss_points,
    )
    kwargs: dict[str, Any] = {}
    if base_pose is not None:
        kwargs["base_pose"] = jnp.asarray(base_pose)
    return Helicoidal.from_segments(
        [segment],
        constitutive_params={0: helix},
        gravity=jnp.array([0.0, 0.0, -gravity]),
        backend="jax",
        **kwargs,
    )


def build_upright_n6_module(
    spec: N6ModuleSpec | None = None, *, num_gauss_points: int = 16
) -> tuple[N6ModuleSpec, HelicoidalHelixParams, Helicoidal]:
    """Module standing on a fixed base, backbone along world +z (compression)."""
    spec = spec or N6ModuleSpec()
    helix = spec.helix_params()
    robot = _build_module(
        helix, gravity=spec.gravity, base_pose=None, num_gauss_points=num_gauss_points
    )
    return spec, helix, robot


def build_horizontal_n6_module(
    spec: N6ModuleSpec | None = None,
    config: TendonBendingConfig | None = None,
) -> tuple[N6ModuleSpec, HelicoidalHelixParams, Helicoidal]:
    """Module clamped at one ring, backbone along world +x, material z up (bending)."""
    spec = spec or N6ModuleSpec()
    config = config or TendonBendingConfig()
    helix = spec.helix_params()
    robot = _build_module(
        helix,
        gravity=config.gravity,
        base_pose=HORIZONTAL_BASE_POSE,
        num_gauss_points=config.num_gauss_points,
    )
    return spec, helix, robot


# --------------------------------------------------------------------------- #
# Tendon bending configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TendonExitGeometry:
    """How the string leaves the clamped ring and reaches the Instron grip.

    CONFIRMED: the string exits through the clamped-ring hole near the clamped
    end and is pulled upward. NOT CONFIRMED: where exactly it is redirected and
    whether a pulley or guide exists. Everything below is therefore an explicit
    assumption that a measurement of the rig can replace.

    Attributes:
        direction: World unit vector along which the string runs from the
            redirect point to the Instron grip. Default: world +z (upward).
        run_length: Length of that external run in the reference configuration
            [m]. It adds a constant to the reference tendon length and sets
            where the string is drawn; it never changes with ``q``.
        redirect_offset: Displacement of the redirect point from the hole on
            the outer face of the clamped ring, in world axes [m]. Zero means
            the string turns at the hole edge itself.
        travel_ratio: Internal tendon shortening per unit crosshead travel.
            Unity (frictionless, inextensible string with a fixed redirect
            point). Any other value must be justified by the real exit
            geometry; it is exposed, not assumed.
    """

    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    run_length: float = 0.120
    redirect_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    travel_ratio: float = 1.0

    def __post_init__(self) -> None:
        direction = np.asarray(self.direction, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("direction must be a finite 3-vector.")
        norm = float(np.linalg.norm(direction))
        if norm <= 0:
            raise ValueError("direction must be nonzero.")
        object.__setattr__(self, "direction", tuple(float(v) for v in direction / norm))
        offset = np.asarray(self.redirect_offset, dtype=float)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("redirect_offset must be a finite 3-vector.")
        object.__setattr__(self, "redirect_offset", tuple(float(v) for v in offset))
        if not math.isfinite(self.run_length) or self.run_length < 0:
            raise ValueError("run_length must be finite and non-negative.")
        if not math.isfinite(self.travel_ratio) or self.travel_ratio <= 0:
            raise ValueError("travel_ratio must be finite and positive.")


@dataclass(frozen=True)
class TendonBendingConfig:
    """Everything the bending experiment adds on top of ``N6ModuleSpec``.

    Confirmed quantities are the ring outer radius (40 mm), the hole diameter
    (3.2 mm) and the radial gap between the ring edge and the hole edge
    (3.4 mm); the hole-centre offset follows as 40 - 3.4 - 1.6 = 35.0 mm.

    Attributes:
        tendon_offset: Radial distance from the module axis to the hole centre
            [m]. Must agree with the ring/hole geometry (checked).
        tendon_azimuth: Angular position of both holes in the material frame,
            measured from the material y axis toward the material z axis
            [rad]. With the horizontal mounting ``pi / 2`` places the tendon on
            the world +z (upper) side, which is the side that bends the free
            end upward when the string is pulled. Both rings share one azimuth
            in the undeformed configuration (holes aligned).
        ring_outer_radius: Outer radius of both end rings [m]. Confirmed.
        hole_diameter: Tendon-hole diameter [m]. Confirmed.
        ring_edge_gap: Radial gap from the ring outer edge to the nearest hole
            edge [m]. Confirmed.
        free_anchor_face: Face of the free ring where the string is anchored:
            ``"outer"`` (knot on the far face; the string runs through the
            full hole) or ``"lattice"`` (anchored at the lattice-side face).
            NOT CONFIRMED; it only changes the constant part of the length.
        exit: External exit geometry (see ``TendonExitGeometry``).
        gravity: Gravitational acceleration used when the module is built
            [m/s^2]. Zero keeps the gravity-free statics of the compression
            study; a positive value adds self-weight along world -z, across
            the horizontal backbone.
        num_gauss_points: Quadrature points of the GVS module.
    """

    tendon_offset: float = 0.035
    tendon_azimuth: float = math.pi / 2
    ring_outer_radius: float = 0.040
    hole_diameter: float = 3.2e-3
    ring_edge_gap: float = 3.4e-3
    free_anchor_face: Literal["outer", "lattice"] = "outer"
    exit: TendonExitGeometry = field(default_factory=TendonExitGeometry)
    gravity: float = 0.0
    num_gauss_points: int = 16

    def __post_init__(self) -> None:
        for name in ("tendon_offset", "ring_outer_radius", "hole_diameter", "ring_edge_gap"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(self.tendon_azimuth):
            raise ValueError("tendon_azimuth must be finite.")
        if not math.isfinite(self.gravity) or self.gravity < 0:
            raise ValueError("gravity must be finite and non-negative.")
        if (
            isinstance(self.num_gauss_points, bool)
            or not isinstance(self.num_gauss_points, int)
            or self.num_gauss_points < 1
        ):
            raise ValueError("num_gauss_points must be a positive integer.")
        if self.free_anchor_face not in ("outer", "lattice"):
            raise ValueError('free_anchor_face must be "outer" or "lattice".')
        if not isinstance(self.exit, TendonExitGeometry):
            raise TypeError("exit must be a TendonExitGeometry.")
        expected = self.ring_outer_radius - self.ring_edge_gap - self.hole_diameter / 2
        if not math.isclose(expected, self.tendon_offset, abs_tol=1e-9):
            raise ValueError(
                "tendon_offset must equal ring_outer_radius - ring_edge_gap - "
                f"hole_diameter / 2 = {expected:.6g} m (got {self.tendon_offset:.6g})."
            )
        if self.tendon_offset + self.hole_diameter / 2 >= self.ring_outer_radius:
            raise ValueError("The hole must lie inside the ring outer radius.")

    @property
    def hole_radius(self) -> float:
        return self.hole_diameter / 2

    def hole_offset_local(self) -> np.ndarray:
        """Hole-centre offset from the backbone in the material frame [m].

        The material x axis is the backbone tangent; the offset lies in the
        ring plane (material y-z plane).
        """
        return np.array(
            [
                0.0,
                self.tendon_offset * math.cos(self.tendon_azimuth),
                self.tendon_offset * math.sin(self.tendon_azimuth),
            ]
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str | bytes) -> TendonBendingConfig:
        data = json.loads(text)
        data["exit"] = TendonExitGeometry(**data.get("exit", {}))
        return cls(**data)


# --------------------------------------------------------------------------- #
# Tendon path: hole positions, string nodes, length and shortening
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TendonPath:
    """Straight-chord tendon between the two ring holes of a horizontal module.

    Nodes along the string, from the anchor to the Instron grip, with the
    material-frame offset ``r`` of the holes and the ring heights ``t_free``
    and ``t_clamped`` from the N6 specification:

    ======  ===========================================  ==================
    node    position                                      depends on ``q``
    ======  ===========================================  ==================
    anchor  free-ring hole on the chosen anchor face       yes (rigid ring)
    P1      free-ring hole, lattice-side opening           yes
    P2      clamped-ring hole, lattice-side opening        no (clamped)
    P3      clamped-ring hole, outer-face opening          no
    P4      redirect point (``P3 + exit.redirect_offset``) no
    grip    ``P4 + exit.run_length * exit.direction``      no
    ======  ===========================================  ==================

    Only the chord ``|P1(q) - P2|`` changes with the configuration; the hole
    runs are rigid-body segments and the external run is constant by the
    assumptions in ``TendonExitGeometry``. ``L(q)`` is the full path length;
    ``shortening(q) = L(q0) - L(q)`` is the string travel the Instron
    crosshead measures.

    The tendon is attached to the free ring, i.e. to the rigid tip frame of
    the GVS backbone, so all quantities come from the tip pose; no intermediate
    backbone station is touched.
    """

    robot: Helicoidal
    spec: N6ModuleSpec
    config: TendonBendingConfig

    def __post_init__(self) -> None:
        chord = float(self.internal_chord(self.reference_configuration()))
        object.__setattr__(self, "_reference_length", chord + self.constant_length())

    # -- rigid-body helpers --------------------------------------------------- #
    @property
    def hole_offset_local(self) -> Array:
        return jnp.asarray(self.config.hole_offset_local())

    @property
    def num_q(self) -> int:
        return self.robot.num_coordinates

    def reference_configuration(self) -> Array:
        return jnp.zeros(self.num_q)

    def tip_pose(self, q: Array) -> Array:
        """Pose of the free end of the lattice (material frame at s = L)."""
        return self.robot.forward_kinematics_tips(jnp.asarray(q))[-1]

    def base_pose(self) -> Array:
        """Pose of the clamped end of the lattice (material frame at s = 0)."""
        return self.robot.forward_kinematics(
            self.reference_configuration(), jnp.asarray(0.0)
        )

    def free_ring_pose(self, q: Array) -> Array:
        """Rigid free ring: frame at the ring mid-plane, following the tip."""
        g = self.tip_pose(q)
        shift = jnp.array([0.5 * self.spec.end_ring_heights[1], 0.0, 0.0])
        return g.at[:3, 3].set(g[:3, 3] + g[:3, :3] @ shift)

    def free_hole_opening(self, q: Array) -> Array:
        """P1: lattice-side opening of the free-ring hole [m, world]."""
        g = self.tip_pose(q)
        return g[:3, 3] + g[:3, :3] @ self.hole_offset_local

    def free_anchor(self, q: Array) -> Array:
        """Anchor of the string on the free ring [m, world]."""
        g = self.tip_pose(q)
        depth = (
            self.spec.end_ring_heights[1]
            if self.config.free_anchor_face == "outer"
            else 0.0
        )
        local = self.hole_offset_local + jnp.array([depth, 0.0, 0.0])
        return g[:3, 3] + g[:3, :3] @ local

    def clamped_hole_opening(self) -> Array:
        """P2: lattice-side opening of the clamped-ring hole [m, world]."""
        g = self.base_pose()
        return g[:3, 3] + g[:3, :3] @ self.hole_offset_local

    def clamped_hole_exit(self) -> Array:
        """P3: outer-face opening of the clamped-ring hole [m, world]."""
        g = self.base_pose()
        local = self.hole_offset_local - jnp.array(
            [self.spec.end_ring_heights[0], 0.0, 0.0]
        )
        return g[:3, 3] + g[:3, :3] @ local

    def redirect_point(self) -> Array:
        """P4: where the string turns toward the Instron [m, world]."""
        return self.clamped_hole_exit() + jnp.asarray(self.config.exit.redirect_offset)

    def grip_point(self) -> Array:
        """Instron grip in the reference configuration [m, world]."""
        exit = self.config.exit
        return self.redirect_point() + exit.run_length * jnp.asarray(exit.direction)

    def nodes(self, q: Array) -> Array:
        """All string nodes from the free anchor to the grip, shape (6, 3)."""
        return jnp.stack(
            [
                self.free_anchor(q),
                self.free_hole_opening(q),
                self.clamped_hole_opening(),
                self.clamped_hole_exit(),
                self.redirect_point(),
                self.grip_point(),
            ]
        )

    # -- length and shortening ------------------------------------------------ #
    def internal_chord(self, q: Array) -> Array:
        """Length of the straight internal segment P1(q) -> P2 [m]."""
        return jnp.linalg.norm(self.free_hole_opening(q) - self.clamped_hole_opening())

    def constant_length(self) -> float:
        """Rigid hole runs plus the external run [m]; independent of ``q``."""
        free_run = (
            self.spec.end_ring_heights[1]
            if self.config.free_anchor_face == "outer"
            else 0.0
        )
        exit = self.config.exit
        return float(
            free_run
            + self.spec.end_ring_heights[0]
            + np.linalg.norm(exit.redirect_offset)
            + exit.run_length
        )

    def length(self, q: Array) -> Array:
        """Effective tendon path length L(q) [m]."""
        return self.internal_chord(q) + self.constant_length()

    def reference_length(self) -> float:
        """L(q0): cached at construction [m]."""
        return self._reference_length

    def shortening(self, q: Array) -> Array:
        """L(q0) - L(q): the string travel; positive when the free end bends toward the clamp."""
        return self.reference_length() - self.length(q)

    # -- free-end pose -------------------------------------------------------- #
    def free_end_pose(self, q: Array) -> dict[str, np.ndarray | float]:
        """Rigid-body pose of the free ring relative to the reference."""
        g = np.asarray(self.free_ring_pose(q))
        g0 = np.asarray(self.free_ring_pose(self.reference_configuration()))
        relative = g0[:3, :3].T @ g[:3, :3]
        rotation_vector = np.asarray(so3.log(jnp.asarray(relative)))
        return {
            "position": g[:3, 3],
            "displacement": g[:3, 3] - g0[:3, 3],
            "rotation_vector": rotation_vector,
            "rotation_angle": float(np.linalg.norm(rotation_vector)),
        }


# --------------------------------------------------------------------------- #
# Chord string load on the free ring
# --------------------------------------------------------------------------- #
class TendonLoad:
    """Wrench of the string on the free ring, ``W = [r x F; F]`` with ``F = T t(q)``.

    The force follows the current straight chord from the free-ring hole to
    the clamped-ring hole (ideal point guide at the lattice-side opening); the
    moment arm follows the rotated ring. ``W`` is expressed about the backbone
    tip frame in world axes, so it enters the equilibrium as ``J_tip^T W`` with
    the inertial-frame tip Jacobian.
    """

    def __init__(self, path: TendonPath) -> None:
        self.path = path

    def direction(self, q: Array) -> Array:
        """Unit string direction at the anchor: toward the clamped-ring hole."""
        chord = self.path.clamped_hole_opening() - self.path.free_hole_opening(q)
        return chord / jnp.linalg.norm(chord)

    def moment_arm(self, q: Array) -> Array:
        """World vector from the backbone tip (Jacobian origin) to the hole."""
        g = self.path.tip_pose(q)
        return g[:3, :3] @ self.path.hole_offset_local

    def wrench(self, q: Array, tension: Array) -> Array:
        """``[M; F]`` about the backbone tip frame, world axes."""
        force = tension * self.direction(q)
        moment = jnp.cross(self.moment_arm(q), force)
        return jnp.concatenate((moment, force))


class RingMonitor:
    """Free ring vs clamped ring collision and internal chord length."""

    def __init__(self, path: TendonPath) -> None:
        self.path = path
        self.ring_radius = path.config.ring_outer_radius
        self.ring_inner = min(
            path.spec.inner_radius, path.config.tendon_offset - path.config.hole_radius
        )
        self.t0, self.t1 = path.spec.end_ring_heights
        ang = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
        local = []
        for xo in (-0.5 * self.t1, 0.5 * self.t1):
            for r in (self.ring_radius, self.ring_inner):
                local.append(
                    np.stack([np.full(ang.size, xo), r * np.cos(ang), r * np.sin(ang)], axis=1)
                )
        self._ring_local = np.concatenate(local)

    def evaluate(self, q: Array) -> dict[str, Any]:
        """Internal chord length [m] and whether the free ring overlaps the clamped ring."""
        path = self.path
        chord = float(
            np.linalg.norm(
                np.asarray(path.free_hole_opening(q)) - np.asarray(path.clamped_hole_opening())
            )
        )
        ring = np.asarray(path.free_ring_pose(q))
        points = ring[:3, 3] + self._ring_local @ ring[:3, :3].T
        radial = np.hypot(points[:, 1], points[:, 2])
        collision = bool(
            np.any(
                (points[:, 0] <= 0.0) & (points[:, 0] >= -self.t0) & (radial <= self.ring_radius)
            )
        )
        return {"internal_chord": chord, "ring_collision": collision}
