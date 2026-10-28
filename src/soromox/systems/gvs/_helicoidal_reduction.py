"""Prepare a backbone law from the reference helix-lattice tangent.

Methodology adapted from CharlesQueen/helicoid-toolkit (Apache-2.0), revision
76c659d7900fb1a5ed01830a4fc1c6c6d473aa59: lattice/assembly2.py, spec/identify.py,
and rod/equiv_rod.py. See the upstream repository for license terms.

This implementation changes frames to SoRoMoX's x-longitudinal convention,
measures compliance at the soft segment ends, and eliminates each span's strain
coordinates analytically before assembling a sparse frame stiffness. It retains
the source's two constant-strain elements per crossing span, square fusion
footprint, and diagonal compliance fit. Each constant-strain element is integrated
exactly. Inadmissible fitted shear requires an inactive shear strain basis.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from soromox.systems.gvs.params import (
    HelicoidalBackboneParams,
    HelicoidalHelixParams,
)
from soromox.systems.params import _contains_tracer
from soromox.utils.lie_algebra import se3


@dataclass(frozen=True)
class _HelicoidalReduction:
    """Internal fitted law and diagnostics from the six-load reduction.

    Attributes:
        backbone: Effective endpoint stiffnesses, damping time constant, and
            admissible strain components for one soft segment.
        tip_compliance: Symmetric positive-definite matrix of shape ``(6, 6)``
            mapping tip moments/forces to small tip rotations/translations,
            in angular-first order and the reference backbone axes.
        shear_compliance_residual: Mean transverse tip compliance remaining
            after subtracting the fitted bending contribution [m/N].
        relative_asymmetry: Frobenius norm of the raw tip compliance minus its
            transpose, divided by the raw compliance norm, before symmetrization.
    """

    backbone: HelicoidalBackboneParams
    tip_compliance: jax.Array
    shear_compliance_residual: float
    relative_asymmetry: float


def _frame(s, phase, rate, length, radius_base, radius_tip):
    """Construct a reference strand frame along a linearly tapered helix.

    Args:
        s: Reference backbone arc length from the soft segment base [m].
        phase: Strand azimuth at the segment base [rad].
        rate: Signed azimuth change per unit backbone length [rad/m].
        length: Soft segment backbone length [m].
        radius_base: Strand centerline radius at the segment base [m].
        radius_tip: Strand centerline radius at the segment tip [m].

    Returns:
        NumPy homogeneous transform of shape ``(4, 4)`` from strand coordinates
        to the reference backbone base frame. Local x follows the strand
        tangent, y follows the outward radial direction projected normal to x,
        and z completes a right-handed frame. Translation is in metres.
    """
    slope = (radius_tip - radius_base) / length
    radius = radius_base + slope * s
    theta = phase + rate * s
    radial = np.array([0.0, np.cos(theta), np.sin(theta)])
    angular = np.array([0.0, -np.sin(theta), np.cos(theta)])
    tangent = np.array([1.0, 0.0, 0.0]) + slope * radial + radius * rate * angular
    tangent /= np.linalg.norm(tangent)
    normal = radial - (radial @ tangent) * tangent
    normal /= np.linalg.norm(normal)
    frame = np.eye(4)
    frame[:3, :3] = np.column_stack((tangent, normal, np.cross(tangent, normal)))
    frame[:3, 3] = np.array([s, 0.0, 0.0]) + radius * radial
    return frame


def _arc_table(length, radius_base, radius_tip, rate, count):
    """Tabulate strand arc length using trapezoidal integration.

    Args:
        length: Soft segment backbone length [m].
        radius_base: Strand centerline radius at the segment base [m].
        radius_tip: Strand centerline radius at the segment tip [m].
        rate: Signed azimuth change per unit backbone length [rad/m].
        count: Number of uniformly spaced backbone samples, including both ends.

    Returns:
        Pair of NumPy arrays of shape ``(count,)``: backbone positions and
        cumulative strand arc lengths, both in metres and starting at zero.
    """
    s = np.linspace(0.0, length, count)
    slope = (radius_tip - radius_base) / length
    radius = radius_base + slope * s
    speed = np.sqrt(1 + slope**2 + (radius * rate) ** 2)
    arc = np.r_[0.0, np.cumsum((speed[1:] + speed[:-1]) * np.diff(s) / 2)]
    return s, arc


def _spans(params):
    """Partition each strand at crossings and the soft segment ends.

    Args:
        params: Validated ``HelicoidalHelixParams`` for one soft segment.
            Opposite winding families share a linearly tapered radius profile.

    Returns:
        Tuple ``(frames, lengths, connections, terminal, tip_index)``. For N
        spans, ``frames`` has shape ``(N, 3, 4, 4)`` and contains strand-to-base
        transforms at each span's start, strand-arc midpoint, and end.
        ``lengths`` has shape ``(N,)`` and stores strand arc lengths [m].
        ``connections`` is a list of start/end node-index pairs; -1 denotes
        the fixed base, crossing nodes start at zero, and ``tip_index`` denotes
        the shared moving tip. ``terminal`` is a boolean array of shape
        ``(N,)`` identifying spans attached to that tip.
    """
    length, r0, r1 = map(float, (params.length, params.radius_base, params.radius_tip))
    n = params.helices_per_direction
    phase, offset, turns = map(
        float, (params.phase, params.winding_family_phase_offset, params.num_turns)
    )
    phase, offset = phase % (2 * np.pi), offset % (2 * np.pi)
    rate = 2 * np.pi * turns / length
    negative = phase + 2 * np.pi * np.arange(n) / n
    positive = negative + offset
    crossings = []
    for i, a in enumerate(negative):
        for j, b in enumerate(positive):
            # Solve a - rate*s = b + rate*s modulo 2*pi without a fixed turn cap.
            lower = (b - a) / (2 * np.pi)
            upper = lower + 2 * turns
            for k in range(int(np.floor(lower)) + 1, int(np.ceil(upper))):
                s = (a - b + 2 * np.pi * k) / (2 * rate)
                if 1e-12 * length < s < (1 - 1e-12) * length:
                    crossings.append((s, (a - rate * s) % (2 * np.pi), i, j))
    crossings.sort()
    tip_index = len(crossings)
    s_table, arc_table = _arc_table(length, r0, r1, rate, 1200)
    inverse_s, inverse_arc = _arc_table(length, r0, r1, rate, 2000)
    frames, lengths, connections, terminal = [], [], [], []
    for family, phases in enumerate((negative, positive)):
        for strand, angle in enumerate(phases):
            marks = [(0.0, -1)]
            marks += [
                (float(np.interp(s, s_table, arc_table)), node)
                for node, (s, _, i, j) in enumerate(crossings)
                if (i if family == 0 else j) == strand
            ]
            marks += [(arc_table[-1], tip_index)]
            for (a, start), (b, end) in zip(marks[:-1], marks[1:], strict=True):
                samples = np.interp([a, (a + b) / 2, b], inverse_arc, inverse_s)
                frames.append(
                    [
                        _frame(
                            s, angle, (-1 if family == 0 else 1) * rate, length, r0, r1
                        )
                        for s in samples
                    ]
                )
                lengths.append(b - a)
                connections.append((start, end))
                terminal.append(end == tip_index)
    return (
        np.asarray(frames),
        np.asarray(lengths),
        connections,
        np.asarray(terminal),
        tip_index,
    )


@jax.jit
def _condensed_spans(frames, lengths, section, fusions):
    """Eliminate the two constant-strain elements within each reference span.

    Args:
        frames: Strand-to-base transforms at span starts, arc midpoints, and
            ends, shape ``(N, 3, 4, 4)``.
        lengths: Strand arc length of each span [m], shape ``(N,)``.
        section: Diagonal strand sectional stiffness, shape ``(6,)``, ordered
            as twist, two bends, extension, and two shears. Rotational entries
            have units N m²; translational entries have units N.
        fusions: Diagonal connection stiffness at each span's distal end,
            shape ``(N, 6)``, expressed in its strand frame. Rotational entries
            have units N m/rad; translational entries have units N/m.

    Returns:
        Condensed stiffness blocks of shape ``(N, 6, 6)`` acting on differences
        between endpoint spatial displacement coordinates expressed about the
        reference backbone base origin, in angular-first order. Each block
        includes strand deformation and distal connection compliance.
    """

    def one(frame, length, fusion):
        """Condense one span and express its stiffness about the base origin.

        Args:
            frame: Three strand-to-base transforms, shape ``(3, 4, 4)``.
            length: Strand arc length of the span [m].
            fusion: Six distal connection stiffnesses in the end strand frame.

        Returns:
            Angular-first stiffness block of shape ``(6, 6)`` for the span.
        """
        first = jnp.linalg.inv(frame[0]) @ frame[1]
        second = jnp.linalg.inv(frame[1]) @ frame[2]
        # Body Jacobians of exp(xi*h) are h*J_right(xi*h).
        j1 = se3.adjoint_inverse(second) @ se3.left_jacobian(
            -se3.log(first, 1e-9), 1e-9
        )
        j2 = se3.left_jacobian(-se3.log(second, 1e-9), 1e-9)
        compliance = (length / 2) * (
            (j1 / section) @ j1.T + (j2 / section) @ j2.T
        ) + jnp.diag(1 / fusion)
        transport = se3.adjoint_inverse(frame[2])
        return transport.T @ jnp.linalg.solve(compliance, transport)

    return jax.vmap(one)(frames, lengths, fusions)


def _tip_compliance(params):
    """Solve six unit-load cases for the unloaded lattice's soft-segment tip.

    Strand bases are fixed. Fused crossings and strand attachments to the tip
    ring have finite connection stiffness. The sparse reference tangent is
    assembled and solved in float64 on the CPU.

    Args:
        params: Validated ``HelicoidalHelixParams`` specifying strand geometry,
            elastic section properties, and connection stiffness multipliers.

    Returns:
        Pair ``(compliance, asymmetry)``. ``compliance`` is a symmetric NumPy
        array of shape ``(6, 6)`` mapping moments [N m] and forces [N] applied
        at the soft tip to small rotations [rad] and translations [m] there.
        Both use angular-first order and the reference backbone axes.
        ``asymmetry`` is the relative Frobenius norm of the antisymmetric part
        computed as ``norm(C - C.T) / norm(C)`` before symmetrization.

    Raises:
        ValueError: If the tangent is singular, the displacement is nonfinite,
            the relative solve residual exceeds 1e-6, or the symmetrized tip
            compliance is not positive definite.
    """
    frames, lengths, connections, terminal, tip = _spans(params)
    properties = params.strand_cross_section_properties
    E, G, area = map(
        float, (params.young_modulus, params.shear_modulus, properties.area)
    )
    section = np.r_[
        G * float(properties.torsion_constant),
        E * np.asarray(properties.second_moments),
        E * area,
        G * np.asarray(properties.shear_areas),
    ]
    side = np.sqrt(area)
    fusion = (
        float(params.fusion_stiffness_multiplier)
        / side
        * np.array(
            [
                G * 0.14 * side**4,
                E * side**4 / 12,
                E * side**4 / 12,
                E * area,
                G * area,
                G * area,
            ]
        )
    )
    fusions = (
        fusion[None, :]
        * np.where(terminal, float(params.tip_attachment_stiffness_multiplier), 1.0)[
            :, None
        ]
    )
    # Preparation always solves in float64; it does not change the user's global
    # JAX precision setting. Output laws use the normal caller dtype policy.
    with jax.enable_x64(), jax.default_device(jax.devices("cpu")[0]):
        blocks = np.asarray(
            _condensed_spans(
                jnp.asarray(frames),
                jnp.asarray(lengths),
                jnp.asarray(section),
                jnp.asarray(fusions),
            )
        )
    rows, cols, values = [], [], []
    for block, (start, end) in zip(blocks, connections, strict=True):
        for a, sign_a in ((start, 1), (end, -1)):
            if a < 0:
                continue
            for b, sign_b in ((start, 1), (end, -1)):
                if b < 0:
                    continue
                rows.extend(np.repeat(6 * a + np.arange(6), 6))
                cols.extend(np.tile(6 * b + np.arange(6), 6))
                values.extend((sign_a * sign_b * block).ravel())
    size = 6 * (tip + 1)
    tangent = sparse.coo_matrix((values, (rows, cols)), shape=(size, size)).tocsc()
    tangent = (tangent + tangent.T) / 2
    # World-twist translations must be evaluated at the load/measurement point.
    length = float(params.length)
    point_map = np.eye(6)
    point_map[4, 2], point_map[5, 1] = length, -length
    loads = np.zeros((size, 6))
    loads[-6:] = point_map.T
    try:
        displacement = splu(tangent).solve(loads)
    except RuntimeError as exc:
        raise ValueError(
            "Helix reference tangent is singular; check geometry and fusion stiffness."
        ) from exc
    residual = np.linalg.norm(tangent @ displacement - loads) / np.linalg.norm(loads)
    if not np.isfinite(displacement).all() or residual > 1e-6:
        raise ValueError(
            f"Helix unit-load solves are ill-conditioned (relative residual {residual:.3g})."
        )
    compliance = point_map @ displacement[-6:]
    asymmetry = np.linalg.norm(compliance - compliance.T) / np.linalg.norm(compliance)
    compliance = (compliance + compliance.T) / 2
    if np.linalg.eigvalsh(compliance).min() <= 0:
        raise ValueError("Helix tip compliance must be positive definite.")
    return compliance, float(asymmetry)


def _fit_backbone(compliance, length, damping_time_constant):
    """Fit the reference diagonal backbone law to soft-segment tip compliance.

    The fit averages the two bending directions and uses their rotation/force
    coupling to determine affine sectional bending compliance. Torsion,
    extension, and shear sectional compliances are constant. The transverse
    compliance left after subtracting bending determines shear admissibility;
    a residual at or below 1e-9 m/N deactivates both shear components.

    Args:
        compliance: Tip compliance matrix of shape ``(6, 6)``, in angular-first
            order and reference backbone axes with x longitudinal. It maps tip
            moments/forces [N m, N] to small rotations/translations [rad, m].
        length: Soft segment backbone length [m].
        damping_time_constant: Backbone Kelvin-Voigt time constant [s], scalar
            or shape ``(6,)`` in native backbone strain order. Passed through
            to scale the fitted sectional stiffness before damping projection.

    Returns:
        Pair ``(law, residual)`` containing ``HelicoidalBackboneParams`` with
        endpoint stiffnesses and admissible strains, and the residual transverse
        tip compliance [m/N]. Inactive shear entries contain positive padding
        values; admissibility flags enforce their exclusion from the strain basis.

    Raises:
        ValueError: If fitted bending compliance is nonpositive at either end
            or the resulting backbone parameters fail validation.
    """
    bending = (compliance[1, 1] + compliance[2, 2]) / 2
    cross = (abs(compliance[1, 5]) + abs(compliance[2, 4])) / 2
    transverse = (compliance[4, 4] + compliance[5, 5]) / 2
    a, b = np.linalg.solve(
        [[length, length**2 / 2], [length**2 / 2, length**3 / 3]], [bending, cross]
    )
    if min(a, a + length * b) <= 0:
        raise ValueError(
            "The affine bending-compliance fit is nonpositive at a segment end."
        )
    residual = float(transverse - (a * length**3 / 3 + b * length**4 / 4))
    shear_active = residual > 1e-9
    distal = np.array(
        [
            compliance[0, 0] / length,
            a,
            a,
            compliance[3, 3] / length,
            residual / length if shear_active else 1.0,
            residual / length if shear_active else 1.0,
        ]
    )
    gradient = np.array([0.0, b, b, 0.0, 0.0, 0.0])
    law = HelicoidalBackboneParams.from_compliance(
        compliance=distal,
        compliance_gradient=gradient,
        length=length,
        damping_time_constant=damping_time_constant,
        admissible_strains=(
            True,
            True,
            True,
            True,
            bool(shear_active),
            bool(shear_active),
        ),
    )
    return law, residual


def reduce_helix(params: HelicoidalHelixParams) -> _HelicoidalReduction:
    """Prepare a diagonal backbone law and diagnostics from strand parameters.

    This eager CPU preparation follows the reference toolkit's six-unit-load
    fit about the unloaded configuration. Recompute the reduction after changing
    strand geometry, elastic properties, or connection stiffness.

    Args:
        params: Helix-domain inputs for one soft segment. Geometry and sectional
            properties describe the individual strands; the damping time
            constant applies to the resulting backbone stiffness.

    Returns:
        Fitted backbone parameters, full soft-tip compliance, residual shear
        compliance, and relative compliance asymmetry in a
        ``_HelicoidalReduction``. Returned JAX arrays use the caller's dtype policy.

    Raises:
        ValueError: If inputs contain JAX tracers, parameter validation fails,
            the lattice solve fails its residual or positivity checks, or the
            fitted backbone law is inadmissible. A nonpositive shear residual
            is represented through inactive shear components in the fitted law.
    """
    if _contains_tracer(params):
        raise ValueError(
            "Helix reduction is eager model preparation; call it outside JAX transformations."
        )
    params.validate_for_update()
    compliance, asymmetry = _tip_compliance(params)
    law, residual = _fit_backbone(
        compliance, float(params.length), params.damping_time_constant
    )
    return _HelicoidalReduction(law, jnp.asarray(compliance), residual, asymmetry)
