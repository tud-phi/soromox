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
    """Internal fitted law and diagnostics from the six-load reduction."""

    backbone: HelicoidalBackboneParams
    tip_compliance: jax.Array
    shear_compliance_residual: float
    relative_asymmetry: float


def _frame(s, phase, rate, length, radius_base, radius_tip):
    """Reference strand frame: tangent x, projected radial y, normal z."""
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
    s = np.linspace(0.0, length, count)
    slope = (radius_tip - radius_base) / length
    radius = radius_base + slope * s
    speed = np.sqrt(1 + slope**2 + (radius * rate) ** 2)
    arc = np.r_[0.0, np.cumsum((speed[1:] + speed[:-1]) * np.diff(s) / 2)]
    return s, arc


def _spans(params):
    """Generate crossings and reference frames using the source arc-length rules."""
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
    """Exact strain elimination in the two-element reference tangent of each span."""

    def one(frame, length, fusion):
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
    frames, lengths, connections, terminal, tip = _spans(params)
    E, G, area = map(
        float, (params.young_modulus, params.shear_modulus, params.strand_area)
    )
    section = np.r_[
        G * float(params.strand_torsion_constant),
        E * np.asarray(params.strand_second_moments),
        E * area,
        G * area * np.asarray(params.strand_shear_factors),
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
    """Fit the reference diagonal law in native x-longitudinal coordinates."""
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
    if _contains_tracer(params):
        raise ValueError(
            "Helix reduction is eager model preparation; call it outside JAX transformations."
        )
    params.validate_for_update()
    compliance, asymmetry = _tip_compliance(params)
    law, residual = _fit_backbone(
        compliance, float(params.length), float(params.damping_time_constant)
    )
    return _HelicoidalReduction(law, jnp.asarray(compliance), residual, asymmetry)
