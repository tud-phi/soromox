"""Shared segment kinematics against independent world endpoint evaluations."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from system_param_builders import pcs_params, planar_pcs_params

from soromox.actuation import RoutingSpanParams, StraightSpanRouting, ThreadlikeRouting
from soromox.actuation import straight_span as straight_module
from soromox.systems import (
    GVS,
    PCS,
    GVSSegment,
    JointSpec,
    LinkSpec,
    PCSStructure,
    PlanarPCS,
    PlanarPCSStructure,
    StrainBasisSpec,
)

jax.config.update("jax_enable_x64", True)


def _world_endpoint(robot, configuration, segment, at_end, offset):
    """Reference endpoint using independent full-backbone frame evaluations."""
    local_derivative = jnp.zeros((3, robot.num_coordinates))
    if not at_end and segment > 0:
        transform, spatial_jacobian, _ = robot.joint_kinematics(configuration, segment)
        offset = transform[:3, :3] @ offset + transform[:3, 3]
        local_derivative = (
            jnp.cross(spatial_jacobian[:3].T, offset).T + spatial_jacobian[3:]
        )
    position = jnp.sum(robot.segment_length[: segment + int(at_end)])
    pose, jacobian = robot.forward_kinematics_and_jacobian_inertialframe(
        configuration, position
    )
    internal = jacobian[..., jacobian.shape[-1] - robot.num_internal_dofs :]
    if robot.is_planar:
        cosine, sine = jnp.cos(pose[0]), jnp.sin(pose[0])
        rotation = jnp.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
        rotated = rotation @ offset
        point = jnp.array([pose[1], pose[2], 0]) + rotated
        derivative = jnp.pad(internal[1:], ((0, 1), (0, 0)))
        derivative = (
            derivative + jnp.array([-rotated[1], rotated[0], 0])[:, None] * internal[0]
        )
    else:
        rotation = pose[:3, :3]
        rotated = rotation @ offset
        point = pose[:3, 3] + rotated
        derivative = internal[3:] + jnp.cross(internal[:3].T, rotated).T
    derivative = jnp.pad(derivative, ((0, 0), (robot.num_base_coordinates, 0)))
    return point, derivative + rotation @ local_derivative


def _robot(kind, floating):
    """Build three unequal links with optional moving joints and variable strain."""
    if kind.startswith("gvs"):
        segments = [
            GVSSegment(
                link=LinkSpec.circular(
                    young_modulus=1e3,
                    shear_modulus=5e2,
                    density=1000.0,
                    length=length,
                    radius=0.03,
                    reference_strain=[0, 0, 0, 1, 0, 0],
                ),
                joint=joint,
                basis=StrainBasisSpec(
                    type="monomial",
                    strain_selector=[1] * 6,
                    basis_order=order,
                ),
                num_gauss_points=points,
            )
            for length, joint, order, points in zip(
                [0.1, 0.08, 0.12],
                [
                    JointSpec.revolute(axis="z"),
                    JointSpec.fixed(),
                    JointSpec.revolute(axis="y"),
                ],
                [[0, 1, 2, 1, 0, 0], [0] * 6, [1, 0, 1, 0, 0, 0]],
                [5, 7, 6],
            )
        ]
        return GVS.from_segments(
            segments,
            floating_base=floating,
            scale_rotational_basis_by_length=kind == "gvs_scaled",
        )
    planar = kind == "planar"
    dimension = 3 if planar else 6
    params = (planar_pcs_params if planar else pcs_params)(
        length=jnp.array([0.1, 0.08, 0.12]),
        radius=jnp.full(3, 0.03),
        density=jnp.full(3, 1000.0),
        young_modulus=jnp.full(3, 1e3),
        shear_modulus=jnp.full(3, 5e2),
        gravity=jnp.zeros(2 if planar else 3),
        damping_matrix=jnp.zeros((3 * dimension, 3 * dimension)),
    )
    if planar:
        return PlanarPCS(
            params, PlanarPCSStructure(num_gauss_points=3), floating_base=floating
        )
    return PCS(params, PCSStructure(num_gauss_points=3), floating_base=floating)


@pytest.mark.parametrize("kind", ["gvs", "gvs_scaled"])
def test_fixed_routing_retains_cached_strain_bases(kind, monkeypatch):
    """Fixed routing shares local strain algebra without sampling new bases."""
    robot = _robot(kind, False)
    configuration = jnp.linspace(-0.12, 0.18, robot.num_coordinates)
    routing = ThreadlikeRouting.linear(
        intercept=jnp.array([[0.0, 0.003, 0.001]]),
        slope=jnp.zeros((1, 3)),
        start_segment_index=(0,),
        end_segment_index=(2,),
    )
    expected = (
        robot._threadlike_path_lengths(configuration, routing),
        robot._threadlike_path_length_jacobian(configuration, routing),
    )
    calls = []
    evaluate = GVS._link_strain_and_basis

    def shared(host, *args):
        """Record use of the common local strain kernel."""
        calls.append("strain")
        return evaluate(host, *args)

    def forbidden(*args, **kwargs):
        """Reject basis resampling and global configuration basis assembly."""
        raise AssertionError("Fixed routing must retain its cached bases")

    monkeypatch.setattr(GVS, "_link_strain_and_basis", shared)
    monkeypatch.setattr(GVS, "_eval_B_segment", forbidden)
    monkeypatch.setattr(GVS, "strain_and_basis", forbidden)
    actual = jax.jit(
        lambda values: (
            robot._threadlike_path_lengths(values, routing),
            robot._threadlike_path_length_jacobian(values, routing),
        )
    )(configuration)
    assert calls
    for value, reference in zip(actual, expected):
        assert_allclose(value, reference, atol=1e-12)


def test_gvs_guides_share_cell_integration(monkeypatch):
    """Multiple guides share one vectorized link integration and boundary scan."""
    robot = _robot("gvs", False)
    offsets = jnp.array([[0, -0.003, 0], [0, 0.001, 0], [0, 0.004, 0]])
    routing = StraightSpanRouting(
        RoutingSpanParams(offsets, offsets, (0, 1, 2), (0, 0, 0), (1, 1, 1))
    )
    calls = []
    segments = GVS.relative_segment_kinematics
    boundaries = straight_module.segment_boundary_poses
    cell = GVS._magnus_jacobian_step_terms

    def integrate(host, configuration):
        """Record the shared segment evaluation."""
        calls.append("segments")
        return segments(host, configuration)

    def compose(base, links, joints):
        """Record the shared boundary traversal."""
        calls.append("boundaries")
        return boundaries(base, links, joints)

    def advance(host, *args):
        """Record the cell body traced once for the vectorized link scan."""
        calls.append("cell")
        return cell(host, *args)

    def forbidden(*args, **kwargs):
        """Reject independent endpoint integration or runtime autodiff."""
        raise AssertionError("Unexpected repeated integration or autodiff")

    monkeypatch.setattr(GVS, "relative_segment_kinematics", integrate)
    monkeypatch.setattr(straight_module, "segment_boundary_poses", compose)
    monkeypatch.setattr(GVS, "_magnus_jacobian_step_terms", advance)
    monkeypatch.setattr(GVS, "forward_kinematics_and_jacobian_inertialframe", forbidden)
    for name in ("grad", "jacfwd", "jacrev"):
        monkeypatch.setattr(jax, name, forbidden)
    solution = routing.solve_routing(
        robot, jnp.zeros(robot.num_coordinates), backend="jax"
    )
    assert bool(solution.converged)
    assert calls == ["segments", "cell", "boundaries"]
    assert_allclose(solution.path_lengths, 0.18, atol=1e-12)


@pytest.mark.parametrize("kind", ["planar", "spatial", "gvs", "gvs_scaled"])
@pytest.mark.parametrize("floating", [False, True])
def test_relative_spans_match_world_endpoints_and_derivatives(
    kind, floating, straight_backend
):
    """Cover shared intervals, upstream motion, floating bases, and padded GVS cells."""
    robot = _robot(kind, floating)
    internal = jnp.linspace(-0.12, 0.18, robot.num_internal_dofs)
    pose = (
        jnp.array([0.3, 0.4, -0.2])
        if robot.is_planar
        else jnp.array([0, 0, np.sin(0.15), np.cos(0.15), 0.4, -0.2, 0.1])
    )
    configuration = (
        robot.pack_configuration(internal, base_pose=pose) if floating else internal
    )
    starts = jnp.array(
        [[0, -0.003, 0], [0, 0.004, 0], [0, 0.002, 0], [0, 0, 0]], dtype=float
    )
    ends = starts.at[:, 1].add(0.001)
    routing = StraightSpanRouting(
        RoutingSpanParams(starts, ends, (0, 1, 2, 3), (0, 0, 1, 2), (1, 1, 1, 2))
    )
    solution = eqx.filter_jit(routing.solve_routing)(
        robot, configuration, backend=straight_backend
    )
    assert bool(solution.converged)
    analytical = routing.path_length_jacobian(robot, configuration, solution)

    def reference(values):
        """Evaluate separate global endpoints without the shared routing kernels."""
        positions, derivatives = [], []
        for index, (first, last) in enumerate(
            zip(routing.spans.start_segment_index, routing.spans.end_segment_index)
        ):
            start, start_jacobian = _world_endpoint(
                robot, values, first, False, starts[index]
            )
            end, end_jacobian = _world_endpoint(robot, values, last, True, ends[index])
            displacement = end - start
            length = jnp.linalg.norm(displacement)
            positions.append(jnp.stack([start, end]))
            derivatives.append(displacement @ (end_jacobian - start_jacobian) / length)
        return jnp.stack(positions), jnp.stack(derivatives)

    positions, derivatives = eqx.filter_jit(reference)(configuration)
    assert_allclose(
        solution.path_lengths,
        jnp.linalg.norm(positions[:, 1] - positions[:, 0], axis=-1),
        atol=1e-12,
    )
    assert_allclose(analytical, derivatives, atol=1e-11)
    for index, span in enumerate(solution.spans):
        assert_allclose(
            jnp.stack([span.start_position, span.end_position]),
            positions[index],
            atol=1e-12,
        )

    lengths = eqx.filter_jit(
        lambda values: (
            routing.solve_routing(robot, values, backend=straight_backend).path_lengths
        )
    )
    assert_allclose(analytical, jax.jacfwd(lengths)(configuration), atol=1e-11)
    assert_allclose(
        analytical.sum(axis=0),
        jax.grad(lambda values: lengths(values).sum())(configuration),
        atol=1e-11,
    )
    direction = jnp.linspace(-0.3, 0.2, robot.num_coordinates)
    finite_difference = (
        lengths(configuration + 1e-5 * direction)
        - lengths(configuration - 1e-5 * direction)
    ) / 2e-5
    assert_allclose(analytical @ direction, finite_difference, atol=1e-9)
    if floating:
        assert_allclose(analytical[:, : robot.num_base_coordinates], 0, atol=0)
    if kind.startswith("gvs"):
        assert_allclose(analytical[:2, robot.num_base_coordinates], 0, atol=0)
