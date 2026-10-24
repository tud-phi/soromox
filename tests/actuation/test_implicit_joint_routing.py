"""Prescribed joint guides and analytical routing derivatives on moving hosts."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from system_param_builders import pcs_params, planar_pcs_params

from soromox.actuation import (
    CircularLumenParams,
    PiecewiseLinearRoutingDiscretization,
    RoutingSpanParams,
    ShortestPathRouting,
    ShortestPathRoutingParams,
    StraightSpanRouting,
)
from soromox.actuation._routing_geometry import guide_kinematics
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


def _joint_robot(kind="revolute", floating=False):
    """Build two straight hollow links with an intervening joint.

    Args:
        kind: Joint family; revolute uses z, prismatic uses x.
        floating: Whether to expose a runtime base pose.

    Returns:
        GVS host with one bending coordinate per link and the selected joint.
    """
    link = LinkSpec.circular(
        young_modulus=1e3,
        shear_modulus=5e2,
        density=1000.0,
        length=0.1,
        radius=0.03,
        reference_strain=[0, 0, 0, 1, 0, 0],
    )
    basis = StrainBasisSpec(
        type="monomial", strain_selector=[0, 0, 1, 0, 0, 0], basis_order=[0] * 6
    )
    return GVS.from_segments(
        [
            GVSSegment(
                link=link, joint=JointSpec.fixed(), basis=basis, num_gauss_points=5
            ),
            GVSSegment(
                link=link,
                joint=JointSpec(
                    type=kind,
                    axis="z" if kind == "revolute" else "x",
                    pitch=0.01 if kind == "helical" else 0.0,
                ),
                basis=basis,
                num_gauss_points=5,
            ),
        ],
        gravity=jnp.zeros(3),
        floating_base=floating,
        backend="jax",
    )


def _guided_routing(offset=0.0, strategy=ShortestPathRouting):
    """Build one tendon with guides at each end of two sections.

    Args:
        offset: Local y offset of all guides in metres.
        strategy: Routing strategy class to construct.

    Returns:
        Shortest path routing with two spans and one joint connection.
    """
    offsets = jnp.array([[0, offset, 0], [0, offset, 0]], dtype=float)
    if strategy is StraightSpanRouting:
        return strategy(RoutingSpanParams(offsets, offsets, (0, 0), (0, 1), (0, 1)))
    return ShortestPathRouting(
        ShortestPathRoutingParams(
            RoutingSpanParams(offsets, offsets, (0, 0), (0, 1), (0, 1)),
            CircularLumenParams.constant(0.02, 2),
        ),
        PiecewiseLinearRoutingDiscretization(elements_per_segment=2),
    )


@pytest.mark.parametrize("offset", [0.0, 0.004])
@pytest.mark.parametrize("angle", [-0.4, 0.4])
@pytest.mark.parametrize("strategy", [ShortestPathRouting, StraightSpanRouting])
def test_revolute_guide_lengths_and_analytical_jacobians(offset, angle, strategy):
    """Check centered and offset guides against the rotational chord formula."""
    robot = _joint_robot()
    routing = _guided_routing(offset, strategy)
    configuration = jnp.array([0, angle, 0], dtype=float)
    solve = eqx.filter_jit(lambda values: routing.solve_routing(robot, values))
    solution = solve(configuration)
    assert bool(solution.converged)
    assert len(solution.guides) == 1
    assert bool(solution.guides[0].differentiable)
    chord = 2 * offset * abs(np.sin(angle / 2))
    assert_allclose(solution.path_lengths, [0.2 + chord], atol=1e-9)
    assert_allclose(solution.guides[0].length, chord, atol=1e-12)
    jacobian = routing.path_length_jacobian(robot, configuration, solution)
    expected = offset * np.cos(angle / 2) * np.sign(angle)
    assert_allclose(jacobian[0, 1], expected, atol=1e-10)
    numerical = (
        solve(configuration.at[1].add(1e-5)).path_lengths
        - solve(configuration.at[1].add(-1e-5)).path_lengths
    ) / 2e-5
    assert_allclose(jacobian[:, 1], numerical, atol=1e-9)
    reverse = jax.grad(lambda values: solve(values).path_lengths.sum())(configuration)
    forward = jax.jacfwd(lambda values: solve(values).path_lengths)(configuration)
    assert_allclose(jacobian[0], reverse, atol=1e-8)
    assert_allclose(jacobian, forward, atol=1e-8)
    fractions = jnp.array([0.0, 0.5, 1.0])
    points = routing.span_positions(robot, configuration, solution, 1, fractions)
    expected_points = jnp.stack(
        (
            0.1 + 0.1 * fractions * np.cos(angle) - offset * np.sin(angle),
            0.1 * fractions * np.sin(angle) + offset * np.cos(angle),
            jnp.zeros_like(fractions),
        ),
        axis=-1,
    )
    base = robot.forward_kinematics(configuration, 0.0)
    expected_points = expected_points @ base[:3, :3].T + base[:3, 3]
    assert_allclose(points, expected_points, atol=1e-9)


@pytest.mark.parametrize("strategy", [ShortestPathRouting, StraightSpanRouting])
def test_collapsed_offset_connection_is_reported_without_poisoning_centered_guides(
    strategy,
):
    """A closing chord is nonsmooth; a guide fixed at the pivot is smooth."""
    robot = _joint_robot()
    configuration = jnp.zeros(robot.num_coordinates)
    for offset in (0.0, 0.004):
        routing = _guided_routing(offset, strategy)
        solve = eqx.filter_jit(
            lambda values, routing=routing: routing.solve_routing(robot, values)
        )
        solution = solve(configuration)
        assert bool(solution.converged)
        assert_allclose(solution.path_lengths, [0.2], atol=1e-9)
        assert_allclose(solution.guides[0].length, 0)
        gradient = routing.path_length_jacobian(robot, configuration, solution)
        reverse = jax.grad(
            lambda values, solve=solve: solve(values).path_lengths.sum()
        )(configuration)
        forward = jax.jacfwd(lambda values, solve=solve: solve(values).path_lengths)(
            configuration
        )
        if offset == 0:
            assert bool(solution.guides[0].differentiable)
            assert_allclose(gradient, 0, atol=1e-10)
            assert_allclose(reverse, 0, atol=1e-10)
            assert_allclose(forward, 0, atol=1e-10)
        else:
            assert not bool(solution.guides[0].differentiable)
            assert bool(jnp.all(jnp.isnan(gradient)))
            assert jnp.isnan(reverse[1])
            assert jnp.isnan(forward[0, 1])


@pytest.mark.parametrize("offset", [0.0, 0.004])
@pytest.mark.parametrize("displacement", [-0.012, 0.012])
@pytest.mark.parametrize("strategy", [ShortestPathRouting, StraightSpanRouting])
def test_prismatic_guides_include_translation_and_joint_force(
    offset, displacement, strategy
):
    """Axial joint displacement contributes its absolute value to cable length."""
    robot = _joint_robot("prismatic")
    routing = _guided_routing(offset, strategy)
    configuration = jnp.array([0, displacement, 0], dtype=float)
    solution = eqx.filter_jit(routing.solve_routing)(robot, configuration)
    assert bool(solution.converged)
    assert_allclose(solution.path_lengths, [0.212], atol=1e-9)
    assert_allclose(
        routing.path_length_jacobian(robot, configuration, solution)[0, 1],
        np.sign(displacement),
        atol=1e-12,
    )


@pytest.mark.parametrize("kind", ["spherical", "free", "helical"])
def test_joint_guide_kinematics_matches_autodiff(kind):
    """Check the analytical joint endpoint derivative for multiple joint types."""
    robot = _joint_robot(kind)
    configuration = jnp.linspace(0.01, 0.1, robot.num_coordinates)
    upstream = jnp.array([0.0, 0.004, 0.001])
    downstream = jnp.array([0.0, -0.002, 0.003])
    displacement, analytical, coincident = guide_kinematics(
        robot, configuration, 1, upstream, downstream
    )
    numerical = jax.jacfwd(
        lambda values: guide_kinematics(robot, values, 1, upstream, downstream)[0]
    )(configuration)
    assert bool(jnp.all(jnp.isfinite(displacement)))
    assert not bool(coincident)
    assert_allclose(analytical, numerical, atol=1e-10, rtol=1e-8)


def test_unsplit_moving_joint_requires_guides():
    """Reject a free span through a joint whose passage is unspecified."""
    routing = ShortestPathRouting(
        ShortestPathRoutingParams(
            RoutingSpanParams(jnp.zeros((1, 3)), jnp.zeros((1, 3)), (0,), (0,), (1,)),
            CircularLumenParams.constant(0.02, 2),
        )
    )
    robot = _joint_robot()
    with pytest.raises(ValueError, match="prescribed guides"):
        routing.solve_routing(robot, jnp.zeros(robot.num_coordinates))


@pytest.mark.parametrize("kind", ["planar", "spatial", "gvs"])
@pytest.mark.parametrize("strategy", [ShortestPathRouting, StraightSpanRouting])
def test_floating_base_routing_invariance_and_velocity_force_dimensions(kind, strategy):
    """Internal routing is invariant to base pose and yields zero base wrench."""
    if kind == "gvs":
        robot = _joint_robot(floating=True)
        routing = _guided_routing(0.004)
        internal = jnp.array([0.2, 0.4, -0.1])
    else:
        factory = planar_pcs_params if kind == "planar" else pcs_params
        dimension = 3 if kind == "planar" else 6
        params = factory(
            length=jnp.array([0.1]),
            radius=jnp.array([0.03]),
            density=jnp.array([1000.0]),
            young_modulus=jnp.array([1e3]),
            shear_modulus=jnp.array([5e2]),
            gravity=jnp.zeros(2 if kind == "planar" else 3),
            damping_matrix=jnp.zeros((dimension, dimension)),
        )
        robot = (
            PlanarPCS(
                params, PlanarPCSStructure(num_gauss_points=3), floating_base=True
            )
            if kind == "planar"
            else PCS(params, PCSStructure(num_gauss_points=3), floating_base=True)
        )
        routing = ShortestPathRouting(
            ShortestPathRoutingParams(
                RoutingSpanParams(
                    jnp.zeros((1, 3)), jnp.array([[0, 0.002, 0]]), (0,), (0,), (0,)
                ),
                CircularLumenParams.constant(0.02, 1),
            ),
            PiecewiseLinearRoutingDiscretization(elements_per_segment=2),
        )
        internal = (
            jnp.zeros(robot.num_internal_dofs).at[0 if kind == "planar" else 2].set(0.2)
        )
    if strategy is StraightSpanRouting:
        routing = strategy(routing.spans)
    base = (
        jnp.array([0.3, 0.1, -0.2])
        if robot.is_planar
        else jnp.array([0.1, -0.2, 0.3, np.cos(0.2), 0, 0, np.sin(0.2)])
    )
    other_base = (
        jnp.array([-0.6, -0.3, 0.4])
        if robot.is_planar
        else jnp.array([-0.3, 0.1, 0.2, np.cos(0.4), np.sin(0.4), 0, 0])
    )
    configuration = robot.pack_configuration(internal, base_pose=base)
    moved = robot.pack_configuration(internal, base_pose=other_base)
    solve = eqx.filter_jit(lambda values: routing.solve_routing(robot, values))
    solution, other = solve(configuration), solve(moved)
    assert bool(solution.converged & other.converged)
    assert_allclose(solution.path_lengths, other.path_lengths, atol=1e-12)
    analytical = routing.path_length_jacobian(robot, configuration, solution)
    numerical = jax.grad(lambda values: solve(values).path_lengths.sum())(configuration)
    assert_allclose(analytical[0], numerical, atol=1e-7)
    assert_allclose(analytical[:, : robot.num_base_coordinates], 0, atol=0)
    velocity = routing.path_velocity_jacobian(robot, configuration, solution)
    assert velocity.shape == (1, robot.num_velocities)
    assert_allclose(velocity[:, : robot.num_base_velocities], 0, atol=0)
    assert_allclose(
        velocity[:, robot.num_base_velocities :],
        analytical[:, robot.num_base_coordinates :],
    )
    force = -velocity.T @ jnp.array([2.0])
    assert force.shape == (robot.num_velocities,)
    assert_allclose(force[: robot.num_base_velocities], 0, atol=0)


def test_length_jacobian_uses_analytical_host_basis(monkeypatch):
    """The public length Jacobian must not construct an autodiff transform."""
    robot = _joint_robot()
    routing = _guided_routing(0.004)
    configuration = jnp.array([0.2, 0.4, -0.1])
    solution = eqx.filter_jit(routing.solve_routing)(robot, configuration)

    def forbidden(*args, **kwargs):
        """Reject any attempt to create a runtime differentiation transform."""
        raise AssertionError("The analytical length Jacobian invoked autodiff")

    monkeypatch.setattr(jax, "jacfwd", forbidden)
    monkeypatch.setattr(jax, "jacrev", forbidden)
    monkeypatch.setattr(jax, "grad", forbidden)
    assert bool(
        jnp.all(
            jnp.isfinite(routing.path_length_jacobian(robot, configuration, solution))
        )
    )


@pytest.mark.parametrize("scaled", [False, True])
def test_variable_strain_basis_matches_configuration_derivative(scaled):
    """Higher order GVS basis columns preserve coordinate order and scaling."""
    segment = GVSSegment(
        link=LinkSpec.circular(
            young_modulus=1e3,
            shear_modulus=5e2,
            density=1000.0,
            length=0.1,
            radius=0.03,
            reference_strain=[0, 0, 0, 1, 0, 0],
        ),
        joint=JointSpec.revolute(axis="z"),
        basis=StrainBasisSpec(
            type="monomial",
            strain_selector=[0, 1, 1, 1, 0, 0],
            basis_order=[0, 1, 2, 1, 0, 0],
        ),
        num_gauss_points=5,
    )
    robot = GVS.from_segments([segment], scale_rotational_basis_by_length=scaled)
    configuration = jnp.linspace(-0.01, 0.01, robot.num_coordinates)
    points = jnp.array([0.01, 0.04, 0.09])
    indices = jnp.zeros(3, dtype=int)
    strains, basis = robot.strain_and_basis(configuration, indices, points)
    numerical = jax.jacfwd(
        lambda values: robot.strain_and_basis(values, indices, points)[0]
    )(configuration)
    assert bool(jnp.all(jnp.isfinite(strains)))
    assert_allclose(basis, numerical, atol=1e-12)
    assert_allclose(basis[:, :, 0], 0, atol=0)
