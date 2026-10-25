"""SLSQP adapter and implicit derivatives for sampled routing problems."""

from __future__ import annotations

import math
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array
from slsqp_jax import SLSQP, QPConfig, SLSQPConfig, ToleranceConfig


class RoutingSolverOptions(eqx.Module):
    """Separate numerical termination, physical feasibility, and sensitivity checks.

    Attributes:
        max_steps: Positive upper bound on SLSQP iterations.
        tolerance: Dimensionless SLSQP termination tolerance.
        endpoint_tolerance: Maximum endpoint position residual in metres.
        wall_tolerance: Maximum radial wall penetration in metres, also applied
            to the continuous-span clearance bound by the routing strategy.
        optimality_tolerance: Maximum normalized stationarity residual for
            forward routing and its analytical envelope force evaluation.
        complementarity_tolerance: Maximum normalized multiplier/residual product.
        dual_tolerance: Maximum allowed negative wall multiplier magnitude.
        sensitivity_tolerance: Stricter normalized KKT residual limit for
            derivatives through the optimized coefficients. Forward acceptance
            alone does not certify these derivatives.
        active_tolerance: Wall residual threshold used to identify active
            constraints; also the strict complementarity margin for derivatives.
        regularization: Positive diagonal addition to the primal KKT Hessian
            during sensitivity solves, after checking the original matrix rank.
    """

    max_steps: int = eqx.field(static=True, default=200)
    tolerance: float = eqx.field(static=True, default=1e-8)
    endpoint_tolerance: float = eqx.field(static=True, default=1e-8)
    wall_tolerance: float = eqx.field(static=True, default=1e-6)
    optimality_tolerance: float = eqx.field(static=True, default=1e-5)
    complementarity_tolerance: float = eqx.field(static=True, default=1e-7)
    dual_tolerance: float = eqx.field(static=True, default=1e-8)
    sensitivity_tolerance: float = eqx.field(static=True, default=1e-7)
    active_tolerance: float = eqx.field(static=True, default=1e-6)
    regularization: float = eqx.field(static=True, default=1e-12)

    def __check_init__(self):
        """Validate parameters after construction.

        Returns:
            result (None): ``None``. Invalid parameter values raise ValueError or
                TypeError.
        """
        if (
            not isinstance(self.max_steps, int)
            or isinstance(self.max_steps, bool)
            or self.max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer.")
        for name in (
            "tolerance",
            "active_tolerance",
            "regularization",
            "endpoint_tolerance",
            "wall_tolerance",
            "optimality_tolerance",
            "complementarity_tolerance",
            "dual_tolerance",
            "sensitivity_tolerance",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.active_tolerance < self.sensitivity_tolerance:
            raise ValueError("active_tolerance must be at least sensitivity_tolerance.")


class RoutingProblem(eqx.Module):
    """Affine tangents and constraints in dimensionless routing variables.

    Tangent arrays have shapes (points, 3, coefficients) and (points, 3).
    Wall arrays use (constraints, transverse dimensions, coefficients).
    Equality constraints follow ``equality_matrix @ coefficients - target``.
    Wall constraints follow ``sum(offset**2) - radius_squared <= 0``.
    Weights already include objective normalization.

    Attributes:
        tangent_matrix: Affine tangent coefficients, shape ``(points, 3, variables)``.
        tangent_offset: Constant tangent terms, shape ``(points, 3)``.
        weights: Normalized integration weights, shape ``(points,)``.
        equality_matrix: Endpoint matrix, shape ``(equalities, variables)``.
        target: Endpoint targets, shape ``(equalities,)``.
        wall_matrix: Offset matrix, shape ``(constraints, dimensions, variables)``.
        wall_offset: Constant offsets, shape ``(constraints, dimensions)``.
        radius_squared: Squared normalized radii, shape ``(constraints,)``.
        length_scale: Metres per normalized offset, used to interpret endpoint
            and radial feasibility. Defaults to one for direct solver users.
    """

    tangent_matrix: Array
    tangent_offset: Array
    weights: Array
    equality_matrix: Array
    target: Array
    wall_matrix: Array
    wall_offset: Array
    radius_squared: Array
    length_scale: Array | float = 1.0


def _norm_and_unit(tangent):
    """Evaluate tangent norms with a finite continuation at zero.

    Args:
        tangent (Array): Vectors with shape ``(..., 3)``.

    Returns:
        result (tuple[Array, Array]): Norms with shape ``(...)`` and unit vectors with
            shape ``(..., 3)``. Below ``1e-12``, the norm is continued quadratically and
            its gradient is the vector divided by ``1e-12``. Such tangents are rejected
            by the routing convergence and derivative checks.
    """
    squared = jnp.sum(tangent**2, axis=-1)
    threshold = jnp.asarray(1e-12, dtype=tangent.dtype)
    regular = squared > threshold**2
    denominator = jnp.sqrt(jnp.where(regular, squared, threshold**2))
    norm = jnp.where(regular, denominator, squared / (2 * threshold) + threshold / 2)
    return norm, tangent / denominator[..., None]


def objective(coefficients, problem):
    """Evaluate normalized length and its analytical coefficient gradient.

    Args:
        coefficients (Array): Normalized routing coefficients, shape ``(variables,)``.
        problem (RoutingProblem): RoutingProblem containing affine tangents, weights,
            and constraints.

    Returns:
        result (tuple[Array, Array]): Normalized scalar length and its analytical
            coefficient gradient, shape ``(variables,)``. The documented finite
            continuation applies at degenerate tangents.
    """
    tangent = problem.tangent_offset + jnp.einsum(
        "pin,n->pi", problem.tangent_matrix, coefficients
    )
    norms, unit = _norm_and_unit(tangent)
    gradient = jnp.einsum("p,pin,pi->n", problem.weights, problem.tangent_matrix, unit)
    return jnp.dot(problem.weights, norms), gradient


def equalities(coefficients, problem):
    """Return endpoint equality residuals.

    Args:
        coefficients (Array): Normalized routing coefficients, shape ``(variables,)``.
        problem (RoutingProblem): RoutingProblem containing affine tangents, weights,
            and constraints.

    Returns:
        result (Array): Endpoint residuals with shape ``(equalities,)``; feasibility
            requires zero.
    """
    return problem.equality_matrix @ coefficients - problem.target


def objective_hvp(coefficients, direction, problem):
    """Apply the exact length Hessian to a coefficient direction.

    Args:
        coefficients (Array): Normalized routing coefficients, shape ``(variables,)``.
        direction (Array): Coefficient perturbation, shape ``(variables,)``.
        problem (RoutingProblem): RoutingProblem containing affine tangents, weights,
            and constraints.

    Returns:
        result (Array): Hessian-vector product with shape ``(variables,)``. At
            degenerate tangents this is the Hessian of the same finite continuation used
            by objective.
    """
    tangent = problem.tangent_offset + jnp.einsum(
        "pin,n->pi", problem.tangent_matrix, coefficients
    )
    _, unit = _norm_and_unit(tangent)
    squared = jnp.sum(tangent**2, axis=-1)
    regular = squared > 1e-24
    norms = jnp.sqrt(jnp.where(regular, squared, 1e-24))
    variation = jnp.einsum("pin,n->pi", problem.tangent_matrix, direction)
    projected = variation - jnp.where(
        regular[:, None],
        unit * jnp.sum(unit * variation, axis=-1, keepdims=True),
        0,
    )
    return jnp.einsum(
        "p,pin,pi->n", problem.weights / norms, problem.tangent_matrix, projected
    )


def inequalities(coefficients, problem):
    """Return containment residuals in the nonpositive convention.

    Args:
        coefficients (Array): Normalized routing coefficients, shape ``(variables,)``.
        problem (RoutingProblem): RoutingProblem containing affine tangents, weights,
            and constraints.

    Returns:
        result (Array): Wall residuals with shape ``(constraints,)``; feasibility
            requires nonpositive entries.
    """
    offsets = problem.wall_offset + jnp.einsum(
        "pin,n->pi", problem.wall_matrix, coefficients
    )
    return jnp.sum(offsets**2, axis=-1) - problem.radius_squared


def inequality_jacobian(coefficients, problem):
    """Evaluate the analytical containment Jacobian.

    Args:
        coefficients (Array): Normalized routing coefficients, shape ``(variables,)``.
        problem (RoutingProblem): RoutingProblem containing affine tangents, weights,
            and constraints.

    Returns:
        result (Array): Wall residual Jacobian with shape ``(constraints, variables)``.
    """
    offsets = problem.wall_offset + jnp.einsum(
        "pin,n->pi", problem.wall_matrix, coefficients
    )
    return 2 * jnp.einsum("pi,pin->pn", offsets, problem.wall_matrix)


def kkt_residual(state, problem, active):
    """Return stationarity, equalities, and masked wall constraints.

    Args:
        state (Array): Coefficients followed by equality and wall multipliers.
        problem (RoutingProblem): RoutingProblem containing affine tangents, weights,
            and constraints.
        active (Array): Boolean wall active-set mask, shape ``(constraints,)``.

    Returns:
        result (Array): Concatenated stationarity, endpoint, and masked wall residuals
            with the same shape as state.
    """
    num_coefficients = problem.tangent_matrix.shape[-1]
    num_equalities = problem.target.size
    coefficients = state[:num_coefficients]
    equality_multipliers = state[num_coefficients : num_coefficients + num_equalities]
    wall_multipliers = state[num_coefficients + num_equalities :]
    stationarity = (
        objective(coefficients, problem)[1]
        + problem.equality_matrix.T @ equality_multipliers
        + inequality_jacobian(coefficients, problem).T @ (active * wall_multipliers)
    )
    return jnp.concatenate(
        (
            stationarity,
            equalities(coefficients, problem),
            jnp.where(active, inequalities(coefficients, problem), wall_multipliers),
        )
    )


def verify_solution(state, problem, options):
    """Verify a solver state using independent physical and optimality limits.

    Args:
        state: Coefficients followed by equality and wall multipliers.
        problem: Normalized affine routing data and physical offset scale.
        options: Independent acceptance thresholds.

    Returns:
        Scalar boolean for the sampled problem. Continuous wall containment
        is checked by the routing strategy after reconstructing its geometry.
    """
    count = problem.tangent_matrix.shape[-1]
    coefficients = state[:count]
    equality_multipliers = state[count : count + problem.target.size]
    wall_multipliers = state[count + problem.target.size :]
    wall_values = inequalities(coefficients, problem)
    stationarity = (
        objective(coefficients, problem)[1]
        + problem.equality_matrix.T @ equality_multipliers
        + inequality_jacobian(coefficients, problem).T @ wall_multipliers
    )
    offsets = problem.wall_offset + jnp.einsum(
        "pin,n->pi", problem.wall_matrix, coefficients
    )
    radial_violation = (
        jnp.linalg.norm(offsets, axis=-1) - jnp.sqrt(problem.radius_squared)
    ) * problem.length_scale
    endpoint_error = (
        jnp.linalg.norm(equalities(coefficients, problem)) * problem.length_scale
    )
    tangent = problem.tangent_offset + jnp.einsum(
        "pin,n->pi", problem.tangent_matrix, coefficients
    )
    return (
        jnp.all(jnp.isfinite(state))
        & jnp.all(jnp.sum(tangent**2, axis=-1) > 1e-24)
        & jnp.all(jnp.isfinite(problem.length_scale) & (problem.length_scale > 0))
        & (endpoint_error <= options.endpoint_tolerance)
        & (jnp.max(radial_violation, initial=0.0) <= options.wall_tolerance)
        & (jnp.max(jnp.abs(stationarity), initial=0.0) <= options.optimality_tolerance)
        & (
            jnp.max(jnp.abs(wall_multipliers * wall_values), initial=0.0)
            <= options.complementarity_tolerance
        )
        & jnp.all(wall_multipliers >= -options.dual_tolerance)
    )


def _forward_solve(problem, initial, options):
    """Run SLSQP and independently verify its optimality conditions.

    Args:
        problem (RoutingProblem): RoutingProblem containing affine tangents, weights,
            and constraints.
        initial (Array): Initial normalized coefficients, shape ``(variables,)``.
        options (RoutingSolverOptions): Static RoutingSolverOptions specifying
            tolerances and iteration limit.

    Returns:
        result (tuple[Array, Array]): Coefficient/multiplier state and a two-entry array
            containing independent verification status and iteration count.
    """
    solver = SLSQP(
        config=SLSQPConfig(
            qp=QPConfig(use_exact_hvp=True),
            tolerance=ToleranceConfig(
                rtol=options.tolerance,
                atol=options.tolerance,
                max_steps=options.max_steps,
            ),
        ),
        eq_constraint_fn=equalities,
        ineq_constraint_fn=lambda coefficients, args: -inequalities(coefficients, args),
        n_eq_constraints=problem.target.size,
        n_ineq_constraints=problem.radius_squared.size,
        obj_grad_fn=lambda coefficients, args: objective(coefficients, args)[1],
        obj_hvp_fn=objective_hvp,
        eq_hvp_fn=lambda coefficients, direction, args: jnp.zeros_like(
            args.equality_matrix
        ),
        ineq_hvp_fn=lambda coefficients, direction, args: (
            -2
            * jnp.einsum("pin,pim,m->pn", args.wall_matrix, args.wall_matrix, direction)
        ),
        eq_jac_fn=lambda coefficients, args: args.equality_matrix,
        ineq_jac_fn=lambda coefficients, args: -inequality_jacobian(coefficients, args),
    )
    result = optx.minimise(
        lambda coefficients, args: objective(coefficients, args)[0],
        solver,
        initial,
        args=problem,
        max_steps=options.max_steps,
        throw=False,
    )
    state = jnp.concatenate(
        (
            result.value,
            -result.stats["multipliers_eq"],
            result.stats["multipliers_ineq"],
        )
    )
    verified = verify_solution(state, problem, options)
    diagnostics = jnp.array(
        [
            verified,
            result.stats["num_steps"],
        ],
        dtype=initial.dtype,
    )
    return state, diagnostics


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def solve_problem(problem, initial, options):
    """Solve with SLSQP and differentiate the converged active KKT equations.

    Args:
        problem (RoutingProblem): RoutingProblem containing affine tangents, weights,
            and constraints.
        initial (Array): Initial normalized coefficients, shape ``(variables,)``.
        options (RoutingSolverOptions): Static RoutingSolverOptions specifying
            tolerances and iteration limit.

    Returns:
        result (tuple[Array, Array]): Coefficient/multiplier state and
            verification/iteration diagnostics. Invalid smooth sensitivities are
            returned as NaNs by the custom JVP.
    """
    return _forward_solve(problem, initial, options)


@solve_problem.defjvp
def _solve_problem_jvp(options, primals, tangents):
    """Differentiate a regular converged routing solution through its KKT equations.

    Args:
        options (RoutingSolverOptions): Static RoutingSolverOptions specifying
            tolerances and iteration limit.
        primals (tuple): Primal RoutingProblem and initial coefficient array.
        tangents (tuple): Problem and initial-guess perturbations; the latter does not
            affect
            implicit sensitivities.

    Returns:
        result (tuple[tuple[Array, Array], tuple[Array, Array]]): Primal
            state/diagnostics and their directional derivatives. The diagnostic
            derivative is zero.
    """
    problem, initial = primals
    problem_tangent, _ = tangents
    state, diagnostics = solve_problem(problem, initial, options)
    num_coefficients = initial.size
    wall_values = inequalities(state[:num_coefficients], problem)
    active = jax.lax.stop_gradient(wall_values >= -options.active_tolerance)
    matrix = jax.jacfwd(kkt_residual)(state, problem, active)
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    regular = singular_values[-1] > 1e-10 * jnp.maximum(singular_values[0], 1)
    matrix = matrix.at[:num_coefficients, :num_coefficients].add(
        options.regularization * jnp.eye(num_coefficients, dtype=state.dtype)
    )
    _, right_hand_side = jax.jvp(
        lambda current_problem: kkt_residual(state, current_problem, active),
        (problem,),
        (problem_tangent,),
    )
    derivative = jnp.linalg.solve(matrix, -right_hand_side)
    residual = kkt_residual(state, problem, active)
    wall_multipliers = state[num_coefficients + problem.target.size :]
    strict_complementarity = jnp.all(
        jnp.where(
            active,
            wall_multipliers > options.active_tolerance,
            wall_values < -options.active_tolerance,
        )
    )
    primal_feasible = jnp.all(wall_values <= options.sensitivity_tolerance)
    dual_feasible = jnp.all(wall_multipliers >= -options.dual_tolerance)
    tangent = problem.tangent_offset + jnp.einsum(
        "pin,n->pi", problem.tangent_matrix, state[:num_coefficients]
    )
    nondegenerate = jnp.all(jnp.linalg.norm(tangent, axis=-1) > 1e-12)
    valid = (
        (diagnostics[0] > 0)
        & (jnp.max(jnp.abs(residual)) <= options.sensitivity_tolerance)
        & regular
        & strict_complementarity
        & primal_feasible
        & dual_feasible
        & nondegenerate
    )
    derivative = derivative * jnp.where(valid, 1.0, jnp.nan)
    return (state, diagnostics), (derivative, jnp.zeros_like(diagnostics))
