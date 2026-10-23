"""SLSQP adapter and implicit derivatives for sampled routing problems."""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


class RoutingSolverOptions(eqx.Module):
    """Dimensionless solver tolerances and bounded iteration count."""

    max_steps: int = eqx.field(static=True, default=200)
    tolerance: float = eqx.field(static=True, default=1e-8)
    active_tolerance: float = eqx.field(static=True, default=1e-6)
    regularization: float = eqx.field(static=True, default=1e-12)

    def __check_init__(self):
        import math

        if (
            not isinstance(self.max_steps, int)
            or isinstance(self.max_steps, bool)
            or self.max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer.")
        for name in ("tolerance", "active_tolerance", "regularization"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.active_tolerance < self.tolerance:
            raise ValueError("active_tolerance must be at least tolerance.")


class RoutingProblem(eqx.Module):
    """Affine tangents and constraints in dimensionless routing variables.

    Tangent arrays have shapes (points, 3, coefficients) and (points, 3).
    Wall arrays use (constraints, transverse dimensions, coefficients).
    Equality constraints follow ``equality_matrix @ coefficients - target``.
    Wall constraints follow ``sum(offset**2) - radius_squared <= 0``.
    Weights already include objective normalization.
    """

    tangent_matrix: Array
    tangent_offset: Array
    weights: Array
    equality_matrix: Array
    target: Array
    wall_matrix: Array
    wall_offset: Array
    radius_squared: Array


def objective(coefficients, problem):
    """Evaluate normalized length and its analytical coefficient gradient."""
    tangent = problem.tangent_offset + jnp.einsum(
        "pin,n->pi", problem.tangent_matrix, coefficients
    )
    norms = jnp.linalg.norm(tangent, axis=-1)
    unit = tangent / jnp.maximum(norms[:, None], jnp.finfo(tangent.dtype).tiny)
    gradient = jnp.einsum("p,pin,pi->n", problem.weights, problem.tangent_matrix, unit)
    return jnp.dot(problem.weights, norms), gradient


def equalities(coefficients, problem):
    """Return endpoint equality residuals."""
    return problem.equality_matrix @ coefficients - problem.target


def objective_hvp(coefficients, direction, problem):
    """Apply the exact length Hessian to a coefficient direction."""
    tangent = problem.tangent_offset + jnp.einsum(
        "pin,n->pi", problem.tangent_matrix, coefficients
    )
    norms = jnp.maximum(
        jnp.linalg.norm(tangent, axis=-1), jnp.finfo(tangent.dtype).tiny
    )
    unit = tangent / norms[:, None]
    variation = jnp.einsum("pin,n->pi", problem.tangent_matrix, direction)
    projected = variation - unit * jnp.sum(unit * variation, axis=-1, keepdims=True)
    return jnp.einsum(
        "p,pin,pi->n", problem.weights / norms, problem.tangent_matrix, projected
    )


def inequalities(coefficients, problem):
    """Return containment residuals in the nonpositive convention."""
    offsets = problem.wall_offset + jnp.einsum(
        "pin,n->pi", problem.wall_matrix, coefficients
    )
    return jnp.sum(offsets**2, axis=-1) - problem.radius_squared


def inequality_jacobian(coefficients, problem):
    """Evaluate the analytical containment Jacobian."""
    offsets = problem.wall_offset + jnp.einsum(
        "pin,n->pi", problem.wall_matrix, coefficients
    )
    return 2 * jnp.einsum("pi,pin->pn", offsets, problem.wall_matrix)


def kkt_residual(state, problem, active):
    """Return stationarity, equalities, and masked wall constraints."""
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


def _forward_solve(problem, initial, options):
    try:
        import optimistix as optx
        from slsqp_jax import SLSQP, QPConfig, SLSQPConfig, ToleranceConfig
    except ImportError as error:
        raise ImportError(
            "Implicit routing requires the 'implicit-routing' extra: "
            "pip install 'soromox[implicit-routing]'."
        ) from error

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
    equality_multipliers = state[initial.size : initial.size + problem.target.size]
    wall_multipliers = state[initial.size + problem.target.size :]
    wall_values = inequalities(result.value, problem)
    stationarity = (
        objective(result.value, problem)[1]
        + problem.equality_matrix.T @ equality_multipliers
        + inequality_jacobian(result.value, problem).T @ wall_multipliers
    )
    tolerance = 10 * options.tolerance
    verified = (
        jnp.all(jnp.isfinite(state))
        & (
            jnp.max(jnp.abs(equalities(result.value, problem)), initial=0.0)
            <= tolerance
        )
        & (jnp.max(wall_values, initial=0.0) <= tolerance)
        & (jnp.max(jnp.abs(stationarity), initial=0.0) <= tolerance)
        & (jnp.max(jnp.abs(wall_multipliers * wall_values), initial=0.0) <= tolerance)
        & jnp.all(wall_multipliers >= -options.tolerance)
    )
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
    """Solve with SLSQP and differentiate the converged active KKT equations."""
    return _forward_solve(problem, initial, options)


@solve_problem.defjvp
def _solve_problem_jvp(options, primals, tangents):
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
    primal_feasible = jnp.all(wall_values <= options.tolerance * 10)
    dual_feasible = jnp.all(wall_multipliers >= -options.tolerance)
    tangent = problem.tangent_offset + jnp.einsum(
        "pin,n->pi", problem.tangent_matrix, state[:num_coefficients]
    )
    nondegenerate = jnp.all(jnp.linalg.norm(tangent, axis=-1) > 1e-12)
    valid = (
        (diagnostics[0] > 0)
        & (jnp.max(jnp.abs(residual)) <= 10 * options.tolerance)
        & regular
        & strict_complementarity
        & primal_feasible
        & dual_feasible
        & nondegenerate
    )
    derivative = derivative * jnp.where(valid, 1.0, jnp.nan)
    return (state, diagnostics), (derivative, jnp.zeros_like(diagnostics))
