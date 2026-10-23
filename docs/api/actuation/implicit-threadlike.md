# Implicit threadlike routing

`ShortestPathRouting` computes a path inside a circular lumen for a supplied
backbone configuration. This is the routing layer of the implicit actuation
model. The runnable example below supplies tendon forces to the robot dynamics
and solves static equilibrium with Optimistix.

Install the optional solver dependencies with:

```bash
pip install 'soromox[implicit-routing]'
```

The current implementation supports fixed-base PCS, PlanarPCS, and GVS with
fixed joints. Segment interfaces must have aligned material frames. It assumes
a taut, frictionless filament with negligible bending stiffness and inertia.
Compression-driven elastic rods need a separate routing strategy.

## Parameters and solving

`RoutingSpanParams` holds shared endpoints and connectivity.
`ShortestPathRoutingParams` combines those spans with `CircularLumenParams`.
All offsets and radii use metres. Indices are zero-based and inclusive.

For an existing one-segment continuum `robot` and its `configuration`:

```python
import equinox as eqx
import jax
import jax.numpy as jnp

from soromox.actuation import (
    CircularLumenParams,
    RoutingDiscretization,
    RoutingSolverOptions,
    RoutingSpanParams,
    ShortestPathRouting,
    ShortestPathRoutingParams,
)

jax.config.update("jax_enable_x64", True)

routing = ShortestPathRouting(
    params=ShortestPathRoutingParams(
        spans=RoutingSpanParams(
            start_offset=jnp.array([[0.0, -0.002, 0.0]]),
            end_offset=jnp.array([[0.0, 0.004, 0.0]]),
            span_path_index=(0,),
            start_segment_index=(0,),
            end_segment_index=(0,),
        ),
        domain=CircularLumenParams.constant(0.005, robot.num_segments),
    ),
    discretization=RoutingDiscretization(elements_per_segment=6),
    solver_options=RoutingSolverOptions(max_steps=200, tolerance=1e-8),
)
solve = eqx.filter_jit(routing.solve_routing)
solution = solve(robot, configuration)
lengths = solution.path_lengths
next_solution = solve(robot, next_configuration, initial_guess=solution)
```

Each span runs one `slsqp-jax` optimization. Spans assigned to the same path share
an actuator channel; their lengths are summed in `path_lengths`. A path crossing
two segments without a guide has one span with indices `(0,)` and `(1,)`. An
intermediate guide requires two spans assigned to the same path, with matching
exit/entry offsets at the guide. The solver does not add a guide at an ordinary
numerical segment boundary.

`inner_radius` has shape `(robot.num_segments,)`; use `constant` to broadcast a
single radius. Endpoint arrays have shape `(num_spans, 3)` in material-frame
`[0, y, z]` order. PlanarPCS also requires zero `z` components. Topology changes
require reconstruction; numeric geometry can be replaced without changing the
compiled array layout:

```python
params = routing.params.replace(
    domain=routing.params.domain.replace(inner_radius=new_radii),
)
routing = routing.with_params(params)
```

## Results and numerical accuracy

Check `solution.converged` before using the path. Each entry in `solution.spans`
contains its physical length, nodal transverse slope coefficients, global
backbone knots, iteration count, active wall mask, and normalized primal,
stationarity, and complementarity residuals. Multipliers correspond to the
normalized problem, with wall residuals in the nonpositive convention.
Convergence is determined from independently evaluated feasibility,
stationarity, complementarity, and multiplier signs. A solver termination code
alone is insufficient: SLSQP can report stagnation at a valid optimum or stop
before these checks pass.

The solver normalizes slopes by the entry radius divided by span length,
objective length by entry-radius squared divided by span length, endpoint
residuals by entry radius, and wall constraints by entry-radius squared.
This makes geometric scaling explicit and keeps solver tolerances dimensionless.
The returned coefficients are converted back to physical offset slopes.

`dense_wall_violation` is the maximum penetration in metres on a denser checking
grid, including element endpoints. A converged sampled optimization can still
have positive penetration between constraint points. Refine the routing mesh
and constraint sampling to meet the application's clearance tolerance. The dense
check is also sampled and is not a proof of continuous containment.

`routing.span_offsets(solution, span_index, points)` samples the material-frame
offset at global backbone coordinates. Supply points within that span. The
offset-derivative basis is piecewise linear; its integral supplies the offsets.
The routing mesh is independent of the host strain basis.

## Derivatives and limitations

The SLSQP adapter supplies analytical gradients, constraint Jacobians, and
Hessian products. A custom JVP differentiates the converged KKT equations rather
than optimizer iterations. Forward and reverse differentiation include geometry
parameters and the host strain field; warm-start guesses are not physical inputs
to the derivative.

Derivatives require a converged, regular solution and a stable active set.
The derivative rule returns NaNs for failed solves, degenerate tangents,
singular KKT systems, and contacts without a strict complementarity margin.
The forward path may still be useful at a contact transition, but a unique smooth
derivative is not guaranteed there. Use FP64 for the documented tolerances.
After checking the unregularized KKT matrix for rank loss, the derivative solve
adds `solver_options.regularization` to its primal Hessian diagonal (default
`1e-12`). This introduces a small sensitivity approximation; it does not change
the routing objective.

The initial release exposes routing independently of `ThreadlikeActuator`.
Floating bases, moving joints, friction, slack, and elastic push-rod mechanics
are not yet supported. Explicit solver state keeps repeated evaluations and
batched JAX calls independent.

## Runnable hollow-section simulations

Run the example from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/soromox-mpl \
  python examples/simulation/pcs/simulate_hollow_tendon_routing.py --check-refinement
```

Install `soromox[implicit-routing,examples]` and provide FFmpeg on `PATH`.
The example saves PNG cutaways, routing and offset plots, MP4 animations,
NPZ trajectories, and `report.json` under
`examples/simulation/pcs/figures/hollow_tendon_routing/`.
Use `--case same-side` or `--case opposed` to select a case, `--output` to change
the destination, and `--render-only` to redraw saved trajectories.

Both cases have two 120 mm sections, 16 mm outer radius, 14 mm lumen radius,
and 4 mm tendon anchor offsets. A separate tendon is anchored at each section's
entry and exit, including the section transition. Each interior route remains
free to move within the lumen. The distal tendon receives an ideal local tension;
the example does not model an additional supply path from a motor at the base.

The model uses annular area and second moment in its elastic, damping, and
distributed inertia parameters. There is one bending coordinate per section;
axial stretch and shear are constrained, and gravity is zero. These are
illustrative planar examples inspired by the paper's hollow sections and fixed
anchors, rather than a reproduction of its experimental parameters or results.

The applied force is evaluated using:

```python
solution = routing.solve_routing(robot, configuration, initial_guess=previous)
length_jacobian = routing.path_length_jacobian(robot, configuration, solution)
generalized_force = -length_jacobian.T @ tension
```

The envelope identity gives the first derivative of optimal path length because
these constraints are independent of robot configuration. The supplied solution
must correspond to the current geometry and configuration. For derivatives of
this force, retain the solution's dependence on configuration through the
implicit routing solver.

The example integrates inertia, Coriolis forces, elasticity, damping, and tendon
forces with RK4. Routing is optimized at every RK stage, with warm starts passed
explicitly. Positive preload keeps the ideal tendon model taut. Returning to
preload allows elastic recoil. Static equilibria are found in the example with
Optimistix bisection; in this zero-gravity model the two static section balances
are independent, although their dynamics are coupled by inertia.

Every integration stage checks convergence and dense wall clearance. The JSON
report includes the energy balance error, maximum routing residual, and wall
penetration. `--check-refinement` repeats both rollouts at half the time step and
checks twice as many routing elements at the largest simulated bend.

The rendered cases use moderate tensions and paths that stay clear of the wall.
Broader wall-contact regions exposed SLSQP convergence failures during example
development. Such results are rejected by the residual checks. Robust simulation
through these contact regimes remains further work; the included animations do
not validate it.
