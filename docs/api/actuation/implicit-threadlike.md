# Configuration-dependent threadlike routing

`ConfigurationDependentThreadlikeRouting` describes paths whose material offsets
change with robot configuration. `StraightSpanRouting` resolves straight cables
between prescribed guides in closed form. `ShortestPathRouting` optimizes paths
inside a circular lumen. Both expose lengths, analytical configuration and
velocity Jacobians, and world-coordinate sampling through `span_positions`.
The runnable example below supplies tendon forces to the robot dynamics and
solves static equilibrium with Optimistix.

## Straight spans without wall contact

Use this strategy when the operating envelope excludes relevant wall contact:

```python
from soromox.actuation import RoutingSpanParams, StraightSpanRouting

spans = RoutingSpanParams(
    start_offset=[[0.0, 0.002, 0.0]],
    end_offset=[[0.0, 0.002, 0.0]],
    span_path_index=(0,),
    start_segment_index=(0,),
    end_segment_index=(0,),
)
routing = StraightSpanRouting(spans)
solution = routing.solve_routing(robot, configuration)
lengths = solution.path_lengths
jacobian = routing.path_velocity_jacobian(robot, configuration, solution)
force = -jacobian.T @ tension
positions = routing.span_positions(robot, configuration, solution, 0, fractions)
```

Each span has length `norm(exit_position - entry_position)`. Its analytical
derivative uses the endpoint position Jacobians. `StraightSpanSolution` stores
world endpoints, length, length Jacobian, and validity. No optimization state or
lumen geometry is required. The caller guarantees clearance; convergence only
checks finite geometry and nonzero span length. Sampling interpolates the world
line directly, including for folded backbones where an offset graph can fail.
This ideal model excludes slack, bending stiffness, and compression buckling.

Both strategies retain prescribed upstream and downstream guides at moving
joints and include their connecting cable lengths. Shared `RoutingSpanParams`
can be reused when switching strategies. `with_params` replaces numerical guide
geometry while preserving topology. No runtime autodiff is used to evaluate the
first length Jacobian.

The routing solver is installed with the core package:

```bash
pip install soromox
```

PCS, PlanarPCS, and GVS implement the required host interface, including floating
bases. Moving GVS joints connect prescribed guides on adjacent spans. It assumes
a taut, frictionless filament with negligible bending stiffness and inertia.
Compression-driven elastic rods need a separate routing strategy.

## Parameters and solving

`RoutingSpanParams` holds shared endpoints and connectivity. A **path** is one
physical tendon or filament associated with an actuator channel. A **span** is
its portion between two prescribed guides. The **lumen** is the hollow
interior of a section within which that portion can move.
`ShortestPathRoutingParams` combines those spans with `CircularLumenParams`.
All offsets and radii use metres. Indices are zero-based and inclusive.

For an existing one-segment continuum `robot` and its `configuration`:

```python
import equinox as eqx
import jax
import jax.numpy as jnp

from soromox.actuation import (
    CircularLumenParams,
    PiecewiseLinearRoutingDiscretization,
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
    discretization=PiecewiseLinearRoutingDiscretization(elements_per_segment=6),
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
intermediate guide requires two spans assigned to the same path. Matching
exit/entry offsets at a continuous interface describe a shared guide. Distinct
offsets describe two guides joined by a straight cable. The solver does not add
a guide at an ordinary numerical segment boundary. Parameters remain alongside
their strategy classes, matching the existing threadlike actuation API.

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
This makes geometric scaling explicit. Optimization residuals are dimensionless;
physical acceptance budgets use metres.
The returned coefficients are converted back to physical offset slopes.

Acceptance thresholds in `RoutingSolverOptions` are independent:

| Attribute | Default | Meaning |
| --- | ---: | --- |
| `tolerance` | `1e-8` | SLSQP termination tolerance |
| `endpoint_tolerance` | `1e-8` m | Maximum endpoint position error |
| `wall_tolerance` | `1e-6` m | Maximum continuous-span radial penetration bound |
| `optimality_tolerance` | `1e-5` | Maximum normalized stationarity residual |
| `complementarity_tolerance` | `1e-7` | Maximum normalized multiplier/residual product |
| `dual_tolerance` | `1e-8` | Allowed negative multiplier magnitude |
| `sensitivity_tolerance` | `1e-7` | KKT residual limit for coefficient derivatives |

These defaults are numerical acceptance budgets, not universal bounds on force
or solution error. Conditioning and discretization still matter. The first
length Jacobian uses the envelope expression at an accepted forward solution.
Differentiating the coefficients or forces additionally requires the stricter
sensitivity checks, an adequately conditioned KKT matrix, and an unambiguous
active set. Such derivatives can be NaN even when the forward solution converges.

`dense_wall_violation` reports observed penetration on the checking grid.
`wall_violation_bound` adds a conservative interpolation error bound: on each
checking interval of width `h`, a quadratic offset differs from its linear
interpolant by at most `norm(offset_second_derivative) * h**2 / 8`.
The maximum endpoint norm plus this error bounds the entire interval's norm.
`solution.converged` requires this continuous bound to meet `wall_tolerance`.
Increase validation sampling if the bound is too conservative; refine the
optimization mesh and wall constraints if actual penetration is excessive.
Rejected geometry also yields invalid length derivatives.

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
Friction, slack, joint obstacle geometry, and elastic push-rod mechanics require
additional models. Explicit solver state keeps repeated evaluations and batched
JAX calls independent.

At tangents shorter than `1e-12`, the optimizer uses a finite quadratic
continuation of the norm so reverse differentiation does not encounter a hidden
division by zero. These degenerate routes are rejected by the convergence and
implicit derivative checks. The physical objective is unchanged above that
threshold.

## Prescribed guides at moving joints

A joint has no modeled thickness. For each physical tendon crossing a moving
joint, end one span in the upstream section and start another in the downstream
section, assigning both to the same path. Each offset is expressed in its own
section's material frame. The joint transform places the two guides relative
to one another. Their straight connecting cable contributes to both path length
and the analytical length Jacobian; no joint housing collision check is implied.

For a two-section host with an intervening joint, use a nonzero guide offset
for guides away from the pivot, or set it to zero for guides at the joint origin:

```python
guide_offset = 0.004
offsets = jnp.array([[0.0, guide_offset, 0.0], [0.0, guide_offset, 0.0]])
spans = RoutingSpanParams(
    start_offset=offsets,
    end_offset=offsets,
    span_path_index=(0, 0),
    start_segment_index=(0, 1),
    end_segment_index=(0, 1),
)
routing = ShortestPathRouting(
    ShortestPathRoutingParams(spans, CircularLumenParams.constant(0.01, 2))
)
solution = routing.solve_routing(robot, configuration)
connection = solution.guides[0]
velocity_jacobian = routing.path_velocity_jacobian(robot, configuration, solution)
force = -velocity_jacobian.T @ tension
```

Centered guides remain coincident across purely rotational joints: their
connecting length and direct joint torque are zero. Offset guides can separate
under rotation and generate joint torque. Translational joints can separate
both centered and offset guides. The joint can also lie upstream of an entire
span; its rigid motion does not change that span's internal routing.

`solution.guides` reports each connection's length, displacement in the upstream
frame, downstream span index, and `differentiable` flag. If moving guides collapse
to zero separation, the forward length remains valid but the norm generally has
no unique derivative. Connections of length at most `1e-12` are flagged and their
path Jacobian rows return NaNs, unless the guides coincide for every joint state.
For always coincident guides, configuration derivatives are zero. Differentiation
with respect to guide geometry at coincidence is outside this smooth convention.

Floating base poses do not affect internal path lengths. `path_length_jacobian`
uses `robot.num_coordinates` columns, including zero base pose columns.
`path_velocity_jacobian` uses `robot.num_velocities` columns, including zero base
velocity columns; use it to compute generalized forces on a floating robot.
This distinction matters for spatial bases, whose quaternion pose has seven
coordinates and whose velocity has six components.

Host implementations provide `routing_strain_and_basis`,
`routing_endpoint_kinematics`, `routing_guide_kinematics`, and
`validate_routing_span`. Endpoint kinematics distinguish entry and exit frames
at a joint and return internal configuration derivatives with zero base columns.
GVS shares its local material-strain evaluator with standard threadlike routing;
global basis assembly is only performed when needed. The routing strategy does not inspect
private attributes to select a rod model. A span crossing a moving joint without
prescribed guides is rejected because there is no joint passage geometry.

## Runnable hollow-section simulations

Run the example from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/soromox-mpl \
  python examples/simulation/pcs/simulate_hollow_tendon_routing.py --check-refinement
```

Install `soromox[examples]` and provide FFmpeg on `PATH`.
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

The analytical tangent derivative in the paper's Appendix B, (B2), integrated
against the unit tangent gives the length Jacobian (equivalently (B5) transposed).
The envelope identity (26) gives the first derivative of optimal path length because
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

## Optimization structure

Choose straight spans explicitly when contact is irrelevant. Wall-aware routing
uses one SLSQP solve per span. At a fixed backbone configuration the sampled
problem has an affine tangent norm objective, linear endpoint equalities, and
convex circular wall constraints. The solver receives analytical gradients,
constraint Jacobians, and Hessian products.

The outer static equilibrium solve belongs to the example. Optimistix updates
the robot configuration, and each residual evaluation resolves the routes at
that configuration before computing tendon forces.

## Routing interfaces

::: soromox.actuation.ConfigurationDependentThreadlikeRouting

::: soromox.actuation.StraightSpanRouting

::: soromox.actuation.ShortestPathRouting

::: soromox.actuation.RoutingSolverOptions
