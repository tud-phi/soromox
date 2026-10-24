# Threadlike actuation

Threadlike paths describe axial force transmission through a continuum robot.
Choose the routing strategy according to how the path is constrained:

- **Fixed material routing:** `ThreadlikeRouting` prescribes offsets along the
  backbone and integrates with `ThreadlikeActuator`.
- **Straight spans between guides:** `StraightSpanRouting` follows prescribed
  guides when wall contact is irrelevant throughout operation.
- **Shortest paths inside a lumen:** `ShortestPathRouting` resolves the path
  subject to circular wall constraints using SLSQP.

The configuration-dependent strategies expose lengths and Jacobians for force
assembly; the example below applies tendon forces to the robot dynamics.

## Fixed material routing

Threadlike actuation models paths embedded in a continuum robot's material
frame. The same geometry supports pulling tendons, pushing rods, simplified
contractile muscles, and pressure chambers represented by an equivalent axial
volume coordinate. A path may span any contiguous inclusive range of body
segments.

!!! info "Threadlike-routing reference"
    The routing geometry follows:

    > Renda, F., Armanini, C., Mathew, A. T., & Boyer, F. (2022).
    > *Geometrically-Exact Inverse Kinematic Control of Soft Manipulators With
    > General Threadlike Actuators' Routing*. IEEE Robotics and Automation
    > Letters, 7(3), 7311–7318.
    > [https://doi.org/10.1109/LRA.2022.3183248](https://doi.org/10.1109/LRA.2022.3183248)

A routing is a material-frame offset from the backbone. `PCS`, `PlanarPCS`, and
`GVS` combine that routing with their native strain kinematics to evaluate path
length, coordinate velocity, and generalized actuator force.

This strategy assumes fixed material-frame routing, one scalar coordinate and
effort per path, and an optional static effort loss along active paths.

### Routing

#### Linear routing

`ThreadlikeRouting` stores a vectorized path family. Linear paths use

\[
r(s)=r_0+r_1s,
\]

where `intercept` and `slope` have shape `(num_paths, 3)` and use local
material-frame `[x, y, z]` order. The undeformed backbone follows local `+x`,
so both local-`x` components must be zero.

```python
from soromox.actuation import ThreadlikeRouting

routing = ThreadlikeRouting.linear(
    intercept=jnp.array([[0.0, 0.01, 0.0], [0.0, -0.01, 0.0]]),
    slope=jnp.zeros((2, 3)),
    start_segment_index=(0, 0),
    end_segment_index=(1, 1),
)
```

Planar PCS uses the local `y` offset and requires the local `x` and `z`
components to be zero. Its material `+x` axis points along the undeformed
backbone, material `+y` is the transverse direction, and positive
`kappa_z` rotates `+x` toward `+y` (counter-clockwise in a right-handed
`xy` frame). Therefore a path at positive offset `d_y` has local tangent

\[
a(s) =
\begin{bmatrix}
\sigma_x-d_y\kappa_z \\
\sigma_y+d_y'
\end{bmatrix}.
\]

In particular, a positive-`y` path lies on the inside of a
positive-curvature bend and shortens relative to the backbone.

#### Custom routing laws

A custom routing supplies immutable parameters and analytical offset
functions. Friction also needs the second derivative; production code does not
differentiate routing functions with autodiff.

```python
import equinox as eqx
import jax
import jax.numpy as jnp

from soromox.actuation import BaseThreadlikeRoutingParams, ThreadlikeRouting


class SinusoidalRoutingParams(BaseThreadlikeRoutingParams):
    amplitude: jax.Array
    frequency: jax.Array
    start_segment_index: tuple[int, ...] = eqx.field(static=True)
    end_segment_index: tuple[int, ...] = eqx.field(static=True)

    @property
    def num_paths(self) -> int:
        """Return the number of paths in this parameter batch.

        Returns:
            Number of routed paths.
        """
        return self.amplitude.shape[0]


def offset(params: SinusoidalRoutingParams, s: jax.Array) -> jax.Array:
    """Return material-frame sinusoidal routing offsets.

    Args:
        params: Batched sinusoidal routing parameters.
        s: Scalar backbone abscissa.

    Returns:
        Material-frame offsets with shape ``(num_paths, 3)``.
    """
    y = params.amplitude * jnp.sin(params.frequency * s)
    return jnp.stack((jnp.zeros_like(y), y, jnp.zeros_like(y)), axis=-1)


def derivative(params: SinusoidalRoutingParams, s: jax.Array) -> jax.Array:
    """Return the first derivative of the routing offset.

    Args:
        params: Batched sinusoidal routing parameters.
        s: Scalar backbone abscissa.

    Returns:
        First derivatives with shape ``(num_paths, 3)``.
    """
    y_s = params.amplitude * params.frequency * jnp.cos(params.frequency * s)
    return jnp.stack((jnp.zeros_like(y_s), y_s, jnp.zeros_like(y_s)), axis=-1)


def second_derivative(params: SinusoidalRoutingParams, s: jax.Array) -> jax.Array:
    """Return the second derivative of the routing offset.

    Args:
        params: Batched sinusoidal routing parameters.
        s: Scalar backbone abscissa.

    Returns:
        Second derivatives with shape ``(num_paths, 3)``.
    """
    y_ss = -(params.frequency**2) * params.amplitude * jnp.sin(
        params.frequency * s
    )
    return jnp.stack((jnp.zeros_like(y_ss), y_ss, jnp.zeros_like(y_ss)), axis=-1)


routing = ThreadlikeRouting(
    params=SinusoidalRoutingParams(
        amplitude=jnp.array([0.01]),
        frequency=jnp.array([20.0]),
        start_segment_index=(0,),
        end_segment_index=(1,),
    ),
    offset_fn=offset,
    derivative_fn=derivative,
    second_derivative_fn=second_derivative,
)
```

Path count and segment spans are static topology. Updating either requires
reconstructing the routing and robot.

### Modality presets

Let \(\ell\) be raw physical path length. Presets choose the signed or scaled
coordinate while positive input retains its named physical meaning.

| Preset | Coordinate | Positive effort | Unit |
| --- | --- | --- | --- |
| `tendons` | \(-\ell\) | tension/pulling | N |
| `push_rods` | \(+\ell\) | compression/pushing | N |
| `muscles` | \(-\ell\) | contraction/tension | N |
| `pressure_chambers` | \(A_\mathrm{eff}\ell\) | pressure | Pa |

Pressure and equivalent volume \(V=A_\mathrm{eff}\ell\) form a
work-conjugate pair.

```python
from soromox.actuation import ThreadlikeActuator

actuator = ThreadlikeActuator.tendons(routing)
robot = PCS(body_params, structure=structure, actuators=actuator)
```

Pass a tuple to combine ordered actuator groups or modalities.

### Coordinates, Jacobians, and force maps

Raw geometry is separate from the actuator coordinate and force map:

```python
actuator = robot.actuators[0]
lengths = actuator.path_lengths(robot, q)
length_rates = actuator.path_velocities(robot, q, qd)
path_length_jacobian = actuator.transmission.path_length_jacobian(robot, q)
points = actuator.path_poses(robot, q, s)

coordinates = robot.actuator_coordinates(q)
coordinate_jacobian = robot.actuator_coordinate_jacobian(q)
actuation_matrix = robot.actuation_matrix(q)
```

The two matrix contracts are

\[
\dot y_a=J_a(q)\dot q,
\qquad
\tau_u=A(q)e,
\]

where `coordinate_jacobian` is \(J_a=\partial y_a/\partial q\) with shape
`(num_actuators, num_velocities)`, and `actuation_matrix` is \(A\) with shape
`(num_velocities, num_actuators)`. For a lossless transmission, virtual work
gives \(A=J_a^T\). Friction can make them differ without changing path
coordinates or velocities.

### Continuum Capstan friction

`ThreadlikeFriction.capstan(coefficient=mu)` attenuates active threadlike effort
along increasing backbone abscissa, starting at each path's configured
anchor. It is available for `PCS`, `PlanarPCS`, and `GVS`. The default
`ThreadlikeFriction.frictionless()` returns unit effort ratio.

#### Path-turn derivation

Let \(s\) be the backbone abscissa in the undeformed configuration,
\(R(s)\in SO(3)\) the material orientation, \(p(s)\in\mathbb{R}^3\) the backbone position, and
\(\xi=(\omega,v)\) the angular and linear Cosserat strain. Then

\[
R'=R[\omega]_\times,
\qquad
p'=Rv.
\]

For material routing offset \(r(s)\), the spatial path is
\(x(s)=p(s)+R(s)r(s)\). Its derivative is

\[
x'=Ra,
\qquad
a=v+\omega\times r+r'.
\]

Define \(\nu=\lVert a\rVert\), material unit tangent \(u=a/\nu\), and
spatial unit tangent \(\hat t=Ru\). Analytical differentiation gives

\[
a'=v'+\omega'\times r+\omega\times r'+r'',
\qquad
u'=\frac{(I-uu^T)a'}{\nu}.
\]

Consequently,

\[
\frac{d\hat t}{ds}=R(\omega\times u+u'),
\qquad
\rho(s)=\left\lVert\omega\times u+u'\right\rVert.
\]

`turn_rate` is \(\rho\), in radians per metre of undeformed backbone. It
includes both material-frame rotation and change of the tangent inside that
frame. Omitting \(u'\) is valid only when the normalized material tangent is
constant. Inside a PCS segment, \(\omega'=v'=0\), so SoRoMoX uses the
constant-strain specialization \(a'=\omega\times r'+r''\). Routing derivatives
can still make \(u'\ne0\). GVS retains the full expression and computes
\(\omega'\) and \(v'\) analytically from the derivative of its strain basis.

The `accumulated_turn` from anchor \(s_0\) is

\[
\Theta(s)=\int_{s_0}^{s}\rho(\sigma)\,d\sigma.
\]

SoRoMoX evaluates this integral using the host quadrature. Genuine one-sided
tangent jumps introduced by a piecewise host discretization contribute their
unsigned angle at the interface.

#### Relation to the cited discrete model

The exponential Capstan relation is established for ropes, belts, and tendons
sliding over contact. SoRoMoX takes inspiration from a discrete spacer-disk
tendon model:

!!! info "Capstan-model reference"
    > Feliu-Talegon, D., Alkayas, A. Y., Adamu, Y. A., Mathew, A. T., &
    > Renda, F. (2025). *Actuation Reading Insights: Estimating Shape and
    > Forces in Tendon-Driven Slender Soft Robots*. IEEE/ASME Transactions on
    > Mechatronics, 30(6), 7878–7888.
    > [https://doi.org/10.1109/TMECH.2025.3581774](https://doi.org/10.1109/TMECH.2025.3581774)

In that model, each disk obeys

\[
T_i=T_{i-1}e^{-\mu\varphi_i},
\qquad
T_n=T_0\exp\left(-\mu\sum_i\varphi_i\right).
\]

SoRoMoX adapts that idea to distributed contact along an embedded continuum
path:

\[
\eta(s)=\frac{e(s)}{e(s_0)}=e^{-\mu\Theta(s)}.
\]

This is not an implementation of discrete spacer disks. The current API does
not represent discrete guide locations, guide radii, or individual guide
reactions.

#### Generalized-force integration

The local routed-length gradient with respect to material-frame strain is
\(\Phi_\xi=[r\times u;u]\). The actuator coordinate is \(y_a=c\ell\), with
\(c=-1\) for tendons and contractile muscles, \(c=+1\) for push rods, and
\(c=A_\mathrm{eff}\) for pressure chambers; see [modality presets](#modality-presets).

If \(B(s)\) maps generalized-coordinate variations to strain variations,
the active force-map column for this path is

\[
A_\mu(q)=c\int_{s_0}^{s_1}
\eta(q,s)B(s)^T\Phi_\xi(q,s)\,ds.
\]

The ratio weights every quadrature contribution; it does not merely scale the
final matrix. Distal contributions therefore receive less of the anchor effort
than proximal contributions.

```python
from soromox.actuation import ThreadlikeFriction

actuator = ThreadlikeActuator.tendons(
    routing,
    friction=ThreadlikeFriction.capstan(coefficient=0.2),
)
```

The coefficient may be a scalar shared by the family or an array of shape
`(num_paths,)`.

#### Custom effort-ratio laws

Custom laws receive only explicit friction state, not a host-specific context:

```python
from soromox.actuation import BaseThreadlikeFrictionParams, ThreadlikeFriction


class HyperbolicFrictionParams(BaseThreadlikeFrictionParams):
    coefficient: Array


def hyperbolic_effort_ratio(
    params: HyperbolicFrictionParams,
    accumulated_turn: Array,
) -> Array:
    """Return a custom effort ratio as a function of accumulated turn.

    Args:
        params: Hyperbolic friction parameters.
        accumulated_turn: Accumulated path turn with shape ``(num_paths,)``.

    Returns:
        Surviving effort ratios with shape ``(num_paths,)``.
    """
    return 1.0 / (1.0 + params.coefficient * accumulated_turn)


friction = ThreadlikeFriction(
    params=HyperbolicFrictionParams(coefficient=jnp.asarray(0.5)),
    effort_ratio_fn=hyperbolic_effort_ratio,
    is_frictionless=False,
)
```

Parameters are differentiable leaves; the law itself is static. A custom law
should return values in \((0,1]\) and must not read the applied effort.

The current model uses the driven anchor-to-distal Capstan branch. It does not
model sticking, direction-dependent backdriving, frictional hysteresis, or
slack. Warp threadlike kernels currently accept only frictionless linear
routing; automatic backend selection evaluates frictional actuation with JAX.

### Passive paths

`ThreadlikeImpedance` implements conservative routed spring-damper mechanics:

```python
from soromox.actuation import ThreadlikeImpedance

passive = ThreadlikeImpedance(
    routing=passive_routing,
    stiffness=jnp.array([50.0]),
    damping=jnp.array([2.0]),
    rest_length=jnp.array([0.19]),
)
```

It contributes elastic force, damping, and elastic energy without consuming an
active control channel. Its `.routing` attribute stays distinct from an active
actuator's `.transmission`; friction is intentionally active-only.

### Parameter updates

Updates use immutable replacement:

```python
params = robot.actuators[0].params
routing_params = params.transmission.routing.replace(intercept=new_intercepts)
transmission_params = params.transmission.replace(routing=routing_params)
robot = robot.update_actuator_params(0, transmission=transmission_params)
```

Routing coefficients, coordinate scales, friction coefficients, effort bounds,
and passive mechanical values are updateable. Path counts and segment spans are
static topology and require reconstruction.

### Rendering

Renderer adapters turn path queries into semantic visual layers. Generic
renderers draw swept geometry when a radius is configured and use polylines
otherwise. I-SUPPORT retains its detailed bellows renderer.

### Current limits

Threadlike paths are fixed in the material frame, use contiguous segment spans,
and have one constant coordinate scale per path. Active effort loss is static
and direction-independent.

## Configuration-dependent routing

`ConfigurationDependentThreadlikeRouting` describes paths whose material offsets
change with robot configuration. `StraightSpanRouting` resolves straight cables
between prescribed guides in closed form. `ShortestPathRouting` optimizes paths
inside a circular lumen. Both expose lengths, analytical configuration and
velocity Jacobians, and world-coordinate sampling through `span_positions`.
The runnable example below supplies tendon forces to the robot dynamics and
solves static equilibrium with Optimistix.

### Straight spans without wall contact

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
derivative uses segment transforms and body Jacobians relative to the span's
entry frame. Each segment is evaluated once for all tendons. GVS integrates a
local Jacobian through its existing Magnus cells, including variable strain
bases. Spans with the same segment interval share the composed transform and
Jacobian. A single traversal supplies world boundary poses for rendering;
upstream rigid motion does not enter the span length calculation.
`StraightSpanSolution` stores
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

The minimum `slsqp-jax` version is 0.21.1, the validated solver release that adds
[stagnation detection during feasibility restoration](https://github.com/lucianopaz/slsqp-jax/compare/v0.21.0...v0.21.1).
This stops restoration steps that make negligible progress in satisfying the
constraints. Routing acceptance still uses the independent checks described below.

PCS, PlanarPCS, and GVS implement the required host interface, including floating
bases. Moving GVS joints connect prescribed guides on adjacent spans. It assumes
a taut, frictionless filament with negligible bending stiffness and inertia.
Compression-driven elastic rods need a separate routing strategy.

### Parameters and solving

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

### Results and numerical accuracy

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

### Derivatives and limitations

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

### Prescribed guides at moving joints

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

The existing `SoftRobot` class provides three general kinematics operations:

- `strain_and_basis(configuration, segment_indices, points)` returns spatial
  strains and their configuration derivatives at specified material points.
- `relative_segment_kinematics(configuration)` returns link transforms, link
  body Jacobians, and preceding joint transforms. Link Jacobians have zero base
  and joint columns.
- `joint_kinematics(configuration, segment)` returns a relative joint transform,
  its spatial twist Jacobian, and constant motion generators. The generators
  allow a common fixed point to be identified for every joint state.

The `moving_joint_indices` property supplies static joint topology. Geometry
utilities compose boundary frames and transform points. The routing layer uses
these operations to evaluate guide connections, identify permanently coincident
guides, and require prescribed guides at moving joints. Straight spans share
segment evaluations and boundary composition across all guides.

GVS evaluates both sampled and cached material strains through the same local
kernel. Standard threadlike routing continues to use cached quadrature bases;
configuration basis assembly is performed only when requested. PCS and PlanarPCS
share their existing `strain` evaluation between both routing families.

### Runnable hollow-section simulations

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

### Optimization structure

Choose straight spans explicitly when contact is irrelevant. Wall-aware routing
uses one SLSQP solve per span. At a fixed backbone configuration the sampled
problem has an affine tangent norm objective, linear endpoint equalities, and
convex circular wall constraints. The solver receives analytical gradients,
constraint Jacobians, and Hessian products.

The outer static equilibrium solve belongs to the example. Optimistix updates
the robot configuration, and each residual evaluation resolves the routes at
that configuration before computing tendon forces.

### Routing interfaces

::: soromox.actuation.ConfigurationDependentThreadlikeRouting

::: soromox.actuation.StraightSpanRouting

::: soromox.actuation.ShortestPathRouting

::: soromox.actuation.RoutingSolverOptions
