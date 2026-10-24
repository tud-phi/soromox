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

Use these strategies when a filament can move relative to the backbone:

- `StraightSpanRouting` connects prescribed guides with straight lines. Use it
  when you can guarantee that wall contact is irrelevant throughout operation.
- `ShortestPathRouting` optimizes each span inside a circular hollow passage
  (the lumen), using `slsqp-jax`, which is installed with SoRoMoX.

Both support PlanarPCS, PCS, and GVS, including floating bases and prescribed
guides at moving joints. They describe taut, frictionless filaments with
negligible bending stiffness and inertia. Slack, friction, joint housing
collisions, and elastic rod buckling require additional models.

### Define guides and choose a strategy

A **path** is one physical filament. A **span** is its portion between two
prescribed guides. `RoutingSpanParams` contains the guides and their connectivity:

- Offsets have shape `(num_spans, 3)`, in local `[0, y, z]` order and metres.
  PlanarPCS requires zero `z` offsets.
- Segment indices are zero-based and inclusive.
- Spans of the same path must be ordered and contiguous. A segment transition
  inside a span does not add a guide.
- `span_path_index` groups spans into paths numbered from zero.

For an existing one-segment `robot` and its `configuration`, start with:

```python
import equinox as eqx
import jax
import jax.numpy as jnp

from soromox.actuation import (
    CircularLumenParams,
    RoutingSpanParams,
    ShortestPathRouting,
    ShortestPathRoutingParams,
    StraightSpanRouting,
)

jax.config.update("jax_enable_x64", True)

spans = RoutingSpanParams(
    start_offset=[[0.0, 0.002, 0.0]],
    end_offset=[[0.0, 0.002, 0.0]],
    span_path_index=(0,),
    start_segment_index=(0,),
    end_segment_index=(0,),
)
routing = StraightSpanRouting(spans)
```

To account for wall contact, use the same guides with a lumen radius per segment:

```python
routing = ShortestPathRouting(
    ShortestPathRoutingParams(
        spans,
        CircularLumenParams.constant(0.005, robot.num_segments),
    )
)
```

`inner_radius` has shape `(robot.num_segments,)`. Each guide must lie inside
its segment's lumen. Use FP64 for the default numerical tolerances.
`with_params` replaces numerical geometry while preserving topology; changing
path or segment connectivity requires constructing new routing parameters.

### Solve, apply forces, and sample geometry

```python
solve = eqx.filter_jit(routing.solve_routing)
solution = solve(robot, configuration)
if not bool(solution.converged):
    raise RuntimeError("Routing did not converge")

lengths = solution.path_lengths
tension = jnp.full(spans.num_paths, 0.1)
jacobian = routing.path_velocity_jacobian(robot, configuration, solution)
force = -jacobian.T @ tension
positions = routing.span_positions(
    robot, configuration, solution, 0, jnp.linspace(0.0, 1.0, 50)
)
```

`path_lengths` has one entry per physical path, including connections between
adjacent span guides. `span_positions` returns world coordinates in metres.
For optimized routes, `span_offsets` also samples material offsets at specified
backbone coordinates. Use a previous optimized solution as an explicit warm start:

```python
next_solution = solve(robot, next_configuration, initial_guess=solution)
```

`path_length_jacobian` differentiates lengths with respect to configuration
coordinates. Use `path_velocity_jacobian` for forces: spatial floating bases
have different numbers of pose coordinates and velocity components. Rigid base
motion contributes zero to internal path lengths and forces.

These routing strategies currently expose geometry and derivatives separately
from `ThreadlikeActuator`. The example below assembles their tendon forces.

### Guides at moving joints

Joints have no modeled thickness. End a span before each moving joint and start
another after it, assigning both to the same path. A straight cable connects
the upstream and downstream guides and contributes to length and force.

For one path through two sections, with guides at their transition:

```python
offsets = [[0.0, 0.004, 0.0], [0.0, 0.004, 0.0]]
spans = RoutingSpanParams(
    start_offset=offsets,
    end_offset=offsets,
    span_path_index=(0, 0),
    start_segment_index=(0, 1),
    end_segment_index=(0, 1),
)
```

Zero offsets give centered guides. At a rotational joint, centered guides remain
coincident and contribute zero connecting length and direct joint torque.
Offset guides can separate under rotation; translation can separate both kinds.
Inspect `solution.guides` for connection lengths and their `differentiable` flags.
A moving connection collapsing to zero length has no unique derivative; its
Jacobian row returns NaNs. Permanently coincident guides have zero derivatives.

### Check accuracy

Always check `solution.converged`. Straight spans check finite geometry and
positive length; wall clearance is the caller's responsibility. Optimized spans
check endpoint error, optimality, and wall containment independently of the
optimizer's termination code.

`RoutingSolverOptions` separates the solver stopping tolerance from physical
limits: `endpoint_tolerance` defaults to `1e-8` m and `wall_tolerance` to `1e-6` m.
Each optimized span reports `dense_wall_violation` at validation samples and
`wall_violation_bound`, a conservative bound between them. Acceptance uses the
bound. Increase `validation_samples_per_element` when the bound is overly
conservative; refine the routing mesh and wall constraint samples when actual
penetration is excessive. Configure these through
`PiecewiseLinearRoutingDiscretization`.

First length Jacobians are analytical. Differentiation through optimized
coefficients additionally requires a regular solution with a stable set of wall
contacts. These derivatives can be NaN even when the forward route converges.
When differentiating forces, retain the solution's dependence on configuration
through `solve_routing`. The [reference paper](https://doi.org/10.1109/TRO.2026.3714660)
describes the routing formulation and force derivatives.

### Run the hollow-section example

Install `soromox[examples]` and FFmpeg, then run:

```bash
OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/soromox-mpl \
  python examples/simulation/pcs/simulate_hollow_tendon_routing.py --check-refinement
```

The example actuates two hollow planar sections, with tendons anchored at each
section's ends. It renders resolved routing, simulates dynamics with routing
resolved at every RK4 stage, and solves static equilibrium using Optimistix.
The parameters are illustrative. The included cases remain clear of the wall;
extended wall contact can still cause rejected SLSQP solves.

Outputs include PNG cutaways, MP4 animations, NPZ trajectories, and a JSON report
under `examples/simulation/pcs/figures/hollow_tendon_routing/`. Use `--case
same-side` or `--case opposed` to select a layout, `--output` to choose a directory,
and `--render-only` to redraw saved trajectories. `--check-refinement` compares
a smaller time step and a finer routing mesh.

### Routing interfaces

::: soromox.actuation.ConfigurationDependentThreadlikeRouting

::: soromox.actuation.StraightSpanRouting

::: soromox.actuation.ShortestPathRouting

::: soromox.actuation.RoutingSolverOptions
