# Threadlike actuation

Threadlike actuation models fixed paths routed through the material-frame
cross-section of a continuum robot. A routing may span any contiguous inclusive
range of body segments. The same geometry supports pulling tendons, pushing
rods, simplified contractile muscles, and pressure chambers represented by an
equivalent axial volume coordinate.

Following the general threadlike-routing formulation of
[Renda et al. (2022)](https://doi.org/10.1109/LRA.2022.3183248), each actuator
is represented by a path embedded in the continuum body. SoRoMoX specifies the
path by its material-frame offset \(d(s)\) from the backbone and its derivative
with respect to arc length. The host combines these routing fields with its
backbone kinematics when evaluating path length and the moment matrix.

The transmission model assumes that:

- the routing is fixed in the material frame and depends only on arc length;
- the actuator coordinate is determined by path length, scaled by a constant
  effective area for the pressure-chamber preset; and
- each path carries one scalar work-conjugate effort, attenuated along its
  route by an optional friction model described below.

Consequently, configuration- or time-dependent rerouting, changes in chamber
cross-section, path slack, and internal actuator dynamics are outside the
transmission model. Representing these effects requires an extended routing,
transmission, or effort model.

Threadlike components require continuum segment topology and the host's native
path-integration hooks. `PCS`, `PlanarPCS`, and `GVS` provide this contract.
Articulated hosts do not, so installing a threadlike component on them raises a
descriptive `TypeError` during construction.

## Routing

`ThreadlikeRouting` stores a vectorized family of paths. Linear paths use

\[
d(s) = d_0 + d_1s
\]

in the local material frame. The `intercept` and `slope` arrays have shape
`(num_paths, 3)`, with the final axis ordered along the local material-frame
axes. SoRoMoX aligns the backbone with the local `x` axis; the local-`x`
component of both arrays must therefore be zero, while the `y` and `z`
components locate the path in the cross-section. A `(3,)` vector defines a
single path.

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

### Custom routing laws

The continuum hosts depend only on the routing contract, not on the linear
parameterization. A custom fixed material-frame routing supplies an immutable
parameter type plus offset and arc-length-derivative functions:

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
    def num_paths(self):
        return self.amplitude.shape[0]

    def validate(self):
        if self.frequency.shape != self.amplitude.shape:
            raise ValueError("frequency must match amplitude")


def offset(params, s):
    y = params.amplitude * jnp.sin(params.frequency * s)
    return jnp.stack((jnp.zeros_like(y), y, jnp.zeros_like(y)), axis=-1)


def derivative(params, s):
    y_s = params.amplitude * params.frequency * jnp.cos(params.frequency * s)
    return jnp.stack((jnp.zeros_like(y_s), y_s, jnp.zeros_like(y_s)), axis=-1)


routing = ThreadlikeRouting(
    params=SinusoidalRoutingParams(
        amplitude=jnp.array([0.01]),
        frequency=jnp.array([20.0]),
        start_segment_index=(0,),
        end_segment_index=(1,),
    ),
    offset_fn=offset,
    derivative_fn=derivative,
)
```

`PCS`, `PlanarPCS`, and `GVS` evaluate these two methods through the same host
integration hooks. Custom parameter types must retain the common contiguous
`start_segment_index` / `end_segment_index` topology contract.

## Modality presets

Raw path length is ℓ. Presets select the signed or scaled coordinate while
positive input always means the named physical action.

| Preset | Coordinate | Positive effort | Unit |
| --- | --- | --- | --- |
| `tendons` | \(-\ell\) | tension/pulling | N |
| `push_rods` | \(+\ell\) | compression/pushing | N |
| `muscles` | \(-\ell\) | contraction/tension | N |
| `pressure_chambers` | \(A_\mathrm{eff}\ell\) | pressure | Pa |

The pressure coordinate is volume (V=A_\mathrm{eff}\ell), so pressure and
volume remain a work-conjugate pair.

```python
from soromox.actuation import ThreadlikeActuator
from soromox.systems import PCS

actuator = ThreadlikeActuator.tendons(routing)
robot = PCS(body_params, structure=structure, actuators=actuator)
```

The same construction works with `PlanarPCS` and `GVS`. Mixed modalities are an
ordered tuple:

```python
robot = PCS(
    body_params,
    structure=structure,
    actuators=(
        ThreadlikeActuator.tendons(tendon_routing),
        ThreadlikeActuator.push_rods(rod_routing),
    ),
)
```

## Geometry and work coordinates

Raw geometry is intentionally separate from the signed/scaled actuator
coordinate:

```python
actuator = robot.actuators[0]
lengths = actuator.path_lengths(robot, q)
length_rates = actuator.path_velocities(robot, q, qd)
points = actuator.path_poses(robot, q, s)
coordinates = robot.actuator_coordinates(q)
```

For tendons, `coordinates == -lengths`; for a pressure preset, coordinates are
equivalent volumes. Without guide friction,
`jax.jacrev(robot.actuator_coordinates)(q) == robot.actuation_matrix(q).T` up
to quadrature tolerance. A friction model breaks that identity by design,
because the moment matrix then carries an attenuation that path length does
not; see [Routing friction](#routing-friction).

PCS integrates the local length gradient and performs the constant-strain projection
once. GVS applies its strain basis inside the existing quadrature. The public
behavior is common, while each host keeps its native evaluation path.

## Routing friction

Ideally, effort is transmitted undiminished along its route, from the motor
until its attachment point. However, friction between the routing and the
robot body is often non-negligible in real applications, and attenuates the
effort along the path. It is possible to simulate this behavior via the
`ThreadlikeFriction` model, which is installed on a `ThreadlikeTransmission`
through the `friction=` keyword of `ThreadlikeActuator` and
`ThreadlikeImpedance`.
The current implemented models are the following:

| Model | Ratio | Parameter | Hosts |
|---|---|---|---|
| `ThreadlikeFriction.frictionless()` | \(1\) | none | all; the default |
| `ThreadlikeFriction.exponential_length(...)` | \(e^{-k\ell}\) | `rate` \(k\), per metre | all |
| `ThreadlikeFriction.capstan(...)` | \(e^{-\mu\Theta}\) | `coefficient` \(\mu\), Coulomb; `eps` \(\epsilon\), per metre | `PCS`, `PlanarPCS` |


### Friction models

The exponential length and Capstan models follow from the same tension balance along the routed path, with \(T\) the tension, \(\lambda\) the attenuation rate per unit arc length, and \(s_0\) the abscissa of the path anchor:

\[
\frac{\mathrm{d}T}{\mathrm{d}s} = -\lambda\,T,
\qquad
\eta(s) = \frac{T(s)}{T(s_0)} = \exp\!\left(-\int_{s_0}^{s}\lambda\,
\mathrm{d}s'\right)
\]

They differ in the origin of the normal load \(N\) pressing the path against the
body.

**Configuration-independent friction.** A first approximation scales uniformly the tension along the path, \(N = cT\), so \(\lambda = \mu c \equiv k\) is
constant and the attenuation depends only on the arc length \(\ell = s - s_0\)
travelled from the anchor:

\[
\eta(\ell) = e^{-k\ell}
\]

**Configuration-dependent friction (Capstan).** The normal load is induced by
the curvature of the path, \(N = Tk_p\), so \(\lambda = \mu k_p(q, s)\):

\[
k_p(q, s) = \lVert\omega \times \hat{u}\rVert,
\qquad
\eta(q, s) = e^{-\mu\Theta(q, s)},
\quad
\Theta(q, s) = \int_{s_0}^{s} k_p(q, s')\,\mathrm{d}s'
\]

Here \(\omega\) is the angular strain, \(\hat{u}\) the unit routed-path tangent,
and \(\Theta\) the wrap angle accumulated from the anchor. Since
\(\mathrm{d}\hat{u}/\mathrm{d}s = \omega \times \hat{u}\), only the component of
\(\omega\) orthogonal to \(\hat{u}\) contributes, so torsion about the axis of
the path does not attenuate. In the plane \(k_p\) reduces to
\(\lvert\kappa_z\rvert\); under torsion an off-axis path traces a helix, whose
curvature \(k_p\) recovers.

Because \(\Theta\) vanishes at a straight configuration, so does
\(\partial\eta/\partial\mu\), and \(\mu\) is not identifiable there. The `eps`
argument floors the curvature at \(\epsilon\), replacing \(k_p\) by
\(\sqrt{k_p^2 + \epsilon^2}\). It never under-estimates \(\Theta\) and
over-estimates it by at most \(\epsilon L\); the default \(\epsilon = 0\)
reproduces the exact curvature.

This model is the continuum limit of the discrete tendon force model of
[Feliu-Talegon et al. (2025)](https://doi.org/10.1109/TMECH.2025.3550846), in
which the tendon passes through a finite set of spacer disks and each contact
attenuates the tension by the classical Capstan (Euler) relation

\[
T_i = T_{i-1}\,e^{-\mu\varphi_i}
\qquad\Longrightarrow\qquad
T_n = T_0\,\exp\!\left(-\mu\sum_{i=1}^{n}\varphi_i\right)
\]

with \(\varphi_i\) the turn at the \(i\)-th contact, independent of the guide
radius. As the disk spacing tends to zero at fixed total turning,
\(\sum_i \varphi_i \to \Theta\) and the product recovers \(\eta\) above, which
can then be evaluated at every quadrature node rather than at discrete guides.

### Applying the friction

\(\eta\) weights the integrand of the moment matrix rather than scaling its
result, so the distal part of a path contributes less than the proximal part:

\[
A_\mu(q) = B_\xi^\top \int_0^L \eta(q, s)\,
\frac{\partial\lVert t\rVert}{\partial\xi}(q, s)\,\mathrm{d}s
\]


```python
robot = PCS.from_links(
    links,
    structure=PCSStructure(num_gauss_points=5),
    # a scalar shares one coefficient; pass shape (num_paths,) for per-path values
    actuators=ThreadlikeActuator.tendons(
        routing, friction=ThreadlikeFriction.capstan(coefficient=0.2)
    ),
)

tau = robot.actuation_force(q, jnp.array([5.0, 0.0]))
```

### Custom friction models

A model is a plain function of `(params, context)`, paired with its parameters by
`ThreadlikeFriction`. The hosts depend only on that contract, not on the shipped
models:

```python
from soromox.actuation import BaseThreadlikeFrictionParams, ThreadlikeFriction


class HyperbolicFrictionParams(BaseThreadlikeFrictionParams):
    a: Array


def hyperbolic_transmission_ratio(params, context):
    return 1.0 / (1.0 + params.a * context.wrap_angle)


actuator = ThreadlikeActuator.tendons(
    routing,
    friction=ThreadlikeFriction(
        params=HyperbolicFrictionParams(a=jnp.asarray(0.5)),
        ratio_fn=hyperbolic_transmission_ratio,
        requires_wrap_angle=True,
        is_frictionless=False,
    ),
)
```

Parameters live in the transmission parameter tree, so they stay differentiable
and replaceable; the model itself is static, so swapping it leads to retracing.
This mirrors how `ThreadlikeRouting` pairs routing parameters with a static
offset function.

The `context` is a `ThreadlikeQuadratureContext` describing the routed path at
one quadrature node.

| Field | Meaning | Portable |
|---|---|---|
| `abscissa` | backbone coordinate \(s\), from the robot base | yes |
| `path_abscissa` | arc length travelled from this path's anchor | yes |
| `segment_index` | segment containing the node | yes |
| `offset`, `offset_derivative` | material-frame offset `(n, 3)` and its \(s\)-derivative | yes |
| `tangent_norm` | routed-path stretch relative to the backbone | yes |
| `active` | whether the node lies in each path's span | yes |
| `unit_tangent` | normalized tangent, `(n, 2)` planar and `(n, 3)` spatial | no |
| `strain` | `(n, 3)` planar and `(n, 6)` spatial | no |
| `wrap_angle` | accumulated turn \(\Theta\), or `None` | only where supplied |

Set `requires_wrap_angle=False` if the model does not read `wrap_angle`, and the
hosts will neither compute it nor refuse the model. Set `is_frictionless=True`
only
for a model that always returns one, and the hosts will skip it entirely. A model
must keep its ratio in \((0, 1]\) and must not read the effort, which would make
the actuation force nonlinear in the control.

### What changes under a friction model

- `moment_matrix` is no longer the transpose of the length gradient, so
  `actuation_space` conversions that assume the identity do not apply.
- The generalized actuation force has no potential: work around a closed
  configuration loop at constant effort is nonzero and orientation dependent.
- For a loaded passive path, `actuation_force(q, u)` is affine in `u` rather
  than linear, and `actuation_force(q, 0)` is nonzero. Prefer
  `actuation_force(q, u)` over `actuation_matrix(q) @ e` wherever passive
  elements exist.
- The moment matrix acquires a mild dependence on `num_gauss_points`, because
  the attenuated integrand is no longer polynomial.
- `path_lengths`, `path_velocities`, and `path_poses` are unchanged. They are
  pure geometry under any friction model.
- The model assumes the effort drives the path. A back-driven path is still
  modelled with the driving branch of Euler's law, so keep the passive
  coefficient conservative or zero for long dynamic rollouts with
  sign-changing path rates.

## Passive paths

Use `ThreadlikeImpedance` for routed spring-damper mechanics:

```python
from soromox.actuation import ThreadlikeImpedance

passive = ThreadlikeImpedance(
    routing=passive_routing,
    stiffness=jnp.array([50.0]),
    damping=jnp.array([2.0]),
    rest_length=jnp.array([0.19]),
)

robot = PCS(
    body_params,
    structure=structure,
    actuators=actuator,
    passive_elements=(passive,),
)
```

The passive element contributes to elastic force, damping, and elastic energy;
it never consumes an active control channel. A passive path may also declare a
`friction=` model, in which case part of its spring force becomes
non-conservative and is applied through `actuation_force`; see
[Routing friction](#routing-friction).

## Updating routing and actuator parameters

Updates follow the same shallow immutable replacement pattern as robot body
parameters:

```python
params = robot.actuators[0].params
routing_params = params.transmission.routing.replace(
    intercept=new_intercepts,
)
transmission_params = params.transmission.replace(routing=routing_params)
robot = robot.update_actuator_params(
    0,
    transmission=transmission_params,
)
```

Routing coefficients, coordinate scales, effort bounds, and passive mechanical
values are updateable. Segment spans and path counts are static topology and
require reconstruction. Colors, radii, and line widths belong to renderer
configuration rather than actuator parameters.

## Rendering

The renderer-side actuation adapter converts threadlike path queries into semantic
`ActuatorVisualLayer` objects with path points, modality kind, and optional input
scalar data. Physics objects do not import renderer primitives or store visual
style. Generic renderers draw swept geometry when a radius is configured and fall
back to polylines otherwise.

```python
from soromox.rendering import ActuatorStyleConfig, RendererColorConfig

visuals = RendererColorConfig(
    actuators=ActuatorStyleConfig(
        kind_colors={"tendon": (0.85, 0.2, 0.15)},
        kind_radii={"tendon": 5e-4},
    )
)
renderer.show(q, actuator_inputs=u, color_config=visuals)
```

I-SUPPORT retains its detailed bellows renderer. Its renderer accepts the same
`actuator_inputs=` argument and also supports `pressures=` as a mutually
exclusive convenience alias.

## Current limits

Threadlike paths are fixed in the material frame, use contiguous segment spans,
and have one constant coordinate scale per path. The current implementation uses
stateless `DirectEffort`; nonlinear and stateful effort laws are future
extensions.
