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
  route only by the optional Capstan friction model described below.

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
components to be zero.

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
to quadrature tolerance. A nonzero friction coefficient breaks that identity by
design, because the moment matrix then carries a transmission loss that path
length does not; see [Guide friction](#guide-friction).

PCS integrates the local path basis and performs the constant-strain projection
once. GVS applies its strain basis inside the existing quadrature. The public
behavior is common, while each host keeps its native evaluation path.

## Guide friction

By default a path transmits its effort undiminished along its whole route. Set
`friction_coefficient` to model Coulomb friction against the guides the path
runs through, following the force model of
[Feliu-Talegon et al. (2025)](https://doi.org/10.1109/TMECH.2025.3550846). A
tendon threaded through spacer disks loses tension at every contact, and the
loss at each contact obeys the classical Capstan (Euler) law
\(T_{out} = T_{in}e^{-\mu\varphi}\), which depends on the accumulated turn
\(\varphi\) alone and not on the guide radius.

Taking the guide spacing to zero turns the product of per-contact losses into

\[
\eta(q, s) = e^{-\mu\Theta(q, s)}, \qquad
\Theta(q, s) = \int_0^s \lVert\omega \times \hat{u}\rVert\,\mathrm{d}s'
\]

the fraction of the base effort that survives to arc length \(s\), with
\(\hat{u}\) the unit routed-path tangent. Because \(\eta\) weights the integrand
of the moment matrix rather than scaling its result, the distal part of a path
contributes less than the proximal part:

\[
A_\mu(q) = B_\xi^\top \int_0^L \eta(q, s)\,\Phi(q, s)\,\mathrm{d}s
\]

\(\eta\) depends on the configuration only, so the actuation force stays linear
in the effort and every controller keeps working. Since \(\eta \in (0, 1]\), the
model can only ever remove effort, never add it, and a coefficient of zero
reproduces the frictionless matrix exactly.

The wrap density reduces to \(\lvert\kappa\rvert\) identically in the plane, for
any offset and any strain. Under torsion it recovers the helix curvature of an
off-axis path, so a twisted robot correctly reports a nonzero wrap angle.

```python
robot = PCS.from_links(
    links,
    structure=PCSStructure(num_gauss_points=5),
    # a scalar shares one coefficient; pass shape (num_paths,) for per-path values
    actuators=ThreadlikeActuator.tendons(routing, friction_coefficient=0.2),
)

tau = robot.actuation_force(q, jnp.array([5.0, 0.0]))
```

`PCS` and `PlanarPCS` implement the wrap-angle model. `GVS` strain varies along
arc length, so its wrap angle is not piecewise linear; it rejects a nonzero
coefficient during construction with a descriptive `TypeError`.

### Identifying the coefficient

The coefficient is an ordinary dynamic leaf, so it is traceable and
differentiable under `jit`:

```python
def with_friction(mu):
    transmission = robot.actuators[0].params.transmission.replace(
        friction_coefficient=mu
    )
    return robot.update_actuator_params(0, transmission=transmission)


@jax.jit
def loss(mu):
    identified = with_friction(mu)
    residual = (
        identified.elastic_force(q_meas)
        + identified.gravitational_force(q_meas)
        - identified.actuation_force(q_meas, u_meas)
    )
    return jnp.sum(residual**2)


value, grad = jax.value_and_grad(loss)(jnp.array(0.2))
```

Zero is a valid optimizer start, but \(\mathrm{d}A/\mathrm{d}\mu\) vanishes
wherever \(\Theta = 0\), so a perfectly straight pose carries no information
about \(\mu\). Set `wrap_angle_smoothing` on the host structure to replace
\(\lvert\kappa\rvert\) with \(\sqrt{\kappa^2 + \epsilon^2}\) if the fit can
approach a straight configuration. It never under-estimates the wrap angle and
over-estimates it by at most \(\epsilon L\), and it also removes the \(C^0\)
kink at zero curvature.

### What changes at a nonzero coefficient

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
  pure geometry at any coefficient.
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
`friction_coefficient`, in which case part of its spring force becomes
non-conservative and is applied through `actuation_force`; see
[Guide friction](#guide-friction).

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
