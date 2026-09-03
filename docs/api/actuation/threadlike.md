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
- each path carries one scalar work-conjugate effort, uniform along its route.

Consequently, configuration- or time-dependent rerouting, changes in chamber
cross-section, along-path friction or slack, and internal actuator dynamics are
outside the transmission model. Representing these effects requires an extended
routing, transmission, or effort model.

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
equivalent volumes. In every case,
`jax.jacrev(robot.actuator_coordinates)(q) == robot.actuation_matrix(q).T` up
to quadrature tolerance.

PCS integrates the local path basis and performs the constant-strain projection
once. GVS applies its strain basis inside the existing quadrature. The public
behavior is common, while each host keeps its native evaluation path.

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
it never consumes an active control channel.

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
