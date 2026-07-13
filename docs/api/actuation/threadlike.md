# Threadlike actuation

Threadlike actuation models fixed paths routed through the material-frame
cross-section of a continuum robot. A routing may span any contiguous inclusive
range of body segments. The same geometry supports pulling tendons, pushing
rods, simplified contractile muscles, and pressure chambers represented by an
equivalent axial volume coordinate.

## Routing

`ThreadlikeRouting` stores a vectorized family of paths. Linear paths use

\[
d(s) = [0,\; y_0 + y_1s,\; z_0 + z_1s]
\]

in the local material frame, where local (x) is the backbone direction.

```python
from soromox.actuation import ThreadlikeRouting

routing = ThreadlikeRouting.linear(
    y_intercept=jnp.array([0.01, -0.01]),
    y_slope=jnp.zeros(2),
    z_intercept=jnp.zeros(2),
    z_slope=jnp.zeros(2),
    start_segment_index=(0, 0),
    end_segment_index=(1, 1),
)
```

Planar PCS uses the local `y` offset and requires the `z` fields to be zero.

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
length_rates = actuator.path_velocities(robot, q, q_dot)
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
    y_intercept=new_intercepts,
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

## Migration from continuum tendon subclasses

| Previous API | Composable API |
| --- | --- |
| `TendonActuatedPCS` | `PCS(..., actuators=ThreadlikeActuator.tendons(...))` |
| `TendonActuatedPlanarPCS` | `PlanarPCS(..., actuators=...)` |
| `TendonActuatedGVS` | `GVS(..., actuators=...)` |
| `LinearTendonRoutingParams(..., attachment_segment_index=end)` | `ThreadlikeRouting.linear(..., start_segment_index=0, end_segment_index=end)` |
| Passive tendon fields on the robot | `ThreadlikeImpedance` in `passive_elements=` |
| `active_tendon_length(q)` | `actuator.path_lengths(robot, q)` |
| Tendon-space coordinates | `robot.actuator_coordinates(q)`; these are `-length` for tendons |
| Tendon-specific renderer methods | `robot.actuator_visual_layers(...)` |

The removed continuum wrappers used a different input-sign convention in some
examples. New tendon inputs are positive tensions; do not negate them manually.

## Current limits

Threadlike paths are fixed in the material frame, use contiguous segment spans,
and have one constant coordinate scale per path. V1 uses stateless
`DirectEffort`; nonlinear and stateful effort laws are future extensions.
