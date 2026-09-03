# Threadlike actuation

Threadlike actuation models paths embedded in a continuum robot's material
frame. The same geometry supports pulling tendons, pushing rods, simplified
contractile muscles, and pressure chambers represented by an equivalent axial
volume coordinate. A path may span any contiguous inclusive range of body
segments.

Following the general routed-path formulation of
[Renda et al. (2022)](https://doi.org/10.1109/LRA.2022.3183248), a routing is a
material-frame offset from the backbone. `PCS`, `PlanarPCS`, and `GVS` combine
that routing with their native strain kinematics to evaluate path length,
coordinate velocity, and generalized actuator force.

The model assumes fixed material-frame routing, one scalar coordinate and
effort per path, and an optional static effort loss along active paths. It does
not model configuration-dependent rerouting, slack, actuator dynamics, or
changes in pressure-chamber cross-section.

## Routing

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

### Custom routing laws

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
    def num_paths(self):
        """Return the number of paths in this parameter batch."""
        return self.amplitude.shape[0]


def offset(params, s):
    """Return material-frame sinusoidal routing offsets at ``s``."""
    y = params.amplitude * jnp.sin(params.frequency * s)
    return jnp.stack((jnp.zeros_like(y), y, jnp.zeros_like(y)), axis=-1)


def derivative(params, s):
    """Return the first arc-length derivative of ``offset``."""
    y_s = params.amplitude * params.frequency * jnp.cos(params.frequency * s)
    return jnp.stack((jnp.zeros_like(y_s), y_s, jnp.zeros_like(y_s)), axis=-1)


def second_derivative(params, s):
    """Return the second arc-length derivative of ``offset``."""
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

## Modality presets

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

## Coordinates, Jacobians, and force maps

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

The singular `coordinate` in `coordinate_jacobian` follows the usual compound
noun: it is the Jacobian of the vector returned by `actuator_coordinates`.
The robot-level prefix is plural because all installed actuators are stacked.
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

## Continuum Capstan friction

`ThreadlikeFriction.capstan(coefficient=mu)` attenuates active threadlike effort
along increasing backbone coordinate, starting at each path's configured
anchor. It is available for `PCS`, `PlanarPCS`, and `GVS`. The default
`ThreadlikeFriction.frictionless()` returns unit effort ratio.

### Path-turn derivation

Let \(s\) be undeformed backbone arc length, \(R(s)\in SO(3)\) the material
orientation, \(p(s)\in\mathbb{R}^3\) the backbone position, and
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
constant. Inside a PCS segment, \(\omega'=v'=0\), but routing derivatives can
still make \(u'\ne0\). GVS computes \(\omega'\) and \(v'\) analytically from
the derivative of its strain basis.

The `accumulated_turn` from anchor \(s_0\) is

\[
\Theta(s)=\int_{s_0}^{s}\rho(\sigma)\,d\sigma.
\]

SoRoMoX evaluates this integral using the host quadrature. Genuine one-sided
tangent jumps introduced by a piecewise host discretization contribute their
unsigned angle at the interface.

### Relation to the cited discrete model

The exponential Capstan relation is established for ropes, belts, and tendons
sliding over contact. SoRoMoX takes inspiration from the discrete spacer-disk
tendon model of
[Feliu-Talegon et al. (2025)](https://doi.org/10.1109/TMECH.2025.3581774), in
which each disk obeys

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

### Generalized-force integration

For spatial strain, the local unscaled length-gradient density is
\(g_\xi=[r\times u;u]\). If \(B(s)\) maps generalized coordinates to strain
and \(c\) is the modality coordinate scale, the active force map is

\[
A_\mu(q)=c\int_{s_0}^{s_1}
\eta(q,s)B(s)^Tg_\xi(q,s)\,ds.
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

### Custom effort-ratio laws

Custom laws receive only explicit friction state, not a host-specific context:

```python
from soromox.actuation import BaseThreadlikeFrictionParams, ThreadlikeFriction


class HyperbolicFrictionParams(BaseThreadlikeFrictionParams):
    coefficient: Array


def hyperbolic_effort_ratio(params, accumulated_turn):
    """Return a custom effort ratio as a function of accumulated turn."""
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

## Passive paths

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

## Parameter updates

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

## Rendering

Renderer adapters turn path queries into semantic visual layers. Physics
objects do not import renderer primitives or store visual style. Generic
renderers draw swept geometry when a radius is configured and use polylines
otherwise. I-SUPPORT retains its detailed bellows renderer.

## Current limits

Threadlike paths are fixed in the material frame, use contiguous segment spans,
and have one constant coordinate scale per path. Active effort loss is static
and direction-independent. Discrete guides and internal actuator dynamics are
not implemented.
