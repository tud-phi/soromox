# Joint-space actuation

Joint-space transmissions map generalized configurations to work-conjugate
actuator coordinates. The generic `AffineJointTransmission` supports any host
with a matching generalized-coordinate dimension. The articulated-tendon
presets additionally require serial articulated routing, as provided by
`Pendulum` and `ArticulatedSoftRobot`.

## Affine transmission coordinates

`AffineJointTransmission` defines

\[
y_\mathrm{a}(q)
= R(q-q_\mathrm{ref}) + y_{\mathrm{a},0},
\qquad
A = R^\mathsf{T}.
\]

Here, each row of \(R\) is a signed actuator routing across the joints. The
offset \(y_{\mathrm{a},0}\) makes the coordinate value explicit at the reference
configuration. The corresponding velocity, generalized force, and power are

\[
\dot y_\mathrm{a}=R\dot q,
\qquad
\tau=Ae,
\qquad
\dot q^\mathsf{T}\tau=\dot y_\mathrm{a}^\mathsf{T}e.
\]

The generic affine transmission places no tendon-specific constraints on
\(R\), so it is also suitable for custom generalized-coordinate work
coordinates. For example, the transmission itself can map PCS coordinates when
its matrix width matches the PCS degrees of freedom. The articulated-tendon
preset adds stronger serial-joint semantics and intentionally rejects continuum
hosts.

The `coordinate_offset` parameter specifies the actuator-space coordinate at
the reference configuration:

\[
y_\mathrm{a}(q_\mathrm{ref}) = y_{\mathrm{a},0}.
\]

It defines the origin of the affine coordinate, while the differential
kinematics remain \(A=R^\mathsf{T}\). For an
`ArticulatedTendonActuator` with `DirectEffort`, the generalized actuation
force is \(\tau=R^\mathsf{T}u\). For `ArticulatedTendonImpedance`,
\(y_\mathrm{a}\) is the elastic deformation; consequently, the elastic force
at the reference configuration is
\(\tau_\mathrm{elastic}=R^\mathsf{T}K y_{\mathrm{a},0}\).

## Articulated tendons

`ArticulatedTendonActuator` interprets \(R\) as a signed tendon-contraction
Jacobian. Positive control input is positive tendon tension.

```python
import jax.numpy as jnp

from soromox.actuation import ArticulatedTendonActuator
from soromox.systems import Pendulum

routing = jnp.array([
    [-0.02, -0.02, 0.0],
    [0.02, 0.02, 0.02],
])

actuator = ArticulatedTendonActuator.from_routing(
    routing,
    reference_configuration=jnp.zeros(3),
    coordinate_offset=jnp.array([0.018, 0.025]),
)
robot = Pendulum(params=body_params, actuators=actuator)
```

Tendon routing rows must be linearly independent and may not skip an
intermediate joint: after a tendon stops crossing joints, its remaining routing
entries must be zero. These checks belong to the tendon preset; the generic
affine transmission does not impose them.

Multiple actuator families can be installed as an ordered tuple. Their control
slices, coordinates, moment-matrix columns, and metadata follow tuple order.

## Passive tendon impedance

Passive routed spring-damper mechanics are represented independently of active
control:

```python
from soromox.actuation import ArticulatedTendonImpedance

passive = ArticulatedTendonImpedance.from_routing(
    routing,
    stiffness=jnp.array([80.0, 80.0]),
    damping=jnp.array([1.5, 1.5]),
    reference_configuration=jnp.zeros(3),
    coordinate_offset=jnp.array([0.018, 0.025]),
)

robot = Pendulum(
    params=body_params,
    actuators=actuator,
    passive_elements=(passive,),
)
```

For diagonal actuator-space stiffness (K) and damping (D),

\[
E=\tfrac12 y_\mathrm{a}^\mathsf{T}K y_\mathrm{a},
\qquad
\tau_\mathrm{elastic}=R^\mathsf{T}K y_\mathrm{a},
\qquad
D_q=R^\mathsf{T}DR.
\]

Body stiffness and damping remain part of the articulated host. Passive-element
contributions are composed with them in elastic force, damping, and energy;
`stiffness_matrix` continues to describe the body alone.

## Optional generalized velocity

The common calls are

```python
yad = robot.actuator_velocities(q, qd)
effort = robot.actuator_efforts(q, u, qd=qd)
tau = robot.actuation_force(q, u, qd=qd)
```

The `qd` argument is optional for effort and generalized-force evaluation. If
it is omitted, zero generalized velocity is used. This is exactly equivalent
for `DirectEffort`, which returns the control without using coordinate or
velocity. Velocity-dependent effort models require the actual `qd`; forward
dynamics supplies it automatically.

Passive damping remains separate and contributes through
`passive_damping_matrix(q) @ qd`.

## Parameter updates

Body, actuator, and passive-element parameters are updated independently:

```python
robot = robot.update_params(mass=new_mass)

params = robot.actuators[0].params
transmission = params.transmission.replace(
    coordinate_offset=new_coordinate_offset,
)
robot = robot.update_actuator_params(0, transmission=transmission)

robot = robot.update_passive_element_params(0, stiffness=new_stiffness)
```

Numeric values can be replaced immutably. Changing the transmission type,
channel count, or routing-matrix shape changes structural topology and requires
reconstructing the component and robot.
