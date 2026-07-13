# Actuation model

SoRoMoX separates actuator geometry from the law that produces actuator effort.
The same actuation objects can therefore be installed on `PCS`, `PlanarPCS`, or
`GVS` without introducing an actuator-specific robot subclass.

## Work-conjugate formulation

A `Transmission` defines actuator coordinates (z(q)). Their Jacobian is the
transpose of the moment matrix:

\[
A(q) = \left(\frac{\partial z}{\partial q}\right)^T,
\qquad \dot z = A(q)^T \dot q.
\]

An `EffortModel` maps user controls (u) to efforts (e) conjugate to (z).
The generalized force and power are

\[
\tau_u = A(q)e,
\qquad \dot q^T\tau_u = \dot z^T e.
\]

`DirectEffort`, the only effort law in the initial release, implements `e = u`.
The split leaves room for later activation, pressure-flow, force-length,
saturation, and actuator-dynamics models without changing the transmission API.

## Construction

Pass one actuator or an ordered tuple through `actuators=`:

```python
robot = PCS(
    params=body_params,
    structure=structure,
    actuators=(left_actuator, right_actuator),
)
```

Tuple order defines control slices, actuator coordinates, moment-matrix columns,
metadata, and rendered layers.

- `actuators=None` installs direct identity actuation, preserving the ordinary
  `PCS`, `PlanarPCS`, and `GVS` behavior.
- `actuators=()` creates an unactuated model.
- Mixed actuator families are supported when their channel counts and robot
  routing topology are valid.

The common robot interface is:

```python
z = robot.actuator_coordinates(q)
z_dot = robot.actuator_velocities(q, q_dot)
A = robot.actuation_matrix(q)
e = robot.actuator_efforts(q, q_dot, u)
tau = robot.actuation_force(q, u, q_dot=q_dot)
metadata = robot.actuator_input_metadata
```

`actuated_coordinates(q)` remains the spelling consumed by
[actuation-space controllers](../control/actuation-space.md). The coordinate
transformation requires the moment matrix to have full column rank at its
reference configuration.

## Active and passive components

An `Actuator` combines a transmission, effort model, parameters, and ordered
control metadata. A `PassiveElement` contributes conservative force, damping,
and energy independently of active control. Passive mechanics are therefore
explicit and can be composed with any active modality. Renderer-side adapters
turn supported component geometry queries into visual layers without coupling
the physics modules to rendering types or styles.

## Parameter updates

Body parameters keep the existing API:

```python
robot = robot.update_params(young_modulus=new_modulus)
robot = robot.with_params(new_body_params)
```

Actuator and passive-element parameters use indexed delegates:

```python
robot = robot.update_actuator_params(0, lower_bounds=new_lower_bounds)
robot = robot.with_actuator_params(0, new_actuator_params)
robot = robot.update_passive_element_params(0, stiffness=new_stiffness)
```

Each component also provides `params`, `with_params(...)`, and
`update_params(...)`. Parameter replacement is immutable. Numeric fields may be
updated, while changing component type, channel count, routing count, or segment
span requires reconstructing the component and robot. Visual styling is configured
on renderers rather than stored in actuator parameters.

See [Threadlike actuation](threadlike.md) for routed continuum examples and the
[parameter guide](../utilities/parameters.md) for the general immutable update
pattern.
