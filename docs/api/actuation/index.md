# Actuation model

SoRoMoX separates actuator geometry from the law that produces actuator effort.
The same actuation objects can therefore be installed on compatible continuum
or articulated hosts without introducing an actuator-specific robot subclass.

## Work-conjugate formulation

A `Transmission` defines actuator coordinates \(y_\mathrm{a}(q)\). Their Jacobian is the
transpose of the moment matrix:

\[
A(q) = \left(\frac{\partial y_\mathrm{a}}{\partial q}\right)^T,
\qquad \dot y_\mathrm{a} = A(q)^T \dot q.
\]

An `EffortModel` maps user controls \(u\) to efforts \(e\) conjugate to
\(y_\mathrm{a}\).
The generalized force and power are

\[
\tau_u = A(q)e,
\qquad \dot q^T\tau_u = \dot y_\mathrm{a}^T e.
\]

`DirectEffort`, the currently available effort law, implements `e = u`.
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

## Host compatibility

Actuator installation validates the geometric host contract before dynamics or
rendering is evaluated. Incompatible combinations raise a descriptive
`TypeError` or `ValueError`; they are not deferred to a missing-method failure.

| Component | Compatible hosts | Contract |
| --- | --- | --- |
| `IdentityActuator` | Every `SoftRobot` | One channel per generalized coordinate |
| `AffineJointTransmission` | `PCS`, `PlanarPCS`, `GVS`, `Pendulum`, `ArticulatedSoftRobot`, or any host with a matching generalized-coordinate dimension | One matrix column per generalized coordinate |
| `ArticulatedTendonActuator`, `ArticulatedTendonImpedance` | `Pendulum`, `ArticulatedSoftRobot`, and hosts explicitly exposing serial articulated routing | Joint-index routing with no skipped joints and full row rank |
| Threadlike actuator and impedance presets | `PCS`, `PlanarPCS`, `GVS`, and continuum hosts implementing the threadlike integration hooks | Material-frame path geometry plus continuum segment topology |
| `ArticulatedMcKibbenActuator` | Spatial articulated hosts implementing the kinematic-frame contract, including `McKibbenActuatedUMArm` | Grouped attachment geometry and valid joint-pair indices |

Compatibility of an `AffineJointTransmission` depends only on the width of its
routing matrix: `routing_matrix.shape[1] == robot.num_dofs`. It can therefore
map GVS, PCS, or articulated generalized coordinates. The articulated-tendon
preset adds serial-joint routing constraints and rejects PCS/GVS hosts, while
threadlike presets require continuum path-integration support.

The common robot interface is:

```python
y_a = robot.actuator_coordinates(q)
y_a_dot = robot.actuator_velocities(q, qd)
A = robot.actuation_matrix(q)
e = robot.actuator_efforts(q, u, qd=qd)
tau = robot.actuation_force(q, u, qd=qd)
metadata = robot.actuator_input_metadata
```

`qd` is optional for effort and force evaluation. For an effort law that
requires velocity, omitting it evaluates the law at zero actuator velocity.

[Actuation-space controllers](../control/actuation-space.md) consume the same
`actuator_coordinates(q)` contract. The coordinate transformation requires the
moment matrix to have full column rank at its reference configuration.

## Active and passive components

An `Actuator` combines a transmission, effort model, parameters, and ordered
control metadata. A `PassiveElement` contributes conservative force, damping,
and energy independently of active control. Passive mechanics are therefore
explicit and can be composed with any active modality. Renderer-side adapters
turn supported component geometry queries into visual layers without coupling
the physics modules to rendering types or styles.

## Parameter updates

Body parameters use top-level or component-specific immutable updates:

```python
robot = robot.update_link_params(stiffness=new_stiffness)
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
span requires reconstructing the component and robot. The `with_*_params`
delegates are JAX-transformable when PyTree structure, shapes, dtypes, and
topology remain fixed. Threadlike segment spans and McKibben joint-pair indices
are precomputed topology; compiled numeric updates retain them. Visual styling
is configured on renderers rather than stored in actuator parameters.

See [Joint-space actuation](joint-space.md) for affine articulated coordinates,
[Threadlike actuation](threadlike.md) for routed continuum examples, and the
[parameter guide](../utilities/parameters.md) for the general immutable update
pattern.
