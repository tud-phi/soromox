# Tendon Actuated Pendulum

The `TendonActuatedPendulum` system is the cable-driven planar entry in the
articulated systems family. It extends the basic pendulum model with active and
passive tendon actuation.

## Overview

The tendon actuated pendulum system implements classical rigid-body dynamics for articulated robots with revolute joints and tendon-based actuation. This system combines the simplicity of planar pendulum dynamics with the complexity of cable-driven mechanisms, making it useful for:

- Cable-driven robot research and development
- Comparison with continuum soft robot models
- Study of hybrid active/passive actuation systems
- Educational demonstrations of tendon routing principles

The system supports both **active tendons** (controlled via input torques) and **passive tendons** (attached to spring-damper systems at the base).

## Key Features

- **Active Tendon Control**: Direct control through motor inputs with configurable pulley routing
- **Passive Tendon Mechanics**: Spring-damper systems providing automatic compliance and stability
- **Flexible Routing**: Configurable tendon paths through pulleys at each joint
- **JAX Compatibility**: Full support for JIT compilation, automatic differentiation, and vectorization
- **Energy Methods**: Computation of kinetic, potential, and elastic energies

## Mathematical Model

### Active Tendons

Active tendons are routed through pulleys with radii specified by the routing matrix **R_at**. The actuation matrix is **A_at = R_at^T**, and the contribution to joint torques is:

```text
τ_at = A_at * u
```

where **u** are the motor inputs.

`active_tendon_reference_configuration` is the joint configuration `q_ref_at`
used for active tendon displacement, `R_at * (q - q_ref_at)`.

### Passive Tendons

Passive tendons are attached to springs and dampers at the base, with:

- Routing matrix **R_pt** specifying pulley radii
- Stiffness coefficients **k_pt** and damping coefficients **d_pt**
- Initial pre-stretch lengths **l_pt0**
- Reference configuration **q_ref_pt** where passive displacement is measured

The passive tendon contributions are:

```text
τ_el_pt = A_pt * K_pt * (R_pt * (q - q_ref_pt) + l_pt0)
τ_d_pt = A_pt * D_pt * R_pt * qd
```

### Total System Dynamics

The equation of motion becomes:

```text
B(q) * qdd + C(q,qd) * qd + G(q) + K_total * q + D_total * qd = A_at * u + τ_ext
```

where:

- **K_total = K + A_pt * K_pt * R_pt** (total stiffness matrix)
- **D_total = D + A_pt * D_pt * R_pt** (total damping matrix)

## Usage

### Basic Setup

```python
import jax.numpy as jnp
from soromox.systems import (
    PassiveTendonParams,
    PendulumParams,
    TendonActuatedPendulum,
    TendonActuatedPendulumParams,
)

# Base pendulum parameters
body_params = PendulumParams(
    mass=jnp.array([10.0, 6.0]),
    moment_inertia=jnp.array([3.0, 2.0]),
    length=jnp.array([2.0, 1.0]),
    center_of_mass_length=jnp.array([1.0, 0.5]),
    joint_stiffness=jnp.diag(jnp.array([1.5, 2.0])),
    joint_damping=jnp.zeros((2, 2)),
    joint_rest_configuration=jnp.zeros(2),
    radius=jnp.array([0.05, 0.05]),
)

params = TendonActuatedPendulumParams(
    body=body_params,
    active_routing_matrix=jnp.array([[0.2, -0.25]]),
    passive_routing_matrix=jnp.array([[-0.15, 0.3]]),
    active_tendon_reference_configuration=jnp.zeros(2),
    passive_tendon_reference_configuration=jnp.zeros(2),
    passive_tendon=PassiveTendonParams(
        stiffness=jnp.array([2.75]),
        damping=jnp.array([0.1]),
        rest_length_offset=jnp.array([0.5]),
    ),
)

# Create robot instance
robot = TendonActuatedPendulum(params=params)
```

### Forward Kinematics and Dynamics

```python
# Configuration and velocities
q = jnp.array([jnp.pi/8, -jnp.pi/4])
qd = jnp.array([0.0, 0.0])

# Forward kinematics
chi_tips = robot.forward_kinematics_tips(q)    # (N, 3): [theta, px, py] per link tip
J_tips = robot.jacobians_tips(q)              # (N, 3, N): Jacobians for each tip

# Dynamic matrices
B = robot.inertia_matrix(q)                   # Inertia matrix
C = robot.coriolis_matrix(q, qd)              # Coriolis matrix
G = robot.gravitational_force(q)             # Gravity vector
K_total = robot.stiffness_matrix()           # Total stiffness matrix
D_total = robot.damping_matrix(q)            # Total damping matrix

# Actuation
A_at = robot.actuation_matrix(q)             # Active tendon actuation matrix
u = jnp.array([1.0])                        # Motor input torques

# Forward dynamics
y = jnp.concatenate([q, qd])                 # State vector [q, qd]
yd = robot.forward_dynamics(0.0, y, (u,))    # State derivative [qd, qdd]
```

### Energy Analysis

```python
# Energy computations
T = robot.kinetic_energy(q, qd)              # Kinetic energy
U_g = robot.gravitational_energy(q)         # Gravitational potential energy  
U_K = robot.elastic_energy(q)               # Elastic potential energy (total)
E_total = T + U_g + U_K                      # Total mechanical energy

# Tendon-specific quantities
L_passive = robot.passive_tendon_length(q)   # Passive tendon lengths
```

### Parameter Updates

```python
# Update robot parameters (returns new instance)
body_updated = robot.params.body.replace(mass=jnp.array([12.0, 7.0]))
robot_updated = robot.update_params(body=body_updated)

# Update tendon parameters
passive_updated = robot.params.passive_tendon.replace(stiffness=jnp.array([3.0]))
robot_updated = robot.update_params(passive_tendon=passive_updated)
```

### Simulation

```python
from diffrax import Tsit5
from soromox.systems import SystemState

# Simulation setup
q0 = jnp.array([jnp.pi/8, -jnp.pi/4])       # Initial angles
qd0 = jnp.zeros_like(q0)                    # Initial velocities
u = jnp.array([2.0])                        # Constant motor input

# Time integration
initial_state = SystemState(t=0.0, y=jnp.concatenate([q0, qd0]))
trajectory = robot.rollout_to(
    initial_state=initial_state,
    u=u,
    t1=5.0,
    solver_dt=1e-3,
    solver=Tsit5()
)
ts = trajectory.t
qs, qds = jnp.split(trajectory.y, 2, axis=1)
us = trajectory.u
```

## Configuration Guidelines

### Tendon Routing Constraints

When specifying routing matrices **R_at** and **R_pt**, the following constraints must be satisfied:

1. **No Joint Skipping**: For each tendon (row), once a zero entry appears, all subsequent entries must be zero
2. **Full Rank**: Each routing matrix must be full rank
3. **Physical Feasibility**: Pulley radii should be positive for proper routing

Example of valid routing:

```python
# Valid: tendon attached at joint 1, affects joints 0 and 1
R_at = jnp.array([[0.1, 0.15, 0.0, 0.0]])  # ✓ Valid

# Invalid: tendon skips joint 2
R_at = jnp.array([[0.1, 0.0, 0.15, 0.0]])  # ✗ Invalid
```

### Active vs Passive Tendons

- **Active tendons** (via `R_at`): Direct motor control, typically fewer in number
- **Passive tendons** (via `R_pt`): Provide compliance and stability, can be numerous

The system automatically computes actuation matrices as **A_at = R_at^T** and **A_pt = R_pt^T**.

## Examples

The package includes a complete simulation example in `examples/simulate_tendon_actuated_pendulum.py` demonstrating:

- System setup with realistic parameters
- Dynamic simulation with visualization  
- Tendon routing visualization
- Energy analysis over time

## API Reference

::: soromox.systems.pendulum.tendon_actuated_pendulum
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source
