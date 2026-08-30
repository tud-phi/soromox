# Examples

See [Parameters, Updates, and Optimization](parameters-and-optimization.md)
for complete construction, immutable update, JAX, and Optax examples.

This page provides a comprehensive overview of the example scripts included with SoRoMoX. Examples are organized by application area to help you find the right starting point for your use case.

!!! tip "Getting Started"
    New to SoRoMoX? Start with the [Quick Start guide](quick-start.md) for a step-by-step tutorial, then return here to explore more examples.

---

## Overview

SoRoMoX examples are organized into two main categories:

- **[Simulation Examples](#simulation-examples)** - Dynamic simulations of various robot systems
- **[Control Examples](#control-examples)** - Control algorithms for different control spaces

Reproducible paper benchmarks, system identification, optimization, and
application studies are catalogued separately in `paper_results/README.md`.

---

## Simulation Examples

Simulation examples demonstrate how to model and simulate different types of robotic systems using SoRoMoX. These examples are located in `examples/simulation/` and are organized by robot type.

### Articulated Systems

Articulated systems model serial rigid-link chains. They include planar
pendulum benchmarks, tendon-actuated pendulums, and spatial articulated soft
robots with optional joint stiffness and damping.

#### N-Link Pendulum
```bash
python examples/simulation/pendulum/simulate_pendulum.py
```

**What you'll learn:**
- Basic pendulum dynamics and numerical integration with JAX
- How to visualize results and compute energy (kinetic and potential)
- Working with `SystemState` and the `rollout_to()` method
- End-effector trajectory computation

**Key concepts:** Rigid-body dynamics, state-space representation, forward kinematics

To simulate a double pendulum, modify the `num_links` variable in the script:
```python
num_links = 2  # Change this line
```

#### Tendon-Actuated Pendulum
```bash
python examples/simulation/pendulum/simulate_tendon_actuated_pendulum.py
```

**What you'll learn:**
- Cable-driven actuation modeling
- Tendon routing and tension mapping
- Control applications with underactuated systems

**Key concepts:** Actuation mapping, tendon kinematics, underactuation

#### Spatial Articulated Soft Robot
```bash
python examples/simulation/articulated/simulate_articulated_soft_robot.py
```

The example simulates a four-link spatial chain, plots joint states, end-effector motion, and energies, then launches a Matplotlib animation by default. Use `--render open3d`, `--render viser`, or `--render all` to try the optional interactive renderers.

**What you'll learn:**
- Building a spatial serial chain with screw-axis joints
- Adding passive joint stiffness and damping
- Simulating with ABA-based forward dynamics through the standard `rollout_to()` API
- Rendering with Matplotlib, Open3D, or Viser backends

**Key concepts:** Spatial articulated dynamics, compliant joints, ABA forward dynamics

### PCS (Piecewise Constant Strain) Robots

PCS robots are continuum robots modeled using the Piecewise Constant Strain approach, which discretizes the robot into segments with constant strain parameters.

#### Planar PCS Robot
```bash
python examples/simulation/pcs/simulate_planar_pcs.py
```

**What you'll learn:**
- Piecewise Constant Strain (PCS) kinematics for 2D continuum robots
- Continuum robot dynamics using Cosserat rod theory
- Forward kinematics and Jacobian computation
- Visualization with Matplotlib and OpenCV renderers

**Key concepts:** PCS modeling, strain parameterization, planar kinematics

#### Spatial PCS Robot
```bash
python examples/simulation/pcs/simulate_pcs.py
```

**What you'll learn:**
- 3D continuum robot modeling with full spatial kinematics
- Six-degree-of-freedom strain representation (bending + extension + shear)
- Spatial forward kinematics and workspace analysis

**Key concepts:** Spatial PCS, 6-DOF strains, 3D kinematics

#### Tendon-Actuated PCS Robots
```bash
# Planar version
python examples/simulation/pcs/simulate_tendon_actuated_planar_pcs.py

# Spatial version
python examples/simulation/pcs/simulate_tendon_actuated_pcs.py

# Batched simulation for performance analysis
python examples/simulation/pcs/simulate_batched_tendon_actuated_pcs.py
```

**What you'll learn:**
- Tendon routing and actuation mapping for continuum robots
- Cable-driven soft robot control
- Batch processing for performance benchmarking

**Key concepts:** Tendon kinematics, actuation space, batch simulation

#### I-SUPPORT Manipulator
```bash
python examples/simulation/pcs/simulate_isupport.py
```

**What you'll learn:**
- Modeling of 3D-printed pneumatic soft robotic arms
- Real-world system parameter adaptation
- Multi-segment pneumatic continuum robots

**Key concepts:** Additive manufacturing, pneumatic actuation, real-world applications

#### Planar HSA Robot

Planar HSA robots are a specialized PCS family using auxetic materials that exhibit unique deformation characteristics under actuation.

```bash
# Planar HSA simulation
python examples/simulation/pcs/simulate_planar_hsa.py
```

**What you'll learn:**
- Handed Shearing Auxetics (HSA) mechanics and unique deformation characteristics
- Practical control applications with HSA robots

**Key concepts:** HSA mechanics, auxetic materials, PCS simulation

### GVS (Geometric Variable Strain) Robots

GVS robots use flexible strain basis functions for advanced continuum robot modeling.

#### Basic GVS Robots
```bash
# Basic GVS simulation
python examples/simulation/gvs/simulate_gvs.py
```

**What you'll learn:**
- Geometric Variable Strain modeling
- Flexible strain basis functions
- Advanced continuum robot dynamics

**Key concepts:** GVS modeling, basis functions, advanced kinematics

#### Soft Inverted Pendulums

```bash
# Fixed-base autonomous pendulum with low stiffness (k = 1)
python examples/simulation/gvs/simulate_soft_inverted_pendulum.py

# The paper's medium-stiffness case (k = 4)
python examples/simulation/gvs/simulate_soft_inverted_pendulum.py --stiffness 4

# The same affine-curvature link clamped to a passive translating cart
python examples/simulation/gvs/simulate_soft_cart_pendulum.py
```

Both examples use the affine curvature
`kappa_z(s) = theta_0 + theta_1 s`, physical parameters, generalized
stiffness and damping matrices, pure-tip-torque map, and initial soft
configuration from [Della Santina (2020)](https://doi.org/10.1109/CDC42340.2020.9303976).
The GVS implementation distributes the reported 1 kg mass uniformly along a
square `0.1 m x 0.1 m` link instead of reproducing the paper's
tip-concentrated inertia.

The cart example is conceptually based on
[Ajithkumar (2023)](https://doi.org/10.13016/dspace/7jir-df1p), but deliberately
clamps the soft link directly to the passive prismatic cart coordinate: it has
no intervening revolute joint and does not implement the thesis controllers.
The two models therefore share exactly the same soft-link structure and
parameters; the cart model adds only its translational coordinate and mass.

Each script saves a summary figure and launches an interactive Viser motion
rendering by default. The cart renderer synchronizes the soft backbone with a
moving cart body, wheels, and rail. Use `--record-viser` to capture one playback
or `--no-viser --no-show` for headless execution. The fixed and cart scripts use
ports 8080 and 8081 by default, respectively, and accept `--viser-port` overrides.

**What you'll learn:**
- Affine-curvature GVS construction with two bending coordinates
- Exact generalized constitutive and tip-torque mappings
- Passive underactuation through a prismatic cart joint
- Interactive trajectory playback and recording with Viser

**Key concepts:** Affine curvature, soft inverted pendulum, passive cart, Viser

#### Tendon-Actuated GVS Robot

```bash
python examples/simulation/gvs/simulate_tendon_actuated_gvs.py
```

**What you'll learn:**
- Routing tendon actuation through a variable-strain continuum model
- Simulating and rendering an actuated GVS robot

**Key concepts:** GVS modeling, tendon actuation, continuum routing

---

## Control Examples

Control examples demonstrate various control algorithms for different control spaces. These examples are located in `examples/control/` and are organized by control space type.

### Operational-Space Controllers

Operational-space controllers operate on the robot's end-effector or task-space coordinates.

#### Synergistic Control
```bash
python examples/control/operational_space/control_tendon_actuated_pcs_with_synergistic.py
```

**What you'll learn:**
- Operational-space control for an underactuated tendon-driven PCS robot
- End-effector position tracking with actuation synergies
- Task-space reference generation

**Key concepts:** Operational-space control, underactuation, synergistic control

### Actuation-Space Controllers

Actuation-space controllers operate directly on the actuator inputs (e.g., tendon tensions, chamber pressures).

#### Setpoint Regulation
```bash
python examples/control/actuation_space/setpoint_regulation_comparison.py
```

**What you'll learn:**
- Control in actuation space
- Direct actuator command generation
- Comparison of actuation-space control strategies

**Key concepts:** Actuation-space control, actuator mapping, direct control

---

## Running Examples

### Prerequisites

Install SoRoMoX with example dependencies:
```bash
pip install -e ".[examples]"
```

This installs additional packages needed for visualization and analysis:
- `matplotlib` - Plotting and visualization
- `open3d` - 3D visualization
- `opencv-python` - 2D rendering for planar robots
- `optimistix` - Optimization (for system identification examples)
- `plotly` - Interactive plotting and visualization
- `seaborn` - Statistical data visualization
- `viser` - Interactive 3D visualization

### Basic Workflow

Follow these steps to run any example:

1. **Navigate to the project root** (if not already there):
   ```bash
   cd /path/to/soromox
   ```

2. **Run the example**:
   ```bash
   python examples/[category]/[subcategory]/[example_name].py
   ```
   
   For example:
   ```bash
   python examples/simulation/pendulum/simulate_pendulum.py
   python examples/control/actuation_space/setpoint_regulation_comparison.py
   ```

3. **Analyze results** in generated plots and videos

**Expected outputs:**
- Interactive plots using Matplotlib (may require GUI backend)
- Video files in the `videos/` directory (if video rendering is enabled)
- Console output with simulation statistics and results
- Analysis plots and comparison figures (for control examples)

---

## Customizing Examples

All examples are designed to be easily customizable. This section shows you how to modify parameters, outputs, and initial conditions.

### Parameter Modification

Most examples allow easy parameter modification. Understanding parameter structure is key to adapting examples to your needs.

**For PCS systems:**

```python
# Modify canonical runtime link parameters
section = robot.params.link.cross_section.replace(
    coefficients=0.9 * robot.params.link.cross_section.coefficients
)
robot = robot.update_link_params(
    length=0.2 * jnp.ones((num_segments,)),
    cross_section=section,
)

# Material variables are caller-owned and applied explicitly.
material = material.replace(
    young_modulus=1e6 * jnp.ones((num_segments,)),
    shear_modulus=5e5 * jnp.ones((num_segments,)),
)
robot = robot.with_isotropic_material(material)
```

**For pendulum systems:**

```python
params = params.replace(
    length=jnp.array([0.5, 0.3]),
    mass=jnp.array([1.0, 0.5]),
    moment_inertia=jnp.array([0.1, 0.05]),
)
robot = robot.with_params(params)
```

**For articulated soft robot systems:**

```python
params = params.replace(
    joint_screw=joint_screws,
    tip_position=tip_positions,
    center_of_mass_position=center_of_mass_positions,
    mass=masses,
    center_of_mass_inertia=center_of_mass_inertia,
    joint_stiffness=jnp.diag(joint_stiffness),
    joint_damping=jnp.diag(joint_damping),
)
robot = robot.with_params(params)
```

### Output Customization

Control visualization and output:

```python
# Adjust simulation time
t0 = 0.0
t1 = 10.0  # 10 seconds

# Change integration parameters
solver_dt = 0.001  # Smaller timestep for higher accuracy
save_dt = 0.01     # Output frequency

# Video settings (if using renderers)
video_width, video_height = 800, 800
```

### Initial Conditions

Modify initial configurations and velocities:

```python
# For pendulum systems
q0 = jnp.array([jnp.pi/4, jnp.pi/6])  # Initial joint angles
qd0 = jnp.zeros_like(q0)                # Initial velocities

# For articulated soft robot systems
q0 = jnp.array([0.35, -0.45, 0.28, -0.20])  # Initial joint coordinates
qd0 = jnp.zeros_like(q0)

# For PCS systems
q0 = jnp.array([0.1, 0.0, 0.0,  # Segment 1 strains
                0.2, 0.0, 0.0]) # Segment 2 strains
qd0 = jnp.zeros_like(q0)
```

### Control Parameters

For control examples, you can modify controller gains:

```python
# PID controller gains
kp = 100.0  # Proportional gain
ki = 10.0   # Integral gain
kd = 5.0    # Derivative gain

# Impedance controller parameters
K_x = 100.0 * jnp.eye(3)  # Stiffness matrix
D_x = 10.0 * jnp.eye(3)   # Damping matrix
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Import errors** | Ensure SoRoMoX is installed with `pip install -e .` |
| **Missing example dependencies** | Install examples group with `pip install -e ".[examples]"` |
| **Missing renderers** | Ensure SoRoMoX is installed with `pip install -e ".[rendering]"` or `pip install -e ".[all]"` |
| **JAX precision warnings or simulation instability** | Most simulations require double precision. Enable with `jax.config.update("jax_enable_x64", True)` at the beginning of the script |
| **Video codec issues** | Install FFmpeg with your system package manager and verify the executable is available with `ffmpeg -version` |
| **Matplotlib backend errors** | Try setting backend: `export MPLBACKEND=Agg` for headless systems |
| **Out of memory errors** | Reduce batch sizes or use smaller simulation parameters |
| **Optimization convergence issues** | Check initial parameter guesses and optimization settings in system identification examples |

### Performance Tips

- **Accuracy vs. Speed**: Use smaller `solver_dt` for higher accuracy and improved simulation stability, larger for faster simulation
- **JIT Compilation**: JAX JIT compilation is automatically used in most examples for optimal performance
- **GPU Acceleration**: Consider using GPU acceleration for large-scale batch simulations
- **Batched Simulations**: For batched simulations, see `simulate_batched_tendon_actuated_pcs.py` as a reference
- **Memory Management**: Be mindful of array sizes in batch operations to avoid memory issues
- **System Identification**: For parameter identification, start with reasonable initial guesses and use appropriate optimization algorithms. Gradient descent will often only work locally.

---

## Creating New Examples

To create a new example:

1. **Choose a base example** similar to your use case from the appropriate category:
   - `examples/simulation/` for new robot systems
   - `examples/control/` for new control algorithms

2. **Copy and modify** parameters and logic
3. **Test thoroughly** with different configurations
4. **Write generated artifacts to sibling `figures/` or `videos/` directories**

### Example Template

Here's a minimal working example you can use as a starting point:

```python
#!/usr/bin/env python3
"""
New Example: Description of what this example demonstrates
"""

import jax
import jax.numpy as jnp
from soromox.systems import YourSystem, SystemState

# Configure JAX (if needed)
jax.config.update("jax_enable_x64", True)  # double precision

# Parameters
params = YourSystemParams(...)

if __name__ == "__main__":
    # Initialize system
    robot = YourSystem(params=params)
    
    # Set initial conditions
    q0 = jnp.array([...])  # Initial configuration
    qd0 = jnp.zeros_like(q0)  # Initial velocities
    
    # Simulation parameters
    t0, t1 = 0.0, 5.0
    solver_dt = 1e-4
    save_dt = 0.01
    
    # Run simulation
    initial_state = SystemState(t=t0, y=jnp.concatenate([q0, qd0]))
    trajectory = robot.rollout_to(
        initial_state=initial_state,
        u=jnp.zeros_like(q0),  # Control inputs
        t1=t1,
        solver_dt=solver_dt,
        save_dt=save_dt,
    )
    
    # Analyze and visualize results
    # ...
```

---

## Next Steps

After exploring the examples:

- **Read the API documentation** to understand the full capabilities of SoRoMoX
- **Check the development guide** for contributing new examples or features
- **Explore the source code** to understand implementation details
- **Join the community** to share your examples and get help

For more detailed information about specific systems or control methods, refer to the [API documentation](../api/overview.md).
