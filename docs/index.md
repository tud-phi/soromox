<div align="center">
  <img src="assets/logo/soromox_logo.png" alt="SoRoMoX Logo" width="280"/>
</div>

# 🤖 Soft Robot Models in jaX (SoRoMoX)

**Welcome to SoRoMoX!** A comprehensive library for soft robot modeling, model-based control, and visualization. Built on JAX for fast, parallelizable, and differentiable execution of soft robot models.

!!! info "Successor to JSRM"

    SoRoMoX is the successor to the [JSRM package](https://github.com/tud-phi/jax-soft-robot-modeling). Key improvements include:

    - **Extended System Support**: Spatial PCS, GVS, and articulated soft robot systems
    - **Numerical Implementation**: Replaced symbolic derivations with numerical implementations for better scalability and significantly reduced JIT compilation times
    - **Object-Oriented Architecture**: Migration from functional to Equinox dataclasses-based design enabling easy extendability and modification of methods
    - **Model-Based Control**: Comprehensive suite of model-based controllers for soft robots
    - **Visualization**: Multiple rendering backends for 2D and 3D visualization

---

## Overview

SoRoMoX provides three core capabilities for soft robotics research and development:

- **🤖 Soft Robot Models**: Kinematic and dynamic models of continuum and articulated soft robots with symbolic foundations and JAX implementation.
- **🎮 Model-Based Control**: Controllers for PID, gravity cancellation, potential shaping, impedance control, and computed torque.
- **🎨 Visualization**: Rendering backends including Matplotlib, Open3D, Viser, and OpenCV.

---

## Key Features

=== "JAX-Powered Performance"

    - **JIT Compilation**: Optimized machine code generation for maximum speed
    - **Automatic Differentiation**: Full gradient support for all operations
    - **GPU/TPU Support**: Seamless acceleration on modern hardware
    - **Parallel Processing**: Batch simulations across multiple devices with `vmap`

=== "Soft Robot Models"

    - **Articulated Systems**: Planar pendulums, tendon-actuated pendulums, and spatial articulated soft robots
    - **PCS (Piecewise Constant Strain)**: Continuum robots with constant strain segments
    - **GVS (Generalized Variable Strain)**: Flexible strain basis functions (Legendre, Chebyshev, Fourier)
    - **HSA (Handed Shearing Auxetics)**: Robots with auxetic material properties
    - **Multiple Actuators**: Support for tendon and pneumatic actuation

=== "Model-Based Controllers"

    - **Configuration-Space**: PID, gravity cancellation, potential compensation/cancellation regulators, computed torque
    - **Operational-Space**: Impedance control, synergistic controllers
    - **Actuation-Space**: Direct control of actuator inputs (tendons, pressures)
    - **Trajectory Tracking**: Feedforward compensation and mixed state feedback trackers

=== "Visualization"

    - **MatplotlibRenderer**: Generic 2D/3D plotting with Matplotlib
    - **ViserRenderer**: Interactive web-based 3D visualization in your browser
    - **Open3DRenderer**: High-quality 3D rendering with Open3D
    - **OpenCVPlanarRenderer**: Fast 2D rendering for planar robots
    - **Video Export**: Generate MP4 videos of simulations

---

## Supported Soft Robot Types

SoRoMoX supports both planar (2D) and spatial (3D) soft robot architectures:

!!! success "Articulated Systems"
    Planar and spatial rigid-link chains. Includes `Pendulum` for planar benchmark dynamics, `TendonActuatedPendulum` for cable-driven articulated mechanisms, and `ArticulatedSoftRobot` for spatial screw-axis chains with optional joint stiffness and damping.

!!! tip "PCS Systems (Piecewise Constant Strain)"
    Continuum soft robots with constant strain segments, available in both planar and spatial variants with tendon or pneumatic actuation

!!! abstract "GVS Systems (Geometric Variable Strain)"
    Advanced continuum robots with flexible strain basis functions (Legendre, Chebyshev, Fourier, etc.) and optional tendon actuation

!!! info "HSA Systems (Handed Shearing Auxetics)"
    Novel soft robots with auxetic material properties for unique deformation characteristics

---

## Quick Start

Get up and running in minutes:

=== "Installation"

    ```bash
    pip install soromox
    ```

    For visualization support:
    ```bash
    pip install soromox[rendering]
    ```

    For running examples:
    ```bash
    pip install soromox[examples]
    ```

    For all optional dependencies:
    ```bash
    pip install soromox[all]
    ```

=== "Simulation"

    ```python
    import jax.numpy as jnp
    from soromox.systems import PlanarPCS, PlanarPCSParams, SystemState

    # Create a planar PCS soft robot
    num_segments = 3
    params = PlanarPCSParams(
        length=0.1 * jnp.ones((num_segments,)),
        radius=0.02 * jnp.ones((num_segments,)),
        density=1070.0 * jnp.ones((num_segments,)),
        reference_strain=jnp.tile(jnp.array([0.0, 1.0, 0.0]), num_segments),
        gravity=jnp.array([0.0, 9.81]),
        young_modulus=2e3 * jnp.ones((num_segments,)),
        shear_modulus=1e3 * jnp.ones((num_segments,)),
        damping_matrix=1e-3 * jnp.eye(3 * num_segments),
        base_angle=jnp.array(jnp.pi / 2),
    )
    robot = PlanarPCS(params=params)

    # Initialize state
    q0 = jnp.zeros(robot.n_q)
    qd0 = jnp.zeros(robot.n_q)
    initial_state = SystemState(t=0.0, y=jnp.concatenate([q0, qd0]))

    # Simulate
    trajectory = robot.rollout_to(
        initial_state=initial_state,
        u=jnp.zeros(robot.n_q),
        t1=5.0,
        solver_dt=1e-4,
        save_dt=0.01,
    )
    ```

=== "Model-Based Control"

    ```python
    from soromox.control import (
        PIDControl,
        PIDControllerState,
        ReferenceTrajectory,
    )
    from soromox.control.configuration_space import (
        PotentialCancellationRegulator,
    )

    # Define PID gains
    pid_control = PIDControl(
        Kp=1e-2 * jnp.ones(robot.n_q),
        Ki=1e-3 * jnp.ones(robot.n_q),
        Kd=1e-4 * jnp.ones(robot.n_q),
    )

    # Define desired setpoint
    q_des = jnp.array([0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0])
    reference = ReferenceTrajectory(q=q_des)

    # Create model-based controller (gravity + elastic compensation)
    controller = PotentialCancellationRegulator(
        robot=robot,
        reference_trajectory=reference,
        pid_control=pid_control,
    )

    # Run closed-loop simulation
    initial_state = SystemState(
        t=0.0,
        y=jnp.concatenate([q0, qd0]),
        control_state=PIDControllerState.zero(robot.n_q),
    )

    trajectory = robot.rollout_closed_loop_to(
        initial_state=initial_state,
        controller=controller,
        t1=5.0,
        solver_dt=1e-4,
        save_dt=0.01,
    )
    ```

=== "Visualization"

    ```python
    from soromox.rendering import ViserRenderer

    # Create interactive 3D visualization
    renderer = ViserRenderer(robot, num_points=50)

    # Render a trajectory (opens in browser)
    renderer.render_sequence(
        ts=trajectory.t,
        q_ts=trajectory.y[:, :robot.n_q],
        playback_speed=1.0,
        loop=True,
        autoplay=True,
    )
    ```

---

## Model-Based Control

SoRoMoX provides a comprehensive suite of model-based controllers organized by control space:

!!! note "Configuration-Space Controllers"
    Control in generalized coordinates (strains, joint angles). Includes PID, gravity cancellation, potential shaping regulators, computed torque, and trajectory trackers.

!!! tip "Operational-Space Controllers"
    Control in task/end-effector space. Includes impedance control and synergistic task/null-space controllers.

!!! abstract "Actuation-Space Controllers"
    Direct control of actuator inputs (tendon forces, pressures). Includes PID and potential-based regulators mapped to actuator space.

---

## Rendering Backends

SoRoMoX supports multiple visualization backends for different use cases:

| Renderer | Use Case | Features |
|----------|----------|----------|
| `ViserRenderer` | Interactive exploration | Web-based 3D, playback controls, browser interface |
| `Open3DRenderer` | High-quality 3D | Point clouds, meshes, camera control |
| `MatplotlibRenderer` | Publication figures | 2D/3D plots, customizable styling |
| `OpenCVPlanarRenderer` | Fast 2D visualization | Real-time rendering, video export |

All renderers support:

- Single frame rendering
- Trajectory playback
- Video export (MP4)
- Customizable color themes and camera configurations

---

## Quick Links

- **📦 [Installation Guide](installation/)**: Get SoRoMoX installed and configured on your system.
- **🚀 [Quick Start](user-guide/quick-start/)**: Jump right in with hands-on tutorials and examples.
- **📖 [Examples](user-guide/examples/)**: Explore examples for simulation, control, and visualization.
- **📋 [API Reference](api/overview/)**: Complete documentation of all classes and functions.

---

## Citation

!!! quote "If you use our software in your research, please cite:"

    ### Software Citation

    ```bibtex
    @software{soromox2025,
      title = {SoRoMoX: Soft Robot Models in JAX},
      author = {Stölzle, Maximilian and Gribonval, Solange and {Feliu Talegón}, Daniel and {Della Santina}, Cosimo},
      year = {2025},
      url = {https://github.com/tud-phi/soromox},
      version = {0.1.0},
      abstract = {Kinematic and dynamic models of continuum and articulated soft robots implemented in JAX.}
    }
    ```

    ### Planar HSA Model Citation

    The model of the kinematics and dynamics of the planar HSA robot are part of the publication **"An Experimental Study of Model-based Control for Planar Handed Shearing Auxetics Robots"** presented at the *18th International Symposium on Experimental Robotics*.

    ```bibtex
    @inproceedings{stolzle2023experimental,
      title={An experimental study of model-based control for planar handed shearing auxetics robots},
      author={St{\"o}lzle, Maximilian and Rus, Daniela and Della Santina, Cosimo},
      booktitle={International Symposium on Experimental Robotics},
      pages={153--167},
      year={2023},
      organization={Springer}
    }
    ```

    ### Model-Based Controllers Citation

    If you found the implementation of the model-based controllers (e.g., PID, gravity cancellation, potential shaping regulators) useful, please consider also citing the following PhD thesis:

    ```bibtex
    @phdthesis{stolzle2025phdthesis,
      title = "Safe yet Precise Soft Robots: Incorporating Physics into Learned Models for Control",
      keywords = "Soft Robotics, Nonlinear Control, Machine Learning, Artificial Intelligence",
      author = "Maximilian St{\"o}lzle",
      year = "2025",
      month = "9",
      day = "15",
      language = "English",
      type = "Dissertation (TU Delft)",
      school = "Mechanical Engineering, Delft University of Technology",
      doi = "10.4233/uuid:24c1f667-8fd6-431a-bb78-11d22f8cb3da",
      isbn = "978-94-6384-836-7",
    }
    ```

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](development/contributing.md) for details on how to get started.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](https://github.com/tud-phi/soromox/blob/main/LICENSE.txt) file for details.
