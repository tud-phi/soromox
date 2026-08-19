# 📦 Installation

**Get started with SoRoMoX in minutes!** Choose from multiple installation methods to get Soft Robot Models in jaX (SoRoMoX) running on your system.

!!! warning "📢 Migration from JSRM"
    
    If you're migrating from the [JSRM package](https://github.com/tud-phi/jax-soft-robot-modeling), please note that SoRoMoX introduces breaking changes:
    
    - **Package Name**: `import jsrm` → `import soromox`
    - **Architecture**: Functional approach → Object-oriented Equinox dataclasses
    - **Performance**: Symbolic derivations → Numerical implementations
    - **New Soft Robot Models**: Support for Spatial PCS, GVS, and articulated soft robot systems
    - **Actuation**: Popular soft robot actuation modalities such as tendon and pressure actuation are implemented into the models (instead of just direct-torque actuation like in JSRM)
    - **Renderers**: SoRoMoX includes built-in renderers for visualization
    - **Control**: Model-based control implementations are included

---

## 🔧 Requirements

!!! note "System Requirements"
    - **Python** >= 3.11
    - **JAX** >= 0.10.0
    - **Diffrax** >= 0.7.2
    - **NumPy**

!!! warning "Python Version Compatibility"
    - **Open3D Rendering**: Open3D rendering is currently not compatible with Python 3.13+
    - **Python 3.14 on Windows**: There may currently exist an incompatibility of Python 3.14 on Windows with the package

---

## 🚀 Quick Install

=== "🐍 pip (Recommended)"

    The easiest way to install SoRoMoX is from PyPI using pip:

    ```bash
    pip install soromox
    ```

    !!! success "Ready to go!"
        This installs the core SoRoMoX package with all essential dependencies.

=== "⚡ uv (Fast Alternative)"

    For faster installation, use [uv](https://github.com/astral-sh/uv) - an extremely fast Python package installer:

    ```bash
    # Install uv if you haven't already
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Install soromox
    uv pip install soromox
    ```

    !!! tip "Speed Boost"
        uv is 10-100x faster than pip and automatically handles virtual environments. Perfect for rapid prototyping!

=== "🔨 From Source"

    For development or to get the latest features:

    ```bash
    git clone https://github.com/tud-phi/soromox.git
    cd soromox
    pip install -e .
    ```

    !!! tip "Development Mode"
        The `-e` flag installs in "editable" mode, so changes to the source code are immediately available.

---

## 🎯 Installation Options

SoRoMoX provides several optional dependency groups for different use cases:

### 🎨 Rendering Dependencies

For 3D visualization and animation:

```bash
pip install soromox[rendering]
```

**Includes:**

- `matplotlib` - For 2D plotting and simple rendering of the backbone shape
- `open3d` - High-quality 3D visualization
- `viser` - Web-based interactive rendering
- `opencv-python` - 2D rendering for planar robots

!!! note "FFmpeg Video Encoding"
    Video output uses the `ffmpeg` executable. Install FFmpeg separately with
    your system package manager and ensure `ffmpeg` is available on `PATH`, for
    example with `brew install ffmpeg` on macOS or `sudo apt install ffmpeg` on
    Debian/Ubuntu. Some renderers fall back to image sequences or OpenCV video
    encoding when FFmpeg is unavailable.

!!! note "Open3D Compatibility"
    Open3D rendering is currently not compatible with Python 3.13+. If you need 3D visualization, please use Python 3.12 or earlier, or use the `viser` renderer instead.

### 📚 Examples Dependencies

To run all examples and tutorials:

```bash
pip install soromox[examples]
```

**Includes:**

- `ipython` - Enhanced interactive Python shell
- `matplotlib` - Publication-ready plotting and visualization
- `open3d` - High-quality 3D visualization
- `opencv-python` - Computer vision and image processing
- `optimistix` - Nonlinear solvers for root finding, minimization, fixed points, and least squares optimization in JAX
- `plotly` - Interactive plotting and visualization
- `seaborn` - Statistical data visualization
- `viser` - Web-based interactive rendering

### 🛠️ Development Dependencies

For contributing to SoRoMoX:

```bash
pip install soromox[dev]
```

**Includes:**

- `pytest` - Testing framework
- `ruff` - Fast Python linter and formatter
- `mypy` - Static type checking
- `pre-commit` - Git hooks for code quality

### 📖 Documentation Dependencies

To build documentation locally:

```bash
pip install soromox[docs]
```

**Includes:**

- `zensical` - Documentation site generator
- `mkdocstrings-python` - API documentation generation

### 🎉 Complete Installation

Get everything at once:

```bash
pip install soromox[all]
```

This installs all optional dependencies including rendering, examples, development tools, and documentation builders.

!!! tip "Using uv"
    All these options work with uv as well:
    ```bash
    uv pip install soromox[rendering,examples]
    ```

---

## ✅ Verification

Test your installation with this quick verification script:

=== "🧪 Basic Test"

    ```python
    import jax.numpy as jnp
    from soromox.systems import PlanarPCS, PlanarPCSParams

    # Create a simple 1-segment PCS robot
    params = PlanarPCSParams(
        length=jnp.array([0.1]),
        radius=jnp.array([0.01]),
        density=jnp.array([1000.0]),
        young_modulus=jnp.array([1e6]),
        shear_modulus=jnp.array([1e5]),
        material_damping_coefficient=jnp.array([318.0]),
        reference_strain=jnp.array([0.0, 1.0, 0.0]),
    )
    robot = PlanarPCS(params=params)
    robot.forward_kinematics(jnp.zeros(robot.num_dofs), s=0.1)

    print("🎉 SoRoMoX installation successful!")
    ```

=== "🚀 Advanced Test"

    ```python
    import jax
    import jax.numpy as jnp
    from soromox.systems import PlanarPCS, PlanarPCSParams

    params = PlanarPCSParams(
        length=jnp.array([0.1, 0.1]),
        radius=jnp.array([0.01, 0.01]),
        density=jnp.array([1000.0, 1000.0]),
        young_modulus=jnp.array([1e6, 1e6]),
        shear_modulus=jnp.array([1e5, 1e5]),
        material_damping_coefficient=jnp.array([318.0, 318.0]),
        reference_strain=jnp.tile(jnp.array([0.0, 1.0, 0.0]), 2),
    )
    robot = PlanarPCS(params=params)

    # Test JAX compilation and differentiation. Same-shape params updates keep
    # the PyTree layout fixed and avoid recompilation.
    @jax.jit
    def test_function(q):
        return robot.forward_kinematics(q, s=0.2)

    # Test with sample configuration
    q = jnp.array([0.1, 0.0, 0.0, 0.1, 0.0, 0.0])
    result = test_function(q)
    
    # Test differentiation
    grad_fn = jax.grad(lambda q: jnp.sum(test_function(q)**2))
    gradient = grad_fn(q)
    
    print(f"✅ Forward kinematics: {result}")
    print(f"✅ Gradient computation: {gradient}")
    print("🎉 Advanced features working perfectly!")
    ```

---

## 🔧 Troubleshooting

!!! question "Common Issues"

    === "🐍 Python Version"
        
        **Problem:** `ImportError` or compatibility issues
        
        **Solution:** Ensure you're using Python 3.11 or later:
        ```bash
        python --version  # Should be >= 3.11
        ```

    === "📦 JAX Installation"
        
        **Problem:** JAX import errors or GPU issues
        
        **Solution:** Install JAX with proper hardware support:
        ```bash
        # For CPU only
        pip install jax[cpu]
        
        # For NVIDIA GPU
        pip install jax[cuda12_pip]
        
        # For Apple Silicon
        pip install jax[metal]
        ```

    === "🔧 Import Errors"

        **Problem:** `ModuleNotFoundError` for soromox modules

        **Solution:** Ensure the package is installed in your current environment:
        ```bash
        # Check installation
        pip list | grep soromox

        # Reinstall if needed
        pip install --force-reinstall soromox

        # For development installations
        pip install -e .

---

## 🆘 Getting Help

!!! info "Need assistance?"

    - 📚 **Documentation**: Check our [API Reference](api/overview.md)
    - 💬 **Discussions**: Join our [GitHub Discussions](https://github.com/tud-phi/soromox/discussions)  
    - 🐛 **Issues**: Report bugs on [GitHub Issues](https://github.com/tud-phi/soromox/issues)
    - 📧 **Email**: Contact us at `mstolzle@mit.edu`
