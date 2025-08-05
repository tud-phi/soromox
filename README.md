<div align="center">
  <img src="docs/assets/logo/soromox_logo.png" alt="SoRoMoX Logo" width="400"/>
  
  # Soft Robot Models in jaX (SoRoMoX)
</div>

<div align="center">

[![Test](https://github.com/tud-phi/soromox/actions/workflows/test.yml/badge.svg)](https://github.com/tud-phi/soromox/actions/workflows/test.yml)
[![Documentation](https://github.com/tud-phi/soromox/actions/workflows/docs.yml/badge.svg)](https://github.com/tud-phi/soromox/actions/workflows/docs.yml)
[![PyPI version](https://badge.fury.io/py/soromox.svg)](https://badge.fury.io/py/soromox)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/tud-phi/soromox.svg)](https://github.com/tud-phi/soromox/blob/main/LICENSE.txt)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen.svg)](https://tud-phi.github.io/soromox)

</div>

> **📢 Note**: SoRoMoX is the successor to the [JSRM package](https://github.com/tud-phi/jax-soft-robot-modeling). It introduces significant improvements including support for an extended set of systems (General/3D PCS and GVS), replacement of symbolic derivations with numerical implementations for better scalability and faster JIT compilation, and migration from a functional to an object-oriented architecture using Equinox dataclasses for enhanced extendability.

We provide implementations of several popular and expressive soft robot models in JAX for fast, parallelizable, and differentiable simulations. Specifically, we focus on strain-based models that are suitable for modeling slender structures like soft robots (i.e., robots with a large length compared to their diameter often referred to as Cosserat rods).
So far, we have implemented the following systems:

- [N-link pendulum](examples/simulate_pendulum.py)
- [General/3D Piecewise Constant Strain (PCS) continuum soft robot](examples/simulate_pcs.py)
- [Planar Piecewise Constant Strain (PCS) continuum soft robot](examples/simulate_planar_pcs.py)
- [Planar Handed Shearing Auxetics (HSA) robot](examples/simulate_planar_hsa.py)

We are happy to receive contributions for other soft robot models. See the [Contributing Guide](development/contributing.md) for more details.

## Citation

Please use the following citation if you use our software in your (scientific) work:

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

## Installation

The plugin can be installed from PyPI:

```bash
pip install soromox
```

or locally from the source code:

```bash
pip install -e .
```

If you want to run the examples, you will also need to install the following dependencies:

```bash
pip install -e ".[examples]"
```

## Usage

Simply, call one of the example scripts to simulate a system. For example, to simulate the N-link pendulum:

```bash
python examples/simulate_pendulum.py
```

or to simulate the planar PCS robot:

```bash
python examples/simulate_planar_pcs.py
```

## Documentation

The full documentation is available at: <https://tud-phi.github.io/soromox>

### Building Documentation Locally

To build and serve the documentation locally:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Serve documentation with live reload (if mkdocs is in PATH)
mkdocs serve

# OR using conda environment
conda run --live-stream --name soromox mkdocs serve
```

The documentation will be available at `http://127.0.0.1:8000`.

### Building Documentation for Production

```bash
# Build static documentation (if mkdocs is in PATH)
mkdocs build

# OR using conda environment
conda run --live-stream --name soromox mkdocs build

# Build with strict mode (for CI/development)
mkdocs build --strict
# OR: conda run --live-stream --name soromox mkdocs build --strict

# Deploy to GitHub Pages (maintainers only)
mkdocs gh-deploy
# OR: conda run --live-stream --name soromox mkdocs gh-deploy
```

You can also use the Makefile targets:

```bash
# Serve locally
make docs-serve

# Build documentation
make docs-build

# Build with strict mode checking
make docs-build-strict

# Deploy to GitHub Pages
make docs-deploy
```

## Version Management and Releases

This project includes automated version management and release creation tools. The version bump system updates version information across all relevant files and can automatically create GitHub releases.

### Version Bumping

You can bump the version using the following Makefile targets:

```bash
# Increment patch version (0.1.0 -> 0.1.1) - for bug fixes
make bump-patch

# Increment minor version (0.1.0 -> 0.2.0) - for new features
make bump-minor

# Increment major version (0.1.0 -> 1.0.0) - for breaking changes
make bump-major

# Set a specific version
make bump-version VERSION=0.2.0
```

### Automated Releases

To create a complete release with automatic GitHub release creation:

```bash
# Create a patch release with GitHub release
make release-patch

# Create a minor release with GitHub release
make release-minor

# Create a major release with GitHub release
make release-major

# Create a specific version release
make release VERSION=1.0.0
```

### Preview Changes (Dry Run)

Before making any changes, you can preview what would be updated using the dry run mode:

```bash
# Preview a patch version bump without making changes
conda run -n jsrm python bump_version.py --patch --dry-run

# Preview a minor version bump
conda run -n jsrm python bump_version.py --minor --dry-run

# Preview a specific version
conda run -n jsrm python bump_version.py 0.2.0 --dry-run
```

The dry run mode will show you:
- Current version
- New version that would be set
- List of files that would be modified
- No actual changes are made to any files

### What Gets Updated

The version bump process automatically updates:
- `pyproject.toml` - Main project version
- `CITATION.cff` - GitHub citation file version
- `docs/development/changelog.md` - Adds new version entry with current date
- `src/soromox.egg-info/PKG-INFO` - Package metadata (if exists)

### Automated Release Process

When you use the `release-*` commands, the following happens automatically:
1. Version numbers are updated in all relevant files
2. Changes are committed to git
3. A version tag (e.g., `v0.1.1`) is created
4. Changes and tags are pushed to GitHub
5. GitHub Actions creates a release with changelog content
6. The package is published to PyPI

For more detailed information, see `VERSION_BUMP_README.md`.

