<div align="center">
  <img src="docs/assets/logo/soromox_logo.png" alt="SoRoMoX Logo" width="400"/>
  
  # Soft Robot Models in jaX (SoRoMoX)
</div>

<div align="center">

[![Test](https://github.com/tud-phi/soromox/actions/workflows/test.yml/badge.svg)](https://github.com/tud-phi/soromox/actions/workflows/test.yml)
[![Documentation](https://github.com/tud-phi/soromox/actions/workflows/docs.yml/badge.svg)](https://github.com/tud-phi/soromox/actions/workflows/docs.yml)
[![PyPI version](https://badge.fury.io/py/soromox.svg)](https://badge.fury.io/py/soromox)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/tud-phi/soromox.svg)](https://github.com/tud-phi/soromox/blob/main/LICENSE.txt)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen.svg)](https://tud-phi.github.io/soromox)
[![Coverage](https://img.shields.io/badge/coverage-local%20report-blue.svg)](#running-tests-with-coverage)

</div>

> **📢 Note**: SoRoMoX is the successor to the [JSRM package](https://github.com/tud-phi/jax-soft-robot-modeling). It introduces significant improvements including support for an extended set of systems (Spatial PCS, GVS, and articulated soft robots), replacement of symbolic derivations with numerical implementations for better scalability and faster JIT compilation, and migration from a functional to an object-oriented architecture using Equinox dataclasses for enhanced extendability.

SoRoMoX provides three core capabilities for soft robotics research:

- **Soft Robot Models**: Kinematic and dynamic models of continuum and articulated soft robots with JAX implementation
- **Model-Based Control**: Comprehensive suite of controllers including PID, gravity cancellation, potential shaping, impedance control, and computed torque
- **Visualization**: Multiple rendering backends (Matplotlib, Open3D, Viser, OpenCV) for 2D and 3D visualization

SoRoMoX includes strain-based continuum models for slender structures (Cosserat rods), planar benchmarks, and spatial articulated soft robot systems. The following systems are implemented:

| System Type | Variants |
|-------------|----------|
| Articulated Systems | [Pendulum](examples/simulation/pendulum/simulate_pendulum.py), [Tendon-Actuated Pendulum](examples/simulation/pendulum/simulate_tendon_actuated_pendulum.py), [Articulated Soft Robot](examples/simulation/articulated/simulate_articulated_soft_robot.py) |
| PCS (Piecewise Constant Strain) | [Planar](examples/simulation/pcs/simulate_planar_pcs.py), [Spatial](examples/simulation/pcs/simulate_pcs.py), [Threadlike-Actuated](examples/simulation/pcs/simulate_tendon_actuated_pcs.py), [I-SUPPORT](examples/simulation/pcs/simulate_isupport.py) |
| GVS (Geometric Variable Strain) | [Spatial](examples/simulation/gvs/simulate_gvs.py), [Tendon-Actuated](examples/simulation/gvs/simulate_tendon_actuated_gvs.py) |
| HSA (Handed Shearing Auxetics) | [Planar](examples/simulation/hsa/simulate_planar_hsa.py) |

We are happy to receive contributions for other soft robot models. See the [Contributing Guide](docs/development/contributing.md) for more details.

## Citation

Please use the following citation if you use our software in your (scientific) work:

### Software Citation

```bibtex
@software{soromox2026,
  title = {SoRoMoX: Soft Robot Models in JAX},
  author = {Stölzle, Maximilian and Gribonval, Solange and {Feliu-Talegon}, Daniel and Perfetta, Vito Daniele and Martini, Michele and Zhang, Chuhan and Wong, Kiwan and Rus, Daniela and {Della Santina}, Cosimo},
  year = {2026},
  url = {https://github.com/tud-phi/soromox},
  version = {0.2.0},
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

## Installation

The package can be installed from PyPI using pip:

```bash
pip install soromox
```

or using [uv](https://docs.astral.sh/uv/) (recommended for faster installation):

```bash
uv pip install soromox
```

### Development Installation

For local development from the source code:

```bash
# Using pip
pip install -e .

# Using uv
uv pip install -e .
```

### Optional Dependencies (extras)

The default install keeps the runtime dependency set small. Install one or more
extras when you need development tools, examples, rendering backends, RL
workflows, or the dependencies used to reproduce the paper results.

| Extra | Use this for | Main packages included |
|-------|--------------|------------------------|
| `dev` | Local development, linting, formatting, tests, coverage, release checks, and pre-commit hooks. | `ruff`, `pytest`, `coverage`, `tox`, `pre-commit`, `bump2version`, `check-manifest`, `seaborn` |
| `docs` | Building and serving the documentation site locally. | `zensical`, `mkdocstrings`, `mkdocstrings-python` |
| `examples` | Running the example scripts under `examples/`. | `matplotlib`, `seaborn`, `ipython`, `optax`, `optimistix`, `open3d`, `opencv-python`, `plotly`, `trimesh`, `viser`, `ffmpeg-python` |
| `rendering` | Using optional rendering backends and exporting plots, videos, or interactive visualizations. | `matplotlib`, `open3d`, `opencv-python`, `plotly`, `trimesh`, `viser`, `ffmpeg-python` |
| `rl` | Training and evaluating reinforcement-learning controllers with SoRoMoX, including parallel RL workflows. | `gymnasium`, `stable-baselines3`, `matplotlib` |
| `paper_results` | Reproducing the research workflows, data, figures, and videos under `paper_results/`, including benchmarks, RL experiments, visualization, and comparison baselines. | `cbfpy`, `elastica`, `gymnasium`, `stable-baselines3`, plus the example and rendering stack |
| `test` | Running the test suite without the full development stack. | `pytest`, `pytest-cov`, `pytest-html`, `coverage`, `tox`, `codecov` |
| `all` | Broad convenience install for users who want one environment with most optional tooling. For reproducible workflows, prefer the task-specific extras above. | See the `all` extra in `pyproject.toml` |

Quote extras in shell commands so shells such as zsh do not interpret the square
brackets as glob patterns. Install from PyPI with:

```bash
pip install "soromox[rendering]"
pip install "soromox[rl]"
```

For a local editable source install, use pip or uv:

```bash
pip install -e ".[dev,docs,examples]"
uv pip install -e ".[rendering]"
```

To reproduce the paper results or run RL experiments from the source checkout:

```bash
uv pip install -e ".[paper_results]"
uv pip install -e ".[rl]"
```

Complete publication workflows, their input and generated data, and committed
canonical outputs live under `paper_results/`. Use the `paper_results` extra for
these workflows; the narrower `rl` extra remains available for
reinforcement-learning-only work.

### Using uv for Project Management

You can also use `uv` to manage the project with a virtual environment:

```bash
# Create a virtual environment and install the package
uv venv
uv pip install -e ".[dev,examples]"

# Or use uv sync for reproducible installs (creates uv.lock)
uv sync --extra dev --extra examples
uv sync --extra paper_results
uv sync --extra rl
```

### Running Tests with Coverage

Coverage.py is included in the `dev`, `test`, and `all` extras. To run the test suite with the project coverage settings:

```bash
uv run --extra test make coverage
```

or call Coverage.py directly:

```bash
uv run --extra test python -m coverage run -m pytest
uv run --extra test python -m coverage report
```

For CI uploads, generate an XML report with:

```bash
uv run --extra test make coverage_xml
```

## Usage

Simply call one of the example scripts to simulate a system. For example, to simulate the N-link pendulum:

```bash
python examples/simulation/pendulum/simulate_pendulum.py
```

or to simulate the planar PCS robot:

```bash
python examples/simulation/pcs/simulate_planar_pcs.py
```

The lightweight example catalogue and output conventions are documented in
[`examples/README.md`](examples/README.md). Reproducible paper workflows,
including the Section Vc model-based control studies, are under
[`paper_results/`](paper_results/README.md):

```bash
python paper_results/secVc_model_based_control/configuration_space_comparison/code/setpoint_regulation_comparison.py
python paper_results/secVc_model_based_control/operational_space_impedance_control/code/control_pcs_with_impedance.py
```

## Documentation

The full documentation is available at: <https://tud-phi.github.io/soromox>

### Building Documentation Locally

To build and serve the documentation locally:

```bash
# Install documentation dependencies
uv sync --extra docs

# Serve documentation with live reload
uv run --extra docs zensical serve
```

The documentation will be available at `http://127.0.0.1:8000`.

### Building Documentation for Production

```bash
# Build static documentation
uv run --extra docs zensical build --clean

# Zensical does not currently support MkDocs-style strict mode.
# The CI workflow runs the same build command.
```

You can also use the Makefile targets:

```bash
# Serve locally
make docs-serve

# Build documentation
make docs-build

# Build with strict mode checking
make docs-build-strict
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
python bump_version.py --patch --dry-run

# Preview a minor version bump
python bump_version.py --minor --dry-run

# Preview a specific version
python bump_version.py 0.2.0 --dry-run
```

The dry run mode will show you:
- Current version
- New version that would be set
- List of files that would be modified
- No actual changes are made to any files

### What Gets Updated

The version bump process automatically updates:
- `pyproject.toml` - Main project version
- `CITATION.cff` - Citation version and release date
- `README.md` and `docs/index.md` - BibTeX version, year, and citation key
- `docs/development/changelog.md` - Moves unreleased notes into a dated release
- `uv.lock` - Local package version

Generated `*.egg-info` metadata is not tracked or edited; package builds
regenerate it from `pyproject.toml`.

### Automated Release Process

When you use the `release-*` commands, the following happens automatically:
1. Version numbers are updated in all relevant files
2. Changes are committed to git
3. A version tag (e.g., `v0.1.1`) is created
4. Changes and tags are pushed to GitHub
5. GitHub Actions verifies the tag, builds and validates the distributions once
6. The validated artifacts are attached to a GitHub release
7. The same artifacts are published to PyPI through trusted publishing

Only the release metadata files listed above are staged automatically. Other
tracked or untracked worktree files are left untouched.

For more detailed information, see `VERSION_BUMP_README.md`.
