# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Shared continuum-component APIs under `soromox.systems.components` for link,
  joint, cross-section, and isotropic-material parameters and specifications.
- Common `from_links`, `params_from_links`, `update_link_params`,
  `link_matrices_from_material`, and `with_isotropic_material` workflows for
  PCS and PlanarPCS, with the same material/update APIs on GVS.
- GVS joint stiffness and damping parameters that participate in global matrix
  assembly, plus `update_joint_params` for immutable joint-local updates.
- Differentiable unit-response mappings from Young's modulus, shear modulus,
  and material damping to canonical generalized link matrices.
- Continuum-component, parameter-update, material-optimization, and
  [parameter-API migration](../user-guide/parameter-api-migration.md)
  documentation with complete PCS and GVS examples.

### Changed

- Harmonized PCS, PlanarPCS, and GVS around nested `params.link` ownership,
  descriptive public construction names, shared link/joint specifications, and
  canonical per-link generalized stiffness and damping matrices.
- Moved shared `LinkSpec` and `JointSpec` functionality out of the GVS package;
  GVS now retains only segment, strain-basis, quadrature, and runtime concepts
  specific to variable-strain mechanics.
- Simplified cross-section parameters to coefficient arrays populated by link
  factories; constant values and `LinearProfile` cover constant and linearly
  varying geometry without a profile-parameter class hierarchy.
- Made isotropic material parameters caller-owned construction/optimization
  PyTrees instead of duplicating material and generalized-matrix
  representations inside system parameters.
- Made geometry updates refresh material unit-response operators without
  silently replacing explicitly supplied canonical matrices.

### Fixed

- Included stored GVS joint stiffness and damping in global stiffness and
  damping assembly.
- Resolved omitted GVS padding values in the advanced `GVS(params, structure)`
  constructor and aligned rectangular geometry output with the shared
  `[height, width]` cross-section convention.
- Normalized caller-provided isotropic material sequences to JAX arrays and
  rejected non-finite or nonphysical link geometry, density, and material
  values during concrete construction and replacement.
- Migrated examples, benchmarks, and paper case studies to the harmonized PCS
  and GVS parameter APIs.

### Breaking changes

- Removed flat PCS and PlanarPCS link/material fields. Access link data through
  `params.link`, and construct systems with `from_links` or
  `params_from_links`.
- Removed the global PCS damping matrix and cross-link damping coupling. Supply
  one generalized damping block per link.
- Removed `GVSLinkParams` and GVS-local exports of shared link, joint, and
  cross-section specifications. Import shared types from `soromox.systems` or
  `soromox.systems.components`.
- Replaced abbreviated GVS construction fields such as `E`, `nu`, `rho`,
  `eta`, `L`, `r_i`, and `r_f` with descriptive material, geometry, and profile
  arguments.
- Moved `reference_strain` from `StrainBasisSpec` to `LinkSpec`; basis specs now
  describe only the selected strains and basis order.
- No compatibility aliases are provided. See the compact
  [PCS/GVS parameter migration guide](../user-guide/parameter-api-migration.md)
  for direct replacements.

## [0.2.1] - 2026-08-03

### Added

- Section-scoped `paper_results/` workflows that colocate generation and
  plotting code, source or generated data, canonical figures and videos, and
  reproduction instructions for the results reported in the paper.
- Reproducible PCS configuration- and operational-space control case studies
  with shared simulation, plotting, serialization, Viser, phase-separated RMSE,
  and canonical paper-figure workflows.
- A shared Matplotlib publication style and case-specific figure composers for
  consistent typography, sizing, colors, labels, grids, and legends.
- Focused Section Vc tests for case-relative paths, NPZ schemas, overwrite
  protection, plotting, controller setup, trajectory primitives, and rendering.
- Complete Section IVa, IVb, Va, Vd, Ve, and Vf generation workflows alongside
  their paper data and outputs, including released RL checkpoints.
- Reproducible Section Vc evaluation metrics for all six configuration
  coordinates and geometric operational-space pose tracking, with canonical
  NPZ and JSON artifacts and a human- or machine-readable reporting CLI.
- A publication-sized Section Vc composite figure combining configuration-space
  comparisons, operational-space tracking and errors, and chronological
  experiment snapshots in committed PDF and SVG outputs.

### Changed

- Separated lightweight demonstrations under `examples/` from complete
  publication reproduction workflows and datasets under `paper_results/`.
- Made generated example figures and videos topic-local instead of writing them
  to the repository root, and organized paper sections into independent
  `code/`, `data/`, and `outputs/` ownership.
- Made affected example and paper entry points independent of the current
  working directory, added configurable artifact paths and overwrite guards,
  and migrated visualization workflows to the current renderer APIs.
- Standardized the Section Vc paper figures around full-width
  configuration-space tracking and partial-feedback-linearization
  operational-space tracking, with explicit strain labels and prominent
  reference trajectories.
- Store trained model checkpoints through Git LFS while leaving generated
  example artifacts and noncanonical paper diagnostics ignored.
- Made the built-in `tanh` PID integral-error saturation unit preserving as
  `Gamma^-1 tanh(Gamma e)`, retained unit slope at the origin, and validated
  scalar, vector, and matrix saturation parameters against their required
  positivity and shape constraints.
- Applied separate physical integral-error saturation scales to rotational and
  linear Section Vc strains, regenerated the canonical data and figures, and
  added steady-state regulation MAE over the final 10% of each evaluated
  setpoint interval.
- Refactored Viser sequence rendering around deterministic scene and frame
  hooks, synchronized video and lossless snapshot capture, explicit snapshot
  validation, and a render-only workflow for the operational-space paper scene.

### Fixed

- Rejected paused or looping playback modes for finite Viser video exports.
- Included `cbfpy` in the test dependency set so the Section Ve rendering tests
  collect across every supported Python version.
- Corrected the documented Section Vc steady-state evaluation window from the
  final quarter to the final 10% (0.3 s) of each 3 s setpoint interval.

## [0.2.0] - 2026-07-29

### Added

- First SoRoMoX package release published to PyPI.
- `ArticulatedSoftRobot` spatial serial-chain system with optional joint
  stiffness and damping, dense matrix dynamics, ABA forward dynamics, examples,
  documentation, and benchmark coverage.
- Geometric Variable-Strain (GVS), tendon-actuated GVS, I-SUPPORT, and
  McKibben-actuated UMArm models.
- Model-based control, system-identification, residual-learning, reinforcement
  learning, rendering, and CPU/GPU benchmarking examples.
- Open3D and Viser renderers with trajectory, actuator, and video support.

### Changed

- Generalized continuum actuation around composable threadlike actuation models.
- Standardized system, geometry, Lie algebra, operational-space, actuation-space,
  parameter, and base-pose APIs.
- Declared support for Python 3.11 through 3.14 and migrated the documentation
  build to Zensical.
- Set JAX 0.10.0 and Diffrax 0.7.2 as the minimum numerical dependency
  versions validated by CI.
- Expanded the software authorship and citation metadata.

### Fixed

- Corrected dynamics, Jacobian, rendering, control, rollout, and multi-segment
  I-SUPPORT behavior across the supported systems.

## [0.1.0] - 2025-09-01

### Added

- Initial SoRoMoX development milestone.

### Features

- Fast, parallelizable, and differentiable soft robot models implemented in JAX
- Implemented systems:
  - N-link pendulum
  - Planar Piecewise Constant Strain (PCS) continuum soft robot
  - Spatial Piecewise Constant Strain (PCS) continuum soft robot
  - Planar Handed Shearing Auxetics (HSA) robot

---

## Version History

- **v0.1.0**: Initial release with core functionality
<!-- - **Future**: Planned additions include 3D robot models, additional soft robot types, and enhanced visualization tools -->
