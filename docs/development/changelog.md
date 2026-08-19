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
- Differentiable unit-response mappings from Young's modulus with either shear
  modulus or Poisson's ratio, plus optional material damping, to canonical
  generalized link matrices.
- Continuum-component, parameter-update, material-optimization, and
  [parameter-API migration](../user-guide/parameter-api-migration.md)
  documentation with complete PCS and GVS examples.

### Changed

### Fixed

## [0.2.3] - 2026-08-19

### Added

- Reverse-mode-safe numerical primitives, strict singularity diagnostics, and
  pytree finiteness reporting for locating and handling degenerate model
  configurations.
- Public, dtype-aware ``jacobian_coefficients`` for forward and inverse
  Lie-group Jacobians, together with a reproducible Lie-algebra benchmark
  covering primal, Jacobian, Hessian, and analytic constant-strain derivative
  paths.
- Public ``constant_strain.se2.operators`` and
  ``constant_strain.se3.operators`` bundles with a fixed named result for
  callers that need adjoints, tangents, and an optional analytic tangent
  derivative at the same strain and arclength.
- Public ``se2`` and ``se3`` left Jacobians and analytic directional
  derivatives for accumulated Lie-algebra coordinates, independent of any
  constant-strain rod interpretation.
- Public ``constant_strain.se3.tangent_and_derivative`` for callers that need
  the selective tangent pair without constructing unused adjoint operators.
- An offline Section Vd post-processing renderer for saved optimization
  trajectories, with reproducible tracking plots, MP4 animations, optional GIF
  conversion, and intermediate metrics.

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
- Reworked ``SO(3)``, ``SE(2)``, and ``SE(3)`` exponential/logarithmic maps,
  constant-strain adjoint and tangent operators, and PlanarPCS pose integration
  around stable closed forms and high-order near-zero series. Constant-strain
  tangent derivatives remain explicit analytic production paths; runtime
  autodiff is used only as a test oracle.
- Organized constant-strain operators into dedicated ``se2`` and ``se3``
  modules with unsuffixed names. General left-Jacobian machinery now belongs
  to the rigid-body Lie-algebra modules; the constant-strain package retains
  arclength scaling, prepared powers, rod-specific adjoints, and the shared
  ``ConstantStrainOperators`` result.
- Fused repeated constant-strain operator evaluation in PlanarPCS, PCS, and
  GVS. The planar path uses direct SE(2) blocks; the spatial operators share
  exact fourth-order matrix powers and tangent coefficients. PCS additionally
  prepares arclength-independent powers once per segment, assembles transported
  tangents directly, and propagates the contracted ``Jd @ qd`` needed by
  forward dynamics instead of materializing unused matrices. PlanarPCS and GVS
  now use the same contracted-derivative strategy in their dynamics-only
  recurrences while preserving full Jacobian derivatives in public kinematics.
- Replaced generated symbolic PlanarHSA expressions with a JAX-native
  PlanarPCS implementation, reusing shared planar kinematics, Jacobians,
  quadrature, and dynamics; migrated parameter files to the descriptive
  ``reference_strain`` schema and removed the obsolete symbolic and
  legacy-parameter modules.
- Compared with ``main`` on CPU/JAX 0.11.0 across five fresh processes, the
  40-case Lie benchmark reduced geometric-mean steady-state and compile time by
  24.3% and 5.3%. A fresh raw before/after table is reported in this PR for
  PlanarPCS, PCS, and GVS.

- Updated the dependency metadata and lockfile for a CUDA 13-enabled JAX and
  PyTorch stack, and declared `ipywidgets` with the optional Open3D extras.
- Simplified rendering, example, paper-result, and complete installations by
  removing an unused video-encoding package; video output continues to use the
  system `ffmpeg` executable.
- Consolidated the Section Vd control-gain optimization scripts around a shared,
  testable optimization loop and independently configured collocated and
  synergistic cases, with shared CLI/output helpers, explicit output locations,
  overwrite protection, paper-result iteration defaults, ``gamma`` derived from
  a configurable tendon-length error scale for collocated integral saturation
  (default 10 mm), finite saturation-parameter validation, stale-artifact
  documentation, and consistent NumPy archive, pickle, and diagnostic-figure
  handling. Diagnostic figures are always saved,
  the obsolete ``--save-figures`` switch is no longer needed, and the primary
  optimization data is persisted before plotting or interactive rendering
  begins; saved trajectory diagnostics can be rendered without rerunning
  optimization.
- Normalized explicit continuous-rollout `save_ts` sequences to JAX arrays and
  preserved traced initial times for JIT-compiled and vectorized open- and
  closed-loop rollouts.

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
- Kept values and reverse-mode gradients finite at degenerate normalization,
  division, square-root, inverse-kinematics, and thread-routing sites across
  PCS, GVS, Planar HSA, articulated, McKibben, rotation, and reference-trajectory
  code paths.
- Sanitized inactive square-root and quotient inputs in SO(3)/SE(3) magnitude
  and quaternion-conversion branches, preventing batched selection from
  exposing NaNs under nested JVP/reverse-mode transforms; the SoftRobot
  forward-kinematics and spatial-Jacobian regressions from PR #148 now pass.
- Kept Lie exponential/logarithmic values, first derivatives, and Hessians
  finite and accurate at zero and small rotations, including batched SE(2) and
  arbitrary-axis SE(3) inputs, without adding autodiff to production paths.
- Made simulation benchmark timing treat Python solver/save parameters as static
  JAX arguments so the existing rollout benchmark can compile and run.
- Prevented automatically generated continuous-rollout save grids from
  overshooting the requested final time, while always including that endpoint
  exactly for both divisible and non-divisible save intervals.
- Set the Section Vd collocated and synergistic optimization defaults to the
  100 iterations used by the paper-result histories, and document that setting
  in the reproduction commands.
- Corrected Section Vd loss/parameter history pairing, rejected non-finite
  optimization candidates, and selected the best collocated and synergistic
  plotting batches independently.
- Guarded the collocated optimization reference against transient setpoints and
  collapsed tendon paths by using stable asymmetric tensions and failing fast
  when the generated steady state is unsuitable for reverse-mode optimization.
- Reported Open3D window-creation failures explicitly, with a clear error for
  environments that lack a usable display or OpenGL context.

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

## [0.2.2] - 2026-08-11

### Added

- Configurable, base-aligned ground planes for the Matplotlib, Open3D, and Viser
  renderers, including inherited support in the I-SUPPORT and UMArm renderers.
- A top-level Paper & Results documentation page with CPU and GPU benchmarks,
  all six application case studies, reproduction pointers, and curated
  web-optimized figures, animations, and videos from the SoRoMoX paper.
- A responsive core-renderer gallery, system-specific rendering samples, and a
  concise top-level contributor guide.

### Changed

- Improved Open3D and Viser efficiency for multi-robot animations with
  renderer-owned vectorized actuator geometry, batched actuator and sphere
  scene objects, automatically merged multi-robot Open3D backbones, and
  instanced or merged Viser geometry with one atomic transaction per frame.
  Viser ground planes now follow the complete batched layout. This substantially
  reduces backend registrations and browser scene handles without adding
  trajectory-rendering concerns to soft-robot mechanics. In a 16-robot,
  80-point, eight-frame
  benchmark, merged Open3D improved initial/frame rendering by 21.0x/2.7x
  (6.51 s to 0.31 s and 41.2 ms to 15.1 ms), while batched Viser improved full
  swept-video/browser-capture time by 2.3x/3.6x (27.9 s to 12.0 s and 2.98 s
  to 0.83 s) with 38.5x fewer live handles.
- Separated Section Vf reinforcement-learning rollout generation from rendering:
  trained and uniform-random baseline trajectories now retain the full parallel
  batch under `data/traj`, while the shared offline renderer derives single-arm
  or automatically framed grid MP4/GIF names from its input data under
  `outputs` and uses the paper's visual style. Renamed the canonical artifacts
  around policy state and environment count, versioned the current trajectory
  and render artifacts, and moved Section Vf MP4/GIF storage to Git LFS.
- Applied the shared camera field of view and tighter scene framing to
  Matplotlib's 3D renderer, bringing its gallery view closer to Open3D and
  Viser while retaining metric axes.
- Standardized the renderer gallery around consistent robot geometry, colors,
  ground planes, and framing, with tighter I-SUPPORT and UMArm close-ups.
- Refocused the README and documentation home on the JAX-native model
  implementations for articulated and continuum soft robots, their
  control-oriented interface, installation, performance, and first use.
- Reorganized the rendering documentation into an overview and gallery,
  backend API guide, and shared configuration reference; normalized Python
  examples throughout the generated package documentation as fenced code.
- Made the SoRoMoX arXiv preprint the primary academic citation, centralized
  citation guidance, and retained an exact-release software citation for
  reproducibility-sensitive work.
- Kept Citation at the top level of the documentation navigation immediately
  after Paper & Results, and grouped the supported systems following the paper.
- Placed Actuation before Control in the API navigation and removed the
  outdated symbolic-derivation documentation page.
- Moved detailed contributor and documentation tooling into the development
  guide and made `VERSION_BUMP_README.md` the authoritative maintainer release
  workflow, including the branch, pull-request, tag, and recovery procedures.
- Added Twine to the development dependency set so the documented distribution
  validation command is available in the project environment.
- Separated backend-specific renderer coverage into dedicated Matplotlib,
  Open3D, and Viser test modules, leaving the shared base-renderer tests focused
  on backend-independent behavior.
- Removed the obsolete duplicate actuation overview page in favor of the
  structured actuation documentation section.

### Fixed

- Renamed the operational-space Viser renderer test module and made Section Vf
  tests skip cleanly when optional paper dependencies are unavailable,
  preventing full-suite collection failures.
- Corrected the I-SUPPORT dynamic-model DOI and added platform references to
  the I-SUPPORT and McKibben UMArm system pages.

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
