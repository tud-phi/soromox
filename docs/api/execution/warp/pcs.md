# PCS Warp Integration API

!!! warning "Integration API"
    SoRoMoX users should call `PCS` or `PlanarPCS` system methods. The symbols
    on this page are intended for packages that reuse PCS kinematics,
    dynamics, or actuation pipelines with preallocated Warp buffers.

## Linear-threadlike actuation

`PCSThreadlikeOperands.from_model(robot)` is the public, ordered operand bundle
for built-in `ThreadlikeRouting.linear` actuator groups. It retains precomputed
physical quadrature coordinates and weights, reference strains, active-strain
indices and scales, segment starts and lengths, routing intercepts and slopes,
inclusive path spans, and signed/physical coordinate scales.
`PCSThreadlikeShapes` specifies caller-owned matrix output `(E, D, A)` and
fused-force output `(E, D)`.

Use one of the planar or spatial launchers below, matching the model dimension:

1. `launch_*_threadlike_actuation_matrix(...)` integrates every segment/path,
   writes inactive spans as zero, and projects through the active-strain map.
2. `launch_*_threadlike_actuation_force(...)` applies
   `coordinate_scale[path] * controls[e, path]` inside path reduction and
   writes every active coordinate without materializing an actuation matrix.

All input and output floating-point arrays must be FP64; routing-span and
active-index arrays are INT32. The caller converts and retains the operand
arrays as Warp views and allocates the outputs before launch. The launchers
perform no allocation, JAX call, synchronization, or host/device transfer.
Each operation is a single write-complete CUDA-graph-capturable launch.
Segment, environment, path, active-DOF, and quadrature counts are
runtime-shaped.

::: soromox.execution.warp.pcs
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
      members_order: source
      docstring_section_style: table
