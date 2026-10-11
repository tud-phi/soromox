# GVS Warp Integration API

!!! warning "Integration API"
    SoRoMoX users should call the public `GVS` kinematics and dynamics methods.
    The symbols on this page are intended for packages that reuse the GVS
    kinematics, dynamics, or actuation pipelines with preallocated Warp
    buffers.

## Heterogeneous worlds

Packed kernels use an array of typed `*Data` structs and a work map of INT32
triples `(world_index, local_index, 0)`. Each struct holds non-owning array views
for one world's inputs, workspaces, and outputs. Keep the referenced allocations
alive for every launch and captured graph. The kernel docstrings specify the
local index and launch dimensions. Worlds can have different coordinate counts,
strain bases, quadrature sizes, and joint types without grouping by morphology.

Direct array kernels serve the JAX/Warp boundary and homogeneous native callers.
Both entry points call the same explicit-index numerical functions. Packed
kernels support independent allocation sizes; their performance depends on the
stage and model sizes. The optional native Cholesky solve and prepared
fixed-joint executor require explicit selection.

## Linear-threadlike actuation

`GVSThreadlikeOperands.from_model(robot)` exposes the ordered built-in linear
routing data together with interior body quadrature, link strain bases and
reference strains, pre-scaled link bases, physical quadrature coordinates and
weights, segment spans, and link-local/global coordinate maps.
`GVSThreadlikeShapes` specifies the shared caller-owned strain workspace
`(E, S, Q, 6)`, matrix output `(E, D, A)`, and fused-force output `(E, D)`.

The exact launch sequences are:

1. `launch_gvs_threadlike_actuation_matrix(...)` prepares every quadrature
   strain once in `(E, S, Q, 6)`, then integrates and writes every matrix entry.
2. `launch_gvs_threadlike_actuation_force(...)` prepares the same workspace,
   applies `coordinate_scale[path] * controls[e, path]`, and reduces paths
   directly into `(E, D)`. CUDA uses one 128-lane block per environment and
   segment; Warp CPU uses a scalar-coordinate kernel.

Both second stages explicitly write inactive paths and joint-only coordinates
as zero, so neither sequence requires a separate output-clear launch. Matrix
and force calls may reuse the same strain workspace when they do not overlap.

Floating-point inputs, workspaces, and outputs are FP64; coordinate maps and
path spans are INT32. Callers own and retain all Warp views and buffers.
Neither launcher allocates, calls JAX, synchronizes, or transfers data, and
both two-launch sequences may be captured in a CUDA graph. Environment,
segment, link-DOF, quadrature, and path dimensions remain runtime-shaped. The
normalization test is exactly
`norm > global_eps`; a collapsed tangent contributes zero, matching JAX.

::: soromox.execution.warp.gvs
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
      members_order: source
      docstring_section_style: table
