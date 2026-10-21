# GVS Warp Integration API

!!! warning "Integration API"
    SoRoMoX users should call the public `GVS` kinematics and dynamics methods.
    The symbols on this page are intended for packages that reuse the GVS
    kinematics, dynamics, or actuation pipelines with preallocated Warp
    buffers.

## Batch layouts and naming

| Entry points | Input layout |
| --- | --- |
| `homogeneous_*_kernel` | One `*Data` struct passed directly, referencing a rectangular batch with common structural dimensions and coordinate layout. |
| `launch_homogeneous_*` | Array arguments at the JAX boundary; the launcher assembles the shared struct and launches the homogeneous kernel. |
| `packed_*_kernel` | Per-world `*Data` structs and an INT32 work map `(world_index, local_index, 0)`; each world may have its own structure and parameters. |
| `*_at` | Shared numerical functions with explicit indices into the supplied array views. |

Values can vary by world when their arrays carry a world axis. Other inputs
are shared: for example, homogeneous cell kernels share bases and geometry,
and homogeneous chain kernels share weighted masses and gravity. Each symbol
documents its sharing rules, indexing and launch dimensions. Callers retain
all referenced arrays. Packed descriptors may share matching model arrays.

The same struct type serves both kernel families. A homogeneous kernel receives
one struct whose batched fields span all `E` worlds. A packed kernel receives an
array of these structs, each containing one world's views (`E = 1`). Homogeneous
kernels compute indices directly from the launch grid; packed kernels select a
descriptor through the work map.

Native callers can bind a struct once and reuse it while its array views remain
valid. Assign fields on the host before launching, and keep the referenced arrays
alive throughout CUDA graph use. Updating array contents preserves the binding;
replacing an allocation requires rebinding the field and recapturing graphs that
use the old allocation.

## Linear-threadlike actuation

`GVSThreadlikeOperands.from_model(robot)` exposes the ordered built-in linear
routing data together with interior body quadrature, link strain bases and
reference strains, pre-scaled link bases, physical quadrature coordinates and
weights, segment spans, and link-local/global coordinate maps.
`GVSThreadlikeShapes` specifies the shared caller-owned strain workspace
`(E, S, Q, 6)`, matrix output `(E, D, A)`, and fused-force output `(E, D)`.

The exact launch sequences are:

1. `launch_homogeneous_gvs_threadlike_actuation_matrix(...)` prepares every quadrature
   strain once in `(E, S, Q, 6)`, then integrates and writes every matrix entry.
2. `launch_homogeneous_gvs_threadlike_actuation_force(...)` prepares the same workspace,
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
