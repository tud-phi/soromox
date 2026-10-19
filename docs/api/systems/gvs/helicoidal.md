# Helicoidal

`Helicoidal` models tendon-driven helicoidal lattice robots using GVS strain
bases, quadrature, and continuous tendon routing. It supports fixed and floating
bases with shared JAX and Warp dynamics.

## Construction

Use `Helicoidal.from_segments(...)` with `GVSSegment` specifications. Helix
inputs derive distributed inertia from strand geometry and material density.
Other segments, including rigid cylindrical assemblies, use their ordinary GVS
cross-section geometry and density. Optional `tip_mass` [kg] adds translational
tip inertia; full attached bodies use the shared
[inertia representation](../continuum-components.md#distributed-and-attached-body-inertia).

The optional pair `mass_per_length` [kg/m] and
`rotational_inertia_per_length` [kg m, three components] overrides distributed
inertia for all links. This is useful for externally identified backbone models.

Supply constitutive parameters through a mapping from segment indices to
`HelicoidalBackboneParams` or `HelicoidalHelixParams`. Each object describes
one segment; a mapping can mix the two types or reuse an object across segments:

```python
robot = Helicoidal.from_segments(
    segments,
    constitutive_params={1: helix, 3: backbone, 5: helix},
)
```

Selected segments receive projected generalized stiffness and damping in
`robot.params.link`. Other segments use the matrices in their `LinkSpec`.

### Backbone Parameters

`HelicoidalBackboneParams` describes the effective lattice response in the local
**backbone material frame**: x is longitudinal, and the component order is
`[twist, bend_y, bend_z, extension, shear_y, shear_z]`.

```python
from soromox.systems import HelicoidalBackboneParams

law = HelicoidalBackboneParams(
    stiffness_base=K_base,
    stiffness_tip=K_tip,
    damping_time_constant=tau,
)
robot = robot.with_constitutive_params({1: law, 3: law, 5: law})
```

Endpoint arrays have shape `(6,)`, with positive rotational stiffnesses
in N m² and translational stiffnesses in N. Compliance interpolates linearly
between their reciprocals; equal endpoint values give constant stiffness.
The law is projected using the configured strain basis and Gauss rule.
`damping_time_constant` accepts a scalar or six values in seconds, ordered as
`[twist, bend_y, bend_z, extension, shear_y, shear_z]` in the backbone material
frame. Each value scales its sectional stiffness before Gauss projection;
a scalar gives `D_link = tau * K_link`. For example:

```python
law = law.replace(damping_time_constant=[0.002, 0.02, 0.02, 0.02, 0.02, 0.02])
robot = robot.with_constitutive_params({1: law, 3: law, 5: law})
```

Values must be finite and nonnegative; zero disables damping for that strain.
Active strains are selected in `StrainBasisSpec`. Numeric updates reuse compiled
methods when array shapes and dtypes remain fixed.

`HelicoidalBackboneParams.from_compliance(...)` also accepts affine compliance
coefficients and a segment length; see its API reference below.

### Helix Parameters

`HelicoidalHelixParams` describes one soft segment with two oppositely wound
strand families. Material and sectional properties belong to cross-sections
**normal to each strand's tangent**. Lengths are in metres and angles in radians.

```python
from math import pi
from soromox.systems import HelicoidalHelixParams, StrandSection

helix = HelicoidalHelixParams.from_section(
    StrandSection.circular(radius=0.002),
    length=0.2,
    radius_base=0.025,
    radius_tip=0.025,
    num_turns=3.0,                      # Each strand revolves around the backbone 3 times.
    helices_per_direction=3,           # Three strands in each winding direction.
    winding_family_phase_offset=pi/6,  # Families' base azimuths differ by 30 degrees.
    density=1220.0,
    young_modulus=30e6,
    shear_modulus=30e6 / (2 * (1 + 0.48)),
    fusion_stiffness_multiplier=4.0,    # Scale the stiffness of fused strand crossings.
    damping_time_constant=0.02,
)
backbone = helix.to_backbone()
# Use backbone.admissible_strains to choose the soft segments' strain bases.
robot = robot.with_constitutive_params({1: backbone, 3: backbone, 5: backbone})
robot = robot.with_helix_inertia({1: helix, 3: helix, 5: helix})
```

The reduction solves six unit-load cases at the soft segment tip relative to its
fixed base and fits a diagonal backbone law. It includes deformation of the
strand connections to the tip ring. The returned backbone parameters carry
`admissible_strains`; the model checks this mask against each segment's basis.
An inadmissible shear fit requires both shear strains to be deactivated.
`helix.reduction_diagnostics()` reruns the reduction to report the full tip
compliance, residual shear compliance, and numerical asymmetry.

Passing helix inputs directly to `constitutive_params` or
`with_constitutive_params` prepares both constitutive matrices and inertia.
Strand inertia includes cross-sectional moments, orientation, backbone offsets,
and the strand-to-backbone arc-length factor. It is sampled at the existing
Gauss points, supporting tapered geometries. Strand volumes add at crossings;
overlap is neglected. Each strand section moves with its backbone material frame.

`StrandSection` supports solid and hollow circles and rectangles through
`circular`, `circular_hollow`, `rectangular`, and `rectangular_hollow` constructors.
Rectangular height follows strand **y** (projected outward radial direction),
and width follows **z**, matching GVS links. Hollow constructors take a uniform
`wall_thickness`.
The same section supplies mechanical properties and viewer geometry. You may
also supply measured area, moments, torsion constant, and shear factors directly
or override the estimates in `from_section`. `damping_time_constant` applies to
the fitted **backbone** stiffness.

Run the cylindrical example with strand-derived mechanics using:

```bash
JAX_PLATFORMS=cpu python examples/simulation/gvs/simulate_helicoidal.py \
    --parameter-domain helix --duration 2 --tension 0.5 --viser
```

The example uses square-root sectional damping and Tsit5 at 250 µs.

## Parameter Updates

Use the same method for either input type:

```python
robot = robot.with_constitutive_params({1: helix.replace(young_modulus=35e6)})
robot = robot.with_constitutive_params({3: backbone.replace(stiffness_tip=K_tip)})
```

Reapply the parameters after changing segment lengths or constitutive inputs.
Helix lengths must match their selected GVS segments. Shared helix objects are
reduced once per call. Numeric backbone updates preserve `admissible_strains`.
Helix inputs also refresh inertia. For density-only updates, or after explicit
conversion with `to_backbone()`, use the differentiable `with_helix_inertia`:

```python
robot = robot.with_helix_inertia({1: helix.replace(density=1250.0)})
```

To change strand shape, replace `strand_section` and reapply the helix inputs:

```python
section = StrandSection.rectangular_hollow(
    width=0.004, height=0.002, wall_thickness=0.0002,
)
helix = helix.replace(strand_section=section)
robot = robot.with_constitutive_params({1: helix})
```

This rederives sectional properties; supply calibrated overrides in the same
`replace` call. Material-only updates preserve existing sectional properties.

Ordinary cylinder density updates use `update_link_params(density=...)`.
Stored helix inertia is refreshed through the helix input; constitutive-only
backbone updates preserve the current inertia.

Helix reduction runs eagerly on the CPU during preparation. Use backbone
parameters for differentiation through constitutive inputs. Prepared models
reuse compiled dynamics when topology, array shapes, and dtypes stay fixed.
Basis choices, quadrature counts, and tendon spans define the static structure.

## Specialized Viser Rendering

`HelicoidalViserRenderer` displays both winding families, connectors,
and tendon routes, following the GVS material frames through deformation.

```python
from soromox.rendering import HelicoidalModuleVisual, HelicoidalViserRenderer

renderer = HelicoidalViserRenderer(
    robot,
    modules=tuple(
        HelicoidalModuleVisual.from_helix_params(i, helix)
        for i in (1, 3, 5)
    ),
)
renderer.show(q)
```

Each module selects a link whose cross-section radius sets the helix centreline
radius. Keep the link's radius profile consistent with the helix inputs.
`from_helix_params` copies the strand section, turns, strand count, and both
phases. Property-only helix inputs have no recoverable shape; pass an explicit
`strand_section` to the adapter in that case. The adapter reads the preparation
inputs; reduced backbone matrices do not retain the original geometry.

`HelicoidalModuleVisual` also accepts `strand_section` directly, defaulting to
`StrandSection.circular(0.002)`.
The shared contour/sweep code builds cavity walls and annular caps for hollow
sections. Rebuild the renderer after changing section geometry or winding inputs.

`HelicoidalVisualConfig` controls colors and mesh resolution. The renderer
supports batched configurations, `render_sequence`, `render_frame`, and live
updates. Image and video capture use a connected Viser browser client.

## API Reference

::: soromox.systems.components.StrandSection
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: soromox.systems.gvs.helicoidal.Helicoidal
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      inherited_members: false
      members_order: source

::: soromox.systems.gvs.params.HelicoidalBackboneParams
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: soromox.systems.gvs.params.HelicoidalHelixParams
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: soromox.rendering.helicoidal.HelicoidalViserRenderer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      inherited_members: false

::: soromox.rendering.helicoidal.HelicoidalModuleVisual
    options:
      heading_level: 3

::: soromox.rendering.helicoidal.HelicoidalVisualConfig
    options:
      heading_level: 3
