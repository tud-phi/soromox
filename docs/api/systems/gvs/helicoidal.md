# Helicoidal

`Helicoidal` models tendon-driven helicoidal lattice robots using GVS strain
bases, quadrature, and continuous tendon routing. It supports fixed and floating
bases with shared JAX and Warp dynamics.

## Construction

Use `Helicoidal.from_segments(...)` with `GVSSegment` specifications and one
entry per link for `mass_per_length` [kg/m] and
`rotational_inertia_per_length` [kg m, three principal components]. Optional
`tip_mass` [kg] adds translational tip inertia. For distributed and attached-body
matrices, see the shared [inertia representation](../continuum-components.md#distributed-and-attached-body-inertia).

Choose either backbone endpoint stiffnesses through `constitutive_law=`, or
strand material and geometry through `helix_params=`. Both prepare the
canonical generalized stiffness and damping in `robot.params.link`.

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
robot = robot.with_constitutive_law(law)
```

Endpoint arrays have shape `(num_links, 6)`, with positive rotational stiffnesses
in N m² and translational stiffnesses in N. Compliance interpolates linearly
between their reciprocals; equal endpoint values give constant stiffness.
The law is projected using the configured strain basis and Gauss rule.
`damping_time_constant` has shape `(num_links,)` in seconds and gives
`D_link = tau * K_link`. Active strains are selected in `StrainBasisSpec`.

`HelicoidalBackboneParams.from_compliance(...)` also accepts affine compliance
coefficients and segment lengths; see its API reference below.

### Helix Parameters

`HelicoidalHelixParams` describes one soft segment with two oppositely wound
strand families. Material and sectional properties belong to cross-sections
**normal to each strand's tangent**. Lengths are in metres and angles in radians.

```python
from math import pi
from soromox.systems import HelicoidalHelixParams

helix = HelicoidalHelixParams.circular(
    length=0.2,
    radius_base=0.025,
    radius_tip=0.025,
    num_turns=3.0,                      # Each strand revolves around the backbone 3 times.
    helices_per_direction=3,           # Three strands in each winding direction.
    winding_family_phase_offset=pi/6,  # Families' base azimuths differ by 30 degrees.
    strand_radius=0.002,
    young_modulus=30e6,
    shear_modulus=30e6 / (2 * (1 + 0.48)),
    fusion_stiffness_multiplier=4.0,    # Scale the stiffness of fused strand crossings.
    damping_time_constant=0.02,
)
reduced = helix.reduce()
# Construct each soft segment's basis using reduced.strain_selector.
# Apply to a robot with matching segment lengths:
robot = robot.with_helix_params({1: helix, 3: helix, 5: helix})
```

The reduction solves six unit-load cases at the soft segment tip relative to its
fixed base and fits a diagonal backbone law. It includes deformation of the
strand connections to the tip ring. `reduced` contains the tip compliance,
backbone law, and admissible strain selector. Inadmissible fitted shear requires
both shear strains to be deactivated in the segment basis.

For noncircular strands, the constructor accepts area, principal area moments,
torsion constant, and shear correction factors. The parameter reference below
defines their frames and the connection stiffness multipliers.
`damping_time_constant` applies to the fitted **backbone** stiffness.

Run the cylindrical example with strand-derived mechanics using:

```bash
JAX_PLATFORMS=cpu python examples/simulation/gvs/simulate_helicoidal.py \
    --parameter-domain helix --duration 2 --tension 0.5 --viser
```

The example uses square-root sectional damping and Tsit5 at 250 µs.

## Parameter Updates

Reapply `with_constitutive_law(law)` after updating lengths or endpoint
stiffnesses. For helix inputs, update the corresponding geometry or material
parameters and call `with_helix_params(...)` again. Shared parameter objects are
reduced once per call; helix lengths must match their selected GVS segments.

Helix reduction runs eagerly on the CPU during preparation. Use backbone
parameters for differentiation through constitutive inputs. Prepared models
reuse compiled dynamics when topology, array shapes, and dtypes stay fixed.
Basis choices, quadrature counts, and tendon spans define the static structure.

## Specialized Viser Rendering

`HelicoidalViserRenderer` displays both winding families, end rings, connectors,
and tendon routes, following the GVS material frames through deformation.

```python
from soromox.rendering import HelicoidalModuleVisual, HelicoidalViserRenderer

renderer = HelicoidalViserRenderer(
    robot,
    modules=tuple(HelicoidalModuleVisual(i, turns=3, strand_radius=0.002)
                  for i in (1, 3, 5)),
)
renderer.show(q)
```

Each module selects a link whose cross-section radius sets the helix centreline
radius. `HelicoidalModuleVisual` specifies the strand geometry;
`HelicoidalVisualConfig` controls colors and mesh resolution. The renderer
supports batched configurations, `render_sequence`, `render_frame`, and live
updates. Image and video capture use a connected Viser browser client.

## API Reference

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

::: soromox.systems.gvs.params.HelicoidalReduction
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
