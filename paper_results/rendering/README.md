# Rendering figure

The figure is 180 x 110 mm, with
serif labels, uppercase panel letters, white gutters, and the paper's coral
`post_opt_1` color (`#f1552e`). The dark base uses RGB `(0.2, 0.2, 0.2)`.

Panel A orders the core backends as OpenCV Planar, Matplotlib, Viser, Open3D.
Panel B shows Technical, Neutral, Bright / Flat, Clay, Dark with Open3D.
The panel letters share the first image-label line. This removes redundant
heading rows and reduces the original 126 mm height by 16 mm while preserving
the physical size of the shaded renderer and preset examples. Display crops
remove empty canvas from OpenCV and Matplotlib; all robot geometry and axis
labels are retained. The crops are recorded in the figure manifest.
Each image panel has a 0.35 pt pale-gray outline (`#d6d6d6`).

## Outputs

- `../final_outputs/renderers_and_presets.pdf`: publication figure; vector labels
  and original raster renderer captures.
- `../final_outputs/renderers_and_presets.svg`: editable composition.
- `../final_outputs/renderers_and_presets.png`: 300 dpi preview.
- `../final_outputs/renderers_and_presets.json`: image hashes and layout metadata.
- `outputs/captures/`: ten renderer captures and their resolved configurations.
- `figure.tex`: inclusion snippet and caption for the manuscript.

## Scene and comparison

The tentacle uses the Section Va fitted GVS model: two tapered links of lengths
0.305 m and 0.055 m, with the same nine active strain coordinates, material
parameters, and tendon routing as `identify_soft_tentacle_parameters.py`.
For fixed tendon tensions `[6.153846153846153, 0]` N, all nine coordinates are
found by solving the static force balance, including gravity:

```text
elastic_force(q) + gravitational_force(q) - actuation_matrix(q) @ tension = 0
```

The solver follows equilibria from zero tension, using a Newton-type root
method with an exact JAX Jacobian. The selected solution has generalized-force
residual norm 4.6e-16 and is locally stable in the damped model's linearization
(largest real eigenvalue -1.89 /s). Its backbone agrees with the corresponding
saved Section Va fitted-model curve to 3.5e-14 m after the mounting yaw.
The saved coordinates, parameter hashes, tension, and checks are in
`data/equilibrium.json`; four experimental tension levels are recorded in
`data/equilibrium_candidates.json`.

The robot hangs under gravity. A -60 degree rotation about the gravity axis
aligns the bending plane with world x-z while preserving the equilibrium.
Both tendons terminate at the end of the first segment; the second segment is
passive. Its curvature follows from the solved mechanical balance.

OpenCV uses a planar view adapter that projects the same GVS forward kinematics
onto x-z; the selected equilibrium is planar to numerical precision.
Matplotlib retains its line representation and uses
an oblique view with sparse axes for legibility. Viser and Open3D share their
camera and neutral studio settings. The six preset panels share geometry,
pose, camera, and framing. Clay applies its uniform material to the mount too.

All images are actual output from the named rendering backend. The composition
does not recolor or retouch them. Viser and Open3D have different lighting and
material implementations, so their output is not pixel-identical.

## Reproduce

From the repository root, with rendering dependencies installed:

```bash
python paper_results/rendering/code/equilibrium_scene.py
python paper_results/rendering/code/build_renderer_figure.py --capture simple
python paper_results/rendering/code/build_renderer_figure.py --capture open3d
python paper_results/rendering/code/build_renderer_figure.py --capture viser
python paper_results/rendering/code/build_renderer_figure.py --assemble
```

For Viser, open the printed local URL within five minutes. The script captures
the browser's rendered scene automatically and shuts down the server. Open3D
requires access to native graphics services and the patched project dependency.
The equilibrium solve needs SciPy and the project's simulation dependencies.
Captures load the saved equilibrium and verify its fitted-parameter hashes.
The commands replace this figure's generated artifacts. Assembly alone uses
the saved captures and does not start a renderer or simulation.

To include the figure in the manuscript, copy the PDF to its `figures/`
directory and insert the contents of `figure.tex` near the rendering discussion.
The source manuscript has not been edited by this figure-generation task.
