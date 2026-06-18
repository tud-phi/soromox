# Robot Parameter Assets

This directory contains robot parameter files used by examples, tests, and
source-checkout workflows. These files are repository assets and are not shipped
as Python package data in the `soromox` wheel.

Library APIs intentionally require explicit paths to these files. Code that runs
from an installed wheel should provide its own parameter file paths instead of
expecting bundled defaults.

## Layout

- `planar_hsa/`: Planar HSA parameter presets stored with `PlanarHSAParams`
  field names.
- `mckibben_umarm/`: Cached UMArm parameters stored with
  `McKibbenActuatedUMArmParams` field names.
