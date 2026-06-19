# Pressure Actuated Planar PCS

Pressure actuated Planar PCS systems with chamber-pressure-based actuation mechanisms, extending the discrete Cosserat approach by Renda et al. (2018).

## Overview

This module extends the planar PCS model (based on the discrete Cosserat approach by Renda et al., 2018) to include pressure actuation, where the robot is actuated by controlling internal chamber pressure. The model maps prescribed chamber pressures to generalized loads and does not model fluid supply dynamics.

## API Reference

::: soromox.systems.pcs.pressure_actuated_planar_pcs
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

## References

The PCS (Piecewise Constant Strain) model was originally proposed in:

Renda, F., Boyer, F., Dias, J., & Seneviratne, L. (2018). Discrete cosserat approach for multisection soft manipulator dynamics. *IEEE Transactions on Robotics*, 34(6), 1518-1533.
