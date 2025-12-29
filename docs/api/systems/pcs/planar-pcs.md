# Planar PCS

The planar Piecewise Constant Strain (PCS) systems provide implementations for 2D soft continuum robots using the Cosserat rod theory with piecewise constant strain assumptions, based on the discrete Cosserat approach by Renda et al. (2018).

## Overview

The PCS model divides the continuum robot into segments, each with constant strain properties. This approach, originally proposed by Renda et al. (2018), provides a good balance between computational efficiency and modeling accuracy for soft continuum robots.

## Main Implementation

::: soromox.systems.pcs.planar_pcs
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
