# Pendulum Systems

This section covers articulated pendulum-based robot systems, from classical rigid-body chains to cable-driven mechanisms.

## Overview

Pendulum systems model articulated robots as N-link chains with revolute joints. These systems are useful for:

- Benchmarking and comparison studies
- Understanding fundamental dynamics
- Cable-driven soft robot research
- Educational purposes

## Available Systems

### [Pendulum](pendulum.md)

Classical (optionally articulated) N-link pendulum implementation.

- Rigid-body dynamics with revolute joints
- Configurable number of links
- Optional elastic and dissipative forces at joints
- Lightweight closed-form dynamics (no symbolic compilation needed)

### [Tendon Actuated Pendulum](tendon-actuated-pendulum.md)

N-link pendulum with active and passive tendon actuation.

- Cable-driven actuation via pulley routing
- Active and passive tendon support
- Spring-damper systems for compliance
- Hybrid rigid-soft robot characteristics

## When to Use

| System | Use Case |
|--------|----------|
| `Pendulum` | Basic articulated robots, benchmarking, rigid-body dynamics |
| `TendonActuatedPendulum` | Cable-driven actuation, passive compliance, tendon routing |
