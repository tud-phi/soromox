__all__ = ["EnvironmentState", "SystemState"]
from dataclasses import dataclass
from typing import Any, TypeAlias

import jax
from jax import Array

EnvironmentState: TypeAlias = Any
"""Type alias for an environment state PyTree.

An ``EnvironmentState`` is any JAX-compatible PyTree (e.g., a dict of
``Array`` values, a nested structure, or a single array) that holds the
auxiliary state of a coupled environment model.  It is integrated alongside
the robot dynamics when an ``environment_model`` is supplied to a rollout
method.
"""


@jax.tree_util.register_pytree_node_class
@dataclass
class SystemState:
    """
    Simulation snapshot of the robot and optional coupled controller/environment state.

    `SystemState` is the public state object passed to controllers and environment
    models, and returned by rollout methods. The robot dynamics state is stored in
    `y`; `control_state` and `environment_state` store optional auxiliary dynamics
    that are coupled to the robot during simulation.

    Attributes:
        t: Current simulation time.
        y: Robot state vector, typically concatenated configuration and velocity.
        u: Actuation input applied at the current time.
        control_state: Additional controller state as a PyTree (e.g., integrator terms).
        environment_state: Additional environment state as a PyTree (e.g., contact state).
    """

    t: Array
    y: Array
    u: Array | None = None
    control_state: Any | None = None
    environment_state: Any | None = None

    def tree_flatten(self):
        children = [self.t, self.y]
        aux = {
            "has_u": self.u is not None,
            "has_control_state": self.control_state is not None,
            "has_environment_state": self.environment_state is not None,
        }
        if self.u is not None:
            children.append(self.u)
        if self.control_state is not None:
            children.append(self.control_state)
        if self.environment_state is not None:
            children.append(self.environment_state)
        return tuple(children), aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        t, y, *rest = children
        idx = 0
        u = rest[idx] if aux["has_u"] else None
        if aux["has_u"]:
            idx += 1
        control_state = rest[idx] if aux["has_control_state"] else None
        if aux["has_control_state"]:
            idx += 1
        has_environment_state = aux.get("has_environment_state", False)
        environment_state = rest[idx] if has_environment_state else None
        return cls(
            t=t,
            y=y,
            u=u,
            control_state=control_state,
            environment_state=environment_state,
        )
