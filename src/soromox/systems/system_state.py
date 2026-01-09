__all__ = ["SystemState"]
from dataclasses import dataclass
from typing import Any

import jax
from jax import Array


@jax.tree_util.register_pytree_node_class
@dataclass
class SystemState:
    """
    Container for the system state and optional controller state.

    Attributes:
        t: Current simulation time.
        y: Robot state vector, typically concatenated configuration and velocity.
        u: Actuation input applied at the current time.
        control_state: Additional controller state as a PyTree (e.g., integrator terms).
    """

    t: Array
    y: Array
    u: Array | None = None
    control_state: Any | None = None

    def tree_flatten(self):
        children = [self.t, self.y]
        aux = {
            "has_u": self.u is not None,
            "has_control_state": self.control_state is not None,
        }
        if self.u is not None:
            children.append(self.u)
        if self.control_state is not None:
            children.append(self.control_state)
        return tuple(children), aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        t, y, *rest = children
        idx = 0
        u = rest[idx] if aux["has_u"] else None
        if aux["has_u"]:
            idx += 1
        control_state = rest[idx] if aux["has_control_state"] else None
        return cls(t=t, y=y, u=u, control_state=control_state)
