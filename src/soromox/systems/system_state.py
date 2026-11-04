__all__ = ["EnvironmentState", "SystemState"]
from dataclasses import dataclass
from typing import Any, TypeAlias

import jax
from jax import Array

from soromox.systems.model_terms import ModelTerms

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
        model_terms: Model terms of the robot at ``(t, y)`` (see
            ``SoftRobot.model_terms``), or None. Model-based controllers read their
            quantities at the current state from them; without them, a controller
            computes the terms of its own model. The computed-torque tracker and
            the synergistic controller set their own terms for both parts of their
            control law. A controller that changes ``y`` before calling another
            controller must set this field to None. Rollout outputs never carry
            terms.
    """

    t: Array
    y: Array
    u: Array | None = None
    control_state: Any | None = None
    environment_state: Any | None = None
    model_terms: ModelTerms | None = None

    def tree_flatten(self):
        children = [self.t, self.y]
        aux = {
            "has_u": self.u is not None,
            "has_control_state": self.control_state is not None,
            "has_environment_state": self.environment_state is not None,
            "has_model_terms": self.model_terms is not None,
        }
        if self.u is not None:
            children.append(self.u)
        if self.control_state is not None:
            children.append(self.control_state)
        if self.environment_state is not None:
            children.append(self.environment_state)
        if self.model_terms is not None:
            children.append(self.model_terms)
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
        if has_environment_state:
            idx += 1
        model_terms = rest[idx] if aux.get("has_model_terms", False) else None
        return cls(
            t=t,
            y=y,
            u=u,
            control_state=control_state,
            environment_state=environment_state,
            model_terms=model_terms,
        )
