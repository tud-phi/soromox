"""Diffrax-compatible numerical integrators for Soromox state layouts."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, ClassVar

import jax
from diffrax import (
    RESULTS,
    AbstractSolver,
    AbstractTerm,
    LocalLinearInterpolation,
)

from soromox.systems.dynamical_system import DynamicalSystem


class SemiImplicitEuler(AbstractSolver):
    """Kick-drift Euler for a Soromox ``[q, qd, auxiliary]`` state.

    Diffrax's generic :class:`diffrax.SemiImplicitEuler` requires a separable
    two-term vector field. Soromox dynamics generally include
    velocity-dependent forces, so this solver evaluates the full state
    derivative at ``(q_n, qd_n)`` and applies

    ``qd_{n+1} = qd_n + dt * qdd_n`` and
    ``q_{n+1} = retract(q_n, dt * qd_{n+1})``.

    The solver is intended for :class:`soromox.systems.DynamicalSystem`
    rollouts. It uses the supplied system's state helpers rather than splitting
    the state in half, so unequal floating-base configuration and velocity
    dimensions, manifold retraction, and trailing system auxiliary states are
    supported. Diffrax-level control and environment state leaves are advanced
    with explicit Euler.

    Attributes:
        system: Soromox system that supplies ``split_state``, ``pack_state``,
            and ``retract_configuration``.
    """

    system: DynamicalSystem
    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    def order(self, terms: AbstractTerm) -> int:
        """Return the deterministic convergence order.

        Args:
            terms: Diffrax term describing the vector field. The order is one
                for every compatible term.

        Returns:
            The deterministic convergence order, which is one.
        """
        del terms
        return 1

    def init(
        self,
        terms: AbstractTerm,
        t0: Any,
        t1: Any,
        y0: Any,
        args: Any,
    ) -> None:
        """Initialize the solver state.

        Args:
            terms: Diffrax term describing the vector field.
            t0: Initial time of the integration interval.
            t1: Final time of the integration interval.
            y0: Initial Diffrax state.
            args: Additional arguments supplied to the vector field.

        Returns:
            ``None`` because the method has no persistent solver state.
        """
        del terms, t0, t1, y0, args
        return None

    def step(
        self,
        terms: AbstractTerm,
        t0: Any,
        t1: Any,
        y0: Any,
        args: Any,
        solver_state: Any,
        made_jump: Any,
    ) -> tuple[Any, None, dict[str, Any], None, RESULTS]:
        """Advance the state by one kick--drift step.

        Args:
            terms: Diffrax term describing the vector field.
            t0: Start time of the step.
            t1: End time of the step.
            y0: Diffrax state at ``t0``.
            args: Additional arguments supplied to the vector field.
            solver_state: Persistent solver state. This method does not use it.
            made_jump: Whether the solution jumped at ``t0``. This method does
                not require special jump handling.

        Returns:
            A Diffrax step tuple containing the state at ``t1``, no local error
            estimate, linear-interpolation data, no persistent solver state,
            and a successful result code.
        """
        del solver_state, made_jump
        control = terms.contr(t0, t1)
        increment = terms.vf_prod(t0, y0, args, control)
        y1 = jax.tree.map(
            lambda initial, delta: None if initial is None else initial + delta,
            y0,
            increment,
            is_leaf=lambda value: value is None,
        )

        q0, qd0, auxiliary0 = self.system.split_state(y0.y)
        _, qd_increment, auxiliary_increment = self.system.split_state(increment.y)
        qd1 = qd0 + qd_increment
        q1 = self.system.retract_configuration(q0, (t1 - t0) * qd1)
        auxiliary1 = auxiliary0 + auxiliary_increment
        primary_state1 = self.system.pack_state(q1, qd1, auxiliary1)
        y1 = dataclasses.replace(y1, y=primary_state1)

        dense_info = {"y0": y0, "y1": y1}
        return y1, None, dense_info, None, RESULTS.successful

    def func(self, terms: AbstractTerm, t0: Any, y0: Any, args: Any) -> Any:
        """Evaluate the underlying vector field.

        Args:
            terms: Diffrax term describing the vector field.
            t0: Time at which to evaluate the vector field.
            y0: Diffrax state at ``t0``.
            args: Additional arguments supplied to the vector field.

        Returns:
            Vector-field evaluation with the same PyTree structure as ``y0``.
        """
        return terms.vf(t0, y0, args)


__all__ = ["SemiImplicitEuler"]
