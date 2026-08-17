"""Shared gain-optimization loop for the Section Vd study.

Both Section Vd generators run the same Optax loop over closed-loop rollouts,
so it lives here and each generator supplies only its own loss and optimizer.

The loop records the parameters each loss was measured at, not the ones the
subsequent update produced, so ``argmin`` over the losses indexes the matching
gains. No unused update is computed after the final requested evaluation.

A candidate is stored only when its loss, trajectory, gradient and parameters
are all finite. The first non-finite candidate stops the loop without being
stored, which keeps the saved history and the best-candidate selection clean.
Checking the gradient matters as much as the trajectory: a forward rollout can
stay healthy while the reverse pass produces ``NaN``. If only an optimizer
update is non-finite, its finite input iterate remains available as the best
fallback and the loop stops before adopting the invalid next state.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
from jax import Array
from jax import numpy as jnp

from soromox.utils.diagnostics import format_nonfinite_report, nonfinite_leaves

__all__ = ["OptimizationHistory", "run_gain_optimization"]


@dataclass
class OptimizationHistory:
    """Per-iteration record of a gain-optimization run.

    Every list is indexed by completed iteration and all lists have the same
    length. Entry ``k`` of :attr:`opt_vars` holds the parameters that produced
    entry ``k`` of :attr:`loss` and :attr:`aux`, so ``argmin(loss)`` indexes the
    parameters that actually attained the best loss.

    Attributes:
        loss: Scalar loss measured at each recorded iterate.
        opt_vars: Optimization variables the corresponding loss was measured at.
        aux: Auxiliary rollout outputs returned alongside each loss.
        time_iter: Wall-clock duration of each iteration in seconds. The first
            entry includes tracing and compilation.
        stopped_early: Whether the loop halted before ``num_iters`` because a
            candidate was non-finite.
        stop_reason: Human-readable description of the early stop, or ``None``.
    """

    loss: list[Array] = field(default_factory=list)
    opt_vars: list[Any] = field(default_factory=list)
    aux: list[dict[str, Array]] = field(default_factory=list)
    time_iter: list[float] = field(default_factory=list)
    stopped_early: bool = False
    stop_reason: str | None = None

    def __len__(self) -> int:
        """Return the number of recorded iterations."""
        return len(self.loss)

    def _require_recorded(self, action: str) -> None:
        """Raise if no iterate was recorded, reporting why the loop stopped.

        Args:
            action: What the caller was trying to do, used in the message.

        Raises:
            RuntimeError: If no finite iterate was recorded.
        """
        if not self.loss:
            raise RuntimeError(
                f"No finite optimization iterate was recorded; nothing to {action}. "
                f"Stop reason: {self.stop_reason}"
            )

    def best_index(self) -> int:
        """Return the index of the lowest recorded loss.

        Returns:
            Index into every history list.

        Raises:
            RuntimeError: If no finite iterate was recorded.
        """
        self._require_recorded("select")
        return int(jnp.argmin(jnp.asarray(self.loss)))

    def stacked_aux(self, key: str) -> Array:
        """Stack one auxiliary field across iterations.

        Args:
            key: Auxiliary dictionary key, e.g. ``"q_ts"``.

        Returns:
            Array with a leading iteration axis.

        Raises:
            RuntimeError: If no finite iterate was recorded.
        """
        self._require_recorded("stack")
        return jnp.stack([entry[key] for entry in self.aux], axis=0)


def run_gain_optimization(
    gradient_fn: Callable[[Any], tuple[tuple[Array, dict[str, Array]], Any]],
    optimizer,
    opt_vars: Any,
    num_iters: int,
    progress_label: str = "Optimization",
) -> OptimizationHistory:
    """Run the Optax gain-optimization loop with consistent history pairing.

    Args:
        gradient_fn: Callable mapping the optimization variables to
            ``((loss, aux), gradient)``, typically
            ``jit(value_and_grad(evaluate_closed_loop_system, has_aux=True))``
            with the controller already bound.
        optimizer: Initialized Optax optimizer (an object exposing ``init`` and
            ``update``).
        opt_vars: Initial optimization variables as a pytree.
        num_iters: Number of iterations to attempt. Must be at least one.
        progress_label: Prefix used in the progress log line.

    Returns:
        :class:`OptimizationHistory` whose entries are pairwise consistent and
        contain only finite values.

    Raises:
        ValueError: If ``num_iters`` is smaller than one.
    """
    if num_iters < 1:
        raise ValueError("num_iters must be at least 1")

    import optax  # Imported lazily; only the paper_results extra provides it.

    opt_state = optimizer.init(opt_vars)
    history = OptimizationHistory()

    for iteration in range(num_iters):
        iter_start = time.time()

        # Keep the parameters the loss belongs to before the update moves them.
        opt_vars_evaluated = opt_vars
        try:
            (loss, aux), grad = gradient_fn(opt_vars)
            jax.block_until_ready((loss, aux, grad, opt_vars_evaluated))
        except KeyboardInterrupt:
            history.stopped_early = True
            history.stop_reason = (
                f"interrupted during iteration {iteration + 1}; preserving "
                f"{len(history)} completed iterations"
            )
            print(f"\n[WARNING] {history.stop_reason}.")
            break

        # Reject an invalid evaluation without poisoning the history.
        candidate = {
            "loss": loss,
            "aux": aux,
            "gradient": grad,
            "opt_vars": opt_vars_evaluated,
        }
        report = nonfinite_leaves(candidate)
        if report:
            history.stopped_early = True
            history.stop_reason = format_nonfinite_report(
                f"iteration {iteration}", report
            )
            print(f"\n[WARNING] {history.stop_reason}. Stopping without recording.")
            break

        # Do not compute an update after the final requested evaluation. For
        # earlier iterations, validate the entire next optimizer state before
        # adopting it. A failed update must not discard this finite evaluation.
        next_report = []
        if iteration < num_iters - 1:
            updates, opt_state_next = optimizer.update(
                grad, opt_state, params=opt_vars_evaluated
            )
            opt_vars_next = optax.apply_updates(opt_vars_evaluated, updates)
            jax.block_until_ready((updates, opt_state_next, opt_vars_next))
            next_report = nonfinite_leaves(
                {
                    "updates": updates,
                    "opt_state_next": opt_state_next,
                    "opt_vars_next": opt_vars_next,
                }
            )
            if not next_report:
                opt_state = opt_state_next
                opt_vars = opt_vars_next

        iter_duration = time.time() - iter_start

        history.loss.append(loss)
        history.opt_vars.append(opt_vars_evaluated)
        history.aux.append(aux)
        history.time_iter.append(iter_duration)

        if next_report:
            history.stopped_early = True
            history.stop_reason = format_nonfinite_report(
                f"iteration {iteration} next optimizer state", next_report
            )
            print(
                f"\n[WARNING] {history.stop_reason}. "
                "Stopping after recording the finite evaluation."
            )
            break

        if iteration == 0:
            print(
                f"\n[INFO] Compilation + first execution time = {iter_duration:.2f} s"
            )
            continue

        steady_times = history.time_iter[1:]
        t_left = (num_iters - iteration - 1) * sum(steady_times) / len(steady_times)
        eta = time.localtime(time.time() + t_left)
        print(
            f"{progress_label}: {100 * (iteration + 1) / num_iters:3.1f} %  |  "
            f"iteration {iteration + 1:>4d} of {num_iters:<4d}  |  "
            f"iter time = {iter_duration:>.2f} s  |  "
            f"ETA = {eta.tm_mday:02d}/{eta.tm_mon:02d}/{eta.tm_year} "
            f"{eta.tm_hour:02d}:{eta.tm_min:02d}",
            end="\r",
        )

    print(
        f"\n{progress_label}: recorded {len(history)} of {num_iters} iterations"
        + ("  (stopped early)" if history.stopped_early else "")
    )

    return history
