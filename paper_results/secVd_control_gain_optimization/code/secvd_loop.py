"""Shared multi-start gain-optimization loop for the Section Vd study."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
from jax import Array
from jax import numpy as jnp

__all__ = ["OptimizationHistory", "run_gain_optimization"]


def _tree_batched_finite(tree: Any, batch_size: int) -> Array:
    """Return a ``(B,)`` mask of which starts are finite across a pytree."""
    flags = jnp.ones((batch_size,), dtype=bool)
    for leaf in jax.tree_util.tree_leaves(tree):
        value = jnp.asarray(leaf)
        if value.ndim >= 1 and value.shape[0] == batch_size:
            per_start = jnp.all(jnp.isfinite(value.reshape(batch_size, -1)), axis=1)
        else:
            per_start = jnp.broadcast_to(jnp.all(jnp.isfinite(value)), (batch_size,))
        flags = flags & per_start
    return flags


def _tree_global_norm(tree: Any, batch_size: int) -> Array:
    """Return the per-start global L2 norm over a batched pytree."""
    total = jnp.zeros((batch_size,))
    for leaf in jax.tree_util.tree_leaves(tree):
        value = jnp.asarray(leaf)
        if value.ndim >= 1 and value.shape[0] == batch_size:
            total = total + jnp.sum(
                jnp.square(value.reshape(batch_size, -1)).astype(jnp.float64), axis=1
            )
    return jnp.sqrt(total)


def _tree_select(new_tree: Any, old_tree: Any, take_new: Array) -> Any:
    """Per-start choice between two like-shaped batched pytrees.

    Args:
        new_tree: Candidate values.
        old_tree: Values to retain where ``take_new`` is ``False``.
        take_new: Boolean mask of shape ``(B,)``.

    Returns:
        A pytree with each start taken from one side or the other.
    """

    def pick(new: Any, old: Any) -> Array:
        new_value, old_value = jnp.asarray(new), jnp.asarray(old)
        mask = take_new.reshape((-1,) + (1,) * (new_value.ndim - 1))
        return jnp.where(mask, new_value, old_value)

    return jax.tree_util.tree_map(pick, new_tree, old_tree)


@dataclass
class OptimizationHistory:
    """Per-iteration record of a multi-start gain-optimization run.

    Attributes:
        loss: Per-iteration losses ``(B,)``.
        opt_vars: Batched optimization variables ``(B,)``.
        grad_norm: Per-start global gradient L2 norm ``(B,)``.
        update_norm: Per-start global update L2 norm ``(B,)``.
        time_iter: Wall-clock duration of each iteration in seconds.
        init_aux: Auxiliary rollout outputs at iteration zero ``B``.
        best_aux: Each start's own lowest-loss rollout ``B``.
        best_iteration: Each start's own lowest-loss iteration ``(B,)``.
        batch_size: Number of independently optimized starts.
        stopped_early: Whether the loop halted before ``num_iters``.
        stop_reason: Human-readable description of the early stop, or ``None``.
    """

    loss: list[Array] = field(default_factory=list)
    opt_vars: list[Any] = field(default_factory=list)
    finite_mask: list[Array] = field(default_factory=list)
    grad_norm: list[Array] = field(default_factory=list)
    update_norm: list[Array] = field(default_factory=list)
    time_iter: list[float] = field(default_factory=list)
    init_aux: dict[str, Array] | None = None
    best_aux: dict[str, Array] | None = None
    best_iteration: Array | None = None
    batch_size: int = 1
    stopped_early: bool = False
    stop_reason: str | None = None

    def __len__(self) -> int:
        """Return the number of recorded iterations."""
        return len(self.loss)

    def _require_recorded(self, action: str) -> None:
        """Raise if no iterate was recorded, reporting why the loop stopped.

        Args:
            action: What the caller requested.

        Raises:
            RuntimeError: If no finite iterate was recorded.
        """
        if not self.loss:
            raise RuntimeError(
                f"No finite optimization iterate was recorded; nothing to {action}. "
                f"Stop reason: {self.stop_reason}"
            )

    def loss_history(self) -> Array:
        """Return all losses as ``(iterations, B)``."""
        self._require_recorded("stack")
        return jnp.stack(self.loss, axis=0)

    def mask_history(self) -> Array:
        """Return the validity mask as ``(iterations, B)``."""
        self._require_recorded("stack")
        return jnp.stack(self.finite_mask, axis=0)

    def grad_norm_history(self) -> Array:
        """Return the gradient norms as ``(iterations, B)``."""
        self._require_recorded("stack")
        return jnp.stack(self.grad_norm, axis=0)

    def update_norm_history(self) -> Array:
        """Return the update norms as ``(iterations, B)``."""
        self._require_recorded("stack")
        return jnp.stack(self.update_norm, axis=0)

    def masked_loss_history(self) -> Array:
        """Return the losses with frozen entries replaced by ``inf``."""
        return jnp.where(self.mask_history(), self.loss_history(), jnp.inf)

    def dead_starts(self) -> list[int]:
        """Return starts that never produced a single finite iterate."""
        self._require_recorded("inspect")
        alive = jnp.any(self.mask_history(), axis=0)
        return [index for index, ok in enumerate(alive.tolist()) if not ok]

    def best_batch(self) -> int:
        """Return the start attaining the lowest loss over all iterations."""
        self._require_recorded("select")
        return int(jnp.argmin(jnp.min(self.masked_loss_history(), axis=0)))


def run_gain_optimization(
    gradient_fn: Callable[[Any], tuple[tuple[Array, dict[str, Array]], Any]],
    optimizer,
    opt_vars: Any,
    num_iters: int,
    batch_size: int = 1,
    progress_label: str = "Optimization",
) -> OptimizationHistory:
    """Run the batched Optax gain-optimization loop.

    Args:
        gradient_fn: Callable mapping optimization variables to ((loss, aux), gradient).
        optimizer: Initialized Optax optimizer.
        opt_vars: Initial optimization variables ``batch_size``.
        num_iters: Number of iterations to attempt. Must be at least one.
        batch_size: Number of independently optimized starts ``B``.
        progress_label: Prefix used in the progress log line.

    Returns:
        :class:`OptimizationHistory` whose entries are pairwise consistent.

    Raises:
        ValueError: If ``num_iters`` or ``batch_size`` is smaller than one.
    """
    if num_iters < 1:
        raise ValueError("num_iters must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    import optax  # Imported lazily

    batched_gradient_fn = jax.jit(jax.vmap(gradient_fn))
    opt_state = jax.vmap(optimizer.init)(opt_vars)
    history = OptimizationHistory(batch_size=batch_size)

    alive = jnp.ones((batch_size,), dtype=bool)
    best_loss = jnp.full((batch_size,), jnp.inf)
    best_iteration = jnp.zeros((batch_size,), dtype=jnp.int64)
    best_aux: dict[str, Array] | None = None
    last_loss = jnp.zeros((batch_size,))

    for iteration in range(num_iters):
        iter_start = time.time()

        opt_vars_evaluated = opt_vars
        try:
            (loss, aux), grad = batched_gradient_fn(opt_vars)
            jax.block_until_ready((loss, aux, grad))
        except KeyboardInterrupt:
            history.stopped_early = True
            history.stop_reason = (
                f"interrupted during iteration {iteration + 1}; preserving "
                f"{len(history)} completed iterations"
            )
            print(f"\n[WARNING] {history.stop_reason}.")
            break

        evaluation_ok = _tree_batched_finite(
            {"loss": loss, "aux": aux, "gradient": grad}, batch_size
        )
        alive = alive & evaluation_ok

        loss = jnp.where(alive, loss, last_loss)
        last_loss = loss

        if iteration == 0:
            history.init_aux = aux
            best_aux = aux

        improved = alive & (loss < best_loss)
        best_loss = jnp.where(improved, loss, best_loss)
        best_iteration = jnp.where(improved, iteration, best_iteration)
        best_aux = _tree_select(aux, best_aux, improved)

        iteration_grad_norm = _tree_global_norm(grad, batch_size)
        iteration_update_norm = jnp.zeros((batch_size,))

        if iteration < num_iters - 1 and bool(jnp.any(alive)):
            updates, opt_state_next = jax.vmap(optimizer.update)(
                grad, opt_state, opt_vars_evaluated
            )
            opt_vars_next = optax.apply_updates(opt_vars_evaluated, updates)
            jax.block_until_ready((updates, opt_state_next, opt_vars_next))
            update_ok = _tree_batched_finite(
                {
                    "updates": updates,
                    "opt_state_next": opt_state_next,
                    "opt_vars_next": opt_vars_next,
                },
                batch_size,
            )
            iteration_update_norm = _tree_global_norm(updates, batch_size)
            advancing = alive & update_ok
            opt_state = _tree_select(opt_state_next, opt_state, advancing)
            opt_vars = _tree_select(opt_vars_next, opt_vars_evaluated, advancing)
            alive = advancing

        history.loss.append(loss)
        history.opt_vars.append(opt_vars_evaluated)
        history.finite_mask.append(evaluation_ok & jnp.isfinite(loss))
        history.grad_norm.append(iteration_grad_norm)
        history.update_norm.append(iteration_update_norm)
        history.time_iter.append(time.time() - iter_start)

        frozen = int(batch_size - int(jnp.sum(alive)))
        if not bool(jnp.any(alive)) and iteration < num_iters - 1:
            history.stopped_early = True
            history.stop_reason = (
                f"all {batch_size} starts became non-finite by iteration "
                f"{iteration + 1}; recorded {len(history)} iterations"
            )
            print(f"\n[WARNING] {history.stop_reason}.")
            break

        if iteration == 0:
            print(
                f"\n[INFO] Compilation + first execution time = "
                f"{history.time_iter[0]:.2f} s"
            )
            continue

        steady_times = history.time_iter[1:]
        t_left = (num_iters - iteration - 1) * sum(steady_times) / len(steady_times)
        eta = time.localtime(time.time() + t_left)
        print(
            f"{progress_label}: {100 * (iteration + 1) / num_iters:3.1f} %  |  "
            f"iteration {iteration + 1:>4d} of {num_iters:<4d}  |  "
            f"iter time = {history.time_iter[-1]:>.2f} s  |  "
            f"frozen {frozen}/{batch_size}  |  "
            f"ETA = {eta.tm_mday:02d}/{eta.tm_mon:02d}/{eta.tm_year} "
            f"{eta.tm_hour:02d}:{eta.tm_min:02d}",
            end="\r",
        )

    history.best_aux = best_aux
    history.best_iteration = best_iteration
    print(
        f"\n{progress_label}: recorded {len(history)} of {num_iters} iterations"
        + ("  (stopped early)" if history.stopped_early else "")
    )
    return history
