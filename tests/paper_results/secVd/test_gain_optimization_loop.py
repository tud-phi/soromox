"""Regression tests for the Section Vd multi-start gain-optimization loop.

Three properties are pinned here:

* the loss/parameter histories must not be offset by one step, because the loop
  once stored the parameters produced by the update next to the loss measured
  *before* it (issue #129);
* a start that goes non-finite must be frozen rather than recorded or allowed to
  abort the whole run, and the mask must say which entries are real;
* the starts must be genuinely independent -- the published Section Vd results
  were six independent optimizers, not one optimizer on an averaged gradient
  (issue #154).

All checks use a cheap synthetic loss so the tests do not integrate a robot.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pytest
from numpy.testing import assert_allclose

SECTION_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secVd_control_gain_optimization"
)
MODULE_DIR = SECTION_DIR / "code"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from gain_optimization_loop import run_gain_optimization  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _quadratic_gradient_fn(target: float = 3.0):
    """Return a ``value_and_grad``-shaped callable for ``(x - target)**2``.

    It takes **one start's** variables; the loop vmaps it. The auxiliary output
    mimics the rollout dictionary the real generators produce, so the loop is
    exercised through the same code path.
    """

    def gradient_fn(opt_vars):
        x = opt_vars["x"]
        loss = jnp.sum((x - target) ** 2)
        grad = {"x": 2.0 * (x - target)}
        aux = {"q_ts": jnp.full((4,), x[0]), "t_ts": jnp.arange(4.0)}
        return (loss, aux), grad

    return gradient_fn


def _batched_start(values):
    """Build ``opt_vars`` for starts whose single coordinate takes ``values``."""
    return {"x": jnp.asarray(values, dtype=jnp.float64)[:, None]}


def test_each_loss_is_paired_with_the_parameters_it_was_measured_at():
    """The core of issue #129: histories must not be offset by one step."""
    gradient_fn = _quadratic_gradient_fn()
    history = run_gain_optimization(
        gradient_fn=gradient_fn,
        optimizer=optax.sgd(learning_rate=0.1),
        opt_vars=_batched_start([0.0, 1.0]),
        num_iters=5,
        batch_size=2,
    )
    assert len(history) == 5
    for stored_vars, stored_loss in zip(history.opt_vars, history.loss):
        (recomputed, _), _ = jax.vmap(gradient_fn)(stored_vars)
        assert_allclose(recomputed, stored_loss, rtol=1e-12)


def test_the_first_entry_is_the_untouched_initialization():
    """Index 0 must be the initialization, not one optimizer step past it."""
    opt_vars = _batched_start([0.0, 1.0])
    history = run_gain_optimization(
        gradient_fn=_quadratic_gradient_fn(),
        optimizer=optax.sgd(learning_rate=0.1),
        opt_vars=opt_vars,
        num_iters=3,
        batch_size=2,
    )
    assert_allclose(history.opt_vars[0]["x"], opt_vars["x"], atol=1e-15)


def test_starts_are_optimized_independently():
    """Six independent optimizers, not one on an averaged gradient (#154)."""
    history = run_gain_optimization(
        gradient_fn=_quadratic_gradient_fn(target=3.0),
        optimizer=optax.sgd(learning_rate=0.1),
        opt_vars=_batched_start([0.0, 6.0]),
        num_iters=4,
        batch_size=2,
    )
    # Symmetric starts about the target must stay symmetric; an averaged
    # gradient would collapse them onto a common trajectory.
    trace = jnp.stack([entry["x"][:, 0] for entry in history.opt_vars])
    assert_allclose(trace[:, 0] - 3.0, -(trace[:, 1] - 3.0), rtol=1e-12)
    assert float(jnp.abs(trace[-1, 0] - trace[-1, 1])) > 1e-6


def test_best_batch_selects_the_start_that_attained_the_lowest_loss():
    history = run_gain_optimization(
        gradient_fn=_quadratic_gradient_fn(target=3.0),
        optimizer=optax.sgd(learning_rate=0.1),
        opt_vars=_batched_start([0.0, 2.9]),
        num_iters=4,
        batch_size=2,
    )
    # Start 1 begins far closer to the target and never gives that up.
    assert history.best_batch() == 1
    losses = history.loss_history()
    assert float(jnp.min(losses[:, 1])) < float(jnp.min(losses[:, 0]))


def test_best_iteration_is_recorded_per_start():
    history = run_gain_optimization(
        gradient_fn=_quadratic_gradient_fn(),
        optimizer=optax.sgd(learning_rate=0.1),
        opt_vars=_batched_start([0.0, 1.0]),
        num_iters=4,
        batch_size=2,
    )
    assert history.best_iteration.shape == (2,)
    expected = jnp.argmin(history.loss_history(), axis=0)
    assert jnp.array_equal(history.best_iteration, expected)


def test_a_single_start_run_is_the_same_code_path():
    """B = 1 must remain a strict special case."""
    history = run_gain_optimization(
        gradient_fn=_quadratic_gradient_fn(),
        optimizer=optax.sgd(learning_rate=0.1),
        opt_vars=_batched_start([0.0]),
        num_iters=3,
        batch_size=1,
    )
    assert history.loss_history().shape == (3, 1)
    assert history.best_batch() == 0
    assert history.dead_starts() == []


def test_a_nonfinite_start_is_frozen_without_stopping_the_others():
    """One bad start must not discard five good ones."""
    target = 3.0

    def gradient_fn(opt_vars):
        x = opt_vars["x"]
        # A start that wanders above this threshold produces NaN.
        loss = jnp.where(x[0] > 10.0, jnp.nan, jnp.sum((x - target) ** 2))
        grad = {"x": 2.0 * (x - target)}
        aux = {"q_ts": jnp.full((4,), x[0]), "t_ts": jnp.arange(4.0)}
        return (loss, aux), grad

    history = run_gain_optimization(
        gradient_fn=gradient_fn,
        optimizer=optax.sgd(learning_rate=0.1),
        opt_vars=_batched_start([0.0, 50.0]),
        num_iters=4,
        batch_size=2,
    )
    assert len(history) == 4
    mask = history.mask_history()
    # Start 0 stays valid throughout; start 1 is invalid from its first look.
    assert bool(jnp.all(mask[:, 0]))
    assert not bool(jnp.any(mask[:, 1]))
    assert history.dead_starts() == [1]
    # The healthy start's own history is untouched and finite.
    assert bool(jnp.all(jnp.isfinite(history.loss_history()[:, 0])))
    assert history.best_batch() == 0


def test_a_start_that_goes_bad_midway_keeps_its_earlier_iterates():
    """Freezing must preserve what a start already achieved."""
    target = 3.0

    def gradient_fn(opt_vars):
        x = opt_vars["x"]
        loss = jnp.sum((x - target) ** 2)
        # A band only start 1 enters, and only after one good evaluation:
        # under sgd(0.1) it steps 5.0 -> 4.6, while start 0 runs 0.0 -> 0.6 -> 1.08.
        loss = jnp.where((x[0] > 4.4) & (x[0] < 4.8), jnp.nan, loss)
        grad = {"x": 2.0 * (x - target)}
        aux = {"q_ts": jnp.full((4,), x[0]), "t_ts": jnp.arange(4.0)}
        return (loss, aux), grad

    history = run_gain_optimization(
        gradient_fn=gradient_fn,
        optimizer=optax.sgd(learning_rate=0.1),
        opt_vars=_batched_start([0.0, 5.0]),
        num_iters=3,
        batch_size=2,
    )
    mask = history.mask_history()
    assert bool(jnp.all(mask[:, 0]))
    # Start 1 banked iteration 0 before going bad, and keeps it.
    assert bool(mask[0, 1])
    assert not bool(jnp.any(mask[1:, 1]))
    assert history.dead_starts() == []
    assert int(history.best_iteration[1]) == 0
    assert bool(jnp.isfinite(history.loss_history()[0, 1]))


def test_the_loop_stops_when_every_start_is_dead():
    def gradient_fn(opt_vars):
        x = opt_vars["x"]
        aux = {"q_ts": jnp.full((4,), x[0]), "t_ts": jnp.arange(4.0)}
        return (jnp.nan * jnp.sum(x), aux), {"x": jnp.zeros_like(x)}

    history = run_gain_optimization(
        gradient_fn=gradient_fn,
        optimizer=optax.sgd(learning_rate=0.1),
        opt_vars=_batched_start([0.0, 1.0]),
        num_iters=6,
        batch_size=2,
    )
    assert history.stopped_early
    assert "non-finite" in history.stop_reason
    assert history.dead_starts() == [0, 1]
    assert len(history) < 6


def test_a_nonfinite_gradient_freezes_a_start_even_when_the_loss_is_clean():
    """The reverse pass can fail from a perfectly healthy rollout."""

    def gradient_fn(opt_vars):
        x = opt_vars["x"]
        loss = jnp.sum(x**2)
        grad = {"x": jnp.where(x[0] > 0.5, jnp.nan, 2.0 * x)}
        aux = {"q_ts": jnp.full((4,), x[0]), "t_ts": jnp.arange(4.0)}
        return (loss, aux), grad

    history = run_gain_optimization(
        gradient_fn=gradient_fn,
        optimizer=optax.sgd(learning_rate=0.1),
        opt_vars=_batched_start([0.1, 1.0]),
        num_iters=3,
        batch_size=2,
    )
    mask = history.mask_history()
    assert bool(jnp.all(mask[:, 0]))
    assert not bool(jnp.any(mask[:, 1]))


def test_init_and_best_rollouts_are_kept_but_not_every_iteration():
    """Per-iteration trajectory histories would run to gigabytes at B=6."""
    history = run_gain_optimization(
        gradient_fn=_quadratic_gradient_fn(),
        optimizer=optax.sgd(learning_rate=0.1),
        opt_vars=_batched_start([0.0, 1.0]),
        num_iters=4,
        batch_size=2,
    )
    assert history.init_aux["q_ts"].shape == (2, 4)
    assert history.best_aux["q_ts"].shape == (2, 4)
    # The initial rollout is the initialization's, not the first update's.
    assert_allclose(history.init_aux["q_ts"][:, 0], [0.0, 1.0], atol=1e-15)
    assert not hasattr(history, "stacked_aux")


def test_accessors_report_the_stop_reason_when_nothing_was_recorded():
    history = run_gain_optimization(
        gradient_fn=_quadratic_gradient_fn(),
        optimizer=optax.sgd(learning_rate=0.1),
        opt_vars=_batched_start([0.0]),
        num_iters=1,
        batch_size=1,
    )
    history.loss.clear()
    for access in (history.best_batch, history.loss_history, history.dead_starts):
        with pytest.raises(RuntimeError, match="No finite optimization iterate"):
            access()


@pytest.mark.parametrize(("num_iters", "batch_size"), [(0, 1), (1, 0)])
def test_degenerate_requests_are_rejected(num_iters, batch_size):
    with pytest.raises(ValueError):
        run_gain_optimization(
            gradient_fn=_quadratic_gradient_fn(),
            optimizer=optax.sgd(learning_rate=0.1),
            opt_vars=_batched_start([0.0]),
            num_iters=num_iters,
            batch_size=batch_size,
        )
