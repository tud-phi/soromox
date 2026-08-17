"""Regression tests for the Section Vd gain-optimization loop (issue #129).

Two defects are pinned here:

* the loss/parameter histories were offset by one step, because the loop stored
  the parameters produced by the update next to the loss measured *before* it;
* non-finite candidates were saved, so best-candidate selection could return
  ``NaN`` gains.

Both are checked with a cheap synthetic loss so the tests do not integrate a
robot, and both would pass silently against a value-only assertion -- the point
is that the histories must be pairwise consistent.
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

    The auxiliary output mimics the rollout dictionary the real generators
    produce, so the loop is exercised through the same code path.
    """

    def gradient_fn(opt_vars):
        x = opt_vars["x"]
        loss = jnp.sum((x - target) ** 2)
        grad = {"x": 2.0 * (x - target)}
        aux = {"q_ts": jnp.full((4,), x[0]), "t_ts": jnp.arange(4.0)}
        return (loss, aux), grad

    return gradient_fn


def test_each_loss_is_paired_with_the_parameters_it_was_measured_at():
    """The core of issue #129: histories must not be offset by one step."""
    gradient_fn = _quadratic_gradient_fn()
    optimizer = optax.sgd(learning_rate=0.1)
    opt_vars = {"x": jnp.array([0.0])}

    history = run_gain_optimization(
        gradient_fn=gradient_fn,
        optimizer=optimizer,
        opt_vars=opt_vars,
        num_iters=5,
    )

    assert len(history) == 5
    for stored_vars, stored_loss in zip(history.opt_vars, history.loss):
        (recomputed_loss, _), _ = gradient_fn(stored_vars)
        assert_allclose(float(stored_loss), float(recomputed_loss), rtol=1e-12)


def test_first_recorded_parameters_are_the_initial_ones():
    """Iteration 0 is evaluated at the initial parameters, before any update."""
    opt_vars = {"x": jnp.array([0.0])}

    history = run_gain_optimization(
        gradient_fn=_quadratic_gradient_fn(),
        optimizer=optax.sgd(learning_rate=0.1),
        opt_vars=opt_vars,
        num_iters=3,
    )

    assert_allclose(history.opt_vars[0]["x"], opt_vars["x"], atol=1e-15)


def test_best_index_selects_the_parameters_that_attained_the_best_loss():
    history = run_gain_optimization(
        gradient_fn=_quadratic_gradient_fn(),
        optimizer=optax.sgd(learning_rate=0.1),
        opt_vars={"x": jnp.array([0.0])},
        num_iters=6,
    )

    best = history.best_index()
    gradient_fn = _quadratic_gradient_fn()
    (best_loss, _), _ = gradient_fn(history.opt_vars[best])

    assert_allclose(float(best_loss), float(min(history.loss)), rtol=1e-12)
    # A descending quadratic reaches its minimum on the last recorded iterate.
    assert best == len(history) - 1


def test_auxiliary_history_matches_the_recorded_parameters():
    history = run_gain_optimization(
        gradient_fn=_quadratic_gradient_fn(),
        optimizer=optax.sgd(learning_rate=0.1),
        opt_vars={"x": jnp.array([0.0])},
        num_iters=4,
    )

    q_ts_tot = history.stacked_aux("q_ts")

    assert q_ts_tot.shape == (4, 4)
    for index, stored_vars in enumerate(history.opt_vars):
        assert_allclose(q_ts_tot[index, 0], stored_vars["x"][0], atol=1e-15)


@pytest.mark.parametrize("bad_value", [jnp.nan, jnp.inf])
def test_non_finite_candidate_is_rejected_rather_than_saved(bad_value):
    """The second half of issue #129: nothing non-finite reaches the history."""
    fail_at = 2

    def gradient_fn(opt_vars):
        x = opt_vars["x"]
        loss = jnp.sum((x - 3.0) ** 2)
        grad = {"x": 2.0 * (x - 3.0)}
        # Poison the loss once the parameter has moved past a threshold.
        loss = jnp.where(x[0] > 0.1 * fail_at, bad_value, loss)
        aux = {"q_ts": jnp.full((4,), x[0]), "t_ts": jnp.arange(4.0)}
        return (loss, aux), grad

    history = run_gain_optimization(
        gradient_fn=gradient_fn,
        optimizer=optax.sgd(learning_rate=0.6),
        opt_vars={"x": jnp.array([0.0])},
        num_iters=8,
    )

    assert history.stopped_early
    assert history.stop_reason is not None
    assert "loss" in history.stop_reason
    assert len(history) < 8
    assert jnp.isfinite(jnp.asarray(history.loss)).all()
    for stored_vars in history.opt_vars:
        assert jnp.isfinite(stored_vars["x"]).all()


def test_non_finite_gradient_is_detected_even_when_the_forward_pass_is_clean():
    """The failure mode the original report described: forward fine, reverse NaN."""

    def gradient_fn(opt_vars):
        x = opt_vars["x"]
        loss = jnp.sum((x - 3.0) ** 2)  # always finite
        grad = {"x": jnp.where(x[0] > 0.5, jnp.nan, 2.0 * (x - 3.0))}
        aux = {"q_ts": jnp.full((4,), x[0]), "t_ts": jnp.arange(4.0)}
        return (loss, aux), grad

    history = run_gain_optimization(
        gradient_fn=gradient_fn,
        optimizer=optax.sgd(learning_rate=0.5),
        opt_vars={"x": jnp.array([0.0])},
        num_iters=6,
    )

    assert history.stopped_early
    assert "gradient" in history.stop_reason


def test_accessors_report_the_stop_reason_when_nothing_was_recorded():
    """An empty history must explain itself rather than fail obscurely.

    ``stacked_aux`` is reached before ``best_index`` in both generators, so an
    unguarded ``jnp.stack`` there would bury the diagnostic the loop just
    printed under a bare "Need at least one array to stack".
    """

    def gradient_fn(opt_vars):
        aux = {"q_ts": jnp.full((4,), jnp.nan), "t_ts": jnp.arange(4.0)}
        return (jnp.array(jnp.nan), aux), {"x": jnp.zeros(1)}

    history = run_gain_optimization(
        gradient_fn=gradient_fn,
        optimizer=optax.sgd(learning_rate=0.1),
        opt_vars={"x": jnp.array([0.0])},
        num_iters=3,
    )

    assert len(history) == 0
    for access in (history.best_index, lambda: history.stacked_aux("t_ts")):
        with pytest.raises(RuntimeError, match="No finite optimization iterate"):
            access()


def test_finite_iterate_is_recorded_before_a_non_finite_update_stops_the_loop():
    """A failed update must not discard the finite baseline it was based on."""

    def gradient_fn(opt_vars):
        x = opt_vars["x"]
        loss = jnp.sum(x**2)
        aux = {"q_ts": jnp.full((4,), x[0]), "t_ts": jnp.arange(4.0)}
        return (loss, aux), {"x": jnp.full_like(x, 1e308)}

    initial_vars = {"x": jnp.array([1.0], dtype=jnp.float64)}
    history = run_gain_optimization(
        gradient_fn=gradient_fn,
        optimizer=optax.sgd(learning_rate=2.0),
        opt_vars=initial_vars,
        num_iters=3,
    )

    assert len(history) == 1
    assert history.stopped_early
    assert history.stop_reason is not None
    assert "next" in history.stop_reason
    assert_allclose(history.opt_vars[0]["x"], initial_vars["x"], atol=0.0)
    assert_allclose(history.loss[0], 1.0, atol=0.0)
    assert history.best_index() == 0


def test_final_evaluation_does_not_compute_an_unused_non_finite_update():
    def gradient_fn(opt_vars):
        x = opt_vars["x"]
        aux = {"q_ts": jnp.full((4,), x[0]), "t_ts": jnp.arange(4.0)}
        return (jnp.sum(x**2), aux), {"x": jnp.full_like(x, 1e308)}

    history = run_gain_optimization(
        gradient_fn=gradient_fn,
        optimizer=optax.sgd(learning_rate=2.0),
        opt_vars={"x": jnp.array([1.0], dtype=jnp.float64)},
        num_iters=1,
    )

    assert len(history) == 1
    assert not history.stopped_early
    assert history.stop_reason is None


def test_keyboard_interrupt_preserves_completed_iterations():
    """Ctrl-C should return the finite prefix so generators can save it."""
    calls = 0
    finite_gradient_fn = _quadratic_gradient_fn()

    def interrupted_gradient_fn(opt_vars):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt
        return finite_gradient_fn(opt_vars)

    history = run_gain_optimization(
        gradient_fn=interrupted_gradient_fn,
        optimizer=optax.sgd(learning_rate=0.1),
        opt_vars={"x": jnp.array([0.0])},
        num_iters=8,
    )

    assert len(history) == 2
    assert history.stopped_early
    assert history.stop_reason == (
        "interrupted during iteration 3; preserving 2 completed iterations"
    )
    assert jnp.isfinite(jnp.asarray(history.loss)).all()


def test_keyboard_interrupt_before_first_evaluation_returns_empty_history():
    def interrupted_gradient_fn(_opt_vars):
        raise KeyboardInterrupt

    history = run_gain_optimization(
        gradient_fn=interrupted_gradient_fn,
        optimizer=optax.sgd(learning_rate=0.1),
        opt_vars={"x": jnp.array([0.0])},
        num_iters=3,
    )

    assert len(history) == 0
    assert history.stopped_early
    assert history.stop_reason == (
        "interrupted during iteration 1; preserving 0 completed iterations"
    )


def test_num_iters_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        run_gain_optimization(
            gradient_fn=_quadratic_gradient_fn(),
            optimizer=optax.sgd(learning_rate=0.1),
            opt_vars={"x": jnp.array([0.0])},
            num_iters=0,
        )
