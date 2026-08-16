"""Tests for the reverse-mode-safe numerical primitives.

Each helper is checked twice: it must agree with the naive expression away from
the singular point, and it must stay finite at it. Transforms are only re-tested
where they change the lowering -- ``grad`` of ``vmap`` rewrites ``lax.cond`` into
``select_n`` -- not for their own sake.
"""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from soromox.autodiff import (
    set_strict_singularities,
    strict_singularities_enabled,
    strict_singularities_mode,
)
from soromox.utils import (
    eps_for_dtype,
    safe_divide,
    safe_norm,
    safe_normalize,
    safe_sqrt,
)

jax.config.update("jax_enable_x64", True)

ATOL = 1e-12


def test_eps_for_dtype_uses_dtype_machine_epsilon_when_none():
    eps = eps_for_dtype(None, jnp.float64)

    assert eps.shape == ()
    assert eps.dtype == jnp.float64
    assert_allclose(eps, jnp.finfo(jnp.float64).eps, rtol=0.0, atol=0.0)


def test_eps_for_dtype_casts_explicit_eps_to_requested_dtype():
    eps = eps_for_dtype(1e-6, jnp.float32)

    assert eps.shape == ()
    assert eps.dtype == jnp.float32
    assert_allclose(eps, jnp.asarray(1e-6, dtype=jnp.float32), rtol=0.0, atol=0.0)


# (label, callable, singular input, regular input) for the behaviour shared by
# every helper: finite at the singularity, unchanged away from it.
SAFE_CASES = [
    ("safe_sqrt", safe_sqrt, jnp.array(0.0), jnp.array(4.0)),
    ("safe_norm", safe_norm, jnp.zeros(3), jnp.array([3.0, 4.0, 0.0])),
    (
        "safe_normalize",
        lambda v: jnp.sum(safe_normalize(v)),
        jnp.zeros(3),
        jnp.array([3.0, 4.0, 0.0]),
    ),
    (
        "safe_divide",
        lambda d: safe_divide(jnp.array(1.0), d, 1e-12),
        jnp.array(0.0),
        jnp.array(3.0),
    ),
]
SAFE_IDS = [case[0] for case in SAFE_CASES]


@pytest.mark.parametrize("_, fn, singular, __", SAFE_CASES, ids=SAFE_IDS)
def test_gradient_is_finite_at_the_singular_point(_, fn, singular, __):
    assert jnp.isfinite(jax.grad(fn)(singular)).all()


class TestSafeSqrt:
    def test_matches_sqrt_away_from_zero(self):
        x = jnp.array([0.25, 1.0, 4.0, 1e6])
        assert_allclose(safe_sqrt(x), jnp.sqrt(x), atol=ATOL)
        assert_allclose(
            jax.grad(lambda v: jnp.sum(safe_sqrt(v)))(x),
            jax.grad(lambda v: jnp.sum(jnp.sqrt(v)))(x),
            atol=ATOL,
        )

    def test_clamps_negative_input_to_zero(self):
        assert float(safe_sqrt(jnp.array(-3.0))) == 0.0

    def test_zero_slope_at_zero_where_plain_sqrt_is_infinite(self):
        assert float(jax.grad(safe_sqrt)(jnp.array(0.0))) == 0.0
        assert not jnp.isfinite(jax.grad(jnp.sqrt)(jnp.array(0.0)))


class TestSafeNorm:
    def test_matches_linalg_norm_away_from_zero(self):
        x = jnp.array([3.0, 4.0, 0.0])
        assert_allclose(safe_norm(x), jnp.linalg.norm(x), atol=ATOL)
        assert_allclose(jax.grad(safe_norm)(x), jax.grad(jnp.linalg.norm)(x), atol=ATOL)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_reduces_over_axis_and_can_keep_dims(self, axis: int):
        x = jnp.array([[3.0, 4.0], [5.0, 12.0], [1.0, 1.0]])
        assert_allclose(safe_norm(x, axis), jnp.linalg.norm(x, axis=axis), atol=ATOL)
        assert safe_norm(x, axis, True).ndim == x.ndim

    def test_zero_gradient_at_the_zero_vector_where_plain_norm_is_nan(self):
        grad = jax.grad(safe_norm)(jnp.zeros(3))
        assert_allclose(grad, jnp.zeros(3), atol=ATOL)
        assert jnp.isnan(jax.grad(jnp.linalg.norm)(jnp.zeros(3))).any()

    def test_gradient_is_finite_for_a_zero_row(self):
        x = jnp.array([[3.0, 4.0], [0.0, 0.0]])
        grad = jax.grad(lambda v: jnp.sum(safe_norm(v, 1)))(x)
        assert jnp.isfinite(grad).all()

    def test_finite_under_grad_of_vmap_at_the_zero_vector(self):
        """``vmap`` is the transform that changes the lowering, so it is re-tested."""
        grad = jax.grad(lambda v: jnp.sum(jax.vmap(safe_norm)(v)))(jnp.zeros((4, 3)))
        assert jnp.isfinite(grad).all()


class TestSquaredNormIdentity:
    """``norm(d, axis=k)**2`` equals ``sum(d**2, axis=k)`` but differentiates better.

    This is the substitution applied to the Section Vd losses and to the
    ISupport parallel-axis term, so both halves of the claim are pinned here.
    """

    def test_values_and_gradients_agree_away_from_zero(self):
        d = jnp.array([[1.0, 2.0], [3.0, 4.0], [0.5, -1.5]])
        norm_form = lambda v: jnp.sum(jnp.linalg.norm(v, axis=1) ** 2)  # noqa: E731
        ssq_form = lambda v: jnp.sum(jnp.sum(v**2, axis=1))  # noqa: E731

        assert_allclose(norm_form(d), ssq_form(d), rtol=1e-15)
        assert_allclose(jax.grad(norm_form)(d), jax.grad(ssq_form)(d), atol=ATOL)

    def test_sum_of_squares_survives_an_exactly_zero_row(self):
        d = jnp.array([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]])
        norm_form = lambda v: jnp.sum(jnp.linalg.norm(v, axis=1) ** 2)  # noqa: E731
        ssq_form = lambda v: jnp.sum(jnp.sum(v**2, axis=1))  # noqa: E731

        assert jnp.isnan(jax.grad(norm_form)(d)).any()
        assert jnp.isfinite(jax.grad(ssq_form)(d)).all()


class TestSafeDivide:
    def test_matches_division_away_from_zero(self):
        num = jnp.array([6.0, -3.0])
        den = jnp.array([3.0, 1.5])
        assert_allclose(safe_divide(num, den, 1e-12), num / den, atol=ATOL)

    def test_returns_fallback_at_a_vanishing_denominator(self):
        result = safe_divide(jnp.array(1.0), jnp.array(0.0), 1e-12, fallback=7.0)
        assert float(result) == 7.0

    def test_mixed_array_denominators_have_finite_reverse_mode_jacobians(self):
        """The sanitized denominator makes both elementwise ``where`` arms safe."""
        numerator = jnp.array([2.0, 4.0, 6.0])
        denominator = jnp.array([2.0, 0.0, -3.0])

        jac_num, jac_den = jax.jacrev(safe_divide, argnums=(0, 1))(
            numerator, denominator, 1e-12
        )

        assert jnp.isfinite(jac_num).all()
        assert jnp.isfinite(jac_den).all()

    def test_guarding_the_quotient_instead_would_produce_nan(self):
        """Document why the denominator, not the output, must be sanitized."""
        broken = lambda d: jnp.where(d > 0, 1.0 / d, 0.0)  # noqa: E731
        assert jnp.isnan(jax.grad(broken)(jnp.array(0.0))).any()


class TestSafeNormalize:
    def test_matches_naive_normalization_away_from_zero(self):
        x = jnp.array([[3.0, 4.0], [5.0, 12.0]])
        expected = x / jnp.linalg.norm(x, axis=1, keepdims=True)
        assert_allclose(safe_normalize(x, axis=1), expected, atol=ATOL)
        assert_allclose(safe_normalize(x[0]), x[0] / jnp.linalg.norm(x[0]), atol=ATOL)

    def test_zero_vector_maps_to_the_fallback(self):
        assert_allclose(safe_normalize(jnp.zeros(3)), jnp.zeros(3), atol=ATOL)

    def test_gradient_is_finite_for_a_zero_row(self):
        x = jnp.array([[3.0, 4.0], [0.0, 0.0]])
        grad = jax.grad(lambda v: jnp.sum(safe_normalize(v, axis=1)))(x)
        assert jnp.isfinite(grad).all()


class TestStrictSingularitiesMode:
    """Opt-in mode that lets singular inputs propagate instead of falling back.

    The fallbacks keep production runs finite but hide *where* a model reaches a
    singular configuration; strict mode trades that back for a diagnostic.
    """

    @pytest.mark.parametrize("_, fn, singular, __", SAFE_CASES, ids=SAFE_IDS)
    def test_singular_gradient_propagates_when_strict(self, _, fn, singular, __):
        with strict_singularities_mode():
            assert not jnp.isfinite(jax.grad(fn)(singular)).all()

    @pytest.mark.parametrize("_, fn, __, regular", SAFE_CASES, ids=SAFE_IDS)
    def test_regular_points_are_unaffected(self, _, fn, __, regular):
        value, grad = fn(regular), jax.grad(fn)(regular)

        with strict_singularities_mode():
            assert_allclose(fn(regular), value, atol=ATOL)
            assert_allclose(jax.grad(fn)(regular), grad, atol=ATOL)

    def test_flag_is_off_by_default_and_always_restored(self):
        assert not strict_singularities_enabled()

        with strict_singularities_mode():
            assert strict_singularities_enabled()
        assert not strict_singularities_enabled()

        with pytest.raises(RuntimeError), strict_singularities_mode():
            raise RuntimeError("boom")
        assert not strict_singularities_enabled()

        try:
            set_strict_singularities(True)
            assert strict_singularities_enabled()
        finally:
            set_strict_singularities(False)
