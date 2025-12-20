import jax

jax.config.update("jax_enable_x64", True)  # double precision

import jax.numpy as jnp
import pytest
from jax import random

from soromox.control import ReferenceTrajectory


class TestReferenceTrajectoryFromDiscrete:
    """Tests for ReferenceTrajectory initialized with discrete data."""

    def test_discrete_to_continuous_conversion(self):
        """Test that x_des_fn is created from x_des_ts."""
        ts = jnp.linspace(0, 1, 10)
        x_des_ts = jnp.column_stack([jnp.sin(ts), jnp.cos(ts)])

        ref = ReferenceTrajectory(ts=ts, x_des_ts=x_des_ts)

        assert ref.x_des_fn is not None
        assert ref.x_des_ts.shape == (10, 2)

    def test_interpolation_at_known_points(self):
        """Test that interpolation returns exact values at known time points."""
        ts = jnp.linspace(0, 1, 10)
        x_des_ts = jnp.column_stack([jnp.sin(ts), jnp.cos(ts)])

        ref = ReferenceTrajectory(ts=ts, x_des_ts=x_des_ts)

        # Check at each known time point
        for i, t in enumerate(ts):
            x_interp = ref.x_des_fn(t)
            assert jnp.allclose(x_interp, x_des_ts[i], atol=1e-10)

    def test_interpolation_between_points(self):
        """Test linear interpolation between known points."""
        ts = jnp.array([0.0, 1.0])
        x_des_ts = jnp.array([[0.0, 0.0], [2.0, 4.0]])

        ref = ReferenceTrajectory(ts=ts, x_des_ts=x_des_ts)

        # At midpoint, should be linear interpolation
        x_mid = ref.x_des_fn(0.5)
        assert jnp.allclose(x_mid, jnp.array([1.0, 2.0]), atol=1e-10)

        # At quarter point
        x_quarter = ref.x_des_fn(0.25)
        assert jnp.allclose(x_quarter, jnp.array([0.5, 1.0]), atol=1e-10)

    def test_boundary_clamping_before_start(self):
        """Test that values before ts[0] clamp to first value."""
        ts = jnp.linspace(0, 1, 10)
        x_des_ts = jnp.column_stack([jnp.sin(ts), jnp.cos(ts)])

        ref = ReferenceTrajectory(ts=ts, x_des_ts=x_des_ts)

        x_before = ref.x_des_fn(-0.5)
        assert jnp.allclose(x_before, x_des_ts[0], atol=1e-10)

    def test_boundary_clamping_after_end(self):
        """Test that values after ts[-1] clamp to last value."""
        ts = jnp.linspace(0, 1, 10)
        x_des_ts = jnp.column_stack([jnp.sin(ts), jnp.cos(ts)])

        ref = ReferenceTrajectory(ts=ts, x_des_ts=x_des_ts)

        x_after = ref.x_des_fn(1.5)
        assert jnp.allclose(x_after, x_des_ts[-1], atol=1e-10)

    def test_velocity_auto_derived(self):
        """Test that velocity is auto-derived from discrete position."""
        ts = jnp.linspace(0, 1, 10)
        x_des_ts = jnp.column_stack([jnp.sin(ts), jnp.cos(ts)])

        ref = ReferenceTrajectory(ts=ts, x_des_ts=x_des_ts)

        assert ref.xd_des_fn is not None
        assert ref.xd_des_ts is not None
        assert ref.xd_des_ts.shape == (10, 2)

    def test_acceleration_auto_derived(self):
        """Test that acceleration is auto-derived from velocity."""
        ts = jnp.linspace(0, 1, 10)
        x_des_ts = jnp.column_stack([jnp.sin(ts), jnp.cos(ts)])

        ref = ReferenceTrajectory(ts=ts, x_des_ts=x_des_ts)

        assert ref.xdd_des_fn is not None
        assert ref.xdd_des_ts is not None
        assert ref.xdd_des_ts.shape == (10, 2)


class TestReferenceTrajectoryFromContinuous:
    """Tests for ReferenceTrajectory initialized with continuous function."""

    def test_continuous_to_discrete_conversion(self):
        """Test that x_des_ts is created from x_des_fn."""
        ts = jnp.linspace(0, 1, 10)

        def x_fn(t):
            return jnp.array([jnp.sin(t), jnp.cos(t)])

        ref = ReferenceTrajectory(ts=ts, x_des_fn=x_fn)

        assert ref.x_des_ts is not None
        assert ref.x_des_ts.shape == (10, 2)

        # Verify values match function evaluation
        for i, t in enumerate(ts):
            assert jnp.allclose(ref.x_des_ts[i], x_fn(t), atol=1e-10)

    def test_velocity_auto_derived_from_function(self):
        """Test that velocity is correctly derived via JAX autograd."""
        ts = jnp.linspace(0, 1, 10)

        def x_fn(t):
            return jnp.array([jnp.sin(t), jnp.cos(t)])

        ref = ReferenceTrajectory(ts=ts, x_des_fn=x_fn)

        # Derivative of sin is cos, derivative of cos is -sin
        t_test = 0.5
        xd = ref.xd_des_fn(t_test)
        expected_xd = jnp.array([jnp.cos(t_test), -jnp.sin(t_test)])

        assert jnp.allclose(xd, expected_xd, atol=1e-6)

    def test_acceleration_auto_derived_from_function(self):
        """Test that acceleration is correctly derived via JAX autograd."""
        ts = jnp.linspace(0, 1, 10)

        def x_fn(t):
            return jnp.array([jnp.sin(t), jnp.cos(t)])

        ref = ReferenceTrajectory(ts=ts, x_des_fn=x_fn)

        # Second derivative of sin is -sin, second derivative of cos is -cos
        t_test = 0.5
        xdd = ref.xdd_des_fn(t_test)
        expected_xdd = jnp.array([-jnp.sin(t_test), -jnp.cos(t_test)])

        assert jnp.allclose(xdd, expected_xdd, atol=1e-6)


class TestReferenceTrajectoryValidation:
    """Tests for input validation."""

    def test_error_when_neither_provided(self):
        """Test that ValueError is raised when neither x_des_ts nor x_des_fn is provided."""
        ts = jnp.linspace(0, 1, 10)

        with pytest.raises(ValueError, match="At least one of"):
            ReferenceTrajectory(ts=ts)

    def test_both_provided_preserves_values(self):
        """Test that when both are provided, original values are preserved."""
        ts = jnp.linspace(0, 1, 10)
        x_des_ts = jnp.column_stack([jnp.sin(ts), jnp.cos(ts)])

        def custom_fn(t):
            return jnp.array([t * 2, t * 3])

        ref = ReferenceTrajectory(ts=ts, x_des_ts=x_des_ts, x_des_fn=custom_fn)

        # x_des_ts should be preserved
        assert jnp.allclose(ref.x_des_ts, x_des_ts, atol=1e-10)

        # x_des_fn should use the custom function
        assert jnp.allclose(ref.x_des_fn(0.5), custom_fn(0.5), atol=1e-10)


class TestReferenceTrajectoryPyTree:
    """Tests for JAX PyTree compatibility."""

    def test_tree_flatten_unflatten_roundtrip(self):
        """Test that flatten/unflatten preserves data."""
        ts = jnp.linspace(0, 1, 10)
        x_des_ts = jnp.column_stack([jnp.sin(ts), jnp.cos(ts)])

        ref = ReferenceTrajectory(ts=ts, x_des_ts=x_des_ts)
        children, aux = ref.tree_flatten()
        ref_restored = ReferenceTrajectory.tree_unflatten(aux, children)

        assert jnp.allclose(ref_restored.ts, ref.ts, atol=1e-10)
        assert jnp.allclose(ref_restored.x_des_ts, ref.x_des_ts, atol=1e-10)
        assert jnp.allclose(ref_restored.xd_des_ts, ref.xd_des_ts, atol=1e-10)
        assert jnp.allclose(ref_restored.xdd_des_ts, ref.xdd_des_ts, atol=1e-10)

    def test_jit_compatibility(self):
        """Test that ReferenceTrajectory works with jax.jit."""
        ts = jnp.linspace(0, 1, 10)

        def x_fn(t):
            return jnp.array([jnp.sin(t), jnp.cos(t)])

        ref = ReferenceTrajectory(ts=ts, x_des_fn=x_fn)

        @jax.jit
        def evaluate(ref_traj, t):
            return ref_traj.x_des_fn(t), ref_traj.xd_des_fn(t)

        x, xd = evaluate(ref, jnp.array(0.25))

        expected_x = x_fn(0.25)
        expected_xd = jnp.array([jnp.cos(0.25), -jnp.sin(0.25)])

        assert jnp.allclose(x, expected_x, atol=1e-6)
        assert jnp.allclose(xd, expected_xd, atol=1e-6)

    def test_vmap_compatibility(self):
        """Test that ReferenceTrajectory functions work with jax.vmap."""
        ts = jnp.linspace(0, 1, 10)

        def x_fn(t):
            return jnp.array([jnp.sin(t), jnp.cos(t)])

        ref = ReferenceTrajectory(ts=ts, x_des_fn=x_fn)

        # vmap over time queries
        t_queries = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        x_batch = jax.vmap(ref.x_des_fn)(t_queries)

        assert x_batch.shape == (5, 2)

        for i, t in enumerate(t_queries):
            assert jnp.allclose(x_batch[i], x_fn(t), atol=1e-6)


class TestReferenceTrajectoryExplicitDerivatives:
    """Tests for explicitly provided derivatives."""

    def test_explicit_velocity_discrete(self):
        """Test that explicitly provided xd_des_ts is used."""
        ts = jnp.linspace(0, 1, 10)
        x_des_ts = jnp.column_stack([jnp.sin(ts), jnp.cos(ts)])
        xd_des_ts = jnp.ones((10, 2))  # Custom velocity

        ref = ReferenceTrajectory(ts=ts, x_des_ts=x_des_ts, xd_des_ts=xd_des_ts)

        assert jnp.allclose(ref.xd_des_ts, xd_des_ts, atol=1e-10)
        # xd_des_fn should interpolate the provided discrete values
        assert jnp.allclose(ref.xd_des_fn(0.5), jnp.ones(2), atol=1e-6)

    def test_explicit_velocity_continuous(self):
        """Test that explicitly provided xd_des_fn is used."""
        ts = jnp.linspace(0, 1, 10)
        x_des_ts = jnp.column_stack([jnp.sin(ts), jnp.cos(ts)])

        def xd_fn(t):
            return jnp.array([1.0, 2.0])  # Constant velocity

        ref = ReferenceTrajectory(ts=ts, x_des_ts=x_des_ts, xd_des_fn=xd_fn)

        assert jnp.allclose(ref.xd_des_fn(0.5), jnp.array([1.0, 2.0]), atol=1e-10)
        # xd_des_ts should be evaluated from the function
        assert ref.xd_des_ts.shape == (10, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
