# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)  # double precision

import jax.numpy as jnp
import pytest

from soromox.control import PIDControl


class TestPIDControlInitialization:
    """Tests for PIDControl initialization with different gain formats."""

    def test_scalar_gains(self):
        """Test initialization with scalar gains."""
        pid = PIDControl(Kp=1.0, Ki=0.5, Kd=0.1)

        assert pid.Kp.shape == (1,)
        assert pid.Ki.shape == (1,)
        assert pid.Kd.shape == (1,)
        assert jnp.allclose(pid.Kp, jnp.array([1.0]))
        assert jnp.allclose(pid.Ki, jnp.array([0.5]))
        assert jnp.allclose(pid.Kd, jnp.array([0.1]))

    def test_diagonal_gains(self):
        """Test initialization with diagonal (1-d array) gains."""
        Kp = jnp.array([1.0, 2.0, 3.0])
        Ki = jnp.array([0.1, 0.2, 0.3])
        Kd = jnp.array([0.01, 0.02, 0.03])

        pid = PIDControl(Kp=Kp, Ki=Ki, Kd=Kd)

        assert pid.Kp.shape == (3,)
        assert pid.Ki.shape == (3,)
        assert pid.Kd.shape == (3,)
        assert jnp.allclose(pid.Kp, Kp)
        assert jnp.allclose(pid.Ki, Ki)
        assert jnp.allclose(pid.Kd, Kd)

    def test_matrix_gains(self):
        """Test initialization with matrix (2-d array) gains."""
        Kp = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        Ki = jnp.array([[0.1, 0.0], [0.0, 0.1]])
        Kd = jnp.array([[0.01, 0.0], [0.0, 0.01]])

        pid = PIDControl(Kp=Kp, Ki=Ki, Kd=Kd)

        assert pid.Kp.shape == (2, 2)
        assert pid.Ki.shape == (2, 2)
        assert pid.Kd.shape == (2, 2)

    def test_default_gamma(self):
        """Test that gamma defaults to 1.0."""
        pid = PIDControl(Kp=1.0, Ki=0.5, Kd=0.1)

        assert jnp.allclose(pid.gamma, jnp.array([1.0]))

    def test_custom_gamma(self):
        """Test initialization with custom gamma."""
        pid = PIDControl(Kp=1.0, Ki=0.5, Kd=0.1, gamma=2.0)

        assert jnp.allclose(pid.gamma, jnp.array([2.0]))


class TestPIDControlOutput:
    """Tests for PID control output computation."""

    def test_proportional_only(self):
        """Test PID with only proportional gain."""
        pid = PIDControl(Kp=2.0, Ki=0.0, Kd=0.0)

        e = jnp.array([1.0, 2.0])
        ed = jnp.array([0.5, 0.5])
        integral_error = jnp.array([10.0, 10.0])

        u, integral_error_dot = pid(e, ed, integral_error)

        # u should be Kp * e
        expected_u = 2.0 * e
        assert jnp.allclose(u, expected_u, atol=1e-10)

    def test_integral_only(self):
        """Test PID with only integral gain."""
        pid = PIDControl(Kp=0.0, Ki=0.5, Kd=0.0)

        e = jnp.array([1.0, 2.0])
        ed = jnp.array([0.5, 0.5])
        integral_error = jnp.array([4.0, 6.0])

        u, integral_error_dot = pid(e, ed, integral_error)

        # u should be Ki * integral_error
        expected_u = 0.5 * integral_error
        assert jnp.allclose(u, expected_u, atol=1e-10)

    def test_derivative_only(self):
        """Test PID with only derivative gain."""
        pid = PIDControl(Kp=0.0, Ki=0.0, Kd=0.3)

        e = jnp.array([1.0, 2.0])
        ed = jnp.array([0.5, 1.0])
        integral_error = jnp.array([10.0, 10.0])

        u, integral_error_dot = pid(e, ed, integral_error)

        # u should be Kd * ed
        expected_u = 0.3 * ed
        assert jnp.allclose(u, expected_u, atol=1e-10)

    def test_full_pid_output(self):
        """Test full PID control output."""
        Kp = jnp.array([1.0, 2.0])
        Ki = jnp.array([0.1, 0.2])
        Kd = jnp.array([0.01, 0.02])
        pid = PIDControl(Kp=Kp, Ki=Ki, Kd=Kd)

        e = jnp.array([1.0, 2.0])
        ed = jnp.array([0.5, 1.0])
        integral_error = jnp.array([5.0, 10.0])

        u, integral_error_dot = pid(e, ed, integral_error)

        # u = Kp * e + Ki * integral_error + Kd * ed
        expected_u = Kp * e + Ki * integral_error + Kd * ed
        assert jnp.allclose(u, expected_u, atol=1e-10)

    def test_matrix_gains_output(self):
        """Test PID with matrix gains."""
        Kp = jnp.array([[2.0, 0.5], [0.5, 2.0]])
        Ki = jnp.array([[0.1, 0.0], [0.0, 0.1]])
        Kd = jnp.array([[0.01, 0.0], [0.0, 0.01]])
        pid = PIDControl(Kp=Kp, Ki=Ki, Kd=Kd)

        e = jnp.array([1.0, 2.0])
        ed = jnp.array([0.5, 1.0])
        integral_error = jnp.array([5.0, 10.0])

        u, integral_error_dot = pid(e, ed, integral_error)

        # u = Kp @ e + Ki @ integral_error + Kd @ ed
        expected_u = Kp @ e + Ki @ integral_error + Kd @ ed
        assert jnp.allclose(u, expected_u, atol=1e-10)


class TestPIDControlIntegralError:
    """Tests for integral error derivative computation."""

    def test_identity_saturation(self):
        """Test that identity saturation returns error unchanged."""
        pid = PIDControl(Kp=1.0, Ki=0.5, Kd=0.1, saturation_fn="identity")

        e = jnp.array([1.0, -2.0, 3.0])
        ed = jnp.zeros(3)
        integral_error = jnp.zeros(3)

        u, integral_error_dot = pid(e, ed, integral_error)

        assert jnp.allclose(integral_error_dot, e, atol=1e-10)

    def test_none_saturation_is_identity(self):
        """Test that None saturation is equivalent to identity."""
        pid = PIDControl(Kp=1.0, Ki=0.5, Kd=0.1, saturation_fn=None)

        e = jnp.array([1.0, -2.0, 3.0])
        ed = jnp.zeros(3)
        integral_error = jnp.zeros(3)

        u, integral_error_dot = pid(e, ed, integral_error)

        assert jnp.allclose(integral_error_dot, e, atol=1e-10)

    def test_tanh_saturation_small_error(self):
        """Test tanh saturation with small errors (approximately linear)."""
        pid = PIDControl(Kp=1.0, Ki=0.5, Kd=0.1, saturation_fn="tanh", gamma=1.0)

        e = jnp.array([0.01, -0.01])
        ed = jnp.zeros(2)
        integral_error = jnp.zeros(2)

        u, integral_error_dot = pid(e, ed, integral_error)

        # For small e, tanh(e) ≈ e
        assert jnp.allclose(integral_error_dot, e, atol=1e-3)

    def test_tanh_saturation_large_error(self):
        """Test tanh saturation with large errors (saturated)."""
        pid = PIDControl(Kp=1.0, Ki=0.5, Kd=0.1, saturation_fn="tanh", gamma=1.0)

        e = jnp.array([100.0, -100.0])
        ed = jnp.zeros(2)
        integral_error = jnp.zeros(2)

        u, integral_error_dot = pid(e, ed, integral_error)

        # For large e, tanh(e) ≈ ±1
        expected = jnp.array([1.0, -1.0])
        assert jnp.allclose(integral_error_dot, expected, atol=1e-10)

    def test_tanh_saturation_with_gamma(self):
        """Test tanh saturation with custom gamma scaling."""
        gamma = 2.0
        pid = PIDControl(Kp=1.0, Ki=0.5, Kd=0.1, saturation_fn="tanh", gamma=gamma)

        e = jnp.array([0.5, -0.5])
        ed = jnp.zeros(2)
        integral_error = jnp.zeros(2)

        u, integral_error_dot = pid(e, ed, integral_error)

        expected = jnp.tanh(gamma * e)
        assert jnp.allclose(integral_error_dot, expected, atol=1e-10)

    def test_tanh_saturation_with_vector_gamma(self):
        """Test tanh saturation with vector gamma."""
        gamma = jnp.array([1.0, 2.0])
        pid = PIDControl(Kp=1.0, Ki=0.5, Kd=0.1, saturation_fn="tanh", gamma=gamma)

        e = jnp.array([0.5, 0.5])
        ed = jnp.zeros(2)
        integral_error = jnp.zeros(2)

        u, integral_error_dot = pid(e, ed, integral_error)

        expected = jnp.tanh(gamma * e)
        assert jnp.allclose(integral_error_dot, expected, atol=1e-10)

    def test_tanh_saturation_with_matrix_gamma(self):
        """Test tanh saturation with matrix gamma."""
        gamma = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        pid = PIDControl(Kp=1.0, Ki=0.5, Kd=0.1, saturation_fn="tanh", gamma=gamma)

        e = jnp.array([0.5, 0.5])
        ed = jnp.zeros(2)
        integral_error = jnp.zeros(2)

        u, integral_error_dot = pid(e, ed, integral_error)

        expected = jnp.tanh(gamma @ e)
        assert jnp.allclose(integral_error_dot, expected, atol=1e-10)

    def test_custom_saturation_function(self):
        """Test PID with custom saturation function."""

        def clip_saturation(e):
            return jnp.clip(e, -0.5, 0.5)

        pid = PIDControl(Kp=1.0, Ki=0.5, Kd=0.1, saturation_fn=clip_saturation)

        e = jnp.array([1.0, -2.0, 0.3])
        ed = jnp.zeros(3)
        integral_error = jnp.zeros(3)

        u, integral_error_dot = pid(e, ed, integral_error)

        expected = jnp.array([0.5, -0.5, 0.3])
        assert jnp.allclose(integral_error_dot, expected, atol=1e-10)


class TestPIDControlValidation:
    """Tests for input validation."""

    def test_unknown_saturation_function_raises_error(self):
        """Test that unknown saturation function raises ValueError."""
        with pytest.raises(ValueError, match="Unknown saturation function"):
            PIDControl(Kp=1.0, Ki=0.5, Kd=0.1, saturation_fn="unknown")


class TestPIDControlJAXCompatibility:
    """Tests for JAX compatibility (jit, vmap, grad)."""

    def test_jit_compatibility(self):
        """Test that PIDControl works with jax.jit."""
        pid = PIDControl(Kp=1.0, Ki=0.5, Kd=0.1)

        @jax.jit
        def compute_control(e, ed, integral_error):
            return pid(e, ed, integral_error)

        e = jnp.array([1.0, 2.0])
        ed = jnp.array([0.5, 0.5])
        integral_error = jnp.array([5.0, 10.0])

        u, integral_error_dot = compute_control(e, ed, integral_error)

        # Verify output is correct
        expected_u = 1.0 * e + 0.5 * integral_error + 0.1 * ed
        assert jnp.allclose(u, expected_u, atol=1e-10)

    def test_vmap_over_batch(self):
        """Test that PIDControl works with jax.vmap over batched inputs."""
        pid = PIDControl(Kp=1.0, Ki=0.5, Kd=0.1)

        # Batch of errors
        e_batch = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        ed_batch = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        integral_error_batch = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        # vmap over the batch dimension
        u_batch, integral_error_dot_batch = jax.vmap(pid)(
            e_batch, ed_batch, integral_error_batch
        )

        assert u_batch.shape == (3, 2)
        assert integral_error_dot_batch.shape == (3, 2)

        # Check individual results
        for i in range(3):
            expected_u = (
                1.0 * e_batch[i] + 0.5 * integral_error_batch[i] + 0.1 * ed_batch[i]
            )
            assert jnp.allclose(u_batch[i], expected_u, atol=1e-10)

    def test_jit_with_tanh_saturation(self):
        """Test jit compatibility with tanh saturation."""
        pid = PIDControl(Kp=1.0, Ki=0.5, Kd=0.1, saturation_fn="tanh", gamma=2.0)

        @jax.jit
        def compute_control(e, ed, integral_error):
            return pid(e, ed, integral_error)

        e = jnp.array([1.0, 2.0])
        ed = jnp.array([0.5, 0.5])
        integral_error = jnp.array([5.0, 10.0])

        u, integral_error_dot = compute_control(e, ed, integral_error)

        expected_integral_error_dot = jnp.tanh(2.0 * e)
        assert jnp.allclose(integral_error_dot, expected_integral_error_dot, atol=1e-10)

    def test_jit_with_custom_saturation(self):
        """Test jit compatibility with custom saturation function."""

        def clip_saturation(e):
            return jnp.clip(e, -1.0, 1.0)

        pid = PIDControl(Kp=1.0, Ki=0.5, Kd=0.1, saturation_fn=clip_saturation)

        @jax.jit
        def compute_control(e, ed, integral_error):
            return pid(e, ed, integral_error)

        e = jnp.array([2.0, -3.0])
        ed = jnp.zeros(2)
        integral_error = jnp.zeros(2)

        u, integral_error_dot = compute_control(e, ed, integral_error)

        expected_integral_error_dot = jnp.array([1.0, -1.0])
        assert jnp.allclose(integral_error_dot, expected_integral_error_dot, atol=1e-10)


class TestPIDControlEquinoxModule:
    """Tests for equinox module properties."""

    def test_is_equinox_module(self):
        """Test that PIDControl is an equinox Module."""
        import equinox as eqx

        pid = PIDControl(Kp=1.0, Ki=0.5, Kd=0.1)
        assert isinstance(pid, eqx.Module)

    def test_parameter_filtering(self):
        """Test that gains can be filtered as parameters."""
        import equinox as eqx

        pid = PIDControl(Kp=1.0, Ki=0.5, Kd=0.1)

        # Get leaves (trainable parameters)
        params, static = eqx.partition(pid, eqx.is_array)

        # Should have Kp, Ki, Kd, gamma as array leaves
        flat_params = jax.tree_util.tree_leaves(params)
        assert len(flat_params) == 4  # Kp, Ki, Kd, gamma


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
