import jax.numpy as jnp
import pytest

from soromox.control.base_controller import BaseController
from soromox.control.closed_form_model_based_controller import (
    ClosedFormModelBasedController,
)
from soromox.systems.system_state import SystemState


def test_base_controller_apply_gain_supports_vectors_matrices_and_rejects_tensors():
    x = jnp.array([2.0, -1.0])

    assert jnp.allclose(
        BaseController._apply_gain(jnp.array([3.0, 4.0]), x), jnp.array([6.0, -4.0])
    )
    assert jnp.allclose(
        BaseController._apply_gain(jnp.array([[1.0, 2.0], [3.0, 4.0]]), x),
        jnp.array([0.0, 2.0]),
    )
    with pytest.raises(ValueError, match="Gain must be"):
        BaseController._apply_gain(jnp.ones((1, 1, 1)), x)


def test_closed_form_model_based_controller_defaults_to_zero_control(
    fake_robot_factory, make_reference_factory
):
    robot = fake_robot_factory(jnp.eye(2))
    controller = ClosedFormModelBasedController(
        robot=robot,
        reference_trajectory=make_reference_factory([0.0, 0.0], [0.0, 0.0]),
    )

    u, control_state_dot = controller(SystemState(t=jnp.array(0.0), y=jnp.zeros((4,))))

    assert jnp.allclose(u, jnp.zeros((2,)))
    assert control_state_dot is None
