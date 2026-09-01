from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from soromox.utils.joint_motion_subspace import (
    joint_dSdq_qd,
    joint_dSTF_dq,
    joint_dSTF_dq_direction,
    joint_motion_subspace_with_derivatives,
)
from soromox.utils.lie_algebra import se3

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("rotation", [1.0e-5, 1.0e-1])
def test_joint_motion_subspace_derivatives_match_autodiff(rotation: float) -> None:
    """Exercise both the stable-series and closed-form derivative branches."""
    interval_length = jnp.array(1.0, dtype=jnp.float64)
    strain_basis = jnp.eye(6, dtype=jnp.float64)
    reference_strain = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    q = jnp.array([rotation, 0.0, 0.0, 0.02, -0.01, 0.03])
    qd = jnp.array([0.2, -0.3, 0.4, -0.1, 0.5, -0.2])
    qdd = jnp.array([-0.4, 0.1, 0.3, 0.2, -0.2, 0.6])
    wrench = jnp.array([0.3, -0.2, 0.1, -0.4, 0.6, -0.5])
    eps = jnp.finfo(q.dtype).eps

    def motion_subspace(q_value: jax.Array) -> jax.Array:
        integrated_strain = interval_length * (
            strain_basis @ q_value + reference_strain
        )
        return se3.left_jacobian(integrated_strain, eps) @ (
            interval_length * strain_basis
        )

    def motion_subspace_dot(q_value: jax.Array) -> jax.Array:
        return jax.jvp(motion_subspace, (q_value,), (qd,))[1]

    expected_transform = se3.exp(
        interval_length * (strain_basis @ q + reference_strain),
        eps=eps,
    )
    expected_subspace = motion_subspace(q)
    expected_subspace_dot = motion_subspace_dot(q)
    expected_deta_dq = jax.jacfwd(lambda value: motion_subspace(value) @ qd)(q)
    expected_dS_qdd_dq = jax.jacfwd(lambda value: motion_subspace(value) @ qdd)(q)
    expected_dSd_qd_dq = jax.jacfwd(lambda value: motion_subspace_dot(value) @ qd)(q)
    expected_dSTF_dq = jax.jacfwd(lambda value: motion_subspace(value).T @ wrench)(q)
    q_directions = jnp.stack((qd, qdd), axis=1)

    (
        transform,
        subspace,
        subspace_dot,
        deta_dq,
        dS_qdd_dq,
        dSd_qd_dq,
    ) = joint_motion_subspace_with_derivatives(
        interval_length,
        strain_basis,
        reference_strain,
        q,
        qd,
        qdd,
        eps,
    )

    assert_allclose(transform, expected_transform, rtol=1.0e-8, atol=1.0e-10)
    assert_allclose(subspace, expected_subspace, rtol=1.0e-8, atol=1.0e-10)
    assert_allclose(subspace_dot, expected_subspace_dot, rtol=1.0e-8, atol=1.0e-10)
    assert_allclose(deta_dq, expected_deta_dq, rtol=1.0e-8, atol=1.0e-10)
    assert_allclose(dS_qdd_dq, expected_dS_qdd_dq, rtol=1.0e-8, atol=1.0e-10)
    assert_allclose(dSd_qd_dq, expected_dSd_qd_dq, rtol=1.0e-8, atol=1.0e-10)
    assert_allclose(
        joint_dSdq_qd(
            interval_length,
            strain_basis,
            reference_strain,
            q,
            qd,
            eps,
        ),
        expected_deta_dq,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    assert_allclose(
        joint_dSTF_dq(
            interval_length,
            strain_basis,
            reference_strain,
            q,
            wrench,
            eps,
        ),
        expected_dSTF_dq,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    assert_allclose(
        joint_dSTF_dq_direction(
            interval_length,
            strain_basis,
            reference_strain,
            q,
            wrench,
            q_directions,
            eps,
        ),
        expected_dSTF_dq @ q_directions,
        rtol=1.0e-8,
        atol=1.0e-10,
    )

    directional_results = joint_motion_subspace_with_derivatives(
        interval_length,
        strain_basis,
        reference_strain,
        q,
        qd,
        qdd,
        eps,
        q_directions,
    )
    assert_allclose(
        directional_results[3],
        expected_deta_dq @ q_directions,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    assert_allclose(
        directional_results[4],
        expected_dS_qdd_dq @ q_directions,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    assert_allclose(
        directional_results[5],
        expected_dSd_qd_dq @ q_directions,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
