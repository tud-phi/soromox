"""Shared JAX/Warp equivalence checks for continuum-system kinematics."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose


def _sample_coordinates(model: Any) -> jax.Array:
    """Return unsorted, duplicated, and boundary backbone coordinates."""

    lengths = getattr(model, "segment_lengths", getattr(model, "L", None))
    total_length = jnp.sum(lengths)
    first_boundary = lengths[0]
    return jnp.asarray(
        [
            total_length,
            0.37 * total_length,
            first_boundary,
            0.0,
            0.81 * total_length,
            0.37 * total_length,
        ],
        dtype=jnp.float64,
    )


def _assert_result_close(actual: Any, expected: Any) -> None:
    """Compare one pose, Jacobian, or fused result PyTree."""

    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        assert_allclose(actual_leaf, expected_leaf, rtol=2e-9, atol=2e-10)


def assert_kinematics_backend_equivalence(model: Any) -> None:
    """Check every supported scalar and vectorized public input form.

    Args:
        model: Exact PlanarPCS, PCS, or GVS model supported by Warp.

    Returns:
        None.
    """

    q = jnp.stack(
        (
            jnp.linspace(-0.025, 0.031, model.num_internal_dofs, dtype=jnp.float64),
            jnp.linspace(0.019, -0.017, model.num_internal_dofs, dtype=jnp.float64),
        )
    )
    samples = _sample_coordinates(model)
    per_environment_samples = jnp.stack((samples, samples[::-1]))

    methods = (
        "forward_kinematics",
        "jacobian_inertialframe",
        "forward_kinematics_and_jacobian_inertialframe",
    )
    for method_name in methods:
        method = getattr(model, method_name)
        expected = method(q[0], samples[1], backend="jax")
        actual = method(q[0], samples[1], backend="warp")
        _assert_result_close(actual, expected)

        jax_spatial = jax.vmap(partial(method, backend="jax"), in_axes=(None, 0))
        warp_spatial = jax.vmap(partial(method, backend="warp"), in_axes=(None, 0))
        expected_spatial = jax_spatial(q[0], samples)
        actual_spatial = warp_spatial(q[0], samples)
        _assert_result_close(actual_spatial, expected_spatial)

        pairwise_samples = samples[: q.shape[0]]
        expected_pairwise = jax.vmap(partial(method, backend="jax"))(
            q, pairwise_samples
        )
        actual_pairwise = jax.vmap(partial(method, backend="warp"))(q, pairwise_samples)
        _assert_result_close(actual_pairwise, expected_pairwise)

        expected_environments = jax.vmap(
            partial(method, backend="jax"), in_axes=(0, None)
        )(q, samples[1])
        actual_environments = jax.vmap(
            partial(method, backend="warp"), in_axes=(0, None)
        )(q, samples[1])
        _assert_result_close(actual_environments, expected_environments)

        expected_cartesian = jax.vmap(jax_spatial, in_axes=(0, None))(q, samples)
        actual_cartesian = jax.vmap(warp_spatial, in_axes=(0, None))(q, samples)
        _assert_result_close(actual_cartesian, expected_cartesian)

    for abscissa_method_name, vectorized_method_name in (
        ("forward_kinematics_abscissa_batched", "forward_kinematics"),
        (
            "jacobian_inertialframe_abscissa_batched",
            "jacobian_inertialframe",
        ),
        (
            "forward_kinematics_and_jacobian_inertialframe_abscissa_batched",
            "forward_kinematics_and_jacobian_inertialframe",
        ),
    ):
        abscissa_method = getattr(model, abscissa_method_name)
        scalar_method = getattr(model, vectorized_method_name)
        expected = jax.vmap(partial(scalar_method, backend="jax"), in_axes=(None, 0))(
            q[0], samples
        )
        actual = abscissa_method(q[0], samples, backend="warp")
        _assert_result_close(actual, expected)

        jax_environment_batch = jax.vmap(
            partial(abscissa_method, backend="jax"), in_axes=(0, None)
        )
        warp_environment_batch = jax.vmap(
            partial(abscissa_method, backend="warp"), in_axes=(0, None)
        )
        expected_environments = jax_environment_batch(q, samples)
        actual_environments = warp_environment_batch(q, samples)
        _assert_result_close(actual_environments, expected_environments)

        expected_per_environment = jax.vmap(partial(abscissa_method, backend="jax"))(
            q, per_environment_samples
        )
        actual_per_environment = jax.vmap(partial(abscissa_method, backend="warp"))(
            q, per_environment_samples
        )
        _assert_result_close(actual_per_environment, expected_per_environment)


def assert_inertial_jacobian_finite_difference(model: Any) -> None:
    """Check inertial Jacobians independently against pose differences.

    Args:
        model: Exact PlanarPCS, PCS, or GVS model supported by Warp.

    Returns:
        None.
    """

    q = jnp.linspace(-0.022, 0.028, model.num_internal_dofs, dtype=jnp.float64)
    s = _sample_coordinates(model)[1]
    step = 2e-6
    columns = []
    pose = model.forward_kinematics(q, s, backend="warp")
    for column in range(model.num_internal_dofs):
        direction = jnp.zeros_like(q).at[column].set(step)
        pose_plus = model.forward_kinematics(q + direction, s, backend="warp")
        pose_minus = model.forward_kinematics(q - direction, s, backend="warp")
        pose_derivative = (pose_plus - pose_minus) / (2.0 * step)
        if model.is_planar:
            columns.append(pose_derivative)
        else:
            rotation_derivative = pose_derivative[:3, :3]
            angular_skew = rotation_derivative @ pose[:3, :3].T
            angular = jnp.asarray(
                [angular_skew[2, 1], angular_skew[0, 2], angular_skew[1, 0]]
            )
            columns.append(jnp.concatenate((angular, pose_derivative[:3, 3])))

    finite_difference = jnp.stack(columns, axis=1)
    actual = model.jacobian_inertialframe(q, s, backend="warp")
    assert_allclose(actual, finite_difference, rtol=3e-5, atol=3e-7)


def assert_warp_derivatives_use_jax(model: Any) -> None:
    """Check that differentiating a Warp request follows the JAX equations.

    Args:
        model: Exact PlanarPCS, PCS, or GVS model supported by Warp.

    Returns:
        None.
    """

    q = jnp.linspace(-0.018, 0.021, model.num_internal_dofs, dtype=jnp.float64)
    tangent = jnp.linspace(0.011, -0.009, model.num_internal_dofs, dtype=jnp.float64)
    s = _sample_coordinates(model)[1]
    for method_name in (
        "forward_kinematics",
        "jacobian_inertialframe",
        "forward_kinematics_and_jacobian_inertialframe",
        "forward_kinematics_abscissa_batched",
        "jacobian_inertialframe_abscissa_batched",
        "forward_kinematics_and_jacobian_inertialframe_abscissa_batched",
    ):
        method = getattr(model, method_name)
        s_value = (
            _sample_coordinates(model)
            if method_name.endswith("abscissa_batched")
            else s
        )
        coordinate_argument = (
            {"s_ps": s_value}
            if method_name.endswith("abscissa_batched")
            else {"s": s_value}
        )
        expected = jax.jvp(
            partial(method, backend="jax", **coordinate_argument),
            (q,),
            (tangent,),
        )
        actual = jax.jvp(
            partial(method, backend="warp", **coordinate_argument),
            (q,),
            (tangent,),
        )
        _assert_result_close(actual, expected)


__all__ = [
    "assert_inertial_jacobian_finite_difference",
    "assert_kinematics_backend_equivalence",
    "assert_warp_derivatives_use_jax",
]
