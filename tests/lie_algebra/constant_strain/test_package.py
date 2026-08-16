"""Tests for the public constant-strain package namespace."""

from soromox.utils.lie_algebra import constant_strain


def test_public_api_is_algebra_namespaced():
    assert constant_strain.__all__ == ["ConstantStrainOperators", "se2", "se3"]

    flat_operator_names = (
        "adjoint_se2",
        "adjoint_inverse_se2",
        "operators_se2",
        "tangent_se2",
        "tangent_derivative_se2",
        "adjoint_se3",
        "adjoint_inverse_se3",
        "operators_se3",
        "tangent_se3",
        "tangent_derivative_se3",
    )
    assert all(not hasattr(constant_strain, name) for name in flat_operator_names)
