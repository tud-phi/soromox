"""Gauss rules must retain the precision requested by JAX."""

import jax
import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox.utils.integration import gauss_quadrature


@pytest.mark.parametrize("count", [5, 8, 12])
@pytest.mark.parametrize("double", [False, True])
def test_gauss_polynomial_exactness(count, double):
    with jax.enable_x64(double):
        nodes, weights, size = gauss_quadrature(count)
        tolerance = 2e-13 if double else 3e-6
        assert size == count + 2
        assert_allclose(np.asarray(nodes)[[0, -1]], [0, 1])
        assert_allclose(np.asarray(weights)[[0, -1]], 0)
        for degree in range(2 * count):
            actual = np.sum(np.asarray(weights) * np.asarray(nodes) ** degree)
            assert_allclose(actual, 1 / (degree + 1), rtol=tolerance, atol=tolerance)
