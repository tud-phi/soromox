"""Shared model factories for Warp execution-backend tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from soromox.systems import (
    GVS,
    PCS,
    GVSSegment,
    JointSpec,
    LinkSpec,
    PCSStructure,
    PlanarPCS,
    PlanarPCSStructure,
    StrainBasisSpec,
)
from soromox.systems.execution import ExecutionBackend

jax.config.update("jax_enable_x64", True)

ModelFactory = Callable[..., Any]


def _spatial_link(length: float) -> LinkSpec:
    """Create a damped circular link for spatial equivalence tests."""

    return LinkSpec.circular(
        length=length,
        radius=0.018,
        density=1040.0,
        reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        young_modulus=2.5e5,
        shear_modulus=9.0e4,
        material_damping_coefficient=0.02,
    )


def _planar_link(length: float) -> LinkSpec:
    """Create a damped circular link for planar equivalence tests."""

    return LinkSpec.circular(
        length=length,
        radius=0.018,
        density=1040.0,
        reference_strain=[0.0, 1.0, 0.0],
        young_modulus=2.5e5,
        shear_modulus=9.0e4,
        material_damping_coefficient=0.02,
    )


@pytest.fixture
def make_gvs_model() -> ModelFactory:
    """Return a factory for shape-generic, general-joint GVS models."""

    segments = (
        GVSSegment(
            link=_spatial_link(0.12),
            joint=JointSpec.revolute(axis="z", stiffness=[[0.2]], damping=[[0.01]]),
            basis=StrainBasisSpec(
                type="legendre",
                strain_selector=[False, True, False, True, False, False],
                basis_order=[0, 1, 0, 1, 0, 0],
            ),
            num_gauss_points=5,
        ),
        GVSSegment(
            link=_spatial_link(0.09),
            joint=JointSpec.prismatic(axis="x", stiffness=[[0.3]], damping=[[0.015]]),
            basis=StrainBasisSpec(
                type="monomial",
                strain_selector=[True, False, False, True, False, True],
                basis_order=[1, 0, 0, 0, 0, 0],
            ),
            num_gauss_points=5,
        ),
    )

    def make(backend: ExecutionBackend) -> GVS:
        """Build a GVS model with identical parameters for one backend."""

        return GVS.from_segments(
            segments,
            gravity=jnp.asarray([0.2, -0.1, -9.81], dtype=jnp.float64),
            scale_rotational_basis_by_length=True,
            backend=backend,
        )

    return make


@pytest.fixture
def make_pcs_model() -> ModelFactory:
    """Return a factory for spatial PCS models."""

    links = (_spatial_link(0.12), _spatial_link(0.09))
    selector = jnp.tile(
        jnp.asarray([True, False, True, True, False, True], dtype=bool),
        2,
    )

    def make(backend: ExecutionBackend, *, num_gauss_points: int = 5) -> PCS:
        """Build a spatial PCS model with identical parameters for one backend."""

        return PCS.from_links(
            links,
            gravity=jnp.asarray([0.2, -0.1, -9.81], dtype=jnp.float64),
            structure=PCSStructure(
                num_gauss_points=num_gauss_points,
                strain_selector=selector,
                scale_rotational_basis_by_length=True,
            ),
            backend=backend,
        )

    return make


@pytest.fixture
def make_planar_pcs_model() -> ModelFactory:
    """Return a factory for planar PCS models."""

    links = (_planar_link(0.12), _planar_link(0.09))
    selector = jnp.tile(
        jnp.asarray([True, False, True], dtype=bool),
        2,
    )

    def make(backend: ExecutionBackend, *, num_gauss_points: int = 5) -> PlanarPCS:
        """Build a planar PCS model with identical parameters for one backend."""

        return PlanarPCS.from_links(
            links,
            gravity=jnp.asarray([0.2, -9.81], dtype=jnp.float64),
            structure=PlanarPCSStructure(
                num_gauss_points=num_gauss_points,
                strain_selector=selector,
                scale_rotational_basis_by_length=True,
            ),
            backend=backend,
        )

    return make


@pytest.fixture
def state_batch() -> Callable[[Any], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Return deterministic nonzero configurations, velocities, and states."""

    def make(model: Any) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Construct two independent environments for ``model``."""

        q = jnp.stack(
            (
                jnp.linspace(-0.018, 0.025, model.num_dofs, dtype=jnp.float64),
                jnp.linspace(0.012, -0.016, model.num_dofs, dtype=jnp.float64),
            )
        )
        qd = jnp.stack((0.35 * q[0], -0.45 * q[1]))
        return q, qd, jnp.concatenate((q, qd), axis=1)

    return make
