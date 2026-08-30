"""Renderer behavior for floating-base continuum systems."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox.rendering.base import BaseSoftRobotRenderer
from soromox.systems import (
    PCS,
    LinkSpec,
    PCSStructure,
    PlanarPCS,
    PlanarPCSStructure,
)

jax.config.update("jax_enable_x64", True)


def _spatial_pcs() -> PCS:
    """Construct a minimal spatial floating PCS renderer fixture."""
    link = LinkSpec.circular(
        length=0.1,
        radius=0.01,
        density=1000.0,
        young_modulus=1.0e5,
        shear_modulus=4.0e4,
        damping=jnp.eye(6),
        reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )
    return PCS.from_links(
        [link],
        structure=PCSStructure(num_gauss_points=3),
        floating_base=True,
        backend="jax",
    )


def _planar_pcs() -> PlanarPCS:
    """Construct a minimal planar floating PCS renderer fixture."""
    link = LinkSpec.circular(
        length=0.1,
        radius=0.01,
        density=1000.0,
        young_modulus=1.0e5,
        shear_modulus=4.0e4,
        damping=jnp.eye(3),
        reference_strain=[0.0, 1.0, 0.0],
    )
    return PlanarPCS.from_links(
        [link],
        structure=PlanarPCSStructure(num_gauss_points=3),
        floating_base=True,
        backend="jax",
    )


class _FloatingRendererProbe(BaseSoftRobotRenderer):
    """Minimal renderer exposing the shared geometry preparation path."""

    def render_frame(self, q):
        """Return an empty frame after satisfying the abstract contract."""
        del q
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def render_sequence(self, ts, q_ts, playback_speed=1.0, record_path=None, **kwargs):
        """Satisfy the abstract sequence-rendering contract."""
        del ts, q_ts, playback_speed, record_path, kwargs

    def show(self, q, **kwargs):
        """Satisfy the abstract interactive-rendering contract."""
        del q, kwargs


@pytest.mark.parametrize("factory", [_spatial_pcs, _planar_pcs])
def test_renderer_follows_runtime_base_pose(factory) -> None:
    """Follow the runtime base in both spatial and planar geometry paths."""
    robot = factory()
    base_pose = (
        jnp.array([0.96, 0.1, -0.2, 0.15, 0.3, -0.2, 0.4])
        if not robot.is_planar
        else jnp.array([0.4, 0.3, -0.2])
    )
    q = robot.pack_configuration(
        jnp.zeros((robot.num_internal_dofs,)), base_pose=base_pose
    )
    renderer = _FloatingRendererProbe(robot, num_points=5)

    backbone_poses = renderer.compute_backbone_poses(q)
    expected_base = robot.base_transform_from_configuration(q)
    assert_allclose(renderer.base_transform, expected_base)
    if robot.is_planar:
        assert_allclose(backbone_poses[0], robot.forward_kinematics(q, 0.0))
    else:
        assert_allclose(backbone_poses[0], expected_base, rtol=1.0e-10, atol=1.0e-11)
