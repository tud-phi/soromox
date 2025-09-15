import cv2
from jax import Array, vmap
from jax import numpy as jnp
import numpy as onp
from typing import Callable, Dict, Optional, Tuple

from soromox.systems.planar_pcs import PlanarPCS


def render_planar_pcs(
    robot: PlanarPCS,
    q: Array,
    width: int,
    height: int,
    origin_uv: Optional[Tuple] = None,
    line_thickness: int = 2,
    num_points: int = 50,
) -> onp.ndarray:
    """
    Renders a planar pcs soft robot in OpenCV.
    Args:
        robot: A PlanarPCS object representing the robot.
        q: An array of joint angles.
        width: The width (i.e. number of horizontal pixels) of the rendered image.
        height: The height (i.e. number of vertical pixels) of the rendered image.
        line_thickness: The thickness of the rendered lines in pixels.
        num_points: The number of points used for discretizing the backbone curve.
    Returns:
        img: A numpy array of shape (width, height, 3) containing the rendered image.
    """
    # plotting in OpenCV
    h, w = height, width  # img height and width
    ppm = h / (1.6 * jnp.sum(robot.L))  # pixel per meter
    robot_color = (0, 0, 0)  # black robot_color in BGR

    # in uv pixel coordinates
    if origin_uv is None:
        origin_uv = (w // 2, h // 2)  # center of the image
    origin_uv = onp.array(origin_uv, dtype=onp.int32)

    # vmap the forward kinematics function
    batched_forward_kinematics_fn = vmap(
        robot.forward_kinematics, in_axes=(None, 0), out_axes=-1
    )

    # we use for plotting N points along the length of the robot
    s_ps = jnp.linspace(0, jnp.sum(robot.L), num_points)

    # poses along the robot of shape (3, N)
    chi_ps = batched_forward_kinematics_fn(q, s_ps)

    img = 255 * onp.ones((h, w, 3), dtype=jnp.uint8)  # initialize background to white
    # transform robot poses to pixel coordinates
    # should be of shape (N, 2)
    curve = onp.array((chi_ps[:2, :].T * ppm), dtype=onp.int32)
    # invert the v pixel coordinate
    curve[:, 1] = -curve[:, 1]

    cv2.polylines(
        img,
        [origin_uv + curve],
        isClosed=False,
        color=robot_color,
        thickness=line_thickness,
    )

    return img