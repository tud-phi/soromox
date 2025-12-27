import cv2
from jax import Array, vmap
from jax import numpy as jnp
import numpy as onp
from typing import Callable, Dict, Optional, Tuple

from soromox.systems import PlanarPCS


def render_planar_pcs(
    robot: PlanarPCS,
    q: Array,
    width: int,
    height: int,
    origin_uv: Optional[Tuple] = None,
    length_scale: float = 2.0,
    backbone_thickness: Optional[int] = None,
    num_points: int = 50,
) -> onp.ndarray:
    """
    Renders a planar pcs soft robot in OpenCV.
    Args:
        robot: A PlanarPCS object representing the robot.
        q: An array of joint angles.
        width: The width (i.e. number of horizontal pixels) of the rendered image.
        height: The height (i.e. number of vertical pixels) of the rendered image.
        length_scale: The length scale of the robot in the image. The robot will occupy height/length_scale in pixels.
        origin_uv: The (u, v) pixel coordinates of the robot base. If None, set to the center of the image.
        backbone_thickness: The thickness of the rendered lines in pixels. If None, set it according to the robot diameter.
        num_points: The number of points used for discretizing the backbone curve.
    Returns:
        img: A numpy array of shape (width, height, 3) containing the rendered image.
    """
    # plotting in OpenCV
    h, w = height, width  # img height and width
    ppm = h / (length_scale * jnp.sum(robot.L))  # pixel per meter
    base_color = (0, 255, 0)  # green base_color in BGR
    robot_color = (0, 0, 0)  # black robot_color in BGR

    if backbone_thickness is None:
        backbone_thicknesses = jnp.clip((ppm * robot.r).astype(jnp.int32), min=1, max=None)
    else:
        backbone_thicknesses = jnp.ones((robot.num_segments,)) * backbone_thickness

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

    # initialize image 
    img = 255 * onp.ones((h, w, 3), dtype=jnp.uint8)  # initialize background to white

    # plot the base
    cv2.circle(img, origin_uv, int(1.5 * robot.r.mean().item() * ppm), base_color, -1)

    # transform robot poses to pixel coordinates
    # should be of shape (N, 2)
    curve = onp.array((chi_ps[1:3, :].T * ppm), dtype=onp.int32)
    # invert the v pixel coordinate
    curve[:, 1] = -curve[:, 1]

    # plot each segment 
    L_cum = jnp.concatenate((jnp.array([0]), jnp.cumsum(robot.L)))
    for segment_idx in range(robot.num_segments):
        curve_selector = (s_ps >= L_cum[segment_idx]) & (s_ps < L_cum[segment_idx + 1])
        cv2.polylines(
            img,
            [origin_uv + curve[curve_selector]],
            isClosed=False,
            color=robot_color,
            thickness=int(backbone_thicknesses[segment_idx].item()),
        )

    return img