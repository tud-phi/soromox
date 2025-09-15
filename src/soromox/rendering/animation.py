import cv2  # importing cv2
from jax import Array
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as onp
import os
from pathlib import Path
from tqdm import tqdm
from typing import Callable, Optional, Union


def animate_cv2(
    rendering_fn: Callable[[Array], onp.ndarray],
    t_ts: onp.ndarray,
    q_ts: onp.ndarray,
    filepath: os.PathLike,
    width: int,
    height: int,
    speed_up: Union[float, Array] = 1,
    skip_step: int = 1,
    rgb_to_bgr: bool = True,
    **kwargs,
):
    """
    Animate using OpenCV
    Args:
        rendering_fn: A function that takes in the time and the configuration and returns an image.
        t_ts: time steps of the data
        q_ts: configurations at each time step
        filepath: path to the output video
        speed_up: The speed up factor of the video.
        skip_step: The number of time steps to skip between animation frames.
        rgb_to_bgr: whether to convert the images from RGB to BGR
        **kwargs: Additional keyword arguments for the rendering function.

    Returns:

    """
    # extract parameters
    dt = onp.mean(onp.diff(t_ts)).item()
    fps = float(speed_up / (skip_step * dt))

    # create video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(
        str(filepath),
        fourcc,
        fps,  # fps,
        (width, height)
    )

    # skip frames
    t_ts = t_ts[::skip_step]
    q_ts = q_ts[::skip_step]

    for time_idx, t in enumerate(t_ts):
        img = rendering_fn(q_ts[time_idx], **kwargs)

        # convert to RBG if grayscale
        if img.shape[-1] == 1:
            img = onp.repeat(img, 3, axis=-1)

        if rgb_to_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        video.write(img)

    video.release()
    print(f"Video saved to {filepath}")
