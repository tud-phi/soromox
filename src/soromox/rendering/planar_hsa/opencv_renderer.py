import cv2  # importing cv2
from jax import Array, jit, vmap
import jax.numpy as jnp
import numpy as onp
from os import PathLike

from soromox.systems.planar_hsa import PlanarHSA

def draw_robot(
    robot: PlanarHSA,
    q: Array,
    width: int = 700,
    height: int = 700,
    num_points: int = 50,
    show: bool = False,
) -> onp.ndarray:
    """
    Draw the robot in OpenCV.
    Args:
        robot: PlanarHSA instance
        q: configuration as shape (3, )
        width: image width
        height: image height
        num_points: number of points to plot along the length of the robot
    """
    # plotting in OpenCV
    h, w = height, width  # img height and width
    ppm = h / (2.0 * jnp.sum(robot.lpc + robot.L + robot.ldc))  # pixel per meter
    base_color = (0, 0, 0)  # black base color in BGR
    backbone_color = (255, 0, 0)  # blue robot color in BGR
    rod_color = (0, 255, 0)  # green rod color in BGR
    platform_color = (0, 0, 255)  # red platform color in BGR

    batched_forward_kinematics_virtual_backbone_fn = vmap(
        robot.forward_kinematics_virtual_backbone_fn, in_axes=(None, 0), out_axes=-1
    )
    batched_forward_kinematics_rod_fn = vmap(
        robot.forward_kinematics_rod_fn, in_axes=(None, 0, None), out_axes=-1
    )
    batched_forward_kinematics_platform_fn = vmap(
        robot.forward_kinematics_platform_fn, in_axes=(None, 0), out_axes=0
    )

    # we use for plotting N points along the length of the robot
    s_ps = jnp.linspace(0, robot.Lmax, num_points)

    # poses along the robot of shape (3, N)
    chiv_ps = batched_forward_kinematics_virtual_backbone_fn(
        q, s_ps
    )  # poses of virtual backbone
    chiL_ps = batched_forward_kinematics_rod_fn(q, s_ps, 0)  # poses of left rod
    chiR_ps = batched_forward_kinematics_rod_fn(q, s_ps, 1)  # poses of left rod
    chip_ps = batched_forward_kinematics_platform_fn(
        q, jnp.arange(0, robot.num_segments)
    )  # poses of the platforms

    img = 255 * onp.ones((w, h, 3), dtype=jnp.uint8)  # initialize background to white
    uv_robot_origin = onp.array(
        [w // 2, h * (1 - 0.1)], dtype=jnp.int32
    )  # in x-y pixel coordinates
    uv_robot_origin_jax = jnp.array(uv_robot_origin)

    @jit
    def chi2u(chi: Array) -> Array:
        """
        Map Cartesian coordinates to pixel coordinates.
        Args:
            chi: Cartesian poses of shape (3)

        Returns:
            uv: pixel coordinates of shape (2)
        """
        uv_off = jnp.array((chi[1:] * ppm), dtype=jnp.int32)
        # invert the v pixel coordinate
        uv_off = uv_off.at[1].set(-uv_off[1])
        # invert the v pixel coordinate
        uv = uv_robot_origin_jax + uv_off
        return uv

    batched_chi2u = vmap(chi2u, in_axes=-1, out_axes=0)

    # draw base
    cv2.rectangle(img, (0, uv_robot_origin[1]), (w, h), color=base_color, thickness=-1)

    # draw the virtual backbone
    # add the first point of the proximal cap and the last point of the distal cap
    chiv_ps = jnp.concatenate(
        [
            (chiv_ps[:, 0] - jnp.array([0.0, 0.0, robot.lpc[0]])).reshape(3, 1),
            chiv_ps,
            (
                chiv_ps[:, -1]
                + jnp.array(
                    [
                        chiv_ps[0, -1],
                        -jnp.sin(chiv_ps[0, -1]) * robot.ldc[-1],
                        jnp.cos(chiv_ps[0, -1]) * robot.ldc[-1],
                    ]
                )
            ).reshape(3, 1),
        ],
        axis=1,
    )
    curve_virtual_backbone = onp.array(batched_chi2u(chiv_ps))
    cv2.polylines(
        img, [curve_virtual_backbone], isClosed=False, color=backbone_color, thickness=5
    )

    # draw the rods
    # add the first point of the proximal cap and the last point of the distal cap
    chiL_ps = jnp.concatenate(
        [
            (chiL_ps[:, 0] - jnp.array([0.0, 0.0, robot.lpc[0]])).reshape(3, 1),
            chiL_ps,
            (
                chiL_ps[:, -1]
                + jnp.array(
                    [
                        chiL_ps[0, -1],
                        -jnp.sin(chiL_ps[0, -1]) * robot.ldc[-1],
                        jnp.cos(chiL_ps[0, -1]) * robot.ldc[-1],
                    ]
                )
            ).reshape(3, 1),
        ],
        axis=1,
    )
    curve_rod_left = onp.array(batched_chi2u(chiL_ps))
    cv2.polylines(
        img,
        [curve_rod_left],
        isClosed=False,
        color=rod_color,
        thickness=10,
        # thickness=2*int(ppm * robot.params["rout"].mean(axis=0)[0])
    )
    # add the first point of the proximal cap and the last point of the distal cap
    chiR_ps = jnp.concatenate(
        [
            (chiR_ps[:, 0] - jnp.array([0.0, 0.0, robot.lpc[0]])).reshape(3, 1),
            chiR_ps,
            (
                chiR_ps[:, -1]
                + jnp.array(
                    [
                        chiR_ps[0, -1],
                        -jnp.sin(chiR_ps[0, -1]) * robot.ldc[-1],
                        jnp.cos(chiR_ps[0, -1]) * robot.ldc[-1],
                    ]
                )
            ).reshape(3, 1),
        ],
        axis=1,
    )
    curve_rod_right = onp.array(batched_chi2u(chiR_ps))
    cv2.polylines(img, [curve_rod_right], isClosed=False, color=rod_color, thickness=10)

    # draw the platform
    for i in range(chip_ps.shape[0]):
        # iterate over the platforms
        platform_R = jnp.array(
            [
                [1, 0, 0],
                [0, jnp.cos(chip_ps[i, 0]), -jnp.sin(chip_ps[i, 0])],
                [0, jnp.sin(chip_ps[i, 0]), jnp.cos(chip_ps[i, 0])],
            ]
        )  # rotation matrix for the platform
        platform_llc = chip_ps[i, :] + platform_R @ jnp.array(
            [
                0,
                -robot.pcudim[i, 0] / 2,  # go half the width to the left
                -robot.pcudim[i, 1] / 2,  # go half the height down
            ]
        )  # lower left corner of the platform
        platform_ulc = chip_ps[i, :] + platform_R @ jnp.array(
            [
                0,
                -robot.pcudim[i, 0] / 2,  # go half the width to the left
                +robot.pcudim[i, 1] / 2,  # go half the height down
            ]
        )  # upper left corner of the platform
        platform_urc = chip_ps[i, :] + platform_R @ jnp.array(
            [
                0,
                +robot.pcudim[i, 0] / 2,  # go half the width to the left
                +robot.pcudim[i, 1] / 2,  # go half the height down
            ]
        )  # upper right corner of the platform
        platform_lrc = chip_ps[i, :] + platform_R @ jnp.array(
            [
                0,
                +robot.pcudim[i, 0] / 2,  # go half the width to the left
                -robot.pcudim[i, 1] / 2,  # go half the height down
            ]
        )  # lower right corner of the platform
        platform_curve = jnp.stack(
            [platform_llc, platform_ulc, platform_urc, platform_lrc, platform_llc],
            axis=1,
        )
        # cv2.polylines(img, [onp.array(batched_chi2u(platform_curve))], isClosed=True, color=platform_color, thickness=5)
        cv2.fillPoly(
            img, [onp.array(batched_chi2u(platform_curve))], color=platform_color
        )

    if show:
        win = "Planar HSA"
        # resizable window (useful on macOS/Linux/HiDPI)
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, img)
        # wait until a key is pressed (close if ESC or 'q' is pressed)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyWindow(win)

    return img


def animate_robot(
    robot: PlanarHSA,
    filepath: PathLike,
    video_ts: Array,
    q_ts: Array,
    video_width: int = 700,
    video_height: int = 700,
):
    """
    Animate the robot and save the video.
    Args:
        filepath: path to save the video
        video_ts: time stamps of the video
        q_ts: configuration time series of shape (N, 3)
        video_width: video width
        video_height: video height
    """
    # create video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    video_dt = jnp.mean(video_ts[1:] - video_ts[:-1]).item()
    print(f"Rendering video with dt={video_dt} and {video_ts.shape[0]} frames")
    video = cv2.VideoWriter(
        str(filepath),
        fourcc,
        1 / video_dt,  # fps
        (video_width, video_height),
    )

    for time_idx, t in enumerate(video_ts):
        q = q_ts[time_idx]
        img = draw_robot(
            robot,
            q,
            video_width,
            video_height,
        )
        video.write(img)

    video.release()
