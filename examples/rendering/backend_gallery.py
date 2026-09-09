"""Export the documentation's tentacle backend comparison in neutral studio lighting.

Run from the repository root with ``--backend matplotlib``, ``open3d``,
``opencv`` or ``viser``. Viser requires opening the printed URL in a browser.
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from tentacle_scene import make_comparison, make_tentacle

from soromox.rendering import (
    MatplotlibRenderer,
    Open3DRenderer,
    OpenCVPlanarRenderer,
    ViserRenderer,
)
from soromox.systems import LinkSpec, PlanarPCS


def make_scene(planar: bool = False):
    """Prepare one upright tentacle with neutral studio settings.

    Args:
        planar: Use a two-link planar PCS counterpart for OpenCV.

    Returns:
        Robot, strain displacement vector, shared rendering configuration and
        base offsets.
        The planar counterpart shares lengths and bending coordinates; OpenCV
        represents its backbone with a line. Geometry lengths are in metres.
    """
    config, q, offsets = make_comparison("neutral", count=1, width=1200, height=800)
    config.geometry.line_width = 16.0
    if not planar:
        return make_tentacle(), q[0], config, offsets
    config.scene.ground.height_reference = "world"
    robot = PlanarPCS.from_links(
        [
            LinkSpec.circular(
                length=length,
                radius=radius,
                density=1500.0,
                young_modulus=5.05e5,
                poisson_ratio=0.45,
                reference_strain=[0, 1, 0],
            )
            for length, radius in ((0.305, 0.01541), (0.055, 0.00642))
        ]
    )
    return robot, np.array([3.6, 0, 0, -7.5, 0, 0]), config, offsets


def main() -> None:
    """Export one backend's gallery PNG using the public renderer API.

    Returns:
        None. Writes an RGB PNG to the requested directory.

    Raises:
        RuntimeError: No Viser browser connects within five minutes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", required=True, choices=("matplotlib", "open3d", "opencv", "viser")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs/assets/rendering",
    )
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    planar = args.backend == "opencv"
    robot, q, config, offsets = make_scene(planar)
    render_offsets = offsets[:, :2] if planar else offsets
    if planar:
        renderer = OpenCVPlanarRenderer(
            robot, config=config, length_scale=1.5, origin_uv=(600, 690)
        )
    elif args.backend == "viser":
        renderer = ViserRenderer(
            robot, config=config, host="127.0.0.1", port=args.port, open_browser=False
        )
    else:
        renderer_type = (
            MatplotlibRenderer if args.backend == "matplotlib" else Open3DRenderer
        )
        renderer = renderer_type(robot, config=config)
    try:
        if args.backend == "viser":
            print(f"Open {renderer.url} to capture the documentation image", flush=True)
            deadline = time.monotonic() + 300
            while not renderer.server.get_clients():
                if time.monotonic() >= deadline:
                    raise RuntimeError("No browser connected within five minutes")
                time.sleep(0.2)
        pixels = renderer.render_frame(
            jnp.asarray(q),
            base_offsets=jnp.asarray(render_offsets),
            render_actuators=False,
        )
        if planar:
            pixels = pixels[..., ::-1]  # OpenCV exports BGR; PNG uses RGB.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        name = "opencv-planar" if planar else args.backend
        path = args.output_dir / f"{name}.png"
        Image.fromarray(pixels).save(path)
        print(f"Saved {path}", flush=True)
    finally:
        if args.backend == "viser":
            renderer.stop()


if __name__ == "__main__":
    main()
