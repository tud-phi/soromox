"""Compare direct and packed cell indexing on identical FP64 inputs.

Uses synthetic cell fixtures with 2 or 8 segments, 7 cells per segment and
4 or 18 local basis columns. Both kernels use two threads per cell. Timing
excludes allocation and compilation, using seven CUDA graph replays of 200
launches each. Results describe the cell stage.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from soromox.execution.warp.gvs.cell import (
    GVSCellTermsData,
    cooperative_cell_terms_kernel,
    packed_cooperative_cell_terms_kernel,
)


def main() -> None:
    """Measure cell launch overhead and verify bitwise output equality.

    Returns:
        None. Writes median microseconds to the requested JSON report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, nargs="+", default=[32, 512])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wp.init()
    wp.set_device("cuda:0")
    report = []
    rng = np.random.default_rng(672)
    for worlds in args.worlds:
        for segments, width in ((2, 4), (8, 18)):
            cells = 7
            nc = segments * cells
            q = rng.normal(0, 0.001, (worlds * segments, width))
            arrays = [
                q,
                q * 0.2,
                rng.normal(0, 0.2, (nc, width)),
                rng.normal(0, 0.2, (nc, width)),
                np.broadcast_to(np.arange(width) % 6, (segments, width)).astype(
                    np.int32
                ),
                np.tile([0.0, 0, 0, 1, 0, 0], (nc, 1)),
                np.tile([0.0, 0, 0, 1, 0, 0], (nc, 1)),
                np.full(segments, 0.12),
                np.full(nc, 1 / cells),
                np.array([cells], np.int32),
                np.array([0], np.int32),
                np.zeros((worlds * nc * 6, 6)),
                np.zeros((worlds * nc * 6, width)),
                np.zeros((worlds * nc * 6, 1)),
                np.zeros((worlds * nc * 6, 1)),
                np.zeros((worlds * nc * 6, 1)),
            ]
            arrays = [
                wp.array(x, dtype=wp.int32 if x.dtype == np.int32 else wp.float64)
                for x in arrays
            ]
            owners = []
            fields = list(GVSCellTermsData.vars)
            for world in range(worlds):
                data = GVSCellTermsData()
                for i, (name, array) in enumerate(zip(fields, arrays, strict=True)):
                    count = segments if i < 2 else nc * 6 if i >= 11 else None
                    view = (
                        array[world * count : (world + 1) * count]
                        if count is not None
                        else array
                    )
                    setattr(data, name, view)
                owners.append(data)
            world_data = wp.array(owners, dtype=GVSCellTermsData)
            work = wp.array(
                [(w, c, 0) for w in range(worlds) for c in range(nc)], dtype=wp.vec3i
            )
            results = {}
            expected = None
            for label, kernel, dim, launch_args in [
                ("direct", cooperative_cell_terms_kernel, worlds * nc * 2, arrays),
                (
                    "packed",
                    packed_cooperative_cell_terms_kernel,
                    (worlds * nc, 2),
                    [world_data, work],
                ),
            ]:
                wp.launch(kernel, dim=dim, inputs=launch_args, block_dim=128)
                actual = [a.numpy() for a in arrays[11:]]
                if expected is None:
                    expected = actual
                else:
                    for a, b in zip(actual, expected, strict=True):
                        np.testing.assert_array_equal(a, b)
                with wp.ScopedCapture() as capture:
                    for _ in range(200):
                        wp.launch(kernel, dim=dim, inputs=launch_args, block_dim=128)
                wp.capture_launch(capture.graph)
                wp.synchronize()
                samples = []
                for _ in range(7):
                    start = time.perf_counter()
                    wp.capture_launch(capture.graph)
                    wp.synchronize()
                    samples.append((time.perf_counter() - start) * 1e6 / 200)
                results[label] = float(np.median(samples))
            report.append(
                dict(worlds=worlds, segments=segments, local_width=width, **results)
            )
            print(report[-1], flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
