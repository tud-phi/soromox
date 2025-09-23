from pathlib import Path

import soromox
from soromox.symbolic_derivation.planar_hsa import symbolically_derive_planar_hsa_model

NUM_SEGMENTS = 1
NUM_RODS_PER_SEGMENT = 2
if __name__ == "__main__":
    sym_exp_filepath = (
        Path(soromox.__file__).parent
        / "symbolic_expressions"
        / f"planar_hsa_ns-{NUM_SEGMENTS}_nrs-{NUM_RODS_PER_SEGMENT}.dill"
    )
    symbolically_derive_planar_hsa_model(
        num_segments=NUM_SEGMENTS,
        filepath=sym_exp_filepath,
        num_rods_per_segment=NUM_RODS_PER_SEGMENT,
        simplify=False,
    )
