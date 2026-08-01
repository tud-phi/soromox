"""Create the canonical Section Vc configuration-space figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from configuration_space_comparison_simulation import load_comparison_npz
from plot_configuration_space_comparison import (
    CANONICAL_INPUT,
    CANONICAL_OUTPUT,
    save_configuration_space_paper_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CANONICAL_INPUT)
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = load_comparison_npz(args.input)
    output = save_configuration_space_paper_figure(
        run, args.output, show=args.show, force=args.force
    )
    print(f"Figure written to {output}")


if __name__ == "__main__":
    main()
