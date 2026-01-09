#!/usr/bin/env python3
"""Visualise Soromox benchmarking data stored as JSON or CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt

if shutil.which("latex"):
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Romand"],
        }
    )

NUMERIC_FIELDS = {
    "size_value": int,
    "jit_compile_time_s": float,
    "jit_execution_time_s": float,
    "duration": float,
    "dt": float,
    "save_dt": int,
    "execution_repeats": int,
}


def _coerce_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    data = dict(row)
    for key, caster in NUMERIC_FIELDS.items():
        if key in data and data[key] is not None:
            try:
                data[key] = caster(data[key])
            except (TypeError, ValueError):
                pass
    return data


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, list):
        raise ValueError("JSON input must be a list of benchmark entries")
    return [_coerce_row(entry) for entry in payload]


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = [_coerce_row(row) for row in reader]
    return rows


def load_results(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json(path)
    if suffix == ".csv":
        return _load_csv(path)
    raise ValueError(f"Unsupported input format '{suffix}'. Use JSON or CSV.")


def plot_results(
    results: Sequence[Mapping[str, Any]],
    output: Path | None,
    show: bool,
    title: str | None,
) -> None:
    if not results:
        raise ValueError("No benchmark entries available to plot")

    systems = sorted({row["system"] for row in results})
    fig, axes = plt.subplots(len(systems), 2, figsize=(12, 4 * len(systems)), squeeze=False, sharex="col")

    for row_idx, system in enumerate(systems):
        subset = [row for row in results if row["system"] == system]
        size_label = subset[0]["size_label"]
        functions = sorted({row["function"] for row in subset})
        ax_compile = axes[row_idx, 0]
        ax_exec = axes[row_idx, 1]
        for fn_name in functions:
            fn_rows = sorted(
                (row for row in subset if row["function"] == fn_name),
                key=lambda item: item["size_value"],
            )
            sizes = [row["size_value"] for row in fn_rows]
            compile_times = [row["jit_compile_time_s"] for row in fn_rows]
            exec_times = [row["jit_execution_time_s"] for row in fn_rows]

            ax_compile.plot(sizes, compile_times, marker="o", label=fn_name)
            ax_exec.plot(sizes, exec_times, marker="o", label=fn_name)

        ax_compile.set_title(f"{system} – compile")
        ax_exec.set_title(f"{system} – execution")
        ax_compile.set_ylabel("time [s]")
        ax_exec.set_ylabel("time [s]")
        ax_compile.set_xlabel(size_label)
        ax_exec.set_xlabel(size_label)
        ax_compile.grid(True, linestyle=":", linewidth=0.5)
        ax_exec.grid(True, linestyle=":", linewidth=0.5)
        ax_exec.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        print(f"Saved plot to {output}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Soromox benchmark results")
    parser.add_argument("input", type=Path, help="Path to the benchmark JSON or CSV file")
    parser.add_argument("--systems", nargs="*", help="Optional list of systems to include")
    parser.add_argument("--functions", nargs="*", help="Optional list of function names to include")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output image path (defaults to CSV input with .pdf suffix)",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot window")
    parser.add_argument("--title", help="Optional figure title")
    args = parser.parse_args(argv)
    if args.output is None and args.input.suffix.lower() == ".csv":
        args.output = args.input.with_suffix(".pdf")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file '{input_path}' does not exist")

    results = load_results(input_path)

    if args.systems:
        systems = set(args.systems)
        results = [row for row in results if row["system"] in systems]
    if args.functions:
        functions = set(args.functions)
        results = [row for row in results if row["function"] in functions]

    if not results:
        raise ValueError("No benchmark entries remain after applying filters")

    plot_results(results, args.output, args.show, args.title)

    if args.output:
        print(f"Visualisation saved to {args.output.resolve()}")
    elif not args.show:
        print("No figure rendered. Use --show or --output to produce a plot.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
