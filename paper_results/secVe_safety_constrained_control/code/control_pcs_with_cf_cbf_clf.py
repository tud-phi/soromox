#!/usr/bin/env python3
"""Run HOCLF/HOCBF simulations and export trajectory data."""

from __future__ import annotations

import argparse
from pathlib import Path

from pcs_cf_cbf_clf_common import (
    CONTROL_STRATEGIES,
    DATA_DIR,
    build_simulation_setup,
    controller_keys,
    run_case,
    save_system_parameters,
    save_trajectory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a tendon-actuated PCS with HOCLF/HOCBF control."
    )
    parser.add_argument(
        "--controller",
        choices=("both", "with-cbf", "without-cbf"),
        default="both",
        help="Controller mode to simulate.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory where .npz trajectory and parameter files are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup = build_simulation_setup()

    print("num_active_strains:", setup.robot.num_active_strains)
    print("num_actuators:", setup.robot.num_actuators)

    parameter_path = save_system_parameters(setup, args.data_dir)
    print(f"Saved system parameters to {parameter_path}")

    for controller_key in controller_keys(args.controller):
        label = CONTROL_STRATEGIES[controller_key]["label"]
        print(f"\nRunning {label} rollout via robot.rollout_closed_loop_to ...")
        result = run_case(setup, controller_key)
        trajectory_file = save_trajectory(result, args.data_dir)
        print(f"Saved {label} trajectory to {trajectory_file}")


if __name__ == "__main__":
    main()
