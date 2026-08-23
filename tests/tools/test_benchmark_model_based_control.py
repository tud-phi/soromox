from tools.benchmarks.benchmark_model_based_control import (
    _benchmark_cases,
    _system_registry,
    parse_args,
)


def test_registry_is_limited_to_controller_benchmark_systems():
    assert tuple(_system_registry()) == ("pcs", "gvs")


def test_cases_track_only_current_default_paths():
    names = {case.name for case in _benchmark_cases()}
    assert names == {
        "configuration_space_computed_torque",
        "configuration_space_feedforward_compensation",
        "actuation_space_feedforward_compensation",
        "operational_space_dynamics_terms",
    }
    assert all("separate" not in name and "fused" not in name for name in names)


def test_parse_args_normalizes_systems_and_selects_methods():
    args = parse_args(
        [
            "--systems",
            "PCS",
            "GVS",
            "--segment-counts",
            "2",
            "--gauss-points",
            "7",
            "--methods",
            "configuration_space_computed_torque",
            "--execution-repeats",
            "3",
        ]
    )

    assert args.systems == ["pcs", "gvs"]
    assert args.segment_counts == [2]
    assert args.gauss_points == [7]
    assert args.methods == ["configuration_space_computed_torque"]
    assert args.execution_repeats == 3
