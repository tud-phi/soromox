"""Exercise the actual PyElastica benchmark imports and force kernels."""

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

MODULE_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secIVa_benchmarking_sequential_cpu"
    / "code"
    / "pyelastica"
)


@pytest.fixture(scope="module")
def generators():
    sys.path.insert(0, str(MODULE_DIR))
    try:
        # Imports must not construct a rod, open a plot, or replace paper data.
        with (
            patch(
                "elastica.CosseratRod.straight_rod",
                side_effect=AssertionError("Simulation started during import"),
            ),
            patch(
                "matplotlib.pyplot.show",
                side_effect=AssertionError("Plot opened during import"),
            ),
            patch(
                "pandas.DataFrame.to_excel",
                side_effect=AssertionError("Paper data written during import"),
            ),
        ):
            modules = {
                case: importlib.import_module(f"simulate_{case}_pyelastica")
                for case in ("planar", "spatial", "tendon_driven")
            }
        yield modules
    finally:
        sys.path.remove(str(MODULE_DIR))


@pytest.mark.parametrize("case", ["planar", "spatial"])
def test_gravity_generator_constructs_and_integrates(generators, case):
    data = generators[case].simulate(final_time=0.005, dt=1e-4, step_skip=1)
    times = np.asarray(data["time"])
    positions = np.asarray(data["position"])

    assert positions.shape == (51, 3, 13)
    assert np.isfinite(positions).all()
    np.testing.assert_allclose(times, np.arange(51) * 1e-4, atol=1e-12)
    assert np.all(np.diff(times) > 0)
    # The gravity load must change the initially straight rod.
    assert np.linalg.norm(positions[-1, :, -1] - positions[0, :, -1]) > 1e-8


@pytest.mark.parametrize("collect_diagnostics", [False, True])
def test_tendon_generator_constructs_and_applies_forces(
    generators, collect_diagnostics
):
    module = generators["tendon_driven"]
    arm = module.BaseArmEnvElastica(
        sim_duration=0.005,
        dt=1e-4,
        COLLECT_DATA_FOR_PROCESSING=collect_diagnostics,
    )
    arm.reset()
    assert len(arm.tendon_groups) == 2

    positions, times = arm.simulate(np.array([1.0, 0.5]))

    assert positions.shape == (50, 3)
    assert np.isfinite(positions).all()
    assert np.isfinite(arm.arm.velocity_collection).all()
    np.testing.assert_allclose(times, np.arange(50) * 1e-4, atol=1e-12)
    assert np.all(np.diff(times) > 0)
    for group, force in zip(arm.tendon_groups, [1.0, 0.5]):
        muscle = group.muscles[0]
        np.testing.assert_allclose(muscle.muscle_force, force)
        np.testing.assert_array_equal(muscle.muscle_area, muscle.rest_muscle_area)
        assert np.isfinite(group.external_force).all()
        assert np.isfinite(group.external_couple).all()
        assert np.linalg.norm(group.external_force) > 0
        assert np.linalg.norm(group.external_couple) > 0

    if collect_diagnostics:
        assert arm.arm_post_processing["position"]
        for callback in arm.tendon_post_processing_list:
            assert callback["internal_force"]
            assert np.isfinite(np.asarray(callback["internal_force"])).all()
    else:
        assert all(
            not callback.get("internal_force")
            for callback in arm.tendon_post_processing_list
        )
