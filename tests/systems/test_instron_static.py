"""Sequential force-controlled static validation: loads, constraints, solver, stepping."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples.simulation.gvs import instron_static as ist

jax.config.update("jax_enable_x64", True)


def _branch(n: int = 50, peak: float = 10.0) -> ist.InstronBranch:
    t = np.linspace(0.0, 1.0, n)
    return ist.InstronBranch(
        source="synthetic", cycle=1, branch="loading", rows=(0, n - 1),
        time_s=t, extension_mm=60.0 * t, load_N=peak * t,
    )


@pytest.fixture(scope="module")
def compression():
    return ist.CompressionExperiment()


@pytest.fixture(scope="module")
def bending():
    return ist.BendingExperiment()


# --------------------------------------------------------------------------- #
# Instron CSV and force schedule
# --------------------------------------------------------------------------- #
def test_cycle_detection_and_branches():
    t = np.arange(600) * 0.01
    ext = np.abs(((t * 2.0 + 1.0) % 4.0) - 2.0)  # triangle wave of 3 peaks
    cycles = ist.detect_cycles(ext)
    assert len(cycles) == 3
    assert ext[cycles[0]["peak"]] == pytest.approx(ext.max(), abs=0.02)
    data = {"time_s": t, "extension_mm": ext, "load_N": 2.0 * ext}
    loading, unloading, cycle = ist.branches(data, 2, "synthetic")
    assert cycle["index"] == 2
    assert loading.rows == (cycles[1]["start"], cycles[1]["peak"])
    assert unloading.rows == (cycles[1]["peak"], cycles[1]["end"])
    assert loading.num_samples + unloading.num_samples == cycles[1]["end"] - cycles[1]["start"] + 2


def test_samples_spread_along_extension_and_keep_order():
    loading = _branch()
    samples = ist.select_samples(loading, num_levels=10)
    assert 2 <= len(samples) <= 10
    assert [s.index for s in samples] == list(range(len(samples)))
    forces = [s.force_N for s in samples]
    assert forces == sorted(forces)
    assert samples[0].measured_displacement_mm == pytest.approx(0.0)
    assert samples[-1].measured_displacement_mm == pytest.approx(60.0)
    capped = ist.select_samples(loading, num_levels=10, max_force=5.0)
    assert max(s.force_N for s in capped) <= 5.0
    every = ist.select_samples(loading, decimate=7)
    assert every[-1].row == 49 and every[0].row == 0


@pytest.mark.skipif(not ist.COMPRESSION_CSV.exists() or not ist.BENDING_CSV.exists(), reason="Instron CSVs not present")
def test_read_instron_csvs():
    comp = ist.read_instron_csv(ist.COMPRESSION_CSV)
    bend = ist.read_instron_csv(ist.BENDING_CSV)
    assert comp["units"] == {"Time": "s", "Extension": "mm", "Load": "N"}
    assert len(ist.detect_cycles(comp["extension_mm"])) == 6
    assert len(ist.detect_cycles(bend["extension_mm"])) == 6
    loading, unloading, _ = ist.load_branches(ist.COMPRESSION_CSV)
    assert loading.rows[0] == 0 and loading.extension_mm.max() > 60
    assert unloading.extension_mm[-1] < 1.0


# --------------------------------------------------------------------------- #
# Loads and constraints
# --------------------------------------------------------------------------- #
def test_compression_wrench_and_platen_constraints(compression):
    exp = compression
    q0 = jnp.zeros(exp.n)
    w = np.asarray(exp.wrench(q0, jnp.asarray(3.0), jnp.zeros(5)))
    assert_allclose(w, [0, 0, 0, 0, 0, -3.0])  # positive force = downward load
    w = np.asarray(exp.wrench(q0, jnp.asarray(3.0), jnp.arange(1.0, 6.0)))
    assert_allclose(w[:5], [1, 2, 3, 4, 5])  # reactions fill the five tied components
    assert exp.num_constraints == exp.num_wrench_unknowns == 5
    assert_allclose(np.asarray(exp.constraints(q0)), 0.0, atol=1e-12)
    solver = ist.StaticSolver(exp)
    expected = (
        exp.robot.elastic_force(q0)
        + exp.robot.gravitational_force(q0)
        - exp.robot.jacobian_tips(q0)[-1].T @ jnp.asarray(w)
    )
    assert_allclose(np.asarray(solver.generalized_residual(q0, jnp.asarray(w))), np.asarray(expected))


def test_bending_load_is_tension_only_with_string_direction_and_moment_arm(bending):
    exp = bending
    q0 = jnp.zeros(exp.n)
    none = jnp.zeros(0)
    assert float(exp.tension(jnp.asarray(-2.0))) == 0.0
    assert_allclose(np.asarray(exp.wrench(q0, jnp.asarray(-2.0), none)), 0.0)
    w = np.asarray(exp.wrench(q0, jnp.asarray(2.0), none))
    # String pulls toward the clamped ring (-x) from a hole offset along +z:
    # force (-T, 0, 0), moment (0, 0, r) x (-T, 0, 0) = (0, -r T, 0).
    assert_allclose(w[3:], [-2.0, 0.0, 0.0], atol=1e-12)
    assert_allclose(w[:3], [0.0, -2.0 * exp.bending_config.tendon_offset, 0.0], atol=1e-12)
    assert exp.num_constraints == 0 and exp.string_points(np.zeros(exp.n)).shape == (5, 3)


# --------------------------------------------------------------------------- #
# Static solver
# --------------------------------------------------------------------------- #
def test_compression_small_force_matches_lattice_axial_stiffness(compression):
    exp = compression
    solver = ist.StaticSolver(exp)
    x = solver.initial_guess()
    displacement = []
    for force in (0.0, 0.1, 0.2):
        x, ok, _, norm = solver.solve(x, force)
        assert ok and norm < 1e-9
        q, wrench = solver.unpack(x, force)
        check = solver.verify(q, wrench)
        assert check["force_residual_rel"] <= ist.FORCE_RESIDUAL_REL_TOL or force == 0.0
        assert check["constraint_residual"] <= ist.CONSTRAINT_TOL
        assert check["min_tangent_eigenvalue"] > 0.0
        assert_allclose(wrench[:5], 0.0, atol=1e-9)  # axial load: no platen reactions
        displacement.append(exp.measure(q)["compression_m"])
    assert displacement[0] == pytest.approx(0.0, abs=1e-12)
    lattice = 1 / float(np.asarray(exp.helix.reduction_diagnostics()["tip_compliance"])[3, 3])
    assert 0.1 / (displacement[1] - displacement[0]) == pytest.approx(lattice, rel=1e-3)


def test_bending_level_raises_the_free_ring_and_shortens_the_string(bending):
    exp = bending
    solver = ist.StaticSolver(exp)
    x, ok, _, norm = solver.solve(solver.initial_guess(), 0.5)
    assert ok and norm < 1e-9
    q, wrench = solver.unpack(x, 0.5)
    check = solver.verify(q, wrench)
    assert check["force_residual_rel"] <= ist.FORCE_RESIDUAL_REL_TOL
    assert check["min_tangent_eigenvalue"] > 0.0
    m = exp.measure(q)
    assert m["travel_m"] > 0.0 and m["ring_dz_m"] > 0.0 and m["bending_angle_rad"] > 0.0
    assert abs(m["ring_dy_m"]) < 1e-9  # in-plane bending
    assert m["lever_arm_m"] == pytest.approx(exp.bending_config.tendon_offset, rel=0.05)
    assert exp.validity(q, m) == (True, "")


# --------------------------------------------------------------------------- #
# Force stepping
# --------------------------------------------------------------------------- #
def test_step_forces_records_verified_equilibria_in_order(compression):
    samples = ist.select_samples(_branch(peak=6.0), num_levels=4)
    records, stop = ist.step_forces(compression, samples, log=None)
    assert stop == "" and len(records) == len(samples)
    assert all(r.converged and r.valid for r in records)
    x = [r.primary for r in records]
    assert x[0] == pytest.approx(0.0, abs=1e-12) and np.all(np.diff(x) > 0)
    row = records[-1].to_row(compression.primary_label)
    assert row["static_displacement_mm"] == pytest.approx(1e3 * x[-1])
    assert row["wrench_f_z"] == pytest.approx(-6.0)
    assert math.isfinite(row["min_tangent_eigenvalue"])


def test_step_forces_stops_at_the_first_nonphysical_level(compression):
    # 60 N drives the linear model past the base: flagged, not plotted as an equilibrium.
    samples = ist.select_samples(_branch(peak=120.0), num_levels=3)
    records, stop = ist.step_forces(compression, samples, log=None)
    assert any(not r.valid for r in records)
    assert "tip z <= 0" in stop or "no verified" in stop
    assert len(records) < len(samples) or not records[-1].valid
