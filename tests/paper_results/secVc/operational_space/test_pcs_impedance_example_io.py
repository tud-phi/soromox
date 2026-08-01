import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

MODULE_DIR = (
    Path(__file__).resolve().parents[4]
    / "paper_results"
    / "secVc_model_based_control"
    / "operational_space_impedance_control"
    / "code"
)

if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import operational_space_impedance_common  # noqa: E402
import trajectory_primitives  # noqa: E402

ENTRY_POINTS = (
    "control_pcs_with_impedance.py",
    "plot_control_pcs_with_impedance.py",
    "plot_operational_space_impedance_paper_figure.py",
    "compare_impedance_feedback_linearization.py",
    "plot_impedance_feedback_linearization.py",
)


def test_save_and_load_run_npz_preserves_rollout_and_config(tmp_path):
    config = trajectory_primitives.TaskSpaceTrajectoryConfig(
        tracking="pose",
        position_primitive="helix",
        orientation_primitive="surface-following",
        surface="sphere",
        period=4.0,
        ramp_time=0.75,
    )
    run = operational_space_impedance_common.PCSImpedanceRun(
        trajectory_config=config,
        t1=0.2,
        solver_dt=1e-4,
        save_dt=0.01,
        t_traj=jnp.array([0.0, 0.1, 0.2]),
        q_traj=jnp.ones((3, 2)),
        qd_traj=2.0 * jnp.ones((3, 2)),
        u_traj=3.0 * jnp.ones((3, 4)),
        x_traj_full=4.0 * jnp.ones((3, 6)),
        x_des_traj_full=5.0 * jnp.ones((3, 6)),
    )

    output = operational_space_impedance_common.save_run_npz(
        tmp_path / "rollout.npz", run
    )
    loaded = operational_space_impedance_common.load_run_npz(output)

    assert loaded.trajectory_config == config
    assert loaded.t1 == run.t1
    assert loaded.solver_dt == run.solver_dt
    assert loaded.save_dt == run.save_dt
    assert jnp.allclose(loaded.t_traj, run.t_traj)
    assert jnp.allclose(loaded.q_traj, run.q_traj)
    assert jnp.allclose(loaded.qd_traj, run.qd_traj)
    assert jnp.allclose(loaded.u_traj, run.u_traj)
    assert jnp.allclose(loaded.x_traj_full, run.x_traj_full)
    assert jnp.allclose(loaded.x_des_traj_full, run.x_des_traj_full)


def test_save_run_npz_requires_force_to_overwrite(tmp_path):
    output = tmp_path / "rollout.npz"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="--force"):
        operational_space_impedance_common.save_run_npz(output, object())


def test_default_paths_are_case_relative():
    case_dir = MODULE_DIR.parent

    assert (
        case_dir / "data" / "control_pcs_with_impedance_trajectory.npz"
    ) == operational_space_impedance_common.DEFAULT_TRAJECTORY_OUTPUT
    assert case_dir / "outputs" == operational_space_impedance_common.OUTPUTS_DIR


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_cli_starts_from_unrelated_working_directory(tmp_path, entry_point):
    subprocess.run(
        [sys.executable, str(MODULE_DIR / entry_point), "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
