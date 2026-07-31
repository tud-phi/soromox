import sys
from pathlib import Path

import jax.numpy as jnp

MODULE_DIR = (
    Path(__file__).resolve().parents[4] / "examples" / "control" / "operational_space"
)

if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import pcs_impedance_example  # noqa: E402
import trajectory_primitives  # noqa: E402


def test_save_and_load_run_npz_preserves_rollout_and_config(tmp_path):
    config = trajectory_primitives.TaskSpaceTrajectoryConfig(
        tracking="pose",
        position_primitive="helix",
        orientation_primitive="surface-following",
        surface="sphere",
        period=4.0,
        ramp_time=0.75,
    )
    run = pcs_impedance_example.PCSImpedanceRun(
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

    output = pcs_impedance_example.save_run_npz(tmp_path / "rollout.npz", run)
    loaded = pcs_impedance_example.load_run_npz(output)

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
