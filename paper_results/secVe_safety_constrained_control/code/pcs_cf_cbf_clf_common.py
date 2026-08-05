from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from cbfpy.cbfs.clf_cbf import CLFCBFConfig
from jax import Array

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.systems import (
    PCS,
    LinkSpec,
    SystemState,
)

jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(
    threshold=10_000,
    linewidth=10_000,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)
jnp.set_printoptions(precision=4, suppress=True)

SECTION_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SECTION_DIR / "data"
OUTPUTS_DIR = SECTION_DIR / "outputs"

CONTROL_STRATEGIES = {
    "with-cbf": {"label": "with CBF", "slug": "with_cbf", "use_cbf": True},
    "without-cbf": {
        "label": "without CBF",
        "slug": "without_cbf",
        "use_cbf": False,
    },
}
CONTROLLER_SELECTIONS = {
    "both": ["with-cbf", "without-cbf"],
    "with-cbf": ["with-cbf"],
    "without-cbf": ["without-cbf"],
}


def controller_keys(selection: str) -> list[str]:
    return CONTROLLER_SELECTIONS[selection]


def trajectory_path(controller_key: str, data_dir: Path = DATA_DIR) -> Path:
    slug = CONTROL_STRATEGIES[controller_key]["slug"]
    return data_dir / f"rollout_{slug}.npz"


def system_parameters_path(data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "system_parameters.npz"


@jax.jit
def pairwise_h(
    p: jnp.ndarray,
    centers: jnp.ndarray,
    r_obs: jnp.ndarray,
    r_robot: jnp.ndarray | float = 0.0,
    safety: float = 0.0,
) -> jnp.ndarray:
    p = jnp.asarray(p)
    centers = jnp.asarray(centers)
    r_obs = jnp.asarray(r_obs)

    r_robot = jnp.asarray(r_robot)
    r_robot = r_robot + jnp.zeros((p.shape[0],), dtype=p.dtype)

    diff = p[:, None, :] - centers[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)
    return dist - (r_obs[None, :] + r_robot[:, None] + safety)


class SoftRobotDynamics:
    def __init__(self, robot: PCS):
        self.robot = robot
        self.n_q = int(robot.num_active_strains)
        self.n_u = int(robot.num_actuators)

    def split(self, y):
        return jnp.split(y, 2)

    @partial(jax.jit, static_argnums=(0,))
    def f(self, y):
        u_zero = jnp.zeros(self.n_u)
        return self.robot.forward_dynamics(jnp.asarray(0.0), y, (u_zero, None))

    @partial(jax.jit, static_argnums=(0,))
    def g(self, y):
        u_zero = jnp.zeros(self.n_u)
        return jax.jacfwd(
            lambda u: self.robot.forward_dynamics(jnp.asarray(0.0), y, (u, None))
        )(u_zero)


class ClosedFormHOCLFHOCBFController:
    """Sequential half-space projections: u0 -> HOCLF -> HOCBF."""

    def __init__(
        self,
        config: CLFCBFConfig,
        dynamics: SoftRobotDynamics,
        projection_eps: float = 1e-12,
    ):
        self.config = config
        self.dyn = dynamics
        self.projection_eps = projection_eps

    @staticmethod
    def _project_upper_halfspace(u: Array, a: Array, upper: Array, eps: float) -> Array:
        a = jnp.ravel(a)
        upper = jnp.ravel(upper)[0]
        violation = a @ u - upper
        denom = a @ a + eps
        return u - jnp.maximum(0.0, violation / denom) * a

    @staticmethod
    def _project_lower_halfspace(u: Array, a: Array, lower: Array, eps: float) -> Array:
        a = jnp.ravel(a)
        lower = jnp.ravel(lower)[0]
        violation = lower - a @ u
        denom = a @ a + eps
        return u + jnp.maximum(0.0, violation / denom) * a

    def _hoclf_state(self, z: Array, z_des: Array) -> Array:
        f_z = self.dyn.f(z)

        def V2_fn(state):
            return self.config.V_2(state, z_des)

        V2, V2_dot = jax.jvp(V2_fn, (z,), (f_z,))
        return V2_dot + self.config.gamma_2(V2)

    def _hocbf_state(self, z: Array) -> Array:
        f_z = self.dyn.f(z)
        h2, h2_dot = jax.jvp(self.config.h_2, (z,), (f_z,))
        return h2_dot + self.config.alpha_2(h2)

    @partial(jax.jit, static_argnums=(0,))
    def hoclf_constraint(self, z: Array, z_des: Array) -> tuple[Array, Array]:
        f_z = self.dyn.f(z)
        g_z = self.dyn.g(z)

        def psi_v_fn(state):
            return self._hoclf_state(state, z_des)

        psi_v, Lf_psi_v = jax.jvp(psi_v_fn, (z,), (f_z,))

        def Lg_col(g_col):
            return jax.jvp(psi_v_fn, (z,), (g_col,))[1]

        Lg_psi_v = jax.vmap(Lg_col, in_axes=1, out_axes=1)(g_z)
        upper = -Lf_psi_v - self.config.gamma(psi_v)
        return jnp.squeeze(Lg_psi_v, axis=0), upper

    @partial(jax.jit, static_argnums=(0,))
    def hocbf_constraint(self, z: Array) -> tuple[Array, Array]:
        f_z = self.dyn.f(z)
        g_z = self.dyn.g(z)

        psi_h, Lf_psi_h = jax.jvp(self._hocbf_state, (z,), (f_z,))

        def Lg_col(g_col):
            return jax.jvp(self._hocbf_state, (z,), (g_col,))[1]

        Lg_psi_h = jax.vmap(Lg_col, in_axes=1, out_axes=1)(g_z)
        lower = -Lf_psi_h - self.config.alpha(psi_h)
        return jnp.squeeze(Lg_psi_h, axis=0), lower

    @partial(jax.jit, static_argnums=(0,))
    def controller(self, z: Array, z_des: Array) -> Array:
        u0 = jnp.zeros(self.config.m)

        a_clf, upper_clf = self.hoclf_constraint(z, z_des)
        u_hoclf = self._project_upper_halfspace(
            u0, a_clf, upper_clf, self.projection_eps
        )

        a_cbf, lower_cbf = self.hocbf_constraint(z)
        return self._project_lower_halfspace(
            u_hoclf, a_cbf, lower_cbf, self.projection_eps
        )


@dataclass
class SimulationSetup:
    robot: PCS
    dyn: SoftRobotDynamics
    parameters: dict[str, Any]

    def make_config(self, use_cbf: bool = True) -> TendonSoroConfig:
        return TendonSoroConfig(self, use_cbf=use_cbf)


class TendonSoroConfig(CLFCBFConfig):
    def __init__(self, setup: SimulationSetup, use_cbf: bool = True):
        self.setup = setup
        self.use_cbf = use_cbf
        self.p_d_2 = setup.parameters["goal_position"]
        self.s_ps = setup.parameters["s_ps"]
        self.robot_radius = setup.parameters["robot_radius"]
        self.safety_margin = setup.parameters["safety_margin"]
        self.obs = {
            "centers": setup.parameters["obs_centers"],
            "radii": setup.parameters["obs_radii"],
        }

        super().__init__(
            n=2 * int(setup.robot.num_active_strains),
            m=int(setup.robot.num_actuators),
            relax_qp=False,
            cbf_relaxation_penalty=1e6,
            clf_relaxation_penalty=10,
        )

    def f(self, y) -> Array:
        return self.setup.dyn.f(y)

    def g(self, y) -> Array:
        return self.setup.dyn.g(y)

    def V_2(self, y, z_des=None) -> Array:
        q, _ = self.setup.dyn.split(y)
        g_ee_2 = self.setup.robot.forward_kinematics(q, jnp.sum(self.setup.robot.L))
        p_2 = g_ee_2[:3, 3]
        p_des = self.p_d_2 if z_des is None else z_des
        e_2 = jnp.linalg.norm(p_2 - p_des[:3])
        return e_2.reshape(-1)

    def h_2(self, y, kappa=2000.0):
        if not self.use_cbf:
            return jnp.array([0.0])

        q, _ = self.setup.dyn.split(y)
        g_ps = self.setup.robot.forward_kinematics_batched(q, self.s_ps)
        p = g_ps[:, :3, 3]

        H = pairwise_h(
            p,
            self.obs["centers"],
            self.obs["radii"],
            self.robot_radius,
            self.safety_margin,
        )

        return 5 + 1000 * jnp.min(H).reshape(-1)

    def alpha_2(self, h_2):
        return h_2 * 20.0

    def gamma_2(self, V_2):
        return V_2 * 10.0


def build_simulation_setup() -> SimulationSetup:
    num_segments = 2
    rho = 1070 * jnp.ones((num_segments,))
    goal_position = jnp.array([0.10, 0.05, 0.32])

    segment_length = 15e-2 * 2 / num_segments * jnp.ones((num_segments,))
    backbone_radius = 3.6e-2 * jnp.ones((num_segments,))
    damping_matrix = 1e-3 * jnp.diag(
        (
            jnp.repeat(
                jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]),
                num_segments,
                axis=0,
            )
            * segment_length[:, None]
        ).flatten()
    )

    r0, r1 = float(backbone_radius[0]) - 0.005, float(backbone_radius[1]) - 0.005
    theta0 = jnp.deg2rad(jnp.array([120, 240, 0]))
    theta1 = jnp.deg2rad(jnp.array([180, 300, 60]))
    ry0, rz0 = r0 * jnp.cos(theta0), r0 * jnp.sin(theta0)
    ry1, rz1 = r1 * jnp.cos(theta1), r1 * jnp.sin(theta1)

    tendon_ry = jnp.concatenate([ry0, ry1])
    tendon_rz = jnp.concatenate([rz0, rz1])
    tendon_my = jnp.zeros(6)
    tendon_mz = jnp.zeros(6)
    tendon_idx_seg_att = jnp.array([0, 0, 0, 1, 1, 1])

    strain_selector = (
        jnp.array([True, True, True, True, True, True])[None, :]
        .repeat(num_segments, axis=0)
        .flatten()
    )

    body_params = PCS.params_from_links(
        [
            LinkSpec.circular(
                length=float(segment_length[index]),
                radius=float(backbone_radius[index]),
                density=float(rho[index]),
                young_modulus=20e3,
                shear_modulus=20e3,
                damping=damping_matrix[
                    6 * index : 6 * (index + 1),
                    6 * index : 6 * (index + 1),
                ],
                reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            )
            for index in range(num_segments)
        ],
        base_pose=jnp.array([0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0]),
        gravity=jnp.array([0.0, 0.0, 9.81]),
    )

    active_tendon_routing = ThreadlikeRouting.linear(
        intercept=jnp.stack(
            (jnp.zeros_like(tendon_ry), tendon_ry, tendon_rz),
            axis=-1,
        ),
        slope=jnp.stack(
            (jnp.zeros_like(tendon_my), tendon_my, tendon_mz),
            axis=-1,
        ),
        start_segment_index=0,
        end_segment_index=tuple(int(index) for index in tendon_idx_seg_att),
    )

    robot = PCS(
        params=body_params,
        actuators=ThreadlikeActuator.tendons(active_tendon_routing),
    )

    q0 = jnp.repeat(
        jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])[None, :],
        num_segments,
        axis=0,
    ).flatten()
    qd0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, qd0])
    t0 = 0.0
    t1 = 8.0
    sim_dt = 1e-3
    save_dt = 1e-3

    parameters = {
        "num_segments": num_segments,
        "rho": rho,
        "goal_position": goal_position,
        "segment_length": segment_length,
        "backbone_radius": backbone_radius,
        "damping_matrix": damping_matrix,
        "tendon_ry": tendon_ry,
        "tendon_rz": tendon_rz,
        "tendon_my": tendon_my,
        "tendon_mz": tendon_mz,
        "tendon_idx_seg_att": tendon_idx_seg_att,
        "strain_selector": strain_selector,
        "base_pose": body_params.base_pose,
        "gravity": body_params.gravity,
        "young_modulus": 20e3 * jnp.ones((num_segments,)),
        "shear_modulus": 20e3 * jnp.ones((num_segments,)),
        "reference_strain": body_params.link.reference_strain,
        "obs_centers": jnp.array(
            [
                [0.10, 0.08, 0.24],
                [0.12, 0.06, 0.32],
                [0.04, 0.055, 0.20],
            ]
        ),
        "obs_radii": jnp.array([0.02, 0.02, 0.02]),
        "s_ps": jnp.linspace(0.0, jnp.sum(segment_length), 10 * num_segments),
        "robot_radius": backbone_radius[0],
        "safety_margin": jnp.array(0.00),
        "q0": q0,
        "qd0": qd0,
        "y0": y0,
        "t0": jnp.array(t0),
        "t1": jnp.array(t1),
        "sim_dt": jnp.array(sim_dt),
        "save_dt": jnp.array(save_dt),
        "force_limit": jnp.array(5.0),
    }
    return SimulationSetup(
        robot=robot, dyn=SoftRobotDynamics(robot), parameters=parameters
    )


def run_case(setup: SimulationSetup, controller_key: str) -> dict[str, Any]:
    strategy = CONTROL_STRATEGIES[controller_key]
    config = setup.make_config(use_cbf=bool(strategy["use_cbf"]))
    controller = ClosedFormHOCLFHOCBFController(config, setup.dyn)
    z_des = config.p_d_2

    def closed_loop_controller(state: SystemState):
        u = controller.controller(state.y, z_des)
        return u, None

    initial_state = SystemState(
        t=setup.parameters["t0"],
        y=setup.parameters["y0"],
        u=jnp.zeros((setup.robot.num_actuators,)),
        control_state=None,
    )

    trajectory = setup.robot.rollout_closed_loop_to(
        initial_state=initial_state,
        controller=closed_loop_controller,
        t1=float(setup.parameters["t1"]),
        solver_dt=float(setup.parameters["sim_dt"]),
        save_ts=jnp.arange(
            float(setup.parameters["t0"]),
            float(setup.parameters["t1"]),
            float(setup.parameters["save_dt"]),
        ),
        max_steps=None,
    )

    q_ts, qd_ts = jnp.split(trajectory.y, 2, axis=1)
    g_body_ts = jax.vmap(
        lambda q: setup.robot.forward_kinematics_batched(q, config.s_ps)
    )(q_ts)
    p_body_ts = g_body_ts[:, :, :3, 3]
    clearances = jnp.linalg.norm(
        p_body_ts[:, :, None, :] - config.obs["centers"][None, None, :, :],
        axis=-1,
    ) - (config.obs["radii"][None, None, :] + config.robot_radius)

    max_force = jnp.maximum(0.0, -1000.0 * clearances.min(axis=(1, 2)))
    forward_kinematics_end_effector = jax.jit(
        partial(setup.robot.forward_kinematics, s=jnp.sum(setup.robot.L))
    )
    g_ee_ts = jax.vmap(forward_kinematics_end_effector)(q_ts)
    goal_distance = jnp.linalg.norm(g_ee_ts[:, :3, 3] - z_des[None, :], axis=1)

    return {
        "controller_key": controller_key,
        "label": strategy["label"],
        "slug": strategy["slug"],
        "use_cbf": strategy["use_cbf"],
        "ts": np.asarray(trajectory.t),
        "y_ts": np.asarray(trajectory.y),
        "q_ts": np.asarray(q_ts),
        "qd_ts": np.asarray(qd_ts),
        "u_ts": np.asarray(trajectory.u),
        "force": np.asarray(max_force),
        "distance": np.asarray(goal_distance),
        "obs_centers": np.asarray(config.obs["centers"]),
        "obs_radii": np.asarray(config.obs["radii"]),
        "target_center": np.asarray(config.p_d_2),
        "s_ps": np.asarray(config.s_ps),
        "robot_radius": np.asarray(config.robot_radius),
        "safety_margin": np.asarray(config.safety_margin),
    }


def save_system_parameters(setup: SimulationSetup, data_dir: Path = DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = system_parameters_path(data_dir)
    np.savez_compressed(
        path,
        **{key: np.asarray(value) for key, value in setup.parameters.items()},
        num_active_strains=np.asarray(setup.robot.num_active_strains),
        num_actuators=np.asarray(setup.robot.num_actuators),
    )
    return path


def save_trajectory(result: dict[str, Any], data_dir: Path = DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = trajectory_path(str(result["controller_key"]), data_dir)
    np.savez_compressed(
        path,
        controller_key=np.asarray(result["controller_key"]),
        label=np.asarray(result["label"]),
        slug=np.asarray(result["slug"]),
        use_cbf=np.asarray(result["use_cbf"]),
        ts=result["ts"],
        y_ts=result["y_ts"],
        q_ts=result["q_ts"],
        qd_ts=result["qd_ts"],
        u_ts=result["u_ts"],
        force=result["force"],
        distance=result["distance"],
        obs_centers=result["obs_centers"],
        obs_radii=result["obs_radii"],
        target_center=result["target_center"],
        s_ps=result["s_ps"],
        robot_radius=result["robot_radius"],
        safety_margin=result["safety_margin"],
    )
    return path


def load_trajectory(path: Path) -> dict[str, Any]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def load_results(
    controller_selection: str = "both", data_dir: Path = DATA_DIR
) -> list[dict[str, Any]]:
    results = []
    for controller_key in controller_keys(controller_selection):
        path = trajectory_path(controller_key, data_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing trajectory file: {path}. Run control_pcs_with_cf_cbf_clf.py first."
            )
        result = load_trajectory(path)
        result["controller_key"] = str(result["controller_key"].item())
        result["label"] = str(result["label"].item())
        result["slug"] = str(result["slug"].item())
        result["use_cbf"] = bool(result["use_cbf"].item())
        results.append(result)
    return results


def load_system_parameters(data_dir: Path = DATA_DIR) -> dict[str, Any]:
    path = system_parameters_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing system parameter file: {path}. Run control_pcs_with_cf_cbf_clf.py first."
        )
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
