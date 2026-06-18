import argparse
import os
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from cbfpy.cbfs.clf_cbf import CLFCBFConfig
from jax import Array

plt.rcParams["pdf.fonttype"] = 42  # For editable text in Illustrator
from soromox.rendering import Open3DRenderer, RendererColorConfig
from soromox.systems import (
    LinearTendonRoutingParams,
    PCSParams,
    SystemState,
    TendonActuatedPCS,
    TendonActuatedPCSParams,
)

jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(
    threshold=10_000,
    linewidth=10_000,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)
jnp.set_printoptions(precision=4, suppress=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a tendon-actuated PCS with HOCLF/HOCBF control."
    )
    parser.add_argument(
        "--controller",
        choices=("both", "with-cbf", "without-cbf"),
        default="both",
        help=(
            "Controller mode to simulate. 'both' reproduces the comparison plot; "
            "the other modes run a single controller."
        ),
    )
    return parser.parse_args()


# ======================================================
# Distance-based barrier
# ======================================================
@jax.jit
def pairwise_h(
    p: jnp.ndarray,  # (P, 3)
    centers: jnp.ndarray,  # (N, 3)
    r_obs: jnp.ndarray,  # (N,)
    r_robot: jnp.ndarray | float = 0.0,  # scalar or (P,)
    safety: float = 0.0,
) -> jnp.ndarray:
    p = jnp.asarray(p)
    centers = jnp.asarray(centers)
    r_obs = jnp.asarray(r_obs)

    r_robot = jnp.asarray(r_robot)
    r_robot = r_robot + jnp.zeros((p.shape[0],), dtype=p.dtype)

    diff = p[:, None, :] - centers[None, :, :]  # (P, N, 3)
    dist = jnp.linalg.norm(diff, axis=-1)  # (P, N)
    H = dist - (r_obs[None, :] + r_robot[:, None] + safety)
    return H


# ======================================================
# Real dynamics wrapper (from file 1 style)
# y = [q, qd]
# ======================================================
class SoftRobotDynamics:
    def __init__(self, robot: TendonActuatedPCS):
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


# ======================================================
# Closed-form HOCLF-HOCBF controller
# ======================================================
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
        """Project onto a @ u <= upper."""
        a = jnp.ravel(a)
        upper = jnp.ravel(upper)[0]
        violation = a @ u - upper
        denom = a @ a + eps
        return u - jnp.maximum(0.0, violation / denom) * a

    @staticmethod
    def _project_lower_halfspace(u: Array, a: Array, lower: Array, eps: float) -> Array:
        """Project onto a @ u >= lower."""
        a = jnp.ravel(a)
        lower = jnp.ravel(lower)[0]
        violation = lower - a @ u
        denom = a @ a + eps
        return u + jnp.maximum(0.0, violation / denom) * a

    def _hoclf_state(self, z: Array, z_des: Array) -> Array:
        """psi_V = V2_dot + gamma2(V2)."""
        f_z = self.dyn.f(z)

        def V2_fn(state):
            return self.config.V_2(state, z_des)

        V2, V2_dot = jax.jvp(V2_fn, (z,), (f_z,))
        return V2_dot + self.config.gamma_2(V2)

    def _hocbf_state(self, z: Array) -> Array:
        """psi_h = h2_dot + alpha2(h2)."""
        f_z = self.dyn.f(z)
        h2, h2_dot = jax.jvp(self.config.h_2, (z,), (f_z,))
        return h2_dot + self.config.alpha_2(h2)

    @partial(jax.jit, static_argnums=(0,))
    def hoclf_constraint(self, z: Array, z_des: Array) -> tuple[Array, Array]:
        """Return a, upper for the HOCLF constraint a @ u <= upper."""
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
        """Return a, lower for the HOCBF constraint a @ u >= lower."""
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
        """Closed-form controller: zero input -> HOCLF projection -> HOCBF projection."""
        u0 = jnp.zeros(self.config.m)

        a_clf, upper_clf = self.hoclf_constraint(z, z_des)
        u_hoclf = self._project_upper_halfspace(
            u0, a_clf, upper_clf, self.projection_eps
        )

        a_cbf, lower_cbf = self.hocbf_constraint(z)
        u_safe = self._project_lower_halfspace(
            u_hoclf, a_cbf, lower_cbf, self.projection_eps
        )
        return u_safe


if __name__ == "__main__":
    args = parse_args()
    num_segments = 2
    rho = 1070 * jnp.ones((num_segments,))
    goal_position = jnp.array([0.10, 0.05, 0.32])

    # --------------------------------------------------
    # Continuum body parameters
    # --------------------------------------------------
    # The robot is modeled as a two-segment piecewise-constant-strain body.
    # Each vector entry corresponds to one segment.
    segment_length = 15e-2 * 2 / num_segments * jnp.ones((num_segments,))
    backbone_radius = 3.6e-2 * jnp.ones((num_segments,))

    # Diagonal strain damping matrix. Translational strain damping is kept
    # lower than rotational/shear-like entries, then scaled by segment length.
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

    # --------------------------------------------------
    # Tendon routing parameters
    # --------------------------------------------------
    # Three straight tendons are attached to each segment. The y/z intercepts
    # place each tendon around the backbone cross-section, slightly inside the
    # outer radius.
    r0, r1 = float(backbone_radius[0]) - 0.005, float(backbone_radius[1]) - 0.005
    theta0 = jnp.deg2rad(jnp.array([120, 240, 0]))
    theta1 = jnp.deg2rad(jnp.array([180, 300, 60]))
    ry0, rz0 = r0 * jnp.cos(theta0), r0 * jnp.sin(theta0)
    ry1, rz1 = r1 * jnp.cos(theta1), r1 * jnp.sin(theta1)

    tendon_routing_params = {
        "ry": jnp.concatenate([ry0, ry1]),
        "rz": jnp.concatenate([rz0, rz1]),
        "my": jnp.zeros(6),
        "mz": jnp.zeros(6),
        "idx_seg_att": jnp.array([0, 0, 0, 1, 1, 1]),
    }

    strain_selector = (
        jnp.array([True, True, True, True, True, True])[None, :]
        .repeat(num_segments, axis=0)
        .flatten()
    )

    # --------------------------------------------------
    # soromox system parameter objects
    # --------------------------------------------------
    body_params = PCSParams(
        base_pose=jnp.array([0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0]),
        length=segment_length,
        radius=backbone_radius,
        density=rho,
        gravity=jnp.array([0.0, 0.0, 9.81]),
        young_modulus=20e3 * jnp.ones((num_segments,)),
        shear_modulus=20e3 * jnp.ones((num_segments,)),
        damping_matrix=damping_matrix,
        reference_strain=jnp.tile(
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), num_segments
        ),
    )

    active_tendon_routing = LinearTendonRoutingParams(
        y_intercept=tendon_routing_params["ry"],
        z_intercept=tendon_routing_params["rz"],
        y_slope=tendon_routing_params["my"],
        z_slope=tendon_routing_params["mz"],
        attachment_segment_index=tendon_routing_params["idx_seg_att"],
    )

    robot = TendonActuatedPCS(
        params=TendonActuatedPCSParams(
            body=body_params,
            active_tendon_routing=active_tendon_routing,
        )
    )

    print("num_active_strains:", robot.num_active_strains)
    print("num_actuators:", robot.num_actuators)

    dyn = SoftRobotDynamics(robot)

    # ======================================================
    # CLF-CBF config with REAL dynamics + V2/h2
    # ======================================================
    class TendonSoroConfig(CLFCBFConfig):
        def __init__(self, use_cbf=True):
            self.use_cbf = use_cbf
            self.p_d_2 = goal_position

            self.s_ps = jnp.linspace(0.0, jnp.sum(segment_length), 10 * num_segments)

            self.robot_radius = backbone_radius[0]
            self.safety_margin = 0.00

            self.obs = {
                "centers": jnp.array(
                    [
                        [0.10, 0.08, 0.24],
                        [0.12, 0.06, 0.32],
                        [0.04, 0.055, 0.20],
                    ]
                ),
                "radii": jnp.array([0.02, 0.02, 0.02]),
            }

            super().__init__(
                n=2 * int(robot.num_active_strains),  # y = [q, qd]
                m=int(robot.num_actuators),
                relax_qp=False,
                cbf_relaxation_penalty=1e6,
                clf_relaxation_penalty=10,
            )

        # --------------------------------------------------
        # Real dynamics from file 1
        # --------------------------------------------------
        def f(self, y) -> Array:
            return dyn.f(y)

        def g(self, y) -> Array:
            return dyn.g(y)

        # --------------------------------------------------
        # CLF #2
        # --------------------------------------------------
        def V_2(self, y, z_des=None) -> Array:
            q, qd = dyn.split(y)
            g_ee_2 = robot.forward_kinematics(q, jnp.sum(robot.L))
            p_2 = g_ee_2[:3, 3]
            p_des = self.p_d_2 if z_des is None else z_des
            e_2 = jnp.linalg.norm(p_2 - p_des[:3])
            return e_2.reshape(-1)

        # --------------------------------------------------
        # CBF #2
        # --------------------------------------------------
        def h_2(self, y, kappa=2000.0):
            if not self.use_cbf:
                return jnp.array([0.0])

            q, qd = dyn.split(y)
            g_ps = robot.forward_kinematics_batched(q, self.s_ps)  # (P,4,4)
            p = g_ps[:, :3, 3]  # (P,3)

            H = pairwise_h(
                p,
                self.obs["centers"],
                self.obs["radii"],
                self.robot_radius,
                self.safety_margin,
            )

            force_min = 5 + 1000 * jnp.min(H).reshape(-1)
            return force_min

        def alpha_2(self, h_2):
            return h_2 * 20.0

        def gamma_2(self, V_2):
            return V_2 * 10.0

    # ======================================================
    # Initial state
    # ======================================================
    q0 = jnp.repeat(
        jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])[None, :],
        num_segments,
        axis=0,
    ).flatten()

    qd0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, qd0])

    # ======================================================
    # Simulation
    # ======================================================
    t0 = 0.0
    t1 = 8.0
    sim_dt = 1e-3
    save_dt = 1e-3

    forward_kinematics_end_effector = jax.jit(
        partial(robot.forward_kinematics, s=jnp.sum(robot.L))
    )

    def run_case(use_cbf: bool, label: str):
        config = TendonSoroConfig(use_cbf=use_cbf)
        controller = ClosedFormHOCLFHOCBFController(config, dyn)
        z_des = config.p_d_2

        def closed_loop_controller(state: SystemState):
            u = controller.controller(state.y, z_des)
            return u, None

        initial_state = SystemState(
            t=jnp.array(t0),
            y=y0,
            u=jnp.zeros((robot.num_actuators,)),
            control_state=None,
        )

        print(f"\nRunning {label} rollout via robot.rollout_closed_loop_to ...")
        trajectory = robot.rollout_closed_loop_to(
            initial_state=initial_state,
            controller=closed_loop_controller,
            t1=t1,
            solver_dt=sim_dt,
            save_ts=jnp.arange(t0, t1, save_dt),
            max_steps=None,
        )

        q_ts, _ = jnp.split(trajectory.y, 2, axis=1)
        g_body_ts = jax.vmap(
            lambda q: robot.forward_kinematics_batched(q, config.s_ps)
        )(q_ts)
        p_body_ts = g_body_ts[:, :, :3, 3]
        clearances = jnp.linalg.norm(
            p_body_ts[:, :, None, :] - config.obs["centers"][None, None, :, :],
            axis=-1,
        ) - (config.obs["radii"][None, None, :] + config.robot_radius)

        # The contact model uses 1000 N/m after the clearance becomes negative.
        max_force = jnp.maximum(0.0, -1000.0 * clearances.min(axis=(1, 2)))

        g_ee_ts = jax.vmap(forward_kinematics_end_effector)(q_ts)
        goal_distance = jnp.linalg.norm(g_ee_ts[:, :3, 3] - z_des[None, :], axis=1)
        return {
            "label": label,
            "ts": onp.asarray(trajectory.t),
            "q_ts": q_ts,
            "config": config,
            "force": onp.asarray(max_force),
            "distance": onp.asarray(goal_distance),
        }

    selected_modes = {
        "both": [("with CBF", True), ("without CBF", False)],
        "with-cbf": [("with CBF", True)],
        "without-cbf": [("without CBF", False)],
    }[args.controller]

    results = [run_case(use_cbf, label) for label, use_cbf in selected_modes]

    # ======================================================
    # Open3D render
    # ======================================================
    if Open3DRenderer is None:
        print("\nOpen3DRenderer unavailable. Install open3d to view the animation.")
    else:
        os.makedirs("videos", exist_ok=True)

        render_result = next(
            (result for result in results if result["label"] == "with CBF"),
            results[0],
        )
        render_config = render_result["config"]
        render_ts = render_result["ts"]
        render_q_ts = render_result["q_ts"]

        obs_centers = onp.asarray(render_config.obs["centers"], dtype=onp.float64)
        obs_radii = onp.asarray(render_config.obs["radii"], dtype=onp.float64)
        obs_colors = onp.tile(
            onp.array([[0.5, 0.5, 0.5]], dtype=onp.float64),
            (obs_radii.shape[0], 1),
        )

        target_center = onp.asarray(render_config.p_d_2, dtype=onp.float64).reshape(
            1, 3
        )
        target_radius = onp.array([0.01], dtype=onp.float64)
        target_color = onp.array([[1.0, 0.1, 0.1]], dtype=onp.float64)

        static_spheres_positions = onp.concatenate([obs_centers, target_center], axis=0)
        static_spheres_radii = onp.concatenate([obs_radii, target_radius], axis=0)
        static_spheres_colors = onp.concatenate([obs_colors, target_color], axis=0)

        render_stride = max(1, len(render_ts) // 300)
        renderer = Open3DRenderer(
            robot,
            num_points=80,
            color_config=RendererColorConfig(),
            width=1280,
            height=800,
            backbone_style="discrete",
            actuator_line_width=2.0,
        )
        renderer.render_sequence(
            ts=render_ts[::render_stride],
            q_ts=render_q_ts[::render_stride],
            playback_speed=1.0,
            loop=True,
            record_path=f"videos/clf_cbf_rollout_{render_result['label'].lower().replace(' ', '_')}.mp4",
            render_actuators=True,
            static_spheres_positions=static_spheres_positions,
            static_spheres_radii=static_spheres_radii,
            static_spheres_colors=static_spheres_colors,
        )

    # ======================================================
    # Figure-style plot: max normal force boundary + goal distance
    # ======================================================
    force_limit = 5.0
    force_axis_max = 1.08 * max(
        force_limit, *(result["force"].max() for result in results)
    )
    fig, ax1 = plt.subplots(figsize=(8, 4))

    force_lines = []
    goal_distance_lines = []

    for result in results:
        linestyle = "-" if result["label"] == "with CBF" else "--"
        force_suffix = "+HOCBF" if result["label"] == "with CBF" else ""
        force_lines.append(
            ax1.plot(
                result["ts"],
                result["force"],
                color="tab:red",
                linewidth=2.5,
                linestyle=linestyle,
                label=f"Force for HOCLF{force_suffix} controller",
            )[0]
        )

    ax1.axhline(force_limit, color="tab:red", linestyle=":", linewidth=2.0)
    ax1.fill_between(
        results[0]["ts"],
        force_limit,
        force_axis_max,
        color="tab:red",
        alpha=0.12,
    )
    ax1.set_xlabel("Time [s]", fontsize=18)
    ax1.set_ylabel("Max Pairwise Normal Force [N]", color="tab:red", fontsize=16)
    ax1.tick_params(axis="x", labelsize=13)
    ax1.tick_params(axis="y", labelcolor="tab:red", labelsize=13)
    ax1.set_ylim(0.0, force_axis_max)
    ax1.grid(True, alpha=0.25)
    ax1.spines["left"].set_color("tab:red")

    ax2 = ax1.twinx()
    for result in results:
        linestyle = "-" if result["label"] == "with CBF" else "--"
        goal_suffix = "+HOCBF" if result["label"] == "with CBF" else ""
        goal_distance_lines.append(
            ax2.plot(
                result["ts"],
                result["distance"],
                color="tab:blue",
                linewidth=2.5,
                linestyle=linestyle,
                label=f"Goal distance for HOCLF{goal_suffix} controller",
            )[0]
        )

    ax2.set_ylabel("Goal Distance [m]", color="tab:blue", fontsize=16)
    ax2.tick_params(axis="y", labelcolor="tab:blue", labelsize=13)
    ax2.spines["right"].set_color("tab:blue")

    if len(results) == 1:
        ax1.legend(
            [force_lines[0], goal_distance_lines[0]],
            [force_lines[0].get_label(), goal_distance_lines[0].get_label()],
            loc="center",
            fontsize=10,
        )
    else:
        lines = [
            force_lines[1],
            force_lines[0],
            goal_distance_lines[1],
            goal_distance_lines[0],
        ]
        ax1.legend(
            lines,
            [line.get_label() for line in lines],
            loc="center",
            fontsize=10,
        )

    fig.tight_layout()
    plt.savefig("force_goal_distance_plot.png", dpi=300, bbox_inches="tight")
    plt.savefig("force_goal_distance_plot.pdf", bbox_inches="tight")
    plt.show()
