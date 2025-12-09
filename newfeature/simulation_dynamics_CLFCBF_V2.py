from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve

import jax
from jax import Array
import jax.numpy as jnp
from cbfpy.cbfs.cbf import CBF, CBFConfig

import matplotlib.pyplot as plt

import numpy as onp
from typing import Dict
from functools import partial

from soromox.systems.tendon_actuated_pcs import TendonActuatedPCS

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from viz.open3d_vis_V2 import visualize_robot_open3d
from viz.matplot_vis import animate_robot_tendons_matplotlib  # currently unused

jax.config.update("jax_enable_x64", True)  # double precision

jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)
jnp.set_printoptions(precision=4, suppress=True)


# ------------------------------------------------------------
#  Geometry / Jacobian utilities
# ------------------------------------------------------------
def backbone_point_position(robot: TendonActuatedPCS, q: Array, s: float) -> Array:
    """Return the 3D position of a backbone point at arc-length s."""
    g = robot.forward_kinematics(q, s)  # (4,4)
    return g[:3, 3]                     # (3,)


def backbone_point_jacobian(robot: TendonActuatedPCS, q: Array, s: float) -> Array:
    """Compute the Jacobian of a backbone point at arc-length s."""
    def fk(q_):
        return backbone_point_position(robot, q_, s)  # (3,)

    J = jax.jacrev(fk)(q)                             # (3, n_strain)
    return J


def backbone_jacobians_along_s(robot: TendonActuatedPCS, q: Array, s_ps: Array) -> Array:
    """Vectorized Jacobian along a list of arc-length samples s_ps."""
    def jac_at_s(s):
        return backbone_point_jacobian(robot, q, s)
    return jax.vmap(jac_at_s)(s_ps)


# ------------------------------------------------------------
#  Spherical obstacles: point-wise contact force + generalized torque
# ------------------------------------------------------------
@jax.jit
def sphere_contact_for_points(
    p_ps: Array,          # (P,3) backbone sample points
    obs_centers: Array,   # (N,3) sphere centers
    obs_radii: Array,     # (N,)   sphere radii
    robot_radius,         # scalar (Python float or JAX scalar)
    k,                    # contact stiffness
    eps: float = 1e-4,    # softplus smoothing parameter
):
    """
    Compute the net contact force at each backbone sample point due to
    spherical obstacles.

    Args:
        p_ps: (P,3) backbone sample points.
        obs_centers: (N,3) sphere centers.
        obs_radii: (N,) sphere radii.
        robot_radius: robot radius (effective).
        k: contact stiffness coefficient.
        eps: softplus smoothing parameter.

    Returns:
        f_total_ps: (P,3) net contact force at each point.
        min_gap_ps: (P,) minimum signed gap to all spheres at each point.
    """
    robot_radius = jnp.asarray(robot_radius)
    k = jnp.asarray(k)

    # (P,1,3) - (1,N,3) → (P,N,3)
    diff = p_ps[:, None, :] - obs_centers[None, :, :]      # (P,N,3)
    dist = jnp.linalg.norm(diff, axis=-1)                  # (P,N)

    # Signed gap: positive outside, negative inside
    gap = dist - (obs_radii[None, :] + robot_radius)       # (P,N)

    # Outward normal: from obstacle center to point
    n_hat = diff / (dist[..., None] + 1e-8)                # (P,N,3)

    # Repulsive force magnitude: only significant when gap < 0
    f_mag = k * jax.nn.softplus(-gap / eps)                # (P,N)
    f_vec = f_mag[..., None] * n_hat                       # (P,N,3)

    # Net force per point: sum over all spheres
    f_total_ps = f_vec.sum(axis=1)                         # (P,3)
    min_gap_ps = gap.min(axis=1)                           # (P,)

    return f_total_ps, min_gap_ps


def compute_sphere_contact_torque(
    q: Array,
    *,
    robot: TendonActuatedPCS,
    s_ps: Array,          # (P,)
    obs_centers: Array,   # (N,3)
    obs_radii: Array,     # (N,)
    robot_radius: float,
    k: float,
    eps: float = 1e-5,
) -> Array:
    """
    Map 3D contact forces with spherical obstacles to generalized torque.

    Args:
        q: generalized strain vector.
        robot: TendonActuatedPCS instance.
        s_ps: (P,) arc-length samples along the backbone.
        obs_centers: (N,3) sphere centers.
        obs_radii: (N,) sphere radii.
        robot_radius: effective robot radius.
        k: contact stiffness coefficient.
        eps: softplus smoothing parameter.

    Returns:
        tau: (n_strain,) generalized torque due to contact.
    """

    # Forward kinematics: (P,4,4) → (P,3)
    g_tsp = robot.forward_kinematics_batched(q, s_ps)
    p_ps = g_tsp[:, :3, 3]                     # (P,3)

    # Net force at each point
    f_total_ps, _ = sphere_contact_for_points(
        p_ps,
        obs_centers,
        obs_radii,
        robot_radius,
        k,
        eps,
    )                                          # (P,3), (P,)

    # Jacobian at each sample point J_i ∈ R^{3×n_strain}
    J_ps = backbone_jacobians_along_s(robot, q, s_ps)  # (P,3,n_strain)

    # τ = Σ_i J_i^T f_i
    tau = jnp.einsum('iaj,ia->j', J_ps, f_total_ps)  # (n_strain,)

    return tau


# ------------------------------------------------------------
#  CBF Config: dynamics + force CBF (3D version)
# ------------------------------------------------------------
class TendonSoroConfig(CBFConfig):

    def __init__(self, robot: TendonActuatedPCS, params: Dict):

        self.robot = robot
        self.params = params

        # Desired 3D end-effector position
        self.p_d_2 = jnp.array([0.10, 0.05, 0.32])

        # Arc-length samples for collision / force CBF
        L_total = jnp.sum(params["L"])
        num_segments = int(robot.num_segments)
        self.s_ps = jnp.linspace(0.0, L_total, 20 * num_segments)

        # Spherical obstacles
        self.robot_radius = jnp.asarray(params["r"][0])  # use first segment radius
        self.safety_margin = 0.00

        self.obs = {
            "centers": jnp.array([
                [0.10, 0.08, 0.24],
                [0.12, 0.06, 0.32],
                [0.04, 0.055, 0.20],
            ]),   # (N,3)
            "radii": jnp.array([0.02, 0.02, 0.02]),  # (N,)
        }

        # Force CBF parameters
        self.contact_spring_constant = 1e4    # stiffness k
        self.maximum_withhold_force = 5.0     # maximum allowable force
        self.softplus_eps = 1e-5              # smoothing parameter

        # Contact torque function
        self.contact_torque_fn = partial(
            compute_sphere_contact_torque,
            robot=robot,
            s_ps=self.s_ps,
            obs_centers=self.obs["centers"],
            obs_radii=self.obs["radii"],
            robot_radius=self.robot_radius,
            k=self.contact_spring_constant,
            eps=self.softplus_eps,
        )

        # ------------------------------------------------
        # State x = [q; qd]
        # ------------------------------------------------
        self.n_q = int(robot.num_active_strains)  # dimension of q
        n = 2 * self.n_q                          # dimension of [q; qd]
        m = int(robot.num_actuators)              # number of tendon actuators

        super().__init__(n=n, m=m)

    def _split_state(self, z: Array) -> tuple[Array, Array]:
        """
        Split state vector z = [q; qd] into (q, qd).
        """
        q, qd = jnp.split(z, 2)
        return q, qd

    # ==========================================================
    # f, g: Dynamics for CBF (consistent with ODE simulation)
    # ==========================================================
    def f(self, z: Array) -> Array:
        """
        Drift term of the control-affine system:

            xdot = f(x) + g(x) u

        where x = [q; qd], u is the tendon input.
        """
        q, qd = self._split_state(z)

        B = self.robot.inertia_matrix(q)        # (n_q, n_q)
        C = self.robot.coriolis_matrix(q, qd)   # (n_q, n_q)
        G = self.robot.gravitational_force(q)   # (n_q,)
        D = self.robot.damping_matrix()         # (n_q, n_q)
        tau_el = self.robot.elastic_force(q)    # (n_q,)

        tau_contact = self.contact_torque_fn(q) # (n_q,)

        B_inv = jnp.linalg.inv(B)

        # B qdd + C qd + D qd + G + tau_el = tau_contact
        # ⇒ qdd = B^{-1}(tau_contact - C qd - D qd - G - tau_el)
        qdd = B_inv @ (tau_contact - C @ qd - D @ qd - G - tau_el)

        return jnp.concatenate([qd, qdd])

    def g(self, z: Array) -> Array:
        """
        Control matrix g(x) such that:

            xdot = f(x) + g(x) u

        and the dynamics satisfy:

            B qdd + C qd + D qd + G + tau_el = tau_contact + A(q) u
        """
        q, _ = self._split_state(z)

        B = self.robot.inertia_matrix(q)        # (n_q, n_q)
        B_inv = jnp.linalg.inv(B)
        A = self.robot.actuation_matrix(q)      # (n_q, m)

        G_bottom = B_inv @ A                    # (n_q, m)
        G_top = jnp.zeros((self.n_q, A.shape[1]))  # (n_q, m)

        # Stack to (2*n_q, m)
        return jnp.vstack([G_top, G_bottom])

    def h_2(self, z: Array) -> Array:
        """
        3D "force CBF" for contact with spherical obstacles.

        For each backbone sample point and each spherical obstacle:
          - Compute smooth penetration depth: penetration ≈ max(0, -gap)
          - Define a force safety margin:
                h = F_max - k * penetration

        Returns:
            A vector of safety margins (P * N,): each entry should be kept ≥ 0.
        """
        q, _ = self._split_state(z)

        # Backbone positions: (P,4,4) → (P,3)
        g_ps = self.robot.forward_kinematics_batched(q, self.s_ps)   # (P,4,4)
        p_ps = g_ps[:, :3, 3]                                        # (P,3)

        centers = self.obs["centers"]                                # (N,3)
        r_obs = self.obs["radii"]                                    # (N,)

        # Pairwise distances: (P,1,3) - (1,N,3) → (P,N,3)
        diff = p_ps[:, None, :] - centers[None, :, :]                # (P,N,3)
        dist = jnp.linalg.norm(diff, axis=-1)                        # (P,N)

        # gap = ||p - c|| - (R_obs + r_robot + safety_margin)
        gap = dist - (
            r_obs[None, :] + self.robot_radius
            # + self.safety_margin  # if you want an extra buffer
        )                                                            # (P,N)
    
        # Smooth penetration depth: penetration ≈ max(0, -gap)
        eps = self.softplus_eps
        penetration = jax.nn.softplus(-gap / eps) * eps              # (P,N)

        k = self.contact_spring_constant
        F_max = self.maximum_withhold_force

        penetration_flat = penetration.reshape(-1)                   # (P*N,)
        safety = F_max - k * penetration_flat                        # (P*N,)

        return safety

    def alpha_2(self, h_2: Array) -> Array:
        """Class-K function α(h) = 10 h for the force CBF."""
        return h_2 * 10.0


# ------------------------------------------------------------
#   Main: 3D soft arm + PD nominal + CBF + spherical contacts
# ------------------------------------------------------------
if __name__ == "__main__":
    num_segments = 2
    rho = 1070 * jnp.ones(
        (num_segments,)
    )  # Volumetric density of Dragon Skin 20 [kg/m^3]

    params = {
        "p0": jnp.array([jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, 0.0]),
        "L": 15e-2 * 2 / num_segments * jnp.ones((num_segments,)),
        "r": 3.6e-2 * jnp.ones((num_segments,)),
        "rho": rho,
        "g": jnp.array([0.0, 0.0, 9.81]),
        "E": 20e3 * jnp.ones((num_segments,)),
        "G": 20e3 * jnp.ones((num_segments,)),
    }

    # Damping matrix
    params["D"] = 1e-3 * jnp.diag(
        (
            jnp.repeat(
                jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]),
                num_segments,
                axis=0,
            )
            * params["L"][:, None]
        ).flatten()
    )

    # Tendon routing parameters
    r0, r1 = float(params["r"][0]) - 0.005, float(params["r"][1]) - 0.005
    theta0 = jnp.deg2rad(jnp.array([120, 240, 0]))
    theta1 = jnp.deg2rad(jnp.array([180, 300, 60]))
    ry0, rz0 = r0 * jnp.cos(theta0), r0 * jnp.sin(theta0)
    ry1, rz1 = r1 * jnp.cos(theta1), r1 * jnp.sin(theta1)

    tendon_routing_params = {
        "ry": jnp.concatenate([ry0, ry1]),   # (6,)
        "rz": jnp.concatenate([rz0, rz1]),   # (6,)
        "my": jnp.zeros(6),
        "mz": jnp.zeros(6),
        "idx_seg_att": jnp.array([0, 0, 0, 1, 1, 1]),
    }

    # ======================================================
    # Robot initialization
    # ======================================================
    strain_selector = (
        jnp.array([True, True, True, True, True, True])[None, :]
        .repeat(num_segments, axis=0)
        .flatten()
    )

    robot = TendonActuatedPCS(
        num_segments=num_segments,
        params=params,
        tendon_routing_params=tendon_routing_params,
        strain_selector=strain_selector,
        order_gauss=5,
    )

    print("num_active_strains:", robot.num_active_strains)
    print("num_actuators:", robot.num_actuators)

    # ======================================================
    # Controller + CBF initialization
    # ======================================================
    config = TendonSoroConfig(robot, params)
    cbf = CBF.from_config(config)

    # End-effector FK + Jacobian
    L_total = jnp.sum(params["L"])

    def ee_pos_fn(q: Array) -> Array:
        g = robot.forward_kinematics(q, L_total)
        return g[:3, 3]

    ee_pos_jac_fn = jax.jit(jax.jacrev(ee_pos_fn))

    @jax.jit
    def nominal_controller(z: Array, p_des: Array) -> Array:
        """
        Workspace PD controller on the end-effector position,
        producing a nominal tendon input u_nom ∈ R^m.
        """
        q, qd = config._split_state(z)

        x = ee_pos_fn(q)                 # (3,)
        J_ee = ee_pos_jac_fn(q)          # (3, n_q)
        x_dot = J_ee @ qd

        x_des = p_des                    # (3,)
        x_dot_des = jnp.zeros_like(x_des)

        e = x_des - x
        e_dot = x_dot_des - x_dot

        Kp = 1.0
        Kd = 1.0
        Ki = 0.0

        f_op = Kp * e + Kd * e_dot       # (3,)

        # Generalized force: add gravity and elastic forces as compensation
        G = robot.gravitational_force(q)   # (n_q,)
        tau_el = robot.elastic_force(q)    # (n_q,)
        # use the k_desired force to compute tendon input

        tau = J_ee.T @ f_op + G + tau_el   # (n_q,)

        A = robot.actuation_matrix(q)      # (n_q, m)
        u_nom = jnp.linalg.pinv(A) @ tau   # (m,)

        return u_nom

    def closed_loop_dyn(t: float, y: Array, args) -> Array:
        """
        Closed-loop dynamics:

            xdot = f(x) + g(x) u_safe

        where u_safe = cbf.safety_filter(x, u_nom).
        """
        p_des = args        # desired end-effector position
        z = y               # z = [q; qd]

        u_nom = nominal_controller(z, p_des)
        u_safe = cbf.safety_filter(z, u_nom)

        q, qd = config._split_state(z)

        B = robot.inertia_matrix(q)
        C = robot.coriolis_matrix(q, qd)
        G = robot.gravitational_force(q)
        D = robot.damping_matrix()
        tau_el = robot.elastic_force(q)
        A = robot.actuation_matrix(q)

        tau_contact = config.contact_torque_fn(q)

        # B qdd + C qd + D qd + G + tau_el = tau_contact + A u_safe
        qdd = jnp.linalg.inv(B) @ (
            tau_contact + A @ u_safe - C @ qd - D @ qd - G - tau_el
        )

        return jnp.concatenate([qd, qdd])

    # =====================================================
    # Initial condition & simulation
    # =====================================================
    q0 = jnp.repeat(
        jnp.array([0.0, 0.0, 0.0 * jnp.pi, 0.0, 0.0, 0.0])[None, :],
        num_segments,
        axis=0,
    ).flatten()
    qd0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, qd0])

    # Simulation time parameters
    t0 = 0.0
    t1 = 8.0
    dt = 1e-3

    z_des = config.p_d_2  # desired end-effector position

    term = ODETerm(closed_loop_dyn)
    sol = diffeqsolve(
        term,
        Tsit5(),
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=y0,
        args=z_des,
        saveat=SaveAt(ts=jnp.arange(t0, t1, dt)),
        max_steps=None,
    )

    ts, y_ts = sol.ts, sol.ys
    q_ts, qd_ts = jnp.split(y_ts, 2, axis=1)

    # =====================================================
    # Clearance statistics
    # =====================================================
    g_ee_tsp = jax.vmap(
        lambda q: robot.forward_kinematics_batched(q, config.s_ps)
    )(q_ts)

    p_tsp = g_ee_tsp[:, :, :3, 3]  # (T, P, 3)
    centers = config.obs["centers"]    # (N, 3)
    radii   = config.obs["radii"]      # (N,)

    dist_tpN = jnp.linalg.norm(
        p_tsp[:, :, None, :] - centers[None, None, :, :],
        axis=-1
    ) - (radii[None, None, :] + config.robot_radius)
    min_clearance_tN = dist_tpN.min(axis=1)

    global_min_t = onp.asarray(min_clearance_tN).min(axis=1)  # (T,)
    ts_np = onp.asarray(ts)

    plt.figure(figsize=(8, 4))
    plt.plot(ts_np, global_min_t)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel("Global min clearance [m]")
    plt.title("Global minimum clearance over time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # =====================================================
    # End-effector position over time
    # =====================================================
    forward_kinematics_end_effector = jax.jit(
        partial(
            robot.forward_kinematics,
            s=jnp.sum(robot.L),  # end-effector at total length
        )
    )
    g_ee_ts = jax.vmap(forward_kinematics_end_effector)(q_ts)

    plt.figure()
    plt.plot(ts, g_ee_ts[:, 0, 3], label="End-effector x [m]")
    plt.plot(ts, g_ee_ts[:, 1, 3], label="End-effector y [m]")
    plt.plot(ts, g_ee_ts[:, 2, 3], label="End-effector z [m]")
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector position [m]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()

    # =====================================================
    # Trajectory visualization (Open3D)
    # =====================================================
    def obs_for_vis(obs_dict, colors=None):
        centers = jnp.asarray(obs_dict["centers"])
        radii   = jnp.asarray(obs_dict["radii"])
        N = len(radii)
        if colors is None:
            colors = [(0.5, 0.5, 0.5)] * N
        assert len(colors) == N, "Number of colors must match number of obstacles"
        return [
            (tuple(centers[i]), float(radii[i]), tuple(colors[i]))
            for i in range(N)
        ]

    stride = 200
    visualize_robot_open3d(
        robot,
        t_list=ts[::stride],
        q_list=q_ts[::stride],
        num_points=80,
        fps=60,
        target_point=config.p_d_2,
        target_radius=0.01,
        target_color=(1.0, 0.1, 0.1),
        obstacles=obs_for_vis(config.obs),
        record={"dir": "frames", "every_n": 1},
    )

    # =====================================================
    # Save CSV for further analysis
    # =====================================================
    ts_onp  = onp.asarray(ts)
    qts_onp = onp.asarray(q_ts)

    data_to_save = onp.column_stack([ts_onp, qts_onp])

    num_dof = qts_onp.shape[1]
    header = "time," + ",".join([f"q{i}" for i in range(num_dof)])

    onp.savetxt(
        "setpoint_results_new.csv",
        data_to_save,
        delimiter=",",
        header=header,
        comments='',
        fmt="%.8f"
    )

    print("✅ Saved to setpoint_results_new.csv")
