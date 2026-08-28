__all__ = ["Pendulum"]


from typing import Any

import equinox as eqx
from jax import Array, vmap
from jax import numpy as jnp

from soromox.actuation.core import Actuator, PassiveElement
from soromox.systems.components import CrossSectionGeometry
from soromox.systems.pendulum.params import PendulumParams
from soromox.systems.soft_robot import SoftRobot


class Pendulum(SoftRobot):
    """
    Planar N-link serial pendulum (revolute chain) with closed-form JAX dynamics.

    This class provides a lightweight rigid-body model of an N-link planar pendulum.
    Kinematics and dynamics are computed on-the-fly using Jacobian-based rigid-body
    formulas that are fully JAX compatible (supporting JIT, vmap, grad, etc.).

    Notes
    ----
    By providing non-zero joint stiffness `K` and/or damping `D`, the model can be
    used as a planar articulated soft robot with linear elastic and viscous joints.
    In this "soft-joint" mode, additional joint torques act as
        τ_el = K q,   τ_d = D qd,
    which endow the otherwise rigid chain with distributed elasticity and dissipation.

    Modeling Overview
    -----------------
    The system consists of N planar links connected by revolute joints (out-of-plane z-axis).
    Joint coordinates q[i] represent relative rotations between consecutive links. The
    cumulative orientation of link i is θ_i = Σ_{k=0..i} q[k]. Each link i has:
        - length L[i]
        - center-of-mass (COM) distance Lc[i] from its proximal joint along the link axis
        - mass m[i]
        - planar moment of inertia about its COM I[i]

    At zero joint angle, each link points along the local positive x-axis of its
    proximal frame. ``Lc[i]`` is measured from the proximal joint along this link
    axis. The base pose may rotate that local convention in the inertial frame.

    Gravity is applied as a planar acceleration vector g = [g_x, g_y]. Optional linear
    joint stiffness K and viscous damping D can be supplied to model elastic and damping
    torques (τ_el = K q, τ_d = D qd).

    Dynamics Formulation
    --------------------
    Using standard rigid-body aggregation, the generalized inertia (mass) matrix is:
        B(q) = Σ_i ( m_i Jv_i^T Jv_i + I_i Jw_i^T Jw_i )
    where Jv_i (2×N) is the linear velocity Jacobian of COM i, and Jw_i (N,) the angular
    velocity Jacobian (planar => binary pattern). Coriolis/centrifugal matrix C(q, qd)
    is obtained from the Christoffel symbols of B(q). Gravity vector:
        G(q) = Σ_i Jv_i^T ( m_i g ).

    Forward kinematics of link tips and COMs is computed cumulatively using link frame
    directions derived from the cumulative joint angles.

    State Convention
    ----------------
    - q ∈ R^N: joint angles (relative)
    - qd ∈ R^N: joint angular velocities

    Attributes:
        num_links: Number of links / DoFs.
        num_actuators: Number of installed actuator channels. This equals
            ``num_links`` for the default identity transmission.
        m, I, L, Lc, r: Physical link properties. Radius is used only for
            circular visualization geometry; mass and inertia are supplied independently.
        g: Planar gravity vector [g_x, g_y].
        K, D: Optional joint stiffness and damping matrices (zeros if omitted).
        q_ref_k: Optional rest configuration of the torsional springs (default is the zero configuration).
    """

    # Static field
    num_links: int = eqx.field(static=True)

    # Stored physical parameters
    m: Array
    I: Array  # noqa: E741
    L: Array
    Lc: Array
    r: Array
    g: Array  # planar gravity acceleration vector

    # Optional linear elasticity/damping to make it an articulated soft robot.
    K: Array
    D: Array
    q_ref_k: Array
    params: PendulumParams

    def __init__(
        self,
        params: PendulumParams,
        *,
        actuators: Actuator | tuple[Actuator, ...] | None = None,
        passive_elements: PassiveElement | tuple[PassiveElement, ...] | None = (),
        **kwargs: Any,
    ) -> None:
        """Initialize the pendulum system with typed parameters."""
        if not isinstance(params, PendulumParams):
            raise TypeError("params must be a PendulumParams instance.")
        params.validate()
        super().__init__(base_pose=params.base_pose, **kwargs)
        self.params = params

        # Basic parameter extraction
        m = jnp.asarray(params.mass)  # (n,)
        inertia = jnp.asarray(params.moment_inertia)  # (n,)
        L = jnp.asarray(params.length)  # (n,)
        Lc = jnp.asarray(params.center_of_mass_length)  # (n,)
        g = jnp.asarray(params.gravity)  # (2,)

        n_q = m.shape[0]
        # Consistency checks (lightweight; JIT friendly if shapes static)
        assert inertia.shape[0] == n_q and L.shape[0] == n_q and Lc.shape[0] == n_q, (
            "Parameter length mismatch"
        )
        assert g.shape[0] == 2, "Gravity vector must be length 2"

        self.num_links = int(n_q)
        self.num_dofs = self.num_links

        # set parameters
        self.m = m
        self.I = inertia
        self.L = L
        self.Lc = Lc
        self.g = g
        self.r = jnp.asarray(params.radius)

        self.K = jnp.asarray(params.joint_stiffness)
        self.D = jnp.asarray(params.joint_damping)
        self.q_ref_k = jnp.asarray(params.joint_rest_configuration)
        self._configure_actuation(actuators, passive_elements)

    @property
    def is_planar(self) -> bool:
        """Pendulum is a planar (2D) model."""
        return True

    @property
    def supports_articulated_tendon_routing(self) -> bool:
        """Pendulum coordinates form a serial articulated joint chain."""
        return True

    @property
    def segment_length(self) -> Array:
        """Per-link lengths."""
        return jnp.asarray(self.L)

    def cross_section_geometry(self, q: Array, s: Array) -> tuple[Array, Array]:
        """Return circular visualization geometry using the per-link radius.

        The radius does not determine the link's mass or planar moment of inertia.
        """
        L_cum = jnp.cumsum(jnp.concatenate([jnp.zeros(1), self.L]))
        segment_idx = jnp.clip(jnp.sum(s > L_cum) - 1, 0, self.num_links - 1)
        radius = self.r[segment_idx]
        tag = jnp.asarray(CrossSectionGeometry.CIRCULAR, dtype=jnp.int32)
        return tag, jnp.array([radius])

    def _current_body_params(self) -> PendulumParams:
        """Return the pendulum body parameters."""
        return self.params

    def _with_pendulum_params(self, params: PendulumParams) -> "Pendulum":
        """Return a copy with pendulum body caches refreshed."""
        current_params = self._current_body_params()
        if not isinstance(params, PendulumParams):
            raise TypeError("params must be a PendulumParams instance.")
        params.validate_for_update()
        if params.mass.shape != current_params.mass.shape:
            raise ValueError(
                "mass shape changes the model structure; construct a new Pendulum."
            )
        return eqx.tree_at(
            lambda x: (
                x.params,
                x.base_pose,
                x.m,
                x.I,
                x.L,
                x.Lc,
                x.g,
                x.r,
                x.K,
                x.D,
                x.q_ref_k,
            ),
            self,
            (
                params,
                jnp.asarray(params.base_pose),
                jnp.asarray(params.mass),
                jnp.asarray(params.moment_inertia),
                jnp.asarray(params.length),
                jnp.asarray(params.center_of_mass_length),
                jnp.asarray(params.gravity),
                jnp.asarray(params.radius),
                jnp.asarray(params.joint_stiffness),
                jnp.asarray(params.joint_damping),
                jnp.asarray(params.joint_rest_configuration),
            ),
        )

    def with_params(self, params: PendulumParams) -> "Pendulum":
        """Return an updated copy with a full typed parameter object."""
        return self._with_pendulum_params(params)

    def update_params(self, **updates: Array) -> "Pendulum":
        """Return an updated copy with selected typed parameter fields replaced."""
        shape_locked_fields = (
            "mass",
            "moment_inertia",
            "length",
            "center_of_mass_length",
            "joint_stiffness",
            "joint_damping",
            "joint_rest_configuration",
            "radius",
            "base_pose",
            "gravity",
        )
        for name in shape_locked_fields:
            if (
                name in updates
                and updates[name] is not None
                and jnp.asarray(updates[name]).shape != getattr(self.params, name).shape
            ):
                raise ValueError(
                    f"{name} shape changes the model structure; construct a new Pendulum."
                )
        return self.with_params(self.params.replace(**updates))

    # -------------------------------------------------
    # Internal helpers (geometry & Jacobians, pure JAX)
    # -------------------------------------------------
    def _cumulative_angles(self, q: Array) -> Array:
        """
        Compute cumulative joint angles from relative joint angles.

        Args:
            q (Array): Relative joint angles, shape (N,) [rad]

        Returns:
            Array: Cumulative angles θ_i = Σ_{k=0..i} q[k], shape (N,) [rad]
        """
        base_theta = jnp.asarray(self.base_pose[0], dtype=q.dtype)
        return base_theta + jnp.cumsum(q)  # (n,)

    def _base_xy(self, dtype: jnp.dtype) -> Array:
        """Return the planar base translation stored in ``base_pose``."""
        return jnp.asarray(self.base_pose[1:3], dtype=dtype)

    def _directions(self, q: Array) -> Array:
        """
        Compute unit direction vectors for each link based on cumulative angles.

        Args:
            q (Array): Relative joint angles, shape (N,) [rad]

        Returns:
            dirs (Array): Unit direction vectors [cos(θ), sin(θ)] for each link, shape (N, 2)
        """
        theta = self._cumulative_angles(q)
        dirs = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)  # (n,2)
        return dirs

    def _joint_positions(self, q: Array) -> Array:
        """
        Compute positions of all joint centers (proximal ends of links).

        Joint 0 is at the origin. Joint i is at the distal end of link i-1.

        Args:
            q (Array): Relative joint angles, shape (N,) [rad]

        Returns:
            p_joints (Array): Joint positions, shape (N, 2) [m]
                   p_joints[i] = position of proximal end of link i as array of shape (2, )
        """
        # Position of proximal joint of each link i (base of link i)
        # joint 0 at origin
        dirs = self._directions(q)
        seg_vecs = self.L[:, None] * dirs  # (n,2)
        tips = jnp.cumsum(seg_vecs, axis=0)  # tip positions of each link
        # Proximal positions p_joint[i] = tip[i-1] with base at 0
        base_xy = self._base_xy(q.dtype)
        p_joints = jnp.concatenate([base_xy[None, :], base_xy + tips[:-1]], axis=0)
        return p_joints

    def _com_positions(self, q: Array) -> Array:
        """
        Compute center-of-mass positions for all links.

        Args:
            q (Array): Relative joint angles, shape (N,) [rad]

        Returns:
            p_coms (Array): COM positions, shape (N, 2) [m]
                   p_coms[i] = center of mass of link i
        """
        dirs = self._directions(q)
        p_joint = self._joint_positions(q)
        p_coms = p_joint + self.Lc[:, None] * dirs  # (N,2)
        return p_coms

    def _tip_positions(self, q: Array) -> Array:
        """
        Compute distal tip positions for all links.

        Args:
            q (Array): Relative joint angles, shape (N,) [rad]

        Returns:
            p_tips (Array): Link tip positions, shape (N, 2) [m]
                   p_tips[i] = distal end position of link i
        """
        dirs = self._directions(q)
        seg_vecs = self.L[:, None] * dirs
        base_xy = self._base_xy(q.dtype)
        p_tips = base_xy + jnp.cumsum(seg_vecs, axis=0)  # (n,2)
        return p_tips

    @eqx.filter_jit
    def forward_kinematics_joints(self, q: Array) -> Array:
        """
        Forward kinematics at the proximal joint positions for all links.

        Joint i corresponds to the proximal end of link i.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            chi_joints (Array): Joint poses stacked as [theta; px; py] with shape (N, 3)
                   - theta: Orientation angle at each joint (rad)
                   - px, py: Cartesian positions of each joint (m)
        """
        p_joints = self._joint_positions(q)  # (N,2)
        theta = self._cumulative_angles(q)  # (N,)
        chi_joints = jnp.concatenate([theta[:, None], p_joints], axis=-1)
        return chi_joints

    @eqx.filter_jit
    def forward_kinematics_tips(self, q: Array) -> Array:
        """
        Forward kinematics of the distal tips of all links.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            chi_tips (Array): Link tip poses [θ, p_x, p_y] of shape (N, 3).
        """
        p_tips = self._tip_positions(q)  # (N,2)
        theta = self._cumulative_angles(q)  # (N,)
        chi_tips = jnp.concatenate([theta[:, None], p_tips], axis=-1)
        return chi_tips

    @eqx.filter_jit
    def forward_kinematics_coms(self, q: Array) -> Array:
        """
        Forward kinematics of the center of mass (COM) of all links.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            chi_coms (Array): COM poses [θ, p_x, p_y] of shape (N, 3).
        """
        p_coms = self._com_positions(q)  # (N,2)
        theta = self._cumulative_angles(q)  # (N,)
        chi_coms = jnp.concatenate([theta[:, None], p_coms], axis=-1)
        return chi_coms

    def _linear_jacobian_coms(self, q: Array) -> Array:
        """
        Compute linear velocity Jacobians for all link centers of mass.

        This method uses vectorized computation to efficiently calculate the Jacobians
        for all links simultaneously, avoiding loops and conditionals.

        Args:
            q (Array): Relative joint angles, shape (N,) [rad]

        Returns:
            Jv_coms (Array): Linear velocity Jacobians, shape (N, 2, N)
                   Jv_coms[i, :, j] = ∂(COM_i position)/∂qd[j]
                   Axis order: (link, cartesian_component, joint)
        """
        n = self.num_links
        p_joint = self._joint_positions(q)  # (n,2)
        p_com = self._com_positions(q)  # (n,2)
        r = p_com[:, None, :] - p_joint[None, :, :]  # (n,n,2)
        col_x = -r[..., 1]  # (n,n)  # z x r
        col_y = r[..., 0]  # (n,n)
        mask = jnp.tril(jnp.ones((n, n), dtype=q.dtype))
        col_x = col_x * mask
        col_y = col_y * mask
        Jv_coms = jnp.stack([col_x, col_y], axis=1)  # (n,2,n)
        return Jv_coms

    def _linear_jacobian_tips(self, q: Array) -> Array:
        """
        Compute linear velocity Jacobians for the distal tips of all links.

        Args:
            q (Array): Relative joint angles, shape (N,) [rad]

        Returns:
            Jv_tips (Array): Linear Jacobians at link tips, shape (N, 2, N)
                            Axis order: (link, cartesian_component, joint)
        """
        n = self.num_links
        p_joint = self._joint_positions(q)  # (n,2)
        p_tip = self._tip_positions(q)  # (n,2)
        r = p_tip[:, None, :] - p_joint[None, :, :]  # (n,n,2)
        col_x = -r[..., 1]  # (n,n)
        col_y = r[..., 0]  # (n,n)
        mask = jnp.tril(jnp.ones((n, n), dtype=q.dtype))
        col_x = col_x * mask
        col_y = col_y * mask
        Jv_tips = jnp.stack([col_x, col_y], axis=1)  # (n,2,n)
        return Jv_tips

    def _angular_jacobians(self) -> Array:
        """
        Compute angular velocity Jacobians for all links.

        For planar pendulums, the angular velocity Jacobian has a simple pattern:
        Jw_links[i,j] = 1 if j <= i, 0 otherwise (lower triangular matrix of ones).

        Returns:
            Jw_links (Array): Angular velocity Jacobians, shape (N, N)
                   Jw_links[i,j] = ∂(link_i angular velocity)/∂qd[j]
        """
        # Jw_i for planar: ones for joints <= i else 0. Shape (n, n)
        n = self.num_links
        idxs = jnp.arange(n)
        Jw_links = (idxs[None, :] <= idxs[:, None]).astype(self.m.dtype)  # (n,n)
        return Jw_links

    @eqx.filter_jit
    def jacobian_joints(self, q: Array) -> Array:
        """
        Spatial Jacobians at proximal joint positions for all links.

        For joint i (proximal end of link i):
          - Linear Jacobian equals the tip Jacobian of link i-1 (i > 0),
            and zeros for i = 0.
          - Angular Jacobian is 1 for joints <= i, else 0.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            J_joints (Array): array of spatial Jacobians with shape (N, 3, N), where for each i:
                   J_joints[i] = [Jw; Jv_x; Jv_y] with shape (3, N) at joint i.
        """
        Jv_tip = self._linear_jacobian_tips(q)  # (N,2,N)
        # Build linear Jacobians at joints: i=0 -> zeros; i>0 -> tip of (i-1)
        Jv_joints = jnp.zeros_like(Jv_tip)
        Jv_joints = Jv_joints.at[1:].set(Jv_tip[:-1])

        Jw = self._angular_jacobians()  # (N,N)
        # Order rows as [Jw; Jv_x; Jv_y]
        J_joints = jnp.concatenate([Jw[:, None, :], Jv_joints], axis=1)  # (N,3,N)
        return J_joints

    @eqx.filter_jit
    def jacobian_tips(self, q: Array) -> Array:
        """
        Spatial Jacobians for all link tips.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            J_tips (Array): array of spatial Jacobians with shape (N, 3, N), where for each i:
                   J_tips[i] = [Jw; Jv_x; Jv_y] with shape (3, N) at tip i.
        """
        Jv_all_tip = self._linear_jacobian_tips(q)  # (N,2,N)
        Jw_all = self._angular_jacobians()  # (N,N)
        J_tips = jnp.concatenate([Jw_all[:, None, :], Jv_all_tip], axis=1)
        return J_tips

    @eqx.filter_jit
    def jacobian_and_time_derivatives_tips(
        self, q: Array, qd: Array
    ) -> tuple[Array, Array]:
        """
        Spatial Jacobians and their time-derivatives at all link tips.

        Returns J_all and Jd_all with rows ordered as [Jw, Jv_x, Jv_y].
        For planar chains Jw is configuration independent, hence d/dt Jw = 0.

        Closed-form expression (no autodiff):
          - Jv_i[:, j]   = z × (p_tip_i - p_joint_j) for j <= i, else 0
          - d/dt Jv_i[:, j] = z × (v_tip_i - v_joint_j) for j <= i, else 0

        Args:
            q (Array): Joint angles, shape (N,) [rad]
            qd (Array): Joint velocities, shape (N,) [rad/s]

        Returns:
            J_tips (Array): Spatial Jacobians at tips, shape (N, 3, N)
            Jd_tips (Array): Time-derivative of Jacobians at tips, shape (N, 3, N)
        """
        N = self.num_links

        # compute the Jacobians at the tips
        J_tips = self.jacobian_tips(q)
        # the linear part of the jacobian
        Jv_tips = J_tips[:, 1:, :]

        # compute the jacobian at the joints
        J_joints = self.jacobian_joints(q)
        # linear part of the jacobian
        Jv_joints = J_joints[:, 1:, :]

        # Linear velocities of tips and joints
        v_tips = jnp.einsum("icj,j->ic", Jv_tips, qd)  # (N,2)
        v_joints = jnp.einsum("icj,j->ic", Jv_joints, qd)  # (N,2)

        # Time derivative of linear Jacobians columns via z × (Δv)
        dv = v_tips[:, None, :] - v_joints[None, :, :]  # (N,N,2) as (i,j,xy)
        Jvdot_x = -dv[..., 1]
        Jvdot_y = dv[..., 0]
        mask = jnp.tril(jnp.ones((N, N), dtype=q.dtype))
        Jvd = jnp.stack([Jvdot_x * mask, Jvdot_y * mask], axis=1)  # (N,2,N)

        # Angular part derivative is zero for planar revolute chains
        Jwd = jnp.zeros((N, 1, N), dtype=q.dtype)
        Jd_tips = jnp.concatenate([Jwd, Jvd], axis=1)  # (N,3,N)
        return J_tips, Jd_tips

    @eqx.filter_jit
    def jacobian_coms(self, q: Array) -> Array:
        """
        Spatial Jacobian for the center of mass (COM) of a specified link.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            J_coms (Array): array of spatial Jacobians with shape (N, 3, N), where for each i:
                   J_coms[i] = [Jw; Jv_x; Jv_y] with shape (3, N) at COM i.
        """
        Jv_all_com = self._linear_jacobian_coms(q)  # (N,2,N)
        Jw_all = self._angular_jacobians()  # (N,N)
        J_coms = jnp.concatenate([Jw_all[:, None, :], Jv_all_com], axis=1)
        return J_coms

    # -----------------------------------------
    # Arc-length parameterized kinematics/Jacobians
    # -----------------------------------------

    @property
    def total_length(self) -> Array:
        """Total arc-length of the pendulum (sum of all link lengths)."""
        return jnp.sum(self.L)

    @property
    def L_cum(self) -> Array:
        """Cumulative link lengths [0, L[0], L[0]+L[1], ...]."""
        return jnp.concatenate([jnp.zeros(1), jnp.cumsum(self.L)])

    @eqx.filter_jit
    def classify_segment(self, s: Array) -> tuple[Array, Array]:
        """
        Determine which link contains the point at arc-length s.

        Args:
            s: Arc-length position in [0, total_length].

        Returns:
            link_idx: Index of the link containing point s (0-indexed).
            s_local: Local arc-length within that link, in [0, L[link_idx]].
        """
        L_cum = self.L_cum
        # Find the link index: s lies in [L_cum[i], L_cum[i+1]]
        # Using side="left" ensures that s at exact boundary L_cum[i+1] stays in segment i
        link_idx = jnp.searchsorted(L_cum[1:], s, side="left")
        link_idx = jnp.clip(link_idx, 0, self.num_links - 1)
        s_local = s - L_cum[link_idx]
        return link_idx, s_local

    @eqx.filter_jit
    def _forward_kinematics(self, q: Array, s: Array) -> Array:
        """
        Compute the forward kinematics at arc-length position s along the pendulum.

        The pose is computed as [theta, x, y] where theta is the orientation angle
        and (x, y) is the Cartesian position. A zero link angle points along the
        local positive x-axis; cumulative joint angles rotate that direction.

        Args:
            q: Joint angles, shape (N,) [rad].
            s: Arc-length position in [0, total_length] (units: m).

        Returns:
            chi: Pose [theta, x, y] at position s, shape (3,).
        """
        link_idx, s_local = self.classify_segment(s)

        # Cumulative angles at each link
        theta_cumsum = self._cumulative_angles(q)

        # Base positions for each link (tip positions of previous links)
        p_tips = self._tip_positions(q)  # (N, 2)
        p_base = jnp.concatenate([jnp.zeros((1, 2)), p_tips[:-1]], axis=0)

        # Get the base position and angle for this link
        p_link_base = p_base[link_idx]
        theta_link = theta_cumsum[link_idx]

        # Direction vector for this link
        dir_vec = jnp.array([jnp.cos(theta_link), jnp.sin(theta_link)])

        # Position at s is base + s_local * direction
        p_s = p_link_base + s_local * dir_vec

        chi = jnp.concatenate([theta_link[None], p_s])
        return chi

    @eqx.filter_jit
    def forward_kinematics_abscissa_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute forward kinematics at multiple arc-length positions.

        Args:
            q: Joint angles, shape (N,) [rad].
            s_ps: Arc-length positions, shape (M,) [m].

        Returns:
            chi_ps: Poses at all positions, shape (M, 3).
        """
        return vmap(lambda s: self.forward_kinematics(q, s))(s_ps)

    @eqx.filter_jit
    def _jacobian(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian at arc-length position s along the pendulum.

        The Jacobian maps joint velocities to Cartesian velocities [omega_z, v_x, v_y].

        Args:
            q: Joint angles, shape (N,) [rad].
            s: Arc-length position in [0, total_length] (units: m).

        Returns:
            J: Jacobian matrix, shape (3, N).
        """
        link_idx, s_local = self.classify_segment(s)
        N = self.num_links

        # Angular Jacobian: J_omega[j] = 1 if joint j affects this point (j <= link_idx)
        j_indices = jnp.arange(N)
        J_omega = (j_indices <= link_idx).astype(q.dtype)

        # Position at s
        chi = self._forward_kinematics(q, s)
        p_s = chi[1:]

        # Joint positions
        p_joints = self._joint_positions(q)  # (N, 2)

        # Linear Jacobian: for joints j <= link_idx
        # J_v[:, j] = z × (p_s - p_joint[j]) = [-dy, dx] where d = p_s - p_joint[j]
        r = p_s[None, :] - p_joints  # (N, 2): vectors from each joint to p_s
        Jv_x = -r[:, 1]  # (N,)
        Jv_y = r[:, 0]  # (N,)

        # Mask: only joints j <= link_idx contribute
        mask = (j_indices <= link_idx).astype(q.dtype)
        Jv_x = Jv_x * mask
        Jv_y = Jv_y * mask

        J = jnp.stack([J_omega, Jv_x, Jv_y], axis=0)  # (3, N)
        return J

    @eqx.filter_jit
    def jacobian_abscissa_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute Jacobians at multiple arc-length positions.

        Args:
            q: Joint angles, shape (N,) [rad].
            s_ps: Arc-length positions, shape (M,) [m].

        Returns:
            J_ps: Jacobians at all positions, shape (M, 3, N).
        """
        return vmap(lambda s: self.jacobian(q, s))(s_ps)

    @eqx.filter_jit
    def _jacobian_and_time_derivative(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its time derivative at arc-length position s.

        Args:
            q: Joint angles, shape (N,) [rad].
            qd: Joint velocities, shape (N,) [rad/s].
            s: Arc-length position in [0, total_length] (units: m).

        Returns:
            J: Jacobian matrix, shape (3, N).
            Jd: Time derivative of the Jacobian, shape (3, N).
        """
        link_idx, s_local = self.classify_segment(s)
        N = self.num_links
        j_indices = jnp.arange(N)

        # Get Jacobian
        J = self._jacobian(q, s)

        # Angular Jacobian time derivative is zero (planar revolute)
        Jd_omega = jnp.zeros(N, dtype=q.dtype)

        # Joint positions and velocities
        J_joints = self.jacobian_joints(q)  # (N, 3, N)
        Jv_joints = J_joints[:, 1:, :]  # (N, 2, N) linear part
        v_joints = jnp.einsum("icj,j->ic", Jv_joints, qd)  # (N, 2)

        # Velocity at point s
        J_s_linear = J[1:, :]  # (2, N)
        v_s = J_s_linear @ qd  # (2,)

        # Time derivative of linear Jacobian
        # d/dt(J_v[:, j]) = z × (v_s - v_joint[j]) for j <= link_idx
        dv = v_s[None, :] - v_joints  # (N, 2)
        Jvd_x = -dv[:, 1]
        Jvd_y = dv[:, 0]

        mask = (j_indices <= link_idx).astype(q.dtype)
        Jvd_x = Jvd_x * mask
        Jvd_y = Jvd_y * mask

        Jd = jnp.stack([Jd_omega, Jvd_x, Jvd_y], axis=0)  # (3, N)
        return J, Jd

    @eqx.filter_jit
    def jacobian_and_time_derivative_abscissa_batched(
        self, q: Array, qd: Array, s_ps: Array
    ) -> tuple[Array, Array]:
        """
        Compute Jacobians and their time derivatives at multiple arc-length positions.

        Args:
            q: Joint angles, shape (N,) [rad].
            qd: Joint velocities, shape (N,) [rad/s].
            s_ps: Arc-length positions, shape (M,) [m].

        Returns:
            J_ps: Jacobians at all positions, shape (M, 3, N).
            Jd_ps: Jacobian time derivatives at all positions, shape (M, 3, N).
        """
        return vmap(lambda s: self.jacobian_and_time_derivative(q, qd, s))(s_ps)

    # -------------------------------
    # Standardized dynamics interface
    # -------------------------------
    @eqx.filter_jit
    def inertia_matrix(self, q: Array) -> Array:
        """
        Compute the generalized mass (inertia) matrix using Jacobian formulation.

        The mass matrix aggregates contributions from all links:
        B(q) = Σ_i ( m_i * Jv_i^T * Jv_i + I_i * Jw_i^T * Jw_i )

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            B (Array): Generalized mass matrix, shape (N, N) [kg⋅m²]
        """
        Jv = self._linear_jacobian_coms(q)  # (n,2,n)
        Jw = self._angular_jacobians()  # (n,n)
        # B = Σ ( m_i Jv_i^T Jv_i + I_i Jw_i^T Jw_i )
        m_terms = jnp.einsum("i,ika,ikb->ab", self.m, Jv, Jv)
        w_terms = jnp.einsum("i,ia,ib->ab", self.I, Jw, Jw)
        B = m_terms + w_terms
        return B

    @eqx.filter_jit
    def coriolis_matrix(self, q: Array, qd: Array) -> Array:
        """
        Closed-form Coriolis/centrifugal matrix using Jacobian time derivatives.

        We avoid autodiff by leveraging that for planar revolute chains:
          - Linear Jacobian columns are Jv_i[:, j] = z × (p_com_i - p_joint_j) for j <= i
          - Their time-derivative columns are d/dt Jv_i[:, j] = z × (v_com_i - v_joint_j)
          - Angular Jacobians are constant (Jw does not depend on q), hence d/dt Jw = 0

        The Coriolis/centrifugal matrix C is defined such that
            C(q, qd) @ qd = Σ_i m_i Jv_i(q)^T (d/dt Jv_i(q) @ qd)
        and we return one valid matrix realization:
            C(q, qd) = Σ_i m_i Jv_i(q)^T d/dt Jv_i(q)

        Args:
            q (Array): Joint angles, shape (N,) [rad]
            qd (Array): Joint velocities, shape (N,) [rad/s]

        Returns:
            C (Array): Coriolis/centrifugal matrix, shape (N, N)
        """
        N = self.num_links
        # Linear Jacobians at COMs and at joints (proximal ends)
        Jv = self._linear_jacobian_coms(q)  # (N, 2, N) for COMs
        Jv_tip = self._linear_jacobian_tips(q)  # (N, 2, N) for tips
        # Build joint linear Jacobians: joint 0 at origin (zero), joint j>0 equals tip of link j-1
        Jv_joints = jnp.zeros_like(Jv_tip)
        Jv_joints = Jv_joints.at[1:].set(Jv_tip[:-1])  # (N, 2, N)

        # COM and joint linear velocities
        v_com = jnp.einsum("icj,j->ic", Jv, qd)  # (N, 2)
        v_joint = jnp.einsum("icj,j->ic", Jv_joints, qd)  # (N, 2)

        # Time derivative of linear Jacobians columns: d/dt Jv_i[:, j]
        # For j <= i: z × (v_com_i - v_joint_j); else 0
        dv = v_com[:, None, :] - v_joint[None, :, :]  # (N, N, 2) indexed by (i, j, xy)
        # z × a in 2D -> [-a_y, a_x]
        Jvdot_x = -dv[..., 1]
        Jvdot_y = dv[..., 0]
        # mask for j <= i
        mask = jnp.tril(jnp.ones((N, N), dtype=q.dtype))
        Jvdot = jnp.stack([Jvdot_x * mask, Jvdot_y * mask], axis=1)  # (n, 2, n)

        # Assemble C = Σ_i m_i Jv_i^T Jvdot_i  (shape (n, n))
        C = jnp.einsum("i,iaj,iak->jk", self.m, Jv, Jvdot)
        return C

    @eqx.filter_jit
    def _gravitational_force(self, q: Array) -> Array:
        """
        Compute the generalized gravitational force vector.

        The gravitational force is computed by projecting gravity effects
        through the linear velocity Jacobians:
        G(q) = -Σ_i Jv_i^T ( m_i * g )

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            G (Array): Generalized gravity vector, shape (N,) [N⋅m]
        """
        Jv = self._linear_jacobian_coms(
            q
        )  # (N,2,N) -> indices (link, component, joint)
        # Force on COM i: f_i = m_i * g
        f = self.m[:, None] * self.g[None, :]  # (N,2)
        # Gravity torque: G = - Σ_i Jv_i^T f_i (shape: (N,))
        # Correctly sum over links and Cartesian components, leaving joint index free.
        G = -jnp.einsum("icj,ic->j", Jv, f)
        return G

    @eqx.filter_jit
    def stiffness_matrix(self) -> Array:
        """
        Return the linear stiffness matrix for joint elasticity.

        Returns:
            K (Array): Stiffness matrix, shape (N, N) [N⋅m/rad]
        """
        return self.K

    @eqx.filter_jit
    def elastic_force(self, q: Array) -> Array:
        """
        Compute the generalized elastic force from joint stiffness.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            tau_el (Array): Elastic force vector τ_el = K @ q, shape (N,) [N⋅m]
        """
        tau_el = self.K @ (q - self.q_ref_k) + self.passive_elastic_force(q)
        return tau_el

    @eqx.filter_jit
    def damping_matrix(self, q: Array) -> Array:
        """
        Return the linear viscous damping matrix.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            Array: Damping matrix, shape (N, N) [N⋅m⋅s/rad]
        """
        return self.D + self.passive_damping_matrix(q)

    @eqx.filter_jit
    def forward_dynamics(
        self,
        t: Array,
        y: Array,
        actuation_args: tuple | None = None,
    ) -> Array:
        """
        Compute forward dynamics for the pendulum system in state space form.

        Solves the equation of motion: B(q) qdd + C(q,qd) qd + G(q) + K q + D qd = τ_u + τ_ext
        and returns the state derivative yd = [qd, qdd].

        Args:
            t (Array): Current time, shape () [s] (unused in autonomous system)
            y (Array): State vector [q, qd], shape (2N,) where N = num_links
                      q: joint angles [rad], qd: joint velocities [rad/s]
            actuation_args (Optional[Tuple]): Actuation specification:
                - None: u=0, tau_ext=0 (free motion)
                - (u,): control torques u, tau_ext=0
                - (u, tau_ext): both control and external torques

        Returns:
            yd (Array): State derivative yd = [qd, qdd], shape (2N,)
                   qd: joint velocities [rad/s]
                   qdd: joint accelerations [rad/s²]
        """
        q, qd = jnp.split(y, 2)

        if actuation_args is None:
            u, tau_ext = None, None
        elif len(actuation_args) == 1:
            u, tau_ext = actuation_args[0], None
        elif len(actuation_args) == 2:
            u, tau_ext = actuation_args
        else:
            raise ValueError("actuation_args must be None, (u,), or (u, tau_ext)")

        if u is None:
            u = jnp.zeros((self.num_actuators,), dtype=q.dtype)
        if tau_ext is None:
            tau_ext = jnp.zeros((self.num_links,))

        B = self.inertia_matrix(q)
        C = self.coriolis_matrix(q, qd)
        G = self._gravitational_force(q)
        D = self.damping_matrix(q)
        tau_el = self.elastic_force(q)
        tau_u = self.actuation_force(q, u, qd=qd)
        rhs = tau_u + tau_ext - C @ qd - G - tau_el - D @ qd
        qdd = jnp.linalg.solve(B, rhs)
        return jnp.concatenate([qd, qdd])

    # ---------------------
    # Energy methods
    # ---------------------

    @eqx.filter_jit
    def _gravitational_energy(self, q: Array) -> Array:
        """
        Compute the gravitational potential energy of the pendulum system.

        The gravitational energy is computed by summing the potential energy
        of each link's center of mass:
        U_G = -Σ_i m_i * g^T @ p_com_i
        where p_com_i is the center of mass position of link i.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            U_g (Array): Gravitational potential energy [J] (scalar)
        """
        p_coms = self._com_positions(q)  # (n, 2)
        # U_G = -Σ_i m_i * g^T @ p_com_i
        U_g = -jnp.sum(self.m * jnp.dot(p_coms, self.g))
        return U_g

    @eqx.filter_jit
    def _elastic_energy(self, q: Array) -> Array:
        """
        Compute the elastic potential energy stored in joint springs.

        The elastic energy is computed as:
        U_k = 0.5 * q^T @ K @ q
        where K is the joint stiffness matrix.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            U_k (Array): Elastic potential energy [J] (scalar)
        """
        U_k = 0.5 * (q - self.q_ref_k).T @ self.stiffness_matrix() @ (
            q - self.q_ref_k
        ) + self.passive_elastic_energy(q)
        return U_k
