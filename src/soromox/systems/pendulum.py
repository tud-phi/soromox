__all__ = ["Pendulum", "normalize_joint_angles"]
import equinox as eqx
import jax
from jax import Array, jit, lax
from jax import numpy as jnp
import numpy as onp
from typing import Dict, List, Tuple, Union, Optional

from soromox.systems.dynamical_system import DynamicalSystem


class Pendulum(DynamicalSystem):
    """
    Planar N-link serial pendulum (revolute chain) with closed-form JAX dynamics.

    This class provides a lightweight rigid-body model of an N-link planar pendulum
    without relying on pre-generated symbolic expressions. Kinematics and dynamics
    are computed on-the-fly using Jacobian-based rigid-body formulas that are fully
    JAX compatible (supporting JIT, vmap, grad, etc.).

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

    Attributes
    ----------
    num_links : int (static)
        Number of links / DoFs.
    num_actuators: int (static)
        Number of joint actuators. Equal to num_links.
    m, I, L, Lc : Array (shape (N,))
        Physical link properties.
    g : Array (shape (2,))
        Planar gravity vector [g_x, g_y].
    K, D : Array (shape (N, N))
        Optional joint stiffness and damping matrices (zeros if omitted).
    q_ref_k :  Array (shape (N,))
        Optional rest configuration of the torsional springs (defualt is the zero configuration)
    """

    # Static field
    num_links: int = eqx.field(static=True)

    # Stored physical parameters
    m: Array
    I: Array
    L: Array
    Lc: Array
    g: Array  # planar gravity acceleration vector

    # Optional linear elasticity/damping to make it an articulated soft robot.
    K: Array
    D: Array
    q_ref_k: Array

    def __init__(
        self,
        params: Dict[str, Array],
    ) -> None:
        """
        Initialize the pendulum system with the given parameters.
        Args:
            params: Dictionary with robot parameters with keys
                - "m": Masses of the links (N,)
                - "I": Planar inertias about COM (N,)
                - "L": Link lengths (N,)
                - "Lc": COM offset from prior joint (N,)
                - "g": Planar gravity vector (2,)
                - "K": stiffness matrix (N,N) (optional)
                - "D": damping matrix (N,N) (optional)
                - "q_ref_k": rest configuration of the torsional springs defined in K (N,) (optional)
        """
        # Basic parameter extraction
        m = jnp.asarray(params["m"])  # (n,)
        I = jnp.asarray(params["I"])  # (n,)
        L = jnp.asarray(params["L"])  # (n,)
        Lc = jnp.asarray(params["Lc"])  # (n,)
        g = jnp.asarray(params["g"])  # (2,)

        n_q = m.shape[0]
        # Consistency checks (lightweight; JIT friendly if shapes static)
        assert I.shape[0] == n_q and L.shape[0] == n_q and Lc.shape[0] == n_q, (
            "Parameter length mismatch"
        )
        assert g.shape[0] == 2, "Gravity vector must be length 2"

        self.num_links = int(n_q)
        self.num_actuators = self.num_links

        # set parameters
        self.m = m
        self.I = I
        self.L = L
        self.Lc = Lc
        self.g = g

        self.K = jnp.asarray(params.get("K", jnp.zeros((n_q, n_q))))
        self.D = jnp.asarray(params.get("D", jnp.zeros((n_q, n_q))))
        self.q_ref_k = jnp.asarray(params.get("q_ref_k", jnp.zeros((n_q,))))

    def update_params(self, params: Dict[str, Array]) -> "Pendulum":
        """
        Update the system parameters and return a new instance (functional style).

        This method creates a new Pendulum instance with updated parameters while
        preserving the original instance (immutable update pattern for JAX compatibility).

        Args:
            params (Dict[str, Array]): Dictionary with robot parameters. Supported keys:
                - "m": Masses of the links, shape (N,) [kg]
                - "I": Planar inertias about COM, shape (N,) [kg⋅m²]
                - "L": Link lengths, shape (N,) [m]
                - "Lc": COM offset from proximal joint, shape (N,) [m]
                - "g": Planar gravity vector, shape (2,) [m/s²]
                - "K": Joint stiffness matrix, shape (N,N) [N⋅m/rad] (optional)
                - "D": Joint damping matrix, shape (N,N) [N⋅m⋅s/rad] (optional)
                - "q_ref_k": rest configuration of the torsional springs defined in K (N,) [rad] (optional)

        Returns:
            Pendulum: New instance with updated parameters.
        """
        updated = self
        if "m" in params:
            updated = eqx.tree_at(lambda x: x.m, updated, jnp.asarray(params["m"]))
        if "I" in params:
            updated = eqx.tree_at(lambda x: x.I, updated, jnp.asarray(params["I"]))
        if "L" in params:
            updated = eqx.tree_at(lambda x: x.L, updated, jnp.asarray(params["L"]))
        if "Lc" in params:
            updated = eqx.tree_at(lambda x: x.Lc, updated, jnp.asarray(params["Lc"]))
        if "g" in params:
            updated = eqx.tree_at(lambda x: x.g, updated, jnp.asarray(params["g"]))
        if "K" in params:
            updated = eqx.tree_at(lambda x: x.K, updated, jnp.asarray(params["K"]))
        if "D" in params:
            updated = eqx.tree_at(lambda x: x.D, updated, jnp.asarray(params["D"]))
        if "q_ref_k" in params:
            updated = eqx.tree_at(lambda x: x.q_ref_k, updated, jnp.asarray(params["q_ref_k"]))
        return updated

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
        return jnp.cumsum(q)  # (n,)

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
        zeros = jnp.zeros((1, 2), dtype=q.dtype)
        p_joints = jnp.concatenate([zeros, tips[:-1]], axis=0)  # (n,2)
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
        p_tips = jnp.cumsum(seg_vecs, axis=0)  # (n,2)
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

    def _linear_jacobians_coms(self, q: Array) -> Array:
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

    def _linear_jacobians_tips(self, q: Array) -> Array:
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
    def jacobians_joints(self, q: Array) -> Array:
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
        Jv_tip = self._linear_jacobians_tips(q)  # (N,2,N)
        # Build linear Jacobians at joints: i=0 -> zeros; i>0 -> tip of (i-1)
        Jv_joints = jnp.zeros_like(Jv_tip)
        Jv_joints = Jv_joints.at[1:].set(Jv_tip[:-1])

        Jw = self._angular_jacobians()  # (N,N)
        # Order rows as [Jw; Jv_x; Jv_y]
        J_joints = jnp.concatenate([Jw[:, None, :], Jv_joints], axis=1)  # (N,3,N)
        return J_joints

    @eqx.filter_jit
    def jacobians_tips(self, q: Array) -> Array:
        """
        Spatial Jacobian for the distal tip of a specified link.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            J_tips (Array): array of spatial Jacobians with shape (N, 3, N), where for each i:
                   J_tips[i] = [Jw; Jv_x; Jv_y] with shape (3, N) at tip i.
        """
        Jv_all_tip = self._linear_jacobians_tips(q)  # (N,2,N)
        Jw_all = self._angular_jacobians()  # (N,N)
        J_tips = jnp.concatenate([Jw_all[:, None, :], Jv_all_tip], axis=1)
        return J_tips

    @eqx.filter_jit
    def jacobians_and_derivatives_tips(
        self, q: Array, qd: Array
    ) -> Tuple[Array, Array]:
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
        J_tips = self.jacobians_tips(q)
        # the linear part of the jacobian
        Jv_tips = J_tips[:, 1:, :]

        # compute the jacobians at the joints
        J_joints = self.jacobians_joints(q)
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
    def jacobians_coms(self, q: Array) -> Array:
        """
        Spatial Jacobian for the center of mass (COM) of a specified link.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            J_coms (Array): array of spatial Jacobians with shape (N, 3, N), where for each i:
                   J_coms[i] = [Jw; Jv_x; Jv_y] with shape (3, N) at COM i.
        """
        Jv_all_com = self._linear_jacobians_coms(q)  # (N,2,N)
        Jw_all = self._angular_jacobians()  # (N,N)
        J_coms = jnp.concatenate([Jw_all[:, None, :], Jv_all_com], axis=1)
        return J_coms

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
        Jv = self._linear_jacobians_coms(q)  # (n,2,n)
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
        Jv = self._linear_jacobians_coms(q)  # (N, 2, N) for COMs
        Jv_tip = self._linear_jacobians_tips(q)  # (N, 2, N) for tips
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
    def gravitational_force(self, q: Array) -> Array:
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
        Jv = self._linear_jacobians_coms(
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
        tau_el = self.K @ (q - self.q_ref_k)
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
        return self.D

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Return the actuation matrix (identity for direct joint torques).

        Args:
            q (Array): Joint angles, shape (N,) [rad] (unused for pendulum)

        Returns:
            A (Array): Actuation matrix (identity), shape (N, N)
        """
        A = jnp.eye(self.num_links)
        return A

    def actuation_force(self, q: Array, u: Array) -> Array:
        """
        Compute generalized actuation forces from control inputs.

        Args:
            q (Array): Joint angles, shape (N,) [rad] (unused)
            u (Array): Control torques, shape (N,) [N⋅m]

        Returns:
            tau_u (Array): Generalized actuation forces τ_u = u, shape (N,) [N⋅m]
        """
        tau_u = self.actuation_matrix(q) @ u
        return tau_u

    @eqx.filter_jit
    def forward_dynamics(
        self,
        t: Array,
        y: Array,
        actuation_args: Optional[Tuple] = None,
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
            u = jnp.zeros((self.num_links,))
        if tau_ext is None:
            tau_ext = jnp.zeros((self.num_links,))

        B = self.inertia_matrix(q)
        C = self.coriolis_matrix(q, qd)
        G = self.gravitational_force(q)
        D = self.damping_matrix(q)
        tau_el = self.elastic_force(q)
        tau_u = self.actuation_force(q, u)
        rhs = tau_u + tau_ext - C @ qd - G - tau_el - D @ qd
        qdd = jnp.linalg.solve(B, rhs)
        return jnp.concatenate([qd, qdd])

    # ---------------------
    # Energy methods
    # ---------------------
    @eqx.filter_jit
    def kinetic_energy(self, q: Array, qd: Array) -> Array:
        """
        Compute the kinetic energy of the pendulum system.

        The kinetic energy is computed as:
        T = 0.5 * qd^T @ B(q) @ qd
        where B(q) is the generalized inertia matrix.

        Args:
            q (Array): Joint angles, shape (N,) [rad]
            qd (Array): Joint velocities, shape (N,) [rad/s]

        Returns:
            T (Array): Kinetic energy [J] (scalar)
        """
        B = self.inertia_matrix(q)
        T = 0.5 * qd.T @ B @ qd
        return T

    @eqx.filter_jit
    def gravitational_energy(self, q: Array) -> Array:
        """
        Compute the gravitational potential energy of the pendulum system.

        The gravitational energy is computed by summing the potential energy
        of each link's center of mass:
        U_G = -Σ_i m_i * g^T @ p_com_i
        where p_com_i is the center of mass position of link i.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            U_G (Array): Gravitational potential energy [J] (scalar)
        """
        p_coms = self._com_positions(q)  # (n, 2)
        # U_G = -Σ_i m_i * g^T @ p_com_i
        U_G = -jnp.sum(self.m * jnp.dot(p_coms, self.g))
        return U_G

    @eqx.filter_jit
    def elastic_energy(self, q: Array) -> Array:
        """
        Compute the elastic potential energy stored in joint springs.

        The elastic energy is computed as:
        U_K = 0.5 * q^T @ K @ q
        where K is the joint stiffness matrix.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            U_K (Array): Elastic potential energy [J] (scalar)
        """
        U_K = 0.5 * (q - self.q_ref_k).T @ self.stiffness_matrix() @ (q - self.q_ref_k)
        return U_K

    @eqx.filter_jit
    def potential_energy(self, q: Array) -> Array:
        """
        Compute the total potential energy of the pendulum system.

        The potential energy is the sum of gravitational and elastic energy:
        U = U_G + U_K

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            U (Array): Total potential energy [J] (scalar)
        """
        U_G = self.gravitational_energy(q)
        U_K = self.elastic_energy(q)
        U = U_G + U_K
        return U

    @eqx.filter_jit
    def total_energy(self, q: Array, qd: Array) -> Array:
        """
        Compute the total energy of the pendulum system.

        The total energy is the sum of kinetic and potential energy:
        E = T + U

        Args:
            q (Array): Joint angles, shape (N,) [rad]
            qd (Array): Joint velocities, shape (N,) [rad/s]

        Returns:
            E (Array): Total energy [J] (scalar)
        """
        T = self.kinetic_energy(q, qd)
        U = self.potential_energy(q)
        E = T + U
        return E

    # --------------------------
    # Operational space dynamics
    # --------------------------
    @eqx.filter_jit
    def operational_space_dynamical_matrices(
        self,
        q: Array,
        qd: Array,
        link_idx: Array,
        operational_space_selector: Tuple = (True, True, True),
    ) -> Tuple[Array, Array, Array, Array, Array]:
        """
        Compute the operational space dynamical matrices for the tip of the specified link.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).
            link_idx (Array): index of the link to compute the operational space dynamics for.
            operational_space_selector (Tuple): Selector for the operational space dimensions.
                Default is (True, True, True) for all dimensions.

        Returns:
            Lambda (Array): Inertia matrix in the operational space, shape (num_operational_space_dims, num_operational_space_dims).
            mu (Array): Coriolis and centrifugal matrix in the operational space, shape (num_operational_space_dims,).
            J (Array): Jacobian of the forward kinematics at point s in the body frame, shape (num_operational_space_dims, num_active_strains).
            Jd (Array): Time-derivative of the Jacobian at point s in the body frame, shape (num_operational_space_dims, num_active_strains).
            JB_pinv (Array): Dynamically-consistent pseudo-inverse of the Jacobian, shape (num_active_strains, num_operational_space_dims).
        """
        # make operational_space_selector a boolean array
        operational_space_selector = onp.array(operational_space_selector, dtype=bool)

        # Jacobian and its time-derivative
        J_tips, Jd_tips = self.jacobians_and_derivatives_tips(q, qd)

        # select the appropiate link
        J = J_tips[link_idx]
        Jd = Jd_tips[link_idx]

        J = J[operational_space_selector, :]
        Jd = Jd[operational_space_selector, :]

        # inverse of the inertia matrix in the configuration space
        B = self.inertia_matrix(q)
        B_inv = jnp.linalg.inv(B)
        C = self.coriolis_matrix(q, qd)

        Lambda = jnp.linalg.inv(
            J @ B_inv @ J.T
        )  # inertia matrix in the operational space
        mu = Lambda @ (
            J @ B_inv @ C - Jd
        )  # coriolis and centrifugal matrix in the operational space

        JB_pinv = (
            B_inv @ J.T @ Lambda
        )  # dynamically-consistent pseudo-inverse of the Jacobian

        return Lambda, mu, J, Jd, JB_pinv


@jit
def normalize_joint_angles(q: Array) -> Array:
    """
    Normalize joint angles to the interval [-pi, pi].

    Args:
        q (Array): Joint angles.

    Returns:
        q_norm (Array): Normalized joint angles.
    """
    q_norm = jnp.mod(q + jnp.pi, 2 * jnp.pi) - jnp.pi
    return q_norm
