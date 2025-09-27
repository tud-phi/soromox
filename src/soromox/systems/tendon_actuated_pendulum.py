__all__ = ["TendonActuatedPendulum"]
import equinox as eqx
import jax
from jax import Array, lax, vmap
from jax import numpy as jnp
from typing import Callable, Dict, Optional, Tuple

import soromox.utils.lie_algebra as lie
import soromox.actuation.tendon_actuation as act

from .pendulum import Pendulum


class TendonActuatedPendulum(Pendulum):
    """
    Planar N-link serial pendulum (revolute chain) with both active and passive
    tendons, in closed-form JAX dynamics.

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

    Additionaly, it is possible to provide the parameters of the tendon actuation by
    specifying matrices R_at and R_pt, denoting the radii of the pulleys that route
    the active and passive tendons respectively. The passive tendons are attached to
    a spring-damper system at the base, therefore requiring the user to specify the
    stiffness and damping coefficients via vectors k_pt and d_pt.

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

    The active tendons are considered inextensible, leading to a contribution via the
    actuation matrix tau_at = A_at u.

    Also the passive tendons act on the system based on the actuation matrix A_pt that
    is defined by the routings, and introduce additional stiffness and damping based on
    the current configuration, as tau_el_pt = A_pt K_pt (R_pt q + l_pt0) and taul_d_pt =
    A_pt D_pt R_pt qd.

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
    m, I, L, Lc : Array (shape (N,))
        Physical link properties.
    g : Array (shape (2,))
        Planar gravity vector [g_x, g_y].
    K, D : Array (shape (N, N))
        Optional joint stiffness and damping matrices (zeros if omitted).
    R_at: Array (shape (Na, N))
        Optional routing matrix of the active tendons, which entries represent the radius of the pulley
        at each joint that route the cable (identity if omitted). If a tendon is attached at the i-th joint
        with i < N, then the entries for that tendon at index > i must be equal to 0.
    R_pt: Array (shape (Np, N))
        Optional routing matrix of the passive tendons (zeros if omitted).
    A_at: Array (shape (N, Na))
        Optional actuation matrix of the active tendons (identity if omitted).
    A_pt: Array (shape (N, Np))
        Optional actuation matrix of the passive tendons (zeros if omitted).
    K_pt: Array (shape (Np, Np))
        Optional stiffness matrix of the passive tendons (zeros if omitted).
    D_pt: Array (shape (Np, Np))
        Optional damping matrix of the passive tendons (zeros if omitted).
    l_pt0: Array (shape (Np,))
        Optional vector of the initial length of the passive tendons (zeros if omitted).
    tau_pt0: Array (shape (Np,))
        Vector of the elastic force due to the pre-stretch of the passive tendons (zeros if l_pt0 is omitted).
    h_q: Array (shape (N, N))
        Transformation matrix from actuation to configuration coordinates, such that q = h_q(y) (identity if omitted).
    h_q_inv: Array (shape (N, N))
        Transformation matrix from configuration to actuation coordinates, such that y = h_q_inv(q) (identity if omitted).
    """

    R_at: Array
    A_at: Array
    R_pt: Array
    A_pt: Array
    K_pt: Array
    D_pt: Array
    l_pt0: Array
    tau_pt0: Array
    h_q: Array
    h_q_inv: Array

    def __init__(
        self,
        params: Dict[str, Array],
        tendon_params: Dict[str, Array] = {},
        *args,
        **kwargs,
    ):
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
            tendon_params: Dictionary with tendon parameters with keys
                - "R_at": radii of the pulleys that route the active tendons (Na,N) (optional)
                - "R_pt": radii of the pulleys that route the passive tendons (Np,N) (optional)
                - "k_pt": stiffnesses of the springs attached to the passive tendons (Np,) (optional)
                - "d_pt": damping coefficients of the dampers attached to the passive tendons (Np,) (optional)
                - "l_pt0": initial length of the passive tendons (Np,) (optional)
        """
        super().__init__(params)

        N = self.num_links

        # Actuation matrix of active tendons
        self.R_at = jnp.asarray(tendon_params.get("R_at", jnp.identity(N)))
        self.A_at = self.R_at.T
        self.num_actuators = self.R_at.shape[0]

        # Actuation matrix of passive tendons
        self.R_pt = jnp.asarray(tendon_params.get("R_pt", jnp.zeros((N, N))))
        self.A_pt = self.R_pt.T
        Np = len(self.R_pt)

        # Stiffness matrix of the passive tendons
        kp_t = jnp.asarray(tendon_params.get("k_pt", jnp.zeros(Np)))
        self.K_pt = jnp.diag(kp_t)

        # Damping amtrix of the passive tendons
        d_pt = jnp.asarray(tendon_params.get("d_pt", jnp.zeros(Np)))
        self.D_pt = jnp.diag(d_pt)

        # Initial length of the passive tendons
        self.l_pt0 = jnp.asarray(tendon_params.get("l_pt0", jnp.zeros(Np)))

        # Elastic force due to pre-stretch of the passive tendons
        self.tau_pt0 = self.A_pt @ self.K_pt @ self.l_pt0

        # Set actuation coordinates transformation
        h_q, h_q_inv = self._set_conf2act_tranformation(self.A_at)
        self.h_q = h_q
        self.h_q_inv = h_q_inv

        # Consistency checks (lightweight; JIT friendly if shapes static)
        assert (
            self.R_pt.shape[0] == Np
            and self.K_pt.shape[0] == Np
            and self.D_pt.shape[0] == Np
            and self.l_pt0.shape[0] == Np
        ), "Mismatch among the size of the tendon parameters."

        assert self.R_at.shape[1] == N and self.R_pt.shape[1] == N, (
            f"R_at and R_pt must have second dimension equal to the number of links = {N}; "
            + f"got R_at = (., {self.R_at.shape[1]}), R_pt = (., {self.R_pt.shape[1]})."
        )

        assert jnp.linalg.matrix_rank(self.R_at) == self.R_at.shape[0], (
            f"R_at must be full rank. Got shape(R_at) = {self.R_at.shape}, "
            + f"rank(R_at) = {jnp.linalg.matrix_rank(self.R_at)}."
        )

        assert "R_pt" not in tendon_params or jnp.linalg.matrix_rank(self.R_pt) == Np, (
            f"R_pt must be full rank. Got shape(R_pt) = {self.R_pt.shape}, "
            + f"rank(R_pt) = {jnp.linalg.matrix_rank(self.R_pt)}."
        )

        assert "R_at" not in tendon_params or self._check_routing_feasibility(
            self.R_at
        ), (
            "Tendons cannot skip joints. For each row in R_at, after the first zero all entries must be zero."
        )

        assert "R_pt" not in tendon_params or self._check_routing_feasibility(
            self.R_pt
        ), (
            "Tendons cannot skip joints. For each row in R_pt, after the first zero all entries must be zero."
        )

        assert ("R_pt" in tendon_params and "k_pt" in tendon_params) or (
            "R_pt" not in tendon_params and "k_pt" not in tendon_params
        ), (
            "'R_pt' and 'k_pt' are mutually dependent. If one of them is specified, the other must be specified too."
        )

    def _check_routing_feasibility(self, A: Array) -> bool:
        """
        Returns True iff for each row of A, all non-zero entries appear
        before the first zero entry. After the first zero, all entries must be zero.
        This ensures that the adopted assumptions on the tendon kinematics are satisfied.

        Args:
            A (Array): matrix to be checked (general 2D shape).

        Returns:
            flag (bool): True if the condition is satisfied for all rows.
        """
        # Boolean mask
        zero_mask = A == 0

        # Index of first zero in each row (if none, set to ncols)
        ncols = A.shape[1]
        col_idx = jnp.arange(ncols)[None, :]
        first_zero = jnp.where(
            jnp.any(zero_mask, axis=1),
            jnp.argmax(zero_mask, axis=1),  # argmax of boolean finds first True
            ncols,
        )

        # Mask everything after the first zero per row
        after_first_zero = col_idx >= first_zero[:, None]

        # Condition: all entries after first zero must be zero
        cond = jnp.all(A == 0, where=after_first_zero, axis=1)

        # Flag
        flag = bool(jnp.all(cond))
        return flag

    def update_tendon_params(
        self, tendon_params: Dict[str, Array]
    ) -> "TendonActuatedPendulum":
        """
        Update the tendon parameters and return a new instance (functional style).

        This method creates a new TendonActuatedPendulum instance with updated parameters
        while preserving the original instance (immutable update pattern for JAX compatibility).

        Args:
           tendon_params: Dictionary with tendon parameters with keys
                - "R_at": radii of the pulleys that route the active tendons (Na,N) (optional)
                - "R_pt": radii of the pulleys that route the passive tendons (Np,N) (optional)
                - "k_pt": stiffnesses of the springs attached to the passive tendons (Np,) (optional)
                - "d_pt": damping coefficients of the dampers attached to the passive tendons (Np,) (optional)
                - "l_pt0": initial length of the passive tendons (Np,) (optional)

        Returns:
            TendonActuatedPendulum: new instance with updated parameters.
        """
        updated = self
        if "R_at" in tendon_params:
            R_at = jnp.asarray(tendon_params["R_at"])
            h_q, h_q_inv = self._set_conf2act_tranformation(R_at.T)
            updated = eqx.tree_at(lambda x: x.R_at, updated, R_at)
            updated = eqx.tree_at(lambda x: x.A_at, updated, R_at.T)
            updated = eqx.tree_at(lambda x: x.h_q, updated, h_q)
            updated = eqx.tree_at(lambda x: x.h_q_inv, updated, h_q_inv)
        if "R_pt" in tendon_params:
            R_pt = jnp.asarray(tendon_params["R_pt"])
            updated = eqx.tree_at(lambda x: x.R_pt, updated, R_pt)
            updated = eqx.tree_at(lambda x: x.A_pt, updated, R_pt.T)
        if "k_pt" in tendon_params:
            k_pt = jnp.asarray(tendon_params["k_pt"])
            updated = eqx.tree_at(lambda x: x.K_pt, updated, jnp.diag(k_pt))
        if "d_pt" in tendon_params:
            d_pt = jnp.asarray(tendon_params["d_pt"])
            updated = eqx.tree_at(lambda x: x.D_pt, updated, jnp.diag(d_pt))
        if "l_pt0" in tendon_params:
            l_pt0 = jnp.asarray(tendon_params["l_pt0"])
            tau_pt0 = updated.A_pt @ updated.K_pt @ updated.l_pt0
            updated = eqx.tree_at(lambda x: x.l_pt0, updated, l_pt0)
            updated = eqx.tree_at(lambda x: x.tau_pt0, updated, tau_pt0)
        return updated

    # -------------------------------------------------
    # Internal helpers (geometry & Jacobians, pure JAX)
    # -------------------------------------------------
    @eqx.filter_jit
    def passive_tendon_length(self, q: Array) -> Array:
        """
        Compute the length of the passive tendons.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            l_p (Array): length of the passive tendons, shape (Np,) [m]
        """
        l_p = self.R_pt @ q + self.l_pt0
        return l_p

    # -------------------------------
    # Standardized dynamics interface
    # -------------------------------
    @eqx.filter_jit
    def stiffness_matrix(self) -> Array:
        """
        Return the linear stiffness matrix for joint elasticity plus the stiffness from the
        springs of the passive tendons.

        Returns:
            K_tot (Array): Stiffness matrix, shape (N, N) [N⋅m/rad]
        """
        K_tot = self.K + self.A_pt @ self.K_pt @ self.R_pt
        return K_tot

    @eqx.filter_jit
    def elastic_force(self, q: Array) -> Array:
        """
        Compute the generalized elastic force from joint stiffness plus the elasticity from the
        springs of the passive tendons.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            tau_el_tot (Array): Total elastic force vector, shape (N,) [N⋅m]
        """
        tau_el_tot = self.stiffness_matrix() @ q + self.tau_pt0
        return tau_el_tot

    @eqx.filter_jit
    def damping_matrix(self) -> Array:
        """
        Return the linear viscous damping matrix inlcuding the damping of the passive tendons.

        Returns:
            D_tot (Array): Total damping matrix, shape (N, N) [N⋅m⋅s/rad]
        """
        D_tot = self.D + self.A_pt @ self.D_pt @ self.R_pt
        return D_tot

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Return the actuation matrix.

        Args:
            q (Array): Joint angles, shape (N,) [rad] (unused)

        Returns:
            A (Array): Actuation matrix, shape (N, Na)
        """
        return self.A_at

    # ---------------------
    # Energy methods
    # ---------------------
    @eqx.filter_jit
    def elastic_energy(self, q: Array) -> Array:
        """
        Compute the elastic potential energy stored in joint and tendon springs.

        The elastic energy is computed as:
        U_K = 0.5 * q^T @ K_tot @ q + 0.5 * l_pt0 @ K_pt @ l_pt0
        where K_tot is the total stiffness matrix, K_pt is the stiffness matrix of
        the tendons, and l_pt0 is the initial pre-stretch of the tendon springs.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            U_K_tot (Array): Total elastic potential energy [J] (scalar)
        """
        U_K_tot = 0.5 * q.T @ self.K @ q + 0.5 * (
            self.R_pt @ q + self.l_pt0
        ).T @ self.K_pt @ (self.R_pt @ q + self.l_pt0)
        return U_K_tot

    # ---------------------
    # Actuation coordinates
    # ---------------------
    def _set_conf2act_tranformation(self, A: Array) -> Tuple[Array, Array]:
        """
        Sets the transformation matrix between actuation (y) and configuration coordinates,
        and adds it as attribute (as it is constant).

        Args:
            A (Array): actuation matrix of the robot, shape (N, Na)

        Returns:
            h_q (Array): transformation matrix such that y = h_q(q), shape (N, N)
            h_q_inv (Array): transformation matrix such that q = h_q_inv(q), shape (N, N)
        """

        def basis_expansion(A: Array, r: int) -> Array:
            """
            Computes the basis expansion of a rectangular matrix via QR method. It is assumed that A is full rank = r.

            Args:
                A (Array): rectangular matrix, shape (n, m)
                r (int): number of columns to expand the matrix (r <= n)

            Returns:
                A_expanded (Array): square matrix which last n - r columns are linearly independent from the first m, shape (n, n)
            """
            # Rank the input matrix ASSUMED to be equal to self.num_actuators
            #r = jnp.linalg.matrix_rank(A)

            # Full orthonormal Q (n, n)
            Q_complete, _ = jnp.linalg.qr(
                A, mode="complete"
            )  # Q_complete @ R = A (R may be padded zeros)

            # Span of the orthogonal complement
            extra = Q_complete[:, r:]  # (n, n - r)

            # Expanded input matrix with n - r linearly independent columns
            A_expanded = jnp.concatenate([A, extra], axis=1)  # (n, m + (n - r))

            return A_expanded

        # Compute basis expansion of the actuation matrix
        A_at_exp = basis_expansion(self.A_at, self.num_actuators)  # (N, N)

        # Define change of coordinates between configuration (q) and actuation space (y)
        h_q_inv = A_at_exp.T
        h_q = jnp.linalg.inv(h_q_inv)

        return h_q, h_q_inv

    def conf2act_coords(self, q: Array) -> Array:
        """
        Compute the actuation coordinates corresponding to the input
        configuration coordinates.

        Args:
            q (Array): configuration coordinates, shape (N,)

        Returns:
            y (Array): actuation coordinates, shape (N,)
        """
        return self.h_q_inv @ q

    def act2conf_coords(self, y: Array) -> Array:
        """
        Compute the configuration coordinates corresponding to the input
        configuration coordinates.

        Args:
            y (Array): actuation coordinates, shape (N,)

        Returns:
            q (Array): configuration coordinates, shape (N,)
        """
        return self.h_q @ y

    def conf2act_jacobian(self, q: Array):
        """
        Jacobian of the configuration to actuation map.

        Args:
            q (Array): configuration coordinates, shape (N,)

        Returns:
            J_h_q_inv (Array): jacobian, shape (N,)
        """
        J_h_q_inv = self.h_q_inv
        return J_h_q_inv

    def act2conf_jacobian(self, y: Array):
        """
        Jacobian of the actuation to configuration map, J_(h_q)^{-1} = d (h_q)^{-1}(y) / d y

        Args:
            y (Array): actuation coordinates, shape (N,)

        Returns:
            J_h_q (Array): jacobian, shape (N,)
        """
        J_h_q = self.h_q
        return J_h_q

    def actuation_space_dynamics(
        self,
        q: Array,
        qd: Array,
    ) -> Tuple[Array, Array, Array, Array]:
        """
        Compute the actuation space dynamical matrices, which can be expressed as
            M_y(q) @ ydd + eta_y(q, qd) @ yd + JhM_pinv.T @ (G(q) + K @ q + D @ qd) = [tau, 0_{n - m}],
        where ydd, yd, y are the acceleration, velocity, and configuration of the actuation coordinates,
        respectively. Similarly for qd, q in configuration coordinates.

        Args:
            q (Array): generalized coordinates of shape (N,).
            qd (Array): time-derivative of the generalized coordinates of shape (N,).

        Returns:
            M_y (Array): Inertia matrix in the actuation space, shape (N, N).
            eta_y (Array): Coriolis and centrifugal matrix in the actuation space, shape (N,).
            J (Array): Jacobian of the h_q map, shape (N, N).
            JhM_pinv (Array): Dynamically-consistent pseudo-inverse of the Jacobian, shape (N, N).
        """
        # Jacobians
        J = self.conf2act_jacobian(q)
        # Even J_inv = self.conf2act_jacobian(y=h_q_inv(q)) in this case becuase it is linear, it is in general not true.
        # Below is reported the right computation in the general case (when (J_h_q)^{-1} != J_(h_q)^{-1})
        J_inv = jnp.linalg.inv(J)

        # Inertia matrix
        M = self.inertia_matrix(q)
        M_inv = jnp.linalg.inv(M)

        # Coriolis
        C = self.coriolis_matrix(q, qd)

        # Actuation space dynamics
        M_y = jnp.linalg.inv(J_inv.T @ M @ J_inv)
        eta_y = M_y @ (J @ M_inv @ C) @ J_inv
        JhM_pinv = J_inv

        return M_y, eta_y, J, JhM_pinv
