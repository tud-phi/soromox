__all__ = ["TendonActuatedPendulum"]

from typing import Any

import equinox as eqx
from jax import Array
from jax import numpy as jnp

from soromox.systems.pendulum.params import TendonActuatedPendulumParams

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

    Attributes:
        num_links: Number of links / DoFs.
        m, I, L, Lc: Physical link properties.
        g: Planar gravity vector [g_x, g_y].
        K, D: Optional joint stiffness and damping matrices (zeros if omitted).
        R_at: Optional routing matrix of the active tendons, which entries represent the radius of the pulley
            at each joint that route the cable (identity if omitted). If a tendon is attached at the i-th joint
            with i < N, then the entries for that tendon at index > i must be equal to 0.
        R_pt: Optional routing matrix of the passive tendons (zero if omitted).
        A_at: Optional actuation matrix of the active tendons (identity if omitted).
        A_pt: Optional actuation matrix of the passive tendons (zero if omitted).
        K_pt: Optional stiffness matrix of the passive tendons (zero if omitted).
        D_pt: Optional damping matrix of the passive tendons (zero if omitted).
        l_pt0: Optional vector of the initial displacement of the passive tendons (zero if omitted).
        tau_pt0: Vector of the elastic force due to the pre-stretch of the passive tendons (zero if l_pt0 is omitted).

    References:
        Murray, R. M., Li, Z., & Sastry, S. S. (2017). A Mathematical Introduction
        to Robotic Manipulation (Chapter 6: Tendon-driven actuation). CRC Press.
    """

    params: TendonActuatedPendulumParams
    R_at: Array
    A_at: Array
    q_ref_at: Array
    R_pt: Array
    A_pt: Array
    q_ref_pt: Array
    K_pt: Array
    D_pt: Array
    l_pt0: Array
    tau_pt0: Array

    def __init__(
        self,
        params: TendonActuatedPendulumParams,
        **kwargs: Any,
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
                - "active_tendon_reference_configuration": joint configuration at which active tendon displacement is zero (N,) (optional)
                - "passive_tendon_reference_configuration": joint configuration at which passive tendon displacement is zero before rest-length offset (N,) (optional)
        """
        if not isinstance(params, TendonActuatedPendulumParams):
            raise TypeError("params must be a TendonActuatedPendulumParams instance.")
        super().__init__(params.body, **kwargs)
        self.params = params

        # Actuation matrix of active tendons
        self.R_at = jnp.asarray(params.active_routing_matrix)
        self.A_at = self.R_at.T
        self.num_actuators = self.R_at.shape[0]

        # Actuation matrix of passive tendons
        self.R_pt = jnp.asarray(params.passive_routing_matrix)
        self.A_pt = self.R_pt.T

        # Stiffness matrix of the passive tendons
        kp_t = jnp.asarray(params.passive_tendon.stiffness)
        self.K_pt = jnp.diag(kp_t)

        # Damping amtrix of the passive tendons
        d_pt = jnp.asarray(params.passive_tendon.damping)
        self.D_pt = jnp.diag(d_pt)

        # Initial displacement of the passive tendons
        self.l_pt0 = jnp.asarray(params.passive_tendon.rest_length_offset)

        # Rest configuration of the active and passive tendons
        self.q_ref_at = jnp.asarray(params.active_tendon_reference_configuration)
        self.q_ref_pt = jnp.asarray(params.passive_tendon_reference_configuration)

        # Elastic force due to pre-stretch of the passive tendons
        self.tau_pt0 = self.A_pt @ self.K_pt @ self.l_pt0

        self._validate_tendon_params(
            self.R_at,
            self.R_pt,
            kp_t,
            d_pt,
            self.l_pt0,
            self.q_ref_at,
            self.q_ref_pt,
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

    def _validate_routing_matrix(self, matrix: Array, name: str) -> None:
        """Validate shape, rank, and no-joint-skipping routing invariants."""
        if matrix.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional.")
        if matrix.shape[1] != self.num_links:
            raise ValueError(
                f"{name} must have one column per link; expected {self.num_links}, "
                f"got {matrix.shape[1]}."
            )
        num_tendons = matrix.shape[0]
        if num_tendons > 0:
            rank = int(jnp.linalg.matrix_rank(matrix))
            if rank != num_tendons:
                raise ValueError(
                    f"{name} must be full row rank. Got shape {matrix.shape} "
                    f"and rank {rank}."
                )
        if not self._check_routing_feasibility(matrix):
            raise ValueError(
                "Tendons cannot skip joints. For each row in "
                f"{name}, after the first zero all entries must be zero."
            )

    def _validate_tendon_params(
        self,
        active_routing_matrix: Array,
        passive_routing_matrix: Array,
        passive_stiffness: Array,
        passive_damping: Array,
        passive_rest_length_offset: Array,
        active_reference_configuration: Array,
        passive_reference_configuration: Array,
    ) -> None:
        """Validate tendon-actuated pendulum routing and passive tendon sizes."""
        self._validate_routing_matrix(active_routing_matrix, "active_routing_matrix")
        self._validate_routing_matrix(passive_routing_matrix, "passive_routing_matrix")
        num_passive_tendons = passive_routing_matrix.shape[0]
        if passive_stiffness.shape != (num_passive_tendons,):
            raise ValueError(
                "passive_tendon stiffness must have one entry per passive routing row."
            )
        if passive_damping.shape != (num_passive_tendons,):
            raise ValueError(
                "passive_tendon damping must have one entry per passive routing row."
            )
        if passive_rest_length_offset.shape != (num_passive_tendons,):
            raise ValueError(
                "passive_tendon rest_length_offset must have one entry per passive routing row."
            )
        if active_reference_configuration.shape != (self.num_links,):
            raise ValueError(
                "active_tendon_reference_configuration must have length equal "
                "to the number of links."
            )
        if passive_reference_configuration.shape != (self.num_links,):
            raise ValueError(
                "passive_tendon_reference_configuration must have length equal "
                "to the number of links."
            )

    def with_params(
        self, params: TendonActuatedPendulumParams
    ) -> "TendonActuatedPendulum":
        """Return an updated copy with a full typed parameter object."""
        if not isinstance(params, TendonActuatedPendulumParams):
            raise TypeError("params must be a TendonActuatedPendulumParams instance.")
        params.validate()
        if params.active_routing_matrix.shape != self.R_at.shape:
            raise ValueError(
                "active_routing_matrix shape changes the actuation layout; construct a new TendonActuatedPendulum."
            )
        if params.passive_routing_matrix.shape != self.R_pt.shape:
            raise ValueError(
                "passive_routing_matrix shape changes the passive tendon layout; construct a new TendonActuatedPendulum."
            )

        updated = self._with_pendulum_params(params.body, stored_params=params)

        R_at = jnp.asarray(params.active_routing_matrix)
        R_pt = jnp.asarray(params.passive_routing_matrix)
        k_pt = jnp.asarray(params.passive_tendon.stiffness)
        d_pt = jnp.asarray(params.passive_tendon.damping)
        l_pt0 = jnp.asarray(params.passive_tendon.rest_length_offset)
        q_ref_at = jnp.asarray(params.active_tendon_reference_configuration)
        q_ref_pt = jnp.asarray(params.passive_tendon_reference_configuration)

        if R_at.shape[1] != self.num_links or R_pt.shape[1] != self.num_links:
            raise ValueError("routing matrices must have one column per link.")
        if k_pt.shape != (R_pt.shape[0],) or d_pt.shape != (R_pt.shape[0],):
            raise ValueError(
                "passive_tendon stiffness and damping must match R_pt rows."
            )
        if l_pt0.shape != (R_pt.shape[0],):
            raise ValueError("passive_tendon rest_length_offset must match R_pt rows.")
        if q_ref_at.shape != (self.num_links,) or q_ref_pt.shape != (self.num_links,):
            raise ValueError(
                "active_tendon_reference_configuration and "
                "passive_tendon_reference_configuration must match num_links."
            )
        self._validate_tendon_params(R_at, R_pt, k_pt, d_pt, l_pt0, q_ref_at, q_ref_pt)

        A_pt = R_pt.T
        K_pt = jnp.diag(k_pt)
        tau_pt0 = A_pt @ K_pt @ l_pt0

        return eqx.tree_at(
            lambda model: (
                model.R_at,
                model.A_at,
                model.R_pt,
                model.A_pt,
                model.K_pt,
                model.D_pt,
                model.l_pt0,
                model.q_ref_at,
                model.q_ref_pt,
                model.tau_pt0,
            ),
            updated,
            (
                R_at,
                R_at.T,
                R_pt,
                A_pt,
                K_pt,
                jnp.diag(d_pt),
                l_pt0,
                q_ref_at,
                q_ref_pt,
                tau_pt0,
            ),
        )

    def update_params(self, **updates: Array) -> "TendonActuatedPendulum":
        """Return an updated copy with selected typed parameter fields replaced."""
        return self.with_params(self.params.replace(**updates))

    # -------------------------------------------------
    # Internal helpers (geometry & Jacobians, pure JAX)
    # -------------------------------------------------
    @eqx.filter_jit
    def active_tendon_displacement(self, q: Array) -> Array:
        """
        Compute the displacement of the active tendons with respect to the initial position.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            l_a (Array): displacement of the active tendons, shape (Na,) [m]
        """
        l_a = self.R_at @ (q - self.q_ref_at)
        return l_a

    @eqx.filter_jit
    def active_tendon_length(self, q: Array) -> Array:
        """
        Compute an approximation of the total length of the active tendons, considering
        that the tendon length at the initial position is equal to the cumulative length
        of the links until the attachment point, when the actuation matrix is user-defined.
        When the actuation matrix is not specified by the user, the tendon length is
        set to the default value of zero.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            l_tot_a (Array): length of the active tendons, shape (Na,) [m]
        """
        cond = jnp.logical_not(jnp.array_equal(self.R_at, jnp.eye(self.num_links)))
        mask = jnp.abs(self.R_at) > 0.0
        l0_a = jnp.sum(self.L * mask * cond, axis=1)
        l_tot_a = self.active_tendon_displacement(q) + l0_a
        return l_tot_a

    @eqx.filter_jit
    def passive_tendon_displacement(self, q: Array) -> Array:
        """
        Compute the displacement of the passive tendons.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            l_p (Array): displacement of the passive tendons, shape (Np,) [m]
        """
        l_p = self.R_pt @ (q - self.q_ref_pt) + self.l_pt0
        return l_p

    @eqx.filter_jit
    def passive_tendon_length(self, q: Array) -> Array:
        """
        Compute an approximation of the total length of the passive tendons, considering
        that the tendon length at the initial position is equal to the cumulative length
        of the links until the attachment point, when the actuation matrix is user-defined.
        When the actuation matrix is not specified by the user, the tendon length is
        set to the default value of zero.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            l_tot_p (Array): length of the passive tendons, shape (Np,) [m]
        """
        cond = jnp.logical_not(
            jnp.array_equal(self.R_pt, jnp.zeros((self.num_links, self.num_links)))
        )
        mask = jnp.abs(self.R_pt) > 0.0
        l0_p = jnp.sum(self.L * mask * cond, axis=1)
        l_tot_p = self.passive_tendon_displacement(q) + l0_p
        return l_tot_p

    tendon_length = active_tendon_length  # Alias for compatibility
    actuated_coordinates = active_tendon_length  # Alias for actuation space dynamics

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
    def damping_matrix(self, q: Array) -> Array:
        """
        Return the linear viscous damping matrix inlcuding the damping of the passive tendons.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

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
    def _elastic_energy(self, q: Array) -> Array:
        """
        Compute the elastic potential energy stored in joint and tendon springs.

        The elastic energy is computed as:
        U_K = 0.5 * q^T @ K_tot @ q + 0.5 * l_pt0 @ K_pt @ l_pt0
        where K_tot is the total stiffness matrix, K_pt is the stiffness matrix of
        the tendons, and l_pt0 is the initial pre-stretch of the tendon springs.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            U_k (Array): Total elastic potential energy [J] (scalar)
        """
        U_k = 0.5 * q.T @ self.K @ q + 0.5 * (
            self.R_pt @ q + self.l_pt0
        ).T @ self.K_pt @ (self.R_pt @ q + self.l_pt0)
        return U_k
