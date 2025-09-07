__all__ = ["TendonActuatedPendulum"]
import equinox as eqx
import jax
from jax import Array, lax, vmap
from jax import numpy as jnp
from typing import Callable, Dict, Optional

import soromox.utils.lie_algebra as lie
import soromox.actuation.tendon_actuation as act

from .pendulum import Pendulum


class TendonActuatedPendulum(Pendulum):
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
    m, I, L, Lc : Array (shape (N,))
        Physical link properties.
    g : Array (shape (2,))
        Planar gravity vector [g_x, g_y].
    K, D : Array (shape (N, N))
        Optional joint stiffness and damping matrices (zeros if omitted).
    Ra: Array (shape (Na, N))
        Optional routing matrix of the active tendons, which entries represent the radius of the pulley
        at each joint that route the cable (identity if omitted). If a tendon is attached at the i-th joint
        with i < N, then the entries for that tendon at index > i must be equal to 0.
    Rp: Array (shape (Np, N))
        Optional routing matrix of the passive tendons (zeros if omitted).
    A: Array (shape (N, Na))
        Optional actuation matrix of the active tendons (identity if omitted).
    Ap: Array (shape (N, Np))
        Optional actuation matrix of the passive tendons (zeros if omitted).
    Kp: Array (shape (N, Np))
        Optional stiffness matrix of the passive tendons (zeros if omitted).
    Dp: Array (shape (N, N))
        Optional damping matrix of the passive tendons (zeros if omitted).
    Lp0: Array (shape (Np, Np)
        Optional vector of the initial length of the passive tendons (zeros if omitted).
    """
    Ra: Array #= eqx.field(static=True)
    A: Array #= eqx.field(static=True)
    Rp: Array #= eqx.field(static=True)
    Ap: Array #= eqx.field(static=True)
    Kp: Array #= eqx.field(static=True)
    Dp: Array #= eqx.field(static=True)
    Lp0: Array #= eqx.field(static=True)

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
                - "Ra": radii of the pulleys that route the active tendons (Na,N) (optional)
                - "Rp": radii of the pulleys that route the passive tendons (Np,N) (optional)
                - "Kp": stiffnesses of the springs attached to the passive tendons (Np,) (optional)
                - "Dp": damping coefficients of the dampers attached to the passive tendons (Np,) (optional)
                - "Lp0": initial length of the passive tendons (Np,) (optional)
        """
        super().__init__(
            params
        )

        N = self.num_links

        # Actuation matrix of active tendons
        self.Ra = jnp.asarray(tendon_params.get("Ra", jnp.identity(N)))
        self.A = self.Ra.T

        # Actuation matrix of passive tendons
        self.Rp = jnp.asarray(tendon_params.get("Rp", jnp.identity(N)))
        self.Ap = self.Rp.T
        Np = len(self.Rp)

        # Stiffness matrix of the passive tendons
        kp = jnp.asarray(tendon_params.get("Kp", jnp.zeros(Np)))
        self.Kp = jnp.diag(kp)

        # Damping amtrix of the passive tendons
        dp = jnp.asarray(tendon_params.get("Dp", jnp.zeros(Np)))
        self.Dp = jnp.diag(dp)

        # Initial length of the passive tendons
        self.Lp0 = jnp.asarray(tendon_params.get("Lp0", jnp.zeros(Np)))

        # Consistency checks (lightweight; JIT friendly if shapes static)
        assert self.Rp.shape[0] == Np and self.Kp.shape[0] == Np and self.Dp.shape[0] == Np and self.Lp0.shape[0] == Np, (
            "Mismatch among the size of the tendon parameters."
        )

        assert self.Ra.shape[1] == N and self.Rp.shape[1] == N, (
            f"Ra and Rp must have second dimension equal to the number of links = {N}; " + \
                f"got Ra = (., {self.Ra.shape[1]}), Rp = (., {self.Rp.shape[1]})."
        )

        assert jnp.linalg.matrix_rank(self.Ra) == self.Ra.shape[0] and jnp.linalg.matrix_rank(self.Rp) == Np, (
            f"Ra and Rp must be full rank. Got shape(Ra) = {self.Ra.shape}, " + \
                f"rank(Ra) = {jnp.linalg.matrix_rank(self.Ra)}; shape(Rp) = {self.Rp.shape}, rank(Rp) = " + \
                f"{jnp.linalg.matrix_rank(self.Rp)}."
        )

        # assert self._check_routing_feasibility(self.Ra) and self._check_routing_feasibility(self.Rp), (
        #     "Tendons cannot skip joints. For each row in Ra and Rp, after the first zero " + \
        #         "all entries must be zero."
        # )


    def _check_routing_feasibility(self, A: jnp.ndarray) -> bool:
        """
        Returns True iff for each row of A, all non-zero entries appear
        before the first zero entry. After the first zero, all entries must be zero.
        This ensures that the adopted assumptions on the tendon kinematics are satisfied.

        Args:
            A (Array): matrix to be checked (general 2D shape).

        Returns:
            flag (bool): True if the condition is satisfied for all rows.
        """
        # Boolean mask: where A == 0
        zero_mask = (A == 0)
        
        # Index of first zero in each row (if none, set to ncols)
        ncols = A.shape[1]
        col_idx = jnp.arange(ncols)[None, :]
        first_zero = jnp.where(
            jnp.any(zero_mask, axis=1),
            jnp.argmax(zero_mask, axis=1),  # argmax of boolean finds first True
            ncols
        )
        
        # Mask everything after the first zero per row
        after_first_zero = col_idx >= first_zero[:, None]
        
        # Condition: all entries after first zero must be zero
        cond = jnp.all(A == 0, where=after_first_zero, axis=1)

        # Flag
        flag = bool(jnp.all(cond))
        return flag

    def update_tendon_params(self, tendon_params: Dict[str, Array]) -> "TendonActuatedPendulum":
        """
        Update the tendon parameters and return a new instance (functional style).

        This method creates a new TendonActuatedPendulum instance with updated parameters
        while preserving the original instance (immutable update pattern for JAX compatibility).

        Args:
           tendon_params: Dictionary with tendon parameters with keys
                - "Ra": radii of the pulleys that route the active tendons (Na,N) (optional)
                - "Rp": radii of the pulleys that route the passive tendons (Np,N) (optional)
                - "Kp": stiffnesses of the springs attached to the passive tendons (Np,) (optional)
                - "Dp": damping coefficients of the dampers attached to the passive tendons (Np,) (optional)
                - "Lp0": initial length of the passive tendons (Np,) (optional)

        Returns:
            TendonActuatedPendulum: New instance with updated parameters.
        """
        updated = self
        if "Ra" in tendon_params:
            Ra = jnp.asarray(tendon_params["Ra"])
            updated = eqx.tree_at(lambda x: x.Ra, updated, Ra)
            updated = eqx.tree_at(lambda x: x.A, updated, Ra.T)
        if "Rp" in tendon_params:
            Rp = jnp.asarray(tendon_params["Rp"])
            updated = eqx.tree_at(lambda x: x.Rp, updated, Rp)
            updated = eqx.tree_at(lambda x: x.Ap, updated, Rp.T)
        if "Kp" in tendon_params:
            kp = jnp.asarray(tendon_params["Kp"])
            updated = eqx.tree_at(lambda x: x.Kp, updated, jnp.diag(kp))
        if "Dp" in tendon_params:
            dp = jnp.asarray(tendon_params["Dp"])
            updated = eqx.tree_at(lambda x: x.Dp, updated, jnp.diag(dp))
        if "Lp0" in tendon_params:
            updated = eqx.tree_at(lambda x: x.Lp0, updated, jnp.asarray(tendon_params["Lp0"]))
        return updated

    @eqx.filter_jit
    def passive_tendon_length(self, q: Array) -> Array:
        """
        Compute the length of the passive tendons.

        Args:
            q (Array): Joint angles, shape (N,) [rad]

        Returns:
            Lp (Array): length of the passive tendons, shape (Np,) [m]
        """
        Lp = self.Rp @ q + self.Lp0
        return Lp
    
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
        Lp = self.passive_tendon_length(q)
        tau_el_tot = self.K @ q + self.Ap @ self.Kp @ Lp
        return tau_el_tot

    @eqx.filter_jit
    def damping_matrix(self) -> Array:
        """
        Return the linear viscous damping matrix inlcuding the damping of the passive tendons.

        Returns:
            D_tot (Array): Total damping matrix, shape (N, N) [N⋅m⋅s/rad]
        """
        D_tot = self.D + self.Ap @ self.Dp @ self.Rp
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
        return self.A

    def actuation_force(self, q: Array, u: Array) -> Array:
        """
        Compute generalized actuation forces from control inputs.

        Args:
            q (Array): Joint angles, shape (N,) [rad]
            u (Array): Control torques, shape (N,) [N⋅m]

        Returns:
            tau_u (Array): Generalized actuation forces τ_u = A @ u, shape (N,) [N⋅m]
        """
        tau_u = self.actuation_matrix(q) @ u
        return tau_u
    
