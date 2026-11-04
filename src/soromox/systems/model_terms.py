__all__ = ["ModelTerms"]

import equinox as eqx
from jax import Array, lax
from jax import numpy as jnp
from jax.scipy.linalg import cho_solve


def _factor_inertia(inertia_matrix: Array) -> Array:
    """
    Return the operand of :func:`_solve_with_inertia_factor` for an inertia matrix.

    The two functions hold the platform policy of every soft-robot inertia solve;
    ``SoftRobot._solve_inertia`` composes them. The CPU uses the lower Cholesky
    factor of the symmetric part, while accelerator platforms keep the matrix for
    the generic dense solve. Both branches return an array with the shape of
    ``inertia_matrix``.

    Args:
        inertia_matrix: Symmetric positive-definite inertia matrix with shape
            ``(num_velocities, num_velocities)``.

    Returns:
        Factor with the shape of ``inertia_matrix``.
    """

    def generic(matrix: Array) -> Array:
        return matrix

    def cholesky(matrix: Array) -> Array:
        return jnp.linalg.cholesky(0.5 * (matrix + matrix.T))

    return lax.platform_dependent(
        inertia_matrix,
        cpu=cholesky,
        cuda=generic,
        rocm=generic,
        default=generic,
    )


def _solve_with_inertia_factor(factor: Array, rhs: Array) -> Array:
    """
    Solve an inertia system with the operand returned by :func:`_factor_inertia`.

    Args:
        factor: Factor returned by :func:`_factor_inertia`.
        rhs: Right-hand side with shape ``(num_velocities,)``, or a matrix of
            right-hand sides with shape ``(num_velocities, k)``.

    Returns:
        Solution with the shape of ``rhs``.
    """

    def generic(matrix: Array, vector: Array) -> Array:
        return jnp.linalg.solve(matrix, vector)

    def cholesky(matrix: Array, vector: Array) -> Array:
        return cho_solve((matrix, True), vector)

    return lax.platform_dependent(
        factor,
        rhs,
        cpu=cholesky,
        cuda=generic,
        rocm=generic,
        default=generic,
    )


class ModelTerms(eqx.Module):
    """
    Model quantities of a soft robot at one state.

    A system computes these quantities once per state with
    :meth:`soromox.systems.SoftRobot.model_terms`. A model-based controller reads
    them from ``SystemState.model_terms``. The terms describe the state at which
    they were computed and must only be used at that state.

    The constructor computes ``inertia_factor`` from ``inertia_matrix``, so
    changed terms must be constructed again: ``dataclasses.replace`` raises,
    and ``eqx.tree_at`` on ``inertia_matrix`` keeps the old factor. The factor
    depends on the platform, so the terms must be used on the platform that
    computed them.

    Attributes:
        inertia_matrix: Generalized inertia matrix ``B(q)`` with shape
            ``(num_velocities, num_velocities)``.
        inertia_factor: Factor of ``B(q)`` used by :meth:`solve_inertia`: the
            lower Cholesky factor of its symmetric part on CPU, and ``B(q)``
            itself on accelerator platforms.
        coriolis_force: Convective force ``C(q, qd) @ qd`` with shape
            ``(num_velocities,)``.
        gravitational_force: Generalized gravitational force ``G(q)`` with shape
            ``(num_velocities,)``.
        elastic_force: Generalized elastic force, including installed passive
            elements, with shape ``(num_velocities,)``.
        damping_matrix: Generalized damping matrix ``D(q)`` with shape
            ``(num_velocities, num_velocities)``.
        actuation_matrix: Actuation matrix ``A(q)``, the map from actuator
            efforts to generalized forces, with shape
            ``(num_velocities, num_actuators)``. The terms hold no
            actuator-coordinate Jacobian ``J_a(q)``: ``A(q)`` equals its
            transpose only for lossless transmissions.
        node_jacobians: Body-frame Jacobians at the nodes of the system's
            dynamics pass, from which its kinematics methods evaluate the
            Jacobian at any backbone coordinate, or ``None`` when the system
            does not provide them.
        node_poses: Poses at the same nodes in the representation of the
            system's forward kinematics, or ``None`` when the system does not
            provide them.
    """

    inertia_matrix: Array
    inertia_factor: Array
    coriolis_force: Array
    gravitational_force: Array
    elastic_force: Array
    damping_matrix: Array
    actuation_matrix: Array
    node_jacobians: Array | None
    node_poses: Array | None

    def __init__(
        self,
        inertia_matrix: Array,
        coriolis_force: Array,
        gravitational_force: Array,
        elastic_force: Array,
        damping_matrix: Array,
        actuation_matrix: Array,
        node_jacobians: Array | None = None,
        node_poses: Array | None = None,
    ) -> None:
        """
        Collect the model quantities and factor the inertia matrix once.

        Args:
            inertia_matrix: Generalized inertia matrix ``B(q)``.
            coriolis_force: Convective force ``C(q, qd) @ qd``.
            gravitational_force: Generalized gravitational force ``G(q)``.
            elastic_force: Generalized elastic force.
            damping_matrix: Generalized damping matrix ``D(q)``.
            actuation_matrix: Actuation matrix ``A(q)``.
            node_jacobians: Optional body-frame Jacobians at the nodes of the
                system's dynamics pass.
            node_poses: Optional poses at the same nodes.
        """
        self.inertia_matrix = inertia_matrix
        self.inertia_factor = _factor_inertia(inertia_matrix)
        self.coriolis_force = coriolis_force
        self.gravitational_force = gravitational_force
        self.elastic_force = elastic_force
        self.damping_matrix = damping_matrix
        self.actuation_matrix = actuation_matrix
        self.node_jacobians = node_jacobians
        self.node_poses = node_poses

    def solve_inertia(self, rhs: Array) -> Array:
        """
        Return ``B(q)^-1 @ rhs`` with the stored factor.

        Args:
            rhs: Right-hand side with shape ``(num_velocities,)``, or a matrix
                of right-hand sides with shape ``(num_velocities, k)``.

        Returns:
            Solution with the shape of ``rhs``.
        """
        return _solve_with_inertia_factor(self.inertia_factor, rhs)
