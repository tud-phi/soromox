from jax import Array
from typing import Callable, Dict

from soromox.systems import euler_lagrangian


def ode_factory(
    dynamical_matrices_fn: Callable, params: Dict[str, Array], tau: Array
) -> Callable[[float, Array], Array]:
    """
    Make an ODE function of the form ode_fn(t, x) -> xd.
    This function assumes a constant torque input (i.e. zero-order hold).
    Args:
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and A matrices. Needs to conform to the signature:
            dynamical_matrices_fn(params, q, qd) -> Tuple[B, C, G, K, D, A]
            where q and qd are the configuration and velocity vectors, respectively,
            B is the inertia matrix of shape (n_q, n_q),
            C is the Coriolis matrix of shape (n_q, n_q),
            G is the gravity vector of shape (n_q, ),
            K is the stiffness vector of shape (n_q, ),
            D is the damping matrix of shape (n_q, n_q),
            A is the actuation matrix of shape (n_q, n_tau).
        params: Dictionary with robot parameters
        tau: torque vector of shape (n_tau, )
    Returns:
        ode_fn: ODE function of the form ode_fn(t, x) -> xd
    """

    def ode_fn(t: float, x: Array, *args) -> Array:
        """
        ODE of the dynamical Lagrangian system.
        Args:
            t: time
            x: state vector of shape (2 * n_q, )
            args: additional arguments
        Returns:
            xd: time-derivative of the state vector of shape (2 * n_q, )
        """
        xd = euler_lagrangian.nonlinear_state_space(
            dynamical_matrices_fn,
            params,
            x,
            tau,
        )
        return xd

    return ode_fn


def ode_with_forcing_factory(
    dynamical_matrices_fn: Callable, params: Dict[str, Array]
) -> Callable[[float, Array], Array]:
    """
    Make an ODE function of the form ode_fn(t, x) -> xd.
    This function assumes a constant torque input (i.e. zero-order hold).
    Args:
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and A matrices. Needs to conform to the signature:
            dynamical_matrices_fn(params, q, qd) -> Tuple[B, C, G, K, D, A]
            where q and qd are the configuration and velocity vectors, respectively,
            B is the inertia matrix of shape (n_q, n_q),
            C is the Coriolis matrix of shape (n_q, n_q),
            G is the gravity vector of shape (n_q, ),
            K is the stiffness vector of shape (n_q, ),
            D is the damping matrix of shape (n_q, n_q),
            A is the actuation matrix of shape (n_q, n_tau).
        params: Dictionary with robot parameters
    Returns:
        ode_fn: ODE function of the form ode_fn(t, x, tau) -> xd
    """

    def ode_fn(
        t: float,
        x: Array,
        tau: Array,
    ) -> Array:
        """
        ODE of the dynamical Lagrangian system.
        Args:
            t: time
            x: state vector of shape (2 * n_q, )
            tau: external torque vector of shape (n_tau, )
        Returns:
            xd: time-derivative of the state vector of shape (2 * n_q, )
        """
        xd = euler_lagrangian.nonlinear_state_space(
            dynamical_matrices_fn,
            params,
            x,
            tau,
        )
        return xd

    return ode_fn
