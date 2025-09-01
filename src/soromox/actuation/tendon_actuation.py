__all__ = ["linear_routing", "linear_routing_derivative"]
from jax import numpy as jnp


def linear_routing(tendon_routing_params, s):
    """
    Routing of linear tendons as a function of ``s``.

    Specifically, ``d_s`` is the vector between the centerline of the robot and the
    tendon position in the cross-sectional plane at abscissa ``s``.

    Args:
        tendon_routing_params (Dict[str, Array]): Parameters of the linear tendons.
                ry: y-coordinate of the pulling point of the tendons [m]
                my: slope coefficient in the x-y plane of the tendons [-]
                rz: z-coordinate of the pulling point of the tendons [m]
                mz: slope coefficient in the x-z plane of the tendons [-]
        s (Array): Abscissa points ``(num_gauss_points,)``.

    Returns:
        d_s (Array): Position of the tendon at ``s`` w.r.t. the centerline ``(3,)``.
    """

    ry = tendon_routing_params["ry"]
    my = tendon_routing_params["my"]
    rz = tendon_routing_params["rz"]
    mz = tendon_routing_params["mz"]
    d_s = jnp.array([0.0, ry + my * s, rz + mz * s])
    return d_s


def linear_routing_derivative(tendon_routing_params, s):
    """
    Spatial derivative of the linear tendon routing as a function of ``s``.

    Specifically, ``dd_s_ds`` is the derivative of ``d_s`` with respect to ``s``.

    Args:
        tendon_routing_params (Dict[str, Array]): Parameters of the linear tendons.
                my: slope coefficient in the x-y plane of the tendons [-]
                mz: slope coefficient in the x-z plane of the tendons [-]
        s (Array): Abscissa points ``(num_gauss_points,)``.

    Returns:
        dd_s_ds (Array): Path of the tendon at ``s`` w.r.t. the centerline ``(3,)``.
    """

    my = tendon_routing_params["my"]
    mz = tendon_routing_params["mz"]
    dd_s_ds = jnp.array([0.0, my, mz])
    return dd_s_ds
