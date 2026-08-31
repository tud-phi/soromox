"""Public dimension-selecting launch helpers for PCS kinematics."""

from __future__ import annotations

from typing import Any

from soromox.execution.warp.pcs.planar_directional_kinematics import (
    launch_planar_forward_kinematics_and_jvp,
    launch_planar_inertial_jacobian_vjp,
)
from soromox.execution.warp.pcs.planar_kinematics import (
    launch_planar_forward_kinematics,
    launch_planar_inertial_jacobians,
    launch_planar_kinematics,
)
from soromox.execution.warp.pcs.spatial_directional_kinematics import (
    launch_spatial_forward_kinematics_and_jvp,
    launch_spatial_inertial_jacobian_vjp,
)
from soromox.execution.warp.pcs.spatial_kinematics import (
    launch_spatial_forward_kinematics,
    launch_spatial_inertial_jacobians,
    launch_spatial_kinematics,
)


def _selected_launcher(is_planar: bool, operation: str):
    """Return the public dimension-specialized PCS launch function.

    Args:
        is_planar: Select PlanarPCS when true and spatial PCS otherwise.
        operation: One of ``"pose"``, ``"jacobian"``, ``"both"``, ``"jvp"``,
            or ``"vjp"``.

    Returns:
        The matching public dimension- and operation-specialized launcher.
    """

    if operation == "pose":
        return (
            launch_planar_forward_kinematics
            if is_planar
            else launch_spatial_forward_kinematics
        )
    if operation == "jacobian":
        return (
            launch_planar_inertial_jacobians
            if is_planar
            else launch_spatial_inertial_jacobians
        )
    if operation == "jvp":
        return (
            launch_planar_forward_kinematics_and_jvp
            if is_planar
            else launch_spatial_forward_kinematics_and_jvp
        )
    if operation == "vjp":
        return (
            launch_planar_inertial_jacobian_vjp
            if is_planar
            else launch_spatial_inertial_jacobian_vjp
        )
    return launch_planar_kinematics if is_planar else launch_spatial_kinematics


def launch_forward_kinematics(
    q: Any,
    s: Any,
    *args: Any,
    is_planar: bool,
    **kwargs: Any,
) -> None:
    """Launch PCS forward kinematics with caller-owned Warp arrays.

    Args:
        q: Batched active configurations with shape ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        *args: Remaining dimension-specific positional arguments documented by
            the specialized forward-kinematics launchers.
        is_planar: Select the PlanarPCS or spatial PCS kernel family.
        **kwargs: Dimension-specific keyword arguments forwarded unchanged.

    Returns:
        None. Outputs are written in place.
    """

    _selected_launcher(is_planar, "pose")(q, s, *args, **kwargs)


def launch_inertial_jacobians(
    q: Any,
    s: Any,
    *args: Any,
    is_planar: bool,
    **kwargs: Any,
) -> None:
    """Launch PCS inertial Jacobians through the shared fused recurrence.

    Args:
        q: Batched active configurations with shape ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        *args: Remaining dimension-specific positional arguments documented by
            the specialized inertial-Jacobian launchers.
        is_planar: Select the PlanarPCS or spatial PCS kernel family.
        **kwargs: Dimension-specific keyword arguments forwarded unchanged.

    Returns:
        None. Outputs are written in place.
    """

    _selected_launcher(is_planar, "jacobian")(q, s, *args, **kwargs)


def launch_forward_kinematics_and_jacobians(
    q: Any,
    s: Any,
    *args: Any,
    is_planar: bool,
    **kwargs: Any,
) -> None:
    """Launch fused PCS poses and inertial Jacobians.

    Args:
        q: Batched active configurations with shape ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        *args: Remaining dimension-specific positional arguments documented by
            the specialized fused launchers.
        is_planar: Select the PlanarPCS or spatial PCS kernel family.
        **kwargs: Dimension-specific keyword arguments forwarded unchanged.

    Returns:
        None. Outputs are written in place.
    """

    _selected_launcher(is_planar, "both")(q, s, *args, **kwargs)


def launch_forward_kinematics_and_jvp(
    q: Any,
    qd: Any,
    s: Any,
    *args: Any,
    is_planar: bool,
    **kwargs: Any,
) -> None:
    """Launch PCS poses and one inertial-Jacobian forward product.

    Args:
        q: Batched active configurations with shape ``(E, D)``.
        qd: Batched generalized-velocity directions with shape ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        *args: Remaining dimension-specific positional arguments documented by
            the specialized JVP launchers.
        is_planar: Select the PlanarPCS or spatial PCS kernel family.
        **kwargs: Dimension-specific keyword arguments forwarded unchanged.

    Returns:
        None. Outputs are written in place.
    """

    _selected_launcher(is_planar, "jvp")(q, qd, s, *args, **kwargs)


def launch_inertial_jacobian_vjp(
    q: Any,
    s: Any,
    inertial_wrenches: Any,
    *args: Any,
    is_planar: bool,
    **kwargs: Any,
) -> None:
    """Launch a PCS inertial-Jacobian transpose product.

    Args:
        q: Batched active configurations with shape ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        inertial_wrenches: Inertial sample cotangents ``(E, N, spatial_dim)``.
        *args: Remaining dimension-specific positional arguments documented by
            the specialized VJP launchers.
        is_planar: Select the PlanarPCS or spatial PCS kernel family.
        **kwargs: Dimension-specific keyword arguments forwarded unchanged.

    Returns:
        None. Outputs are written in place.
    """

    _selected_launcher(is_planar, "vjp")(
        q,
        s,
        inertial_wrenches,
        *args,
        **kwargs,
    )


__all__ = [
    "launch_forward_kinematics",
    "launch_forward_kinematics_and_jacobians",
    "launch_forward_kinematics_and_jvp",
    "launch_inertial_jacobian_vjp",
    "launch_inertial_jacobians",
]
