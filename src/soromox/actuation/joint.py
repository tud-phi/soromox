"""Joint-space transmissions and articulated tendon mechanics."""

from __future__ import annotations

from typing import Any

import equinox as eqx
from jax import Array
from jax import numpy as jnp

from soromox.systems.params import BaseSystemParams

from .core import (
    Actuator,
    ActuatorMetadata,
    DirectEffort,
    PassiveElement,
    Transmission,
)


def _channel_array(value: Any, count: int, name: str) -> Array:
    """Broadcast a scalar or validate one value per actuator channel."""
    array = jnp.asarray(value)
    if array.ndim == 0:
        array = jnp.full((count,), array)
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},), got {array.shape}.")
    return array


def _validate_tendon_routing_structure(matrix: Array, num_dofs: int, name: str) -> None:
    """Validate articulated tendon shape against a robot."""
    matrix = jnp.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if matrix.shape[1] != num_dofs:
        raise ValueError(
            f"{name} must have one column per degree of freedom; expected "
            f"{num_dofs}, got {matrix.shape[1]}."
        )


def _validate_tendon_routing_values(matrix: Array, num_dofs: int, name: str) -> None:
    """Validate concrete articulated tendon routing values."""
    matrix = jnp.asarray(matrix)
    count = matrix.shape[0]
    if count > 0 and int(jnp.linalg.matrix_rank(matrix)) != count:
        raise ValueError(f"{name} must be full row rank.")
    zero_mask = matrix == 0
    column_indices = jnp.arange(num_dofs)[None, :]
    first_zero = jnp.where(
        jnp.any(zero_mask, axis=1), jnp.argmax(zero_mask, axis=1), num_dofs
    )
    after_first_zero = column_indices >= first_zero[:, None]
    if not bool(jnp.all(jnp.where(after_first_zero, matrix == 0, True))):
        raise ValueError(
            f"Tendons cannot skip joints in {name}; after the first zero in each "
            "row, all remaining entries must be zero."
        )


def _validate_articulated_tendon_host(robot) -> None:
    """Require an explicit serial articulated routing capability."""
    if not robot.supports_articulated_tendon_routing:
        raise TypeError(
            "Articulated tendon components require a serial articulated host "
            "such as Pendulum or ArticulatedSoftRobot. Use a continuum "
            "transmission, such as ThreadlikeActuator, for PCS or GVS hosts."
        )


class AffineJointTransmissionParams(BaseSystemParams):
    """Parameters for an affine mapping to actuator coordinates.

    Attributes:
        routing_matrix: Matrix ``R`` with shape
            ``(num_channels, num_dofs)``. Each row is the derivative of one
            actuator coordinate with respect to the generalized coordinates.
        reference_configuration: Generalized configuration ``q_ref`` at which
            the affine displacement term is zero.
        coordinate_offset: Per-channel actuator coordinate ``y_a0`` at
            ``q_ref``, in the same units as the actuator coordinate. Thus
            ``y_a(q_ref) = y_a0`` and the differential map is ``dy_a/dq = R``.
            For passive tendon impedance, ``y_a0`` specifies the deformation
            and corresponding preload at ``q_ref``.
    """

    routing_matrix: Array
    reference_configuration: Array
    coordinate_offset: Array

    def __check_init__(self) -> None:
        self.validate_for_update()

    @property
    def num_channels(self) -> int:
        return int(jnp.asarray(self.routing_matrix).shape[0])

    @property
    def num_dofs(self) -> int:
        return int(jnp.asarray(self.routing_matrix).shape[1])

    def validate_structure(self) -> None:
        matrix = jnp.asarray(self.routing_matrix)
        if matrix.ndim != 2:
            raise ValueError("routing_matrix must be two-dimensional.")
        if jnp.asarray(self.reference_configuration).shape != (matrix.shape[1],):
            raise ValueError(
                f"reference_configuration must have shape ({matrix.shape[1]},)."
            )
        if jnp.asarray(self.coordinate_offset).shape != (matrix.shape[0],):
            raise ValueError(f"coordinate_offset must have shape ({matrix.shape[0]},).")

    def assert_same_topology(self, other: AffineJointTransmissionParams) -> None:
        if not isinstance(other, AffineJointTransmissionParams):
            raise ValueError("Changing transmission type requires reconstruction.")
        if other.routing_matrix.shape != self.routing_matrix.shape:
            raise ValueError(
                "Changing the routing matrix shape requires reconstruction."
            )


class AffineJointTransmission(Transmission):
    """Affine actuator coordinates ``R @ (q - q_ref) + y_a0``."""

    params: AffineJointTransmissionParams

    def __init__(self, params: AffineJointTransmissionParams) -> None:
        if not isinstance(params, AffineJointTransmissionParams):
            raise TypeError("params must be AffineJointTransmissionParams.")
        params.validate_for_update()
        self.params = params

    @property
    def num_channels(self) -> int:
        return self.params.num_channels

    def coordinates(self, robot, q: Array) -> Array:
        del robot
        return (
            self.params.routing_matrix @ (q - self.params.reference_configuration)
            + self.params.coordinate_offset
        )

    def moment_matrix(self, robot, q: Array) -> Array:
        del robot, q
        return self.params.routing_matrix.T

    def with_params(
        self, params: AffineJointTransmissionParams
    ) -> AffineJointTransmission:
        self.params.assert_same_topology(params)
        params.validate_for_update()
        return AffineJointTransmission(params)

    def update_params(self, **updates: Any) -> AffineJointTransmission:
        return self.with_params(self.params.replace(**updates))


class ArticulatedTendonActuatorParams(BaseSystemParams):
    """Physical parameters for direct-effort articulated tendons."""

    transmission: AffineJointTransmissionParams
    lower_bounds: Array
    upper_bounds: Array

    def __check_init__(self) -> None:
        self.validate_for_update()

    def validate_structure(self) -> None:
        self.transmission.validate_structure()
        count = self.transmission.num_channels
        for name in ("lower_bounds", "upper_bounds"):
            value = jnp.asarray(getattr(self, name))
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape ({count},).")

    def validate_values(self) -> None:
        self.transmission.validate_values()


class ArticulatedTendonActuator(Actuator):
    """Positive-tension actuation through fixed signed joint routing."""

    _params: ArticulatedTendonActuatorParams
    _transmission: AffineJointTransmission
    _effort_model: DirectEffort
    _metadata: ActuatorMetadata
    name: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        params: ArticulatedTendonActuatorParams,
        labels: tuple[str, ...] | None = None,
        name: str = "articulated_tendons",
    ) -> None:
        if not isinstance(params, ArticulatedTendonActuatorParams):
            raise TypeError("params must be ArticulatedTendonActuatorParams.")
        params.validate_for_update()
        count = params.transmission.num_channels
        if labels is None:
            labels = tuple(f"tendon_tension_{index}" for index in range(count))
        if len(labels) != count:
            raise ValueError("labels must contain one entry per tendon.")
        self._params = params
        self._transmission = AffineJointTransmission(params.transmission)
        self._effort_model = DirectEffort()
        self._metadata = ActuatorMetadata(
            labels=labels,
            units="N",
            kind="tendon",
            lower_bounds=params.lower_bounds,
            upper_bounds=params.upper_bounds,
        )
        self.name = name

    @classmethod
    def from_routing(
        cls,
        routing_matrix: Array,
        *,
        reference_configuration: Array | None = None,
        coordinate_offset: Array | float = 0.0,
        lower_bounds: Array | float = 0.0,
        upper_bounds: Array | float = jnp.inf,
        labels: tuple[str, ...] | None = None,
        name: str = "articulated_tendons",
    ) -> ArticulatedTendonActuator:
        """Construct direct-tension actuation from signed joint routing.

        The resulting actuator coordinate is
        ``routing_matrix @ (q - reference_configuration) + coordinate_offset``.
        Each routing row is the signed contraction Jacobian of one tendon.

        Args:
            routing_matrix: Signed tendon routing with one row per tendon and
                one column per generalized coordinate.
            reference_configuration: Configuration ``q_ref`` used as the
                origin of the routing displacement. Defaults to zero.
            coordinate_offset: Actuator coordinate ``y_a0`` at ``q_ref``, as a
                scalar shared by all tendons or one value per tendon. Under
                ``DirectEffort``, the generalized force is ``R.T @ control``.
            lower_bounds: Lower tension bounds, shared or per tendon.
            upper_bounds: Upper tension bounds, shared or per tendon.
            labels: Optional input label for each tendon.
            name: Component name used for identification and rendering.

        Returns:
            An articulated tendon actuator with direct tension effort.
        """
        matrix = jnp.asarray(routing_matrix)
        if matrix.ndim != 2:
            raise ValueError("routing_matrix must be two-dimensional.")
        count, num_dofs = matrix.shape
        if reference_configuration is None:
            reference_configuration = jnp.zeros((num_dofs,), dtype=matrix.dtype)
        transmission = AffineJointTransmissionParams(
            routing_matrix=matrix,
            reference_configuration=jnp.asarray(reference_configuration),
            coordinate_offset=_channel_array(
                coordinate_offset, count, "coordinate_offset"
            ),
        )
        return cls(
            params=ArticulatedTendonActuatorParams(
                transmission=transmission,
                lower_bounds=_channel_array(lower_bounds, count, "lower_bounds"),
                upper_bounds=_channel_array(upper_bounds, count, "upper_bounds"),
            ),
            labels=labels,
            name=name,
        )

    @property
    def params(self) -> ArticulatedTendonActuatorParams:
        return self._params

    @property
    def transmission(self) -> AffineJointTransmission:
        return self._transmission

    @property
    def effort_model(self) -> DirectEffort:
        return self._effort_model

    @property
    def metadata(self) -> ActuatorMetadata:
        return self._metadata

    def validate_structure_for_robot(self, robot) -> None:
        _validate_articulated_tendon_host(robot)
        _validate_tendon_routing_structure(
            self.params.transmission.routing_matrix,
            robot.num_internal_dofs,
            "routing_matrix",
        )

    def validate_values_for_robot(self, robot) -> None:
        _validate_tendon_routing_values(
            self.params.transmission.routing_matrix,
            robot.num_internal_dofs,
            "routing_matrix",
        )

    def _robot_value_validation_tree(self):
        return self.params.transmission.routing_matrix

    def with_params(self, params: BaseSystemParams) -> ArticulatedTendonActuator:
        if not isinstance(params, ArticulatedTendonActuatorParams):
            raise TypeError("params must be ArticulatedTendonActuatorParams.")
        self.params.transmission.assert_same_topology(params.transmission)
        params.validate_for_update()
        return ArticulatedTendonActuator(
            params=params,
            labels=self.metadata.labels,
            name=self.name,
        )


class ArticulatedTendonImpedanceParams(BaseSystemParams):
    """Spring-damper parameters for articulated tendon coordinates."""

    transmission: AffineJointTransmissionParams
    stiffness: Array
    damping: Array

    def __check_init__(self) -> None:
        self.validate_for_update()

    def validate_structure(self) -> None:
        self.transmission.validate_structure()
        count = self.transmission.num_channels
        for name in ("stiffness", "damping"):
            if jnp.asarray(getattr(self, name)).shape != (count,):
                raise ValueError(f"{name} must have shape ({count},).")

    def validate_values(self) -> None:
        self.transmission.validate_values()


class ArticulatedTendonImpedance(PassiveElement):
    """Passive spring-damper mechanics in affine tendon coordinates."""

    _params: ArticulatedTendonImpedanceParams
    transmission: AffineJointTransmission
    name: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        params: ArticulatedTendonImpedanceParams,
        name: str = "passive_articulated_tendons",
    ) -> None:
        if not isinstance(params, ArticulatedTendonImpedanceParams):
            raise TypeError("params must be ArticulatedTendonImpedanceParams.")
        params.validate_for_update()
        self._params = params
        self.transmission = AffineJointTransmission(params.transmission)
        self.name = name

    @classmethod
    def from_routing(
        cls,
        routing_matrix: Array,
        *,
        stiffness: Array,
        damping: Array,
        reference_configuration: Array | None = None,
        coordinate_offset: Array | float = 0.0,
        name: str = "passive_articulated_tendons",
    ) -> ArticulatedTendonImpedance:
        """Construct passive spring-damper mechanics in tendon coordinates.

        Args:
            routing_matrix: Signed tendon routing with one row per passive
                tendon and one column per generalized coordinate.
            stiffness: Actuator-space stiffness, one value per tendon.
            damping: Actuator-space damping, one value per tendon.
            reference_configuration: Configuration ``q_ref`` used as the
                origin of the routing displacement. Defaults to zero.
            coordinate_offset: Tendon deformation ``y_a0`` at ``q_ref``, as a
                scalar shared by all tendons or one value per tendon. Zero
                makes ``q_ref`` unstrained; a nonzero value creates elastic
                preload ``R.T @ (stiffness * y_a0)``.
            name: Component name used for identification and rendering.

        Returns:
            A passive articulated tendon impedance component.
        """
        matrix = jnp.asarray(routing_matrix)
        if matrix.ndim != 2:
            raise ValueError("routing_matrix must be two-dimensional.")
        count, num_dofs = matrix.shape
        if reference_configuration is None:
            reference_configuration = jnp.zeros((num_dofs,), dtype=matrix.dtype)
        return cls(
            params=ArticulatedTendonImpedanceParams(
                transmission=AffineJointTransmissionParams(
                    routing_matrix=matrix,
                    reference_configuration=jnp.asarray(reference_configuration),
                    coordinate_offset=_channel_array(
                        coordinate_offset, count, "coordinate_offset"
                    ),
                ),
                stiffness=_channel_array(stiffness, count, "stiffness"),
                damping=_channel_array(damping, count, "damping"),
            ),
            name=name,
        )

    @property
    def params(self) -> ArticulatedTendonImpedanceParams:
        return self._params

    def validate_structure_for_robot(self, robot) -> None:
        _validate_articulated_tendon_host(robot)
        _validate_tendon_routing_structure(
            self.params.transmission.routing_matrix,
            robot.num_dofs,
            "routing_matrix",
        )

    def validate_values_for_robot(self, robot) -> None:
        _validate_tendon_routing_values(
            self.params.transmission.routing_matrix,
            robot.num_dofs,
            "routing_matrix",
        )

    def _robot_value_validation_tree(self):
        return self.params.transmission.routing_matrix

    def with_params(self, params: BaseSystemParams) -> ArticulatedTendonImpedance:
        if not isinstance(params, ArticulatedTendonImpedanceParams):
            raise TypeError("params must be ArticulatedTendonImpedanceParams.")
        self.params.transmission.assert_same_topology(params.transmission)
        params.validate_for_update()
        return ArticulatedTendonImpedance(params=params, name=self.name)

    def coordinates(self, robot, q: Array) -> Array:
        return self.transmission.coordinates(robot, q)

    def velocities(self, robot, q: Array, qd: Array) -> Array:
        return self.transmission.velocities(robot, q, qd)

    def elastic_force(self, robot, q: Array) -> Array:
        coordinate = self.coordinates(robot, q)
        return self.transmission.moment_matrix(robot, q) @ (
            self.params.stiffness * coordinate
        )

    def damping_matrix(self, robot, q: Array) -> Array:
        matrix = self.transmission.moment_matrix(robot, q)
        return matrix @ (self.params.damping[:, None] * matrix.T)

    def elastic_energy(self, robot, q: Array) -> Array:
        coordinate = self.coordinates(robot, q)
        return 0.5 * jnp.sum(self.params.stiffness * coordinate**2)


__all__ = [
    "AffineJointTransmission",
    "AffineJointTransmissionParams",
    "ArticulatedTendonActuator",
    "ArticulatedTendonActuatorParams",
    "ArticulatedTendonImpedance",
    "ArticulatedTendonImpedanceParams",
]
