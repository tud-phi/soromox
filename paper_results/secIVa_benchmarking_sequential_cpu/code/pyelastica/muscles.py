from typing import Union, Iterable, Dict

from collections import defaultdict
import numpy as np
from numba import njit

import elastica
from elastica._linalg import _batch_cross
from elastica._calculus import quadrature_kernel
from elastica.external_forces import inplace_addition
from _rod_tool import average2D, difference2D, sigma_to_shear

from actuation import (
    _force_induced_couple,
    _internal_to_external_load,
    ContinuousActuation,
    ApplyActuations,
)


class MuscleInfo:
    # TODO: Maybe try to implement this class as @dataclass
    """MuscleInfo.
    Data class to store muscle's meta data
    """

    def __init__(self, type_name: str="muscle", index: int=0, **kwargs) -> None:
        """Muscle information constructor

        Parameters
        ----------
        type_name : str
            Name of the muscle type
        index : int
            Muscle index
        """
        super().__init__(**kwargs)
        self.type_name = type_name
        self.index = index

    def __str__(self) -> str:
        return f"{self.index}_{self.type_name}"


class Muscle(MuscleInfo, ContinuousActuation):
    """Muscle base class"""

    def __init__(
        self,
        ratio_muscle_position: np.ndarray,
        rest_muscle_area: np.ndarray,
        index: int = 0,
        **kwargs,
    ) -> None:
        """__init__.

        Parameters
        ----------
        ratio_muscle_position : np.ndarray
            shape: (3, n_element)
        rest_muscle_area : np.ndarray
            shape: (n_element)
        index : int
        """
        super().__init__(
            n_elements=rest_muscle_area.shape[0],
            index=index,
            **kwargs,
        )

        self.s = np.linspace(0, 1, self.n_elements + 1)
        self.muscle_normalized_length = np.zeros(self.n_elements)
        self.muscle_rest_length = np.ones(self.n_elements)
        self.muscle_length = np.zeros(self.n_elements)
        self.muscle_tangent = np.zeros((3, self.n_elements))
        self.muscle_strain = np.zeros((3, self.n_elements))
        self.muscle_position = np.zeros((3, self.n_elements))
        self.ratio_muscle_position = ratio_muscle_position.copy()
        self.rest_muscle_area = rest_muscle_area.copy()
        self.muscle_area = self.rest_muscle_area.copy()

    def __call__(self, system: elastica.rod.RodBase) -> None:
        """__call__.

        Parameters
        ----------
        system : elastica.rod.RodBase
        """

        self.calculate_muscle_position(
            self.muscle_position, system.radius, self.ratio_muscle_position
        )
        self.calculate_muscle_strain(
            self.muscle_strain,
            self.muscle_position,
            system.sigma,
            system.kappa,
            system.rest_voronoi_lengths,
            system.voronoi_dilatation,
        )
        self.calculate_muscle_tangent(self.muscle_tangent, self.muscle_strain)

    def set_current_length_as_rest_length(self, system: elastica.rod.RodBase) -> None:
        """set_current_length_as_rest_length.

        Parameters
        ----------
        system : elastica.rod.RodBase
        """
        self.__call__(system)
        self.calculate_muscle_length(self.muscle_length, self.muscle_strain)
        self.muscle_rest_length[:] = self.muscle_length

    @staticmethod
    @njit(cache=True)
    def calculate_muscle_position(muscle_position, radius, ratio_muscle_position):
        muscle_position[:, :] = radius * ratio_muscle_position

    @staticmethod
    @njit(cache=True)
    def calculate_muscle_strain(
        muscle_strain,
        off_center_displacement,
        sigma,
        kappa,
        rest_voronoi_lengths,
        voronoi_dilatation,
    ):
        shear = sigma_to_shear(sigma)
        muscle_position_derivative = difference2D(off_center_displacement) / (
            rest_voronoi_lengths * voronoi_dilatation
        )
        muscle_strain[:, :] = shear + quadrature_kernel(
            _batch_cross(kappa, average2D(off_center_displacement))
            + muscle_position_derivative
        )

    @staticmethod
    @njit(cache=True)
    def calculate_muscle_tangent(muscle_tangent, muscle_strain):
        blocksize = muscle_strain.shape[1]
        for i in range(blocksize):
            muscle_tangent[:, i] = muscle_strain[:, i] / np.sqrt(
                muscle_strain[0, i] ** 2
                + muscle_strain[1, i] ** 2
                + muscle_strain[2, i] ** 2
            )

    @staticmethod
    @njit(cache=True)
    def calculate_muscle_length(muscle_length, muscle_strain):
        blocksize = muscle_length.shape[0]
        for i in range(blocksize):
            muscle_length[i] = np.sqrt(
                muscle_strain[0, i] ** 2
                + muscle_strain[1, i] ** 2
                + muscle_strain[2, i] ** 2
            )

    @staticmethod
    @njit(cache=True)
    def calculate_muscle_normalized_length(
        muscle_normalized_length, muscle_length, muscle_rest_length
    ):
        muscle_normalized_length[:] = muscle_length / muscle_rest_length


class MuscleForce(Muscle):
    """MuscleForce"""

    def __init__(
        self,
        ratio_muscle_position: np.ndarray,
        rest_muscle_area: np.ndarray,
        max_muscle_stress: Union[float, np.ndarray],
        **kwargs,
    ):
        """
        Muscle force class implementation

        Parameters
        ----------
        ratio_muscle_position : np.ndarray
            shape: (3, n_element)
        rest_muscle_area : np.ndarray
            shape: (n_element)
        max_muscle_stress : Union[float, np.array]
            shape: (n_element)
        """
        super().__init__(
            ratio_muscle_position=ratio_muscle_position,
            rest_muscle_area=rest_muscle_area,
            muscle_type="muscle_force",
            **kwargs,
        )
        self.activation = np.zeros(self.n_elements)
        self.s_activation = (self.s[:-1] + self.s[1:]) / 2
        if isinstance(max_muscle_stress, float):
            self.max_muscle_stress = max_muscle_stress
        elif isinstance(max_muscle_stress, np.ndarray):
            self.max_muscle_stress = max_muscle_stress.copy()
        else:
            raise TypeError(
                f"{max_muscle_stress=} must be either float or np.ndarray. "
            )
        self.muscle_force = np.zeros(self.n_elements)
        self.s_force = 0.5 * (self.s[:-1] + self.s[1:])
        self.force_length_weight = kwargs.get("force_length_weight", np.ones_like)

    def __call__(self, system: elastica.rod.RodBase):
        """__call__.

        Parameters
        ----------
        system : elastica.rod.RodBase
        """
        super().__call__(system)
        self.calculate_muscle_length(self.muscle_length, self.muscle_strain)
        self.calculate_muscle_normalized_length(
            self.muscle_normalized_length, self.muscle_length, self.muscle_rest_length
        )
        # self.calculate_muscle_force(
        #     self.muscle_force,
        #     self.get_activation(),
        #     self.max_muscle_stress,
        #     self.force_length_weight(self.muscle_normalized_length),
        #     self.muscle_area,
        # )
        self.muscle_force[:] = self.get_activation()
        self.calculate_force_and_couple(
            self.muscle_force,
            self.muscle_tangent,
            self.muscle_position,
            self.internal_force,
            self.internal_couple,
            self.external_force,
            self.external_couple,
            system.director_collection,
            system.kappa,
            system.tangents,
            system.rest_lengths,
            system.rest_voronoi_lengths,
            system.dilatation,
            system.voronoi_dilatation,
        )

    # @staticmethod
    # @njit(cache=True)
    # def calculate_muscle_force(
    #     muscle_force, muscle_activation, max_muscle_stress, weight, muscle_area
    # ):
    #     muscle_force[:] = (muscle_activation * max_muscle_stress * weight) * muscle_area

    @staticmethod
    @njit(cache=True)
    def calculate_force_and_couple(
        muscle_force,
        muscle_tangent,
        muscle_position,
        internal_force,
        internal_couple,
        external_force,
        external_couple,
        director_collection,
        kappa,
        tangents,
        rest_lengths,
        rest_voronoi_lengths,
        dilatation,
        voronoi_dilatation,
    ):  
        internal_force[:, :] = muscle_force * muscle_tangent
        _force_induced_couple(internal_force, muscle_position, internal_couple)
        _internal_to_external_load(
            director_collection,
            kappa,
            tangents,
            rest_lengths,
            rest_voronoi_lengths,
            dilatation,
            voronoi_dilatation,
            internal_force,
            internal_couple,
            external_force,
            external_couple,
        )

    def apply_activation(self, activation: Union[float, np.ndarray]):
        """apply_activation.

        Parameters
        ----------
        activation : Union[float, np.ndarray]
            If array of activation is given, the shape of activation is expected to
            match the shape of muscle_activation.
        """
        self.set_activation(self.activation, activation)

    @staticmethod
    @njit(cache=True)
    def set_activation(muscle_activation, activation):
        muscle_activation[:] = activation

    def get_activation(self) -> Union[float, np.ndarray]:
        """
        activation getter

        Returns
        -------
        activation: Union[float, np.ndarray]
        """
        return self.activation


class MuscleGroup(MuscleInfo, ContinuousActuation):
    """MuscleGroup.
    Group of muscle. Provides convinience tools to operate group-activation.
    """

    def __init__(
        self,
        muscles: Iterable[Muscle],
        type_name: str = "muscle_group",
        index: int = 0,
        **kwargs,
    ):
        """__init__.

        Parameters
        ----------
        muscles : Iterable[Muscle]
        """
        super().__init__(
            n_elements=muscles[0].n_elements, type_name=type_name, index=index, **kwargs
        )

        self.muscles = muscles
        for m, muscle in enumerate(self.muscles):
            muscle.index = m
        self.activation = np.zeros(self.muscles[0].activation.shape)
        self.s_activation = self.muscles[0].s_activation.copy()

    def __call__(self, system: elastica.rod.RodBase):
        """__call__.

        Parameters
        ----------
        system : elastica.rod.RodBase
        """
        self.reset_actuation()
        for muscle in self.muscles:
            muscle(system)
            inplace_addition(self.internal_force, muscle.internal_force)
            inplace_addition(self.external_force, muscle.external_force)
            inplace_addition(self.internal_couple, muscle.internal_couple)
            inplace_addition(self.external_couple, muscle.external_couple)

    def set_current_length_as_rest_length(self, system: elastica.rod.RodBase):
        """set_current_length_as_rest_length.

        Parameters
        ----------
        system : elastica.rod.RodBase
        """
        for muscle in self.muscles:
            muscle.set_current_length_as_rest_length(system)

    def apply_activation(self, activation: Union[float, np.ndarray]):
        """apply_activation.

        MuscleGroup apply activation

        Parameters
        ----------
        activation : Union[float, np.ndarray]
            If array of activation is given, the shape of activation is expected to
            match the shape of muscle_activation.
        """
        self.set_activation(self.activation, activation)
        for muscle in self.muscles:
            muscle.set_activation(muscle.activation, self.activation)

    @staticmethod
    @njit(cache=True)
    def set_activation(muscle_activation, activation):
        muscle_activation[:] = activation

    def get_activation(self) -> Union[float, np.ndarray]:
        """
        activation getter

        Returns
        -------
        activation: Union[float, np.ndarray]
        """
        return self.activation


class ApplyMuscles(ApplyActuations):
    """ApplyMuscles."""

    def __init__(
        self, muscles: Iterable[Muscle], step_skip: int, COLLECT_DATA_FOR_POSTPROCESSING, callback_params_list: list
    ):
        """__init__.

        Parameters
        ----------
        muscles : Iterable[Muscle]
        step_skip : int
        callback_params_list : list
        """
        super().__init__(muscles, step_skip, COLLECT_DATA_FOR_POSTPROCESSING, callback_params_list)
        for m, muscle in enumerate(muscles):
            muscle.index = m

    def callback_func(
        self, muscles: Iterable[Muscle], callback_params_list: Iterable[Dict]
    ):
        """callback_func.

        Parameters
        ----------
        muscles : Iterable[Muscle]
        callback_params_list : Iterable[Dict]
        """
        for muscle, callback_params in zip(muscles, callback_params_list):
            callback_params["muscle_info"].append(str(muscle))
            callback_params["s_activation"].append(muscle.s_activation.copy())
            callback_params["activation"].append(muscle.activation.copy())
            callback_params["muscle_length"].append(muscle.muscle_length.copy())
            callback_params["muscle_normalized_length"].append(
                muscle.muscle_normalized_length.copy()
            )
            callback_params["force_length_weight"].append(
                muscle.force_length_weight(muscle.muscle_normalized_length).copy()
            )
            callback_params["muscle_position"].append(muscle.muscle_position.copy())
            callback_params["internal_force"].append(muscle.internal_force.copy())
            callback_params["internal_couple"].append(muscle.internal_couple.copy())
            callback_params["external_force"].append(muscle.external_force.copy())
            callback_params["external_couple"].append(muscle.external_couple.copy())


class ApplyMuscleGroups(ApplyMuscles):
    """ApplyMuscleGroups."""

    def __init__(
        self, muscle_groups: MuscleGroup, step_skip: int, COLLECT_DATA_FOR_POSTPROCESSING, callback_params_list: list
    ):
        """__init__.

        Parameters
        ----------
        muscle_groups : MuscleGroup
        step_skip : int
        callback_params_list : list
        """
        super().__init__(muscle_groups, step_skip, COLLECT_DATA_FOR_POSTPROCESSING, callback_params_list)
        for muscle_group, callback_params in zip(
            muscle_groups, self.callback_params_list
        ):
            callback_params["muscles"] = [
                defaultdict(list) for _ in muscle_group.muscles
            ]

    def callback_func(
        self, muscle_groups: MuscleGroup, callback_params_list: Iterable[Dict]
    ):
        """callback_func.

        Parameters
        ----------
        muscle_groups : MuscleGroup, Iterable[Muscle]
        callback_params_list : Iterable[Dict]
        """
        for muscle_group, callback_params in zip(muscle_groups, callback_params_list):
            callback_params["muscle_group_info"].append(str(muscle_group))
            callback_params["s_activation"].append(muscle_group.s_activation.copy())
            callback_params["activation"].append(muscle_group.activation.copy())
            callback_params["internal_force"].append(muscle_group.internal_force.copy())
            callback_params["internal_couple"].append(
                muscle_group.internal_couple.copy()
            )
            callback_params["external_force"].append(muscle_group.external_force.copy())
            callback_params["external_couple"].append(
                muscle_group.external_couple.copy()
            )
            ApplyMuscles.callback_func(
                self, muscle_group.muscles, callback_params["muscles"]
            )