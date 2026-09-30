from typing import Union

import numpy as np
from numba import njit

from muscles import MuscleForce

def get_x_y_from_z_tendon_parallel(z, x0, y0):
    x = np.full_like(z, x0, dtype=float)
    y = np.full_like(z, y0, dtype=float)
    return x, y



def get_ratio_muscle_position_tendon_parallel(radius_list, x0, y0):
    radius_arr = np.asarray(radius_list)
    n = radius_arr.shape[0]

    z = np.linspace(0, 0.25, n)

    x, y = get_x_y_from_z_tendon_parallel(z, x0, y0)

    x_ratio = x / radius_arr
    y_ratio = y / radius_arr
    
    # print(f"Tendon muscle position ratios (first 5 elements): x: {x_ratio[:5]}, y: {y_ratio[:5]}")

    return np.vstack([x_ratio, y_ratio, np.zeros(n)])



class Tendon1(MuscleForce):

    def __init__(
        self,
        rest_muscle_area,
        max_muscle_stress,
        radius_list,
        **kwargs,
    ):
        n_elem = rest_muscle_area.shape[0]
        super().__init__(
            ratio_muscle_position=get_ratio_muscle_position_tendon_parallel(radius_list, 0.02, 0.0),
            rest_muscle_area=rest_muscle_area,
            max_muscle_stress=max_muscle_stress,
            type_name="tendon1",
            **kwargs,
        )


class Tendon2(MuscleForce):

    def __init__(
        self,
        rest_muscle_area,
        max_muscle_stress,
        radius_list,
        **kwargs,
    ):
        n_elem = rest_muscle_area.shape[0]
        super().__init__(
            ratio_muscle_position=get_ratio_muscle_position_tendon_parallel(radius_list, 0.00, 0.02),
            rest_muscle_area=rest_muscle_area,
            max_muscle_stress=max_muscle_stress,
            type_name="tendon2",
            **kwargs,
        )
        
        
class Tendon3(MuscleForce):

    def __init__(
        self,
        rest_muscle_area,
        max_muscle_stress,
        radius_list,
        **kwargs,
    ):
        n_elem = rest_muscle_area.shape[0]
        super().__init__(
            ratio_muscle_position=get_ratio_muscle_position_tendon_parallel(radius_list, 0.0, 0.02),
            rest_muscle_area=rest_muscle_area,
            max_muscle_stress=max_muscle_stress,
            type_name="tendon3",
            **kwargs,
        )


class Tendon4(MuscleForce):

    def __init__(
        self,
        rest_muscle_area,
        max_muscle_stress,
        radius_list,
        **kwargs,
    ):
        n_elem = rest_muscle_area.shape[0]
        super().__init__(
            ratio_muscle_position=get_ratio_muscle_position_tendon_parallel(radius_list, 0.0, -0.02),
            rest_muscle_area=rest_muscle_area,
            max_muscle_stress=max_muscle_stress,
            type_name="tendon4",
            **kwargs,
        )