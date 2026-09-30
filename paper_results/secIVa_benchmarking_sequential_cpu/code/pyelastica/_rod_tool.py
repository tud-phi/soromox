import numpy as np
from numba import njit

from elastica._linalg import _batch_matvec, _batch_norm
from elastica._calculus import _difference, _average
from elastica._rotations import _get_rotation_matrix, _inv_rotate

@njit(cache=True)
def inverse(matrix_collection):
    output_matrix_collection = np.empty(matrix_collection.shape)
    for n in range(output_matrix_collection.shape[2]):
        output_matrix_collection[:, :, n] = (
            np.linalg.inv(matrix_collection[:, :, n])
        )
    return output_matrix_collection

@njit(cache=True)
def _lab_to_material(directors, lab_vectors):
    return _batch_matvec(directors, lab_vectors)

@njit(cache=True)
def _material_to_lab(directors, material_vectors):
    blocksize = material_vectors.shape[1]
    lab_vectors = np.zeros((3, blocksize))
    for n in range(blocksize):
        for i in range(3):
            for j in range(3):
                lab_vectors[i, n] += (
                    directors[j, i, n] * material_vectors[j, n]
                )
    return lab_vectors

@njit(cache=True)
def average1D(vector_collection):
    blocksize = vector_collection.shape[0]-1
    output_vector = np.zeros(blocksize)
    for n in range(blocksize):
        output_vector[n] = (vector_collection[n]+vector_collection[n+1])/2
    return output_vector

@njit(cache=True)
def average2D(vector_collection):
    blocksize = vector_collection.shape[1]-1
    output_vector = np.zeros((3, blocksize))
    for n in range(blocksize):
        for i in range(3):
            output_vector[i, n] = (
                (vector_collection[i, n]+vector_collection[i, n+1])/2
            )
    return output_vector

@njit(cache=True)
def difference2D(vector_collection):
    blocksize = vector_collection.shape[1]-1
    output_vector = np.zeros((3, blocksize))
    for n in range(blocksize):
        for i in range(3):
            output_vector[i, n] = (
                vector_collection[i, n+1]-vector_collection[i, n]
            )
    return output_vector

@njit(cache=True)
def calculate_dilatation(sigma):
    shear = sigma_to_shear(sigma)
    dilatation = _batch_norm(shear)
    voronoi_dilatation = (dilatation[:-1] + dilatation[1:])/2
    return dilatation, voronoi_dilatation

# @njit(cache=True)
# def calculate_length(position_collection):
#     length = 0
#     for n in range(position_collection.shape[1]-1):
#         length += np.sqrt(
#             (position_collection[0, n+1] - position_collection[0, n])**2 +
#             (position_collection[1, n+1] - position_collection[1, n])**2 +
#             (position_collection[2, n+1] - position_collection[2, n])**2
#         )
#     return length

@njit(cache=True)
def calculate_distance_to_a_point(position_collection, point_position):
    blocksize = position_collection.shape[1]
    distance_collection = np.zeros(blocksize)
    for n in range(blocksize):
        distance_collection[n] = (
            (position_collection[0, n]-point_position[0])**2 +
            (position_collection[1, n]-point_position[1])**2 +
            (position_collection[2, n]-point_position[2])**2
        )**0.5
    return distance_collection

@njit(cache=True)
def calculate_distance(position_collection_1, position_collection_2):
    blocksize = position_collection_1.shape[1]
    distance_collection = np.zeros(blocksize)
    for n in range(blocksize):
        distance_collection[n] = (
            (position_collection_1[0, n]-position_collection_2[0, n])**2 +
            (position_collection_1[1, n]-position_collection_2[1, n])**2 +
            (position_collection_1[2, n]-position_collection_2[2, n])**2
        )**0.5
    return distance_collection

@njit(cache=True)
def sigma_to_shear(sigma):
    shear = np.zeros(sigma.shape)
    for n in range(shear.shape[1]):
        shear[0, n] = sigma[0, n]
        shear[1, n] = sigma[1, n]
        shear[2, n] = sigma[2, n] + 1
    return shear

@njit(cache=True)
def kappa_to_curvature(kappa, voronoi_dilatation):
    curvature = np.zeros(kappa.shape)
    for n in range(curvature.shape[1]):
        curvature[0, n] = kappa[0, n] / voronoi_dilatation[n]
        curvature[1, n] = kappa[1, n] / voronoi_dilatation[n]
        curvature[2, n] = kappa[2, n] / voronoi_dilatation[n]
    return curvature

class StaticRod:
    def __init__(
            self, rest_position, rest_director, rest_radius, shear_matrix, bend_matrix
        ):
        self.n_elements = rest_radius.shape[0]
        self.shear_matrix = shear_matrix.copy()
        self.bend_matrix = bend_matrix.copy()
        self.position_collection = rest_position.copy()
        self.director_collection = rest_director.copy()
        
        self.rest_radius = rest_radius.copy()
        self.rest_lengths = _batch_norm(
            _difference(self.position_collection)  # Position difference
        )
        self.rest_voronoi_lengths = 0.5 * (
            self.rest_lengths[1:] + self.rest_lengths[:-1]
        )

        self.lengths = np.zeros(self.n_elements)
        self.tangents = np.zeros((3, self.n_elements))
        self.radius = np.zeros(self.n_elements)

        self.dilatation = np.zeros(self.n_elements)
        self.voronoi_dilatation = np.zeros(self.n_elements-1)

        self.sigma = np.zeros((3, self.n_elements))
        self.kappa = np.zeros((3, self.n_elements-1))

        self._compute_geometry_from_state(
            self.position_collection, self.rest_lengths, self.rest_radius,
            self.lengths, self.tangents, self.radius
        )
        self._compute_all_dilatations(
            self.lengths,
            self.rest_lengths,
            self.rest_voronoi_lengths,
            self.dilatation,
            self.voronoi_dilatation,
        )
        self._compute_shear_stretch_strains(
            self.tangents,
            self.dilatation,
            self.director_collection,
            self.sigma,
        )
        self._compute_bending_twist_strains(
            self.director_collection, 
            self.rest_voronoi_lengths,
             self.kappa
        )

        self.rest_sigma = self.sigma.copy()
        self.rest_kappa = self.kappa.copy()

    @staticmethod
    @njit(cache=True)
    def _compute_bending_twist_strains(
        director_collection, rest_voronoi_lengths, kappa
    ):
        
        temp = _inv_rotate(director_collection)
        blocksize = rest_voronoi_lengths.shape[0]
        for k in range(blocksize):
            # kappa[0, k] = temp[0, k] / rest_voronoi_lengths[k]
            # kappa[1, k] = temp[1, k] / rest_voronoi_lengths[k]
            # kappa[2, k] = temp[2, k] / rest_voronoi_lengths[k]

            # The following should be the right implementation once the _inv_rotate is fixed
            kappa[0, k] = -temp[0, k] / rest_voronoi_lengths[k]
            kappa[1, k] = -temp[1, k] / rest_voronoi_lengths[k]
            kappa[2, k] = -temp[2, k] / rest_voronoi_lengths[k]

    @staticmethod
    @njit(cache=True)
    def _compute_shear_stretch_strains(
        tangents,
        dilatation,
        director_collection,
        sigma,
    ):
        # Quick trick : Instead of evaliation Q(et-d^3), use property that Q*d3 = (0,0,1), a constant
        z_vector = np.array([0.0, 0.0, 1.0]).reshape(3, -1)
        sigma[:] = dilatation * _batch_matvec(director_collection, tangents) - z_vector

    @staticmethod
    @njit(cache=True)
    def _compute_all_dilatations(
        lengths,
        rest_lengths,
        rest_voronoi_lengths,
        dilatation,
        voronoi_dilatation,
    ):
        for k in range(lengths.shape[0]):
            dilatation[k] = lengths[k] / rest_lengths[k]

        # Cmopute eq (3.4) from 2018 RSOS paper
        voronoi_lengths = _average(lengths)  # Position average

        # Cmopute eq (3.5) from 2018 RSOS paper
        for k in range(voronoi_lengths.shape[0]):
            voronoi_dilatation[k] = voronoi_lengths[k] / rest_voronoi_lengths[k]

    @staticmethod
    @njit(cache=True)
    def _compute_geometry_from_state(
        position_collection, rest_lengths, rest_radius, lengths, tangents, radius
    ):
        # Compute eq (3.3) from 2018 RSOS paper
        position_diff = _difference(position_collection)  # Position difference
        lengths[:] = _batch_norm(position_diff)
        for k in range(lengths.shape[0]):
            tangents[0, k] = position_diff[0, k] / lengths[k]
            tangents[1, k] = position_diff[1, k] / lengths[k]
            tangents[2, k] = position_diff[2, k] / lengths[k]
            # recalculate radius based on volume conservation
            radius[k] = rest_radius[k] * np.sqrt(rest_lengths[k]/lengths[k])

    def update_from_strain(self, sigma, kappa):
        self.sigma[:, :] = sigma.copy()
        self.kappa[:, :] = kappa.copy()
        self.static_pose_evolution(
            self.rest_lengths, self.sigma, self.kappa,
            self.position_collection, self.director_collection
        )
        self._compute_geometry_from_state(
            self.position_collection, self.rest_lengths, self.rest_radius,
            self.lengths, self.tangents, self.radius
        )
        self._compute_all_dilatations(
            self.lengths,
            self.rest_lengths,
            self.rest_voronoi_lengths,
            self.dilatation,
            self.voronoi_dilatation,
        )

    @staticmethod
    @njit(cache=True)
    def static_pose_evolution(
        rest_lengths, sigma, kappa,
        position_collection, director_collection
    ):
        shear = sigma_to_shear(sigma)
        for k in range(rest_lengths.shape[0]-1):
            next_position(
                director_collection[:, :, k],
                shear[:, k] * rest_lengths[k],
                position_collection[:, k:k+2]
                )
            next_director(
                kappa[:, k] * rest_lengths[k],
                director_collection[:, :, k:k+2]
                )

        next_position(
            director_collection[:, :, -1],
            shear[:, -1] * rest_lengths[-1],
            position_collection[:, -2:]
            )

    @classmethod
    def get_rod(cls, rest_cosserat_rod):
        return StaticRod(
            rest_cosserat_rod.position_collection,
            rest_cosserat_rod.director_collection,
            rest_cosserat_rod.radius,
            rest_cosserat_rod.shear_matrix,
            rest_cosserat_rod.bend_matrix
        )

@njit(cache=True)
def next_position(director, delta, positions):
    positions[:, 1] = positions[:, 0]
    for i in range(3):
        for j in range(3):
            positions[i, 1] += (
                director[j, i] * delta[j]
            )

@njit(cache=True)
def next_director(rotation, directors):
    # FIXME The following should be the right implementation once the _get_rotation_matrix is fixed
    # Rotation = _get_rotation_matrix(-1, rotation.reshape((3, 1)))[:, :, 0]
    
    Rotation = _get_rotation_matrix(1, rotation.reshape((3, 1)))[:, :, 0]
    for i in range(3):
        for j in range(3):
            directors[i, j, 1] = 0
            for k in range(3):
                directors[i, j, 1] += (
                    Rotation[i, k] * directors[k, j, 0]
                )