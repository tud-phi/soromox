import jax
jax.config.update("jax_enable_x64", True)  # double precision
from jax import Array
from jax import numpy as jnp
from numpy.testing import assert_allclose
from typing import Dict, Tuple

from soromox.systems.tendon_actuated_pcs import TendonActuatedPCS
from soromox.utils.tolerance import Tolerance


def test_actuation_matrix_pcs():
    """
    Test the methods responsible for the computation of the actuation matrix of the
    tendon actuated Piecewise Constant Strain class.
    """

    def createRobot(segment_lengths: Array, tendon_params: Dict[str, Array]) -> TendonActuatedPCS:
        """
        Creates an object of the class TendonActuatedPCS with the specified segments' length
        and tendon routing parameters.

        Args:
            segment_lengths (Array): lengths of the segments of the robot (num_segments, )
            tendon_params (Dict[str, Array]): routing parameters of the given tendon

        Returns:
            robot (TendonActuatedPCS): object representing the soft robot
        """
        num_segments = segment_lengths.shape[0]
        rho = 1070 * jnp.ones(
            (num_segments,)
        )  # Volumetric density of Dragon Skin 20 [kg/m^3]
        params = {
            "p0": jnp.zeros(6),  # Initial position and orientation
            "r": 1.0 * jnp.ones((num_segments,)),  # default: 2e-2
            "rho": rho,
            "g": jnp.array([0.0, 0.0, 9.81]),  # Gravity vector [m/s^2]
            "E": 2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
            "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
        }
        params["L"] = segment_lengths
        params["D"] = 1e-3 * jnp.diag(
            (
                jnp.repeat(
                    jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), num_segments, axis=0
                )
                * params["L"][:, None]
            ).flatten()
        )

        robot = TendonActuatedPCS(
            num_segments=num_segments,
            params=params,
            order_gauss=5,
            tendon_routing_params=tendon_params,
        )

        return robot

    def reference_actuation_matrix(tendon_params: Dict[str, Array], l_tot: Array) -> Array:
        """
        Compute the actuation matrix of a soft robot of length l_tot w.r.t. one
        linear tendon at the straight configuration corresponding to vector state
        q0 = zeros(6*num_segments), considered as reference.
        This configuration represents the rest (stress-free) shape of the robot
        and it is easily tractable analitically.

        Args:
            tendon_params (Dict[str, Array]): routing parameters of the given tendon
            l_tot (Array): total length of the robot

        Returns:
            A (Array): actuation matrix at straight configuration
        """
        ry, rz, my, mz = (
            tendon_params["ry"][0],
            tendon_params["rz"][0],
            tendon_params["my"][0],
            tendon_params["mz"][0],
        )
        A = jnp.array(
            [
                l_tot * (-my * rz + mz * ry),
                l_tot**2 * mz / 2 + l_tot * rz,
                -(l_tot**2) * my / 2 - l_tot * ry,
                l_tot,
                l_tot * my,
                l_tot * mz,
            ]
        )

        return A / jnp.sqrt(my**2 + mz**2 + 1)

    def reference_actuation_basis(tendon_params: Dict[str, Array], q: Array, s: Array) -> Array:
        """
        Computes the vector representing the actuation basis of the given tendon
        at configuration q (assuming the strain basis to be the identity) and at
        abscissa point s for a single CS segment.

        Args:
            tendon_params (Dict[str, Array]): routing parameters of the given tendon
            q (Array): strains of the robot made by one single segment (6,)
            s (Array): abscissa point ()

        Returns:
            Phi_a (Array): actuation basis of the tendon at q, s (6,)
        """
        xi = jnp.eye(q.shape[0]) @ q + xi_ref
        ry, rz, my, mz = (
            tendon_params["ry"][0],
            tendon_params["rz"][0],
            tendon_params["my"][0],
            tendon_params["mz"][0],
        )
        xi_1, xi_2, xi_3, xi_4, xi_5, xi_6 = xi[0], xi[1], xi[2], xi[3], xi[4], xi[5]
        Phi_a = jnp.array(
            [
                (my * s + ry) * (mz + xi_1 * (my * s + ry) + xi_6)
                + (-mz * s - rz) * (my - xi_1 * (mz * s + rz) + xi_5),
                (mz * s + rz) * (xi_2 * (mz * s + rz) - xi_3 * (my * s + ry) + xi_4),
                (-my * s - ry) * (xi_2 * (mz * s + rz) - xi_3 * (my * s + ry) + xi_4),
                xi_2 * (mz * s + rz) - xi_3 * (my * s + ry) + xi_4,
                my - xi_1 * (mz * s + rz) + xi_5,
                mz + xi_1 * (my * s + ry) + xi_6,
            ]
        )
        norm = jnp.linalg.norm(Phi_a[3:])
        Phi_a = Phi_a / norm

        return Phi_a

    # ========================================
    # Test of the functions
    # ========================================

    # test actuation matrix
    print("\nTesting actuation matrix... ------------------------")

    test_cases = [
        (
            jnp.array([0.5]),
            {
                "ry": jnp.array([0.1]),
                "rz": jnp.array([0.05]),
                "my": jnp.array([0.0]),
                "mz": jnp.array([0.0]),
                "idx_seg_att": jnp.array([0]),
            },
        ),
        (
            jnp.array([0.5, 0.6]),
            {
                "ry": jnp.array([0.15]),
                "rz": jnp.array([0.0]),
                "my": jnp.array([0.01]),
                "mz": jnp.array([0.01]),
                "idx_seg_att": jnp.array([1]),
            },
        ),
        (
            jnp.array([0.5, 0.6, 0.7]),
            {
                "ry": jnp.array([0.1]),
                "rz": jnp.array([-0.05]),
                "my": jnp.array([0.01]),
                "mz": jnp.array([-0.05]),
                "idx_seg_att": jnp.array([2]),
            },
        ),
    ]

    for segment_lengths, tendon_params in test_cases:
        robot = createRobot(segment_lengths, tendon_params)
        num_segments = segment_lengths.shape[0]
        q0 = jnp.zeros(6 * num_segments)
        target_actuation_matrix = robot.actuation_matrix(q0).reshape(
            (6, num_segments), order="F"
        )
        target_actuation_matrix = jnp.sum(target_actuation_matrix, axis=1)
        expected_actuation_matrix = reference_actuation_matrix(
            tendon_params, jnp.sum(segment_lengths)
        )
        assert not jnp.isnan(target_actuation_matrix).any(), (
            "Actuation matrix contains NaN!"
        )
        assert_allclose(
            target_actuation_matrix,
            expected_actuation_matrix,
            rtol=Tolerance.rtol(),
            atol=Tolerance.atol(),
        )
        print("[Valid test]\n")

    # test actuation basis
    print("\nTesting actuation basis... ------------------------")
    tendon_params = {
        "ry": jnp.array([0.1]),
        "rz": jnp.array([-0.05]),
        "my": jnp.array([0.01]),
        "mz": jnp.array([-0.05]),
        "idx_seg_att": jnp.array([0]),
    }
    robot = createRobot(jnp.array([0.5]), tendon_params)
    xi_ref = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    test_cases = [
        (
            jnp.zeros_like(xi_ref),
            robot.L_cum[-1] * 0.25,
        ),
        (
            jnp.array([0.0, jnp.pi * 3, jnp.pi * 2, 0.1, 0.2, 0.0]),
            robot.L_cum[-1] * 0.25,
        ),
        (
            jnp.zeros_like(xi_ref),
            robot.L_cum[-1] * 0.7,
        ),
        (
            jnp.array([0.0, jnp.pi * 3, jnp.pi * 2, 0.1, 0.2, 0.0]),
            robot.L_cum[-1] * 0.7,
        ),
    ]

    for q, s in test_cases:
        target_actuation_basis = robot._local_actuation_basis(q, s).squeeze()
        expected_actuation_basis = reference_actuation_basis(tendon_params, q, s)
        assert not jnp.isnan(target_actuation_basis).any(), (
            "Actuation basis contains NaN!"
        )
        assert_allclose(
            target_actuation_basis,
            expected_actuation_basis,
            rtol=Tolerance.rtol(),
            atol=Tolerance.atol(),
        )
        print("[Valid test]\n")


if __name__ == "__main__":
    print("Running tests for tendon actuated Piecewise Constant Strain...")
    test_actuation_matrix_pcs()
