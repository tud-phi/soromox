__all__ = ["B_Fixed", "B_Free", "B_Spherical", "B_Planar", "B_Cylindrical", "B_Helical", "B_Prismatic", "B_Revolute"]
import jax.numpy as jnp
from jax import Array
from jax import lax

from soromox.systems.gvs.operands import JointOperand


def unit_vector_6(index) -> Array:
    """
    Function to return a unit vector in 6D space corresponding to the given index.

    Args:
        index (int): Index of the unit vector to return (0 to 5).

    Returns:
        vec (Array): shape (6,) JAX array.
            A JAX array representing the unit vector in 6D space.
    """
    # vec = jnp.eye(6)[:, index]
    eye = jnp.eye(6)
    vec = lax.dynamic_index_in_dim(eye, index, axis=1, keepdims=False)
    return vec


def vector_with_p_concatenate(index_one, index_pitch, pitch) -> Array:
    """
    Function to create a vector in 6D space with a unit vector for the first index and a scaled unit vector for the pitch.

    Args:
        index_one (int): Index for the unit vector in the first three dimensions (0 to 2).
        index_pitch (int): Index for the pitch vector in the last three dimensions (0 to 2).
        pitch (float): The scaling factor for the pitch vector.

    Returns:
        vec (Array): shape (6,) JAX array.
            A JAX array of shape (6,) representing the concatenated vector.
    """
    v_index_one = jnp.eye(3)[:, index_one]
    v_index_pitch = jnp.eye(3)[:, index_pitch] * pitch
    vec = jnp.concatenate([v_index_one, v_index_pitch])
    return vec


def B_Fixed(operand: JointOperand) -> Array:
    """
    Function to return the basis matrix for a fixed joint.

    Returns:
        B (Array): shape (6, 0) JAX array
            An empty array of shape (6, 0) representing the fixed joint basis.
    """
    return jnp.zeros((6, 0), dtype=jnp.float64)


def B_Free(operand: JointOperand) -> Array:
    """
    Function to return the basis matrix for a free joint.

    Returns:
        B (Array): shape (6, 6) JAX array
            An identity matrix of shape (6, 6) representing the free joint basis.
    """
    return jnp.eye(6, dtype=jnp.float64)


def B_Spherical(operand: JointOperand) -> Array:
    """
    Function to return the basis matrix for a spherical joint.

    Returns:
        B (Array): shape (6, 3) JAX array
            A matrix with three unit vectors in the first three dimensions.
    """
    return jnp.stack([unit_vector_6(0), unit_vector_6(1), unit_vector_6(2)], axis=1)


def B_Planar(operand: JointOperand) -> Array:
    """
    Function to return the basis matrix for a planar joint.

    Args:
        plane_idx (int, 0/1/2): 0:'xy', 1:'yz', 2:'xz'
            The plane of motion for the joint.

    Returns:
        B (Array): shape (6, 3) JAX array
            A matrix with three unit vectors corresponding to the specified plane.
    """

    def case_xy(_):
        return (2, 3, 4)

    def case_yz(_):
        return (0, 4, 5)

    def case_xz(_):
        return (1, 3, 5)

    idx = lax.switch(
        index=operand.plane_idx,
        branches=[
            case_xy,
            case_yz,
            case_xz,
        ],
        operand=None,
    )

    return jnp.stack(
        [unit_vector_6(idx[0]), unit_vector_6(idx[1]), unit_vector_6(idx[2])], axis=1
    )


def B_Cylindrical(operand: JointOperand) -> Array:
    """
    Function to return the basis matrix for a cylindrical joint.

    Args:
        operand.axis_idx (int, 0/1/2): 0:'x', 1:'y', 2:'z'
            The axis of motion for the joint.

    Returns:
        B (Array): shape (6, 2) JAX array
            A matrix with two unit vectors corresponding to the specified axis.
    """
    return jnp.stack(
        [unit_vector_6(operand.axis_idx), unit_vector_6(operand.axis_idx + 3)], axis=1
    )


def B_Helical(operand: JointOperand) -> Array:
    """
    Function to return the basis matrix for a helical joint.

    Args:
        operand.axis_idx (int, 0/1/2): 0:'x', 1:'y', 2:'z'
            The axis of motion for the joint.
        pitch (float):
            The pitch of the helical joint.

    Returns:
        B (Array): shape (6, 1) JAX array
            A matrix with a unit vector in the first three dimensions and a scaled unit vector in the last three dimensions.
    """
    return vector_with_p_concatenate(operand.axis_idx, operand.axis_idx, operand.pitch)[
        None
    ].T


def B_Prismatic(operand: JointOperand) -> Array:
    """
    Function to return the basis matrix for a prismatic joint.

    Args:
        operand.axis_idx (int, 0/1/2): 0:'x', 1:'y', 2:'z'
            The axis of translation for the joint.

    Returns:
        B (Array): shape (6, 1) JAX array
            A matrix with a unit vector in the specified axis of translation.
    """
    return unit_vector_6(operand.axis_idx + 3)[None].T


def B_Revolute(operand: JointOperand) -> Array:
    """
    Function to return the basis matrix for a revolute joint.

    Args:
        operand.axis_idx (int, 0/1/2): 0:'x', 1:'y', 2:'z'
            The axis of rotation for the joint.

    Returns:
        B (Array): shape (6, 1) JAX array
            A matrix with a unit vector in the specified axis of rotation.
    """
    return unit_vector_6(operand.axis_idx)[None].T


if __name__ == "__main__":
    # Example usage
    operand = JointOperand(axis_idx=0, plane_idx=0, pitch=1.0)

    print("B_Fixed:\n", B_Fixed(operand))
    print("B_Free:\n", B_Free(operand))
    print("B_Spherical:\n", B_Spherical(operand))

    for i in range(3):
        operand.plane_idx = i
        print(f"B_Planar (plane {i}):\n", B_Planar(operand))

    for i in range(3):
        operand.axis_idx = i
        print(f"B_Cylindrical (axis {i}):\n", B_Cylindrical(operand))

    for i in range(3):
        operand.axis_idx = i
        print(f"B_Helical (axis {i}):\n", B_Helical(operand))

    for i in range(3):
        operand.axis_idx = i
        print(f"B_Prismatic (axis {i}):\n", B_Prismatic(operand))

    for i in range(3):
        operand.axis_idx = i
        print(f"B_Revolute (axis {i}):\n", B_Revolute(operand))
