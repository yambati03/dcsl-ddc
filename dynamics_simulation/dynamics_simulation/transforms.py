import numpy as np


def get_transform_matrix(rot: float, trans: np.ndarray) -> np.ndarray:
    cos = np.cos(rot)
    sin = np.sin(rot)

    T = np.array([[cos, sin, trans[0]], [sin, -cos, trans[1]], [0, 0, 1]])
    return T


def get_rotation_matrix(rot: float) -> np.ndarray:
    cos = np.cos(rot)
    sin = np.sin(rot)

    T = np.array([[cos, -sin], [sin, cos]])
    return T


if __name__ == "__main__":
    rot = -np.pi / 2

    T_reflect = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    T = get_transform_matrix(rot, np.array([512, 512]))
    vec = np.array([0, 1, 1])
    transformed_vec = T @ vec

    print(transformed_vec[:2])
