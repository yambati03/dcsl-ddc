import numpy as np


def angle_diff(theta1, theta2):
    difference = (theta1 - theta2) % (2 * np.pi)
    if difference > np.pi:
        difference -= 2 * np.pi
    return abs(difference)


def compute_arc_length(waypoints: np.ndarray):
    diffs = np.diff(waypoints, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    s = np.zeros(len(waypoints))
    s[1:] = np.cumsum(segment_lengths)

    return s


def get_point_by_s_idx(s: float, path: np.ndarray):
    distances = path[:, 0] - s
    idx = np.where(distances > 0, distances, np.inf).argmin()

    return idx
