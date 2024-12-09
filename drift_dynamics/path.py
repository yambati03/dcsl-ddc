import numpy as np
from scipy.interpolate import interp1d


class Path:
    # waypoints should have shape (n, 3) where n is the number of waypoints, and the columns are (x, y, theta)
    def __init__(self, waypoints: np.ndarray):
        self.waypoints = waypoints
        self.arc_lengths = self.compute_arc_length()

        self.init_interpolators()

    def compute_arc_length(self):
        """
        Compute the approximate arc length of the path.
        """
        diffs = np.diff(self.waypoints[:, :2], axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)

        s = np.zeros(len(self.waypoints))
        s[1:] = np.cumsum(segment_lengths)
        return s

    def init_interpolators(self):
        if self.arc_lengths is None:
            raise RuntimeError(
                "Arc lengths must be defined before calling init_interpolators."
            )

        self.interp_x = interp1d(
            self.arc_lengths,
            self.waypoints[:, 0],
            kind="cubic",
            fill_value="extrapolate",
        )
        self.interp_y = interp1d(
            self.arc_lengths,
            self.waypoints[:, 1],
            kind="cubic",
            fill_value="extrapolate",
        )

        unwrapped_headings = np.unwrap(self.waypoints[:, 2])
        self.interp_h = interp1d(
            self.arc_lengths,
            unwrapped_headings,
            kind="linear",
            fill_value="extrapolate",
        )

    def get_state_by_s(self, s: float):
        if not self.interp_x or not self.interp_y or not self.interp_h:
            raise RuntimeError("Interpolators not defined.")

        s = s % self.arc_lengths[-1]
        return np.array([self.interp_x(s), self.interp_y(s), self.interp_h(s)])
