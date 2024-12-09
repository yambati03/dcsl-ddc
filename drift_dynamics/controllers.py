import numpy as np
from scipy.optimize import minimize, LinearConstraint
from models import DynamicsModel
from path import Path
import utils


class MPC:
    """
    Class to control a dynamic bicycle model using Model Predictive Control (MPC)

    args:
        N: int, number of steps to simulate
        init_state: State, initial state of the system
        dt: float, time step for simulation
    """

    def __init__(
        self,
        predHorizon: int,
        model: DynamicsModel,
        dt: float = 0.01,
        time_range: float = 1.0,
        debug: bool = False,
    ):
        self.predHorizon = predHorizon
        self.dt = dt
        self.simSteps = round(time_range / dt)

        # Control: [steering, throttle]
        self.control = np.array([0.0, 0.0])
        self.controls_history = []

        # (x, y, h, V, beta, r, w)
        self.Q = np.diag([1.2, 1.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.R = np.diag([0.0, 0.0])

        self.desired_error = np.array([0.1, 0.1, 0.26, 1.0, 1.0, 1.0, 1.0, 1.0])

        self.min_delta_dot, self.max_delta_dot = -np.pi, np.pi
        self.min_T_dot, self.max_T_dot = -25, 25

        self.min_delta, self.max_delta = -np.pi / 6, np.pi / 6
        self.min_T, self.max_T = 0, 5

        self.model = model
        self.debug = debug

        self.lookahead_distance = 0.5

    # def find_desired_state(self, vehicle_pos, path, look_ahead_distance=0.5):
    #     distances = np.linalg.norm(path[:, :2] - vehicle_pos[:2], axis=1)
    #     closest_idx = np.argmin(distances)

    #     i = closest_idx
    #     while np.linalg.norm(path[i, :2] - vehicle_pos[:2]) < look_ahead_distance:
    #         i = (i + 1) % len(path)

    #     return path[i]

    def mpc_cost(
        self,
        control_seq: np.ndarray,
        init_state: np.ndarray,
    ):
        J = 0
        X = init_state

        for i in range(self.predHorizon):
            U = control_seq[2 * i : 2 * i + 2]
            ref_state = self.path.get_state_by_s(X[7] + self.lookahead_distance)

            X_diff = X - np.array(
                [ref_state[0], ref_state[1], ref_state[2], 0, 0, 0, 0, 0]
            )
            X_diff[2] = utils.angle_diff(X[2], ref_state[2])

            X_diff = X_diff / self.desired_error

            J += X_diff.T @ self.Q @ X_diff + U.T @ self.R @ U

            curr_path_state = self.path.get_state_by_s(X[7])
            X = self.model.step_and_return(X, U, curr_path_state[2], dt=self.dt)

        return J

    def solve(self, init_state: np.ndarray, init_controls: np.ndarray):
        bounds = [
            (self.min_delta, self.max_delta),
            (self.min_T, self.max_T),
        ] * self.predHorizon

        min_control_dots = np.array(
            [self.min_delta_dot, self.min_T_dot] * self.predHorizon
        )
        max_control_dots = np.array(
            [self.max_delta_dot, self.max_T_dot] * self.predHorizon
        )

        constraint = LinearConstraint(
            (np.eye(2 * self.predHorizon) - np.eye(2 * self.predHorizon, k=2))
            / self.dt,
            min_control_dots,
            max_control_dots,
        )

        result = minimize(
            self.mpc_cost,
            init_controls,
            args=(init_state),
            bounds=bounds,
            constraints=[constraint],
        )
        return result

    def run(self, path: np.ndarray):
        init_controls = np.zeros(2 * self.predHorizon)
        self.path = Path(path)

        for _ in range(self.simSteps):
            controls_mpc = self.solve(self.model.state, init_controls).x
            init_controls = controls_mpc

            self.control = controls_mpc[0:2]
            self.controls_history.append(self.control)

            curr_path_state = self.path.get_state_by_s(self.model.state[7])
            self.model.step(self.control, curr_path_state[2], dt=self.dt)

    @property
    def controls(self):
        return np.array(self.controls_history)
