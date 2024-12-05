import numpy as np
from scipy.optimize import minimize, LinearConstraint
from models import DynamicsModel


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
        self.Q = np.diag([10.0, 10.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        self.R = np.diag([0.0, 0.0])

        self.min_delta_dot, self.max_delta_dot = -np.pi / 2, np.pi / 2
        self.min_acc, self.max_acc = -2, 2

        self.model = model
        self.debug = debug

    def find_desired_state(self, vehicle_pos, path, look_ahead_distance=0.5):
        distances = np.linalg.norm(path[:, :2] - vehicle_pos[:2], axis=1)
        closest_idx = np.argmin(distances)

        for i in range(closest_idx, len(path)):
            if np.linalg.norm(path[i, :2] - vehicle_pos[:2]) >= look_ahead_distance:
                return path[i]

        return path[-1]

    def angle_diff(self, theta1, theta2):
        difference = (theta1 - theta2) % (2 * np.pi)
        if difference > np.pi:
            difference -= 2 * np.pi
        return abs(difference)

    def mpc_cost(
        self,
        control_seq: np.ndarray,
        init_state: np.ndarray,
        path: np.ndarray,
    ):
        J = 0
        X = init_state

        for i in range(self.predHorizon):
            U = control_seq[2 * i : 2 * i + 2]
            ref_state = self.find_desired_state(X[:2], path)

            X_diff = X - np.array(
                [ref_state[0], ref_state[1], ref_state[2], 0, 0, 0, 0]
            )
            X_diff[2] = self.angle_diff(X[2], ref_state[2])

            J += X_diff.T @ self.Q @ X_diff + U.T @ self.R @ U
            X = self.model.step_and_return(X, U, self.dt)

        return J

    def solve(
        self, init_state: np.ndarray, init_controls: np.ndarray, path: np.ndarray
    ):
        bounds = [(-np.pi / 4, np.pi / 4), (0.0, 6.0)] * self.predHorizon

        delta_steering = 4 * np.pi / 3
        delta_throttle = 50.0
        delta_controls = np.array([delta_steering, delta_throttle] * self.predHorizon)

        constraint = LinearConstraint(
            (np.eye(2 * self.predHorizon) - np.eye(2 * self.predHorizon, k=2))
            / self.dt,
            -delta_controls,
            delta_controls,
        )

        result = minimize(
            self.mpc_cost,
            init_controls,
            args=(init_state, path),
            bounds=bounds,
            constraints=[constraint],
        )
        return result

    def run(self, path: np.ndarray):
        init_controls = np.zeros(2 * self.predHorizon)

        for _ in range(self.simSteps):
            controls_mpc = self.solve(self.model.state, init_controls, path).x
            init_controls = controls_mpc

            self.control = controls_mpc[0:2]
            self.controls_history.append(self.control)
            self.model.step(self.control, self.dt)

    @property
    def controls(self):
        return np.array(self.controls_history)


"""
TODO:
* Need to normalize error terms by dividing by the max desired error value of each state variable
* Tune contraints and costs
* Need to add a terminal cost to the cost function
* Potentially switch control vector to be desired [delta_dot, acceleration] instead of [desired delta, desired velocity]
"""
