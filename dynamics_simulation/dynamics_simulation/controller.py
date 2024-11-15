import numpy as np
import cv2
from scipy.optimize import minimize
from dataclasses import dataclass
import matplotlib.pyplot as plt
from simulator import Simulator


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    h: float = 0.0
    r: float = 0.0

    def to_tuple(self):
        return (self.x, self.y, self.vx, self.vy, self.h, self.r)

    def to_numpy(self):
        return np.array(self.to_tuple())

    @classmethod
    def from_numpy(cls, arr):
        return cls(*arr)


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
        init_state: State,
        dt: float = 0.01,
        time_range: float = 1.0,
    ):
        self.predHorizon = predHorizon
        self.dt = dt
        self.simSteps = round(time_range / dt)

        self.state = init_state
        self.control = np.array([0.0, 0.0])

        self.Q = np.diag([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        self.R = np.diag([1.0, 0.5])

        self.min_delta_dot, self.max_delta_dot = -np.pi / 2, np.pi / 2
        self.min_acc, self.max_acc = -2, 2

        self.sim = Simulator()

    def find_desired_state(self, vehicle_pos, path, look_ahead_distance=0.5):
        distances = np.linalg.norm(path[:, :2] - vehicle_pos[:2], axis=1)
        closest_idx = np.argmin(distances)

        for i in range(closest_idx, len(path)):
            if np.linalg.norm(path[i, :2] - vehicle_pos[:2]) >= look_ahead_distance:
                return path[i]
        return path[-1]

    def step_kinematic(self, state: State, control: np.ndarray, dt=0.01):
        x, y, vx, vy, h, r = state.to_tuple()
        steering, throttle = control[0], control[1]

        lf = 0.1651  # m
        lr = 0.1651  # m

        beta = np.arctan(lr / (lr + lf) * np.tan(steering))
        norm = lambda a, b: (a**2 + b**2) ** 0.5

        d_vx = throttle - vx
        vx = vx + d_vx * dt
        vy = norm(vx, vy) * np.sin(beta)
        r = vx / (lf + lr) * np.tan(steering)

        # Get velocity in global frame
        vx_g = vx * np.cos(h) - vy * np.sin(h)
        vy_g = vx * np.sin(h) + vy * np.cos(h)

        x += vx_g * dt
        y += vy_g * dt
        h += r * dt

        return State(x, y, vx, vy, h, r)

    def step(self, state: State, control: np.ndarray, dt=0.01) -> State:
        x, y, vx, vy, h, r = state.to_tuple()
        steering, throttle = control[0], control[1]

        if vx < 0.05:
            return self.step_kinematic(state, control, dt)

        l_f = 0.1651  # m
        l_r = 0.1651  # m
        m = 4.202  # kg
        iz = 0.08502599670201208  # kg m^2

        Cf = 79.0
        Cr = 70.0

        # Calculate slip angles
        slip_f = np.arctan((vy + l_f * r) / vx) - steering
        slip_r = np.arctan((vy - l_r * r) / vx)

        Fyf = -Cf * slip_f
        Fyr = -Cr * slip_r

        d_vx = throttle - vx  # m/s^2
        d_vy = -vx * r + ((Fyr + Fyf * np.cos(steering)) / m)  # m/s^2
        d_r = (l_f * Fyf * np.cos(steering) - l_r * Fyr) / iz  # rad/s^2

        vx += d_vx * dt
        vy += d_vy * dt
        r += d_r * dt

        # Get velocity in global frame
        vx_g = vx * np.cos(h) - vy * np.sin(h)
        vy_g = vx * np.sin(h) + vy * np.cos(h)

        # Update state
        x += vx_g * dt
        y += vy_g * dt
        h += r * dt

        return State(x, y, vx, vy, h, r)

    def mpc_cost(self, control_seq: np.ndarray, init_state: State, path: np.ndarray):
        J = 0
        X = init_state.to_numpy()

        for i in range(self.predHorizon):
            U = control_seq[2 * i : 2 * i + 2]
            ref_state = self.find_desired_state(X[:2], path)
            X_ref = np.array([ref_state[0], ref_state[1], 0, 0, ref_state[2], 0])

            J += (X - X_ref).T @ self.Q @ (X - X_ref) + U.T @ self.R @ U
            new_state = self.step(
                State.from_numpy(X), control_seq[2 * i : 2 * i + 2], self.dt
            )
            X = new_state.to_numpy()

        return J

    def solve(self, init_state: State, init_controls: np.ndarray, path: np.ndarray):
        bounds = [(-np.pi * 4, np.pi * 4), (-1.0, 2.0)] * self.predHorizon

        result = minimize(
            self.mpc_cost,
            init_controls,
            args=(init_state, path),
            bounds=bounds,
            constraints=None,
            method="SLSQP",
        )
        return result

    def run(self, path: np.ndarray):
        init_controls = np.zeros(2 * self.predHorizon)

        print(f"Running simulation for {self.simSteps} steps...")

        for i in range(self.simSteps):
            self.sim.clear_img()

            self.control = self.solve(self.state, init_controls, path).x[0:2]
            print(f"[{i}] Applying control: {self.control}")

            self.state = self.step(self.state, self.control, self.dt)
            print(f"[{i}] New state: {self.state.to_tuple()}")

            self.sim.draw_steering(self.control[0])
            self.sim.show_raw_state(self.state.to_tuple(), tuple(self.control))
            self.sim.draw_car(self.state.x, self.state.y, self.state.h)

            cv2.imshow("Simulator", self.sim.get_img())
            cv2.waitKey(1)


if __name__ == "__main__":
    init_state = State(0, 0, 1.5, 0, 0, 0)
    mpc = MPC(10, init_state, time_range=10)

    theta = np.linspace(0, 2 * np.pi, 100)
    path = np.column_stack((5 * np.cos(theta), 5 * np.sin(theta), theta))

    plt.plot(path[:, 0], path[:, 1])
    plt.show()

    mpc.run(path)


"""
In MPC, Q represents the weight on the state variables and R represents the weight on the control input.
General form: J = sum(Q * (x - x_ref)^2 + R * u^2)

In the bicycle model, the state variables are x, y, h, vx, vy, r and the control inputs are steering and throttle.
The reference states are x_ref, y_ref, h_ref

Given the above information, the cost function can be defined as:
J = Q_x * (x - x_ref)^2 + Q_y * (y - y_ref)^2 + Q_h * (h - h_ref)^2 + R_steering * steering^2 + R_throttle * throttle^2
"""

# Set Speed, sideslip and steering angle. Vary velocity of the vehicle and generate a phase plot of the vehicle

# Try cross-track error, orientation error, and extra penalty for steering angle
# Impose a linear constraint on the rate of change of the control input
