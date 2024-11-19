import numpy as np
import cv2
from scipy.optimize import minimize, LinearConstraint
from simulator import Simulator


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
        init_state: np.ndarray,
        dt: float = 0.01,
        time_range: float = 1.0,
        debug: bool = False,
    ):
        self.predHorizon = predHorizon
        self.dt = dt
        self.simSteps = round(time_range / dt)

        # State: [x, y, vx, vy, h, r, beta]
        self.state = init_state

        # Control: [steering, throttle]
        self.control = np.array([0.0, 0.0])

        self.Q = np.diag([10.0, 10.0, 0.0, 0.0, 0.0, 0.0, -0.5])
        self.R = np.diag([0.0, 0.0])

        self.min_delta_dot, self.max_delta_dot = -np.pi / 2, np.pi / 2
        self.min_acc, self.max_acc = -2, 2

        self.sim = Simulator()
        self.debug = debug

    def find_desired_state(self, vehicle_pos, path, look_ahead_distance=0.5):
        distances = np.linalg.norm(path[:, :2] - vehicle_pos[:2], axis=1)
        closest_idx = np.argmin(distances)

        for i in range(closest_idx, len(path)):
            if np.linalg.norm(path[i, :2] - vehicle_pos[:2]) >= look_ahead_distance:
                return path[i]

        return path[-1]

    def step_kinematic(self, state: np.ndarray, control: np.ndarray, dt=0.01):
        x, y, vx, vy, h, r, beta = state
        steering, throttle = control

        l_f, l_r = 0.1651, 0.1651  # m

        beta = np.arctan(l_r / (l_r + l_f) * np.tan(steering))
        norm = lambda a, b: (a**2 + b**2) ** 0.5

        d_vx = throttle - vx
        vx = vx + d_vx * dt
        vy = norm(vx, vy) * np.sin(beta)
        r = vx / (l_f + l_r) * np.tan(steering)

        # Get velocity in global frame
        vx_g = vx * np.cos(h) - vy * np.sin(h)
        vy_g = vx * np.sin(h) + vy * np.cos(h)

        x += vx_g * dt
        y += vy_g * dt
        h += r * dt

        return np.array(
            [x, y, vx, vy, h % (2 * np.pi), r, np.arctan2(vy, vx) if vx != 0 else 0]
        )

    def step(self, state: np.ndarray, control: np.ndarray, dt=0.01) -> np.ndarray:
        x, y, vx, vy, h, r, beta = state
        steering, throttle = control

        if vx < 0.05:
            return self.step_kinematic(state, control, dt)

        iz = 0.08502599670201208  # kg m^2
        l_f, l_r = 0.1651, 0.1651  # m
        m = 4.202  # kg

        Cf, Cr = 79.0, 70.0  # N/rad

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

        return np.array(
            [x, y, vx, vy, h % (2 * np.pi), r, np.arctan2(vy, vx) if vx != 0 else 0]
        )

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
                [ref_state[0], ref_state[1], 0, 0, ref_state[2], 0, 0]
            )
            X_diff[4] = self.angle_diff(X[4], ref_state[2])

            J += X_diff.T @ self.Q @ X_diff + U.T @ self.R @ U
            X = self.step(X, U, self.dt)

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
        paused = False

        if self.debug:
            print(f"Running simulation for {self.simSteps} steps...")

        for i in range(self.simSteps):
            if not paused:
                self.sim.clear_img()
                desired_state = self.find_desired_state(self.state[:2], path)

                controls_mpc = self.solve(self.state, init_controls, path).x
                self.control = controls_mpc[0:2]
                init_controls = controls_mpc
                self.state = self.step(self.state, self.control, self.dt)

                if self.debug:
                    print(f"[{i}] Desired state: {desired_state}")
                    print(f"[{i}] Applying control: {self.control}")
                    print(f"[{i}] New state: {self.state}")

                self.sim.draw_steering(self.control[0])
                self.sim.draw_car(self.state[0], self.state[1], self.state[4])
                self.sim.draw_polyline(path)
                self.sim.draw_point(desired_state)

                self.sim.show_dict(
                    {
                        "desired_x": desired_state[0],
                        "desired_y": desired_state[1],
                        "desired_theta": desired_state[2],
                        "x": self.state[0],
                        "y": self.state[1],
                        "theta": self.state[4],
                        "steering": self.control[0],
                        "throttle": self.control[1],
                    }
                )

            cv2.imshow("Simulator", self.sim.get_img())
            key = cv2.waitKey(1) & 0xFF
            if key == ord("p"):
                paused = not paused

            while paused:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("p"):
                    paused = not paused


if __name__ == "__main__":
    init_state = np.array([0, -1.5, 0.5, 0, 0, 0, 0])
    mpc = MPC(5, init_state, time_range=10, debug=False)

    r = 1.5
    theta = np.linspace(0, 2 * np.pi, 100)
    path = np.column_stack((r * np.cos(theta), r * np.sin(theta), theta + np.pi / 2))

    path[:, 2] = path[:, 2] % (2 * np.pi)

    mpc.run(path)
