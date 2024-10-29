import numpy as np
from scipy.optimize import minimize, LinearConstraint
from dataclasses import dataclass

@dataclass
class State:
    x: float = 0.0
    vx_g: float = 0.0
    y: float = 0.0
    vy_g: float = 0.0
    h: float = 0.0
    r: float = 0.0

    def to_tuple(self):
        return (self.x, self.vx_g, self.y, self.vy_g, self.h, self.r)
    
    def to_numpy(self):
        return np.array(self.to_tuple())

class MPC:
    """
    Class to control a dynamic bicycle model using Model Predictive Control (MPC)

    args:
        N: int, number of steps to simulate
        init_state: State, initial state of the system
        dt: float, time step for simulation
    """
    def __init__(self, N: int, init_state: State, dt: float = 0.1, time_range: float = 1.0):
        self.N = N
        self.dt = dt
        self.predHorizon = round(time_range / dt)

        self.state = init_state
        self.control= np.array([0.0, 0.0])

        # Parameters for MPC

        self.Q = np.diag([1., 1., 1., 1., 1., 1.])
        self.R = np.diag([1., 1.])

        self.min_delta_dot, self.max_delta_dot = -np.pi/2, np.pi/2
        self.min_acc, self.max_acc = -2, 2

    def step(self, state: State, control: np.ndarray, dt=0.01):
        x, vx_g, y, vy_g, h, r = self.state.to_numpy()
        steering, throttle = self.control

        l_f = 0.1651  # m
        l_r = 0.1651  # m
        m = 4.202  # kg
        iz = 0.08502599670201208 # kg m^2

        # Get velocity in local frame (x is forward, y is left -- this follows the right hand rule)
        vx = vx_g * np.cos(h) + vy_g * np.sin(h)
        vy = -vx_g * np.sin(h) + vy_g * np.cos(h)

        if vx < 0.05:
            self.step_kinematic(dt)
            return

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

        self.state = State(x, vx_g, y, vy_g, h, r)

    def mpc_cost(self, controls: np.ndarray, init_state: State):
        J = 0
        X = init_state.to_diagonal_matrix()

        # Simulate the next N steps of the system
        for i in range(self.N):
            J += (X - self.state.to_diagonal_matrix()).T @ self.Q @ (X - self.state.to_diagonal_matrix()) + controls[i].T @ self.R @ controls[i]
            self.step(self.state, controls, self.dt)

        return J
    
    def solve(self, init_state: State):
        result = minimize(self.mpc_cost, init_state, args=(init_state), bounds=[(-1, 1), (-1, 1)], constraints=None, method="SLSQP")
        return result
    
    def run(self):
        # Generate desired trajectory in parametric form
        x_ref = np.sin(t)
        y_ref = np.cos(t)

        for i in range(self.L):
            # Gener

"""
In MPC, Q represents the weight on the state variables and R represents the weight on the control input.
General form: J = sum(Q * (x - x_ref)^2 + R * u^2)

In the bicycle model, the state variables are x, y, h, vx, vy, r and the control inputs are steering and throttle.
The reference states are x_ref, y_ref, h_ref

Given the above information, the cost function can be defined as:
J = Q_x * (x - x_ref)^2 + Q_y * (y - y_ref)^2 + Q_h * (h - h_ref)^2 + R_steering * steering^2 + R_throttle * throttle^2
"""

# Set Speed, sideslip and steering angle. Vary velocity of the vehicle and generate a phase plot of the vehicle

# Try cross-track error, orientatoin error, and extra penalty for steering angle
# Impose a linear constraint on the rate of change of the control input