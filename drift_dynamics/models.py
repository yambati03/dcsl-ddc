import numpy as np


class DynamicsModel:
    def __init__(self, state: np.ndarray):
        self.state = state

    def _step_kinematic(
        self, state: np.ndarray, control: np.ndarray, dt=0.01
    ) -> np.ndarray:
        raise NotImplementedError

    def _step_dynamic(
        self, state: np.ndarray, control: np.ndarray, dt=0.01
    ) -> np.ndarray:
        raise NotImplementedError

    def step_and_return(
        self,
        state: np.ndarray,
        control: np.ndarray,
        dt=0.01,
        v_idx=3,
        v_threshold=0.05,
    ) -> np.ndarray:
        if state[v_idx] < v_threshold:
            return self._step_kinematic(state, control, dt=dt)
        return self._step_dynamic(state, control, dt=dt)

    def step(
        self,
        control: np.ndarray,
        dt=0.01,
        v_idx=3,
        v_threshold=0.05,
    ):
        self.state = self.step_and_return(
            self.state, control, dt=dt, v_idx=v_idx, v_threshold=v_threshold
        )


"""
Derived from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10349953
"""


class DynamicBicycleModel(DynamicsModel):
    def __init__(self, Izz, m, l_f, l_r, Cf, Cx, Cy, mu, R, J, init_state: np.ndarray):
        super().__init__(init_state)

        # Vehicle parameters
        self.Izz = Izz  # Moment of inertia around z-axis
        self.m = m  # Mass of vehicle
        self.l_f = l_f  # Distance from CoG to front axle
        self.l_r = l_r  # Distance from CoG to rear axle
        self.Cf = Cf  # Front cornering stiffness
        self.Cx = Cx  # Longitudinal stiffness
        self.Cy = Cy  # Lateral stiffness
        self.mu = mu  # Friction coefficient
        self.R = R  # Effective rear wheel radius
        self.J = J  # Drivetrain inertia

    def front_slip_angle(self, V, beta, r, delta):
        return np.arctan2(V * np.sin(beta) + self.l_f * r, V * np.cos(beta)) - delta

    def rear_slip_angle(self, V, beta, r):
        return np.arctan2(V * np.sin(beta) - self.l_r * r, V * np.cos(beta))

    def rear_slip_ratio(self, omega_R, V, beta):
        return (self.R * omega_R - V * np.cos(beta)) / max(
            V * np.cos(beta), self.R * omega_R
        )

    def tire_forces(self, alpha_F, alpha_R, s):
        """Compute front and rear tire forces."""

        # Front lateral force with saturation
        FyF = -self.Cf * alpha_F
        FzF = self.m * 9.81 * self.l_r / (self.l_f + self.l_r)
        FyF = np.clip(FyF, -self.mu * FzF, self.mu * FzF)

        # Combined-slip model for rear tire forces
        f = np.sqrt(
            (self.Cx * s / (s + 1)) ** 2 + (self.Cy * np.tan(alpha_R) / (s + 1)) ** 2
        )

        if f == 0:
            return 0, 0, 0

        FzR = self.m * 9.81 * self.l_f / (self.l_f + self.l_r)
        F = f if f <= self.mu * FzR else self.mu * FzR

        FxR = (F * self.Cx * s) / (f * (s + 1))
        FyR = -F * self.Cy * np.tan(alpha_R) / (f * (s + 1))

        return FyF, FxR, FyR

    def compute_r_dot(self, FyF, FyR, delta):
        return (1 / self.Izz) * (self.l_f * FyF * np.cos(delta) - self.l_r * FyR)

    def compute_V_dot(self, FxR, FyF, FyR, beta, delta):
        return (
            FxR * np.cos(beta) - FyF * np.sin(delta - beta) + FyR * np.sin(beta)
        ) / self.m

    def compute_beta_dot(self, FyF, FxR, FyR, beta, r, V, delta):
        return (
            FyF * np.cos(delta - beta) - FxR * np.sin(beta) + FyR * np.cos(beta)
        ) / (self.m * V) - r

    def compute_w_dot(self, torque, FxR):
        return (torque - self.R * FxR) / self.J

    def compute_dynamics(self, state: np.ndarray, control: np.ndarray):
        # Unpack state and control
        _, _, _, V, beta, r, w = state
        delta, torque = control

        # Compute slip angles and slip ratio
        alpha_F = self.front_slip_angle(V, beta, r, delta)
        alpha_R = self.rear_slip_angle(V, beta, r)
        s = self.rear_slip_ratio(w, V, beta)

        # Tire forces
        FyF, FxR, FyR = self.tire_forces(alpha_F, alpha_R, s)

        # Dynamics
        r_dot = self.compute_r_dot(FyF, FyR, delta)
        V_dot = self.compute_V_dot(FxR, FyF, FyR, beta, delta)
        beta_dot = self.compute_beta_dot(FyF, FxR, FyR, beta, r, V, delta)
        w_dot = self.compute_w_dot(torque, FxR)

        return V_dot, beta_dot, r_dot, w_dot

    def _step_dynamic(self, state: np.ndarray, control: np.ndarray, dt=0.01):
        """Compute the vehicle's dynamics.

        args:
            state (np.ndarray): Vehicle state vector (x, y, h, V, beta, r, w)
            control (np.ndarray): Control input vector (delta, torque)
            dt (float): Time step width
        """

        V_dot, beta_dot, r_dot, w_dot = self.compute_dynamics(state, control)

        # Unpack state
        x, y, h, V, beta, r, w = state

        # Update position
        Vx = V * np.cos(beta)
        Vy = V * np.sin(beta)

        VxG = Vx * np.cos(h) - Vy * np.sin(h)
        VyG = Vx * np.sin(h) + Vy * np.cos(h)

        next_state = np.zeros_like(state)

        next_state[0] = x + VxG * dt
        next_state[1] = y + VyG * dt

        # Update state
        next_state[2] = h + r * dt
        next_state[3] = V + V_dot * dt
        next_state[4] = beta + beta_dot * dt
        next_state[5] = r + r_dot * dt
        next_state[6] = w + w_dot * dt

        return next_state

    def _step_kinematic(self, state: np.ndarray, control: np.ndarray, dt=0.01):
        # Unpack state and control
        x, y, h, V, beta, _, _ = state  # Ignore `r` and `w` for initial calculation
        delta, torque = control  # Include torque to update velocity

        # Kinematic equations of motion
        L = self.l_f + self.l_r  # Total wheelbase
        beta = np.arctan(self.l_r / L * np.tan(delta))  # Slip angle due to steering

        # Compute velocities in the global frame
        VxG = V * np.cos(h + beta)
        VyG = V * np.sin(h + beta)

        # Approximate acceleration using torque
        acceleration = torque / (self.m * self.R)
        V_new = max(V + acceleration * dt, 0)  # Enforce non-negative velocity

        # Update state using kinematic equations
        next_state = np.zeros_like(state)
        next_state[0] = x + VxG * dt  # x position
        next_state[1] = y + VyG * dt  # y position
        next_state[2] = h + (V / L) * np.sin(beta) * dt  # Heading
        next_state[3] = V_new  # Update velocity
        next_state[4] = beta  # Beta remains constant for this step

        # Update yaw rate and wheel speed for consistency with dynamics
        next_state[5] = (V / L) * np.sin(beta)  # Approximate yaw rate
        next_state[6] = V_new / self.R  # Approximate wheel speed

        return next_state
