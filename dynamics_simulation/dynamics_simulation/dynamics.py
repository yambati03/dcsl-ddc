# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10349953

import numpy as np


class DriftDynamicsModel:
    def __init__(self, Izz, m, l_f, l_r, Cf, Cr, mu, R_e):
        # Vehicle parameters
        self.Izz = Izz  # Moment of inertia around z-axis
        self.m = m  # Mass of vehicle
        self.l_f = l_f  # Distance from CoG to front axle
        self.l_r = l_r  # Distance from CoG to rear axle
        self.Cf = Cf  # Front cornering stiffness
        self.Cr = Cr  # Rear cornering stiffness
        self.mu = mu  # Friction coefficient
        self.R_e = R_e  # Effective rear wheel radius

    def compute_beta(self, Vx, Vy):
        """Compute sideslip angle beta."""
        return np.arctan2(Vy, Vx)

    def front_slip_angle(self, V, beta, r, delta):
        """Compute front slip angle alpha_F."""
        return np.arctan2(V * np.sin(beta) + self.l_f * r, V * np.cos(beta)) - delta

    def rear_slip_angle(self, V, beta, r):
        """Compute rear slip angle alpha_R."""
        return np.arctan2(V * np.sin(beta) - self.l_r * r, V * np.cos(beta))

    def rear_slip_ratio(self, omega_R, V, beta):
        """Compute rear slip ratio kR."""
        return (self.R_e * omega_R - V * np.cos(beta)) / max(
            V * np.cos(beta), 1e-5
        )  # Prevent divide by zero

    def tire_forces(self, alpha_F, alpha_R, kR, r, V, beta, omega_R):
        """Compute front and rear tire forces."""
        # Front lateral force with saturation
        FyF = -self.Cf * alpha_F
        FzF = self.m * 9.81 * self.l_r / (self.l_f + self.l_r)
        FyF = np.clip(FyF, -self.mu * FzF, self.mu * FzF)

        # Combined-slip model for rear tire forces
        f = np.sqrt(
            (self.Cr * kR / (kR + 1)) ** 2 + (self.Cr * np.tan(alpha_R) / (kR + 1)) ** 2
        )
        FzR = self.m * 9.81 * self.l_f / (self.l_f + self.l_r)
        # F = min(f, 3 * self.mu * FzR)
        F = f if f <= 3 * self.mu * FzR else self.mu * FzR
        FxR = self.mu * FzR * np.sin(self.Cr * np.arctan(kR))  # F * kR / (kR + 1)
        FyR = -F * np.tan(alpha_R) / (kR + 1)

        # # Simplified model
        # FzF = self.m * 9.81 * self.l_r / (self.l_f + self.l_r)
        # zF = np.tan(alpha_F)
        # a_slF = np.arctan(3 * self.mu * FzF / self.Cf)

        # FyF_noslip = -self.Cf * zF + ((self.Cf * self.Cf) / (3 * self.mu * FzF)) * np.abs(zF) * zF - ((self.Cf * self.Cf * self.Cf) / (27 * (self.mu * self.mu) * (FzF * FzF))) * zF * zF * zF
        # FyF_slip = -self.mu * FzF * np.sign(alpha_F)
        # FyF = FyF_noslip if np.abs(zF) < np.tan(a_slF) else FyF_slip

        # FyR = self.l_r * r - V * np.sin(beta)
        # FxR = omega_R * self.R_e - V * np.cos(beta)

        return FyF, FxR, FyR

    def dynamics(self, Vx, Vy, r, omega_R, delta, torque, logger=None):
        """Compute the vehicle's dynamics."""
        beta = self.compute_beta(Vx, Vy)
        V = np.sqrt(Vx**2 + Vy**2)

        # Compute slip angles and slip ratio
        alpha_F = self.front_slip_angle(V, beta, r, delta)
        alpha_R = self.rear_slip_angle(V, beta, r)
        kR = self.rear_slip_ratio(omega_R, V, beta)

        # Tire forces
        FyF, FxR, FyR = self.tire_forces(alpha_F, alpha_R, kR, r, V, beta, omega_R)

        # Dynamics
        r_dot = (1 / self.Izz) * (self.l_f * FyF * np.cos(delta) - self.l_r * FyR)
        V_dot = (
            FxR * np.cos(beta) - FyF * np.sin(delta - beta) + FyR * np.sin(beta)
        ) / self.m
        beta_dot = (1 / (self.m * V)) * (
            FyF * np.cos(delta - beta) - FxR * np.sin(beta) + FyR * np.cos(beta)
        ) - r

        rolling_resistance = 0.02 * (self.m * 9.81 * self.R_e)
        w_dot = (torque - self.R_e * FxR - rolling_resistance) / 0.0375

        return V_dot, beta_dot, r_dot, w_dot
