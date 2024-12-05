class DynamicBicycleModel(DynamicsModel):
    """
    This class implements a dynamic bicycle model for vehicle dynamics simulation.

    Args:
        Izz (float): Moment of inertia around z-axis
        m (float): Mass of vehicle
        l_f (float): Distance from CoG to front axle
        l_r (float): Distance from CoG to rear axle
        C_f (float): Front cornering stiffness
        C_r (float): Rear cornering stiffness
        mu (float): Friction coefficient
        r_e (float): Effective rear wheel radius
    """

    def __init__(
        self,
        Izz: float,
        m: float,
        l_f: float,
        l_r: float,
        C_f: float,
        C_r: float,
        mu: float,
        r_e: float,
        init_state: np.ndarray,
    ):
        super().__init__(init_state)

        self.Izz = Izz
        self.m = m
        self.l_f = l_f
        self.l_r = l_r
        self.C_f = C_f
        self.C_r = C_r
        self.mu = mu
        self.r_e = r_e

    def _step_kinematic(self, state: np.ndarray, control: np.ndarray, dt=0.01):
        x, y, vx, vy, h, r, beta = state
        steering, throttle = control

        beta = np.arctan(self.l_r / (self.l_r + self.l_f) * np.tan(steering))
        norm = lambda a, b: (a**2 + b**2) ** 0.5

        d_vx = throttle - vx
        vx = vx + d_vx * dt
        vy = norm(vx, vy) * np.sin(beta)
        r = vx / (self.l_f + self.l_r) * np.tan(steering)

        # Get velocity in global frame
        vx_g = vx * np.cos(h) - vy * np.sin(h)
        vy_g = vx * np.sin(h) + vy * np.cos(h)

        x += vx_g * dt
        y += vy_g * dt
        h += r * dt

        return np.array(
            [x, y, vx, vy, h % (2 * np.pi), r, np.arctan2(vy, vx) if vx != 0 else 0]
        )

    def _step_dynamic(
        self, state: np.ndarray, control: np.ndarray, dt=0.01
    ) -> np.ndarray:
        x, y, vx, vy, h, r, _ = state
        steering, throttle = control

        # Calculate slip angles
        slip_f = np.arctan((vy + self.l_f * r) / vx) - steering
        slip_r = np.arctan((vy - self.l_r * r) / vx)

        Fyf = -self.C_f * slip_f
        Fyr = -self.C_r * slip_r

        d_vx = throttle - vx  # m/s^2
        d_vy = -vx * r + ((Fyr + Fyf * np.cos(steering)) / self.m)  # m/s^2
        d_r = (self.l_f * Fyf * np.cos(steering) - self.l_r * Fyr) / self.Izz  # rad/s^2

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
