from dynamics_simulation.dynamics import DriftDynamicsModel
import numpy as np

if __name__ == "__main__":
    dynamics = DriftDynamicsModel(
        Izz=0.08502599670201208,
        m=4.202,
        l_f=0.1651,
        l_r=0.1651,
        Cf=79.0,
        Cr=70.0,
        mu=1.0,
        R_e=0.05,
    )

    dt = 0.01

    Vx = 2.0
    Vy = 0.0
    r = 0.0
    omega = Vx / dynamics.R_e
    beta = np.arctan2(Vy, Vx)

    state = np.array([Vx, Vy, r, omega, beta])
    
    for _ in range(10):
        print(f"\nInitial state: {state}")

        delta = 0.34
        torque = 0.2

        V_dot, beta_dot, r_dot, w_dot = dynamics.dynamics(Vx, Vy, r, omega, delta, torque)

        print(f"V_dot: {V_dot}, beta_dot: {beta_dot}, r_dot: {r_dot}, w_dot: {w_dot}")

        V = np.sqrt(Vx**2 + Vy**2)
        beta = np.arctan2(Vy, Vx)

        V += V_dot * dt
        beta += beta_dot * dt
        r += r_dot * dt
        omega += w_dot * dt

        # Get velocity components in local frame
        Vx = V * np.cos(beta)
        Vy = V * np.sin(beta) + r * dynamics.l_r

        state = np.array([Vx, Vy, r, omega, beta])
        print(f"Next state: {state}")

