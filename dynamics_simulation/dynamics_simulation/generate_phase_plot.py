import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Define the dynamics model
def vehicle_dynamics(state, t, vx, steering):
    beta, r = state  # Unpack state: sideslip angle and yaw rate
    l_f, l_r = 0.1651, 0.1651  # Vehicle length (m)
    Cf, Cr = 79.0, 70.0  # Cornering stiffness (N/rad)
    m, iz = 4.202, 0.0398378  # Mass (kg) and moment of inertia (kg*m^2)

    # Slip angles
    slip_f = np.arctan((vx * np.sin(beta) + l_f * r) / vx) - steering
    slip_r = np.arctan((vx * np.sin(beta) - l_r * r) / vx)

    # Lateral forces
    Fyf = -Cf * slip_f
    Fyr = -Cr * slip_r

    # Dynamics equations
    d_beta = r - (Fyf * np.cos(steering) + Fyr) / (m * vx)
    d_r = (l_f * Fyf * np.cos(steering) - l_r * Fyr) / iz

    return [d_beta, d_r]


# Simulation parameters
vx = 1.2
steering = -0.34
ts = np.linspace(0, 2, 500)
beta_values = np.linspace(-0.5, 0.5, 20)
r_values = np.linspace(-2, 2, 20)

# Initialize mesh for vector field
B, R = np.meshgrid(beta_values, r_values)
dB, dR = np.zeros(B.shape), np.zeros(R.shape)

plt.figure(figsize=(10, 8))
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        state0 = [B[i, j], R[i, j]]
        traj = odeint(vehicle_dynamics, state0, ts, args=(vx, steering))
        plt.plot(traj[:, 0], traj[:, 1], "b-", alpha=0.3)

        d_state = vehicle_dynamics([B[i, j], R[i, j]], 0, vx, steering)
        dB[i, j], dR[i, j] = d_state

plt.quiver(B, R, dB, dR, color="red", angles="xy", alpha=0.7)


plt.xlabel("Sideslip angle (beta)")
plt.ylabel("Yaw rate (r)")
plt.title("Phase plot for sideslip angle (veta) and yaw rate (r)")
plt.grid()
plt.show()
