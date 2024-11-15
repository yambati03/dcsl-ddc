import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from dynamics import DriftDynamicsModel

model = DriftDynamicsModel(
    Izz=0.08502599670201208,
    m=4.202,
    l_f=0.1651,
    l_r=0.1651,
    Cf=79.0,
    Cr=70.0,
    mu=1.0,
    R_e=0.1,
)

def vehicle_dynamics(t, state, V, delta, omega_R):
    beta, r = state
    Vx = V * np.cos(beta)
    Vy = V * np.sin(beta)

    _, beta_dot, r_dot, w_dot = model.dynamics(Vx, Vy, r, omega_R, delta, 0)

    return [beta_dot, r_dot]

V = 2.0
delta = -0.34
omega_R = 5.0

ts = np.linspace(0, 2, 500)
beta_values = np.linspace(-0.5, 0.5, 20)
r_values = np.linspace(-5, 5, 20)

# Initialize mesh for vector field
B, R = np.meshgrid(beta_values, r_values)
dB, dR = np.zeros(B.shape), np.zeros(R.shape)

plt.figure(figsize=(10, 8))
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        state0 = [B[i, j], R[i, j]]
        
        if i % 2 == 0 and j % 2 == 0:
            sol = solve_ivp(vehicle_dynamics, [0, 2], state0, t_eval=ts, args=(V, delta, omega_R))
            plt.plot(sol.y[0], sol.y[1], "b-", alpha=0.5)

        d_state = vehicle_dynamics(0, [B[i, j], R[i, j]], V, delta, omega_R)
        dB[i, j], dR[i, j] = d_state

plt.quiver(B, R, dB, dR, color="red", angles="xy", alpha=0.6, width=0.002, scale=5000)
plt.xlabel("Sideslip angle (beta)")
plt.ylabel("Yaw rate (r)")
plt.title("Phase plot for sideslip angle (beta) and yaw rate (r)")
plt.grid()
plt.show()
