import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from models import DynamicBicycleModel

init_state = np.array([0, -1.5, 1.04, 1.0, -0.5, 0, 0])
init_state[6] = init_state[3] / 0.05

model = DynamicBicycleModel(
    Izz=0.08502599670201208,
    m=4.202,
    Lf=0.1651,
    Lr=0.1651,
    Cf=20.5,
    Cx=46.5,
    Cy=53.0,
    mu=1.0,
    R=0.05,
    J=0.005,
    init_state=init_state,
)


# State: (x, y, h, V, beta, r, w)
def vehicle_dynamics(t, state, V, delta, omega_R):
    beta, r = state
    state = np.array([0, 0, 0, V, beta, r, 0])
    control = np.array([delta, 0])

    _, beta_dot, r_dot, _ = model.compute_dynamics(state, control)

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
            sol = solve_ivp(
                vehicle_dynamics, [0, 2], state0, t_eval=ts, args=(V, delta, omega_R)
            )
            plt.plot(sol.y[0], sol.y[1], "b-", alpha=0.5)

        d_state = vehicle_dynamics(0, [B[i, j], R[i, j]], V, delta, omega_R)
        dB[i, j], dR[i, j] = d_state

plt.quiver(B, R, dB, dR, color="red", angles="xy", alpha=0.6, width=0.002, scale=5000)
plt.xlabel("Sideslip angle (beta)")
plt.ylabel("Yaw rate (r)")
plt.title("Phase plot for sideslip angle (beta) and yaw rate (r)")
plt.grid()
plt.show()
