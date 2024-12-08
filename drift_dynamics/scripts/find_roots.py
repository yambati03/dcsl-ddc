import numpy as np
from scipy.optimize import fsolve
from models import DynamicBicycleModel

model = DynamicBicycleModel(
    Izz=0.08502599670201208,
    m=4.202,
    Lf=0.1651,
    Lr=0.1651,
    Cf=16.86,
    Cx=14.05,
    Cy=15.46,
    mu=1.0,
    R=0.05,
    J=0.005,
    init_state=np.zeros(7),
)
initial_state = np.array([3.6, -0.56, 0.5, 40])
control_input = np.array([np.pi / 6, 10.0])


def func(state):
    state_input = np.concatenate(([0, 0, 0], state))
    V_dot, beta_dot, r_dot, w_dot = model.compute_dynamics(
        state_input, np.array([np.pi / 6, 10.0])
    )
    return [V_dot, beta_dot, r_dot, w_dot]


# Solve for steady-state conditions
roots = fsolve(func, initial_state)
print("Steady-state solution:", roots)
if np.allclose(func(roots), np.zeros(4), atol=1e-3):
    print("Solution is correct.")
else:
    print(f"Solution is incorrect. Error: {func(roots)}")
