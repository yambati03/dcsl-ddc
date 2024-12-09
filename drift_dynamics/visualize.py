import numpy as np
from models import DynamicBicycleModel
from controllers import MPC
from simulator import Simulator
import time
import cv2


# State: (x, y, h, V, beta, r, w, s)
init_state = np.array([0, 0, 0, 1.0, 0, 0, 0, 0])
init_state[6] = init_state[3] / 0.05

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
    J=0.0005,
    init_state=init_state,
)

mpc = MPC(10, model, time_range=20, debug=False)

path = None
path_type = "figure8"

if path_type == "circle":
    r = 2.0
    theta = np.linspace(0, 2 * np.pi, 100)
    path = np.column_stack(
        (r * np.cos(theta), r * np.sin(theta), (theta + np.pi / 2) % (2 * np.pi))
    )
elif path_type == "figure8":
    r = 2.0
    theta_left = np.linspace(-np.pi / 2, 3 * np.pi / 2, 100)
    theta_right = np.linspace(np.pi / 2, -3 * np.pi / 2, 100)

    left_circle = np.column_stack(
        (
            r * np.cos(theta_left),
            r * np.sin(theta_left) + r,
            (theta_left + np.pi / 2) % (2 * np.pi),
        )
    )
    right_circle = np.column_stack(
        (
            r * np.cos(theta_right),
            r * np.sin(theta_right) - r,
            (theta_right - np.pi / 2) % (2 * np.pi),
        )
    )
    path = np.vstack((left_circle[:-1, :], right_circle))

print(f"Running MPC for {mpc.simSteps} steps...")
start_time = time.time()
mpc.run(path)
print(f"Ran MPC in {time.time() - start_time} seconds.")

sim = Simulator()
paused = True
debug = False
state = init_state

for i in range(mpc.simSteps):
    sim.clear_img()

    control = mpc.controls_history[i]

    curr_path_state = mpc.path.get_state_by_s(state[7])
    state = model.step_and_return(state, control, curr_path_state[2], dt=mpc.dt)

    if debug:
        print(f"[{i}] Applying control: {control}")
        print(f"[{i}] New state: {state}")

    sim.draw_steering(control[0])
    sim.draw_car(state[0], state[1], state[2])
    sim.draw_polyline(path, is_last_heading=True)

    # Draw velocity vector
    sim.draw_vec(
        state[0:2],
        np.array([state[3] * np.cos(state[4]), state[3] * np.sin(state[4])]),
        state[2],
    )

    x, y, h, V, beta, r, w, path_dist = state
    delta, torque = control

    desired_pos = mpc.path.get_state_by_s(path_dist + mpc.lookahead_distance)
    sim.draw_point(desired_pos[:2], color=(0, 255, 0))

    # Compute slip angles and slip ratio
    alpha_F = model.front_slip_angle(V, beta, r, delta)
    alpha_R = model.rear_slip_angle(V, beta, r)
    s = model.rear_slip_ratio(w, V, beta)

    # Tire forces
    FyF, FxR, FyR = model.tire_forces(alpha_F, alpha_R, s)

    sim.draw_vec(state[0:2], np.array([0, FyF * 0.5]), state[2], color=(0, 255, 0))
    sim.draw_vec(state[0:2], np.array([0, FyR * 0.5]), state[2], color=(0, 0, 255))
    sim.draw_vec(state[0:2], np.array([FxR * 0.5, 0]), state[2], color=(255, 0, 0))
    sim.draw_vec(desired_pos[:2], np.array([1, 0]), desired_pos[2], color=(0, 255, 0))

    # Dynamics
    r_dot = model.compute_r_dot(FyF, FyR, delta)
    V_dot = model.compute_V_dot(FxR, FyF, FyR, beta, delta)
    beta_dot = model.compute_beta_dot(FyF, FxR, FyR, beta, r, V, delta)
    w_dot = model.compute_w_dot(torque, FxR)

    sim.show_dict(
        {
            "x": state[0],
            "y": state[1],
            "h": state[2],
            "V": state[3],
            "beta": state[4],
            "r": state[5],
            "w": state[6],
            "V_dot": V_dot,
            "beta_dot": beta_dot,
            "r_dot": r_dot,
            "w_dot": w_dot,
            "FxR": FxR,
            "FyF": FyF,
            "FyR": FyR,
            "alpha_F": alpha_F,
            "alpha_R": alpha_R,
            "path_dist": path_dist,
            "steering": control[0],
            "throttle": control[1],
        }
    )

    cv2.imshow("Simulator", sim.get_img())
    key = cv2.waitKey(1) & 0xFF
    if key == ord("p"):
        paused = not paused

    while paused:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("p"):
            paused = not paused
