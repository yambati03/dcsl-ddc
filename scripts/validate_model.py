import cv2
import math
import time
import click
import numpy as np
from scipy.signal import savgol_filter

import tire
from simulator import Simulator
from log import Log
from load_log import load_ground_truth_from_bag


def step(state, control, dt=0.01):
    x, vx_g, y, vy_g, h, r = state
    steering, throttle = control

    l_f = 0.1651 # m
    l_r = 0.1651 # m
    m = 3.17 # kg
    iz = 0.0398378 # kg m^2
 
    # Get velocity in local frame
    vx = vx_g * np.cos(h) + vy_g * np.sin(h)
    vy = -vx_g * np.sin(h) + vy_g * np.cos(h)

    beta = np.arctan((l_r / (l_f + l_r)) * np.tan(steering))

    # Calculate slip angles
    slip_f = np.arctan(beta + (l_f * r) / vx) - steering
    slip_r = np.arctan(beta - (l_r * r) / vx)

    # Calculate lateral forces
    tire_curve_f = tire.get_tire_curve_f()
    tire_curve_r = tire.get_tire_curve_r()

    Fyf = tire_curve_f(slip_f)
    Fyr = tire_curve_r(slip_r)
 
    d_vx = throttle # m/s^2
    d_vy = -vx * r + (Fyr + Fyf * np.cos(steering)) / m # m/s^2
    d_r = (l_f * Fyf * np.cos(steering) - l_r * Fyr) / iz # rad/s^2

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

    return (x, vx_g, y, vy_g, h, r)

@click.command()
@click.argument("bag", type=click.Path(exists=True))
@click.option("--plot_predicted", "-p", is_flag=True)
def main(bag, plot_predicted):
    sim = Simulator()
    log = Log(bag)

    log = load_ground_truth_from_bag(bag)
    lookahead_steps = 50
  
    # Initial state
    t = log[:, 0]
    x = log[:, 1]
    y = log[:, 2]
    h = log[:, 6]

    steering = log[:, 7]
    throttle = log[:, 8]

    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    r = np.gradient(h, t)

    # Smooth velocity
    vx = savgol_filter(vx, 51, 2)
    vy = savgol_filter(vy, 51, 2)
 
    for i in range(1, log.shape[0] - lookahead_steps):

        state = (x[i], vx[i], y[i], vy[i], h[i], r[i])
        control = (steering[i], throttle[i]) 

        sim.clear_img()

        if plot_predicted:
            predicted_states = [state]

            for j in range(i + 1, i + lookahead_steps):
                state = step(state, control, dt=(t[j] - t[j - 1]))
                predicted_states.append(state)
                control = (steering[j], throttle[j])

            predicted_states = np.array(predicted_states)

            predicted_future_traj = np.vstack([predicted_states[:, 0],predicted_states[:, 2]]).T
            sim.draw_polyline(predicted_future_traj * 100 + 512, color=(0, 255, 0))

        actual_future_traj = np.vstack([x[i:i + lookahead_steps],y[i:i + lookahead_steps]]).T
        sim.draw_polyline(actual_future_traj * 100 + 512)
        
        sim.draw_car(x[i] * 100 + 512, y[i] * 100 + 512, math.pi / 2 - h[i])

        cv2.imshow("Simulator", cv2.putText(sim.get_img(), f"Ground Truth: {bag}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA))
        cv2.waitKey(1)
        time.sleep(0.001)

if __name__ == "__main__":
    main()