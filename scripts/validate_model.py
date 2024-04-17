import cv2
import time
import click
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

import tire
from simulator import Simulator
from log import Log
from load_log import load_ground_truth_from_bag


def pretty(d):
    print("{")
    for key, value in d.items():
        print(f"\t {str(key)}: {str(value)}")
    print("}")


def step_kinematic(state, control, dt=0.01):
    x, vx_g, y, vy_g, h, r = state
    steering, throttle = control

    lf = 0.1651  # m
    lr = 0.1651  # m

    vx = vx_g * np.cos(h) + vy_g * np.sin(h)
    vy = -vx_g * np.sin(h) + vy_g * np.cos(h)

    v = vx

    beta = np.arctan(np.tan(steering) * lf / (lf + lr))
    dXdt = v * np.cos(h + beta)
    dYdt = v * np.sin(h + beta)
    dvdt = throttle - v
    r = v / lr * np.sin(beta)

    x += dt * dXdt
    y += dt * dYdt
    v += dt * dvdt
    h += dt * r

    vx = v
    vy = 0

    # Get velocity in global frame
    vx_g = vx * np.cos(h) - vy * np.sin(h)
    vy_g = vx * np.sin(h) + vy * np.cos(h)

    return (x, vx_g, y, vy_g, h, r)


def step(state, control, dt=0.01):
    x, vx_g, y, vy_g, h, r = state
    steering, throttle = control

    l_f = 0.1651  # m
    l_r = 0.1651  # m
    m = 4.202  # kg
    iz = 0.08502599670201208  # 3 #98378  # kg m^2

    # Get velocity in local frame
    vx = vx_g * np.cos(h) + vy_g * np.sin(h)
    vy = -vx_g * np.sin(h) + vy_g * np.cos(h)

    beta = np.arctan(vy / vx)

    # Calculate slip angles
    slip_f = np.arctan(beta + (l_f * r) / vx) - steering
    slip_r = np.arctan(beta - (l_r * r) / vx)

    # Calculate lateral forces
    tire_curve_f = tire.get_tire_curve_f()
    tire_curve_r = tire.get_tire_curve_r()

    Fyf = tire_curve_f(slip_f)
    Fyr = tire_curve_r(slip_r)

    d_vx = throttle - vx  # m/s^2
    d_vy = vx * r - ((Fyr + Fyf * np.cos(steering)) / m)  # m/s^2
    d_r = (l_f * Fyf * np.cos(steering) - l_r * Fyr) / iz  # rad/s^2

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


# wrap to -pi,pi
def wrap_continuous(val):
    wrap = lambda x: np.mod(x + np.pi, 2 * np.pi) - np.pi
    dval = np.diff(val)
    dval = wrap(dval)
    retval = np.hstack([0, np.cumsum(dval)]) + val[0]
    return retval


@click.command()
@click.argument("bag", type=click.Path(exists=True))
@click.option("--plot_predicted", "-p", is_flag=True)
@click.option("--plot_state", "-s", is_flag=True)
def main(bag, plot_predicted, plot_state):
    sim = Simulator()

    log = load_ground_truth_from_bag(bag)
    lookahead_steps = 50

    # Initialize state
    t = log[:, 0] - log[0, 0]
    t = np.linspace(0, t[-1], t.shape[0])
    x = log[:, 1]
    y = log[:, 2]
    h = savgol_filter(wrap_continuous(log[:, 6]), 51, 2)

    steering = log[:, 7]
    throttle = log[:, 8]

    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    r = np.hstack([0, np.diff(h)]) / (t[1] - t[0])

    # Smooth velocity
    vx = savgol_filter(vx, 51, 2)
    vy = savgol_filter(vy, 51, 2)

    # Plot time versus h and r using ax
    if plot_state:
        _, ax = plt.subplots(2, 2, figsize=(15, 10))
        ax[0][0].plot(t, h, label="h")
        ax[0][0].set_title("Heading")
        ax[1][0].plot(t, r, label="r")
        ax[1][0].set_title("Yaw Rate")
        ax[0][1].plot(t, vx, label="vx")
        ax[0][1].set_title("Velocity (x)")
        ax[1][1].plot(t, vy, label="vy")
        ax[1][1].set_title("Velocity (y)")
        plt.show()

    for i in range(1, log.shape[0] - lookahead_steps):

        state = (x[i], vx[i], y[i], vy[i], h[i], r[i])
        control = (steering[i], throttle[i])

        sim.clear_img()

        sim.draw_steering(steering[i])
        sim.show_raw_state(state, control)

        if plot_predicted:
            predicted_states = [state]

            for j in range(i + 1, i + lookahead_steps):
                state = step(state, control, dt=(t[j] - t[j - 1]))
                predicted_states.append(state)
                control = (steering[j], throttle[j])

            predicted_states = np.array(predicted_states)

            predicted_future_traj = np.vstack(
                [predicted_states[:, 0], predicted_states[:, 2]]
            ).T

            sim.draw_polyline(predicted_future_traj, color=(0, 255, 0))

        actual_future_traj = np.vstack(
            [x[i : i + lookahead_steps], y[i : i + lookahead_steps]]
        ).T

        sim.draw_polyline(actual_future_traj)
        sim.draw_car(x[i], y[i], h[i])

        sim.draw_text(f"Ground Truth: {bag}", 20, 40)

        cv2.imshow("Simulator", sim.get_img())
        cv2.waitKey(1)
        time.sleep(0.001)


if __name__ == "__main__":
    main()
