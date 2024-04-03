import numpy as np
import click
import math
from scipy.signal import savgol_filter
from load_log import load_ground_truth_from_bag
from tire import pacejka
from functools import partial
import itertools
from tqdm import tqdm


def get_tire_curve(D, C, B):
    func = partial(pacejka, D, C, B)
    return func


def step(state, control, Bf, Br, dt=0.01):
    x, vx_g, y, vy_g, h, r = state
    steering, throttle = control

    l_f = 0.1651  # m
    l_r = 0.1651  # m
    m = 3.17  # kg
    iz = 0.0398378  # 3 #98378  # kg m^2

    # Get velocity in local frame
    vx = vx_g * np.cos(h) + vy_g * np.sin(h)
    vy = -vx_g * np.sin(h) + vy_g * np.cos(h)

    beta = np.arctan(vy / vx)

    # Calculate slip angles
    slip_f = np.arctan(beta + (l_f * r) / vx) - steering
    slip_r = np.arctan(beta - (l_r * r) / vx)

    # Calculate lateral forces
    tire_curve_f = get_tire_curve(1, 1, Bf)
    tire_curve_r = get_tire_curve(1, 1, Br)

    Fyf = tire_curve_f(slip_f)
    Fyr = tire_curve_r(slip_r)

    d_vx = throttle - vx  # m/s^2
    d_vy = -vx * r + (Fyr + Fyf * np.cos(steering)) / m  # m/s^2
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


# Compute the average displacement error between actual and predicted trajectories
def ade(actual, predicted):
    return np.mean(np.linalg.norm(actual - predicted, axis=1))


def parse_log(bag):
    log = load_ground_truth_from_bag(bag)

    # Initialize state
    t = log[:, 0] - log[0, 0]
    t = np.linspace(0, t[-1], t.shape[0])
    x = log[:, 1]
    y = log[:, 2]
    h = savgol_filter(wrap_continuous(log[:, 6]), 51, 2)

    steering = log[:, 7]
    throttle = log[:, 8]

    vx = savgol_filter(np.gradient(x, t), 51, 2)
    vy = savgol_filter(np.gradient(y, t), 51, 2)
    r = np.hstack([0, np.diff(h)]) / (t[1] - t[0])

    return np.vstack((t, x, y, h, steering, throttle, vx, vy, r)).T


def search_best_params(log):
    lookahead_steps = 50

    t = log[:, 0]
    x = log[:, 1]
    y = log[:, 2]
    h = log[:, 3]
    steering = log[:, 4]
    throttle = log[:, 5]
    vx = log[:, 6]
    vy = log[:, 7]
    r = log[:, 8]

    best_error = float(math.inf)
    best_params = None

    for Bf, Br in tqdm(
        itertools.product(np.arange(1, 20, 0.1), np.arange(1, 20, 0.1)), total=36100
    ):
        error = 0.0

        for i in range(1, 50):  # log.shape[0] - lookahead_steps):
            state = (x[i], vx[i], y[i], vy[i], h[i], r[i])
            control = (steering[i], throttle[i])

            predicted_states = [state]

            for j in range(i + 1, i + lookahead_steps):
                state = step(state, control, Bf, Br, dt=(t[j] - t[j - 1]))
                predicted_states.append(state)
                control = (steering[j], throttle[j])

            predicted_states = np.array(predicted_states)

            predicted_future_traj = np.vstack(
                [predicted_states[:, 0], predicted_states[:, 2]]
            ).T

            actual_future_traj = np.vstack(
                [x[i : i + lookahead_steps], y[i : i + lookahead_steps]]
            ).T

            error += ade(actual_future_traj, predicted_future_traj)

        if error < best_error:
            best_error = error
            best_params = (Bf, Br)

    return best_params


@click.command()
@click.argument("bag", type=click.Path(exists=True))
def main(bag):
    log = parse_log(bag)

    best_params = search_best_params(log)
    print(f"Best params: {best_params}")


if __name__ == "__main__":
    main()
