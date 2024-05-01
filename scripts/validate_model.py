import cv2
import time
import click
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import tire
from simulator import Simulator
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

    # Get velocity in local frame (x is forward, y is left -- this follows the right hand rule)
    vx = vx_g * np.cos(h) + vy_g * np.sin(h)
    vy = -vx_g * np.sin(h) + vy_g * np.cos(h)

    Cf = 79.0
    Cr = 70.0

    # Calculate slip angles
    slip_f = np.arctan((vy + l_f * r) / vx) - steering
    slip_r = np.arctan((vy - l_r * r) / vx)

    Fyf = -Cf * slip_f
    Fyr = -Cr * slip_r

    d_vx = throttle - vx  # m/s^2
    d_vy = -vx * r + ((Fyr + Fyf * np.cos(steering)) / m)  # m/s^2
    d_r = (l_f * Fyf * np.cos(steering) - l_r * Fyr) / iz  # rad/s^2

    # pretty(
    #     {
    #         "d_vx": d_vx,
    #         "d_vy": d_vy,
    #         "d_r": d_r,
    #         "vx": vx,
    #         "vy": vy,
    #         "r": r,
    #         "steering": steering,
    #         "Fyf": Fyf,
    #         "Fyr": Fyr,
    #         "slip_f": slip_f,
    #         "slip_r": slip_r,
    #         "d_vy (c1)": -vx * r,
    #         "d_vy (c2)": ((Fyr + Fyf * np.cos(steering)) / m),
    #     }
    # )

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


def get_local_velocity(vx_g, vy_g, h):
    vx = vx_g * np.cos(h) + vy_g * np.sin(h)
    vy = -vx_g * np.sin(h) + vy_g * np.cos(h)
    return vx, vy


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

    print(t[1] - t[0])

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

    vx_l, vy_l = get_local_velocity(vx, vy, h)

    max_x = -1000
    min_x = 1000

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
        print(i)
        state = (x[i], vx[i], y[i], vy[i], h[i], r[i])
        control = (steering[i], throttle[i])

        sim.clear_img()

        sim.draw_steering(steering[i])
        sim.show_raw_state(state, control)

        beta = np.arctan(vy_l[i] / vx_l[i])
        speed = np.sqrt(vx_l[i] * vx_l[i] + vy_l[i] * vy_l[i])

        if abs(beta) > 0.24 and speed > 0.05:
            sim.draw_text(f"SLIPPING!", 20, 110, color=(0, 0, 255))
        else:
            sim.draw_text(f"NOT SLIPPING", 20, 110, color=(0, 255, 0))

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
            # sim.draw_car(
            #     predicted_states[-1][0],
            #     predicted_states[-1][2],
            #     predicted_states[-1][4],
            # )

        actual_future_traj = np.vstack(
            [x[i : i + lookahead_steps], y[i : i + lookahead_steps]]
        ).T

        sim.draw_polyline(actual_future_traj)
        sim.draw_car(x[i], y[i], h[i])
        # sim.draw_car(
        #     x[i + lookahead_steps], y[i + lookahead_steps], h[i + lookahead_steps]
        # )

        sim.draw_text(f"Ground Truth: {bag}", 20, 40)

        cv2.imshow("Simulator", sim.get_img())

        if cv2.waitKey(1) == 27:
            cv2.waitKey(0)

        time.sleep(0.001)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
