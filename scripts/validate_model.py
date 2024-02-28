import numpy as np
from tire_curve import get_tire_curve
from simulator import Simulator
import cv2
from load_log import load_ground_truth_from_bag
import math
import time
import click

# TODO: Simulate predicted trajectory using the model versus ground truth

def step(state, control, dt=0.01):
    x, vx_car, y, vy_car, h, r = tuple(state)
    steering, throttle = tuple(control)

    l_f = 0.1651
    l_r = 0.1651
    m = 3.17
    iz = 0.0398378

    vx = vx_car * np.cos(h) + vy_car * np.sin(h)
    vy = -vx_car * np.sin(h) + vy_car * np.cos(h)

    beta = np.arctan((l_r / (l_f + l_r)) * np.tan(steering))

    slip_f = np.arctan(beta + (l_f * r) / vx) - steering
    slip_r = np.arctan(beta - (l_r * r) / vx)

    tire_curve = get_tire_curve()
    Fyf = tire_curve(slip_f)
    Fyr = tire_curve(slip_r)

    d_vx = 1
    d_vy = -vx * r + (Fyr + Fyf * np.cos(steering)) / m
    d_r = (l_f * Fyf * np.cos(steering) - l_r * Fyr) / iz

    # Update state
    vx = vx * d_vx * dt
    vy = vy * d_vy * dt
    r += r * d_r * dt

    # Get velocity in car frame
    vx_car = vx * np.cos(h) - vy * np.sin(h)
    vy_car = vx * np.sin(h) + vy * np.cos(h)

    # Update state
    x += vx_car * dt
    y += vy_car * dt
    h += r * dt

    return (x, vx_car, y, vy_car, h, r)

@click.command()
@click.argument("bag", type=click.Path(exists=True))
def main(bag):
    sim = Simulator()
    
    log = load_ground_truth_from_bag(bag)
    lookahead_steps = 50
  
    # Initial state
    t = log[:, 0]
    x = (log[:, 1] * 100) + 512
    y = (log[:, 2] * 100) + 512
    z = log[:, 3]
    rx = log[:, 4]
    ry = log[:, 5]
    rz = log[:, 6]
 
    for i in range(1, log.shape[0] - lookahead_steps):
        actual_future_traj = np.vstack([x[i:i + lookahead_steps],y[i:i + lookahead_steps]]).T

        sim.clear_img()

        sim.draw_polyline(actual_future_traj)
        sim.draw_car(x[i], y[i], math.pi / 2 - rz[i])



        cv2.imshow("Simulator", cv2.putText(sim.get_img(), f"Ground Truth: {bag}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA))
        cv2.waitKey(1)
        time.sleep(0.001)

if __name__ == "__main__":
    main()