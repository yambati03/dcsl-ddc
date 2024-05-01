import numpy as np


def main():
    l_f = 0.1651  # m
    l_r = 0.1651  # m

    m = 4.202  # kg
    steering = -0.18  # rad
    v_x = 1.83  # m/s

    slip_f = 0.001  # rad
    slip_r = 0.1  # rad

    r = -1.7  # rad/s

    B_f = (l_r * m * v_x * r) / ((l_f + l_r) * np.cos(steering) * slip_f)
    B_r = (v_x * r * m + B_f * slip_f * np.cos(steering)) / slip_r

    print(f"B_f: {B_f}")
    print(f"B_r: {B_r}")

    return B_f, B_r


if __name__ == "__main__":
    main()
