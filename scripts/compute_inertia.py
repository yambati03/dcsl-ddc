import numpy as np
from load_log import load_vicon_from_bag
import click
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


@click.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--plot", "-p", is_flag=True)
def analyze_bag(filepath, plot):
    # Compute the inertia of the robot
    vicon = load_vicon_from_bag(filepath)

    t = vicon[:, 0] - vicon[0, 0]
    # t = np.linspace(0, t[-1], t.shape[0])
    h = vicon[:, 6]

    mask = np.where(t > 33)
    t = t[mask]
    h = h[mask]

    # Find peaks
    peaks, _ = find_peaks(h, width=4)

    print(f"Time between oscillations: {np.diff(t[peaks])}")

    if plot:
        plt.plot(t, h)
        plt.scatter(t[peaks], h[peaks], color="r")
        for t_peak, h_peak in zip(t[peaks], h[peaks]):
            plt.annotate(np.round(t_peak, 3), xy=(t_peak, h_peak))
        plt.title("Heading")
        plt.xlabel("Time (s)")
        plt.ylabel("Heading (rad)")
        plt.show()


def compute_car_inertia():
    m_car = 4.202  # kg
    m_pendulum = 1.425  # kg, includes tray and ropes
    m_eq = m_car + m_pendulum  # kg
    R = 0.55 / 2  # m
    g = 9.81  # m/s^2
    h = 0.78  # m

    t_period_no_car = (67.578 - 44.66) / 20  # s
    t_period_with_car = (62.412 - 42.819) / 20  # s

    print(f"Period without car: {t_period_no_car} s")
    print(f"Period with car: {t_period_with_car} s")

    i_eq = (m_eq * g * R * R * t_period_with_car * t_period_with_car) / (
        4 * np.pi * np.pi * h
    )
    i_p = (m_pendulum * g * R * R * t_period_no_car * t_period_no_car) / (
        4 * np.pi * np.pi * h
    )

    print(f"Calculated inertia of the car: {i_eq - i_p} kg m^2")
    return i_eq - i_p


if __name__ == "__main__":
    analyze_bag()
