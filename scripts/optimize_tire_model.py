import click
import yaml
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid

from load_log import load_log_from_bag


with open("../config/car_params.yaml", "r") as f:
    params = yaml.safe_load(f)


# Define the Pacejka tire model
def pacejka(D, C, B, alphas):
    return D * np.sin(C * np.arctan(B * alphas))


# Define the slip angle
def slip_angle(v_xs: np.ndarray, rs: np.ndarray, deltas: np.ndarray, betas: np.ndarray, l_f):
    return np.arctan(betas + (l_f * rs) / v_xs) - deltas


# We want to minimize the sum of the squared differences between
# the measured lateral force and the model's lateral force
def objective(D, C, B, alphas: np.ndarray, Fys: np.ndarray):
    return np.sum((Fys - pacejka(D, C, B, alphas)) ** 2)


def plot_func(D, C, B):
    func = lambda a: D * np.sin(C * np.arctan(B * a))

    _, ax = plt.subplots(figsize=(15, 10))
    x = np.linspace(-1, 1, 1000)
    y = func(x)
    l, = ax.plot(x, y, 'b-')

    ax.title.set_text('Optimized Pacejka Model')
    ax.set_xlabel('Slip Angle (rad)')
    ax.set_ylabel('Lateral Force (N)')
    ax.grid(True)
    ax.set_xticks(np.arange(-1, 1, 0.1))
    
    plt.show()


@click.command()
@click.argument("bags", type=click.Path(exists=True), nargs=-1)
def optimize(bags):
    i_z = params["i_z"]
    l_f = params["l_f"]
    l_r = params["l_r"]
    wheelbase = params["wheelbase"]
    m = params["mass"]

    imu_topic = params["imu_topic"]
    steering_topic = params["steering_topic"]

    alphas_ = np.array([])
    Fys_ = np.array([])

    for bag in bags:
        t, a_xs, a_ys, rs, deltas = load_log_from_bag(bag, imu_topic, steering_topic)

        rdots = np.gradient(rs, t)
        v_xs = cumulative_trapezoid(a_xs, t, initial=0)

        betas = np.arctan((l_r / (l_f + l_r)) * np.tan(deltas))

        alphas = slip_angle(v_xs, rs, deltas, betas, l_f)
        Fys = (i_z * rdots + l_r * m * a_ys) / (wheelbase * np.cos(deltas))

        non_zero_idx = np.nonzero(v_xs)
        alphas = alphas[non_zero_idx]
        Fys = Fys[non_zero_idx]

        alphas_ = np.concatenate((alphas_, alphas))
        Fys_ = np.concatenate((Fys_, Fys))


    # Initial guess taken from https://www.edy.es/dev/docs/pacejka-94-parameters-explained-a-comprehensive-guide/ for dry tarmac.
    x0 = [1.0, 1.9, 10.0]

    print("Optimizing...")

    # Minimize the sum of residuals. Bounds are taken from the same source as the initial guess.
    result = minimize(
        lambda x: objective(x[0], x[1], x[2], alphas_, Fys_),
        x0,
        bounds=[(0.1, 1.9), (1.0, 2.0), (4.0, 12.0)]
    )

    D, C, B = result.x

    print(f"D: {D}, C: {C}, B: {B}")

    plot_func(D, C, B)


if __name__ == "__main__":
    optimize()
