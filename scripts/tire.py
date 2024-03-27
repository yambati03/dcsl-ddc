import numpy as np
import matplotlib.pyplot as plt
from functools import partial


def pacejka(D, C, B, alpha):
    return D * np.sin(C * np.arctan(B * alpha))


def get_tire_curve_f():
    D = 1.23306
    C = 1.0
    B = 4.0

    func = partial(pacejka, D, C, B)
    return func


def get_tire_curve_r():
    D = 0.79573
    C = 2.0
    B = 12.0

    func = partial(pacejka, D, C, B)
    return func


def plot_tire_curve(tire_curve):
    func_ = lambda a: tire_curve(a)

    _, ax = plt.subplots(figsize=(15, 10))
    x = np.linspace(-2, 2, 1000)
    y = func_(x)
    l, = ax.plot(x, y, 'b-')

    ax.title.set_text('Tire Curve')
    ax.set_xlabel('Slip Angle (rad)')
    ax.set_ylabel('Lateral Force (N)')
    ax.grid(True)
    ax.set_xticks(np.arange(-2, 2, 0.2))
    
    plt.show()