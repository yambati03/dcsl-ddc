import click
import numpy as np
import rosbag2_py
from common import get_rosbag_options


# Define the Pacejka tire model
def pacejka(D, C, B, alpha):
    return D * np.sin(C * np.arctan(B * alpha))


# Define the slip angle
def slip_angle(v_x, r, delta, beta, l_f):
    return np.arctan(beta + (l_f * r) / v_x) - delta


# We want to minimize the sum of the squared differences between
# the measured lateral force and the model's lateral force
def objective(D, C, B, alpha, Fy):
    return np.sum((Fy - pacejka(D, C, B, alpha)) ** 2)


@click.command()
@click.argument("data", type=click.Path(exists=True))
def optimize(data):
    reader = rosbag2_py.SequentialReader()

    storage_options, converter_options = get_rosbag_options(data, 'sqlite3')
    reader.open(storage_options, converter_options)
