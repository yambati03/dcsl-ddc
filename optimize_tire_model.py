import click
import yaml
import numpy as np

from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid

import rosbag2_py
from lib.bag import get_rosbag_options
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu
from ackermann_msgs.msg import AckermannDriveStamped

import matplotlib.pyplot as plt

with open("config/car_params.yaml", "r") as f:
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


def lerp(t1, t2, v2):
    lerped_vals = np.zeros_like(t1)

    for i, t in enumerate(t1):
        idx = np.searchsorted(t2, t, side="right") - 1

        if idx < 0:
            lerped_vals[i] = v2[0]
        elif idx >= len(t2) - 1:
            lerped_vals[i] = v2[-1]
        else:
            lerped_vals[i] = v2[idx] + (v2[idx + 1] - v2[idx]) * (t - t2[idx]) / (t2[idx + 1] - t2[idx])

    return lerped_vals


def plot_func(D, C, B):
    func = lambda a: D * np.sin(C * np.arctan(B * a))

    _, ax = plt.subplots(figsize=(15, 10))
    x = np.linspace(-5, 5, 1000)
    y = func(x)
    l, = ax.plot(x, y, 'b-')

    ax.title.set_text('Optimized Pacejka Model')
    ax.set_xlabel('Slip Angle (rad)')
    ax.set_ylabel('Lateral Force (N)')
    ax.grid(True)
    ax.set_xticks(np.arange(-5, 6, 1))
    
    plt.show()


@click.command()
@click.argument("data", type=click.Path(exists=True))
def optimize(data):
    reader = rosbag2_py.SequentialReader()

    storage_options, converter_options = get_rosbag_options(data, "sqlite3")
    reader.open(storage_options, converter_options)

    # Load params
    i_z = params["i_z"]
    l_f = params["l_f"]
    l_r = params["l_r"]
    wheelbase = params["wheelbase"]
    m = params["mass"]

    imu_topic = params["imu_topic"]
    steering_topic = params["steering_topic"]

    topic_types = reader.get_all_topics_and_types()
    type_map = {
        topic_types[i].name: topic_types[i].type for i in range(len(topic_types))
    }

    a_xs = []
    a_ys = []
    rs = []
    t_imu = []

    deltas = []
    t_delta = []

    print("Reading data...")

    while reader.has_next():
        (topic, data, _) = reader.read_next()

        if topic == imu_topic:
            msg_type = get_message(type_map[topic])
            imu_msg = deserialize_message(data, msg_type)

            assert isinstance(imu_msg, Imu)

            a_x = imu_msg.linear_acceleration.x
            a_y = imu_msg.linear_acceleration.y
            r = imu_msg.angular_velocity.z
            t = imu_msg.header.stamp.sec + imu_msg.header.stamp.nanosec * 1e-9

            a_xs.append(a_x)
            a_ys.append(a_y)
            rs.append(r)
            t_imu.append(t)

        if topic == steering_topic:
            msg_type = get_message(type_map[topic])
            steering_msg = deserialize_message(data, msg_type)

            assert isinstance(steering_msg, AckermannDriveStamped)

            t = steering_msg.header.stamp.sec + steering_msg.header.stamp.nanosec * 1e-9
            delta = steering_msg.drive.steering_angle
            
            deltas.append(delta)
            t_delta.append(t)

    # Convert the lists to numpy arrays
    a_xs = np.array(a_xs)
    a_ys = np.array(a_ys)
    rs = np.array(rs)

    rdots = np.gradient(rs, t_imu)
    v_xs = cumulative_trapezoid(a_xs, t_imu, initial=0)

    deltas = np.array(deltas)
    deltas = lerp(t_imu, t_delta, deltas)

    # Calculate sideslip angles
    betas = np.arctan((l_r / (l_f + l_r)) * np.tan(deltas))
    alphas = slip_angle(v_xs, rs, deltas, betas, l_f)

    # Calculate the lateral forces
    Fys = (i_z * rdots + l_r * m * a_ys) / (wheelbase * np.cos(deltas))

    # Initial guess taken from https://www.edy.es/dev/docs/pacejka-94-parameters-explained-a-comprehensive-guide/ for dry tarmac.
    x0 = [1.0, 1.9, 10.0]

    print("Optimizing...")

    result = minimize(
        lambda x: objective(x[0], x[1], x[2], alphas, Fys),
        x0
    )

    D, C, B = result.x

    print(f"D: {D}, C: {C}, B: {B}")

    plot_func(D, C, B)


if __name__ == "__main__":
    optimize()
