import click
import numpy as np
import rosbag2_py
from lib.bag import get_rosbag_options
import yaml
from scipy.optimize import minimize

from sensor_msgs.msg import Imu
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

with open("config/car_params.yaml", "r") as f:
    params = yaml.safe_load(f)


# Define the Pacejka tire model
def pacejka(D, C, B, alpha):
    return D * np.sin(C * np.arctan(B * alpha))


# Define the slip angle
def slip_angle(v_xs: np.ndarray, rs: np.ndarray, deltas: np.ndarray, beta, l_f):
    return np.arctan(beta + (l_f * rs) / v_xs) - deltas


# We want to minimize the sum of the squared differences between
# the measured lateral force and the model's lateral force
def objective(D, C, B, alphas, Fys: np.ndarray):
    return np.sum((Fys - pacejka(D, C, B, alphas)) ** 2)


@click.command()
@click.argument("data", type=click.Path(exists=True))
def optimize(data):
    reader = rosbag2_py.SequentialReader()

    storage_options, converter_options = get_rosbag_options(data, 'sqlite3')
    reader.open(storage_options, converter_options)

    # Load params
    i_z = params["i_z"]
    l_f = params["l_f"]
    l_r = params["l_r"]
    wheelbase = params["wheelbase"]
    m = params["mass"]

    imu_topic = params["imu_topic"]
    steering_topic = params["steering_topic"]

    # Initialize the parameters
    D = 1.0
    C = 1.0
    B = 1.0

    # Initialize the arrays to store the data
    topic_types = reader.get_all_topics_and_types()
    type_map = {
        topic_types[i].name: topic_types[i].type for i in range(len(topic_types))
    }

    a_xs = []
    a_ys = []
    rs = []
    deltas = []

    while reader.has_next():
        (topic, data, t) = reader.read_next()

        if topic == imu_topic:
            msg_type = get_message(type_map[topic])
            imu_msg = deserialize_message(data, msg_type)

            assert isinstance(imu_msg, Imu)

            a_x = imu_msg.linear_acceleration.x
            a_y = imu_msg.linear_acceleration.y
            r = imu_msg.angular_velocity.z

            a_xs.append(a_x)
            a_ys.append(a_y)
            rs.append(r)

        if topic == steering_topic:
            msg_type = get_message(type_map[topic])
            steering_msg = deserialize_message(data, msg_type)

            delta = steering_msg.drive.steering_angle
            deltas.append(delta)

    # Convert the lists to numpy arrays
    a_xs = np.array(a_xs)
    a_ys = np.array(a_ys)
    rs = np.array(rs)
    deltas = np.array(deltas)

    # Calculate sideslip angles
    betas = np.arctan((l_r / (l_f + l_r)) * np.tan(deltas))
    alphas = np.zeros(len(betas))

    # Calculate the lateral forces
    Fys = (i_z * rs + l_f * m * a_ys) / (wheelbase * np.cos(deltas))

    # Optimize the parameters
    result = minimize(
        lambda x: objective(x[0], x[1], x[2], alphas, Fys),
        [D, C, B],
        bounds=[(0, None), (0, None), (0, None)],
    )

    D, C, B = result.x

    print(f"D: {D}, C: {C}, B: {B}")


if __name__ == '__main__':
    optimize()
