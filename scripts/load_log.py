import numpy as np
from datetime import datetime

import rosbag2_py
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu
from ackermann_msgs.msg import AckermannDriveStamped


def get_rosbag_options(path, storage_id, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(
        uri=path, storage_id=storage_id)

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options


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


def load_log_from_bag(filename, imu_topic="/imu/data", teleop_topic="/teleop", save=False):
    reader = rosbag2_py.SequentialReader()
    storage_options, converter_options = get_rosbag_options(filename, "sqlite3")
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {
        topic_types[i].name: topic_types[i].type for i in range(len(topic_types))
    }

    t_imu = []
    a_xs = []
    a_ys = []
    rs = []
    
    t_delta = []
    deltas = []

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

        if topic == teleop_topic:
            msg_type = get_message(type_map[topic])
            steering_msg = deserialize_message(data, msg_type)

            assert isinstance(steering_msg, AckermannDriveStamped)

            t = steering_msg.header.stamp.sec + steering_msg.header.stamp.nanosec * 1e-9
            delta = steering_msg.drive.steering_angle
            
            deltas.append(delta)
            t_delta.append(t)

    t = np.array(t_imu)
    t_delta = np.array(t_delta)
    a_xs = np.array(a_xs)
    a_ys = np.array(a_ys)
    rs = np.array(rs)
    deltas = lerp(t, t_delta, np.array(deltas))

    if save:
        formatted_date = datetime.now().strftime("%Y_%m_%d-%p%I_%M_%S")
        np.save(f"{formatted_date}.npy", np.vstack((t_imu, a_xs, a_ys, rs, deltas)))

    return t, a_xs, a_ys, rs, deltas
    

def load_log_from_npy(filename):
    log = np.load(filename)
    return log[0, :], log[1, :], log[2, :], log[3, :], log[4, :]

    