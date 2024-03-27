import numpy as np
from datetime import datetime

import rosbag2_py
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu
from ackermann_msgs.msg import AckermannDriveStamped
from vicon_msgs.msg import ViconObject


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


def load_log_from_bag(filename, imu_topic="/imu/data", teleop_topic="/teleop", vicon_topic="/vicon", save=False):
    reader = rosbag2_py.SequentialReader()
    storage_options, converter_options = get_rosbag_options(filename, "sqlite3")
    reader.open(storage_options, converter_options)
    reader.set_filter(rosbag2_py.StorageFilter([imu_topic, teleop_topic, vicon_topic]))

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

    vicon = []

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

        if topic == vicon_topic:
            msg_type = get_message(type_map[topic])
            vicon_msg = deserialize_message(data, msg_type)

            assert isinstance(vicon_msg, ViconObject)

            t = vicon_msg.header.stamp.sec + vicon_msg.header.stamp.nanosec * 1e-9
            x = vicon_msg.position.x
            y = vicon_msg.position.y
            z = vicon_msg.position.z
            rx = vicon_msg.rotation.x
            ry = vicon_msg.rotation.y
            rz = vicon_msg.rotation.z

            vicon.append([t, x, y, z, rx, ry, rz])

    vicon = np.array(vicon)
    t = np.array(t_imu)
    t_delta = np.array(t_delta)
    a_xs = np.array(a_xs)
    a_ys = np.array(a_ys)
    rs = np.array(rs)
    deltas = lerp(t, t_delta, np.array(deltas))

    if save:
        save_file(np.vstack((t_imu, a_xs, a_ys, rs, deltas)))

    return t, a_xs, a_ys, rs, deltas #, vicon


def save_file(data, filename=None):
    if filename is None:
        filename = datetime.now().strftime("%Y_%m_%d-%p%I_%M_%S")
    np.save(f"{filename}.npy", data)


def load_ground_truth_from_bag(filename, vicon_topic="/vicon", teleop_topic="/ackermann_cmd"):
    reader = rosbag2_py.SequentialReader()
    storage_options, converter_options = get_rosbag_options(filename, "sqlite3")
    reader.open(storage_options, converter_options)
    reader.set_filter(rosbag2_py.StorageFilter([vicon_topic, teleop_topic]))

    topic_types = reader.get_all_topics_and_types()
    type_map = {
        topic_types[i].name: topic_types[i].type for i in range(len(topic_types))
    }

    vicon = []
    t_delta = []
    deltas = []
    throttles = []

    while reader.has_next():
        (topic, data, _) = reader.read_next()

        if topic == vicon_topic:
            msg_type = get_message(type_map[topic])
            vicon_msg = deserialize_message(data, msg_type)

            assert isinstance(vicon_msg, ViconObject)

            t = vicon_msg.header.stamp.sec + vicon_msg.header.stamp.nanosec * 1e-9
            x = vicon_msg.position.x
            y = vicon_msg.position.y
            z = vicon_msg.position.z
            rx = vicon_msg.rotation.x
            ry = vicon_msg.rotation.y
            rz = vicon_msg.rotation.z

            vicon.append([t, x, y, z, rx, ry, rz])

        if topic == teleop_topic:
            msg_type = get_message(type_map[topic])
            steering_msg = deserialize_message(data, msg_type)

            assert isinstance(steering_msg, AckermannDriveStamped)

            t = steering_msg.header.stamp.sec + steering_msg.header.stamp.nanosec * 1e-9
            delta = steering_msg.drive.steering_angle
            throttle = steering_msg.drive.speed
            
            deltas.append(delta)
            t_delta.append(t)
            throttles.append(throttle)

    vicon = np.array(vicon)
    t_delta = np.array(t_delta)
    deltas = np.array(deltas)
    throttles = np.array(throttles)

    deltas = lerp(vicon[:, 0], t_delta, deltas)
    throttles = lerp(vicon[:, 0], t_delta, throttles)

    return np.hstack((vicon, deltas[:, np.newaxis], throttles[:, np.newaxis]))


def load_log_from_npy(filename):
    log = np.load(filename)
    return log[0, :], log[1, :], log[2, :], log[3, :], log[4, :]

    