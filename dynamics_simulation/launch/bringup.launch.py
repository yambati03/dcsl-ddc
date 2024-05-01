from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    joy_teleop_config = os.path.join(
        get_package_share_directory("dynamics_simulation"), "config", "joy_teleop.yaml"
    )

    joy_la = DeclareLaunchArgument(
        "joy_config",
        default_value=joy_teleop_config,
        description="Descriptions for joy and joy_teleop configs",
    )

    ld = LaunchDescription([joy_la])

    joy_node = Node(
        package="joy",
        executable="joy_node",
        name="joy",
        parameters=[LaunchConfiguration("joy_config")],
    )
    joy_teleop_node = Node(
        package="joy_teleop",
        executable="joy_teleop",
        name="joy_teleop",
        parameters=[LaunchConfiguration("joy_config")],
    )
    sim_node = Node(
        package="dynamics_simulation",
        executable="simulator_node",
        name="simulator_node",
    )

    # finalize
    ld.add_action(joy_node)
    ld.add_action(joy_teleop_node)
    ld.add_action(sim_node)

    return ld
