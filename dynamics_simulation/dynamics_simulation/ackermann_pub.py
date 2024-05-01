import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped


class AckermannPub(Node):
    def __init__(self):
        super().__init__("ackermann_pub")
        self.get_logger().info("Ackermann publisher node started...")

        self.ackermann_pub = self.create_publisher(
            AckermannDriveStamped, "/ackermann_cmd", 10
        )

        self.pub_rate_ = 0.02  # 50 Hz
        self.timer = self.create_timer(self.pub_rate_, self.pub)

    def pub(self):
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = -0.18
        msg.drive.speed = 1.83

        self.ackermann_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    pub_node = AckermannPub()

    rclpy.spin(pub_node)

    pub_node.destroy_node()


if __name__ == "__main__":
    main()
