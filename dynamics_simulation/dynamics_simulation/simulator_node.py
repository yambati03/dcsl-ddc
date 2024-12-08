import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from dynamics_simulation.simulator import Simulator
from dynamics_simulation.dynamics import DynamicBicycleModel
import cv2
import numpy as np


class SimulatorNode(Node):
    def __init__(self):
        super().__init__("simulator_node")
        self.get_logger().info("Simulator node started...")

        self.ackermann_sub = self.create_subscription(
            AckermannDriveStamped, "/ackermann_cmd", self.ackermann_callback, 10
        )

        self.sim = Simulator()
        self.control = np.zeros(2)

        self.update_rate_ = 0.01  # 100 Hz
        self.timer = self.create_timer(self.update_rate_, self.update_sim)

        self.model = DynamicBicycleModel(
            Izz=0.08502599670201208,
            m=4.202,
            Lf=0.1651,
            Lr=0.1651,
            Cf=16.86,
            Cx=14.05,
            Cy=15.46,
            mu=1.0,
            R=0.05,
            J=0.0005,
            init_state=np.zeros(7),
        )

    def ackermann_callback(self, msg):
        self.control[0] = msg.drive.steering_angle
        self.control[1] = msg.drive.speed

    def update_sim(self):
        self.sim.clear_img()
        self.model.step(self.control, dt=self.update_rate_)

        self.sim.draw_steering(self.control[0])
        self.sim.show_dict(
            {
                "x": self.model.state[0],
                "y": self.model.state[1],
                "h": self.model.state[4],
                "V": self.model.state[3],
                "beta": self.model.state[4],
                "r": self.model.state[5],
                "w": self.model.state[6],
                "steering": self.control[0],
                "torque": self.control[1],
            }
        )
        self.sim.draw_car(self.model.state[0], self.model.state[1], self.model.state[2])

        cv2.imshow("Simulator", self.sim.get_img())
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    sim_node = SimulatorNode()

    rclpy.spin(sim_node)

    sim_node.destroy_node()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
