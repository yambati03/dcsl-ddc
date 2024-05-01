import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from dataclasses import dataclass
from dynamics_simulation.simulator import Simulator
import cv2
import numpy as np


@dataclass
class State:
    x: float = 0.0
    vx_g: float = 0.0
    y: float = 0.0
    vy_g: float = 0.0
    h: float = 0.0
    r: float = 0.0

    def to_tuple(self):
        return (self.x, self.vx_g, self.y, self.vy_g, self.h, self.r)


class SimulatorNode(Node):
    def __init__(self):
        super().__init__("simulator_node")
        self.get_logger().info("Simulator node started...")

        self.ackermann_sub = self.create_subscription(
            AckermannDriveStamped, "/ackermann_cmd", self.ackermann_callback, 10
        )

        self.sim = Simulator()
        self.control = (0.0, 0.0)
        self.state = State()

        self.update_rate_ = 0.01  # 100 Hz
        self.timer = self.create_timer(self.update_rate_, self.update_sim)

    def ackermann_callback(self, msg):
        self.control = (msg.drive.steering_angle, msg.drive.speed)

    def update_sim(self):
        self.sim.clear_img()

        self.step(dt=self.update_rate_)

        self.sim.draw_steering(self.control[0])
        self.sim.show_raw_state(self.state.to_tuple(), self.control)
        self.sim.draw_car(self.state.x, self.state.y, self.state.h)

        cv2.imshow("Simulator", self.sim.get_img())
        cv2.waitKey(1)

    def step_kinematic(self, dt=0.01):
        x, vx_g, y, vy_g, h, r = self.state.to_tuple()
        steering, throttle = self.control

        lf = 0.1651  # m
        lr = 0.1651  # m

        vx = vx_g * np.cos(h) + vy_g * np.sin(h)
        vy = -vx_g * np.sin(h) + vy_g * np.cos(h)

        beta = np.arctan(lr / (lr + lf) * np.tan(steering))
        norm = lambda a, b: (a**2 + b**2) ** 0.5

        d_vx = throttle - vx
        vx = vx + d_vx * dt
        vy = norm(vx, vy) * np.sin(beta)
        r = vx / (lf + lr) * np.tan(steering)

        # Get velocity in global frame
        vx_g = vx * np.cos(h) - vy * np.sin(h)
        vy_g = vx * np.sin(h) + vy * np.cos(h)

        x += vx_g * dt
        y += vy_g * dt
        h += r * dt

        self.state = State(x, vx_g, y, vy_g, h, r)

    def step(self, dt=0.01):
        x, vx_g, y, vy_g, h, r = self.state.to_tuple()
        steering, throttle = self.control

        l_f = 0.1651  # m
        l_r = 0.1651  # m
        m = 4.202  # kg
        iz = 0.08502599670201208  # 3 #98378  # kg m^2

        # Get velocity in local frame (x is forward, y is left -- this follows the right hand rule)
        vx = vx_g * np.cos(h) + vy_g * np.sin(h)
        vy = -vx_g * np.sin(h) + vy_g * np.cos(h)

        if vx < 0.05:
            self.step_kinematic(dt)
            return

        Cf = 79.0
        Cr = 70.0

        # Calculate slip angles
        slip_f = np.arctan((vy + l_f * r) / vx) - steering
        slip_r = np.arctan((vy - l_r * r) / vx)

        Fyf = -Cf * slip_f
        Fyr = -Cr * slip_r

        d_vx = throttle - vx  # m/s^2
        d_vy = -vx * r + ((Fyr + Fyf * np.cos(steering)) / m)  # m/s^2
        d_r = (l_f * Fyf * np.cos(steering) - l_r * Fyr) / iz  # rad/s^2

        vx += d_vx * dt
        vy += d_vy * dt
        r += d_r * dt

        # Get velocity in global frame
        vx_g = vx * np.cos(h) - vy * np.sin(h)
        vy_g = vx * np.sin(h) + vy * np.cos(h)

        # Update state
        x += vx_g * dt
        y += vy_g * dt
        h += r * dt

        self.state = State(x, vx_g, y, vy_g, h, r)


def main(args=None):
    rclpy.init(args=args)
    sim_node = SimulatorNode()

    rclpy.spin(sim_node)

    sim_node.destroy_node()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
