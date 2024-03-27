import cv2
import numpy as np


class Simulator:
    def __init__(self, origin=(512, 512)):
        self.img = np.zeros((1024, 1024, 3), np.uint8)
        self.img.fill(255)
        self.origin = origin
        self.w = 10  # m

        self.resolution = self.img.shape[0] / self.w

        theta = -np.pi / 2
        cos = np.cos(theta)
        sin = np.sin(theta)

        self.T = np.array(
            [[cos, sin, self.origin[0]], [sin, -cos, self.origin[1]], [0, 0, 1]]
        )
        self.T_rot = np.array([[cos, sin], [sin, -cos]])

    def get_img(self):
        return self.img

    def draw_frame(self):
        cv2.arrowedLine(
            self.img, self.origin, (self.origin[0], self.origin[1] - 40), (0, 0, 0), 1
        )
        cv2.arrowedLine(
            self.img, self.origin, (self.origin[0] - 40, self.origin[1]), (0, 0, 0), 1
        )

    def clear_img(self):
        self.img.fill(255)
        cv2.circle(self.img, self.origin, radius=4, color=(0, 0, 255), thickness=-1)
        self.draw_frame()

    def vicon_to_image(self, point):
        transformed_point = self.T @ np.array(
            [point[0] * self.resolution, point[1] * self.resolution, 1]
        )
        return transformed_point[:2]

    def vicon_to_image_rot(self, point):
        return self.T_rot @ np.array(
            [point[0] * self.resolution, point[1] * self.resolution]
        )

    def draw_car(self, x, y, theta, color=(255, 0, 0)):
        car = np.array([[-0.15, -0.1], [-0.15, 0.1], [0.15, 0.1], [0.15, -0.1]])
        front = np.array([0.15, 0])

        car = np.array([self.vicon_to_image_rot(p) for p in car])
        front = self.vicon_to_image_rot(front)

        front = front @ np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        car = car @ np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

        transformed_point = self.vicon_to_image([x, y])
        car += np.array([transformed_point[0], transformed_point[1]])
        front += np.array([transformed_point[0], transformed_point[1]])

        car = car.astype(np.int32)
        front = front.astype(np.int32)

        cv2.fillPoly(self.img, [car], color=color)
        cv2.circle(
            self.img, (front[0], front[1]), radius=4, color=(0, 0, 255), thickness=-1
        )

    def draw_polyline(self, points, color=(0, 0, 255)):
        points = np.array([self.vicon_to_image(p) for p in points]).astype(np.int32)
        cv2.polylines(self.img, [points], isClosed=False, color=color, thickness=2)

    def draw_text(self, text, x, y):
        cv2.putText(
            self.img,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    def show_raw_state(self, state, control):
        x, vx, y, vy, h, r = state
        cv2.rectangle(self.img, (780, 100), (900, 290), (0, 0, 255), 1)
        self.draw_text(f"x: {x:.2f}", 800, 120)
        self.draw_text(f"y: {y:.2f}", 800, 140)
        self.draw_text(f"vx: {vx:.2f}", 800, 160)
        self.draw_text(f"vy: {vy:.2f}", 800, 180)
        self.draw_text(f"h: {h:.2f}", 800, 200)
        self.draw_text(f"r: {r:.2f}", 800, 220)
        vx_l = vx * np.cos(h) + vy * np.sin(h)
        vy_l = -vx * np.sin(h) + vy * np.cos(h)
        self.draw_text(f"vx_l: {vx_l:.2f}", 800, 240)
        self.draw_text(f"vy_l: {vy_l:.2f}", 800, 260)
        self.draw_text(f"steer: {control[0]:.2f}", 800, 280)

    def draw_steering(self, steering):
        def map(val, in_l, in_h, out_low, out_high):
            # out of bound flag
            oob = False
            if val < in_l:
                val = in_l
                oob = True
            elif val > in_h:
                val = in_h
                oob = True
            return (val - in_l) / (in_h - in_l) * (out_high - out_low) + out_low, oob

        x1 = 100
        y1 = 40

        # Add steering bar
        steering, oob = map(steering, -0.35, 0.35, 0, 100)
        cv2.putText(
            self.img,
            f"Steering:",
            (20, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        ),
        cv2.rectangle(self.img, (x1, y1 + 25), (x1 + 100, y1 + 40), (0, 0, 255), 1)
        if oob:
            cv2.rectangle(
                self.img,
                (x1 + 50, y1 + 25),
                (x1 + int(steering), y1 + 40),
                (0, 0, 255),
                -1,
            )
        else:
            cv2.rectangle(
                self.img,
                (x1 + 50, y1 + 25),
                (x1 + int(steering), y1 + 40),
                (0, 255, 0),
                -1,
            )
