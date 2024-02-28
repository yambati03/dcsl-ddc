import cv2
import numpy as np

class Simulator():
    def __init__(self):
        self.img = np.zeros((1024, 1024, 3), np.uint8)
        self.img.fill(255)

    def get_img(self):
        return self.img
    
    def clear_img(self):
        self.img.fill(255)
    
    def draw_car(self, x, y, theta, color=(255, 0, 0)):
        car = np.array([[-12, -25], [-12, 25], [12, 25], [12, -25]])
        car = car @ np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        car += np.array([x, y])
        car = car.astype(np.int32)
        cv2.fillPoly(self.img, [car], color=color)

    def draw_polyline(self, points, color=(0, 0, 255)):
        points = points.astype(np.int32)
        cv2.polylines(self.img, [points], isClosed=False, color=color, thickness=2)

