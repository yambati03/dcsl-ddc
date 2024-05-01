from simulator import Simulator
import cv2
import numpy as np

sim = Simulator()

sim.clear_img()

sim.draw_car(1, 1, 0)
cv2.imshow("Simulator", sim.get_img())
cv2.waitKey(0)
