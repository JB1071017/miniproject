import numpy as np
import cv2

class ScreenCalibrator:
    def __init__(self):
        self.points = []
        self.done = False
        self.matrix = None
        self.screen = [(0,0), (1920,0), (1920,1080), (0,1080)]

    def add_point(self, x, y):
        if len(self.points) < 4:
            self.points.append((x, y))
            print("Calibration point:", len(self.points))

        if len(self.points) == 4:
            self.compute()

    def compute(self):
        src = np.array(self.points, dtype="float32")
        dst = np.array(self.screen, dtype="float32")
        self.matrix = cv2.getPerspectiveTransform(src, dst)
        self.done = True
        print("Calibration complete")

    def map_to_screen(self, x, y):
        pt = np.array([[[x, y]]], dtype="float32")
        mapped = cv2.perspectiveTransform(pt, self.matrix)
        return int(mapped[0][0][0]), int(mapped[0][0][1])
