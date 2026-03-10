import cv2
import math
import numpy as np


def calculate_distance(point1, point2):
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])


def smooth_position(prev_x, prev_y, new_x, new_y, alpha=0.20):
    smooth_x = int(prev_x + alpha * (new_x - prev_x))
    smooth_y = int(prev_y + alpha * (new_y - prev_y))
    return smooth_x, smooth_y


def point_inside_polygon(point, polygon):
    polygon_np = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(polygon_np, point, False) >= 0


def map_point_with_homography(point, matrix):
    src = np.array([[[point[0], point[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, matrix)
    mapped_x = int(dst[0][0][0])
    mapped_y = int(dst[0][0][1])
    return mapped_x, mapped_y