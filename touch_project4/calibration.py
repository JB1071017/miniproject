import cv2


def mouse_click(event, x, y, flags, param):
    state = param

    if state.mode != "calibration":
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(state.calibration_points) < 4:
            state.calibration_points.append((x, y))
            print(f"Point {len(state.calibration_points)} selected: ({x}, {y})")


def draw_calibration_overlay(frame, points):
    for i, point in enumerate(points):
        cv2.circle(frame, point, 8, (0, 255, 0), -1)
        cv2.putText(
            frame,
            str(i + 1),
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], (0, 255, 255), 2)

    if len(points) == 4:
        cv2.line(frame, points[3], points[0], (0, 255, 255), 2)