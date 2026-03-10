import cv2


def draw_dwell_progress(frame, point, progress, radius=22, color=(0, 0, 255)):
    progress = max(0.0, min(1.0, progress))
    cv2.circle(frame, point, radius, (0, 255, 255), 2)

    if progress > 0:
        end_angle = int(360 * progress)
        cv2.ellipse(
            frame,
            point,
            (radius, radius),
            -90,
            0,
            end_angle,
            color,
            4
        )


def draw_mode_info(frame, mode, calibration_points, left_dwell_time, right_dwell_time,
                   drag_dwell_time, double_dwell_time, drawing_mode):
    cv2.putText(
        frame,
        f"Mode: {mode.upper()}",
        (330, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if mode == "interaction" else (0, 255, 255),
        2
    )

    draw_text = "DRAW MODE: ON" if drawing_mode else "DRAW MODE: OFF"
    draw_color = (0, 255, 0) if drawing_mode else (0, 0, 255)
    cv2.putText(
        frame,
        draw_text,
        (330, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        draw_color,
        2
    )

    cv2.putText(
        frame,
        "K: Calibration  S: Start Interaction  D: Draw Toggle  R: Reset  Q: Quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2
    )

    if mode == "idle":
        cv2.putText(
            frame,
            "Press K to enter calibration mode",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )
    elif mode == "calibration":
        if len(calibration_points) < 4:
            cv2.putText(
                frame,
                f"Click Point {len(calibration_points) + 1} (TL, TR, BR, BL)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )
        else:
            cv2.putText(
                frame,
                "4 points selected. Press S to start interaction",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
    else:
        if drawing_mode:
            cv2.putText(
                frame,
                "Drawing mode: index-only hold 0.8s to draw | index+middle = instant click",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                f"Left dwell: {left_dwell_time:.1f}s | Double dwell: {double_dwell_time:.1f}s | Right dwell: {right_dwell_time:.1f}s | Drag dwell: {drag_dwell_time:.1f}s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 0),
                2
            )