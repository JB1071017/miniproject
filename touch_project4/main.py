import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

from config import (
    LEFT_DWELL_TIME,
    DOUBLE_DWELL_TIME,
    RIGHT_DWELL_TIME,
    DRAG_DWELL_TIME,
    MOVEMENT_TOLERANCE,
    DRAWING_DWELL_TIME,
    DRAWING_MOVEMENT_TOLERANCE,
    SCROLL_THRESHOLD,
    SCROLL_COOLDOWN,
    SCROLL_AMOUNT,
    NORMAL_SMOOTH_ALPHA,
    DRAWING_SMOOTH_ALPHA,
    INTERPOLATION_STEPS,
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    WINDOW_NAME,
)
from state import AppState
from calibration import mouse_click, draw_calibration_overlay
from hand_tracking import detect_hands, draw_hand_info
from drawing_mode import draw_dwell_progress, draw_mode_info
from utils import calculate_distance, smooth_position, point_inside_polygon, map_point_with_homography
from actions import move_cursor_interpolated

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


def reset_interaction_state(state):
    state.dwell_anchor = None
    state.dwell_start_time = None
    state.click_triggered = False
    state.drag_click_triggered = False
    state.active_dwell_type = None
    state.scroll_anchor_y = None


def reset_drawing_state(state):
    state.drawing_anchor = None
    state.drawing_start_time = None
    state.drawing_click_ready = True


def release_all_mouse_states(state):
    if state.drag_active:
        pyautogui.mouseUp(button='left')
        state.drag_active = False

    if state.drawing_mouse_down:
        pyautogui.mouseUp(button='left')
        state.drawing_mouse_down = False


def main():
    state = AppState()

    hands_module = mp.solutions.hands
    drawing_utils = mp.solutions.drawing_utils

    hand_detector = hands_module.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_click, state)

    print("\nStage 9 - Wall Control + Drawing Mode")
    print("-------------------------------------")
    print("K -> Enter calibration mode")
    print("S -> Start interaction mode")
    print("D -> Toggle drawing mode ON/OFF")
    print("R -> Reset calibration")
    print("Q -> Quit")
    print("\nNormal mode gestures:")
    print("- Index only -> move cursor + dwell left click")
    print("- Thumb-index pinch -> dwell double click")
    print("- Index + middle only -> dwell right click")
    print("- Index + middle + ring up, thumb+pinky closed -> drag / text selection")
    print("- Open palm -> vertical scroll")
    print("\nDrawing mode:")
    print("- Index only + stable 0.8s -> start drawing")
    print("- Index + middle -> instant click")
    print("- Leaving valid gesture -> release drawing")
    print()

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Error: Could not read frame.")
                break
            #frame = cv2.flip(frame, 1)
            draw_calibration_overlay(frame, state.calibration_points)
            draw_mode_info(
                frame,
                state.mode,
                state.calibration_points,
                LEFT_DWELL_TIME,
                RIGHT_DWELL_TIME,
                DRAG_DWELL_TIME,
                DOUBLE_DWELL_TIME,
                state.drawing_mode
            )

            if state.mode == "interaction" and state.transform_matrix is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = detect_hands(rgb_frame, hand_detector)
                rgb_frame.flags.writeable = True

                fingertip_position, palm_position, gesture_type, drawing_state = draw_hand_info(
                    frame, results, hands_module, drawing_utils, state.drawing_mode
                )

                if fingertip_position is not None:
                    inside_region = point_inside_polygon(fingertip_position, state.calibration_points)

                    if inside_region:
                        mapped_x, mapped_y = map_point_with_homography(
                            fingertip_position,
                            state.transform_matrix
                        )

                        mapped_x = max(0, min(state.screen_width - 1, mapped_x))
                        mapped_y = max(0, min(state.screen_height - 1, mapped_y))

                        current_alpha = DRAWING_SMOOTH_ALPHA if state.drawing_mode else NORMAL_SMOOTH_ALPHA

                        state.smooth_cursor_x, state.smooth_cursor_y = smooth_position(
                            state.smooth_cursor_x,
                            state.smooth_cursor_y,
                            mapped_x,
                            mapped_y,
                            alpha=current_alpha
                        )

                        cv2.putText(
                            frame,
                            "Finger Inside Region",
                            (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 180, 0),
                            2
                        )

                        current_cursor_point = (state.smooth_cursor_x, state.smooth_cursor_y)

                        if state.drawing_mode:
                            if state.drag_active:
                                pyautogui.mouseUp(button='left')
                                state.drag_active = False

                            reset_interaction_state(state)

                            move_cursor_interpolated(
                                state.prev_cursor_x,
                                state.prev_cursor_y,
                                state.smooth_cursor_x,
                                state.smooth_cursor_y,
                                steps=INTERPOLATION_STEPS
                            )
                            state.prev_cursor_x, state.prev_cursor_y = state.smooth_cursor_x, state.smooth_cursor_y

                            if drawing_state == "draw_ready":
                                state.drawing_click_ready = True

                                if state.drawing_anchor is None:
                                    state.drawing_anchor = current_cursor_point
                                    state.drawing_start_time = time.time()

                                movement = calculate_distance(current_cursor_point, state.drawing_anchor)

                                if state.drawing_mouse_down:
                                    cv2.putText(
                                        frame,
                                        "DRAWING...",
                                        (10, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.9,
                                        (0, 255, 0),
                                        3
                                    )
                                else:
                                    if movement <= DRAWING_MOVEMENT_TOLERANCE:
                                        elapsed = time.time() - state.drawing_start_time
                                        progress = elapsed / DRAWING_DWELL_TIME

                                        draw_dwell_progress(
                                            frame,
                                            fingertip_position,
                                            progress,
                                            color=(0, 255, 0)
                                        )

                                        cv2.putText(
                                            frame,
                                            f"Draw Dwell: {elapsed:.1f}/{DRAWING_DWELL_TIME:.1f}s",
                                            (10, 200),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.8,
                                            (0, 255, 0),
                                            2
                                        )

                                        if elapsed >= DRAWING_DWELL_TIME:
                                            pyautogui.mouseDown(button='left')
                                            state.drawing_mouse_down = True

                                            cv2.putText(
                                                frame,
                                                "DRAW STARTED",
                                                (10, 240),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.9,
                                                (0, 255, 0),
                                                3
                                            )
                                    else:
                                        state.drawing_anchor = current_cursor_point
                                        state.drawing_start_time = time.time()

                                        if state.drawing_mouse_down:
                                            pyautogui.mouseUp(button='left')
                                            state.drawing_mouse_down = False

                                        cv2.putText(
                                            frame,
                                            "Hold steady 0.8s to draw",
                                            (10, 200),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.8,
                                            (0, 200, 255),
                                            2
                                        )

                            elif drawing_state == "instant_click":
                                if state.drawing_mouse_down:
                                    pyautogui.mouseUp(button='left')
                                    state.drawing_mouse_down = False

                                state.drawing_anchor = None
                                state.drawing_start_time = None

                                if state.drawing_click_ready:
                                    pyautogui.click()
                                    state.drawing_click_ready = False

                                    cv2.putText(
                                        frame,
                                        "INSTANT CLICK",
                                        (10, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.9,
                                        (255, 0, 255),
                                        3
                                    )
                                else:
                                    cv2.putText(
                                        frame,
                                        "CLICK GESTURE HELD",
                                        (10, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8,
                                        (255, 0, 255),
                                        2
                                    )

                            else:
                                if state.drawing_mouse_down:
                                    pyautogui.mouseUp(button='left')
                                    state.drawing_mouse_down = False

                                reset_drawing_state(state)

                                cv2.putText(
                                    frame,
                                    "Drawing paused - use index only / index+middle",
                                    (10, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 165, 255),
                                    2
                                )

                        else:
                            pyautogui.moveTo(state.smooth_cursor_x, state.smooth_cursor_y)
                            state.prev_cursor_x, state.prev_cursor_y = state.smooth_cursor_x, state.smooth_cursor_y

                            if gesture_type == "double_click":
                                if state.drawing_mouse_down:
                                    pyautogui.mouseUp(button='left')
                                    state.drawing_mouse_down = False

                                state.drag_click_triggered = False
                                state.scroll_anchor_y = None

                                if state.active_dwell_type != "double":
                                    state.dwell_anchor = current_cursor_point
                                    state.dwell_start_time = time.time()
                                    state.click_triggered = False
                                    state.active_dwell_type = "double"
                                else:
                                    movement = calculate_distance(current_cursor_point, state.dwell_anchor)

                                    if movement <= MOVEMENT_TOLERANCE:
                                        elapsed = time.time() - state.dwell_start_time
                                        progress = elapsed / DOUBLE_DWELL_TIME

                                        draw_dwell_progress(frame, fingertip_position, progress, color=(255, 128, 0))

                                        cv2.putText(
                                            frame,
                                            f"Double Dwell: {elapsed:.1f}/{DOUBLE_DWELL_TIME:.1f}s",
                                            (10, 200),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.8,
                                            (255, 180, 0),
                                            2
                                        )

                                        if elapsed >= DOUBLE_DWELL_TIME and not state.click_triggered:
                                            pyautogui.doubleClick()
                                            state.click_triggered = True

                                            cv2.putText(
                                                frame,
                                                "DOUBLE CLICK TRIGGERED",
                                                (10, 240),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.9,
                                                (255, 128, 0),
                                                3
                                            )
                                    else:
                                        state.dwell_anchor = current_cursor_point
                                        state.dwell_start_time = time.time()
                                        state.click_triggered = False

                            elif gesture_type == "left_click":
                                if state.drawing_mouse_down:
                                    pyautogui.mouseUp(button='left')
                                    state.drawing_mouse_down = False

                                if state.drag_active:
                                    pyautogui.mouseUp(button='left')
                                    state.drag_active = False

                                state.drag_click_triggered = False
                                state.scroll_anchor_y = None

                                if state.active_dwell_type != "left":
                                    state.dwell_anchor = current_cursor_point
                                    state.dwell_start_time = time.time()
                                    state.click_triggered = False
                                    state.active_dwell_type = "left"
                                else:
                                    movement = calculate_distance(current_cursor_point, state.dwell_anchor)

                                    if movement <= MOVEMENT_TOLERANCE:
                                        elapsed = time.time() - state.dwell_start_time
                                        progress = elapsed / LEFT_DWELL_TIME

                                        draw_dwell_progress(frame, fingertip_position, progress, color=(0, 0, 255))

                                        cv2.putText(
                                            frame,
                                            f"Left Dwell: {elapsed:.1f}/{LEFT_DWELL_TIME:.1f}s",
                                            (10, 200),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.8,
                                            (0, 255, 255),
                                            2
                                        )

                                        if elapsed >= LEFT_DWELL_TIME and not state.click_triggered:
                                            pyautogui.click()
                                            state.click_triggered = True

                                            cv2.putText(
                                                frame,
                                                "LEFT CLICK TRIGGERED",
                                                (10, 240),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.9,
                                                (0, 0, 255),
                                                3
                                            )
                                    else:
                                        state.dwell_anchor = current_cursor_point
                                        state.dwell_start_time = time.time()
                                        state.click_triggered = False

                            elif gesture_type == "right_click":
                                if state.drawing_mouse_down:
                                    pyautogui.mouseUp(button='left')
                                    state.drawing_mouse_down = False

                                if state.drag_active:
                                    pyautogui.mouseUp(button='left')
                                    state.drag_active = False

                                state.drag_click_triggered = False
                                state.scroll_anchor_y = None

                                if state.active_dwell_type != "right":
                                    state.dwell_anchor = current_cursor_point
                                    state.dwell_start_time = time.time()
                                    state.click_triggered = False
                                    state.active_dwell_type = "right"
                                else:
                                    movement = calculate_distance(current_cursor_point, state.dwell_anchor)

                                    if movement <= MOVEMENT_TOLERANCE:
                                        elapsed = time.time() - state.dwell_start_time
                                        progress = elapsed / RIGHT_DWELL_TIME

                                        draw_dwell_progress(frame, fingertip_position, progress, color=(255, 0, 255))

                                        cv2.putText(
                                            frame,
                                            f"Right Dwell: {elapsed:.1f}/{RIGHT_DWELL_TIME:.1f}s",
                                            (10, 200),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.8,
                                            (255, 200, 255),
                                            2
                                        )

                                        if elapsed >= RIGHT_DWELL_TIME and not state.click_triggered:
                                            pyautogui.click(button='right')
                                            state.click_triggered = True

                                            cv2.putText(
                                                frame,
                                                "RIGHT CLICK TRIGGERED",
                                                (10, 240),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.9,
                                                (255, 0, 255),
                                                3
                                            )
                                    else:
                                        state.dwell_anchor = current_cursor_point
                                        state.dwell_start_time = time.time()
                                        state.click_triggered = False

                            elif gesture_type == "drag":
                                if state.drawing_mouse_down:
                                    pyautogui.mouseUp(button='left')
                                    state.drawing_mouse_down = False

                                state.scroll_anchor_y = None
                                state.click_triggered = False

                                if state.active_dwell_type != "drag":
                                    state.dwell_anchor = current_cursor_point
                                    state.dwell_start_time = time.time()
                                    state.drag_click_triggered = False
                                    state.active_dwell_type = "drag"
                                else:
                                    movement = calculate_distance(current_cursor_point, state.dwell_anchor)

                                    if movement <= MOVEMENT_TOLERANCE:
                                        elapsed = time.time() - state.dwell_start_time
                                        progress = elapsed / DRAG_DWELL_TIME

                                        draw_dwell_progress(frame, fingertip_position, progress, color=(0, 255, 255))

                                        cv2.putText(
                                            frame,
                                            f"Drag Dwell: {elapsed:.1f}/{DRAG_DWELL_TIME:.1f}s",
                                            (10, 200),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.8,
                                            (0, 255, 255),
                                            2
                                        )

                                        if elapsed >= DRAG_DWELL_TIME and not state.drag_click_triggered:
                                            pyautogui.mouseDown(button='left')
                                            state.drag_active = True
                                            state.drag_click_triggered = True

                                        if state.drag_active:
                                            cv2.putText(
                                                frame,
                                                "DRAG ACTIVE",
                                                (10, 240),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.9,
                                                (0, 255, 255),
                                                3
                                            )
                                    else:
                                        state.dwell_anchor = current_cursor_point
                                        state.dwell_start_time = time.time()
                                        state.drag_click_triggered = False

                                        cv2.putText(
                                            frame,
                                            "Drag Dwell Reset (movement detected)",
                                            (10, 200),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.8,
                                            (0, 200, 255),
                                            2
                                        )

                            elif gesture_type == "scroll":
                                if state.drawing_mouse_down:
                                    pyautogui.mouseUp(button='left')
                                    state.drawing_mouse_down = False

                                if state.drag_active:
                                    pyautogui.mouseUp(button='left')
                                    state.drag_active = False

                                state.dwell_anchor = None
                                state.dwell_start_time = None
                                state.click_triggered = False
                                state.drag_click_triggered = False
                                state.active_dwell_type = None

                                if palm_position is not None:
                                    current_palm_y = palm_position[1]

                                    if state.scroll_anchor_y is None:
                                        state.scroll_anchor_y = current_palm_y
                                    else:
                                        dy = current_palm_y - state.scroll_anchor_y
                                        now = time.time()

                                        cv2.putText(
                                            frame,
                                            f"Scroll DY: {dy}",
                                            (10, 200),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.8,
                                            (255, 255, 0),
                                            2
                                        )

                                        if abs(dy) >= SCROLL_THRESHOLD and (now - state.last_scroll_time) >= SCROLL_COOLDOWN:
                                            if dy < 0:
                                                pyautogui.scroll(SCROLL_AMOUNT)
                                                cv2.putText(
                                                    frame,
                                                    "SCROLL UP",
                                                    (10, 240),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.9,
                                                    (255, 255, 0),
                                                    3
                                                )
                                            else:
                                                pyautogui.scroll(-SCROLL_AMOUNT)
                                                cv2.putText(
                                                    frame,
                                                    "SCROLL DOWN",
                                                    (10, 240),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.9,
                                                    (255, 255, 0),
                                                    3
                                                )

                                            state.last_scroll_time = now
                                            state.scroll_anchor_y = current_palm_y

                            else:
                                release_all_mouse_states(state)
                                reset_interaction_state(state)

                                cv2.putText(
                                    frame,
                                    "No active action gesture",
                                    (10, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 165, 255),
                                    2
                                )
                    else:
                        release_all_mouse_states(state)
                        reset_drawing_state(state)
                        reset_interaction_state(state)

                        cv2.putText(
                            frame,
                            "Finger Outside Region",
                            (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )
                else:
                    release_all_mouse_states(state)
                    reset_drawing_state(state)
                    reset_interaction_state(state)

            current_time = time.time()
            fps = 1 / (current_time - state.previous_time) if state.previous_time > 0 else 0
            state.previous_time = current_time

            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Exiting...")
                break

            elif key == ord('k'):
                release_all_mouse_states(state)
                state.drawing_mode = False
                state.mode = "calibration"
                state.calibration_points = []
                state.transform_matrix = None
                reset_interaction_state(state)
                reset_drawing_state(state)
                print("Calibration mode enabled. Click 4 points.")

            elif key == ord('r'):
                release_all_mouse_states(state)
                state.drawing_mode = False
                state.calibration_points = []
                state.transform_matrix = None
                state.mode = "idle"
                reset_interaction_state(state)
                reset_drawing_state(state)
                print("Reset complete. Back to idle mode.")

            elif key == ord('s'):
                if len(state.calibration_points) == 4:
                    src_points = np.array(state.calibration_points, dtype=np.float32)
                    dst_points = np.array([
                        [0, 0],
                        [state.screen_width - 1, 0],
                        [state.screen_width - 1, state.screen_height - 1],
                        [0, state.screen_height - 1]
                    ], dtype=np.float32)

                    state.transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                    state.mode = "interaction"

                    reset_interaction_state(state)
                    reset_drawing_state(state)

                    state.prev_cursor_x, state.prev_cursor_y = pyautogui.position()
                    state.smooth_cursor_x, state.smooth_cursor_y = state.prev_cursor_x, state.prev_cursor_y

                    print("Interaction mode started.")
                else:
                    print("Please select exactly 4 calibration points first.")

            elif key == ord('d'):
                if state.mode != "interaction":
                    print("Start interaction mode first, then use D for drawing mode.")
                else:
                    state.drawing_mode = not state.drawing_mode

                    release_all_mouse_states(state)
                    reset_interaction_state(state)
                    reset_drawing_state(state)

                    state.prev_cursor_x, state.prev_cursor_y = pyautogui.position()
                    state.smooth_cursor_x, state.smooth_cursor_y = state.prev_cursor_x, state.prev_cursor_y

                    if state.drawing_mode:
                        print("Drawing mode ON")
                    else:
                        print("Drawing mode OFF")

    finally:
        release_all_mouse_states(state)
        camera.release()
        cv2.destroyAllWindows()
        hand_detector.close()


if __name__ == "__main__":
    main()