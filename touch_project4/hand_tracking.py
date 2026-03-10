import cv2
from gestures import get_gesture_type, get_drawing_mode_state


def detect_hands(rgb_frame, hand_detector):
    return hand_detector.process(rgb_frame)


def get_landmark_position(frame, hand_landmarks, landmark_id):
    height, width, _ = frame.shape
    landmark = hand_landmarks.landmark[landmark_id]

    x = int(landmark.x * width)
    y = int(landmark.y * height)

    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))

    return x, y


def draw_hand_info(frame, results, hands_module, drawing_utils, drawing_mode=False):
    fingertip_position = None
    palm_position = None
    gesture_type = "none"
    drawing_state = "blocked"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                hands_module.HAND_CONNECTIONS
            )

            index_tip = get_landmark_position(
                frame, hand_landmarks, hands_module.HandLandmark.INDEX_FINGER_TIP
            )
            palm_center = get_landmark_position(
                frame, hand_landmarks, hands_module.HandLandmark.MIDDLE_FINGER_MCP
            )

            fingertip_position = index_tip
            palm_position = palm_center

            if drawing_mode:
                drawing_state = get_drawing_mode_state(hand_landmarks, hands_module)

                if drawing_state == "draw_ready":
                    tip_color = (0, 255, 0)
                    gesture_text = "DRAW READY - INDEX ONLY"
                    gesture_color = (0, 255, 0)
                elif drawing_state == "instant_click":
                    tip_color = (255, 0, 255)
                    gesture_text = "INSTANT CLICK - INDEX + MIDDLE"
                    gesture_color = (255, 0, 255)
                else:
                    tip_color = (0, 165, 255)
                    gesture_text = "DRAW BLOCKED - USE INDEX ONLY / INDEX+MIDDLE"
                    gesture_color = (0, 165, 255)
            else:
                gesture_type = get_gesture_type(hand_landmarks, hands_module)

                if gesture_type == "left_click":
                    tip_color = (0, 255, 0)
                    gesture_text = "LEFT CLICK GESTURE"
                    gesture_color = (0, 255, 0)
                elif gesture_type == "double_click":
                    tip_color = (255, 128, 0)
                    gesture_text = "DOUBLE CLICK GESTURE (PINCH)"
                    gesture_color = (255, 128, 0)
                elif gesture_type == "right_click":
                    tip_color = (255, 0, 255)
                    gesture_text = "RIGHT CLICK GESTURE"
                    gesture_color = (255, 0, 255)
                elif gesture_type == "drag":
                    tip_color = (0, 255, 255)
                    gesture_text = "DRAG GESTURE (3 fingers up)"
                    gesture_color = (0, 255, 255)
                elif gesture_type == "scroll":
                    tip_color = (255, 255, 0)
                    gesture_text = "SCROLL GESTURE"
                    gesture_color = (255, 255, 0)
                else:
                    tip_color = (0, 165, 255)
                    gesture_text = "Show valid gesture"
                    gesture_color = (0, 165, 255)

            cv2.circle(frame, index_tip, 10, tip_color, cv2.FILLED)
            cv2.circle(frame, palm_center, 8, (255, 255, 255), cv2.FILLED)

            cv2.putText(
                frame,
                f"Index Tip: ({index_tip[0]}, {index_tip[1]})",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )
            cv2.putText(
                frame,
                gesture_text,
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                gesture_color,
                2
            )
            break
    else:
        cv2.putText(
            frame,
            "No hand detected",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    return fingertip_position, palm_position, gesture_type, drawing_state