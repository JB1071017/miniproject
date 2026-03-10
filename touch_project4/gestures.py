import math
import mediapipe as mp


def is_finger_extended(hand_landmarks, tip_id, pip_id):
    tip = hand_landmarks.landmark[tip_id]
    pip = hand_landmarks.landmark[pip_id]
    return tip.y < pip.y


def is_thumb_closed(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
    wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]

    palm_size = math.hypot(index_mcp.x - wrist.x, index_mcp.y - wrist.y)
    thumb_to_index = math.hypot(thumb_tip.x - index_mcp.x, thumb_tip.y - index_mcp.y)

    return thumb_to_index < (palm_size * 0.9)


def is_thumb_index_pinch(hand_landmarks, hands_module, pinch_ratio_threshold=0.35):
    thumb_tip = hand_landmarks.landmark[hands_module.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[hands_module.HandLandmark.INDEX_FINGER_TIP]
    wrist = hand_landmarks.landmark[hands_module.HandLandmark.WRIST]
    middle_mcp = hand_landmarks.landmark[hands_module.HandLandmark.MIDDLE_FINGER_MCP]

    pinch_distance = math.hypot(
        thumb_tip.x - index_tip.x,
        thumb_tip.y - index_tip.y
    )

    palm_size = math.hypot(
        middle_mcp.x - wrist.x,
        middle_mcp.y - wrist.y
    )

    if palm_size == 0:
        return False

    normalized_distance = pinch_distance / palm_size
    return normalized_distance < pinch_ratio_threshold


def get_gesture_type(hand_landmarks, hands_module):
    index_extended = is_finger_extended(
        hand_landmarks,
        hands_module.HandLandmark.INDEX_FINGER_TIP,
        hands_module.HandLandmark.INDEX_FINGER_PIP
    )
    middle_extended = is_finger_extended(
        hand_landmarks,
        hands_module.HandLandmark.MIDDLE_FINGER_TIP,
        hands_module.HandLandmark.MIDDLE_FINGER_PIP
    )
    ring_extended = is_finger_extended(
        hand_landmarks,
        hands_module.HandLandmark.RING_FINGER_TIP,
        hands_module.HandLandmark.RING_FINGER_PIP
    )
    pinky_extended = is_finger_extended(
        hand_landmarks,
        hands_module.HandLandmark.PINKY_TIP,
        hands_module.HandLandmark.PINKY_PIP
    )

    thumb_closed = is_thumb_closed(hand_landmarks)
    thumb_index_pinch = is_thumb_index_pinch(hand_landmarks, hands_module)

    if thumb_index_pinch and (not middle_extended) and (not ring_extended) and (not pinky_extended):
        return "double_click"

    if index_extended and (not middle_extended) and (not ring_extended) and (not pinky_extended) and (not thumb_index_pinch):
        return "left_click"

    if index_extended and middle_extended and (not ring_extended) and (not pinky_extended):
        return "right_click"

    if index_extended and middle_extended and ring_extended and (not pinky_extended) and thumb_closed:
        return "drag"

    if index_extended and middle_extended and ring_extended and pinky_extended:
        return "scroll"

    return "none"


def get_drawing_mode_state(hand_landmarks, hands_module):
    index_extended = is_finger_extended(
        hand_landmarks,
        hands_module.HandLandmark.INDEX_FINGER_TIP,
        hands_module.HandLandmark.INDEX_FINGER_PIP
    )
    middle_extended = is_finger_extended(
        hand_landmarks,
        hands_module.HandLandmark.MIDDLE_FINGER_TIP,
        hands_module.HandLandmark.MIDDLE_FINGER_PIP
    )
    ring_extended = is_finger_extended(
        hand_landmarks,
        hands_module.HandLandmark.RING_FINGER_TIP,
        hands_module.HandLandmark.RING_FINGER_PIP
    )
    pinky_extended = is_finger_extended(
        hand_landmarks,
        hands_module.HandLandmark.PINKY_TIP,
        hands_module.HandLandmark.PINKY_PIP
    )

    if index_extended and (not middle_extended) and (not ring_extended) and (not pinky_extended):
        return "draw_ready"

    if index_extended and middle_extended and (not ring_extended) and (not pinky_extended):
        return "instant_click"

    return "blocked"