# Touch-Projectors - MediaPipe Hand Detection with Perspective Transform

import cv2
import numpy as np
import pyautogui
from pynput.mouse import Button, Controller
import time
import math
import mediapipe as mp

# Initialize video capture
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("http://192.168.137.244:4747/video")
time.sleep(1.1)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure MediaPipe Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Detect only one hand for simplicity
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=1  # 0=Lite, 1=Full, 2=Heavy (better accuracy but slower)
)

# Initialize variables
mouse = Controller()
pts = [(0, 0), (0, 0), (0, 0), (0, 0)]
pointIndex = 0
AR = (740, 1280)  # Height, Width of projection area
oppts = np.float32([[0, 0], [AR[1], 0], [0, AR[0]], [AR[1], AR[0]]])

# Touch detection parameters
touch_zone_y = int(AR[0] * 0.8)  # Bottom 20% of screen is touch zone
touch_threshold = 30  # Pixels from bottom to trigger touch
click_cooldown = 0
click_cooldown_max = 15  # Frames between clicks
touch_active = False

# Gesture detection parameters
pinch_threshold = 0.05  # Normalized distance for pinch detection
click_method = 0  # 0: Touch zone, 1: Pinch gesture, 2: Finger stay
method_names = ["Touch Zone", "Pinch Gesture", "Finger Stay"]

# For tracking fingertip position
prev_index_tip = None
stay_counter = 0
stay_threshold = 10  # Frames to stay for click

# Landmark indices for fingers
LANDMARKS = {
    'WRIST': 0,
    'THUMB_CMC': 1,
    'THUMB_MCP': 2,
    'THUMB_IP': 3,
    'THUMB_TIP': 4,
    'INDEX_MCP': 5,
    'INDEX_PIP': 6,
    'INDEX_DIP': 7,
    'INDEX_TIP': 8,
    'MIDDLE_MCP': 9,
    'MIDDLE_PIP': 10,
    'MIDDLE_DIP': 11,
    'MIDDLE_TIP': 12,
    'RING_MCP': 13,
    'RING_PIP': 14,
    'RING_DIP': 15,
    'RING_TIP': 16,
    'PINKY_MCP': 17,
    'PINKY_PIP': 18,
    'PINKY_DIP': 19,
    'PINKY_TIP': 20
}

def draw_circle(event, x, y, flags, param):
    """Mouse callback function for selecting perspective points"""
    global img
    global pointIndex
    global pts

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        pts[pointIndex] = (x, y)
        print(f"Point {pointIndex + 1} selected: ({x}, {y})")
        pointIndex = pointIndex + 1

def show_window():
    """Display window for perspective point selection"""
    while True:
        cv2.imshow('img', img)

        if pointIndex == 4:
            print("All 4 points selected!")
            break

        if cv2.waitKey(20) & 0xFF == 27:
            break

def get_persp(image, pts):
    """Apply perspective transform to get top-down view"""
    ippts = np.float32(pts)
    Map = cv2.getPerspectiveTransform(ippts, oppts)
    warped = cv2.warpPerspective(image, Map, (AR[1], AR[0]))
    return warped

def calculate_distance(landmark1, landmark2, image_shape):
    """Calculate Euclidean distance between two landmarks in pixels"""
    h, w = image_shape[:2]
    x1, y1 = int(landmark1.x * w), int(landmark1.y * h)
    x2, y2 = int(landmark2.x * w), int(landmark2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_normalized_distance(landmark1, landmark2):
    """Calculate normalized distance between two landmarks (0-1)"""
    dx = landmark2.x - landmark1.x
    dy = landmark2.y - landmark1.y
    return math.sqrt(dx**2 + dy**2)

def detect_pinch(hand_landmarks, image_shape, threshold=pinch_threshold):
    """
    Detect pinch gesture between thumb and index finger
    Returns: (is_pinching, pinch_center)
    """
    thumb_tip = hand_landmarks.landmark[LANDMARKS['THUMB_TIP']]
    index_tip = hand_landmarks.landmark[LANDMARKS['INDEX_TIP']]
    
    # Calculate normalized distance
    distance = calculate_normalized_distance(thumb_tip, index_tip)
    
    # Calculate pinch center in pixel coordinates
    h, w = image_shape[:2]
    center_x = int((thumb_tip.x + index_tip.x) * w / 2)
    center_y = int((thumb_tip.y + index_tip.y) * h / 2)
    
    return distance < threshold, (center_x, center_y), distance

def is_touching_screen(fingertip_y, touch_zone_y, threshold=touch_threshold):
    """
    Detect if fingertip is touching the screen based on vertical position
    """
    return fingertip_y > (touch_zone_y - threshold)

def get_finger_state(hand_landmarks):
    """
    Determine which fingers are extended
    Returns dictionary of finger states
    """
    h, w = 1, 1  # Normalized coordinates, so we can use raw values
    
    finger_tips = {
        'THUMB': LANDMARKS['THUMB_TIP'],
        'INDEX': LANDMARKS['INDEX_TIP'],
        'MIDDLE': LANDMARKS['MIDDLE_TIP'],
        'RING': LANDMARKS['RING_TIP'],
        'PINKY': LANDMARKS['PINKY_TIP']
    }
    
    finger_pips = {
        'THUMB': LANDMARKS['THUMB_IP'],  # Using IP for thumb
        'INDEX': LANDMARKS['INDEX_PIP'],
        'MIDDLE': LANDMARKS['MIDDLE_PIP'],
        'RING': LANDMARKS['RING_PIP'],
        'PINKY': LANDMARKS['PINKY_PIP']
    }
    
    states = {}
    for finger in finger_tips:
        tip = hand_landmarks.landmark[finger_tips[finger]]
        pip = hand_landmarks.landmark[finger_pips[finger]]
        
        # Finger is extended if tip is above pip (in normalized coordinates)
        # Note: In image coordinates, y increases downward, so tip.y < pip.y means extended
        states[finger] = tip.y < pip.y
    
    return states

def draw_finger_states(image, states, hand_landmarks):
    """Draw finger state indicators on the image"""
    h, w = image.shape[:2]
    y_offset = 60
    
    for i, (finger, extended) in enumerate(states.items()):
        color = (0, 255, 0) if extended else (0, 0, 255)
        status = "UP" if extended else "DOWN"
        cv2.putText(image, f"{finger}: {status}", (10, y_offset + i*25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def draw_landmarks_with_labels(image, hand_landmarks):
    """Draw hand landmarks with numbered labels for debugging"""
    h, w = image.shape[:2]
    
    # Draw connections
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())
    
    # Draw landmark numbers (optional - for debugging)
    for idx, landmark in enumerate(hand_landmarks.landmark):
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.putText(image, str(idx), (x-10, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

# Setup window for perspective point selection
cv2.namedWindow('img')
cv2.setMouseCallback('img', draw_circle)
print('Select points in order: Top Left, Top Right, Bottom Right, Bottom Left')
_, img = cap.read()
show_window()

# Main detection loop
print("\n" + "="*60)
print("MEDIAPIPE HAND DETECTION - Touch Projector")
print("="*60)
print("Instructions:")
print("  - Show your hand to the camera")
print("  - MediaPipe will automatically detect and track your hand")
print("  - Index finger tip controls mouse cursor")
print("\nClick Methods:")
print("  1. Touch Zone - Touch bottom area of screen")
print("  2. Pinch Gesture - Pinch thumb and index finger")
print("  3. Finger Stay - Hold finger still for a moment")
print("\nControls:")
print("  ESC - Exit")
print("  m - Switch click method")
print("  + / - - Adjust touch threshold")
print("  p - Print hand landmarks (debug)")
print("="*60)

# Initialize variables
touch_zone_y = int(AR[0] * 0.8)  # Bottom 20% of screen
touch_threshold = 30

while True:
    # Read frame
    _, frame = cap.read()
    if frame is None:
        print("Failed to grab frame")
        break

    # Apply perspective transform
    warped = get_persp(frame, pts)
    
    # Create a copy for display
    display = warped.copy()
    
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    
    # Process with MediaPipe
    results = hands.process(rgb_frame)
    
    # Convert back to BGR for OpenCV display
    rgb_frame.flags.writeable = True
    
    # Draw touch zone
    cv2.line(display, (0, touch_zone_y), (AR[1], touch_zone_y), (255, 255, 0), 2)
    cv2.putText(display, f"Touch Zone ({method_names[click_method]})", 
                (10, touch_zone_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Check if hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks with labels
            draw_landmarks_with_labels(display, hand_landmarks)
            
            # Get index finger tip (landmark 8)
            index_tip = hand_landmarks.landmark[LANDMARKS['INDEX_TIP']]
            
            # Convert to pixel coordinates
            h, w, _ = display.shape
            fingertip_x = int(index_tip.x * w)
            fingertip_y = int(index_tip.y * h)
            fingertip = (fingertip_x, fingertip_y)
            
            # Get thumb tip for pinch detection
            thumb_tip = hand_landmarks.landmark[LANDMARKS['THUMB_TIP']]
            
            # Draw fingertip with highlight
            cv2.circle(display, fingertip, 15, (0, 0, 255), 3)
            cv2.circle(display, fingertip, 5, (255, 255, 255), -1)
            cv2.putText(display, "Index", 
                       (fingertip[0] - 30, fingertip[1] - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Get finger states
            finger_states = get_finger_state(hand_landmarks)
            draw_finger_states(display, finger_states, hand_landmarks)
            
            # Convert to screen coordinates
            screen_width, screen_height = pyautogui.size()
            x_percent = (fingertip_x / AR[1]) * 100
            y_percent = (fingertip_y / AR[0]) * 100
            screen_x = int((screen_width * x_percent) / 100)
            screen_y = int((screen_height * y_percent) / 100)
            
            # Move mouse to fingertip position (always)
            mouse.position = (screen_x, screen_y)
            
            # Click detection based on selected method
            click_triggered = False
            
            if click_method == 0:  # Touch Zone
                if is_touching_screen(fingertip_y, touch_zone_y, touch_threshold):
                    if not touch_active and click_cooldown <= 0:
                        mouse.click(Button.left)
                        print(f"\rTOUCH ZONE CLICK at ({screen_x}, {screen_y})", end="")
                        click_triggered = True
                        touch_active = True
                        click_cooldown = click_cooldown_max
                else:
                    touch_active = False
                    
            elif click_method == 1:  # Pinch Gesture
                is_pinching, pinch_center, pinch_dist = detect_pinch(hand_landmarks, display.shape)
                
                # Draw pinch visualization
                if is_pinching:
                    cv2.circle(display, pinch_center, 20, (0, 255, 255), 2)
                    cv2.line(display, fingertip, 
                            (int(thumb_tip.x * w), int(thumb_tip.y * h)), 
                            (0, 255, 255), 2)
                    cv2.putText(display, f"Pinch: {pinch_dist:.2f}", 
                               (pinch_center[0] - 30, pinch_center[1] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                if is_pinching and not touch_active and click_cooldown <= 0:
                    mouse.click(Button.left)
                    print(f"\rPINCH CLICK at ({screen_x}, {screen_y})", end="")
                    click_triggered = True
                    touch_active = True
                    click_cooldown = click_cooldown_max
                elif not is_pinching:
                    touch_active = False
                    
            elif click_method == 2:  # Finger Stay
                if prev_index_tip is not None:
                    # Calculate movement distance
                    distance = math.sqrt((fingertip_x - prev_index_tip[0])**2 + 
                                        (fingertip_y - prev_index_tip[1])**2)
                    
                    if distance < 5:  # Finger is still
                        stay_counter += 1
                        # Draw progress bar
                        progress = min(stay_counter / stay_threshold, 1.0)
                        bar_width = 50
                        filled_width = int(bar_width * progress)
                        cv2.rectangle(display, (fingertip_x - bar_width//2, fingertip_y - 40),
                                     (fingertip_x + bar_width//2, fingertip_y - 30),
                                     (100, 100, 100), -1)
                        cv2.rectangle(display, (fingertip_x - bar_width//2, fingertip_y - 40),
                                     (fingertip_x - bar_width//2 + filled_width, fingertip_y - 30),
                                     (0, 255, 0), -1)
                        
                        if stay_counter >= stay_threshold and click_cooldown <= 0:
                            mouse.click(Button.left)
                            print(f"\rSTAY CLICK at ({screen_x}, {screen_y})", end="")
                            click_triggered = True
                            click_cooldown = click_cooldown_max
                            stay_counter = 0
                    else:
                        stay_counter = max(0, stay_counter - 1)
                
                prev_index_tip = (fingertip_x, fingertip_y)
            
            # If click triggered, show visual feedback
            if click_triggered:
                cv2.putText(display, "CLICK!", (fingertip_x - 30, fingertip_y - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    else:
        # No hand detected
        cv2.putText(display, "No Hand Detected", (AR[1]//2 - 100, AR[0]//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        prev_index_tip = None
        stay_counter = 0
    
    # Update cooldown
    if click_cooldown > 0:
        click_cooldown -= 1
        cv2.putText(display, f"Cooldown: {click_cooldown}", (10, AR[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add info text
    info_lines = [
        f"Method: {method_names[click_method]}",
        f"Touch Zone: {touch_zone_y}",
        f"Threshold: {touch_threshold}",
        f"Stay Counter: {stay_counter}"
    ]
    
    for i, line in enumerate(info_lines):
        cv2.putText(display, line, (AR[1] - 250, 30 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw the selected points on original frame
    for i, point in enumerate(pts):
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
        cv2.putText(frame, str(i + 1), point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)
    
    # Display windows
    cv2.imshow('Original with Points', frame)
    cv2.imshow('MediaPipe Hand Detection', display)
    
    # Create debug view with landmarks only (optional)
    debug_view = np.zeros_like(display)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                debug_view,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    cv2.imshow('Landmarks Only', debug_view)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('m'):  # Switch click method
        click_method = (click_method + 1) % 3
        print(f"\nSwitched to {method_names[click_method]} method")
    elif key == ord('+'):  # Increase touch threshold
        touch_threshold = min(100, touch_threshold + 5)
        touch_zone_y = AR[0] - touch_threshold
        print(f"\nTouch threshold: {touch_threshold}")
    elif key == ord('-'):  # Decrease touch threshold
        touch_threshold = max(10, touch_threshold - 5)
        touch_zone_y = AR[0] - touch_threshold
        print(f"\nTouch threshold: {touch_threshold}")
    elif key == ord('p'):  # Print landmarks (debug)
        if results.multi_hand_landmarks:
            print("\nHand Landmarks:")
            for idx, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
                print(f"  {idx}: ({landmark.x:.3f}, {landmark.y:.3f}, {landmark.z:.3f})")

# Cleanup
hands.close()
cv2.destroyAllWindows()
cap.release()
print("\nProgram terminated")