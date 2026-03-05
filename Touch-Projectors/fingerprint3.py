# Touch-Projectors - Single Calibration with Feature Tracking
# No physical markers needed - just click 4 corners once!

import cv2
import numpy as np
import pyautogui
from pynput.mouse import Button, Controller
import time
import math
import mediapipe as mp

# ============================================
# INITIALIZATION
# ============================================

# Initialize video capture
# Use webcam: cap = cv2.VideoCapture(0)
# Use phone camera: 
cap = cv2.VideoCapture("http://10.202.214.63:4747/video")
time.sleep(1.1)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Initialize mouse
mouse = Controller()

# ============================================
# PROJECTION AREA DIMENSIONS
# ============================================

# This should match your actual projection area aspect ratio
# Format: (Height, Width)
AR = (740, 1280)  # Standard 16:9 aspect ratio

# Destination points for perspective transform (the "ideal" top-down view)
dst_pts = np.float32([
    [0, 0],           # Top-Left
    [AR[1], 0],       # Top-Right
    [AR[1], AR[0]],   # Bottom-Right
    [0, AR[0]]        # Bottom-Left
])

# ============================================
# CALIBRATION VARIABLES
# ============================================

calibration_points = []  # Stores the 4 clicked points
calibration_done = False
transform_matrix = None
reference_frame = None
homography_matrix = None

# Feature tracking variables
orb = cv2.ORB_create(nfeatures=2000)  # ORB feature detector
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # Feature matcher
reference_kp = None  # Keypoints from reference frame
reference_des = None  # Descriptors from reference frame

# ============================================
# TOUCH DETECTION PARAMETERS
# ============================================

# THESE ARE GLOBAL VARIABLES - accessible everywhere
touch_zone_ratio = 0.8  # Bottom 20% is touch zone
click_cooldown = 0
click_cooldown_max = 10
touch_active = False
click_method = 0  # 0: Touch Zone, 1: Pinch Gesture, 2: Finger Stay
method_names = ["Touch Zone", "Pinch Gesture", "Finger Stay"]

# For gesture tracking
prev_index_tip = None
stay_counter = 0
stay_threshold = 10

# Landmark indices for MediaPipe
LM_INDEX = {
    'WRIST': 0,
    'THUMB_TIP': 4,
    'INDEX_TIP': 8,
    'MIDDLE_TIP': 12,
    'RING_TIP': 16,
    'PINKY_TIP': 20
}

# ============================================
# MOUSE CALLBACK FOR CALIBRATION
# ============================================

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function for selecting calibration points
    Click the 4 corners of your projection area in order
    """
    global calibration_points, reference_frame, calibration_done
    global transform_matrix, reference_kp, reference_des, homography_matrix
    global calibration_display, current_frame
    
    if event == cv2.EVENT_LBUTTONDOWN and not calibration_done:
        # Draw circle at clicked point
        cv2.circle(calibration_display, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(calibration_display, str(len(calibration_points)+1), 
                   (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Store point
        calibration_points.append((x, y))
        
        # Get corner name
        corner_names = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
        print(f"Point {len(calibration_points)}: ({x}, {y}) - {corner_names[len(calibration_points)-1]}")
        
        # If we have all 4 points, calculate perspective transform
        if len(calibration_points) == 4:
            print("\n" + "="*50)
            print("CALIBRATION COMPLETE!")
            print("="*50)
            
            # Source points from user clicks
            src_pts = np.float32(calibration_points)
            
            # Calculate perspective transform
            transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            # Also calculate homography (inverse)
            homography_matrix = cv2.getPerspectiveTransform(dst_pts, src_pts)
            
            # Save reference frame for feature tracking
            reference_frame = current_frame.copy()
            
            # Extract features from reference frame
            reference_kp, reference_des = orb.detectAndCompute(reference_frame, None)
            
            print(f"Detected {len(reference_kp)} features in reference frame")
            print("System will now track perspective automatically!")
            print("\nYou can now use hand gestures to control the mouse.")
            
            calibration_done = True

# ============================================
# FUNCTION: Get Current Transform
# ============================================

def get_current_transform(current_frame):
    """
    Calculate current perspective transform using feature matching
    This adapts to phone movement automatically
    """
    global homography_matrix, reference_kp, reference_des
    
    if reference_kp is None or reference_des is None:
        return None, None
    
    # Detect features in current frame
    current_kp, current_des = orb.detectAndCompute(current_frame, None)
    
    if current_des is None or len(current_des) < 10:
        return None, None
    
    # Match features with reference frame
    matches = bf.match(reference_des, current_des)
    
    # Sort by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Use top matches (filter out poor matches)
    good_matches = []
    for match in matches[:100]:  # Take top 100 matches
        if match.distance < 50:  # Threshold for good matches
            good_matches.append(match)
    
    if len(good_matches) < 10:
        return None, None
    
    # Get matching points
    src_pts = np.float32([reference_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([current_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    try:
        # Find homography matrix
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is not None:
            return H, good_matches
        else:
            return None, None
            
    except Exception as e:
        return None, None

# ============================================
# FUNCTION: Convert Warped to Screen Coordinates
# ============================================

def warped_to_screen(warped_point):
    """Convert warped image coordinates to screen coordinates"""
    screen_width, screen_height = pyautogui.size()
    screen_x = int((warped_point[0] / AR[1]) * screen_width)
    screen_y = int((warped_point[1] / AR[0]) * screen_height)
    return (screen_x, screen_y)

# ============================================
# FUNCTION: Detect Pinch Gesture
# ============================================

def detect_pinch(hand_landmarks, image_shape, threshold=0.05):
    """
    Detect pinch gesture between thumb and index finger
    """
    h, w = image_shape[:2]
    
    thumb_tip = hand_landmarks.landmark[LM_INDEX['THUMB_TIP']]
    index_tip = hand_landmarks.landmark[LM_INDEX['INDEX_TIP']]
    
    # Calculate normalized distance
    dx = thumb_tip.x - index_tip.x
    dy = thumb_tip.y - index_tip.y
    distance = math.sqrt(dx**2 + dy**2)
    
    # Calculate pinch center
    center_x = int((thumb_tip.x + index_tip.x) * w / 2)
    center_y = int((thumb_tip.y + index_tip.y) * h / 2)
    
    return distance < threshold, (center_x, center_y), distance

# ============================================
# FUNCTION: Get Finger States
# ============================================

def get_finger_states(hand_landmarks):
    """
    Determine which fingers are extended
    """
    states = {}
    
    # Thumb: compare x-coordinates (for right hand)
    thumb_tip = hand_landmarks.landmark[LM_INDEX['THUMB_TIP']]
    thumb_ip = hand_landmarks.landmark[3]  # THUMB_IP
    states['THUMB'] = thumb_tip.x < thumb_ip.x
    
    # Index finger
    index_tip = hand_landmarks.landmark[LM_INDEX['INDEX_TIP']]
    index_pip = hand_landmarks.landmark[6]  # INDEX_PIP
    states['INDEX'] = index_tip.y < index_pip.y
    
    # Middle finger
    middle_tip = hand_landmarks.landmark[LM_INDEX['MIDDLE_TIP']]
    middle_pip = hand_landmarks.landmark[10]  # MIDDLE_PIP
    states['MIDDLE'] = middle_tip.y < middle_pip.y
    
    # Ring finger
    ring_tip = hand_landmarks.landmark[LM_INDEX['RING_TIP']]
    ring_pip = hand_landmarks.landmark[14]  # RING_PIP
    states['RING'] = ring_tip.y < ring_pip.y
    
    # Pinky finger
    pinky_tip = hand_landmarks.landmark[LM_INDEX['PINKY_TIP']]
    pinky_pip = hand_landmarks.landmark[18]  # PINKY_PIP
    states['PINKY'] = pinky_tip.y < pinky_pip.y
    
    return states

# ============================================
# MAIN PROGRAM
# ============================================

def main():
    global calibration_display, current_frame, calibration_done
    global click_method, touch_active, click_cooldown
    global prev_index_tip, stay_counter, calibration_points
    global touch_zone_ratio  # Add this line to access global variable
    
    print("\n" + "="*70)
    print("TOUCH PROJECTOR - Single Calibration Mode")
    print("="*70)
    print("\nSTEP 1: CALIBRATION")
    print("Click the 4 corners of your projection area in this order:")
    print("  1. TOP-LEFT corner")
    print("  2. TOP-RIGHT corner")
    print("  3. BOTTOM-RIGHT corner")
    print("  4. BOTTOM-LEFT corner")
    print("\nControls after calibration:")
    print("  m - Switch click method")
    print("  + / - - Adjust touch zone")
    print("  c - Recalibrate")
    print("  ESC - Exit")
    print("="*70)
    
    # Create window and set mouse callback
    cv2.namedWindow('Calibration')
    cv2.setMouseCallback('Calibration', mouse_callback)
    
    # Main loop
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        current_frame = frame.copy()
        calibration_display = frame.copy()
        
        if not calibration_done:
            # ============================================
            # CALIBRATION MODE
            # ============================================
            
            # Draw instructions
            cv2.putText(calibration_display, f"CALIBRATION: Click corner {len(calibration_points)+1}/4", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw points that have been clicked
            for i, pt in enumerate(calibration_points):
                cv2.circle(calibration_display, pt, 8, (0, 255, 0), -1)
                cv2.putText(calibration_display, str(i+1), (pt[0]+15, pt[1]-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw guide lines if we have at least 2 points
            if len(calibration_points) >= 2:
                for i in range(len(calibration_points)-1):
                    cv2.line(calibration_display, calibration_points[i], 
                            calibration_points[i+1], (255, 255, 0), 2)
            
            cv2.imshow('Calibration', calibration_display)
            
        else:
            # ============================================
            # TRACKING MODE
            # ============================================
            
            # Get current homography using feature matching
            current_H, matches = get_current_transform(frame)
            
            if current_H is not None:
                # Apply transform to get warped view
                warped = cv2.warpPerspective(frame, current_H, (AR[1], AR[0]))
                
                # Draw projection area boundary on main frame
                corners = np.float32([[0,0], [AR[1],0], [AR[1],AR[0]], [0,AR[0]]]).reshape(-1,1,2)
                projected_corners = cv2.perspectiveTransform(corners, np.linalg.inv(current_H))
                cv2.polylines(calibration_display, [np.int32(projected_corners)], True, (0, 255, 0), 2)
                
                # Draw touch zone on warped image
                touch_line_y = int(AR[0] * touch_zone_ratio)
                cv2.line(warped, (0, touch_line_y), (AR[1], touch_line_y), (255, 255, 0), 3)
                cv2.putText(warped, f"TOUCH ZONE ({method_names[click_method]})", 
                           (10, touch_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Process hand detection on warped image
                rgb_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_warped)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(
                            warped,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Get finger states
                        finger_states = get_finger_states(hand_landmarks)
                        
                        # Get index fingertip position
                        h, w, _ = warped.shape
                        index_tip = hand_landmarks.landmark[LM_INDEX['INDEX_TIP']]
                        fingertip = (int(index_tip.x * w), int(index_tip.y * h))
                        
                        # Draw fingertip
                        cv2.circle(warped, fingertip, 15, (0, 0, 255), 3)
                        cv2.circle(warped, fingertip, 5, (255, 255, 255), -1)
                        
                        # Convert to screen coordinates
                        screen_x, screen_y = warped_to_screen(fingertip)
                        
                        # Move mouse to fingertip position
                        mouse.position = (screen_x, screen_y)
                        
                        # ============================================
                        # CLICK DETECTION
                        # ============================================
                        click_triggered = False
                        
                        if click_method == 0:  # Touch Zone
                            if fingertip[1] > touch_line_y:
                                if not touch_active and click_cooldown <= 0:
                                    mouse.click(Button.left)
                                    print(f"\r✓ TOUCH ZONE CLICK at ({screen_x}, {screen_y})", end="")
                                    click_triggered = True
                                    touch_active = True
                                    click_cooldown = click_cooldown_max
                            else:
                                touch_active = False
                        
                        elif click_method == 1:  # Pinch Gesture
                            is_pinching, pinch_center, pinch_dist = detect_pinch(hand_landmarks, warped.shape)
                            
                            # Draw pinch visualization
                            thumb_tip = hand_landmarks.landmark[LM_INDEX['THUMB_TIP']]
                            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                            
                            if is_pinching:
                                cv2.line(warped, fingertip, thumb_pos, (0, 255, 255), 3)
                                cv2.circle(warped, pinch_center, 20, (0, 255, 255), 2)
                                cv2.putText(warped, f"PINCH", (pinch_center[0]-30, pinch_center[1]-20),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            
                            if is_pinching and not touch_active and click_cooldown <= 0:
                                mouse.click(Button.left)
                                print(f"\r✓ PINCH CLICK at ({screen_x}, {screen_y})", end="")
                                click_triggered = True
                                touch_active = True
                                click_cooldown = click_cooldown_max
                            elif not is_pinching:
                                touch_active = False
                        
                        elif click_method == 2:  # Finger Stay
                            if prev_index_tip is not None:
                                # Calculate movement distance
                                distance = math.sqrt(
                                    (fingertip[0] - prev_index_tip[0])**2 + 
                                    (fingertip[1] - prev_index_tip[1])**2
                                )
                                
                                if distance < 5:  # Finger is still
                                    stay_counter += 1
                                    
                                    # Draw progress bar
                                    bar_width = 60
                                    bar_height = 10
                                    progress = min(stay_counter / stay_threshold, 1.0)
                                    filled_width = int(bar_width * progress)
                                    
                                    # Background
                                    cv2.rectangle(warped, 
                                                (fingertip[0] - bar_width//2, fingertip[1] - 40),
                                                (fingertip[0] + bar_width//2, fingertip[1] - 40 + bar_height),
                                                (100, 100, 100), -1)
                                    # Progress
                                    cv2.rectangle(warped, 
                                                (fingertip[0] - bar_width//2, fingertip[1] - 40),
                                                (fingertip[0] - bar_width//2 + filled_width, fingertip[1] - 40 + bar_height),
                                                (0, 255, 0), -1)
                                    
                                    if stay_counter >= stay_threshold and click_cooldown <= 0:
                                        mouse.click(Button.left)
                                        print(f"\r✓ STAY CLICK at ({screen_x}, {screen_y})", end="")
                                        click_triggered = True
                                        click_cooldown = click_cooldown_max
                                        stay_counter = 0
                                else:
                                    stay_counter = max(0, stay_counter - 1)
                            
                            prev_index_tip = fingertip
                        
                        # Visual feedback for click
                        if click_triggered:
                            cv2.putText(warped, "CLICK!", (fingertip[0]-30, fingertip[1]-60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        
                        # Display finger states
                        y_offset = 60
                        for i, (finger, extended) in enumerate(finger_states.items()):
                            color = (0, 255, 0) if extended else (0, 0, 255)
                            status = "UP" if extended else "DOWN"
                            cv2.putText(warped, f"{finger}: {status}", (10, y_offset + i*20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                else:
                    cv2.putText(warped, "No hand detected", (AR[1]//2-80, AR[0]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    prev_index_tip = None
                    stay_counter = 0
                
                # Show warped view
                cv2.imshow('Warped View (Top-Down)', warped)
                
                # Show tracking status
                cv2.putText(calibration_display, "✓ TRACKING ACTIVE", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            else:
                # Lost tracking
                cv2.putText(calibration_display, "✗ TRACKING LOST - Hold still", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Update cooldown
            if click_cooldown > 0:
                click_cooldown -= 1
            
            # Add info text on main display
            info_lines = [
                f"Method: {method_names[click_method]}",
                f"Touch Zone: {int(touch_zone_ratio*100)}%",
                f"Cooldown: {click_cooldown}"
            ]
            
            for i, line in enumerate(info_lines):
                cv2.putText(calibration_display, line, (10, calibration_display.shape[0] - 80 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Calibration', calibration_display)
        
        # ============================================
        # KEYBOARD CONTROLS
        # ============================================
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('m') and calibration_done:  # Switch click method
            click_method = (click_method + 1) % 3
            print(f"\nSwitched to {method_names[click_method]} method")
        elif key == ord('+') and calibration_done:  # Increase touch zone
            touch_zone_ratio = min(0.95, touch_zone_ratio + 0.05)
            print(f"Touch zone: {int(touch_zone_ratio*100)}%")
        elif key == ord('-') and calibration_done:  # Decrease touch zone
            touch_zone_ratio = max(0.5, touch_zone_ratio - 0.05)
            print(f"Touch zone: {int(touch_zone_ratio*100)}%")
        elif key == ord('c'):  # Recalibrate
            calibration_points = []
            calibration_done = False
            reference_frame = None
            reference_kp = None
            reference_des = None
            print("\nCalibration reset. Click the 4 corners again.")
    
    # Cleanup
    hands.close()
    cv2.destroyAllWindows()
    cap.release()
    print("\nProgram terminated")

# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cv2.destroyAllWindows()
        cap.release()